"""
Parametric skin / subcutaneous-fat tetrahedral shell generator.

Produces a single-layer-thick structured hexahedral grid (split into tets)
that drapes over the anterior torso.  Inner-layer vertices are placed by
casting rays *inward* from an egg-shaped (ellipsoidal) base surface and
finding the closest intersection with the underlying anatomy meshes:
  breast_l, breast_r, ribcage, clavicle_l, clavicle_r, arm_l, arm_r.

The intersection point (plus a small outward gap) becomes the inner-shell
vertex.  Barycentric coordinates relative to the hit triangle are stored
directly in ``SkinShellData`` for real-time kinematic tracking without any
additional nearest-search pass.

Coverage region:
  • top    = neckline (just above the clavicles)
  • bottom = inferior rib margin
  • left/right = lateral shoulder line

Coordinate convention (same as anatomy.py):
  x  medial(0)/lateral(+)   left breast at +x
  y  inferior(−)/superior(+)
  z  posterior(−)/anterior(+, outward)
"""

from __future__ import annotations

from collections import namedtuple
from enum import IntEnum

import numpy as np

try:
    from PBD_Taichi.geom import gtet
except ImportError:
    from geom import gtet


# ---------------------------------------------------------------------------
# Anchor-type enum
# ---------------------------------------------------------------------------

class AnchorType(IntEnum):
    FREE       = 0
    BREAST_L   = 1
    BREAST_R   = 2
    CLAVICLE_L = 3
    CLAVICLE_R = 4
    ARM_L      = 5
    ARM_R      = 6
    RIBCAGE    = 7


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------

SkinShellData = namedtuple('SkinShellData', [
    'verts',           # (n_verts, 3)   float32
    'tets_flat',       # (n_tets*4,)    int32  – flat tet indices
    'faces_flat',      # (n_faces*3,)   int32  – flat surface-triangle indices
    'n_u',             # int – grid columns
    'n_v',             # int – grid rows
    'inner_idx',       # (n_u*n_v,)     int32  – global vertex indices of inner layer
    'outer_idx',       # (n_u*n_v,)     int32  – global vertex indices of outer layer
    'anchor_type',     # (n_u*n_v,)     int32  – AnchorType per inner vert
    'anchor_target',   # (n_u*n_v,)     int32  – DEPRECATED (always -1, kept for compat)
    'anchor_bary_face',  # (n_u*n_v,)   int32  – triangle idx in anchor mesh (-1 if FREE)
    'anchor_bary_uvw',   # (n_u*n_v, 3) float32 – barycentric coords for ALL non-FREE types
])


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def generate_skin_shell(
        skeleton,
        breast_l_verts: np.ndarray,
        breast_r_verts: np.ndarray,
        breast_l_faces: np.ndarray | None = None,
        breast_r_faces: np.ndarray | None = None,
        ribcage_verts: np.ndarray | None = None,
        ribcage_faces: np.ndarray | None = None,
        n_u: int = 20,
        n_v: int = 30,
        thickness: float = 0.005,
        egg_depth: float = 0.15,
        gap: float = 0.001,
) -> SkinShellData:

    """
    Parameters
    ----------
    skeleton : anatomy.Skeleton
        Must already be at its rest pose (``update()`` called).
    breast_l_verts, breast_r_verts : (N, 3) float32
        World-space positions of *all* breast mesh vertices (not just outer).
    breast_l_faces, breast_r_faces : (F, 3) int32 or None
        Surface triangle face arrays for the breast meshes.  If None the
        breast is absent from the ray-cast soup (no BREAST_L/R anchors).
    ribcage_verts : (K, 3) float32 or None
        World-space ribcage surface verts.
    ribcage_faces : (M, 3) int32 or None
        Surface triangle face array for the ribcage mesh.
    n_u, n_v : int
        Grid resolution (columns × rows).
    thickness : float  (metres)
        Shell thickness (skin + subcutaneous fat, typically 3–8 mm).
    egg_depth : float  (metres)
        How far anterior of the deepest anatomy z-coordinate to place the
        depth-map ray origins (i.e. the ``z_offset`` passed to
        :func:`_build_depthmap_surface`).  0.10–0.15 m is a safe default.
        (Formerly controlled the egg-ellipsoid protrusion depth; that code
        has been replaced by the depth-map approach.)
    gap : float  (metres)
        Outward offset applied to each ray-hit point before placing the
        inner vert – prevents geometry interpenetration (default 1 mm).
    """

    # ── 1. determine grid coverage region ─────────────────────────────────
    clav_l = skeleton.get_clavicle_left_world_np()
    clav_r = skeleton.get_clavicle_right_world_np()
    arm_l  = skeleton.get_upper_arm_left_surface_np()
    arm_r  = skeleton.get_upper_arm_right_surface_np()

    cp     = skeleton.chest_pos                         # (3,) world root
    clav_y = cp[1] + skeleton._clavicle_l_offset[1]    # SC joint y
    y_top  = clav_y + 0.015                             # slightly above SC joint
    y_bot  = cp[1] - 0.3                               # inferior rib margin

    all_pts = np.concatenate(
        [breast_l_verts, breast_r_verts, clav_l, clav_r, arm_l, arm_r], axis=0)
    x_half = float(np.abs(all_pts[:, 0]).max()) + 0.01
    x_half = max(x_half, 0.16)

    grid_x = np.linspace(-x_half, x_half, n_u, dtype=np.float32)
    grid_y = np.linspace(y_top,   y_bot,  n_v, dtype=np.float32)   # top→bottom

    # ── 2. build tagged triangle soup ─────────────────────────────────────
    #   Soup is needed before surface generation (depth-map uses it).
    _empty_f = np.zeros((0, 3), dtype=np.int32)
    _empty_v = np.zeros((0, 3), dtype=np.float32)

    soup_tris, soup_tags, soup_normals, soup_face_idx = _build_tagged_soup(
        skeleton,
        breast_l_verts,
        breast_l_faces if breast_l_faces is not None else _empty_f,
        breast_r_verts,
        breast_r_faces if breast_r_faces is not None else _empty_f,
        ribcage_verts  if ribcage_verts  is not None else _empty_v,
        ribcage_faces  if ribcage_faces  is not None else _empty_f,
    )
    print(f"[SkinShell] triangle soup: {len(soup_tris)} tris "
          f"from {len(np.unique(soup_tags))} anchor type(s)")

    # ── 3. build depth-map starting surface ───────────────────────────────
    #   For each (x_i, y_j) grid column, rasterise the triangle soup in XY
    #   and find the per-column z_max.  Ray origins are placed at
    #   (x_i, y_j, z_col_max + egg_depth) and all rays shoot along [0,0,-1].
    #   This replaces the old ellipsoid ("egg") prior with a surface that is
    #   always anterior to the anatomy and axis-aligned for clean raycasting.
    surf_pos, surf_normals = _build_depthmap_surface(
        grid_x, grid_y, soup_tris, z_offset=egg_depth)
    # surf_pos     : (n_v, n_u, 3) – ray start positions
    # surf_normals : (n_v, n_u, 3) – all [0, 0, 1] (anterior)

    # ── 4. ray-cast inward (−z) from every surface vertex ─────────────────
    K        = n_u * n_v
    ray_orig = surf_pos.reshape(K, 3)                                         # (K, 3)
    ray_dir  = np.tile(np.array([[0.0, 0.0, -1.0]], dtype=np.float32), (K, 1))  # (K, 3)

    hit_t, hit_tri_soup, hit_u, hit_v = _cast_rays_inward(
        ray_orig, ray_dir, soup_tris, soup_normals)

    # ── 5. place inner verts and fill anchor arrays (vectorised) ──────────
    n_inner  = K
    # Outward direction is always +z (anterior) for the depth-map approach.
    outward  = np.tile(np.array([[0.0, 0.0, 1.0]], dtype=np.float32), (K, 1))  # (K, 3)

    inner_pos        = ray_orig.copy().astype(np.float32)       # default: start surface
    anchor_type      = np.full(n_inner, int(AnchorType.FREE),  dtype=np.int32)
    anchor_target    = np.full(n_inner, -1,                    dtype=np.int32)  # deprecated
    anchor_bary_face = np.full(n_inner, -1,                    dtype=np.int32)
    anchor_bary_uvw  = np.zeros((n_inner, 3),                  dtype=np.float32)

    hit_mask = hit_tri_soup >= 0   # (K,)
    if hit_mask.any():
        hi      = np.where(hit_mask)[0]              # (H,) indices of hit verts
        t_hi    = hit_t[hi, None]                    # (H, 1)
        hit_pts = ray_orig[hi] + t_hi * ray_dir[hi]  # (H, 3) on anatomy surface
        # push slightly outward (+z) by gap so skin does not interpenetrate anatomy
        inner_pos[hi] = (hit_pts + gap * outward[hi]).astype(np.float32)

        tri_idx              = hit_tri_soup[hi]       # (H,) soup triangle indices
        anchor_type[hi]      = soup_tags[tri_idx]
        anchor_bary_face[hi] = soup_face_idx[tri_idx]

        u_hi = hit_u[hi]; v_hi = hit_v[hi]
        anchor_bary_uvw[hi, 0] = 1.0 - u_hi - v_hi  # w  (weight for face vertex 0)
        anchor_bary_uvw[hi, 1] = u_hi                # u  (weight for face vertex 1)
        anchor_bary_uvw[hi, 2] = v_hi                # v  (weight for face vertex 2)

    # ── 6. single-layer vertex array (no extrusion yet) ──────────────────
    verts     = inner_pos.astype(np.float32)          # (n_inner, 3)
    inner_idx = np.arange(n_inner, dtype=np.int32)
    outer_idx = np.arange(n_inner, dtype=np.int32)    # same layer – no outer yet

    # ── 7. grid quads → triangle surface mesh ─────────────────────────────
    tris: list[list[int]] = []
    for j in range(n_v - 1):
        for i in range(n_u - 1):
            v0 = j       * n_u + i;         v1 = j       * n_u + (i + 1)
            v2 = (j + 1) * n_u + (i + 1);   v3 = (j + 1) * n_u + i
            tris.append([v0, v1, v2])
            tris.append([v0, v2, v3])

    tets_flat  = np.zeros(0, dtype=np.int32)
    faces_flat = np.array(tris, dtype=np.int32).flatten().astype(np.int32)

    n_tris = len(tris)
    print(f"[SkinShell] {n_inner} verts, {n_tris} tris (single-layer surface), "
          f"grid {n_u}×{n_v}")
    if anchor_type is not None:
        _print_anchor_stats(anchor_type)

    return SkinShellData(
        verts=verts,
        tets_flat=tets_flat,
        faces_flat=faces_flat,
        n_u=n_u,
        n_v=n_v,
        inner_idx=inner_idx,
        outer_idx=outer_idx,
        anchor_type=anchor_type,
        anchor_target=anchor_target,
        anchor_bary_face=anchor_bary_face,
        anchor_bary_uvw=anchor_bary_uvw,
    )


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _build_egg_surface(
    grid_x: np.ndarray,   # (n_u,)
    grid_y: np.ndarray,   # (n_v,)
    y_c:    float,
    a_x:    float,
    a_y:    float,
    a_z:    float,
    z_c:    float,
) -> tuple[np.ndarray, np.ndarray]:
    """Parameterise the anterior half-ellipsoid over the UV grid.

    The ellipsoid is centred at ``(0, y_c, z_c)`` with semi-axes
    ``a_x`` (lateral), ``a_y`` (vertical), ``a_z`` (anterior depth).
    Only the anterior hemisphere (z >= z_c) is generated.  Grid points
    outside the ellipse in XY (s² ≥ 1) are clamped to the equatorial
    plane z = z_c and given a flat forward normal ``[0, 0, 1]``.

    Returns
    -------
    egg_pos     : (n_v, n_u, 3) float32 – world positions on the egg
    egg_normals : (n_v, n_u, 3) float32 – outward unit surface normals
    """
    X, Y  = np.meshgrid(grid_x, grid_y)               # both (n_v, n_u)
    s2    = (X / a_x) ** 2 + ((Y - y_c) / a_y) ** 2  # (n_v, n_u)

    Z = (z_c + a_z * np.sqrt(np.clip(1.0 - s2, 0.0, 1.0))).astype(np.float32)

    egg_pos = np.stack([X.astype(np.float32),
                        Y.astype(np.float32),
                        Z], axis=-1)   # (n_v, n_u, 3)

    # Outward normal: normalised gradient of the implicit ellipsoid function
    # F = (x/ax)² + ((y-yc)/ay)² + ((z-zc)/az)² - 1,  ∇F ∝ [x/ax², …]
    nx_raw = X / (a_x ** 2)
    ny_raw = (Y - y_c) / (a_y ** 2)
    nz_raw = (Z - z_c) / (a_z ** 2)
    nlen   = np.sqrt(nx_raw ** 2 + ny_raw ** 2 + nz_raw ** 2) + 1e-12

    # Equatorial/exterior points (Z = z_c → nz_raw ≈ 0): fall back to [0,0,1]
    # so the inward ray points straight backward and can still hit the chest wall.
    eq = s2 >= 100.0
    nx = np.where(eq, 0.0, nx_raw / nlen).astype(np.float32)
    ny = np.where(eq, 0.0, ny_raw / nlen).astype(np.float32)
    nz = np.where(eq, 1.0, nz_raw / nlen).astype(np.float32)

    egg_normals = np.stack([nx, ny, nz], axis=-1)   # (n_v, n_u, 3)
    return egg_pos, egg_normals


def _build_depthmap_surface(
    grid_x: np.ndarray,    # (n_u,)  ascending
    grid_y: np.ndarray,    # (n_v,)  may be descending (superior → inferior)
    soup_tris: np.ndarray, # (T, 3, 3) float32
    z_offset: float = 0.05,
) -> tuple[np.ndarray, np.ndarray]:
    """Build the anterior depth-map starting surface for shrinkwrap raycasting.

    For each ``(x_i, y_j)`` grid column, rasterises the triangle soup onto the
    grid in the XY plane and records the maximum z of any triangle that covers
    that column.  The ray origin is placed at
    ``(x_i, y_j, z_col_max + z_offset)`` and all ray directions are
    ``[0, 0, −1]`` (shooting posteriorly).

    Grid columns with no triangle coverage fall back to the global z maximum
    so that every ray still has a valid (if slightly conservative) start.

    Parameters
    ----------
    grid_x, grid_y : 1-D float32 arrays
        Grid axis coordinates.  ``grid_y`` may be descending.
    soup_tris : (T, 3, 3) float32
        Triangle vertex array from :func:`_build_tagged_soup`.
    z_offset : float
        How far anterior of the deepest anatomy z-coordinate to start each ray.
        A value of 0.05–0.15 m is typically sufficient.

    Returns
    -------
    surf_pos : (n_v, n_u, 3) float32
        Ray origins; z component equals per-column z_max + z_offset.
    surf_normals : (n_v, n_u, 3) float32
        Outward unit normals – all ``[0, 0, 1]`` (anterior).
    """
    n_u = len(grid_x)
    n_v = len(grid_y)
    z_max_grid = np.full((n_v, n_u), -np.inf, dtype=np.float64)

    if len(soup_tris) > 0:
        global_z_max = float(soup_tris.reshape(-1, 3)[:, 2].max())

        for tri in soup_tris:
            A, B, C = tri[0], tri[1], tri[2]
            ax, ay, az = float(A[0]), float(A[1]), float(A[2])
            bx, by, bz = float(B[0]), float(B[1]), float(B[2])
            cx, cy, cz = float(C[0]), float(C[1]), float(C[2])

            # --- XY bounding box; clip to grid extent -------------------------
            x_lo = min(ax, bx, cx);  x_hi = max(ax, bx, cx)
            y_lo = min(ay, by, cy);  y_hi = max(ay, by, cy)

            # grid_x is ascending – use searchsorted
            i_lo = max(0, int(np.searchsorted(grid_x, x_lo, 'left'))  - 1)
            i_hi = min(n_u - 1, int(np.searchsorted(grid_x, x_hi, 'right')))
            if i_lo > i_hi:
                continue

            # grid_y may be descending – use a boolean mask
            j_ids = np.where((grid_y >= y_lo) & (grid_y <= y_hi))[0]
            if len(j_ids) == 0:
                continue
            j_lo, j_hi = int(j_ids[0]), int(j_ids[-1])

            # --- vectorised 2-D barycentric test over the sub-grid patch ------
            gx_patch = grid_x[i_lo:i_hi + 1]   # (ni,)
            gy_patch = grid_y[j_lo:j_hi + 1]   # (nj,)
            GX, GY = np.meshgrid(gx_patch, gy_patch)  # (nj, ni)

            e1x, e1y = bx - ax, by - ay
            e2x, e2y = cx - ax, cy - ay
            denom = e1x * e2y - e1y * e2x
            if abs(denom) < 1e-12:
                continue   # degenerate triangle in XY projection

            epx = GX - ax   # (nj, ni)
            epy = GY - ay
            u = (epx * e2y - epy * e2x) / denom
            v = (e1x * epy - e1y * epx) / denom

            inside = (u >= -1e-6) & (v >= -1e-6) & (u + v <= 1.0 + 1e-6)
            z_interp = az + u * (bz - az) + v * (cz - az)  # (nj, ni)

            patch = z_max_grid[j_lo:j_hi + 1, i_lo:i_hi + 1]
            z_max_grid[j_lo:j_hi + 1, i_lo:i_hi + 1] = np.where(
                inside, np.maximum(patch, z_interp), patch)

        # Columns not covered by any triangle fall back to the global max
        z_max_grid = np.where(np.isinf(z_max_grid), global_z_max, z_max_grid)
    else:
        z_max_grid[:] = 0.0

    z_start  = (z_max_grid + z_offset).astype(np.float32)
    X, Y     = np.meshgrid(grid_x.astype(np.float32),
                           grid_y.astype(np.float32))   # both (n_v, n_u)
    surf_pos = np.stack([X, Y, z_start], axis=-1)       # (n_v, n_u, 3)

    surf_normals = np.zeros_like(surf_pos)
    surf_normals[:, :, 2] = 1.0   # all normals point anteriorly (+z)

    return surf_pos, surf_normals




def _build_tagged_soup(
    skeleton,
    bl_v: np.ndarray, bl_f: np.ndarray,  # breast L verts + surface faces
    br_v: np.ndarray, br_f: np.ndarray,  # breast R verts + surface faces
    rc_v: np.ndarray, rc_f: np.ndarray,  # ribcage  verts + surface faces
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Flatten all anatomy meshes into a single tagged triangle soup.

    Returns
    -------
    soup_tris     : (T, 3, 3) float32  – triangle vertices [A, B, C]
    soup_tags     : (T,) int32         – AnchorType per triangle
    soup_normals  : (T, 3) float32     – pre-computed outward face normals
    soup_face_idx : (T,) int32         – local face index in the source mesh
    """
    clav_l   = skeleton.get_clavicle_left_world_np()
    clav_r   = skeleton.get_clavicle_right_world_np()
    arm_l    = skeleton.get_upper_arm_left_surface_np()
    arm_r    = skeleton.get_upper_arm_right_surface_np()
    clav_l_f = skeleton.get_clavicle_left_faces_np()
    clav_r_f = skeleton.get_clavicle_right_faces_np()
    arm_l_f  = skeleton.get_upper_arm_left_faces_np()
    arm_r_f  = skeleton.get_upper_arm_right_faces_np()

    # Traversal order: deepest (posterior) structures first, then breast last.
    # The ray-cast picks minimum-t regardless, so order only affects tie-breaking.
    sources = [
        (AnchorType.RIBCAGE,    rc_v,   rc_f),
        (AnchorType.CLAVICLE_L, clav_l, clav_l_f),
        (AnchorType.CLAVICLE_R, clav_r, clav_r_f),
        (AnchorType.ARM_L,      arm_l,  arm_l_f),
        (AnchorType.ARM_R,      arm_r,  arm_r_f),
        (AnchorType.BREAST_L,   bl_v,   bl_f),
        (AnchorType.BREAST_R,   br_v,   br_f),
    ]

    tris_list     = []
    tags_list     = []
    normals_list  = []
    face_idx_list = []

    for atype, verts, faces in sources:
        verts = np.asarray(verts, dtype=np.float32)
        faces = np.asarray(faces, dtype=np.int32).reshape(-1, 3)  # handle flat or (F,3)
        if len(verts) == 0 or len(faces) == 0:
            continue
        A = verts[faces[:, 0]]   # (F, 3)
        B = verts[faces[:, 1]]
        C = verts[faces[:, 2]]
        tris = np.stack([A, B, C], axis=1)   # (F, 3, 3)

        e1 = B - A;  e2 = C - A
        n  = np.cross(e1, e2)
        n  = n / (np.linalg.norm(n, axis=1, keepdims=True) + 1e-12)

        tris_list.append(tris)
        tags_list.append(np.full(len(faces), int(atype), dtype=np.int32))
        normals_list.append(n.astype(np.float32))
        face_idx_list.append(np.arange(len(faces), dtype=np.int32))

    if not tris_list:
        return (np.zeros((0, 3, 3), dtype=np.float32),
                np.zeros(0,         dtype=np.int32),
                np.zeros((0, 3),    dtype=np.float32),
                np.zeros(0,         dtype=np.int32))

    return (np.concatenate(tris_list,     axis=0),
            np.concatenate(tags_list,     axis=0),
            np.concatenate(normals_list,  axis=0),
            np.concatenate(face_idx_list, axis=0))


def _cast_rays_inward(
    ray_orig:     np.ndarray,   # (K, 3) float32
    ray_dir:      np.ndarray,   # (K, 3) float32 – unit inward normals
    soup_tris:    np.ndarray,   # (T, 3, 3) float32
    soup_normals: np.ndarray,   # (T, 3) float32
    eps: float = 1e-9,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Vectorised Möller–Trumbore ray-triangle intersection.

    For each of the K rays finds the triangle in ``soup_tris`` with the
    smallest positive t (closest visible surface).  Back-faces (triangles
    whose normal faces the same direction as the ray) are rejected so that
    for convex closed meshes the inward ray hits the outer surface first.

    Returns
    -------
    hit_t    : (K,) float32 – ray parameter; np.inf on miss
    hit_tri  : (K,) int32   – soup triangle index; -1 on miss
    hit_u    : (K,) float32 – MT barycentric u
    hit_v    : (K,) float32 – MT barycentric v
    """
    K = len(ray_orig)
    T = len(soup_tris)

    hit_t   = np.full(K, np.inf, dtype=np.float32)
    hit_tri = np.full(K, -1,     dtype=np.int32)
    hit_u   = np.zeros(K,        dtype=np.float32)
    hit_v   = np.zeros(K,        dtype=np.float32)

    if T == 0:
        return hit_t, hit_tri, hit_u, hit_v

    # Pre-compute edges once; reused for every ray.
    A  = soup_tris[:, 0, :]       # (T, 3)
    e1 = soup_tris[:, 1, :] - A  # (T, 3)
    e2 = soup_tris[:, 2, :] - A  # (T, 3)

    for k in range(K):
        d = ray_dir[k]    # (3,)
        o = ray_orig[k]   # (3,)

        # Back-face filter: dot(ray_dir, face_normal) > 0 → back face, skip.
        backface = (soup_normals @ d) > 0.0   # (T,) bool

        # Möller–Trumbore ---------------------------------------------------
        h   = np.cross(d, e2)                           # (T,3); d broadcasts as (1,3)
        det = np.einsum('ti,ti->t', e1, h)              # (T,)

        valid = (~backface) & (np.abs(det) > eps)
        inv_det = np.where(valid, 1.0 / np.where(valid, det, 1.0), 0.0)

        s   = o - A                                      # (T, 3); o broadcasts as (1,3)
        u   = np.einsum('ti,ti->t', s, h) * inv_det     # (T,)
        valid = valid & (u >= 0.0) & (u <= 1.0)

        q     = np.cross(s, e1)                          # (T, 3)
        v_val = (q @ d) * inv_det                        # (T,)
        valid = valid & (v_val >= 0.0) & (u + v_val <= 1.0)

        t_param = np.einsum('ti,ti->t', e2, q) * inv_det  # (T,)
        valid   = valid & (t_param > eps)

        t_masked = np.where(valid, t_param, np.inf)
        best     = int(np.argmin(t_masked))
        if t_masked[best] < np.inf:
            hit_t[k]   = float(t_masked[best])
            hit_tri[k] = best
            hit_u[k]   = float(u[best])
            hit_v[k]   = float(v_val[best])

    return hit_t, hit_tri, hit_u, hit_v


def _print_anchor_stats(anchor_type: np.ndarray) -> None:
    for at in AnchorType:
        n = int((anchor_type == at).sum())
        if n > 0:
            print(f"  {at.name:14s}: {n}")


# ---------------------------------------------------------------------------
# Deprecated helpers – kept so existing external callers do not break
# ---------------------------------------------------------------------------

def _project_point_to_triangle(P, A, B, C):
    """Return ``(uvw, dist)`` – deprecated.

    Closest-point projection using the Christer Ericson vertex/edge/face
    region algorithm (Real-Time Collision Detection §5.1.5).
    """
    AB = B - A;  AC = C - A;  AP = P - A
    d1 = float(np.dot(AB, AP));  d2 = float(np.dot(AC, AP))
    if d1 <= 0.0 and d2 <= 0.0:
        return np.array([1.0, 0.0, 0.0], dtype=np.float32), float(np.linalg.norm(P - A))
    BP = P - B
    d3 = float(np.dot(AB, BP));  d4 = float(np.dot(AC, BP))
    if d3 >= 0.0 and d4 <= d3:
        return np.array([0.0, 1.0, 0.0], dtype=np.float32), float(np.linalg.norm(P - B))
    vc = d1 * d4 - d3 * d2
    if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
        v = d1 / (d1 - d3)
        return np.array([1.0 - v, v, 0.0], dtype=np.float32), float(np.linalg.norm(P - (A + v * AB)))
    CP = P - C
    d5 = float(np.dot(AB, CP));  d6 = float(np.dot(AC, CP))
    if d6 >= 0.0 and d5 <= d6:
        return np.array([0.0, 0.0, 1.0], dtype=np.float32), float(np.linalg.norm(P - C))
    vb = d5 * d2 - d1 * d6
    if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
        w = d2 / (d2 - d6)
        return np.array([1.0 - w, 0.0, w], dtype=np.float32), float(np.linalg.norm(P - (A + w * AC)))
    va = d3 * d6 - d5 * d4
    if va <= 0.0 and (d4 - d3) >= 0.0 and (d5 - d6) >= 0.0:
        denom = (d4 - d3) + (d5 - d6)
        w = (d4 - d3) / denom if denom != 0.0 else 0.0
        closest = B + w * (C - B)
        return np.array([0.0, 1.0 - w, w], dtype=np.float32), float(np.linalg.norm(P - closest))
    denom = va + vb + vc
    if denom == 0.0:
        return np.array([1/3, 1/3, 1/3], dtype=np.float32), float(np.linalg.norm(P - A))
    inv = 1.0 / denom
    v = vb * inv;  w = vc * inv;  u = 1.0 - v - w
    closest = u * A + v * B + w * C
    return np.array([u, v, w], dtype=np.float32), float(np.linalg.norm(P - closest))


def _nearest_triangle_bary(point: np.ndarray,
                            verts: np.ndarray,
                            faces: np.ndarray):
    """Find the nearest triangle to *point* – deprecated, kept for compatibility.

    Parameters
    ----------
    point : (3,) float32
    verts : (N, 3) float32
    faces : (M, 3) int32

    Returns
    -------
    face_idx : int
    uvw      : (3,) float32 – barycentric coordinates summing to 1
    """
    best_dist = np.inf
    best_face = 0
    best_uvw  = np.array([1/3, 1/3, 1/3], dtype=np.float32)
    for fi in range(len(faces)):
        A = verts[faces[fi, 0]]
        B = verts[faces[fi, 1]]
        C = verts[faces[fi, 2]]
        uvw, dist = _project_point_to_triangle(point, A, B, C)
        if dist < best_dist:
            best_dist = dist
            best_face = fi
            best_uvw  = uvw
    return best_face, best_uvw

