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

from PBD_Taichi.geom.gtet import TetMesh
from PBD_Taichi.geom.obj import BoundBox3D

try:
    from PBD_Taichi.geom import gtet, anatomy
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
# Public API
# ---------------------------------------------------------------------------

def _build_tagged_triangle_soup(
        skeleton: anatomy.Skeleton,
        meshes: tuple[str, np.ndarray, np.ndarray],
) -> namedtuple[
            'TaggedTriangleSoup',
            ['verts', 'faces', 'tag', 'face_idx']
            ]:
    meshes = (meshes + [
        ('clav_l', skeleton.get_clavicle_left_world_np(), skeleton.get_upper_arm_left_faces_np()),
        ('clav_r', skeleton.get_clavicle_right_world_np(), skeleton.get_upper_arm_right_faces_np())
        ('arm_l', skeleton.get_upper_arm_left_surface_np(), skeleton.get_upper_arm_left_faces_np()),
        ('arm_r', skeleton.get_upper_arm_right_surface_np(), skeleton.get_upper_arm_right_faces_np()),
    ])

    soup_faces = []
    #for tag, verts, faces in meshes:

class Mesh:
    vertices: np.ndarray # (N, 3) vertices
    faces: np.ndarray # (N, 3) int indices

class BarycentricBindingDefinition:
    """
    Defines a spring binding between a vertex on the skin tetra mesh and a given anchor triangle on a given driving
    (kinematic, deformaable) mesh.
    The triangle end of the spring is bound to the triangle using barycentric coordinates with a distance offset.
    """
    anchor_type: AnchorType
    anchor_bary_face: np.ndarray  # vertex indices on the mesh indicated by anchor_type
    anchor_bary_uvw: np.ndarray   # barycentric UVW
    skin_vertex_index: int        # index of the vertex on the skin shell tetra mesh
    skin_vertex_distance: float   # offset distance from the anchor triangle (spring length at initial pos)


def generate_skin_shell(
    skeleton,
    breast_l_verts: np.ndarray,
    breast_r_verts: np.ndarray,
    breast_l_faces: np.ndarray | None = None,
    breast_r_faces: np.ndarray | None = None,
    ribcage_verts: np.ndarray | None = None,
    ribcage_faces: np.ndarray | None = None,
    target_n_tets: int = 600,
    thickness: float = 0.005,
) -> tuple[TetMesh, list[BarycentricBindingDefinition]]:

    from PBD_Taichi.geom import distance_field as df

    _empty_f = np.zeros((0, 3), dtype=np.int32)
    _empty_v = np.zeros((0, 3), dtype=np.float64)

    # ── collect skeleton surfaces ──────────────────────────────────────────
    clav_l_v = np.asarray(skeleton.get_clavicle_left_world_np(),     dtype=np.float64)
    clav_r_v = np.asarray(skeleton.get_clavicle_right_world_np(),    dtype=np.float64)
    arm_l_v  = np.asarray(skeleton.get_upper_arm_left_surface_np(),  dtype=np.float64)
    arm_r_v  = np.asarray(skeleton.get_upper_arm_right_surface_np(), dtype=np.float64)
    clav_l_f = np.asarray(skeleton.get_clavicle_left_faces_np(),     dtype=np.int32).reshape(-1, 3)
    clav_r_f = np.asarray(skeleton.get_clavicle_right_faces_np(),    dtype=np.int32).reshape(-1, 3)
    arm_l_f  = np.asarray(skeleton.get_upper_arm_left_faces_np(),    dtype=np.int32).reshape(-1, 3)
    arm_r_f  = np.asarray(skeleton.get_upper_arm_right_faces_np(),   dtype=np.int32).reshape(-1, 3)

    bl_v = np.asarray(breast_l_verts, dtype=np.float64)
    br_v = np.asarray(breast_r_verts, dtype=np.float64)
    bl_f = np.asarray(breast_l_faces, dtype=np.int32).reshape(-1, 3) if breast_l_faces is not None else _empty_f
    br_f = np.asarray(breast_r_faces, dtype=np.int32).reshape(-1, 3) if breast_r_faces is not None else _empty_f
    rc_v = np.asarray(ribcage_verts,  dtype=np.float64) if ribcage_verts is not None else _empty_v
    rc_f = np.asarray(ribcage_faces,  dtype=np.int32).reshape(-1, 3) if ribcage_faces is not None else _empty_f

    # ── 1. Boolean merge all anatomy meshes → merged_surface ──────────────
    merge_inputs: list[df.Mesh] = []
    for v, f, label in [
        (rc_v, rc_f, 'ribcage'),
        (bl_v, bl_f, 'breast_l'),
        (br_v, br_f, 'breast_r'),
        (clav_l_v, clav_l_f, 'clav_l'), (clav_r_v, clav_r_f, 'clav_r'),
        (arm_l_v,  arm_l_f, 'arm_l'),  (arm_r_v,  arm_r_f, 'arm_r'),
    ]:
        if len(v) > 0 and len(f) > 0:
            btr = df.BasicTriMesh(verts=v, faces=f, label=label)
            if label == 'ribcage':
                btr = btr.deduplicated()
            merge_inputs.append(btr)

    if not merge_inputs:
        return []

    print(f"[generate_skin_shell] Boolean-merging {len(merge_inputs)} anatomy meshes...")
    merged_surface = df.boolean_merge_meshes(merge_inputs, debug_save_path="generate_skin_shell.ply")
    print(f"[generate_skin_shell] Merged surface: {len(merged_surface.verts)} verts, "
          f"{len(merged_surface.faces)} faces")

    # ── 2. Build skin shell as a single-layer tetrahedral boundary layer ───
    # Option C (remesh_surface=True): pymeshlab isotropic remesh + pure-numpy
    # prism extrusion gives an exact tet count and a guaranteed vertex layout:
    #   verts[0 : n_v]        → inner layer (anatomy-facing, on merged_surface)
    #   verts[n_v : 2*n_v]    → outer layer (offset by thickness)
    # where N=1 is hardcoded in build_boundary_layer.
    print(f"[generate_skin_shell] Building boundary layer "
          f"(target_n_tets={target_n_tets}, thickness={thickness})…")
    shell = df.build_boundary_layer_sdf(
        merged_surface,
        layer_thickness=thickness,
        target_tet_count=target_n_tets,
        debug_save_path="generate_skin_shell_sdf.msh"
    )
    n_v         = len(shell.verts) // 2   # N=1 → two equal rings
    inner_verts = shell.verts[:n_v]       # (n_v, 3) – on the merged-surface side
    print(f"[generate_skin_shell] Shell: {len(shell.verts)} verts ({n_v} inner), "
          f"{len(shell.tets)} tets")

    # ── 3. Build per-inner-vertex BarycentricBindings ─────────────────────
    # Flatten all anatomy meshes into a tagged triangle soup so that for each
    # inner vertex we can find both the nearest triangle *and* which anatomy
    # mesh it belongs to (AnchorType), plus its local face index.
    soup_tris, soup_tags, _, soup_face_idx = _build_tagged_soup(
        skeleton,
        bl_v.astype(np.float32), bl_f,
        br_v.astype(np.float32), br_f,
        rc_v.astype(np.float32), rc_f,
    )

    bindings: list[BarycentricBindingDefinition] = []

    if len(soup_tris) == 0:
        print("[generate_skin_shell] Warning: empty tagged soup – no bindings produced.")
        return shell, bindings

    best_tri_arr, best_uvw_arr, best_dist_arr = _batch_closest_triangles(
        inner_verts.astype(np.float64), soup_tris,
    )

    for vi in range(n_v):
        ti = int(best_tri_arr[vi])
        b = BarycentricBindingDefinition()
        b.anchor_type          = AnchorType(int(soup_tags[ti]))
        b.anchor_bary_face     = int(soup_face_idx[ti])
        b.anchor_bary_uvw      = best_uvw_arr[vi].astype(np.float32)  # (w_A, w_B, w_C)
        b.skin_vertex_index    = vi
        b.skin_vertex_distance = float(best_dist_arr[vi])
        bindings.append(b)

    print(f"[generate_skin_shell] {len(bindings)} bindings produced.")
    _print_anchor_stats(np.array([int(b.anchor_type) for b in bindings], dtype=np.int32))

    # ── 4. Return the bindings ─────────────────────────────────────────────
    return shell, bindings


def generate_skin_shell_archaic_method(
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
        How far the egg surface protrudes anteriorly from ``chest_pos[2]``.
        Must exceed the most anterior z-coordinate of any anatomy mesh.
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

    # ── 2. build egg surface (anterior half-ellipsoid) ────────────────────
    y_c = float((y_top + y_bot) * 0.5)
    a_x = float(x_half)
    a_y = float((y_top - y_bot) * 0.5)
    a_z = float(egg_depth)
    z_c = float(cp[2])

    egg_pos, egg_normals = _build_egg_surface(
        grid_x, grid_y, y_c, a_x, a_y, a_z, z_c)
    # egg_pos:     (n_v, n_u, 3) float32 – world positions on the egg shell
    # egg_normals: (n_v, n_u, 3) float32 – outward surface normals

    debug_egg_surface = False
    if debug_egg_surface:
        verts = egg_pos.reshape((-1, 3)) + egg_normals.reshape((-1, 3)) * 0.01
        n_inner = 0
        anchor_type = None
        inner_idx = None
        outer_idx = None
        anchor_target = None
        anchor_bary_face = None
        anchor_bary_uvw = None
    else:
        # ── 3. build tagged triangle soup ─────────────────────────────────────
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

        # ── 4. ray-cast inward from every egg vertex ───────────────────────────
        K        = n_u * n_v
        ray_orig = egg_pos.reshape(K, 3)          # (K, 3)
        ray_dir  = (-egg_normals).reshape(K, 3)   # (K, 3) inward

        hit_t, hit_tri_soup, hit_u, hit_v = _cast_rays_inward(
            ray_orig, ray_dir, soup_tris, soup_normals)

        # ── 5. place inner verts and fill anchor arrays (vectorised) ──────────
        n_inner  = K
        outward  = egg_normals.reshape(K, 3).astype(np.float32)  # (K, 3)

        inner_pos        = ray_orig.copy().astype(np.float32)       # default: egg surface
        anchor_type      = np.full(n_inner, int(AnchorType.FREE),  dtype=np.int32)
        anchor_target    = np.full(n_inner, -1,                    dtype=np.int32)  # deprecated
        anchor_bary_face = np.full(n_inner, -1,                    dtype=np.int32)
        anchor_bary_uvw  = np.zeros((n_inner, 3),                  dtype=np.float32)

        hit_mask = hit_tri_soup >= 0   # (K,)
        if hit_mask.any():
            hi      = np.where(hit_mask)[0]              # (H,) indices of hit verts
            t_hi    = hit_t[hi, None]                    # (H, 1)
            hit_pts = ray_orig[hi] + t_hi * ray_dir[hi]  # (H, 3) on anatomy surface
            # push slightly outward by gap so skin does not interpenetrate anatomy
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
          f"grid {n_u}×{n_v}")#  ({n_hits}/{n_inner} anchored by raycast)")
    if anchor_type:
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


def _batch_closest_triangles(
    points: np.ndarray,     # (K, 3) float64 – query points
    soup_tris: np.ndarray,  # (T, 3, 3) float32 – triangle soup
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorised Ericson closest-point-on-triangle search.

    For each of the K points finds the closest triangle in the T-triangle
    soup using the Christer Ericson vertex / edge / face region algorithm
    (Real-Time Collision Detection §5.1.5), vectorised over T for each query.

    Vertex regions have the highest priority (applied last, overwriting edge
    and interior results) to match the original if-elif chain.

    Returns
    -------
    best_tri  : (K,)    int32   – index of nearest soup triangle
    best_uvw  : (K, 3)  float32 – barycentric weights (w_A, w_B, w_C)
    best_dist : (K,)    float32 – Euclidean distance to nearest point
    """
    K = len(points)
    T = len(soup_tris)

    best_tri  = np.zeros(K,       dtype=np.int32)
    best_uvw  = np.full((K, 3), 1.0 / 3.0, dtype=np.float32)
    best_dist = np.full(K, np.inf, dtype=np.float64)

    if T == 0:
        return best_tri, best_uvw, best_dist.astype(np.float32)

    A  = soup_tris[:, 0, :].astype(np.float64)  # (T, 3)
    B  = soup_tris[:, 1, :].astype(np.float64)
    C  = soup_tris[:, 2, :].astype(np.float64)
    AB = B - A   # (T, 3)
    AC = C - A

    for k in range(K):
        P  = points[k].astype(np.float64)        # (3,)
        AP = P - A;  BP = P - B;  CP = P - C     # (T, 3) each

        d1 = np.einsum('ti,ti->t', AB, AP)
        d2 = np.einsum('ti,ti->t', AC, AP)
        d3 = np.einsum('ti,ti->t', AB, BP)
        d4 = np.einsum('ti,ti->t', AC, BP)
        d5 = np.einsum('ti,ti->t', AB, CP)
        d6 = np.einsum('ti,ti->t', AC, CP)

        # Voronoi region scalars
        va = d3 * d6 - d5 * d4
        vb = d5 * d2 - d1 * d6
        vc = d1 * d4 - d3 * d2

        # Interior barycentric (initialise; region masks overwrite below)
        denom = va + vb + vc
        inv_d = np.where(np.abs(denom) > 1e-12, 1.0 / denom, 0.0)
        u = 1.0 - (vb + vc) * inv_d   # w_A
        v = vb * inv_d                 # w_B
        w = vc * inv_d                 # w_C

        # ── edge regions (lower priority than vertices) ───────────────────
        # Edge AB: vc <= 0, d1 >= 0, d3 <= 0
        m = (vc <= 0) & (d1 >= 0) & (d3 <= 0)
        if m.any():
            den = d1[m] - d3[m]
            t_  = np.clip(np.where(den > 1e-12, d1[m] / den, 0.0), 0.0, 1.0)
            u[m] = 1.0 - t_;  v[m] = t_;  w[m] = 0.0

        # Edge AC: vb <= 0, d2 >= 0, d6 <= 0
        m = (vb <= 0) & (d2 >= 0) & (d6 <= 0)
        if m.any():
            den = d2[m] - d6[m]
            t_  = np.clip(np.where(den > 1e-12, d2[m] / den, 0.0), 0.0, 1.0)
            u[m] = 1.0 - t_;  v[m] = 0.0;  w[m] = t_

        # Edge BC: va <= 0, d4-d3 >= 0, d5-d6 >= 0
        m = (va <= 0) & ((d4 - d3) >= 0) & ((d5 - d6) >= 0)
        if m.any():
            num = d4[m] - d3[m]
            den = num + (d5[m] - d6[m])
            t_  = np.clip(np.where(den > 1e-12, num / den, 0.0), 0.0, 1.0)
            u[m] = 0.0;  v[m] = 1.0 - t_;  w[m] = t_

        # ── vertex regions (highest priority – applied last) ──────────────
        m = (d1 <= 0) & (d2 <= 0)          # vertex A
        u[m] = 1.0;  v[m] = 0.0;  w[m] = 0.0
        m = (d3 >= 0) & (d4 <= d3)         # vertex B
        u[m] = 0.0;  v[m] = 1.0;  w[m] = 0.0
        m = (d6 >= 0) & (d5 <= d6)         # vertex C
        u[m] = 0.0;  v[m] = 0.0;  w[m] = 1.0

        q     = u[:, None] * A + v[:, None] * B + w[:, None] * C  # (T, 3)
        dist2 = np.sum((P - q) ** 2, axis=1)                       # (T,)
        best_t = int(np.argmin(dist2))

        best_tri[k]  = best_t
        best_uvw[k]  = [float(u[best_t]), float(v[best_t]), float(w[best_t])]
        best_dist[k] = float(np.sqrt(max(0.0, dist2[best_t])))

    return best_tri, best_uvw, best_dist.astype(np.float32)


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

