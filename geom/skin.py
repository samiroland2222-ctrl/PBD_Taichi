"""
Parametric skin / subcutaneous-fat tetrahedral shell generator.

Produces a single-layer-thick structured hexahedral grid (split into tets)
that drapes over the anterior torso.  Inner-layer vertices are classified
by the nearest underlying anatomical structure so that they can be anchored
(kinematically or via bilateral springs) at simulation time.

Coverage region:
  • top   = neckline (just above the clavicles)
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
    'anchor_target',   # (n_u*n_v,)     int32  – nearest-vert index in the anchor structure
])


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_skin_shell(
    skeleton,
    breast_l_outer_verts: np.ndarray,
    breast_r_outer_verts: np.ndarray,
    ribcage_verts: np.ndarray | None = None,
    n_u: int = 20,
    n_v: int = 30,
    thickness: float = 0.005,
    query_radius: float = 0.025,
    max_anchor_dist: float = 0.04,
) -> SkinShellData:
    """Generate the skin-shell tet mesh.

    Parameters
    ----------
    skeleton : anatomy.Skeleton
        Must already be at its rest pose (``update()`` called).
    breast_l_outer_verts, breast_r_outer_verts : (N, 3)
        World-space positions of the *outer* (anterior) breast surface verts.
    ribcage_verts : (K, 3) or None
        World-space ribcage surface verts, if available.
    n_u, n_v : int
        Grid resolution (columns × rows).
    thickness : float  (metres)
        Shell thickness (skin + subcutaneous fat ≈ 3–8 mm).
    query_radius : float
        Horizontal search radius when building the heightfield.
    max_anchor_dist : float
        Maximum 3-D distance for an inner vert to be anchored to a structure.
    """

    # ── 1. collect tagged surface points ──────────────────────────────────
    clav_l  = skeleton.get_clavicle_left_world_np()
    clav_r  = skeleton.get_clavicle_right_world_np()
    arm_l   = skeleton.get_upper_arm_left_surface_np()
    arm_r   = skeleton.get_upper_arm_right_surface_np()

    tagged: list[tuple[int, np.ndarray]] = [
        (AnchorType.BREAST_L,   breast_l_outer_verts),
        (AnchorType.BREAST_R,   breast_r_outer_verts),
        (AnchorType.CLAVICLE_L, clav_l),
        (AnchorType.CLAVICLE_R, clav_r),
        (AnchorType.ARM_L,      arm_l),
        (AnchorType.ARM_R,      arm_r),
    ]
    if ribcage_verts is not None and len(ribcage_verts) > 0:
        tagged.append((AnchorType.RIBCAGE, ribcage_verts))

    # Flatten all surface points for heightfield query
    all_pts = np.concatenate([p for _, p in tagged], axis=0)   # (P, 3)

    # ── 2. determine grid coverage region ─────────────────────────────────
    cp = skeleton.chest_pos                          # (3,)
    clav_y = cp[1] + skeleton._clavicle_l_offset[1]  # SC joint y
    y_top  = clav_y + 0.015                           # slightly above SC joint
    y_bot  = cp[1] - 0.15                             # inferior rib margin
    x_half = float(np.abs(all_pts[:, 0]).max()) + 0.01
    x_half = max(x_half, 0.16)

    grid_x = np.linspace(-x_half, x_half, n_u, dtype=np.float32)
    grid_y = np.linspace(y_top,   y_bot,  n_v, dtype=np.float32)   # top→bottom
    dx = grid_x[1] - grid_x[0] if n_u > 1 else 1.0
    # dy = grid_y[1] - grid_y[0] if n_v > 1 else 1.0  # negative

    # ── 3. build heightfield (inner-surface z) ────────────────────────────
    # Base torso profile: a gentle parabolic cross-section
    chest_z    = float(cp[2])
    base_depth = 0.04                                 # how far the base torso extends forward
    inner_z = np.empty((n_v, n_u), dtype=np.float32)

    for j in range(n_v):
        for i in range(n_u):
            xg = grid_x[i]
            yg = grid_y[j]
            # parabolic cross-section
            t_lat = min((xg / x_half) ** 2, 1.0)
            z_base = chest_z + base_depth * (1.0 - t_lat)
            # query underlying anatomy
            dxy  = np.sqrt((all_pts[:, 0] - xg) ** 2 +
                           (all_pts[:, 1] - yg) ** 2)
            near = dxy < query_radius
            z_anat = all_pts[near, 2].max() if near.any() else z_base
            inner_z[j, i] = max(z_base, z_anat)

    # ── 4. compute outward normals via finite differences ─────────────────
    normals = np.zeros((n_v, n_u, 3), dtype=np.float32)
    for j in range(n_v):
        for i in range(n_u):
            if 0 < i < n_u - 1:
                dzdx = (inner_z[j, i + 1] - inner_z[j, i - 1]) / (2 * dx)
            elif i == 0:
                dzdx = (inner_z[j, 1] - inner_z[j, 0]) / dx
            else:
                dzdx = (inner_z[j, -1] - inner_z[j, -2]) / dx

            dy_val = grid_y[1] - grid_y[0] if n_v > 1 else 1.0
            if 0 < j < n_v - 1:
                dzdy = (inner_z[j + 1, i] - inner_z[j - 1, i]) / (2 * dy_val)
            elif j == 0:
                dzdy = (inner_z[1, i] - inner_z[0, i]) / dy_val
            else:
                dzdy = (inner_z[-1, i] - inner_z[-2, i]) / dy_val

            n = np.array([-dzdx, -dzdy, 1.0], dtype=np.float32)
            normals[j, i] = n / (np.linalg.norm(n) + 1e-12)

    # ── 5. build inner + outer vertex arrays ──────────────────────────────
    n_inner = n_u * n_v
    verts = np.empty((n_inner * 2, 3), dtype=np.float32)

    for j in range(n_v):
        for i in range(n_u):
            idx = j * n_u + i
            p_inner = np.array([grid_x[i], grid_y[j], inner_z[j, i]],
                               dtype=np.float32)
            verts[idx]           = p_inner
            verts[idx + n_inner] = p_inner + thickness * normals[j, i]

    inner_idx = np.arange(n_inner, dtype=np.int32)
    outer_idx = np.arange(n_inner, n_inner * 2, dtype=np.int32)

    # ── 6. build hex → 6-tet decomposition ────────────────────────────────
    off = n_inner  # offset from inner to outer layer
    tets: list[list[int]] = []

    for j in range(n_v - 1):
        for i in range(n_u - 1):
            # inner (bottom) quad: v0, v1, v2, v3
            v0 = j * n_u + i
            v1 = j * n_u + (i + 1)
            v2 = (j + 1) * n_u + (i + 1)
            v3 = (j + 1) * n_u + i
            # outer (top) quad:  v4, v5, v6, v7
            v4, v5, v6, v7 = v0 + off, v1 + off, v2 + off, v3 + off

            # Prism A  (bottom: v0,v1,v2  top: v4,v5,v6)
            tets.append([v0, v1, v2, v5])
            tets.append([v0, v5, v2, v6])
            tets.append([v0, v5, v6, v4])
            # Prism B  (bottom: v0,v2,v3  top: v4,v6,v7)
            tets.append([v0, v2, v3, v6])
            tets.append([v0, v6, v3, v7])
            tets.append([v0, v6, v7, v4])

    tets_np = np.array(tets, dtype=np.int32)
    faces   = gtet.extract_surface_triangles(verts, tets_np)

    tets_flat  = tets_np.flatten().astype(np.int32)
    faces_flat = faces.flatten().astype(np.int32)

    # ── 7. classify inner verts by nearest structure ──────────────────────
    anchor_type   = np.full(n_inner, AnchorType.FREE, dtype=np.int32)
    anchor_target = np.full(n_inner, -1,               dtype=np.int32)

    for tag, pts in tagged:
        if len(pts) == 0:
            continue
        # vectorised 3-D distances from each inner vert to each structure pt
        # inner_verts: (n_inner, 3)   pts: (M, 3)
        iv = verts[:n_inner]
        diff = iv[:, None, :] - pts[None, :, :]    # (n_inner, M, 3)
        dists = np.linalg.norm(diff, axis=2)        # (n_inner, M)
        min_idx  = dists.argmin(axis=1)             # (n_inner,)
        min_dist = dists[np.arange(n_inner), min_idx]

        # update where this structure is closer than the current best
        update = (min_dist < max_anchor_dist)
        # only overwrite if this structure is closer than existing assignment
        for k in np.where(update)[0]:
            # compute existing best distance (if any)
            if anchor_type[k] == AnchorType.FREE:
                existing = np.inf
            else:
                existing = _anchor_dist(k, anchor_type[k], anchor_target[k],
                                        verts, tagged)
            if min_dist[k] < existing:
                anchor_type[k]   = int(tag)
                anchor_target[k] = int(min_idx[k])

    n_tets = len(tets_np)
    print(f"[SkinShell] {n_inner * 2} verts, {n_tets} tets, "
          f"grid {n_u}×{n_v}, thickness={thickness*1000:.1f}mm")
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
    )


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _anchor_dist(k, atype, aidx, verts, tagged):
    """3-D distance from inner vert *k* to its current anchor target."""
    for tag, pts in tagged:
        if int(tag) == int(atype):
            return float(np.linalg.norm(verts[k] - pts[aidx]))
    return np.inf


def _print_anchor_stats(anchor_type):
    for at in AnchorType:
        n = int((anchor_type == at).sum())
        if n > 0:
            print(f"  {at.name:14s}: {n}")

