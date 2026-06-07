"""
16×16×16 (or custom resolution) deformation cage for live breast mesh rebuild.

A structured axis-aligned lattice captures the deformation field of a breast
mesh.  At rebuild time the per-cell affine maps (rest → current) are
transferred to the new rest-pose mesh so the simulation continues smoothly.

Public API
----------
    build_deform_cage(verts, res)                     → DeformCage
    compute_cage_deformation(cage, rest, current)     → affine_maps (n_cells, 3, 4)
    warp_new_verts(cage, new_rest, affine_maps)       → (M, 3) warped positions
    morph_breast_mesh(old_rest, new_target, old_current, res=8)
                                                      → (new_rest, new_current)

``morph_breast_mesh`` is the high-level entry point used by
``UnifiedTorso.rebuild_breasts()``.  It returns morphed arrays with the SAME
vertex count as the input ``old_rest``/``old_current`` so Taichi fields can be
updated in-place without reallocation.
"""
from __future__ import annotations

import dataclasses
import numpy as np


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class DeformCage:
    """Axis-aligned lattice cage for one breast mesh.

    Attributes
    ----------
    origin    : (3,) float32 – world-space minimum corner of the cage bbox
    cell_size : (3,) float32 – per-axis cell dimensions
    res       : (rx, ry, rz) – lattice resolution
    vert_cell : (N, 3) int32 – (i,j,k) cell index per source vertex
    vert_bary : (N, 8) float32 – trilinear weights to the 8 cell corners
    verts_rest: (N, 3) float32 – rest-pose vertex positions at build time
    """
    origin:     np.ndarray   # (3,) float32
    cell_size:  np.ndarray   # (3,) float32
    res:        tuple        # (rx, ry, rz) ints
    vert_cell:  np.ndarray   # (N, 3) int32
    vert_bary:  np.ndarray   # (N, 8) float32
    verts_rest: np.ndarray   # (N, 3) float32


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _assign_cells(
    verts:     np.ndarray,   # (N, 3)
    origin:    np.ndarray,   # (3,)
    cell_size: np.ndarray,   # (3,)
    res:       tuple,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (ci, cell_lin) – per-vertex cell assignment.

    ci       : (N, 3) int32 – (i,j,k) per axis, clamped to [0, res-1]
    cell_lin : (N,)   int32 – linear index = i*ry*rz + j*rz + k
    """
    rx, ry, rz = res
    norm = (verts - origin) / cell_size          # (N, 3) normalised coord
    ci   = np.floor(norm).astype(np.int32)
    ci   = np.clip(ci, 0, np.array([rx-1, ry-1, rz-1], dtype=np.int32))
    cell_lin = (ci[:, 0] * ry * rz
              + ci[:, 1] * rz
              + ci[:, 2]).astype(np.int32)
    return ci, cell_lin


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------

def build_deform_cage(verts: np.ndarray, res: tuple = (16, 16, 16)) -> DeformCage:
    """Build a cage bounding *verts* at the given lattice resolution.

    A 10 % padding is added on each side of the bounding box so that mesh
    boundary vertices do not sit on the cage edge.

    Parameters
    ----------
    verts : (N, 3) float32
    res   : lattice resolution, default ``(16, 16, 16)``
    """
    verts = np.asarray(verts, dtype=np.float32)
    lo    = verts.min(axis=0)
    hi    = verts.max(axis=0)
    pad   = 0.10 * np.maximum(hi - lo, 1e-6)
    lo   -= pad
    hi   += pad
    cell_size = (hi - lo) / np.array(res, dtype=np.float32)

    ci, _ = _assign_cells(verts, lo, cell_size, res)

    # Trilinear weights for the 8 corners of the cell
    frac = (verts - lo) / cell_size - ci.astype(np.float32)  # (N, 3) in [0,1]
    fx, fy, fz = frac[:, 0], frac[:, 1], frac[:, 2]
    bary = np.stack([
        (1-fx)*(1-fy)*(1-fz),  # 000
        (1-fx)*(1-fy)*(  fz),  # 001
        (1-fx)*(  fy)*(1-fz),  # 010
        (1-fx)*(  fy)*(  fz),  # 011
        (  fx)*(1-fy)*(1-fz),  # 100
        (  fx)*(1-fy)*(  fz),  # 101
        (  fx)*(  fy)*(1-fz),  # 110
        (  fx)*(  fy)*(  fz),  # 111
    ], axis=1).astype(np.float32)   # (N, 8)

    return DeformCage(
        origin    = lo.copy(),
        cell_size = cell_size.copy(),
        res       = tuple(int(r) for r in res),
        vert_cell = ci.copy(),
        vert_bary = bary,
        verts_rest = verts.copy(),
    )


# ---------------------------------------------------------------------------
# Affine maps: rest → current
# ---------------------------------------------------------------------------

def _cell_lin_to_ijk(lin: np.ndarray, ry: int, rz: int) -> np.ndarray:
    """Convert linear cell index array to (i, j, k) array."""
    i = lin // (ry * rz)
    j = (lin % (ry * rz)) // rz
    k = lin % rz
    return np.stack([i, j, k], axis=-1)


def compute_cage_deformation(
    cage:          DeformCage,
    verts_rest:    np.ndarray,   # (N, 3) float32 – original rest positions
    verts_current: np.ndarray,   # (N, 3) float32 – live deformed positions
) -> np.ndarray:
    """Compute a per-cell affine map: p_rest → p_current.

    For cells with ≥ 2 verts a 3 × 4 least-squares affine [A | b] is computed.
    Single-vert cells use a translation-only map.
    Empty cells inherit the map of the nearest populated cell.

    Returns
    -------
    maps : (n_cells, 3, 4) float32
        Each ``maps[c]`` encodes p_current ≈ maps[c, :, :3] @ p_rest + maps[c, :, 3].
    """
    verts_rest    = np.asarray(verts_rest,    dtype=np.float32)
    verts_current = np.asarray(verts_current, dtype=np.float32)
    rx, ry, rz = cage.res
    n_cells = rx * ry * rz

    _, cell_lin = _assign_cells(verts_rest, cage.origin, cage.cell_size, cage.res)

    # Default: identity rotation, zero translation
    maps = np.tile(np.eye(3, 4, dtype=np.float32), (n_cells, 1, 1))

    cell_verts: dict[int, list[int]] = {}
    for vi, cl in enumerate(cell_lin.tolist()):
        cell_verts.setdefault(cl, []).append(vi)

    for c, vis in cell_verts.items():
        P = verts_rest[vis].astype(np.float64)
        Q = verts_current[vis].astype(np.float64)
        if len(vis) == 1:
            maps[c, :, 3] = (Q[0] - P[0]).astype(np.float32)
        else:
            A_mat = np.hstack([P, np.ones((len(vis), 1))])
            try:
                X, _, _, _ = np.linalg.lstsq(A_mat, Q, rcond=None)  # (4, 3)
                maps[c] = X.T.astype(np.float32)                     # (3, 4)
            except np.linalg.LinAlgError:
                pass

    # Fill empty cells from nearest populated
    populated = np.array(list(cell_verts.keys()), dtype=np.int32)
    if len(populated) > 0:
        pop_ijk = _cell_lin_to_ijk(populated, ry, rz).astype(np.float32)
        all_c   = np.arange(n_cells, dtype=np.int32)
        unpop   = np.array([c for c in all_c if c not in cell_verts], dtype=np.int32)
        if len(unpop) > 0:
            up_ijk = _cell_lin_to_ijk(unpop, ry, rz).astype(np.float32)
            # Broadcast distance: (n_unpop, n_pop)
            d2     = np.sum((up_ijk[:, None, :] - pop_ijk[None, :, :])**2, axis=-1)
            nn     = populated[np.argmin(d2, axis=1)]
            maps[unpop] = maps[nn]

    return maps


# ---------------------------------------------------------------------------
# Apply maps to new mesh
# ---------------------------------------------------------------------------

def warp_new_verts(
    cage:        DeformCage,
    new_rest:    np.ndarray,   # (M, 3) float32
    affine_maps: np.ndarray,   # (n_cells, 3, 4) float32
) -> np.ndarray:
    """Apply per-cell affine maps to warp *new_rest* positions.

    Vertices outside the cage bbox are clamped to the nearest boundary cell.

    Returns (M, 3) float32.
    """
    nv = np.asarray(new_rest, dtype=np.float32)
    rx, ry, rz = cage.res
    n_cells = rx * ry * rz
    _, cell_lin = _assign_cells(nv, cage.origin, cage.cell_size, cage.res)
    cell_lin = np.clip(cell_lin, 0, n_cells - 1)

    Am = affine_maps[cell_lin]            # (M, 3, 4)
    # warped[i] = Am[i, :, :3] @ nv[i] + Am[i, :, 3]
    warped = np.einsum('mij,mj->mi', Am[:, :, :3], nv.astype(np.float64)) \
             + Am[:, :, 3]
    return warped.astype(np.float32)


# ---------------------------------------------------------------------------
# High-level convenience: morph existing breast mesh in-place
# ---------------------------------------------------------------------------

def morph_breast_mesh(
    old_rest:    np.ndarray,    # (N, 3) float32 – existing rest positions
    new_target:  np.ndarray,    # (M, 3) float32 – new target rest positions
    old_current: np.ndarray,    # (N, 3) float32 – existing live positions
    res: tuple = (8, 8, 8),
) -> tuple[np.ndarray, np.ndarray]:
    """Morph an existing breast mesh toward a new target shape.

    Uses a coarse cage built over the UNION bounding box of *old_rest* and
    *new_target* so both meshes are fully covered.  A per-cell shift vector
    (mean of new_target verts in cell – mean of old_rest verts in cell) is
    computed and applied to each old vert.  The same shift is added to the
    current (live/deformed) positions so the existing physics deformation is
    preserved.

    The output has the SAME vertex count as *old_rest* / *old_current* — only
    positions change, not topology.

    Parameters
    ----------
    old_rest    : (N, 3) – rest positions of the existing mesh
    new_target  : (M, 3) – rest positions of the new target mesh (any count)
    old_current : (N, 3) – current live positions of the existing mesh
    res         : cage resolution (default 8³)

    Returns
    -------
    new_rest    : (N, 3) float32 – morphed rest positions
    new_current : (N, 3) float32 – morphed current positions
    """
    old_rest    = np.asarray(old_rest,    dtype=np.float32)
    new_target  = np.asarray(new_target,  dtype=np.float32)
    old_current = np.asarray(old_current, dtype=np.float32)

    # Build cage from union bbox with 15% padding
    all_v = np.concatenate([old_rest, new_target], axis=0)
    lo    = all_v.min(axis=0)
    hi    = all_v.max(axis=0)
    pad   = 0.15 * np.maximum(hi - lo, 1e-6)
    lo   -= pad
    hi   += pad
    cell_size = (hi - lo) / np.array(res, dtype=np.float32)

    rx, ry, rz = res
    n_cells = rx * ry * rz

    def _lin(verts):
        norm = (verts - lo) / cell_size
        ci   = np.clip(np.floor(norm).astype(np.int32),
                       0, np.array([rx-1, ry-1, rz-1], dtype=np.int32))
        return (ci[:, 0]*ry*rz + ci[:, 1]*rz + ci[:, 2]).astype(np.int32)

    cl_old = _lin(old_rest)
    cl_new = _lin(new_target)

    # Per-cell mean for old and new, using vectorised scatter
    old_sum = np.zeros((n_cells, 3), dtype=np.float64)
    old_cnt = np.zeros(n_cells, dtype=np.int32)
    new_sum = np.zeros((n_cells, 3), dtype=np.float64)
    new_cnt = np.zeros(n_cells, dtype=np.int32)

    np.add.at(old_sum, cl_old, old_rest.astype(np.float64))
    np.add.at(old_cnt, cl_old, 1)
    np.add.at(new_sum, cl_new, new_target.astype(np.float64))
    np.add.at(new_cnt, cl_new, 1)

    both = (old_cnt > 0) & (new_cnt > 0)
    cell_shift = np.zeros((n_cells, 3), dtype=np.float32)
    if np.any(both):
        cell_shift[both] = ((new_sum[both] / new_cnt[both, None])
                          - (old_sum[both] / old_cnt[both, None])).astype(np.float32)

    # Fill cells with old verts but no new verts from nearest "both" cell
    only_old  = (old_cnt > 0) & ~both
    both_idx  = np.where(both)[0]
    if len(both_idx) > 0 and np.any(only_old):
        # (i,j,k) for both-populated cells
        bi = both_idx // (ry * rz)
        bj = (both_idx % (ry * rz)) // rz
        bk = both_idx % rz
        both_ijk = np.stack([bi, bj, bk], axis=1).astype(np.float32)
        for c in np.where(only_old)[0]:
            ci = c // (ry * rz)
            cj = (c % (ry * rz)) // rz
            ck = c % rz
            d2 = np.sum((both_ijk - np.array([ci, cj, ck], dtype=np.float32))**2, axis=1)
            cell_shift[c] = cell_shift[int(both_idx[int(np.argmin(d2))])]

    # Global fallback when cage has no overlap cells at all
    if len(both_idx) == 0:
        g = (new_target.mean(axis=0) - old_rest.mean(axis=0)).astype(np.float32)
        cell_shift[:] = g

    # ── Trilinear interpolation of node displacements ────────────────────
    # Convert per-cell mean shifts to per-node displacements by averaging
    # adjacent cells.  Each interior lattice node (i,j,k) is shared by up to
    # 8 cells; its displacement = mean of those cells' shifts.
    # This produces a C0-continuous field that prevents tet inversions at cell
    # boundaries (the piecewise-constant approach caused degenerate tets when
    # neighbouring cells had very different shifts).

    node_shift_sum = np.zeros((rx+1, ry+1, rz+1, 3), dtype=np.float64)
    node_shift_cnt = np.zeros((rx+1, ry+1, rz+1),    dtype=np.int32)

    # For each cell, scatter its shift to all 8 surrounding corner nodes
    cell_shift_3d = cell_shift.reshape(rx, ry, rz, 3)
    for di in range(2):
        for dj in range(2):
            for dk in range(2):
                # Slice: cell (cx,cy,cz) contributes to node (cx+di, cy+dj, cz+dk)
                node_shift_sum[di:rx+di, dj:ry+dj, dk:rz+dk] += cell_shift_3d
                node_shift_cnt[di:rx+di, dj:ry+dj, dk:rz+dk] += 1

    # Average; leave zero where no adjacent cells have shift data
    has_data = node_shift_cnt > 0
    node_disp = np.zeros((rx+1, ry+1, rz+1, 3), dtype=np.float32)
    node_disp[has_data] = (node_shift_sum[has_data]
                           / node_shift_cnt[has_data, None]).astype(np.float32)

    # ── Per-vert trilinear blend of the 8 surrounding lattice nodes ───────
    # Fractional position in node grid: (v - lo) / cell_size ∈ [0, rx] × ...
    node_frac = (old_rest.astype(np.float64) - lo.astype(np.float64)) / cell_size.astype(np.float64)
    n_ijk = np.clip(np.floor(node_frac).astype(np.int32),
                    np.zeros(3, dtype=np.int32),
                    np.array([rx-1, ry-1, rz-1], dtype=np.int32))
    frac = (node_frac - n_ijk.astype(np.float64)).astype(np.float32)
    fx, fy, fz = frac[:, 0], frac[:, 1], frac[:, 2]

    # Trilinear weights for 8 corner nodes: (N, 8)
    w8 = np.stack([
        (1-fx)*(1-fy)*(1-fz),  # 000
        (1-fx)*(1-fy)*fz,      # 001
        (1-fx)*fy*(1-fz),      # 010
        (1-fx)*fy*fz,          # 011
        fx*(1-fy)*(1-fz),      # 100
        fx*(1-fy)*fz,          # 101
        fx*fy*(1-fz),          # 110
        fx*fy*fz,              # 111
    ], axis=1).astype(np.float32)   # (N, 8)

    shift_pv = np.zeros((len(old_rest), 3), dtype=np.float32)
    for c_idx, (di, dj, dk) in enumerate(
            [(0,0,0),(0,0,1),(0,1,0),(0,1,1),(1,0,0),(1,0,1),(1,1,0),(1,1,1)]):
        ni = np.clip(n_ijk[:, 0] + di, 0, rx)
        nj = np.clip(n_ijk[:, 1] + dj, 0, ry)
        nk = np.clip(n_ijk[:, 2] + dk, 0, rz)
        shift_pv += (w8[:, c_idx:c_idx+1] * node_disp[ni, nj, nk]).astype(np.float32)

    return (old_rest    + shift_pv).astype(np.float32), \
           (old_current + shift_pv).astype(np.float32)


