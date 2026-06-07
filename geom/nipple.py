"""
Nipple geometry: deform cage + high-resolution surface + barycentric bindings.

Two-level representation
------------------------
1. **Deform cage** – ~4 tetrahedra; this is the simulated body that lives inside
   the PBD solver.  Base vertices are barycentric-bound to the breast surface so
   they track breast movement.  The cage rest shape is blended between
   "flat/relaxed" and "erect/firm" via a scalar *firmness* parameter in [0, 1].

2. **Hi-res surface** – a smooth revolved surface (areola disc + nipple shaft)
   whose vertices are pinned to the cage via tet-barycentric coordinates.  No
   independent DOF; pure skinning at render time.

Coordinate convention
---------------------
Matches the rest of the codebase (anatomy.py):
  x  medial(0)/lateral(+)
  y  inferior(−)/superior(+)
  z  posterior(−)/anterior(+)

The nipple is built in a local *apex frame* (n̂ = outward, t̂, b̂ = tangents)
and then transformed into world space.

Public API
----------
    from PBD_Taichi.geom.nipple import (
        find_apex_frame,
        build_nipple_cage,
        build_nipple_hires,
        bind_hires_to_cage,
        eval_hires_positions,
        NippleGeometry,
        make_nipple_pair,
    )
"""

from __future__ import annotations

import dataclasses

import numpy as np

try:
    from PBD_Taichi.geom.skin import BarycentricBindingDefinition, AnchorType
except ImportError:
    from geom.skin import BarycentricBindingDefinition, AnchorType


# ---------------------------------------------------------------------------
# Tuneable defaults (all in metres)
# ---------------------------------------------------------------------------

_AREOLA_RADIUS    = 0.022   # radius of the pigmented areola disc
_NIPPLE_RADIUS    = 0.007   # radius of the nipple base cylinder
_NIPPLE_H_RELAX   = 0.001   # cage height in relaxed (flat) state
_NIPPLE_H_FIRM    = 0.010   # cage height in firm/erect state
_NIPPLE_R_TIP_RELAX = 0.006 # tip radius in relaxed state (wide, flat)
_NIPPLE_R_TIP_FIRM  = 0.003 # tip radius in firm state (narrow, pointed)
_CAGE_N_SIDES     = 4       # number of sides for the cage base polygon
_HIRES_RINGS      = 7       # shaft profile rings
_HIRES_SEGS       = 16      # circumferential segments
_NIPPLE_BASE_LIFT = 0.0005  # metres — raise hi-res mesh off breast surface to
                             # prevent z-fighting / occlusion by breast geometry


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class NippleGeometry:
    """All geometry for one nipple (left or right).

    Cage fields
    -----------
    cage_verts      : (N_c, 3) float32  – current cage vertex positions (world)
    cage_rest_flat  : (N_c, 3) float32  – rest positions in flat/relaxed state
    cage_rest_firm  : (N_c, 3) float32  – rest positions in firm/erect state
    cage_tets       : (T_c, 4) int32    – tet vertex indices
    cage_faces      : (F_c, 3) int32    – surface triangle indices (for rendering)
    cage_bindings   : list[BarycentricBindingDefinition]
                        Breast→cage bindings (one per cage BASE vertex)

    Hi-res surface fields
    ---------------------
    hires_verts     : (N_h, 3) float32  – current hi-res vertex positions (world)
    hires_rest      : (N_h, 3) float32  – hi-res rest positions (apex frame)
    hires_faces     : (F_h, 3) int32    – hi-res triangle indices
    hires_tet_idx   : (N_h,)   int32    – which cage tet each hi-res vert is in
    hires_bary      : (N_h, 4) float32  – barycentric coords inside that tet

    State
    -----
    firmness        : float in [0, 1]   – 0=flat/relaxed, 1=firm/erect
    side            : str               – 'left' or 'right'
    apex_world      : (3,) float32      – world-space apex position
    apex_frame      : (3, 3) float32    – columns = [n̂, t̂, b̂] world-space axes
    """

    # Cage
    cage_verts:     np.ndarray
    cage_rest_flat: np.ndarray
    cage_rest_firm: np.ndarray
    cage_tets:      np.ndarray
    cage_faces:     np.ndarray
    cage_bindings:  list

    # Hi-res
    hires_verts:    np.ndarray
    hires_rest:     np.ndarray
    hires_faces:    np.ndarray
    hires_tet_idx:  np.ndarray
    hires_bary:     np.ndarray

    # Meta
    firmness:       float
    side:           str
    apex_world:     np.ndarray
    apex_frame:     np.ndarray   # (3, 3) [n̂ | t̂ | b̂]


# ---------------------------------------------------------------------------
# 1. Apex detection
# ---------------------------------------------------------------------------

def find_apex_frame(
    breast_verts: np.ndarray,                       # (V, 3) float32
    breast_faces: np.ndarray,                       # (F, 3) int32
    base_vertex_indices: np.ndarray | None = None,  # (K,) int32 — optional base-face verts
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(apex_pos, apex_frame)`` for a breast mesh.

    When *base_vertex_indices* is supplied (strongly recommended) the apex is
    found robustly as the surface vertex furthest from the attachment-base
    centroid along the base→breast-centroid direction.

    Without base indices we fall back to a surface-only scoring scheme:
    the surface vertex that maximises ``dot(pos - surface_centroid, vn)``
    — the outward-signed protrusion from the surface centroid.  This handles
    tilted / spread meshes without relying on a fixed world-space axis.

    ``apex_frame`` is a ``(3, 3)`` float32 matrix whose columns are:
      col 0 : **n̂** – outward normal at apex
      col 1 : **t̂** – first tangent (chosen to be near global +x)
      col 2 : **b̂** – second tangent = n̂ × t̂

    Parameters
    ----------
    breast_verts        : (V, 3) float32  — may include interior tet verts
    breast_faces        : (F, 3) int32    — surface triangles only
    base_vertex_indices : (K,)  int32     — vertices on the chest-attachment face
                          When given, the apex is found as the surface vertex
                          furthest from the base in the base→centroid direction.

    Returns
    -------
    apex_pos   : (3,) float32
    apex_frame : (3, 3) float32  — columns [n̂, t̂, b̂]
    """
    verts = np.asarray(breast_verts, dtype=np.float64)
    faces = np.asarray(breast_faces, dtype=np.int32)

    v0 = verts[faces[:, 0]]; v1 = verts[faces[:, 1]]; v2 = verts[faces[:, 2]]
    fn = np.cross(v1 - v0, v2 - v0)   # (F, 3) area-weighted face normals

    # Per-vertex area-weighted normals (only surface verts get non-zero values)
    vn = np.zeros_like(verts)
    np.add.at(vn, faces[:, 0], fn)
    np.add.at(vn, faces[:, 1], fn)
    np.add.at(vn, faces[:, 2], fn)
    vn_mag  = np.linalg.norm(vn, axis=1, keepdims=True) + 1e-12
    vn_unit = vn / vn_mag   # (V, 3) unit per-vertex outward normals

    # Surface vertex set (only these are candidates for the apex)
    surf_idx = np.unique(faces.flatten())   # sorted array of surface vertex indices

    if base_vertex_indices is not None:
        # ── Robust path: use base centroid to define the protrusion axis ──────
        base_idx  = np.asarray(base_vertex_indices, dtype=np.int32)
        base_cent = verts[base_idx].mean(axis=0)        # (3,) base-face centroid
        all_cent  = verts.mean(axis=0)                  # (3,) overall mesh centroid
        apex_dir  = all_cent - base_cent
        apex_dir_n = apex_dir / (np.linalg.norm(apex_dir) + 1e-12)

        # Exclude base vertices; among remaining surface verts pick max projection
        base_set = set(base_idx.tolist())
        cands = np.array([i for i in surf_idx if i not in base_set], dtype=np.int32)
        if len(cands) == 0:
            cands = surf_idx    # fallback: use all surface verts

        projs = verts[cands] @ apex_dir_n   # (len(cands),)
        apex_idx = int(cands[np.argmax(projs)])

    else:
        # ── Fallback path: no base labels → score by outward protrusion ───────
        # score_i = dot(verts[i] - surface_centroid, vn_unit[i])
        # = how far vertex i protrudes from the surface centroid in its own
        #   outward-normal direction.  The nipple scores highest because it is
        #   both far from the centroid AND has a fully consistent outward normal.
        # Only surface vertices are considered (interior verts have vn=0 → score=0).
        surf_cent = verts[surf_idx].mean(axis=0)       # (3,)
        radial    = verts - surf_cent                  # (V, 3)
        score     = np.einsum('vi,vi->v', radial, vn_unit)  # (V,) dot per vertex
        # Zero out interior vertices (they have vn≈0 already, but be explicit)
        mask = np.zeros(len(verts), dtype=bool)
        mask[surf_idx] = True
        score[~mask] = -np.inf
        apex_idx = int(np.argmax(score))

    apex_pos = verts[apex_idx].astype(np.float32)

    # Apex frame: n̂ = unit per-vertex normal at apex vertex
    n_hat = vn[apex_idx]
    n_hat = n_hat / (np.linalg.norm(n_hat) + 1e-12)

    # Build tangent frame: choose t̂ approximately along global +x (or +y if n̂ ≈ ±x)
    hint = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(n_hat, hint)) > 0.9:
        hint = np.array([0.0, 1.0, 0.0])
    t_hat = np.cross(hint, n_hat)
    t_hat = t_hat / (np.linalg.norm(t_hat) + 1e-12)
    b_hat = np.cross(n_hat, t_hat)
    b_hat = b_hat / (np.linalg.norm(b_hat) + 1e-12)

    frame = np.stack([n_hat, t_hat, b_hat], axis=1).astype(np.float32)  # (3,3)
    return apex_pos, frame


# ---------------------------------------------------------------------------
# 2. Cage geometry
# ---------------------------------------------------------------------------

def _make_cage_shape(
    apex_pos:   np.ndarray,    # (3,) world-space apex
    apex_frame: np.ndarray,    # (3, 3) [n̂ | t̂ | b̂]
    r_base:     float,         # base polygon radius
    h:          float,         # cage height along n̂
    r_tip:      float,         # tip polygon radius
    n_sides:    int = _CAGE_N_SIDES,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build one cage shape (flat or firm) as (verts, tets, faces).

    Layout
    ------
    Vertices 0..n_sides-1  : base ring (on the breast surface plane)
    Vertices n_sides..2*n_sides-1 : tip ring (h above apex)
    Vertex   2*n_sides     : tip centre (single point at top)

    Tets (n_sides of them)
    ----------------------
    Each lateral quad  [base[i], base[(i+1)%s], tip[i], tip[(i+1)%s]]
    is split into 2 tets.  Add 1 cap tet per tip-ring quad closing to
    the centre vertex → total = 2*n_sides tets + n_sides cap tets.

    Actually we use a simpler split: n_sides "wedge" tets from tip centre
    to each base edge, plus n_sides "side" tets.

    Returns (verts, tets, faces) in *local* apex frame first, then
    rotated into world space.
    """
    S = n_sides
    angles = np.linspace(0, 2 * np.pi, S, endpoint=False, dtype=np.float64)

    # Local coords:  col-0 = n̂ axis (outward),  col-1 = t̂,  col-2 = b̂
    # Base ring: at n̂=0, radius r_base in (t̂, b̂) plane
    base_local = np.zeros((S, 3), dtype=np.float64)
    base_local[:, 1] = r_base * np.cos(angles)
    base_local[:, 2] = r_base * np.sin(angles)

    # Tip ring: at n̂=h, radius r_tip
    tip_local = np.zeros((S, 3), dtype=np.float64)
    tip_local[:, 0] = h
    tip_local[:, 1] = r_tip * np.cos(angles)
    tip_local[:, 2] = r_tip * np.sin(angles)

    # Tip centre
    tip_centre_local = np.array([[h + r_tip * 0.2, 0.0, 0.0]], dtype=np.float64)

    # Concatenate local verts: [base | tip | tip_centre]
    verts_local = np.concatenate([base_local, tip_local, tip_centre_local], axis=0)

    # Transform to world
    # world_pt = apex_pos + frame @ local_pt
    verts_world = (apex_frame @ verts_local.T).T + apex_pos   # (S*2+1, 3)

    # ── Build tets ──────────────────────────────────────────────────────────
    tets: list[list[int]] = []
    for i in range(S):
        j = (i + 1) % S
        bi  = i;    bj  = j           # base ring indices
        ti  = S + i; tj = S + j       # tip ring indices
        tc  = 2 * S                   # tip centre index

        # Side tet: base_i, base_j, tip_i, tip_j
        tets.append([bi, bj, ti, tj])
        # Cap tet: tip_i, tip_j, tip_centre
        # Need 4th vertex — use one base vertex for volume
        tets.append([ti, tj, tc, bi])

    tets_np = np.array(tets, dtype=np.int32)   # (2*S, 4)

    # ── Build surface triangles ──────────────────────────────────────────────
    tris: list[list[int]] = []
    for i in range(S):
        j = (i + 1) % S
        bi = i;      bj = j
        ti = S + i;  tj = S + j
        tc = 2 * S

        # Bottom base triangle (faces inward toward breast — kept for debug)
        # tris.append([bi, bj, 0])   # skip base; it merges with breast
        # Side quad → 2 tris
        tris.append([bi, ti, bj])
        tris.append([bj, ti, tj])
        # Cap triangle
        tris.append([ti, tc, tj])

    faces_np = np.array(tris, dtype=np.int32)   # (3*S, 3)

    return verts_world.astype(np.float32), tets_np, faces_np


def build_nipple_cage(
    apex_pos:      np.ndarray,              # (3,) float32 — world-space breast apex
    apex_frame:    np.ndarray,              # (3, 3) float32 — [n̂ | t̂ | b̂]
    breast_verts:  np.ndarray,              # (V, 3) float32 — used for binding fallback
    breast_faces:  np.ndarray,              # (F, 3) int32   — used for binding fallback
    anchor_type:   AnchorType = AnchorType.BREAST_L,
    r_base:        float = _NIPPLE_RADIUS,
    h_relax:       float = _NIPPLE_H_RELAX,
    h_firm:        float = _NIPPLE_H_FIRM,
    r_tip_relax:   float = _NIPPLE_R_TIP_RELAX,
    r_tip_firm:    float = _NIPPLE_R_TIP_FIRM,
    n_sides:       int   = _CAGE_N_SIDES,
    anchor_verts:  np.ndarray | None = None,  # override: mesh to bind cage base to
    anchor_faces:  np.ndarray | None = None,  # override: faces of that mesh
) -> tuple[np.ndarray, np.ndarray, np.ndarray,
           np.ndarray, np.ndarray,
           list[BarycentricBindingDefinition]]:
    """Build the nipple deform cage and its barycentric bindings.

    The cage geometry (shape, tets, surface triangles) is always built from
    *apex_pos* and *apex_frame*.

    The cage BASE vertices are barycentric-bound to a surface mesh.  When
    *anchor_verts* / *anchor_faces* are supplied (e.g. the outer skin-shell
    layer) those are used instead of *breast_verts* / *breast_faces*.  This
    lets the nipple track the deformable skin layer rather than the underlying
    breast flesh.

    Returns
    -------
    cage_verts      : (N_c, 3) float32  — initial world positions (flat state)
    cage_rest_flat  : (N_c, 3) float32  — rest positions, flat/relaxed
    cage_rest_firm  : (N_c, 3) float32  — rest positions, firm/erect
    cage_tets       : (T_c, 4) int32
    cage_faces      : (F_c, 3) int32
    bindings        : list[BarycentricBindingDefinition]  (one per BASE vertex)
    """
    verts_flat, tets, faces = _make_cage_shape(
        apex_pos, apex_frame, r_base, h_relax, r_tip_relax, n_sides)
    verts_firm, _, _ = _make_cage_shape(
        apex_pos, apex_frame, r_base, h_firm, r_tip_firm, n_sides)

    # Choose which mesh to bind to: skin shell if provided, otherwise breast flesh
    bind_verts = anchor_verts if anchor_verts is not None else breast_verts
    bind_faces = anchor_faces if anchor_faces is not None else breast_faces

    # ── Bind base vertices (indices 0..n_sides-1) to the chosen surface ─────
    bindings = _bind_cage_to_breast(
        verts_flat[:n_sides],   # base ring only
        np.asarray(bind_verts, dtype=np.float32),
        np.asarray(bind_faces, dtype=np.int32),
        vertex_offset=0,
        anchor_type=anchor_type,
    )

    return (verts_flat.copy(), verts_flat.copy(), verts_firm.copy(),
            tets, faces, bindings)


def _bind_cage_to_breast(
    cage_base_verts: np.ndarray,     # (S, 3) float32 — base ring world positions
    breast_verts:    np.ndarray,     # (V, 3) float32
    breast_faces:    np.ndarray,     # (F, 3) int32
    vertex_offset:   int,
    anchor_type:     AnchorType,
) -> list[BarycentricBindingDefinition]:
    """For each cage base vertex find the nearest breast triangle and compute
    barycentric coordinates → ``BarycentricBindingDefinition``.
    """
    from PBD_Taichi.geom.skin import _batch_closest_triangles

    bv = breast_verts.astype(np.float64)
    bf = breast_faces.astype(np.int32)

    # Build triangle soup
    A = bv[bf[:, 0]]; B = bv[bf[:, 1]]; C = bv[bf[:, 2]]
    soup = np.stack([A, B, C], axis=1).astype(np.float32)  # (F, 3, 3)

    best_tri, best_uvw, best_dist = _batch_closest_triangles(
        cage_base_verts.astype(np.float64), soup)

    bindings: list[BarycentricBindingDefinition] = []
    for i, (ti, uvw, dist) in enumerate(zip(best_tri, best_uvw, best_dist)):
        b = BarycentricBindingDefinition()
        b.anchor_type          = anchor_type
        b.anchor_bary_face     = int(ti)
        b.anchor_bary_uvw      = uvw.astype(np.float32)
        b.skin_vertex_index    = vertex_offset + i
        b.skin_vertex_distance = float(dist)
        bindings.append(b)
    return bindings


# ---------------------------------------------------------------------------
# 3. High-resolution nipple surface
# ---------------------------------------------------------------------------

def build_nipple_hires(
    apex_pos:     np.ndarray,   # (3,) float32
    apex_frame:   np.ndarray,   # (3, 3) float32
    areola_radius: float = _AREOLA_RADIUS,
    nipple_radius: float = _NIPPLE_RADIUS,
    h_tip:         float = _NIPPLE_H_FIRM,
    segs:          int   = _HIRES_SEGS,
    rings:         int   = _HIRES_RINGS,
    firmness:      float = 0.0,
    base_lift:     float = _NIPPLE_BASE_LIFT,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate the high-resolution nipple + areola surface mesh.

    The mesh consists of:
    - **Areola disc**: a flat annular ring, inner radius ``nipple_radius``,
      outer radius ``areola_radius``.
    - **Nipple shaft**: revolved profile from base to tip.
    - **Nipple cap**: single apex vertex stitched to the top ring.

    The height of the shaft is lerp(0, h_tip, firmness) so the mesh is flat
    in the relaxed state and tall in the firm state.

    ``base_lift`` raises the entire mesh by this amount along n̂ so that even
    the flat areola disc sits slightly above the breast surface and is not
    occluded / z-fighting with the breast geometry.

    Returns
    -------
    verts : (N_h, 3) float32  — world-space positions
    faces : (F_h, 3) int32
    """
    apex_pos   = np.asarray(apex_pos, dtype=np.float64)
    apex_frame = np.asarray(apex_frame, dtype=np.float64)  # (3,3) [n̂|t̂|b̂]

    actual_h = h_tip * float(firmness)
    angles = np.linspace(0, 2 * np.pi, segs, endpoint=False)

    # ── A. Areola disc (annular ring) ──────────────────────────────────────
    # n_sides + 1 rings from inner to outer radius, all at n̂ = 0.0
    areola_rings = 3
    areola_radii = np.linspace(nipple_radius, areola_radius, areola_rings + 1)
    n_areola_verts = (areola_rings + 1) * segs
    areola_verts_local = np.zeros((n_areola_verts, 3), dtype=np.float64)
    for ri, r in enumerate(areola_radii):
        base = ri * segs
        areola_verts_local[base:base + segs, 1] = r * np.cos(angles)
        areola_verts_local[base:base + segs, 2] = r * np.sin(angles)

    # Areola faces: quads from ring i to ring i+1
    areola_faces: list[list[int]] = []
    for ri in range(areola_rings):
        for si in range(segs):
            a = ri * segs + si
            b = ri * segs + (si + 1) % segs
            c = (ri + 1) * segs + si
            d = (ri + 1) * segs + (si + 1) % segs
            areola_faces.append([a, b, d])
            areola_faces.append([a, d, c])

    # ── B. Nipple shaft (revolved profile) ─────────────────────────────────
    # Profile: cosine-smoothed cylinder; at each ring k ∈ [0..rings-1]
    #   t = k / (rings - 1)
    #   h = t * actual_h
    #   r = nipple_radius * lerp(1.0, 0.4, smoothstep(t))  (tapers to ~0.4× at tip)
    offset_shaft = n_areola_verts
    shaft_verts_local = np.zeros((rings * segs, 3), dtype=np.float64)
    for ri in range(rings):
        t = ri / max(rings - 1, 1)
        h = t * actual_h
        r = nipple_radius * (1.0 - 0.6 * _smoothstep(t))
        base = ri * segs
        shaft_verts_local[base:base + segs, 0] = h
        shaft_verts_local[base:base + segs, 1] = r * np.cos(angles)
        shaft_verts_local[base:base + segs, 2] = r * np.sin(angles)

    # Shaft faces: quads between successive rings
    shaft_faces: list[list[int]] = []
    for ri in range(rings - 1):
        for si in range(segs):
            a = offset_shaft + ri * segs + si
            b = offset_shaft + ri * segs + (si + 1) % segs
            c = offset_shaft + (ri + 1) * segs + si
            d = offset_shaft + (ri + 1) * segs + (si + 1) % segs
            shaft_faces.append([a, b, d])
            shaft_faces.append([a, d, c])

    # ── C. Nipple cap (single apex vertex) ──────────────────────────────────
    cap_centre_local = np.array([[actual_h, 0.0, 0.0]], dtype=np.float64)
    offset_cap = n_areola_verts + rings * segs
    cap_faces: list[list[int]] = []
    top_ring_base = offset_shaft + (rings - 1) * segs
    for si in range(segs):
        a = top_ring_base + si
        b = top_ring_base + (si + 1) % segs
        cap_faces.append([a, b, offset_cap])

    # ── D. Join areola inner ring to shaft base ring ─────────────────────────
    # The areola inner ring (ring 0, indices 0..segs-1) connects to the shaft
    # base ring (shaft ring 0, indices offset_shaft..offset_shaft+segs-1).
    join_faces: list[list[int]] = []
    for si in range(segs):
        a = si                                   # areola inner ring
        b = (si + 1) % segs
        c = offset_shaft + si                    # shaft base ring
        d = offset_shaft + (si + 1) % segs
        join_faces.append([a, b, d])
        join_faces.append([a, d, c])

    # ── E. Assemble all verts and faces ─────────────────────────────────────
    verts_local = np.concatenate([
        areola_verts_local,
        shaft_verts_local,
        cap_centre_local,
    ], axis=0)   # (N_h, 3)

    # Lift the entire mesh off the breast surface so the flat areola disc is
    # not coplanar with (and hidden by) the breast geometry.  Column 0 is the
    # n̂ (outward) axis in local space.
    verts_local[:, 0] += base_lift

    all_faces = (areola_faces + shaft_faces + join_faces + cap_faces)
    faces_np = np.array(all_faces, dtype=np.int32)

    # Transform to world:  world = apex_pos + frame @ local
    verts_world = (apex_frame @ verts_local.T).T + apex_pos
    return verts_world.astype(np.float32), faces_np


def _smoothstep(t: float) -> float:
    """Cubic Hermite smoothstep: 3t²−2t³ in [0, 1]."""
    t = float(np.clip(t, 0.0, 1.0))
    return t * t * (3.0 - 2.0 * t)


# ---------------------------------------------------------------------------
# 4. Barycentric binding: hi-res surface → cage tets
# ---------------------------------------------------------------------------

def bind_hires_to_cage(
    hires_verts: np.ndarray,   # (N_h, 3) float32
    cage_verts:  np.ndarray,   # (N_c, 3) float32
    cage_tets:   np.ndarray,   # (T_c, 4) int32
) -> tuple[np.ndarray, np.ndarray]:
    """Compute tet-barycentric coordinates for each hi-res vertex.

    For each hi-res vertex finds the cage tet whose centroid is closest (a fast
    spatial proxy), then computes exact barycentric coordinates inside that tet.
    If the vertex is outside all tets (e.g. areola disc sits outside the shaft
    cage) the nearest tet is used with clamped barycentrics.

    Returns
    -------
    tet_idx : (N_h,) int32   — cage tet index per hi-res vertex
    bary    : (N_h, 4) float32 — barycentric weights summing to ~1
    """
    N_h = len(hires_verts)
    T_c = len(cage_tets)

    hv = hires_verts.astype(np.float64)
    cv = cage_verts.astype(np.float64)

    # Tet centroids for fast nearest-tet lookup
    tet_centroids = cv[cage_tets].mean(axis=1)   # (T_c, 3)

    tet_idx = np.zeros(N_h, dtype=np.int32)
    bary    = np.zeros((N_h, 4), dtype=np.float32)

    for i, p in enumerate(hv):
        # 1. Find nearest tet centroid
        d2 = np.sum((tet_centroids - p) ** 2, axis=1)
        ti = int(np.argmin(d2))
        tet_idx[i] = ti

        # 2. Exact barycentric coords in that tet
        t = cage_tets[ti]
        A = cv[t[0]]; B = cv[t[1]]; C = cv[t[2]]; D = cv[t[3]]
        bary[i] = _tet_bary(p, A, B, C, D)

    return tet_idx, bary


def _tet_bary(
    P: np.ndarray,   # (3,) query
    A: np.ndarray, B: np.ndarray, C: np.ndarray, D: np.ndarray,   # tet verts
) -> np.ndarray:
    """Barycentric coordinates of P in tet ABCD.  Clamped to [0,1] and normalised."""
    T = np.stack([B - A, C - A, D - A], axis=1)   # (3, 3)
    det = np.linalg.det(T)
    if abs(det) < 1e-14:
        return np.array([0.25, 0.25, 0.25, 0.25], dtype=np.float32)
    uvw = np.linalg.solve(T, P - A)   # (3,) = weights for B, C, D relative to A
    w1 = float(uvw[0])
    w2 = float(uvw[1])
    w3 = float(uvw[2])
    w0 = 1.0 - w1 - w2 - w3
    bary = np.array([w0, w1, w2, w3], dtype=np.float32)
    # Clamp and renormalise (handles points outside the tet gracefully)
    bary = np.clip(bary, 0.0, 1.0)
    s = bary.sum()
    if s > 1e-8:
        bary /= s
    else:
        bary[:] = 0.25
    return bary


def eval_hires_positions(
    cage_verts_current: np.ndarray,   # (N_c, 3) float32
    cage_tets:          np.ndarray,   # (T_c, 4) int32
    tet_idx:            np.ndarray,   # (N_h,) int32
    bary:               np.ndarray,   # (N_h, 4) float32
) -> np.ndarray:
    """Skin hi-res vertex positions from current cage positions.

    Each hi-res vertex position is the barycentric-weighted sum of its tet's
    four vertex positions.

    Returns (N_h, 3) float32.
    """
    # Gather the 4 cage verts for every hi-res vertex: (N_h, 4, 3)
    tet_verts = cage_verts_current[cage_tets[tet_idx]]   # (N_h, 4, 3)
    # Weighted sum: bary (N_h, 4) × tet_verts (N_h, 4, 3) → (N_h, 3)
    return np.einsum('hi,hic->hc', bary, tet_verts).astype(np.float32)


# ---------------------------------------------------------------------------
# 5. Firmness state management
# ---------------------------------------------------------------------------

def set_firmness(nipple: NippleGeometry, firmness: float) -> None:
    """Update nipple.firmness and recompute cage rest positions.

    Does NOT update the hi-res mesh (call ``refresh_hires`` after this if you
    want the hi-res mesh to also change shape).

    Parameters
    ----------
    firmness : float in [0, 1]
        0 = flat/relaxed, 1 = firm/erect.
    """
    firmness = float(np.clip(firmness, 0.0, 1.0))
    nipple.firmness = firmness
    # Linear interpolation of rest positions
    nipple.cage_verts[:] = (
        (1.0 - firmness) * nipple.cage_rest_flat
        + firmness       * nipple.cage_rest_firm
    )


def refresh_hires(
    nipple:        NippleGeometry,
    areola_radius: float = _AREOLA_RADIUS,
    nipple_radius: float = _NIPPLE_RADIUS,
    h_tip:         float = _NIPPLE_H_FIRM,
    segs:          int   = _HIRES_SEGS,
    rings:         int   = _HIRES_RINGS,
    base_lift:     float = _NIPPLE_BASE_LIFT,
) -> None:
    """Rebuild the hi-res mesh for the current firmness state and rebind to cage.

    Anchors the new hi-res mesh to the **live cage centroid and live normal**
    (not the stale build-time apex_world / apex_frame) so that firmness changes
    during a running simulation always produce a mesh that is correctly aligned
    with the current breast surface.

    Mutates ``nipple.hires_verts``, ``nipple.hires_faces``,
    ``nipple.hires_tet_idx``, ``nipple.hires_bary`` in-place.
    """
    S = len(nipple.cage_bindings)   # n_sides

    # ── Build live apex position from current cage base-ring centroid ─────────
    live_centroid = nipple.cage_verts[:S].mean(axis=0).astype(np.float32)

    # ── Use live smoothed normal if available; fall back to stored frame ──────
    if hasattr(nipple, '_smooth_n'):
        live_n = nipple._smooth_n.astype(np.float64)
    else:
        live_n = nipple.apex_frame[:, 0].astype(np.float64)

    # Gram-Schmidt orthonormalise t_hat against live n_hat
    t_hat = nipple.apex_frame[:, 1].astype(np.float64)
    t_hat -= np.dot(t_hat, live_n) * live_n
    t_norm = np.linalg.norm(t_hat)
    if t_norm > 1e-8:
        t_hat /= t_norm
    else:
        hint  = np.array([0.0, 1.0, 0.0]) if abs(live_n[1]) < 0.9 else np.array([1.0, 0.0, 0.0])
        t_hat = hint - np.dot(hint, live_n) * live_n
        t_hat /= np.linalg.norm(t_hat) + 1e-12

    b_hat = np.cross(live_n, t_hat)
    b_hat /= np.linalg.norm(b_hat) + 1e-12

    live_frame = np.stack([live_n, t_hat, b_hat], axis=1).astype(np.float32)  # (3,3)

    new_verts, new_faces = build_nipple_hires(
        live_centroid, live_frame,
        areola_radius=areola_radius,
        nipple_radius=nipple_radius,
        h_tip=h_tip,
        segs=segs,
        rings=rings,
        firmness=nipple.firmness,
        base_lift=base_lift,
    )
    new_tet_idx, new_bary = bind_hires_to_cage(
        new_verts, nipple.cage_verts, nipple.cage_tets)

    nipple.hires_verts   = new_verts
    nipple.hires_rest    = new_verts.copy()
    nipple.hires_faces   = new_faces
    nipple.hires_tet_idx = new_tet_idx
    nipple.hires_bary    = new_bary


# ---------------------------------------------------------------------------
# 6. High-level factory
# ---------------------------------------------------------------------------

def make_nipple(
    breast_verts: np.ndarray,   # (V, 3) float32
    breast_faces: np.ndarray,   # (F, 3) int32
    side: str = 'left',         # 'left' or 'right'
    base_vertex_indices: np.ndarray | None = None,  # (K,) int32 — base attachment verts
    anchor_verts: np.ndarray | None = None,  # skin-shell verts to bind cage to
    anchor_faces: np.ndarray | None = None,  # skin-shell faces to bind cage to
    areola_radius: float = _AREOLA_RADIUS,
    nipple_radius: float = _NIPPLE_RADIUS,
    h_relax:       float = _NIPPLE_H_RELAX,
    h_firm:        float = _NIPPLE_H_FIRM,
    r_tip_relax:   float = _NIPPLE_R_TIP_RELAX,
    r_tip_firm:    float = _NIPPLE_R_TIP_FIRM,
    n_sides:       int   = _CAGE_N_SIDES,
    hires_segs:    int   = _HIRES_SEGS,
    hires_rings:   int   = _HIRES_RINGS,
    base_lift:     float = _NIPPLE_BASE_LIFT,
    initial_firmness: float = 0.0,
) -> NippleGeometry:
    """Build a complete ``NippleGeometry`` for one breast.

    Parameters
    ----------
    breast_verts : (V, 3) float32  — world-space breast mesh vertices (used for
                   apex detection; may be the breast flesh or skin shell)
    breast_faces : (F, 3) int32    — breast surface triangles (apex detection)
    side         : 'left' or 'right'
    base_vertex_indices : (K,) int32 — chest-attachment vertices for robust apex
                   detection (strongly recommended; see find_apex_frame)
    anchor_verts : (V2, 3) float32 — *optional* mesh whose surface the cage base
                   vertices are bound to.  When supplied (e.g. the extruded outer
                   skin-shell layer) the nipple cage tracks the skin surface rather
                   than the underlying breast flesh.  Defaults to breast_verts.
    anchor_faces : (F2, 3) int32   — surface triangles of anchor_verts.
    ...          : geometry tuning knobs (see module-level constants)
    initial_firmness : float [0,1]

    Returns
    -------
    NippleGeometry — ready to use; hi-res verts are in the flat state.
    """
    bv = np.asarray(breast_verts, dtype=np.float32)
    bf = np.asarray(breast_faces, dtype=np.int32)

    anchor = AnchorType.BREAST_L if side == 'left' else AnchorType.BREAST_R

    # 1. Apex detection
    # Always detect on the breast flesh first — gives us the correct protrusion direction
    # even when the skin shell has low resolution near the tip.
    flesh_apex, apex_frame = find_apex_frame(bv, bf, base_vertex_indices=base_vertex_indices)

    if anchor_verts is not None and anchor_faces is not None:
        # The nipple must sit on the OUTER skin-shell surface, which is ~skin_thickness
        # further outward than the breast flesh apex.  Placing the cage at the flesh apex
        # leaves it floating inside the mesh with ~14-20 mm binding distances.
        #
        # Fix: among all outer-skin-shell vertices within (2 × breast_radius) of the flesh
        # apex, pick the one furthest along the breast protrusion direction n̂.  This is the
        # skin surface apex in the breast region.
        av_arr = np.asarray(anchor_verts, dtype=np.float32)
        n_hat  = apex_frame[:, 0].astype(np.float64)   # breast outward normal

        breast_r_est = float(
            np.linalg.norm(bv.astype(np.float64) - bv.mean(axis=0), axis=1).max()
        ) * 2.0   # generous search radius
        dists_to_flesh = np.linalg.norm(av_arr - flesh_apex, axis=1)
        near_mask = dists_to_flesh < breast_r_est
        if near_mask.sum() >= 1:
            near_idx  = np.where(near_mask)[0]
            projs     = av_arr[near_idx].astype(np.float64) @ n_hat
            best_vi   = near_idx[int(np.argmax(projs))]
            apex_pos  = av_arr[best_vi].astype(np.float32)
        else:
            # Fallback: no outer-skin verts found nearby → use flesh apex
            apex_pos = flesh_apex
    else:
        apex_pos = flesh_apex

    # 2. Cage — bind to skin shell if provided, else fall back to breast flesh
    # Use areola_radius (not nipple_radius) as the cage base ring radius so that
    # all hi-res areola disc verts fall INSIDE the cage tets and receive correct
    # barycentric coordinates (using nipple_radius caused out-of-cage verts to get
    # clamped bary coords, dragging the areola off the surface on erection).
    (cage_verts, cage_rest_flat, cage_rest_firm,
     cage_tets, cage_faces, bindings) = build_nipple_cage(
        apex_pos, apex_frame, bv, bf,
        anchor_type  = anchor,
        r_base       = areola_radius,   # was nipple_radius — expanded to cover full areola
        h_relax      = h_relax,
        h_firm       = h_firm,
        r_tip_relax  = r_tip_relax,
        r_tip_firm   = r_tip_firm,
        n_sides      = n_sides,
        anchor_verts = anchor_verts,
        anchor_faces = anchor_faces,
    )

    # 3. Hi-res mesh (start flat)
    hires_verts, hires_faces = build_nipple_hires(
        apex_pos, apex_frame,
        areola_radius = areola_radius,
        nipple_radius = nipple_radius,
        h_tip         = h_firm,
        segs          = hires_segs,
        rings         = hires_rings,
        firmness      = initial_firmness,
        base_lift     = base_lift,
    )

    # 4. Bind hi-res to cage
    hires_tet_idx, hires_bary = bind_hires_to_cage(
        hires_verts, cage_verts, cage_tets)

    nipple = NippleGeometry(
        cage_verts     = cage_verts,
        cage_rest_flat = cage_rest_flat,
        cage_rest_firm = cage_rest_firm,
        cage_tets      = cage_tets,
        cage_faces     = cage_faces,
        cage_bindings  = bindings,
        hires_verts    = hires_verts,
        hires_rest     = hires_verts.copy(),
        hires_faces    = hires_faces,
        hires_tet_idx  = hires_tet_idx,
        hires_bary     = hires_bary,
        firmness       = 0.0,
        side           = side,
        apex_world     = apex_pos,
        apex_frame     = apex_frame,
    )

    # Apply initial firmness if requested
    if initial_firmness > 0.0:
        set_firmness(nipple, initial_firmness)

    return nipple


def make_nipple_pair(
    breast_l_verts: np.ndarray,   # (V, 3) float32
    breast_l_faces: np.ndarray,   # (F, 3) int32
    breast_r_verts: np.ndarray,   # (V, 3) float32
    breast_r_faces: np.ndarray,   # (F, 3) int32
    base_l_vertex_indices: np.ndarray | None = None,  # (K,) int32
    base_r_vertex_indices: np.ndarray | None = None,  # (K,) int32
    anchor_verts: np.ndarray | None = None,  # shared skin-shell verts (both nipples)
    anchor_faces: np.ndarray | None = None,  # shared skin-shell faces
    **kwargs,
) -> tuple[NippleGeometry, NippleGeometry]:
    """Build ``NippleGeometry`` for both breasts.

    *base_l_vertex_indices* and *base_r_vertex_indices* are the chest-attachment
    vertex indices for the left and right breast respectively.  When supplied
    (recommended), ``find_apex_frame`` uses them to robustly identify the nipple
    apex even for tilted/spread breast meshes.

    *anchor_verts* / *anchor_faces* is the surface mesh the cage base vertices
    are bound to.  Pass the outer skin-shell layer here so the nipple tracks the
    deformable skin rather than the underlying breast flesh.  Both nipples share
    the same anchor mesh (the skin shell covers both breasts).

    All remaining ``**kwargs`` are forwarded to ``make_nipple`` for both sides.

    Returns (nipple_left, nipple_right).
    """
    nipple_l = make_nipple(breast_l_verts, breast_l_faces, side='left',
                           base_vertex_indices=base_l_vertex_indices,
                           anchor_verts=anchor_verts, anchor_faces=anchor_faces,
                           **kwargs)
    nipple_r = make_nipple(breast_r_verts, breast_r_faces, side='right',
                           base_vertex_indices=base_r_vertex_indices,
                           anchor_verts=anchor_verts, anchor_faces=anchor_faces,
                           **kwargs)
    return nipple_l, nipple_r

