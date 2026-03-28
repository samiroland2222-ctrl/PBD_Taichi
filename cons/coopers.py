"""
Cooper's Ligaments constraint for the breast simulation.

Phase 1 – Simple anchor springs
────────────────────────────────
Each ligament is a distance spring between:
  • a fixed world-space anchor point on the clavipectoral fascia
  • the closest surface vertex of the breast mesh

The spring is one-sided when rest_length is used as an upper bound
(only pulls, never pushes), which mimics the collagen strand behaviour:
ligaments resist elongation but offer no compression resistance.

Phase 2 – Branching (planned)
──────────────────────────────
Each fascia anchor fans out to a small cluster of nearby breast vertices
with weights that fall off with distance, modelling the dendritic
branching described in the anatomical literature.
"""

import taichi as ti
import numpy as np


@ti.data_oriented
class CoopersLigaments:
    """
    Phase 1: one spring per anchor point, connecting a world-space fascia
    anchor to one breast-surface vertex.

    Parameters
    ----------
    breast_pos      : ti.MatrixField (n_vert × 3) – live breast positions
    breast_invm     : ti.Field       (n_vert,)     – inverse masses
    anchor_pos_np   : np.ndarray     (n_anchors, 3) – initial anchor world pos
    surface_idx_np  : np.ndarray     (n_anchors,)   – breast vertex index per anchor
    dt              : float
    alpha           : float – compliance (higher = softer ligament)
    pull_only       : bool  – if True, springs only pull (resist stretch),
                              never push (no compression)
    """

    def __init__(self,
                 breast_pos,
                 breast_invm,
                 anchor_pos_np: np.ndarray,
                 surface_idx_np: np.ndarray,
                 dt: float,
                 alpha: float = 1e-4,
                 pull_only: bool = True,
                 pretension: float = 1.0):
        """
        pretension: rest_length = pretension * initial_distance.
            < 1.0 means springs are always pulling even from rest pose,
            making the alpha slider immediately visible.
            1.0 = no pretension (springs only activate when stretched).
        """

        self.n = len(surface_idx_np)
        self.breast_pos  = breast_pos
        self.breast_invm = breast_invm
        self.pull_only   = pull_only
        self.dt          = dt
        self.pretension  = pretension

        # Compliance as Taichi scalar field so it can be updated at runtime
        self._alpha = ti.field(dtype=ti.f32, shape=())
        self._alpha[None] = alpha / (dt * dt)

        # Anchor world positions (updated each frame by the skeleton)
        self.anchor_pos = ti.Vector.field(3, dtype=ti.f32, shape=self.n)
        self.anchor_pos.from_numpy(anchor_pos_np.astype(np.float32))

        # Which breast vertex each anchor attaches to
        self.surface_idx = ti.field(dtype=ti.i32, shape=self.n)
        self.surface_idx.from_numpy(surface_idx_np.astype(np.int32))

        # Rest lengths – computed once from initial configuration
        self.rest_length = ti.field(dtype=ti.f32, shape=self.n)
        self.lambdaf     = ti.field(dtype=ti.f32, shape=self.n)

        # Visualisation: line segments (anchor → breast vertex)
        self._vis_verts = ti.Vector.field(3, dtype=ti.f32, shape=self.n * 2)
        self._vis_idx   = ti.field(dtype=ti.i32, shape=self.n * 2)
        idx_np = np.array([[i*2, i*2+1] for i in range(self.n)],
                          dtype=np.int32).flatten()
        self._vis_idx.from_numpy(idx_np)

    # ------------------------------------------------------------------
    @property
    def alpha(self):
        return self._alpha[None] * (self.dt * self.dt)

    @alpha.setter
    def alpha(self, v):
        self._alpha[None] = v / (self.dt * self.dt)

    # ------------------------------------------------------------------
    def update_anchors(self, anchor_pos_np: np.ndarray):
        """Call each frame with the skeleton's current fascia vertex positions."""
        self.anchor_pos.from_numpy(anchor_pos_np.astype(np.float32))

    # ------------------------------------------------------------------
    @ti.kernel
    def _compute_rest_lengths(self, pretension: ti.f32):
        for k in range(self.n):
            j = self.surface_idx[k]
            self.rest_length[k] = (self.anchor_pos[k] - self.breast_pos[j]).norm() * pretension

    def init_rest_status(self):
        self._compute_rest_lengths(self.pretension)

    def preupdate_cons(self):
        self.lambdaf.fill(0.0)

    def update_cons(self):
        self._solve(self.pull_only)

    @ti.kernel
    def _solve(self, pull_only: bool):
        for k in range(self.n):
            j = self.surface_idx[k]
            xj  = self.breast_pos[j]
            xa  = self.anchor_pos[k]
            xaj = xj - xa
            L   = xaj.norm()
            if L < 1e-8:
                continue
            C = L - self.rest_length[k]
            if pull_only and C <= 0.0:
                continue   # compressed → do nothing
            wj = self.breast_invm[j]
            # anchor is part of the skeleton → infinite mass → w=0
            # so only the breast vertex moves
            denom = wj + self._alpha[None]
            delta_lambda = -(C + self._alpha[None] * self.lambdaf[k]) / denom
            self.lambdaf[k] += delta_lambda
            self.breast_pos[j] += wj * delta_lambda * (xaj / L)

    # ------------------------------------------------------------------
    @ti.kernel
    def _update_vis(self):
        for k in range(self.n):
            j = self.surface_idx[k]
            self._vis_verts[k * 2]     = self.anchor_pos[k]
            self._vis_verts[k * 2 + 1] = self.breast_pos[j]

    def get_render_draw(self, color=(0.9, 0.9, 0.4), width=1.5):
        def draw(scene):
            self._update_vis()
            scene.lines(self._vis_verts, width=width,
                        indices=self._vis_idx, color=color)
        return draw


# ---------------------------------------------------------------------------
# Factory: build a CoopersLigaments from a Skeleton + breast mesh
# ---------------------------------------------------------------------------

def build_coopers(skeleton,
                  breast_pos_np: np.ndarray,
                  breast_surface_idx_np: np.ndarray,
                  breast_pos_field,
                  breast_invm_field,
                  dt: float,
                  alpha: float = 1e-4,
                  pull_only: bool = True,
                  max_attach_dist: float = 0.25,
                  n_ligaments: int = 60,
                  outer_z_min: float = 0.005,
                  pretension: float = 0.7):
    """
    Build Cooper's ligaments from the LEFT pectoral bone surface to the
    outer surface of the left breast.

    Anchors are sampled from the left pectoral surface verts.
    Targets are the outer breast surface verts (z > outer_z_min), chosen
    to be evenly spread using farthest-point-style greedy selection so
    ligaments cover the whole breast surface rather than clustering.

    Parameters
    ----------
    skeleton              : geom.anatomy.Skeleton
    breast_pos_np         : (n_vert, 3) reference positions of ALL breast verts
    breast_surface_idx_np : (n_surface,) indices into breast_pos_np that are
                            on the surface (from TetMesh.f_i)
    breast_pos_field      : ti.MatrixField – live positions
    breast_invm_field     : ti.Field       – inverse masses
    max_attach_dist       : anchors whose nearest outer-surface vert is farther
                            than this are discarded
    n_ligaments           : how many ligament springs to create
    outer_z_min           : minimum z to be considered "outer" surface
                            (excludes the flat base at z≈0)
    """
    # ── outer breast surface verts only ──────────────────────────────────
    # Cooper's ligaments run from the clavicle/fascia (above) through
    # breast tissue to the outer surface. Target the full outer surface
    # (excluding only the flat posterior base at z≈0).
    all_surf_verts = breast_pos_np[breast_surface_idx_np]
    outer_mask     = all_surf_verts[:, 2] > outer_z_min
    outer_local    = np.where(outer_mask)[0]
    outer_global   = breast_surface_idx_np[outer_local]
    outer_verts    = breast_pos_np[outer_global]

    # ── evenly distribute n_ligaments targets across outer surface ────────
    # Greedy farthest-point sampling ensures even coverage.
    n_targets = min(n_ligaments, len(outer_global))
    selected  = [0]
    min_dists = np.full(len(outer_verts), np.inf)
    for _ in range(n_targets - 1):
        d = np.linalg.norm(outer_verts - outer_verts[selected[-1]], axis=1)
        min_dists = np.minimum(min_dists, d)
        selected.append(int(np.argmax(min_dists)))
    target_global = outer_global[selected]   # (n_targets,) breast vert indices
    target_verts  = outer_verts[selected]    # (n_targets, 3)

    # ── left pectoral surface anchors ────────────────────────────────────
    pec_anchors = skeleton.get_pec_left_surface_anchors_np()  # (n_pec_surf, 3)

    # For each target breast vert, pick the nearest pectoral surface vert
    # as its anchor — this naturally spreads anchors across the bone.
    chosen_anchor_pos = []
    chosen_anchor_idx = []   # index into pec_anchors array
    kept_target_global = []

    for ti_idx, tgt in zip(target_global, target_verts):
        dists = np.linalg.norm(pec_anchors - tgt, axis=1)
        nearest_ai   = int(np.argmin(dists))
        nearest_dist = dists[nearest_ai]
        if nearest_dist > max_attach_dist:
            continue
        chosen_anchor_pos.append(pec_anchors[nearest_ai])
        chosen_anchor_idx.append(nearest_ai)
        kept_target_global.append(int(ti_idx))

    if len(kept_target_global) == 0:
        raise RuntimeError(
            "[CoopersLigaments] No ligaments could be attached — "
            "pectoral bone may be too far from the breast surface. "
            f"min pec-to-breast dist: "
            f"{np.linalg.norm(pec_anchors[:,None]-target_verts[None],axis=2).min():.4f}m, "
            f"max_attach_dist={max_attach_dist}")

    anchor_pos_np     = np.array(chosen_anchor_pos,  dtype=np.float32)
    surface_idx_np    = np.array(kept_target_global,  dtype=np.int32)
    chosen_anchor_idx = np.array(chosen_anchor_idx,   dtype=np.int32)

    print(f"[CoopersLigaments] {len(surface_idx_np)} ligaments attached "
          f"across outer breast surface "
          f"({n_targets - len(surface_idx_np)} discarded as too far)")

    lig = CoopersLigaments(
        breast_pos    = breast_pos_field,
        breast_invm   = breast_invm_field,
        anchor_pos_np = anchor_pos_np,
        surface_idx_np= surface_idx_np,
        dt            = dt,
        alpha         = alpha,
        pull_only     = pull_only,
        pretension    = pretension,
    )
    return lig, chosen_anchor_idx

