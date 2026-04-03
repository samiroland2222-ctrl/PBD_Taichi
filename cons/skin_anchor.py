"""
Skin-spring XPBD constraints.

BaryBreastSkinConstraint
    Bilateral spring between a skin inner vert and a *barycentric point*
    on a breast surface triangle.  Both the skin vert and the three breast
    verts are free to move; the XPBD denominator accounts for all four.

KinematicSkinSpringConstraint
    One-sided spring from a skin inner vert to a kinematic target on the
    skeleton surface (clavicle / upper arm).  The target is updated every
    frame via ``update_targets()`` and treated as infinitely heavy.
"""
import taichi as ti
import numpy as np


# ---------------------------------------------------------------------------
# BaryBreastSkinConstraint
# ---------------------------------------------------------------------------

@ti.data_oriented
class BaryBreastSkinConstraint:
    """Bilateral barycentric spring: skin vert ↔ barycentric point on breast triangle.

    XPBD denominator:  w_i + u²·w_j0 + v²·w_j1 + bw²·w_j2 + α̃
    Updates (d = unit vec from breast point → skin vert):
        x_i  +=  w_i          · Δλ · d
        x_j0 -= u  · w_j0    · Δλ · d
        x_j1 -= v  · w_j1    · Δλ · d
        x_j2 -= bw · w_j2    · Δλ · d
    """

    def __init__(self, v_p, v_invm,
                 skin_idx_np: np.ndarray,   # (N,)   global skin vert index
                 tri_v0_np:   np.ndarray,   # (N,)   global breast tri vert 0
                 tri_v1_np:   np.ndarray,   # (N,)   global breast tri vert 1
                 tri_v2_np:   np.ndarray,   # (N,)   global breast tri vert 2
                 bary_uvw_np: np.ndarray,   # (N, 3) barycentric weights
                 dt: float, alpha: float = 1e-2, pretension: float = 1.0):
        self.n          = len(skin_idx_np)
        self.v_p        = v_p
        self.v_invm     = v_invm
        self.dt         = dt
        self.pretension = pretension

        self._alpha = ti.field(dtype=ti.f32, shape=())
        self._alpha[None] = alpha / (dt * dt) if self.n > 0 else 0.0

        sz = max(self.n, 1)
        self.skin_idx    = ti.field(dtype=ti.i32, shape=sz)
        self.tri_v0      = ti.field(dtype=ti.i32, shape=sz)
        self.tri_v1      = ti.field(dtype=ti.i32, shape=sz)
        self.tri_v2      = ti.field(dtype=ti.i32, shape=sz)
        self.bary_u      = ti.field(dtype=ti.f32, shape=sz)
        self.bary_v      = ti.field(dtype=ti.f32, shape=sz)
        self.bary_w      = ti.field(dtype=ti.f32, shape=sz)
        self.rest_length = ti.field(dtype=ti.f32, shape=sz)
        self.lambdaf     = ti.field(dtype=ti.f32, shape=sz)

        if self.n > 0:
            self.skin_idx.from_numpy(skin_idx_np.astype(np.int32))
            self.tri_v0.from_numpy(tri_v0_np.astype(np.int32))
            self.tri_v1.from_numpy(tri_v1_np.astype(np.int32))
            self.tri_v2.from_numpy(tri_v2_np.astype(np.int32))
            uvw = bary_uvw_np.reshape(-1, 3).astype(np.float32)
            self.bary_u.from_numpy(uvw[:, 0])
            self.bary_v.from_numpy(uvw[:, 1])
            self.bary_w.from_numpy(uvw[:, 2])

        vis_sz = max(self.n, 1)
        self._vis_verts = ti.Vector.field(3, dtype=ti.f32, shape=vis_sz * 2)
        self._vis_idx   = ti.field(dtype=ti.i32,           shape=vis_sz * 2)
        if self.n > 0:
            self._vis_idx.from_numpy(np.arange(self.n * 2, dtype=np.int32))

    @property
    def alpha(self):
        return self._alpha[None] * (self.dt * self.dt)

    @alpha.setter
    def alpha(self, v):
        self._alpha[None] = v / (self.dt * self.dt)

    @ti.kernel
    def _compute_rest_lengths(self, pretension: ti.f32):
        for k in range(self.n):
            i  = self.skin_idx[k]
            j0 = self.tri_v0[k]; j1 = self.tri_v1[k]; j2 = self.tri_v2[k]
            bu = self.bary_u[k]; bv = self.bary_v[k]; bw = self.bary_w[k]
            xt = bu * self.v_p[j0] + bv * self.v_p[j1] + bw * self.v_p[j2]
            self.rest_length[k] = (self.v_p[i] - xt).norm() * pretension

    def init_rest_status(self):
        if self.n > 0:
            self._compute_rest_lengths(self.pretension)

    def preupdate_cons(self):
        if self.n > 0:
            self.lambdaf.fill(0.0)

    def update_cons(self):
        if self.n > 0:
            self._solve()

    @ti.kernel
    def _solve(self):
        for k in range(self.n):
            i  = self.skin_idx[k]
            j0 = self.tri_v0[k]; j1 = self.tri_v1[k]; j2 = self.tri_v2[k]
            bu = self.bary_u[k]; bv = self.bary_v[k]; bw = self.bary_w[k]

            xi = self.v_p[i]
            xt = bu * self.v_p[j0] + bv * self.v_p[j1] + bw * self.v_p[j2]
            diff = xi - xt
            L = diff.norm()
            if L < 1e-8:
                continue
            C = L - self.rest_length[k]

            wi  = self.v_invm[i]
            wj0 = self.v_invm[j0] * bu * bu
            wj1 = self.v_invm[j1] * bv * bv
            wj2 = self.v_invm[j2] * bw * bw
            denom = wi + wj0 + wj1 + wj2 + self._alpha[None]
            if denom < 1e-12:
                continue

            dl = -(C + self._alpha[None] * self.lambdaf[k]) / denom
            self.lambdaf[k] += dl
            d = diff / L
            self.v_p[i]  += wi                    * dl * d
            self.v_p[j0] -= bu * self.v_invm[j0] * dl * d
            self.v_p[j1] -= bv * self.v_invm[j1] * dl * d
            self.v_p[j2] -= bw * self.v_invm[j2] * dl * d

    @ti.kernel
    def _update_vis(self):
        for k in range(self.n):
            i  = self.skin_idx[k]
            j0 = self.tri_v0[k]; j1 = self.tri_v1[k]; j2 = self.tri_v2[k]
            bu = self.bary_u[k]; bv = self.bary_v[k]; bw = self.bary_w[k]
            self._vis_verts[k * 2]     = self.v_p[i]
            self._vis_verts[k * 2 + 1] = (bu * self.v_p[j0]
                                          + bv * self.v_p[j1]
                                          + bw * self.v_p[j2])

    def get_render_draw(self, color=(0.85, 0.25, 0.15), width: float = 1.5):
        """Reddish line segments: skin vert → barycentric point on breast tri."""
        if self.n == 0:
            return lambda scene: None
        def draw(scene):
            self._update_vis()
            scene.lines(self._vis_verts, width=width,
                        indices=self._vis_idx, color=color)
        return draw


# ---------------------------------------------------------------------------
# KinematicSkinSpringConstraint
# ---------------------------------------------------------------------------

@ti.data_oriented
class KinematicSkinSpringConstraint:
    """One-sided spring: skin vert → kinematic target on skeleton surface.

    Target positions are pushed in each frame via ``update_targets()``.
    Only the skin vert moves; the target is infinitely heavy.
    """

    def __init__(self, v_p, v_invm,
                 skin_idx_np:    np.ndarray,   # (N,)   global skin vert indices
                 init_target_np: np.ndarray,   # (N, 3) initial target positions
                 dt: float, alpha: float = 1e-3, pretension: float = 1.0):
        self.n          = len(skin_idx_np)
        self.v_p        = v_p
        self.v_invm     = v_invm
        self.dt         = dt
        self.pretension = pretension

        self._alpha = ti.field(dtype=ti.f32, shape=())
        self._alpha[None] = alpha / (dt * dt) if self.n > 0 else 0.0

        sz = max(self.n, 1)
        self.skin_idx    = ti.field(dtype=ti.i32,           shape=sz)
        self.target_pos  = ti.Vector.field(3, dtype=ti.f32, shape=sz)
        self.rest_length = ti.field(dtype=ti.f32,           shape=sz)
        self.lambdaf     = ti.field(dtype=ti.f32,           shape=sz)

        if self.n > 0:
            self.skin_idx.from_numpy(skin_idx_np.astype(np.int32))
            self.target_pos.from_numpy(init_target_np.astype(np.float32))

        vis_sz = max(self.n, 1)
        self._vis_verts = ti.Vector.field(3, dtype=ti.f32, shape=vis_sz * 2)
        self._vis_idx   = ti.field(dtype=ti.i32,           shape=vis_sz * 2)
        if self.n > 0:
            self._vis_idx.from_numpy(np.arange(self.n * 2, dtype=np.int32))

    @property
    def alpha(self):
        return self._alpha[None] * (self.dt * self.dt)

    @alpha.setter
    def alpha(self, v):
        self._alpha[None] = v / (self.dt * self.dt)

    def update_targets(self, new_targets_np: np.ndarray):
        """Push updated kinematic target positions (N, 3) into Taichi field."""
        if self.n > 0:
            self.target_pos.from_numpy(new_targets_np.astype(np.float32))

    @ti.kernel
    def _compute_rest_lengths(self, pretension: ti.f32):
        for k in range(self.n):
            i = self.skin_idx[k]
            self.rest_length[k] = (self.v_p[i] - self.target_pos[k]).norm() * pretension

    def init_rest_status(self):
        if self.n > 0:
            self._compute_rest_lengths(self.pretension)

    def preupdate_cons(self):
        if self.n > 0:
            self.lambdaf.fill(0.0)

    def update_cons(self):
        if self.n > 0:
            self._solve()

    @ti.kernel
    def _solve(self):
        for k in range(self.n):
            i  = self.skin_idx[k]
            xi = self.v_p[i]
            xt = self.target_pos[k]
            diff = xi - xt
            L = diff.norm()
            if L < 1e-8:
                continue
            C = L - self.rest_length[k]
            wi = self.v_invm[i]
            denom = wi + self._alpha[None]
            if denom < 1e-12:
                continue
            dl = -(C + self._alpha[None] * self.lambdaf[k]) / denom
            self.lambdaf[k] += dl
            self.v_p[i] += wi * dl * (diff / L)

    @ti.kernel
    def _update_vis(self):
        for k in range(self.n):
            i = self.skin_idx[k]
            self._vis_verts[k * 2]     = self.v_p[i]
            self._vis_verts[k * 2 + 1] = self.target_pos[k]

    def get_render_draw(self, color=(0.25, 0.60, 0.95), width: float = 1.5):
        """Bluish line segments: skin vert → kinematic target on skeleton."""
        if self.n == 0:
            return lambda scene: None
        def draw(scene):
            self._update_vis()
            scene.lines(self._vis_verts, width=width,
                        indices=self._vis_idx, color=color)
        return draw
