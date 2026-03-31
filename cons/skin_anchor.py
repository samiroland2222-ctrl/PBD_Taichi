"""
Skin-anchor XPBD constraint -- bilateral distance springs between
inner-skin verts and breast surface verts.
Rigid anchors (clavicle / arm / ribcage) are handled by pinning
(invm=0) and writing kinematic positions directly in UnifiedTorso.
"""
import taichi as ti
import numpy as np
@ti.data_oriented
class SkinAnchorConstraint:
    """Bilateral distance springs between inner-skin verts and breast verts."""
    def __init__(self, v_p, v_invm,
                 skin_idx_np: np.ndarray,
                 breast_idx_np: np.ndarray,
                 dt: float, alpha: float = 1e-2,
                 pretension: float = 1.0):
        self.n = len(skin_idx_np)
        self.v_p = v_p
        self.v_invm = v_invm
        self.dt = dt
        self.pretension = pretension
        self._alpha = ti.field(dtype=ti.f32, shape=())
        self._alpha[None] = alpha / (dt * dt) if self.n > 0 else 0.0
        sz = max(self.n, 1)
        self.skin_idx    = ti.field(dtype=ti.i32, shape=sz)
        self.breast_idx  = ti.field(dtype=ti.i32, shape=sz)
        self.rest_length = ti.field(dtype=ti.f32, shape=sz)
        self.lambdaf     = ti.field(dtype=ti.f32, shape=sz)
        if self.n > 0:
            self.skin_idx.from_numpy(skin_idx_np.astype(np.int32))
            self.breast_idx.from_numpy(breast_idx_np.astype(np.int32))
    # -- compliance property ------------------------------------------------
    @property
    def alpha(self):
        return self._alpha[None] * (self.dt * self.dt)
    @alpha.setter
    def alpha(self, v):
        self._alpha[None] = v / (self.dt * self.dt)
    # -- standard XPBD interface -------------------------------------------
    @ti.kernel
    def _compute_rest_lengths(self, pretension: ti.f32):
        for k in range(self.n):
            i = self.skin_idx[k]
            j = self.breast_idx[k]
            self.rest_length[k] = (self.v_p[i] - self.v_p[j]).norm() * pretension
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
            i = self.skin_idx[k]
            j = self.breast_idx[k]
            xi = self.v_p[i]
            xj = self.v_p[j]
            diff = xi - xj
            L = diff.norm()
            if L < 1e-8:
                continue
            C = L - self.rest_length[k]
            wi = self.v_invm[i]
            wj = self.v_invm[j]
            denom = wi + wj + self._alpha[None]
            if denom < 1e-12:
                continue
            delta_lambda = -(C + self._alpha[None] * self.lambdaf[k]) / denom
            self.lambdaf[k] += delta_lambda
            d = diff / L
            self.v_p[i] += wi * delta_lambda * d
            self.v_p[j] -= wj * delta_lambda * d
