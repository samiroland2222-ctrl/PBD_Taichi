"""
UnifiedTorso -- merges left + right breast tet meshes and a skin/subcut
fat shell into a single TetMesh driven by one XPBD framework.
"""
import numpy as np
import taichi as ti

try:
    from PBD_Taichi.cons import framework, deform3d, coopers, skin_anchor
    from PBD_Taichi.cons.breast import Breast
    from PBD_Taichi.geom import gtet, skin as skin_mod
    from PBD_Taichi.geom.skin import AnchorType
except ImportError:
    from cons import framework, deform3d, coopers, skin_anchor
    from cons.breast import Breast
    from geom import gtet, skin as skin_mod
    from geom.skin import AnchorType
# ---------------------------------------------------------------------------
# Lightweight adapter so build_coopers can work on a region of the unified mesh
# ---------------------------------------------------------------------------
class BreastRegion:
    """Provides the Breast-like interface that build_coopers expects."""
    def __init__(self, mesh: gtet.TetMesh,
                 base_idx_np: np.ndarray,
                 top_idx_np: np.ndarray):
        self.mesh = mesh
        self.base_idx_np = base_idx_np
        self.top_idx_np  = top_idx_np
    @property
    def verts_np(self):
        return self.mesh.v_p_ref.to_numpy()
# ---------------------------------------------------------------------------
# UnifiedTorso
# ---------------------------------------------------------------------------
class UnifiedTorso:
    """
    Unified deformable body: left breast + right breast + skin shell.
    Construction
    ------------
    1. Generate left/right breast mesh data (numpy).
    2. Generate skin shell (numpy) using the breast surface + skeleton data.
    3. Merge all three into one TetMesh.
    4. Pin breast bases + rigid skin anchors.
    5. Build PBD framework, Deform3D, Cooper's ligaments, skin-anchor springs.
    The caller drives the simulation via ``substep()``.
    """
    def __init__(
        self,
        skeleton,
        *,
        breast_height=0.05,
        breast_radius=0.08,
        breast_k=0.7,
        breast_spread=0.5,
        breast_tilt=0.2,
        breast_target_tets=300,
        rng=None,
        ribcage_verts_np=None,
        skin_n_u=20,
        skin_n_v=30,
        skin_thickness=0.005,
        skin_query_radius=0.025,
        skin_max_anchor_dist=0.04,
        g=(0.0, -9.8, 0.0),
        dt=None,
        fps=60,
        substep=6,
    ):
        import random as _random
        rng = rng or _random.Random(42)
        def _rand(center, spread_pct):
            return center + spread_pct * center * (rng.random() * 2 - 1)
        self.skeleton = skeleton
        if dt is None:
            dt = 1.0 / (fps * substep)
        self.dt = dt
        # ── 1. breast numpy data ──────────────────────────────────────────
        v_l, t_l, f_l, base_l, top_l = Breast.make_numpy(
            spread=_rand(breast_spread, 0.1),
            tilt=_rand(breast_tilt, 0.1),
            radius=_rand(breast_radius, 0.1),
            height=_rand(breast_height, 0.1),
            k=_rand(breast_k, 0.2),
            target_tets=breast_target_tets,
        )
        v_r, t_r, f_r, base_r, top_r = Breast.make_numpy(
            spread=_rand(-breast_spread, 0.1),
            tilt=_rand(breast_tilt, 0.1),
            radius=_rand(breast_radius, 0.1),
            height=_rand(breast_height, 0.1),
            k=_rand(breast_k, 0.2),
            target_tets=breast_target_tets,
        )
        self.n_left_verts  = len(v_l)
        self.n_right_verts = len(v_r)
        self.n_left_tets   = len(t_l) // 4
        self.n_right_tets  = len(t_r) // 4
        # ── 2. skin shell ─────────────────────────────────────────────────
        breast_l_outer = v_l[top_l]
        breast_r_outer = v_r[top_r]
        shell = skin_mod.generate_skin_shell(
            skeleton,
            breast_l_outer_verts=breast_l_outer,
            breast_r_outer_verts=breast_r_outer,
            ribcage_verts=ribcage_verts_np,
            n_u=skin_n_u, n_v=skin_n_v,
            thickness=skin_thickness,
            query_radius=skin_query_radius,
            max_anchor_dist=skin_max_anchor_dist,
        )
        self.shell = shell
        self.n_skin_verts = len(shell.verts)
        self.n_skin_tets  = len(shell.tets_flat) // 4
        # ── 3. merge into one TetMesh ─────────────────────────────────────
        merged_v, merged_t, merged_f = gtet.merge_numpy(
            (v_l, t_l, f_l),
            (v_r, t_r, f_r),
            (shell.verts, shell.tets_flat, shell.faces_flat),
        )
        self.mesh = gtet.TetMesh(v=merged_v, t=merged_t, f=merged_f,
                                 rho=1.0, scale=1.0)
        # ── index offsets ─────────────────────────────────────────────────
        self.left_offset  = 0
        self.right_offset = self.n_left_verts
        self.skin_offset  = self.n_left_verts + self.n_right_verts
        # breast vertex indices in the unified mesh
        self.base_l = base_l + self.left_offset
        self.top_l  = top_l  + self.left_offset
        self.base_r = base_r + self.right_offset
        self.top_r  = top_r  + self.right_offset
        # ── 4. pin breast bases ───────────────────────────────────────────
        pin_idx = list(self.base_l) + list(self.base_r)
        # pin rigid skin anchors (everything except BREAST and FREE)
        rigid_types = {AnchorType.CLAVICLE_L, AnchorType.CLAVICLE_R,
                       AnchorType.ARM_L, AnchorType.ARM_R, AnchorType.RIBCAGE}
        self._rigid_skin_local = []          # (local inner idx, atype, target idx)
        self._rigid_skin_global = []         # global vert idx in unified mesh
        for k in range(len(shell.anchor_type)):
            atype = shell.anchor_type[k]
            if atype in rigid_types:
                global_idx = k + self.skin_offset   # inner verts are first in shell
                self._rigid_skin_global.append(global_idx)
                self._rigid_skin_local.append((k, atype, shell.anchor_target[k]))
                pin_idx.append(global_idx)
        pin_np = np.array(pin_idx, dtype=np.int32)
        pin_ti = ti.field(dtype=ti.i32, shape=len(pin_np))
        pin_ti.from_numpy(pin_np)
        self.mesh.set_fixed_point(len(pin_np), pin_ti)
        self._pin_np = pin_np
        self._pin_ti = pin_ti
        # ── precompute rigid target scatter arrays ────────────────────────
        self._build_rigid_target_maps()
        # ── 5. set initial kinematic positions for rigid skin verts ──────
        self._write_rigid_skin_positions()
        # Recompute mass from final vertex positions so tet volumes are
        # consistent (some hex cells collapse when adjacent verts snap to the
        # same skeleton attachment point).
        self.mesh.reset_mass(rho=1.0)
        self.mesh.set_fixed_point(len(pin_np), pin_ti)
        self.breast_l = BreastRegion(self.mesh, self.base_l, self.top_l)
        self.breast_r = BreastRegion(self.mesh, self.base_r, self.top_r)
        # ── 7. build PBD framework ────────────────────────────────────────
        g_vec = ti.Vector(list(g))
        self.xpbd = framework.pbd_framework(
            g=g_vec, n_vert=self.mesh.n_vert, v_p=self.mesh.v_p,
            dt=dt, damp=0.99, invm=self.mesh.v_invm)
        # Deform3D for all tets (single stiffness set for now)
        self.deform = deform3d.Deform3D(
            n=self.mesh.n_tet, indices=self.mesh.t_i,
            invm=self.mesh.v_invm, pos=self.mesh.v_p,
            pos_ref=self.mesh.v_p_ref, tet_mass=self.mesh.t_mass,
            dt=dt, hydro_alpha=1e-2, devia_alpha=1e1)
        self.xpbd.add_cons(self.deform)
        # ── 8. Cooper's ligaments ─────────────────────────────────────────
        self.ligaments_l, _ = coopers.build_coopers(
            skeleton=skeleton, breast=self.breast_l, dt=dt,
            alpha=1e3, pull_only=True, max_attach_dist=0.5,
            n_ligaments=90, pretension=1.0, side='left')
        self.xpbd.add_cons(self.ligaments_l)
        self.ligaments_r, _ = coopers.build_coopers(
            skeleton=skeleton, breast=self.breast_r, dt=dt,
            alpha=1e3, pull_only=True, max_attach_dist=0.5,
            n_ligaments=90, pretension=1.0, side='right')
        self.xpbd.add_cons(self.ligaments_r)
        # ── 9. skin anchor springs (bilateral: breast-coupled) ────────────
        bilateral_skin_idx, bilateral_breast_idx = self._build_bilateral_pairs()
        self.skin_anchors = skin_anchor.SkinAnchorConstraint(
            v_p=self.mesh.v_p, v_invm=self.mesh.v_invm,
            skin_idx_np=bilateral_skin_idx,
            breast_idx_np=bilateral_breast_idx,
            dt=dt, alpha=1e-2, pretension=1.0)
        self.xpbd.add_cons(self.skin_anchors)
        # ── 10. bounding box collision ────────────────────────────────────
        # (caller may add one later via self.xpbd.add_collision)
        # ── 11. init rest status ──────────────────────────────────────────
        self.xpbd.init_rest_status()
        print(f"[UnifiedTorso] unified mesh: {self.mesh.n_vert} verts, "
              f"{self.mesh.n_tet} tets  "
              f"(breast L={self.n_left_tets}, R={self.n_right_tets}, "
              f"skin={self.n_skin_tets})")
    # ------------------------------------------------------------------
    # Rigid-skin kinematic helpers
    # ------------------------------------------------------------------
    def _build_rigid_target_maps(self):
        """Precompute per-structure scatter arrays for fast target update."""
        self._rigid_global_np = np.array(self._rigid_skin_global, dtype=np.int32)
        n = len(self._rigid_skin_local)
        if n == 0:
            self._rigid_targets_np = np.zeros((0, 3), dtype=np.float32)
            return
        self._rigid_targets_np = np.zeros((n, 3), dtype=np.float32)
        # build per-type masks and target-index arrays
        self._rigid_type  = np.array([x[1] for x in self._rigid_skin_local], dtype=np.int32)
        self._rigid_tidx  = np.array([x[2] for x in self._rigid_skin_local], dtype=np.int32)
    def _write_rigid_skin_positions(self):
        """Write kinematic target positions into v_p for pinned skin verts."""
        if len(self._rigid_skin_local) == 0:
            return
        positions = self._gather_rigid_positions()
        # write into mesh v_p and v_p_ref (rest + current)
        all_v = self.mesh.v_p.to_numpy()
        all_r = self.mesh.v_p_ref.to_numpy()
        for k, gidx in enumerate(self._rigid_skin_global):
            all_v[gidx] = positions[k]
            all_r[gidx] = positions[k]
        self.mesh.v_p.from_numpy(all_v.astype(np.float32))
        self.mesh.v_p_ref.from_numpy(all_r.astype(np.float32))
    def _gather_rigid_positions(self) -> np.ndarray:
        """Gather world positions for all rigid skin anchors from skeleton."""
        skel = self.skeleton
        clav_l = skel.get_clavicle_left_world_np()
        clav_r = skel.get_clavicle_right_world_np()
        arm_l  = skel.get_upper_arm_left_surface_np()
        arm_r  = skel.get_upper_arm_right_surface_np()
        src = {
            int(AnchorType.CLAVICLE_L): clav_l,
            int(AnchorType.CLAVICLE_R): clav_r,
            int(AnchorType.ARM_L):      arm_l,
            int(AnchorType.ARM_R):      arm_r,
        }
        n = len(self._rigid_skin_local)
        out = np.empty((n, 3), dtype=np.float32)
        for k in range(n):
            _, atype, tidx = self._rigid_skin_local[k]
            arr = src.get(int(atype))
            if arr is not None and tidx < len(arr):
                out[k] = arr[tidx]
            else:
                out[k] = self.mesh.v_p_ref.to_numpy()[self._rigid_skin_global[k]]
        return out
    def update_kinematic_skin(self):
        """Call once per frame (after skeleton.update()) to move pinned skin verts."""
        if len(self._rigid_skin_local) == 0:
            return
        positions = self._gather_rigid_positions()
        all_v = self.mesh.v_p.to_numpy()
        for k, gidx in enumerate(self._rigid_skin_global):
            all_v[gidx] = positions[k]
        self.mesh.v_p.from_numpy(all_v.astype(np.float32))
    # ------------------------------------------------------------------
    # Bilateral breast-skin pair construction
    # ------------------------------------------------------------------
    def _build_bilateral_pairs(self):
        """Return (skin_global_idx, breast_global_idx) arrays for bilateral springs."""
        shell = self.shell
        skin_idx_list   = []
        breast_idx_list = []
        # reference positions of the unified mesh (before Taichi allocation alters them)
        all_ref = self.mesh.v_p_ref.to_numpy()
        for k in range(len(shell.anchor_type)):
            atype = shell.anchor_type[k]
            if atype == AnchorType.BREAST_L:
                skin_global = k + self.skin_offset
                # anchor_target is index into breast_l_outer_verts which were
                # indexed by top_l into the left breast vertex array
                breast_global = int(self.top_l[shell.anchor_target[k]])
                skin_idx_list.append(skin_global)
                breast_idx_list.append(breast_global)
            elif atype == AnchorType.BREAST_R:
                skin_global = k + self.skin_offset
                breast_global = int(self.top_r[shell.anchor_target[k]])
                skin_idx_list.append(skin_global)
                breast_idx_list.append(breast_global)
        print(f"[UnifiedTorso] {len(skin_idx_list)} bilateral skin-breast springs")
        return (np.array(skin_idx_list,   dtype=np.int32),
                np.array(breast_idx_list, dtype=np.int32))
    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------
    def get_render_draws(self, breast_color=(0.85, 0.65, 0.55),
                         skin_color=(0.90, 0.78, 0.70)):
        """Return a list of scene-draw callables for the unified mesh."""
        mesh = self.mesh
        def draw_all(scene):
            scene.mesh(mesh.v_p, mesh.f_i, color=breast_color,
                       show_wireframe=False, two_sided=True)
        return [draw_all]
    def get_ligament_draws(self):
        return [self.ligaments_l.get_render_draw(),
                self.ligaments_r.get_render_draw()]
    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------
    def reset(self):
        self.mesh.v_p.copy_from(self.mesh.v_p_ref)
        self.mesh.reset_mass(rho=1.0)
        self.mesh.set_fixed_point(len(self._pin_np), self._pin_ti)
        self.xpbd.v_v.fill(0)
        self._write_rigid_skin_positions()
        self.xpbd.init_rest_status()
