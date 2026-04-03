"""
UnifiedTorso -- merges left + right breast tet meshes and a skin/subcut
fat shell into a single TetMesh driven by one XPBD framework.
"""
import numpy as np
import taichi as ti

from PBD_Taichi.geom.distance_field import BasicTetMesh

try:
    from PBD_Taichi.cons import framework, deform3d, coopers, skin_anchor
    from PBD_Taichi.cons.breast import Breast
    from PBD_Taichi.cons.skin_anchor import BaryBreastSkinConstraint, KinematicSkinSpringConstraint
    from PBD_Taichi.geom import gtet, skin as skin_mod
    from PBD_Taichi.geom.skin import AnchorType, BarycentricBindingDefinition
except ImportError:
    from cons import framework, deform3d, coopers, skin_anchor
    from cons.breast import Breast
    from cons.skin_anchor import BaryBreastSkinConstraint, KinematicSkinSpringConstraint
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

    Anchor types:
      RIBCAGE          – pinned (invm=0), kinematic position written each frame
      CLAVICLE_L/R     – KinematicSkinSpringConstraint, barycentric on clavicle mesh
      ARM_L/R          – KinematicSkinSpringConstraint, barycentric on arm mesh
      BREAST_L/R       – BaryBreastSkinConstraint, barycentric on breast tet surface
      FREE             – no constraint
    """
    # Types that stay kinematically pinned (invm=0)
    _PINNED_TYPES = frozenset({AnchorType.RIBCAGE})
    # Types that become elastic springs toward the skeleton surface
    _SPRING_TYPES = frozenset({AnchorType.CLAVICLE_L, AnchorType.CLAVICLE_R,
                               AnchorType.ARM_L, AnchorType.ARM_R})

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
        ribcage_faces_np=None,
        skin_target_n_tets=1000,
        skin_thickness=0.01,
        arm_init_abduction=0.4,   # radians – A-pose for skin shell init
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
        # Set arms to A-pose so the skin grid overlaps the arm capsule and
        # can be properly anchored/initialised against it.
        # The skeleton is left in this pose; the simulation starts here and
        # gravity + joint sliders drive it from A-pose onward.
        skeleton.arm_left_abduction  = arm_init_abduction
        skeleton.arm_right_abduction = arm_init_abduction
        skeleton.update()

        breast_l_outer = v_l[top_l]  # kept for compatibility (not used in skin shell)
        breast_r_outer = v_r[top_r]
        skin_shell, barycentric_bindings = skin_mod.generate_skin_shell(
            skeleton,
            breast_l_verts=v_l,
            breast_l_faces=f_l.reshape(-1, 3),
            breast_r_verts=v_r,
            breast_r_faces=f_r.reshape(-1, 3),
            ribcage_verts=ribcage_verts_np,
            ribcage_faces=(ribcage_faces_np
                           if ribcage_faces_np is not None
                           else skeleton.get_ribcage_faces_np()),
            target_n_tets=skin_target_n_tets,
            thickness=skin_thickness,
        )
        self.skin_shell: BasicTetMesh = skin_shell
        self.barycentric_bindings: list[BarycentricBindingDefinition] = barycentric_bindings

        # ── 3. merge breast meshes only (skin is visual-only for now) ─────
        merged_v, merged_t, merged_f = gtet.merge_numpy(
            (v_l, t_l, f_l),
            (v_r, t_r, f_r),
        )
        self.mesh = gtet.TetMesh(v=merged_v, t=merged_t, f=merged_f,
                                 rho=1.0, scale=1.0)

        # ── 3b. skin trimesh – Taichi fields for rendering only ───────────
        n_sv = len(shell.verts)
        self.skin_v = ti.Vector.field(3, dtype=ti.f32, shape=max(1, n_sv))
        if n_sv > 0:
            self.skin_v.from_numpy(shell.verts.astype(np.float32))
        n_sf = len(shell.faces_flat)
        self.skin_f = ti.field(dtype=ti.i32, shape=max(1, n_sf))
        if n_sf > 0:
            self.skin_f.from_numpy(shell.faces_flat.astype(np.int32))

        # ── index offsets ─────────────────────────────────────────────────
        self.left_offset  = 0
        self.right_offset = self.n_left_verts

        # breast vertex indices in the unified mesh
        self.base_l = base_l + self.left_offset
        self.top_l  = top_l  + self.left_offset
        self.base_r = base_r + self.right_offset
        self.top_r  = top_r  + self.right_offset

        # ── 4. pin breast bases only (skin not yet in simulation) ─────────
        pin_idx = list(self.base_l) + list(self.base_r)

        pin_np = np.array(pin_idx, dtype=np.int32)
        pin_ti = ti.field(dtype=ti.i32, shape=len(pin_np))
        pin_ti.from_numpy(pin_np)
        self.mesh.set_fixed_point(len(pin_np), pin_ti)
        self._pin_np = pin_np
        self._pin_ti = pin_ti

        self.breast_l = BreastRegion(self.mesh, self.base_l, self.top_l)
        self.breast_r = BreastRegion(self.mesh, self.base_r, self.top_r)

        # ── 5. build PBD framework ────────────────────────────────────────
        g_vec = ti.Vector(list(g))
        self.xpbd = framework.pbd_framework(
            g=g_vec, n_vert=self.mesh.n_vert, v_p=self.mesh.v_p,
            dt=dt, damp=0.99, invm=self.mesh.v_invm)

        # Deform3D for all tets
        self.deform = deform3d.Deform3D(
            n=self.mesh.n_tet, indices=self.mesh.t_i,
            invm=self.mesh.v_invm, pos=self.mesh.v_p,
            pos_ref=self.mesh.v_p_ref, tet_mass=self.mesh.t_mass,
            dt=dt, hydro_alpha=1e-2, devia_alpha=1e1)
        self.xpbd.add_cons(self.deform)

        # ── 6. Cooper's ligaments ─────────────────────────────────────────
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

        # ── 7. skin constraints (empty – skin not yet in simulation) ──────
        _empty_i = np.zeros(0, dtype=np.int32)
        _empty_f3 = np.zeros((0, 3), dtype=np.float32)
        self.skin_anchors = BaryBreastSkinConstraint(
            v_p=self.mesh.v_p, v_invm=self.mesh.v_invm,
            skin_idx_np=_empty_i,
            tri_v0_np=_empty_i, tri_v1_np=_empty_i, tri_v2_np=_empty_i,
            bary_uvw_np=_empty_f3,
            dt=dt, alpha=1e-2, pretension=1.0)
        self.xpbd.add_cons(self.skin_anchors)

        self.skeleton_skin_springs = KinematicSkinSpringConstraint(
            v_p=self.mesh.v_p, v_invm=self.mesh.v_invm,
            skin_idx_np=_empty_i, init_target_np=_empty_f3,
            dt=dt, alpha=1e-3, pretension=1.0)
        self.xpbd.add_cons(self.skeleton_skin_springs)

        # ── 8. init rest status ───────────────────────────────────────────
        self.xpbd.init_rest_status()
        print(f"[UnifiedTorso] {self.mesh.n_vert} verts, {self.mesh.n_tet} tets  "
              f"(L={self.n_left_tets} R={self.n_right_tets})  ")

    # ------------------------------------------------------------------
    # Skin kinematic update – no-op until skin is brought into simulation
    # ------------------------------------------------------------------
    def update_kinematic_skin(self):
        pass

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------
    def get_render_draws(self, breast_color=(0.85, 0.65, 0.55)):
        mesh = self.mesh
        def draw_all(scene):
            scene.mesh(mesh.v_p, mesh.f_i, color=breast_color,
                       show_wireframe=False, two_sided=True)
        return [draw_all]

    def get_skin_draws(self, color=(0.90, 0.78, 0.68)):
        """Render the raycast skin surface (visual only)."""
        sv, sf = self.skin_v, self.skin_f
        def draw_skin(scene):
            scene.mesh(sv, sf, color=color, two_sided=True)
        return [draw_skin]

    def get_ligament_draws(self):
        return [self.ligaments_l.get_render_draw(),
                self.ligaments_r.get_render_draw()]

    def get_skin_anchor_draws(self):
        """Spring visualisations – empty until skin enters simulation."""
        draws = []
        if self.skin_anchors.n > 0:
            draws.append(self.skin_anchors.get_render_draw(
                color=(0.85, 0.25, 0.15), width=1.5))
        if self.skeleton_skin_springs.n > 0:
            draws.append(self.skeleton_skin_springs.get_render_draw(
                color=(0.25, 0.60, 0.95), width=1.5))
        return draws

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------
    def reset(self):
        self.mesh.v_p.copy_from(self.mesh.v_p_ref)
        self.mesh.reset_mass(rho=1.0)
        self.mesh.set_fixed_point(len(self._pin_np), self._pin_ti)
        self.xpbd.v_v.fill(0)
        self.xpbd.init_rest_status()
