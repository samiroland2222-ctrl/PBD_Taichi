"""
UnifiedTorso -- merges left + right breast tet meshes and a skin/subcut
fat shell into a single TetMesh driven by one XPBD framework.
"""
import numpy as np
import taichi as ti

from PBD_Taichi.geom.distance_field import BasicTetMesh
import random as _random

# ---------------------------------------------------------------------------
# Module-level defaults exposed for GUI initialisation (deform_3d.py imports)
# ---------------------------------------------------------------------------
BREAST_HYDRO_ALPHA = 1e-2
BREAST_DEVIA_ALPHA = 1e1
SKIN_HYDRO_ALPHA   = 1e-2   # firmer than breast (skin)
SKIN_DEVIA_ALPHA   = 2e-1

rng = _random.Random(42)


def _rand(center, spread_pct):
    return center + spread_pct * center * (rng.random() * 2 - 1)


from PBD_Taichi.cons import framework, deform3d, coopers
from PBD_Taichi.cons.breast import Breast, BreastRegion
from PBD_Taichi.cons.skin_anchor import BaryBreastSkinConstraint, KinematicSkinSpringConstraint
from PBD_Taichi.geom import gtet, skin as skin_mod
from PBD_Taichi.geom.skin import AnchorType, BarycentricBindingDefinition

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
        skin_target_n_tets=2000,
        skin_thickness=0.01,
        arm_init_abduction=0.4,   # radians – A-pose for skin shell init
        g=(0.0, -9.8, 0.0),
        dt=None,
        fps=60,
        substep=6,
    ):

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
        # Store the ribcage face array used in the bindings so we can resolve
        # RIBCAGE targets in _compute_kinematic_targets each frame.
        _rc_faces_raw = (ribcage_faces_np if ribcage_faces_np is not None
                         else skeleton.get_ribcage_faces_np())
        self._ribcage_faces = np.asarray(_rc_faces_raw, dtype=np.int32).reshape(-1, 3)

        # ── 2b. Breast-base → fascia barycentric bindings ─────────────────
        # skeleton.update() was already called above, so fascia world verts are current.
        fascia_l_v_np = skeleton.fascia_l_v.to_numpy().astype(np.float32)
        fascia_r_v_np = skeleton.fascia_r_v.to_numpy().astype(np.float32)
        fascia_l_f    = np.asarray(skeleton._fascia_l_f_np, dtype=np.int32).reshape(-1, 3)
        fascia_r_f    = np.asarray(skeleton._fascia_r_f_np, dtype=np.int32).reshape(-1, 3)

        def _vf_to_soup(v, f):
            """Build (T, 3, 3) float32 triangle soup from verts+faces."""
            return np.stack([v[f[:,0]], v[f[:,1]], v[f[:,2]]], axis=1).astype(np.float32)

        soup_l = _vf_to_soup(fascia_l_v_np, fascia_l_f)
        soup_r = _vf_to_soup(fascia_r_v_np, fascia_r_f)

        # Breast base verts are in the same world space (origin = simulation root)
        bl_base_pts = v_l[base_l].astype(np.float64)   # (n_base_l, 3)
        br_base_pts = v_r[base_r].astype(np.float64)   # (n_base_r, 3)

        best_tri_l, best_uvw_l, _ = skin_mod._batch_closest_triangles(bl_base_pts, soup_l)
        best_tri_r, best_uvw_r, _ = skin_mod._batch_closest_triangles(br_base_pts, soup_r)

        # Persist for _compute_fascia_targets() called every frame
        self._fascia_l_faces    = fascia_l_f
        self._fascia_r_faces    = fascia_r_f
        self._fascia_l_bind_tri = best_tri_l.astype(np.int32)
        self._fascia_l_bind_uvw = best_uvw_l.astype(np.float32)   # (n_base_l, 3)
        self._fascia_r_bind_tri = best_tri_r.astype(np.int32)
        self._fascia_r_bind_uvw = best_uvw_r.astype(np.float32)   # (n_base_r, 3)

        print(f"[UnifiedTorso] fascia bindings: "
              f"L={len(best_tri_l)}  R={len(best_tri_r)}")

        # ── 3. merge breast + skin meshes into one unified simulation mesh ──
        self.skin_offset = self.n_left_verts + self.n_right_verts
        if self.skin_shell:
            skin_v_np   = self.skin_shell.verts.astype(np.float32)
            skin_t_np   = self.skin_shell.tets.flatten().astype(np.int32)
            skin_f_surf = self.skin_shell.surface_faces()           # (F, 3) local
            self.n_skin_verts = len(skin_v_np)
            self.n_skin_tets  = len(self.skin_shell.tets)
            merged_v, merged_t, merged_f = gtet.merge_numpy(
                (v_l, t_l, f_l),
                (v_r, t_r, f_r),
                (skin_v_np, skin_t_np, skin_f_surf.flatten()),
            )
        else:
            skin_f_surf = None
            self.n_skin_verts = 0
            self.n_skin_tets  = 0
            merged_v, merged_t, merged_f = gtet.merge_numpy(
                (v_l, t_l, f_l),
                (v_r, t_r, f_r),
            )
        self.mesh = gtet.TetMesh(v=merged_v, t=merged_t, f=merged_f,
                                 rho=1.0, scale=1.0)

        # ── 3b. skin face-index field (global indices into unified mesh) ──
        if skin_f_surf is not None and len(skin_f_surf) > 0:
            global_skin_f = (skin_f_surf + self.skin_offset).flatten().astype(np.int32)
            self.skin_f_i = ti.field(dtype=ti.i32, shape=len(global_skin_f))
            self.skin_f_i.from_numpy(global_skin_f)
        else:
            self.skin_f_i = None


        # ── index offsets ─────────────────────────────────────────────────
        self.left_offset  = 0
        self.right_offset = self.n_left_verts
        # self.skin_offset already set in step 3 above

        # breast vertex indices in the unified mesh
        self.base_l = base_l + self.left_offset
        self.top_l  = top_l  + self.left_offset
        self.base_r = base_r + self.right_offset
        self.top_r  = top_r  + self.right_offset

        # ── 4. no hard pins for breast bases (held by fascia springs below) ─
        pin_np = np.zeros(0, dtype=np.int32)
        _pin_dummy = ti.field(dtype=ti.i32, shape=1)
        self.mesh.set_fixed_point(0, _pin_dummy)   # noop – loops 0 times
        self._pin_np = pin_np
        self._pin_ti = _pin_dummy

        self.breast_l = BreastRegion(self.mesh, self.base_l, self.top_l)
        self.breast_r = BreastRegion(self.mesh, self.base_r, self.top_r)

        # ── 5. build PBD framework ────────────────────────────────────────
        g_vec = ti.Vector(list(g))
        self.xpbd = framework.pbd_framework(
            g=g_vec, n_vert=self.mesh.n_vert, v_p=self.mesh.v_p,
            dt=dt, damp=0.99, invm=self.mesh.v_invm)

        # ── 5a. Deform3D for breast tets (breast L + breast R) ────────────
        n_breast_tets = self.n_left_tets + self.n_right_tets
        _breast_t_np  = merged_t[:n_breast_tets * 4]
        self._breast_t_field = ti.field(dtype=ti.i32, shape=len(_breast_t_np))
        self._breast_t_field.from_numpy(_breast_t_np)

        _tm_all = self.mesh.t_mass.to_numpy()
        self._breast_tm_field = ti.field(dtype=ti.f32, shape=max(1, n_breast_tets))
        self._breast_tm_field.from_numpy(_tm_all[:n_breast_tets])

        self.deform_breast = deform3d.Deform3D(
            n=n_breast_tets, indices=self._breast_t_field,
            invm=self.mesh.v_invm, pos=self.mesh.v_p,
            pos_ref=self.mesh.v_p_ref, tet_mass=self._breast_tm_field,
            dt=dt, hydro_alpha=1e-2, devia_alpha=1e1)
        self.xpbd.add_cons(self.deform_breast)

        # ── 5b. Deform3D for skin tets (softer defaults) ──────────────────
        if self.n_skin_tets > 0:
            _skin_t_np = merged_t[n_breast_tets * 4:]
            self._skin_t_field = ti.field(dtype=ti.i32, shape=len(_skin_t_np))
            self._skin_t_field.from_numpy(_skin_t_np)
            self._skin_tm_field = ti.field(dtype=ti.f32, shape=self.n_skin_tets)
            self._skin_tm_field.from_numpy(_tm_all[n_breast_tets:])
            self.deform_skin = deform3d.Deform3D(
                n=self.n_skin_tets, indices=self._skin_t_field,
                invm=self.mesh.v_invm, pos=self.mesh.v_p,
                pos_ref=self.mesh.v_p_ref, tet_mass=self._skin_tm_field,
                dt=dt, hydro_alpha=SKIN_HYDRO_ALPHA, devia_alpha=SKIN_DEVIA_ALPHA)
            self.xpbd.add_cons(self.deform_skin)
        else:
            self._skin_t_field  = None
            self._skin_tm_field = None
            self.deform_skin    = None

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

        # ── 7. skin constraints built from barycentric_bindings ──────────
        bl_faces = f_l.reshape(-1, 3)   # (F, 3) local breast-L face indices
        br_faces = f_r.reshape(-1, 3)   # (F, 3) local breast-R face indices

        # 7a. BaryBreastSkinConstraint – skin inner vert ↔ breast surface tri
        breast_b = [b for b in self.barycentric_bindings
                    if b.anchor_type in (AnchorType.BREAST_L, AnchorType.BREAST_R)]
        if breast_b:
            _si, _v0, _v1, _v2, _uvw = [], [], [], [], []
            for b in breast_b:
                gi = self.skin_offset + b.skin_vertex_index
                if b.anchor_type == AnchorType.BREAST_L:
                    f3 = bl_faces[b.anchor_bary_face]; off = self.left_offset
                else:
                    f3 = br_faces[b.anchor_bary_face]; off = self.right_offset
                _si.append(gi)
                _v0.append(int(f3[0]) + off)
                _v1.append(int(f3[1]) + off)
                _v2.append(int(f3[2]) + off)
                _uvw.append(b.anchor_bary_uvw)
            skin_idx_np = np.array(_si, dtype=np.int32)
            v0_np  = np.array(_v0, dtype=np.int32)
            v1_np  = np.array(_v1, dtype=np.int32)
            v2_np  = np.array(_v2, dtype=np.int32)
            uvw_np = np.array(_uvw, dtype=np.float32)
        else:
            skin_idx_np = v0_np = v1_np = v2_np = np.zeros(0, dtype=np.int32)
            uvw_np = np.zeros((0, 3), dtype=np.float32)

        self.skin_anchors = BaryBreastSkinConstraint(
            v_p=self.mesh.v_p, v_invm=self.mesh.v_invm,
            skin_idx_np=skin_idx_np,
            tri_v0_np=v0_np, tri_v1_np=v1_np, tri_v2_np=v2_np,
            bary_uvw_np=uvw_np,
            dt=dt, alpha=1e-2, pretension=1.0)
        self.xpbd.add_cons(self.skin_anchors)

        # 7b. KinematicSkinSpringConstraint – skin vert → skeleton surface target
        self._kinematic_bindings = [
            b for b in self.barycentric_bindings
            if b.anchor_type in self._SPRING_TYPES
        ]
        if self._kinematic_bindings:
            kin_skin_idx = np.array(
                [self.skin_offset + b.skin_vertex_index
                 for b in self._kinematic_bindings], dtype=np.int32)
            init_targets = self._compute_kinematic_targets(self._kinematic_bindings)
        else:
            kin_skin_idx = np.zeros(0, dtype=np.int32)
            init_targets = np.zeros((0, 3), dtype=np.float32)

        self.skeleton_skin_springs = KinematicSkinSpringConstraint(
            v_p=self.mesh.v_p, v_invm=self.mesh.v_invm,
            skin_idx_np=kin_skin_idx, init_target_np=init_targets,
            dt=dt, alpha=1e-3, pretension=1.0)
        self.xpbd.add_cons(self.skeleton_skin_springs)

        # 7c. KinematicSkinSpringConstraint – skin vert → ribcage surface target
        #     (weaker than breast-skin springs – ribcage is not glued, just anchored)
        self._ribcage_bindings = [
            b for b in self.barycentric_bindings
            if b.anchor_type == AnchorType.RIBCAGE
        ]
        if self._ribcage_bindings:
            rc_skin_idx = np.array(
                [self.skin_offset + b.skin_vertex_index
                 for b in self._ribcage_bindings], dtype=np.int32)
            rc_init_targets = self._compute_kinematic_targets(self._ribcage_bindings)
        else:
            rc_skin_idx     = np.zeros(0, dtype=np.int32)
            rc_init_targets = np.zeros((0, 3), dtype=np.float32)

        self.ribcage_skin_springs = KinematicSkinSpringConstraint(
            v_p=self.mesh.v_p, v_invm=self.mesh.v_invm,
            skin_idx_np=rc_skin_idx, init_target_np=rc_init_targets,
            dt=dt, alpha=1e0, pretension=1.0)   # weaker default
        self.xpbd.add_cons(self.ribcage_skin_springs)

        # 7d. KinematicSkinSpringConstraint – breast base verts → fascia surface
        #     Replaces the old hard-pin: breast base follows the clavipectoral
        #     fascia barycentrically as the skeleton moves.
        def _fascia_init_targets(v_np, faces, tri_idx, uvw):
            if len(tri_idx) == 0:
                return np.zeros((0, 3), dtype=np.float32)
            f = faces[tri_idx]
            return (uvw[:, 0:1] * v_np[f[:, 0]]
                    + uvw[:, 1:2] * v_np[f[:, 1]]
                    + uvw[:, 2:3] * v_np[f[:, 2]]).astype(np.float32)

        init_tgts_l = _fascia_init_targets(
            fascia_l_v_np, fascia_l_f, best_tri_l, best_uvw_l)
        init_tgts_r = _fascia_init_targets(
            fascia_r_v_np, fascia_r_f, best_tri_r, best_uvw_r)

        self.breast_l_fascia_springs = KinematicSkinSpringConstraint(
            v_p=self.mesh.v_p, v_invm=self.mesh.v_invm,
            skin_idx_np=self.base_l.astype(np.int32),
            init_target_np=init_tgts_l,
            dt=dt, alpha=1e-4, pretension=1.0)
        self.xpbd.add_cons(self.breast_l_fascia_springs)

        self.breast_r_fascia_springs = KinematicSkinSpringConstraint(
            v_p=self.mesh.v_p, v_invm=self.mesh.v_invm,
            skin_idx_np=self.base_r.astype(np.int32),
            init_target_np=init_tgts_r,
            dt=dt, alpha=1e-4, pretension=1.0)
        self.xpbd.add_cons(self.breast_r_fascia_springs)

        # ── 8. init rest status ───────────────────────────────────────────
        self.xpbd.init_rest_status()
        print(f"[UnifiedTorso] {self.mesh.n_vert} verts, {self.mesh.n_tet} tets  "
              f"(L={self.n_left_tets} R={self.n_right_tets} S={self.n_skin_tets})  "
              f"bindings: {len(breast_b)} breast, "
              f"{len(self._kinematic_bindings)} kinematic, "
              f"{len(self._ribcage_bindings)} ribcage-spring")

    # ------------------------------------------------------------------
    # Skin kinematic update – push fresh skeleton targets each frame
    # ------------------------------------------------------------------
    def update_kinematic_skin(self):
        """Re-evaluate skeleton surface positions and push into spring constraints."""
        if self.skeleton_skin_springs.n > 0 and self._kinematic_bindings:
            new_targets = self._compute_kinematic_targets(self._kinematic_bindings)
            self.skeleton_skin_springs.update_targets(new_targets)
        if self.ribcage_skin_springs.n > 0 and self._ribcage_bindings:
            rc_targets = self._compute_kinematic_targets(self._ribcage_bindings)
            self.ribcage_skin_springs.update_targets(rc_targets)
        # Fascia breast-base springs — targets follow the live skeleton fascia mesh
        sk = self.skeleton
        if self.breast_l_fascia_springs.n > 0:
            tgts_l = self._compute_fascia_targets(
                sk.fascia_l_v.to_numpy().astype(np.float32),
                self._fascia_l_faces, self._fascia_l_bind_tri, self._fascia_l_bind_uvw)
            self.breast_l_fascia_springs.update_targets(tgts_l)
        if self.breast_r_fascia_springs.n > 0:
            tgts_r = self._compute_fascia_targets(
                sk.fascia_r_v.to_numpy().astype(np.float32),
                self._fascia_r_faces, self._fascia_r_bind_tri, self._fascia_r_bind_uvw)
            self.breast_r_fascia_springs.update_targets(tgts_r)

    @staticmethod
    def _compute_fascia_targets(fascia_v: np.ndarray,
                                fascia_f: np.ndarray,
                                tri_idx:  np.ndarray,
                                uvw:      np.ndarray) -> np.ndarray:
        """Evaluate barycentric positions on the current fascia mesh.

        Parameters
        ----------
        fascia_v : (N, 3) float32 – current world-space fascia vertex positions
        fascia_f : (F, 3) int32   – fascia face index array (static)
        tri_idx  : (K,)   int32   – per-binding face index into fascia_f
        uvw      : (K, 3) float32 – barycentric weights [w_A, w_B, w_C]

        Returns
        -------
        targets : (K, 3) float32
        """
        if len(tri_idx) == 0:
            return np.zeros((0, 3), dtype=np.float32)
        f = fascia_f[tri_idx]   # (K, 3) vertex indices
        return (uvw[:, 0:1] * fascia_v[f[:, 0]]
                + uvw[:, 1:2] * fascia_v[f[:, 1]]
                + uvw[:, 2:3] * fascia_v[f[:, 2]]).astype(np.float32)

    def _compute_kinematic_targets(self, bindings: list) -> np.ndarray:
        """Return (N, 3) float32 world-space target positions for *bindings*.

        Each binding references a local face index on a skeleton surface mesh
        plus barycentric coordinates.  The target position is the interpolated
        point on the current (world-space) skeleton mesh.
        """
        if not bindings:
            return np.zeros((0, 3), dtype=np.float32)
        sk = self.skeleton
        mesh_map = {
            AnchorType.CLAVICLE_L: (
                np.asarray(sk.get_clavicle_left_world_np(),    dtype=np.float32),
                np.asarray(sk.get_clavicle_left_faces_np(),    dtype=np.int32).reshape(-1, 3)),
            AnchorType.CLAVICLE_R: (
                np.asarray(sk.get_clavicle_right_world_np(),   dtype=np.float32),
                np.asarray(sk.get_clavicle_right_faces_np(),   dtype=np.int32).reshape(-1, 3)),
            AnchorType.ARM_L: (
                np.asarray(sk.get_upper_arm_left_surface_np(), dtype=np.float32),
                np.asarray(sk.get_upper_arm_left_faces_np(),   dtype=np.int32).reshape(-1, 3)),
            AnchorType.ARM_R: (
                np.asarray(sk.get_upper_arm_right_surface_np(), dtype=np.float32),
                np.asarray(sk.get_upper_arm_right_faces_np(),   dtype=np.int32).reshape(-1, 3)),
            AnchorType.RIBCAGE: (
                np.asarray(sk.get_ribcage_verts_world_np(),    dtype=np.float32),
                self._ribcage_faces),
        }
        targets = []
        for b in bindings:
            verts, faces = mesh_map[b.anchor_type]
            tri = faces[b.anchor_bary_face]        # [v0, v1, v2]
            uvw = b.anchor_bary_uvw                # [w0, w1, w2]
            pt  = (uvw[0] * verts[tri[0]]
                   + uvw[1] * verts[tri[1]]
                   + uvw[2] * verts[tri[2]])
            targets.append(pt)
        return np.array(targets, dtype=np.float32)

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------
    def get_render_draws(self, breast_color=(0.85, 0.65, 0.55)):
        """Draw all breast+skin faces with per-pixel Phong normals."""
        v_p  = self.mesh.v_p
        f_i  = self.mesh.f_i
        def draw_all(scene):
            scene.mesh(v_p, f_i,
                       color=breast_color, show_wireframe=False, two_sided=False)
        return [draw_all]

    def get_skin_draws(self, color=(0.90, 0.78, 0.68)):
        """Render the simulated skin surface with per-pixel Phong normals.

        Normals are computed (or reused) from the full mesh surface so that
        vertices at the breast-skin boundary are shaded smoothly.
        """
        if self.skin_f_i is None:
            return []
        v_p     = self.mesh.v_p
        skin_fi = self.skin_f_i
        def draw_skin(scene):
            scene.mesh(v_p, skin_fi,
                       color=color, two_sided=False, show_wireframe=True)
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
        # Re-sync split tet-mass fields from the recomputed unified mass field.
        _tm = self.mesh.t_mass.to_numpy()
        n_bt = self.n_left_tets + self.n_right_tets
        self._breast_tm_field.from_numpy(_tm[:n_bt])
        if self.n_skin_tets > 0:
            self._skin_tm_field.from_numpy(_tm[n_bt:])
        # No hard pins (breast bases held by fascia springs).
        self.mesh.set_fixed_point(0, self._pin_ti)
        self.xpbd.v_v.fill(0)
        self.xpbd.init_rest_status()
