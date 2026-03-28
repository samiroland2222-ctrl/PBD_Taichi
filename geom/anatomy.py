"""
Skeletal proxy geometry for the breast simulation.

Hierarchy:
  chest (root, static)
  └── clavipectoral_fascia  – broad band across upper chest, child of chest,
                              no independent motion
  └── pectoral_left         – left pectoralis major, child of chest,
                              can pitch (up/down) and yaw (forward/back)
  └── pectoral_right        – right pectoralis major, same DOF

The breast mesh models the LEFT breast, positioned at ~x=+0.06 (patient's
left when facing the camera, +x in our coordinate system).

Coordinate convention (matches breast mesh):
  x  – medial(0)/lateral(+)  left breast is at positive x
  y  – inferior(−)/superior(+)
  z  – posterior(0, chest wall) / anterior(+, outward)

Cooper's ligaments run from the surface of the left pectoral bone to the
outer surface of the left breast.
"""

import numpy as np
import taichi as ti


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _rot_x(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1,0,0],[0,c,-s],[0,s,c]], dtype=np.float32)

def _rot_y(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c,0,s],[0,1,0],[-s,0,c]], dtype=np.float32)

def _make_ti_mesh(verts_np, faces_np):
    v = ti.Vector.field(3, dtype=ti.f32, shape=len(verts_np))
    f = ti.field(dtype=ti.i32, shape=len(faces_np) * 3)
    v.from_numpy(verts_np.astype(np.float32))
    f.from_numpy(faces_np.flatten().astype(np.int32))
    return v, f


# ---------------------------------------------------------------------------
# Proxy mesh generators
# ---------------------------------------------------------------------------

def _fascia_verts_local():
    """
    Clavipectoral fascia: a broad concave band ~20 cm wide, 6 cm tall,
    curving gently around the chest wall.  Local origin = band centre.
    """
    cols, rows  = 9, 5
    width       = 0.20    # m, total width across sternum
    height      = 0.06    # m
    curve_depth = 0.012   # z concavity (wraps chest)

    verts = []
    for r in range(rows):
        tv = r / (rows - 1)
        for c in range(cols):
            tu = c / (cols - 1) - 0.5
            x  = tu * width
            y  = (0.5 - tv) * height
            z  = -curve_depth * (1.0 - 4.0 * tu * tu)
            verts.append([x, y, z])

    faces = []
    for r in range(rows - 1):
        for c in range(cols - 1):
            i0 = r * cols + c
            i1 = i0 + 1
            i2 = i0 + cols
            i3 = i2 + 1
            faces += [[i0, i2, i1], [i1, i2, i3]]

    return np.array(verts, dtype=np.float32), np.array(faces, dtype=np.int32)


def _bone_capsule_verts_local(length=0.15, radius=0.018,
                               rings=8, segs=10):
    """
    Capsule with long axis along +x.  Local origin = MEDIAL end (pivot).
    x runs from 0 (medial/sternum end, pivot) to +length (lateral/shoulder).

    Returns verts (N,3), faces (M,3), lateral_surf_idx (indices of cylinder
    surface verts on the lateral half, used as Cooper's ligament anchors).
    """
    verts = []
    faces = []

    # ── cylinder body ────────────────────────────────────────────────────
    # x runs 0..length (medial to lateral); origin at medial end
    for ri in range(rings + 1):
        t  = ri / rings          # 0..1  (medial..lateral)
        x  = t * length
        for si in range(segs):
            angle = 2 * np.pi * si / segs
            y = radius * np.cos(angle)
            z = radius * np.sin(angle)
            verts.append([x, y, z])

    for ri in range(rings):
        for si in range(segs):
            a  = ri * segs + si
            b  = ri * segs + (si + 1) % segs
            c  = (ri + 1) * segs + si
            d  = (ri + 1) * segs + (si + 1) % segs
            faces += [[a, c, b], [b, c, d]]

    cap_rings = 4

    # ── lateral cap (+x, shoulder end) ───────────────────────────────────
    base_lat = len(verts)
    lat_ring_start = rings * segs
    for ci in range(cap_rings):
        phi = np.pi / 2 * (ci + 1) / cap_rings
        x   = length + radius * np.sin(phi)
        r2  = radius * np.cos(phi)
        for si in range(segs):
            angle = 2 * np.pi * si / segs
            verts.append([x, r2 * np.cos(angle), r2 * np.sin(angle)])
    apex_lat = len(verts)
    verts.append([length + radius, 0.0, 0.0])

    for ci in range(cap_rings):
        if ci == 0:
            for si in range(segs):
                a = lat_ring_start + si
                b = lat_ring_start + (si + 1) % segs
                c = base_lat + si
                d = base_lat + (si + 1) % segs
                faces += [[a, c, b], [b, c, d]]
        else:
            prev = base_lat + (ci - 1) * segs
            cur  = base_lat + ci * segs
            for si in range(segs):
                a = prev + si
                b = prev + (si + 1) % segs
                c = cur  + si
                d = cur  + (si + 1) % segs
                faces += [[a, c, b], [b, c, d]]
    last_lat = base_lat + (cap_rings - 1) * segs
    for si in range(segs):
        faces.append([last_lat + si, apex_lat, last_lat + (si + 1) % segs])

    # ── medial cap (x=0, sternum/pivot end) ──────────────────────────────
    base_med = len(verts)
    med_ring_start = 0
    for ci in range(cap_rings):
        phi = np.pi / 2 * (ci + 1) / cap_rings
        x   = -radius * np.sin(phi)
        r2  = radius * np.cos(phi)
        for si in range(segs):
            angle = 2 * np.pi * si / segs
            verts.append([x, r2 * np.cos(angle), r2 * np.sin(angle)])
    apex_med = len(verts)
    verts.append([-radius, 0.0, 0.0])

    for ci in range(cap_rings):
        if ci == 0:
            for si in range(segs):
                a = med_ring_start + si
                b = med_ring_start + (si + 1) % segs
                c = base_med + si
                d = base_med + (si + 1) % segs
                faces += [[a, b, c], [b, d, c]]
        else:
            prev = base_med + (ci - 1) * segs
            cur  = base_med + ci * segs
            for si in range(segs):
                a = prev + si
                b = prev + (si + 1) % segs
                c = cur  + si
                d = cur  + (si + 1) % segs
                faces += [[a, b, c], [b, d, c]]
    last_med = base_med + (cap_rings - 1) * segs
    for si in range(segs):
        faces.append([last_med + si, last_med + (si + 1) % segs, apex_med])

    verts = np.array(verts, dtype=np.float32)
    faces = np.array(faces, dtype=np.int32)

    # Lateral half of cylinder surface verts (x > length/2) → ligament anchors
    cyl_count = (rings + 1) * segs
    cyl_verts = verts[:cyl_count]
    lateral_mask = cyl_verts[:, 0] > length * 0.1
    lateral_surf_idx = np.where(lateral_mask)[0].astype(np.int32)

    return verts, faces, lateral_surf_idx


# ---------------------------------------------------------------------------
# Skeleton
# ---------------------------------------------------------------------------

@ti.data_oriented
class Skeleton:
    """
    Skeletal proxy for pectoral anatomy surrounding the LEFT breast.

    The pectoral bones run horizontally (long axis = X).  Meaningful DOF:
      pec_left_pitch  – tilt the bone's lateral end up/down  (rot around Z)
      pec_left_yaw    – swing the bone anteriorly/posteriorly (rot around Y)
      (same for right)

    chest_pos is the world-space root at the chest wall (z ≈ 0).
    The left breast mesh sits at x ∈ [−0.02, +0.14], y ∈ [−0.08, +0.08],
    z ∈ [0, 0.1].
    """

    def __init__(self, chest_pos=(0.0, 0.08, 0.0)):
        self.chest_pos = np.array(chest_pos, dtype=np.float32)

        self.pec_left_pitch  = 0.0   # rot around Z: tilts lateral end up/down
        self.pec_left_yaw    = 0.0   # rot around Y: swings bone forward/back
        self.pec_right_pitch = 0.0
        self.pec_right_yaw   = 0.0

        # ── local geometry ────────────────────────────────────────────────
        self._fascia_v_local, self._fascia_f_np = _fascia_verts_local()

        (self._pec_l_v_local, self._pec_l_f_np,
         self._pec_l_surf_idx) = _bone_capsule_verts_local()
        (self._pec_r_v_local, self._pec_r_f_np,
         self._pec_r_surf_idx) = _bone_capsule_verts_local()

        # ── joint offsets from chest_pos ──────────────────────────────────
        # fascia: wide band across upper chest at chest wall
        self._fascia_offset  = np.array([0.0,  -0.01,  0.005], dtype=np.float32)

        # Left pectoral: local x=0 is the pivot (medial/sternum end).
        # Place it so the medial end is at world x ≈ 0.01 (just right of midline).
        # offset is added to chest_pos=(0, 0.13, 0), so world medial end =
        # chest_pos + offset = (0.02, 0.12, 0.02).
        self._pec_l_offset   = np.array([ 0.02,  -0.01,  0.02], dtype=np.float32)

        # Right pectoral: medial end at x ≈ -0.02.
        # The right bone geometry needs its x-axis flipped so it extends toward -x.
        self._pec_r_offset   = np.array([-0.02,  -0.01,  0.02], dtype=np.float32)

        # ── Taichi fields ─────────────────────────────────────────────────
        self.fascia_v, self.fascia_f = _make_ti_mesh(
            self._fascia_v_local, self._fascia_f_np)
        self.pec_l_v,  self.pec_l_f  = _make_ti_mesh(
            self._pec_l_v_local,  self._pec_l_f_np)
        self.pec_r_v,  self.pec_r_f  = _make_ti_mesh(
            self._pec_r_v_local,  self._pec_r_f_np)

        # world-space cache of left pectoral surface verts (for anchors)
        self._pec_l_world = self._pec_l_v_local.copy()

        self.update()

    # ------------------------------------------------------------------
    def _transform_verts(self, v_local, R, offset):
        return (v_local @ R.T) + self.chest_pos + offset

    def update(self):
        R_id = np.eye(3, dtype=np.float32)
        self.fascia_v.from_numpy(
            self._transform_verts(self._fascia_v_local, R_id,
                                  self._fascia_offset))

        # Long axis is +x (medial→lateral).
        # Pitch = tilt lateral end up/down  → rot around Z.
        # Yaw   = swing bone forward/back   → rot around Y.
        # Pivot is at local origin (medial end) — no extra translation needed.
        def _rot_z(a):
            c, s = np.cos(a), np.sin(a)
            return np.array([[c,-s,0],[s,c,0],[0,0,1]], dtype=np.float32)

        R_l = _rot_y(self.pec_left_yaw)  @ _rot_z(self.pec_left_pitch)
        R_r = _rot_y(self.pec_right_yaw) @ _rot_z(-self.pec_right_pitch)

        self._pec_l_world = self._transform_verts(
            self._pec_l_v_local, R_l, self._pec_l_offset)

        # Right bone: flip x so it extends toward -x (right shoulder)
        pec_r_local_mirrored = self._pec_r_v_local.copy()
        pec_r_local_mirrored[:, 0] *= -1
        pec_r_world = self._transform_verts(
            pec_r_local_mirrored, R_r, self._pec_r_offset)

        self.pec_l_v.from_numpy(self._pec_l_world)
        self.pec_r_v.from_numpy(pec_r_world)

    # ------------------------------------------------------------------
    def get_pec_left_surface_anchors_np(self):
        """World-space positions of the left pectoral surface vertices."""
        return self.pec_l_v.to_numpy()[self._pec_l_surf_idx]   # (n_surf, 3)

    def get_pec_right_surface_anchors_np(self):
        """World-space positions of the right pectoral surface vertices."""
        return self.pec_r_v.to_numpy()[self._pec_r_surf_idx]   # (n_surf, 3)

    # ------------------------------------------------------------------
    def reset_pose(self):
        """Zero all joint angles and recompute world positions."""
        self.pec_left_pitch  = 0.0
        self.pec_left_yaw    = 0.0
        self.pec_right_pitch = 0.0
        self.pec_right_yaw   = 0.0
        self.update()

    # ------------------------------------------------------------------
    def get_render_draws(self, fascia_color=(0.85, 0.80, 0.95),
                         pec_color=(0.60, 0.35, 0.35)):
        def draw_fascia(scene):
            scene.mesh(self.fascia_v, self.fascia_f,
                       color=fascia_color, two_sided=True)
        def draw_pec_l(scene):
            scene.mesh(self.pec_l_v, self.pec_l_f,
                       color=pec_color, two_sided=True)
        def draw_pec_r(scene):
            scene.mesh(self.pec_r_v, self.pec_r_f,
                       color=pec_color, two_sided=True)
        return [draw_fascia, draw_pec_l, draw_pec_r]
