"""
Skeletal proxy geometry for the breast simulation.

Hierarchy:
  chest (root, static)
  └── clavicle_left         – left clavicle, child of chest,
                              can pitch (up/down) and yaw (forward/back)
  └── clavicle_right        – right clavicle, same DOF
  └── fascia_left           – clavipectoral fascia, deformable quad surface
  └── fascia_right          – clavipectoral fascia, deformable quad surface

A models each breast. The LEFT breast is positioned at ~x=+0.06 (patient's
left when facing the camera, +x in our coordinate system).

Coordinate convention (matches breast mesh):
  x  – medial(0)/lateral(+)  left breast is at positive x
  y  – inferior(−)/superior(+)
  z  – posterior(0, chest wall) / anterior(+, outward)

Cooper's ligaments run from the surface of the clavipectoral fascia to the
outer surface of the breasts.
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

def _bone_capsule_verts_local(length=0.136, radius=0.007,
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
# Clavipectoral fascia quad surface
# ---------------------------------------------------------------------------

def _fascia_verts_local(rows=6, cols=6,
                         top_x0=0.02, top_x1=0.16,
                         top_y=0.0,   top_z=0.02,
                         bot_x0=0.01, bot_x1=0.14,
                         bot_y=-0.08, bot_z=0.0):
    """
    Generate a (rows × cols) quad grid representing the clavipectoral fascia
    for the LEFT side (positive x).

    Row 0 = top edge  → tracks the clavicle (bound to clavicle frame each frame)
    Row rows-1 = bottom edge → fixed to chest root (static)
    Interior rows are linearly interpolated.

    Returns:
      verts_local (rows*cols, 3)  – reference positions (chest-space, unrotated)
      faces       (M, 3)          – triangle indices
      top_row_idx (cols,)         – vertex indices of the top row
      bot_row_idx (cols,)         – vertex indices of the bottom row
    """
    verts = []
    for ri in range(rows):
        t = ri / (rows - 1)           # 0 = top, 1 = bottom
        for ci in range(cols):
            s = ci / (cols - 1)       # 0 = medial, 1 = lateral
            # Interpolate top and bottom edges
            top_x = top_x0 + s * (top_x1 - top_x0)
            top_pt = np.array([top_x, top_y, top_z], dtype=np.float32)
            bot_x  = bot_x0 + s * (bot_x1 - bot_x0)
            bot_pt = np.array([bot_x, bot_y, bot_z], dtype=np.float32)
            p = (1.0 - t) * top_pt + t * bot_pt
            verts.append(p)

    faces = []
    for ri in range(rows - 1):
        for ci in range(cols - 1):
            a = ri * cols + ci
            b = ri * cols + ci + 1
            c = (ri + 1) * cols + ci
            d = (ri + 1) * cols + ci + 1
            faces += [[a, b, c], [b, d, c]]

    verts = np.array(verts, dtype=np.float32)
    faces = np.array(faces, dtype=np.int32)
    top_row_idx = np.arange(0, cols, dtype=np.int32)
    bot_row_idx = np.arange((rows - 1) * cols, rows * cols, dtype=np.int32)
    return verts, faces, top_row_idx, bot_row_idx


# ---------------------------------------------------------------------------
# Skeleton
# ---------------------------------------------------------------------------

@ti.data_oriented
class Skeleton:
    """
    Skeletal proxy for pectoral anatomy surrounding the breasts.

    The clavicles run horizontally (long axis = X).  Meaningful DOF:
      clavicle_left_pitch  – tilt the bone's lateral end up/down  (rot around Z)
      clavicle_left_yaw    – swing the bone anteriorly/posteriorly (rot around Y)
      (same for right)

    chest_pos is the world-space root at the chest wall (z ≈ 0).
    The left breast mesh sits at x ∈ [−0.02, +0.14], y ∈ [−0.08, +0.08],
    z ∈ [0, 0.1].
    """

    def __init__(self, chest_pos=(0.0, 0.08, 0.0)):
        self.chest_pos = np.array(chest_pos, dtype=np.float32)

        self.clavicle_left_yaw    = 0.0   # rot around Y: swings bone forward/back
        self.clavicle_right_yaw   = 0.0
        # pitch set to anatomical default after geometry is built (below)

        # ── local geometray ────────────────────────────────────────────────
        (self._clavicle_l_v_local, self._clavicle_l_f_np,
         self._clavicle_l_surf_idx) = _bone_capsule_verts_local()
        (self._clavicle_r_v_local, self._clavicle_r_f_np,
         self._clavicle_r_surf_idx) = _bone_capsule_verts_local()

        # ── fascia local geometry (left side, positive x) ──────────────────
        _FASCIA_ROWS = 6
        _FASCIA_COLS = 6
        (self._fascia_l_v_local, self._fascia_l_f_np,
         self._fascia_l_top_idx, self._fascia_l_bot_idx) = _fascia_verts_local(
             rows=_FASCIA_ROWS, cols=_FASCIA_COLS,
             top_x0=0.0,   top_x1=0.136,   # along clavicle local-x (medial→lateral)
             top_y=0.0,    top_z=0.0,       # top edge at clavicle pivot origin
             bot_x0=0.01,  bot_x1=0.14,    # chest-local bottom edge
             bot_y=-0.08,  bot_z=0.0)

        # Right fascia: mirror x — top runs from 0 toward -0.136
        (self._fascia_r_v_local, self._fascia_r_f_np,
         self._fascia_r_top_idx, self._fascia_r_bot_idx) = _fascia_verts_local(
             rows=_FASCIA_ROWS, cols=_FASCIA_COLS,
             top_x0=0.0,    top_x1=-0.136,
             top_y=0.0,     top_z=0.0,
             bot_x0=-0.01,  bot_x1=-0.14,
             bot_y=-0.08,   bot_z=0.0)

        # ── joint offsets from chest_pos ──────────────────────────────────
        # Sternoclavicular joint sits at the manubrium, level with R1.
        # In local coords (relative to chest_pos): x ≈ ±0.010, y ≈ +0.035, z = 0.
        # The clavicle has a natural superior bow (~5°) and slight anterior curve,
        # represented by the pitch DOF; the rest pose here is anatomical neutral.
        self._clavicle_l_offset = np.array([ 0.010,  0.035, 0.0], dtype=np.float32)
        self._clavicle_r_offset = np.array([-0.010,  0.035, 0.0], dtype=np.float32)

        # Natural resting pitch: clavicle rises ~5° superiorly from medial to lateral
        self.clavicle_left_pitch  =  np.deg2rad(5.0)
        self.clavicle_right_pitch =  np.deg2rad(5.0)

        # ── Taichi fields ─────────────────────────────────────────────────
        self.clavicle_l_v,  self.clavicle_l_f  = _make_ti_mesh(
            self._clavicle_l_v_local,  self._clavicle_l_f_np)
        self.clavicle_r_v,  self.clavicle_r_f  = _make_ti_mesh(
            self._clavicle_r_v_local,  self._clavicle_r_f_np)

        # Fascia Taichi vertex fields (world space, updated each frame)
        n_fascia_l = len(self._fascia_l_v_local)
        n_fascia_r = len(self._fascia_r_v_local)
        self.fascia_l_v = ti.Vector.field(3, dtype=ti.f32, shape=n_fascia_l)
        self.fascia_l_f_ti = ti.field(dtype=ti.i32, shape=len(self._fascia_l_f_np) * 3)
        self.fascia_l_f_ti.from_numpy(self._fascia_l_f_np.flatten().astype(np.int32))
        self.fascia_r_v = ti.Vector.field(3, dtype=ti.f32, shape=n_fascia_r)
        self.fascia_r_f_ti = ti.field(dtype=ti.i32, shape=len(self._fascia_r_f_np) * 3)
        self.fascia_r_f_ti.from_numpy(self._fascia_r_f_np.flatten().astype(np.int32))

        # world-space cache of left clavicle surface verts (for anchors)
        self._clavicle_l_world = self._clavicle_l_v_local.copy()

        # World-space caches for fascia
        self._fascia_l_world = self._fascia_l_v_local + self.chest_pos
        self._fascia_r_world = self._fascia_r_v_local + self.chest_pos

        self.update()

    # ------------------------------------------------------------------
    def _transform_verts(self, v_local, R, offset):
        return (v_local @ R.T) + self.chest_pos + offset

    def update(self):
        R_id = np.eye(3, dtype=np.float32)

        # Long axis is +x (medial→lateral).
        # Pitch = tilt lateral end up/down  → rot around Z.
        # Yaw   = swing bone forward/back   → rot around Y.
        # Pivot is at local origin (medial end) — no extra translation needed.
        def _rot_z(a):
            c, s = np.cos(a), np.sin(a)
            return np.array([[c,-s,0],[s,c,0],[0,0,1]], dtype=np.float32)

        R_l = _rot_y(self.clavicle_left_yaw) @ _rot_z(self.clavicle_left_pitch)
        R_r = _rot_y(self.clavicle_right_yaw) @ _rot_z(-self.clavicle_right_pitch)

        self._clavicle_l_world = self._transform_verts(
            self._clavicle_l_v_local, R_l, self._clavicle_l_offset)

        # Right bone: flip x so it extends toward -x (right shoulder)
        clavicle_r_local_mirrored = self._clavicle_r_v_local.copy()
        clavicle_r_local_mirrored[:, 0] *= -1
        clavicle_r_world = self._transform_verts(
            clavicle_r_local_mirrored, R_r, self._clavicle_r_offset)

        self.clavicle_l_v.from_numpy(self._clavicle_l_world)
        self.clavicle_r_v.from_numpy(clavicle_r_world)

        # ── update fascia ─────────────────────────────────────────────────
        # Top edge of fascia tracks the clavicle local frame (same R + offset).
        # Bottom edge is fixed to chest root (static).
        # Interior rows are linearly interpolated (t=0 top, t=1 bottom).
        self._fascia_l_world = self._update_fascia(
            self._fascia_l_v_local,
            self._fascia_l_top_idx, self._fascia_l_bot_idx,
            R_l, self._clavicle_l_offset)
        self.fascia_l_v.from_numpy(self._fascia_l_world.astype(np.float32))

        # For right fascia: R_r acts on mirrored geometry (x already negative in local)
        self._fascia_r_world = self._update_fascia(
            self._fascia_r_v_local,
            self._fascia_r_top_idx, self._fascia_r_bot_idx,
            R_r, self._clavicle_r_offset)
        self.fascia_r_v.from_numpy(self._fascia_r_world.astype(np.float32))

    # ------------------------------------------------------------------
    def _update_fascia(self, v_local, top_idx, bot_idx, R, offset):
        """
        Update fascia world positions.
        Top row → transformed by clavicle rotation (same as clavicle bone).
        Bottom row → fixed to chest root (no rotation, just chest_pos + local).
        Interior rows → linearly blended by row-parameter t.
        """
        n_rows = 0
        # Determine number of rows from the index arrays
        # top_idx = row 0, bot_idx = row (n_rows-1)
        n_cols = len(top_idx)
        n_verts = len(v_local)
        n_rows = n_verts // n_cols

        world = np.empty_like(v_local)

        # Compute top row (clavicle frame)
        top_local = v_local[top_idx]
        top_world = self._transform_verts(top_local, R, offset)

        # Bottom row (static – chest root, no rotation)
        bot_local = v_local[bot_idx]
        bot_world = bot_local + self.chest_pos   # identity rotation

        for ri in range(n_rows):
            t = ri / (n_rows - 1)   # 0 = top, 1 = bottom
            row_start = ri * n_cols
            row_end   = row_start + n_cols
            world[row_start:row_end] = (1.0 - t) * top_world + t * bot_world

        return world

    # ------------------------------------------------------------------
    def get_clavicle_left_surface_anchors_np(self):
        """World-space positions of the left clavicle surface vertices."""
        return self.clavicle_l_v.to_numpy()[self._clavicle_l_surf_idx]   # (n_surf, 3)

    def get_clavicle_right_surface_anchors_np(self):
        """World-space positions of the right clavicle surface vertices."""
        return self.clavicle_r_v.to_numpy()[self._clavicle_r_surf_idx]   # (n_surf, 3)

    def get_fascia_left_surface_anchors_np(self):
        """World-space positions of ALL left fascia vertices (used as ligament anchors)."""
        return self._fascia_l_world.copy()   # (n_fascia, 3)

    def get_fascia_right_surface_anchors_np(self):
        """World-space positions of ALL right fascia vertices (used as ligament anchors)."""
        return self._fascia_r_world.copy()   # (n_fascia, 3)

    # ------------------------------------------------------------------
    def reset_pose(self):
        """Restore all joint angles to anatomical neutral and recompute world positions."""
        self.clavicle_left_pitch  = np.deg2rad(5.0)   # natural superior bow
        self.clavicle_right_pitch = np.deg2rad(5.0)
        self.clavicle_left_yaw    = 0.0
        self.clavicle_right_yaw   = 0.0
        self.update()

    # ------------------------------------------------------------------
    def get_render_draws(self, clavicle=(0.60, 0.35, 0.35),
                         ribcage=(0.55, 0.55, 0.65),
                         fascia=(0.70, 0.80, 0.60)):
        def draw_clavicle_l(scene):
            scene.mesh(self.clavicle_l_v, self.clavicle_l_f,
                       color=clavicle, two_sided=True)
        def draw_clavicle_r(scene):
            scene.mesh(self.clavicle_r_v, self.clavicle_r_f,
                       color=clavicle, two_sided=True)
        def draw_fascia_l(scene):
            scene.mesh(self.fascia_l_v, self.fascia_l_f_ti,
                       color=fascia, two_sided=True)
        def draw_fascia_r(scene):
            scene.mesh(self.fascia_r_v, self.fascia_r_f_ti,
                       color=fascia, two_sided=True)
        return [draw_clavicle_l, draw_clavicle_r,
                draw_fascia_l, draw_fascia_r]
