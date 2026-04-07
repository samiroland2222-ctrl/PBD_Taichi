"""
Skeletal proxy geometry for the breast simulation.

Hierarchy:
  chest (root, static)
  └── clavicle_left         – left clavicle, child of chest,
                              can pitch (up/down) and yaw (forward/back)
      └── upper_arm_left    – upper arm capsule, child of clavicle_left,
                              ball-and-socket at glenohumeral (GH) joint;
                              DOF: flexion (rot X) and abduction (rot Z)
  └── clavicle_right        – right clavicle, same DOF
      └── upper_arm_right   – upper arm capsule, mirror of left
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

from PBD_Taichi.geom.distance_field import BasicTriMesh


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _rot_x(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1,0,0],[0,c,-s],[0,s,c]], dtype=np.float32)

def _rot_y(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c,0,s],[0,1,0],[-s,0,c]], dtype=np.float32)

def _rot_z(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c,-s,0],[s,c,0],[0,0,1]], dtype=np.float32)

def _make_ti_mesh(verts_np, faces_np):
    v = ti.Vector.field(3, dtype=ti.f32, shape=len(verts_np))
    f = ti.field(dtype=ti.i32, shape=len(faces_np) * 3)
    v.from_numpy(verts_np.astype(np.float32))
    f.from_numpy(faces_np.flatten().astype(np.int32))
    return v, f


# ---------------------------------------------------------------------------
# Proxy mesh generators
# ---------------------------------------------------------------------------

# Clavicle acromial length – used both in the bone capsule and to compute the
# glenohumeral joint pivot position.
_CLAVICLE_LENGTH = 0.136    # metres

def _clavicle_capsule_verts_local():
    """
    Capsule with long axis along +x.  Local origin = MEDIAL end (pivot).
    x runs from 0 (medial/sternum end, pivot) to +length (lateral/shoulder).

    Returns verts (N,3), faces (M,3), lateral_surf_idx (indices of cylinder
    surface verts on the lateral half, used as Cooper's ligament anchors).
    """
    clavicle_mesh = _make_capsule(length=_CLAVICLE_LENGTH, radius=0.02, rings=8, segs=10, long_axis='x')
    return clavicle_mesh

# ---------------------------------------------------------------------------
# Upper-arm proxy mesh
# ---------------------------------------------------------------------------
#
# Anthropometric source: ANSUR II (2012), U.S. Army female population.
# Targeting a slender, athletic build:
#   Upper-arm length    31.5 cm   – acromion → lateral epicondyle,
#                                   ≈ 50th-percentile stature-scaled length
#   Mid-arm circumference  ~25 cm  – radius 4.0 cm,
#                                   ≈ 10th-percentile circumference (toned build)
#
# Glenohumeral (GH) joint offset from the clavicle's LATERAL (acromial) tip,
# expressed in clavicle-local space (+x along bone, +y superior, +z anterior).
# The humeral head centre sits ~10 mm lateral and ~12 mm inferior to the
# acromioclavicular (AC) joint with negligible anterior offset.
_UPPER_ARM_LENGTH  = 0.315   # metres
_UPPER_ARM_RADIUS  = 0.040   # metres  (radius, not circumference)
_GH_LOCAL_OFFSET   = np.array([ 0.010, -0.012,  0.000], dtype=np.float32)

# Pre-computed GH pivot positions in clavicle-local space (before world xform).
# Left  : lateral tip at (+_CLAVICLE_LENGTH, 0, 0) + offset
# Right : clavicle x is mirrored, so tip is at (−_CLAVICLE_LENGTH, 0, 0)
_GH_L_LOCAL = np.array([ _CLAVICLE_LENGTH + _GH_LOCAL_OFFSET[0],
                          _GH_LOCAL_OFFSET[1], _GH_LOCAL_OFFSET[2]], dtype=np.float32)
_GH_R_LOCAL = np.array([-(  _CLAVICLE_LENGTH + _GH_LOCAL_OFFSET[0]),
                          _GH_LOCAL_OFFSET[1], _GH_LOCAL_OFFSET[2]], dtype=np.float32)


def _make_upper_arm_capsule(length=_UPPER_ARM_LENGTH, radius=_UPPER_ARM_RADIUS,
                            rings=8, segs=12) -> BasicTriMesh:
    """
    Capsule for the upper arm.  Long axis along −y (anatomical hanging position).

    Origin (y = 0) = glenohumeral joint pivot (shoulder, proximal end).
    y = −length     = elbow (distal end).

    This geometry is shared for both arms; the right side mirrors x before
    the world-space transform so it is a perfect bilateral reflection.
    """
    return _make_capsule(length=length, radius=radius, rings=rings, segs=segs)


def _make_capsule(length, radius, rings=8, segs=12, long_axis='-y') -> BasicTriMesh:
    """
    Capsule with long axis along −y (anatomical hanging position).
    """

    import gmsh
    import numpy as np
    gmsh.initialize()
    gmsh.model.add("upper_arm_capsule")
    if long_axis == '-y':
        end = [0, -length, 0]
    elif long_axis == 'x':
        end = [length, 0, 0]
    else:
        print(f"Unrecognized long_axis {long_axis} -> using x")
        end = [length, 0, 0]
    # Cylinder axis: from (0, 0, 0) to end
    cyl = gmsh.model.occ.addCylinder(0, 0, 0, end[0], end[1], end[2], radius)
    # Proximal hemisphere (shoulder)
    sph_prox = gmsh.model.occ.addSphere(0, 0, 0, radius)
    # Distal hemisphere (elbow)
    sph_dist = gmsh.model.occ.addSphere(end[0], end[1], end[2], radius)
    # Fragment to get a capsule
    gmsh.model.occ.fuse([(3, cyl)], [(3, sph_prox), (3, sph_dist)], removeObject=True, removeTool=True)
    gmsh.model.occ.synchronize()
    # Mesh options
    gmsh.model.mesh.generate(2)
    node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
    verts = np.array(node_coords, dtype=np.float32).reshape(-1, 3)
    elem_types, elem_tags, elem_node_tags = gmsh.model.mesh.getElements(dim=2)
    faces = np.array(elem_node_tags[0], dtype=np.int32).reshape(-1, 3) - 1
    gmsh.finalize()
    return BasicTriMesh(verts, faces)


# ---------------------------------------------------------------------------
# Clavipectoral fascia quad surface
# ---------------------------------------------------------------------------

def _fascia_verts_local(rows=6, cols=10,
                         top_x0=0.02, top_x1=0.16,
                         top_y=0.0,   top_z=0.02,
                         bot_x0=0.01, bot_x1=0.14,
                         bot_y=-0.08,
                         bot_z0=0.0, bot_z1=0.02):
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
            bot_z  = bot_z0 + s * (bot_z1 - bot_z0)
            bot_pt = np.array([bot_x, bot_y, bot_z], dtype=np.float32)
            p = (1.0 - t) * top_pt + t * bot_pt
            verts.append(p)

    swap_ab = top_x0 < top_x1
    swap_cd = top_x0 < bot_x1
    faces = []
    for ri in range(rows - 1):
        for ci in range(cols - 1):
            a = ri * cols + ci
            b = ri * cols + ci + 1
            c = (ri + 1) * cols + ci
            d = (ri + 1) * cols + ci + 1
            face0 = [b, a, c] if swap_ab else [a, b, c]
            face1 = [b, c, d] if swap_cd else [b, d, c]
            faces += [face0, face1]

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

    Glenohumeral (shoulder) DOF — expressed in the parent clavicle's local frame:
      arm_left_flexion     – arm swings anterior (+) / posterior (−)  [rot around +X]
      arm_left_abduction   – arm swings lateral (+) / adducts  (−)    [rot around +Z]
      (same for right; positive abduction always moves the arm outward)

    chest_pos is the world-space root at the chest wall (z ≈ 0).
    The left breast mesh sits at x ∈ [−0.02, +0.14], y ∈ [−0.08, +0.08],
    z ∈ [0, 0.1].
    """

    def __init__(self, chest_pos=(0.0, 0.12, -0.05)):
        self.chest_pos = np.array(chest_pos, dtype=np.float32)

        self.clavicle_rest_pitch = np.deg2rad(5.0)
        self.clavicle_rest_yaw = 0.4

        self.clavicle_left_yaw    =  self.clavicle_rest_yaw   # rot around Y: swings bone forward/back
        self.clavicle_right_yaw   = -self.clavicle_rest_yaw
        # pitch set to anatomical default after geometry is built (below)

        # ── local geometray ────────────────────────────────────────────────
        self._clavicle_l = _clavicle_capsule_verts_local()
        self._clavicle_r = _clavicle_capsule_verts_local().mirrored_x()

        # ── fascia local geometry (left side, positive x) ──────────────────
        _FASCIA_ROWS = 6
        _FASCIA_COLS = 6
        (self._fascia_l_v_local, self._fascia_l_f_np,
         self._fascia_l_top_idx, self._fascia_l_bot_idx) = _fascia_verts_local(
             rows=_FASCIA_ROWS, cols=_FASCIA_COLS,
             top_x0=0.0,   top_x1=0.136,   # along clavicle local-x (medial→lateral)
             top_y=0.0,    top_z=0.0,       # top edge at clavicle pivot origin
             bot_x0=0.01,  bot_x1=0.1,    # chest-local bottom edge
             bot_y=-0.21,
             bot_z0=0.04, bot_z1=-0.0)

        # Right fascia: mirror x — top runs from 0 toward -0.136
        (self._fascia_r_v_local, self._fascia_r_f_np,
         self._fascia_r_top_idx, self._fascia_r_bot_idx) = _fascia_verts_local(
             rows=_FASCIA_ROWS, cols=_FASCIA_COLS,
             top_x0=0.0,    top_x1=-0.136,
             top_y=0.0,     top_z=0.0,
             bot_x0=-0.01,  bot_x1=-0.1,
             bot_y=-0.21,
             bot_z0=0.04, bot_z1=-0.0)

        # ── joint offsets from chest_pos ──────────────────────────────────
        # Sternoclavicular joint sits at the manubrium, level with R1.
        # In local coords (relative to chest_pos): x ≈ ±0.010, y ≈ +0.025, z = 0.
        # The clavicle has a natural superior bow (~5°) and slight anterior curve,
        # represented by the pitch DOF; the rest pose here is anatomical neutral.
        self._clavicle_l_offset = np.array([ 0.010,  0.035, 0.0], dtype=np.float32)
        self._clavicle_r_offset = np.array([-0.010,  0.035, 0.0], dtype=np.float32)

        # Natural resting pitch: clavicle rises ~5° superiorly from medial to lateral
        self.clavicle_left_pitch  = self.clavicle_rest_pitch
        self.clavicle_right_pitch = self.clavicle_rest_pitch

        # ── Taichi fields ─────────────────────────────────────────────────
        self.clavicle_l_v,  self.clavicle_l_f  = _make_ti_mesh(
            self._clavicle_l.verts,  self._clavicle_l.faces)
        self.clavicle_r_v,  self.clavicle_r_f  = _make_ti_mesh(
            self._clavicle_r.verts,  self._clavicle_r.faces)

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
        self._clavicle_l_world = self._clavicle_l.verts.copy()

        # World-space caches for fascia
        self._fascia_l_world = self._fascia_l_v_local + self.chest_pos
        self._fascia_r_world = self._fascia_r_v_local + self.chest_pos

        # ── upper-arm geometry (child of clavicle, pivot = GH joint) ──────────
        # Both arms share identical capsule geometry; right side mirrors x.
        # Local frame: y = 0 at shoulder (GH pivot), y = −length at elbow.
        _ua_l = _make_upper_arm_capsule()
        _ua_r = _ua_l.mirrored_x()
        self._upper_arm_l_v_local = _ua_l.verts
        self._upper_arm_r_v_local = _ua_r.verts
        self._upper_arm_l_f_np = _ua_l.faces
        self._upper_arm_r_f_np = _ua_r.faces            # same topology; two_sided rendering

        # Glenohumeral DOF angles (in parent clavicle-local frame).
        # flexion  > 0 → arm swings anterior;  abduction > 0 → arm swings lateral.
        self.arm_left_flexion    = 0.0
        self.arm_left_abduction  = 0.0
        self.arm_right_flexion   = 0.0
        self.arm_right_abduction = 0.0

        # Taichi render fields for upper arms
        self.upper_arm_l_v, self.upper_arm_l_f = _make_ti_mesh(
            self._upper_arm_l_v_local, self._upper_arm_l_f_np)
        self.upper_arm_r_v, self.upper_arm_r_f = _make_ti_mesh(
            self._upper_arm_r_v_local, self._upper_arm_r_f_np)

        # World-space caches (populated properly by first update() call below)
        self._upper_arm_l_world = self._upper_arm_l_v_local.copy()
        self._upper_arm_r_world = self._upper_arm_r_v_local.copy()

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

        R_l = _rot_y(self.clavicle_left_yaw) @ _rot_z(self.clavicle_left_pitch)
        R_r = _rot_y(self.clavicle_right_yaw) @ _rot_z(-self.clavicle_right_pitch)

        self._clavicle_l_world = self._transform_verts(
            self._clavicle_l.verts, R_l, self._clavicle_l_offset)
        self._clavicle_r_world = self._transform_verts(
            self._clavicle_r.verts, R_r, self._clavicle_r_offset)

        self.clavicle_l_v.from_numpy(self._clavicle_l_world)
        self.clavicle_r_v.from_numpy(self._clavicle_r_world)

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

        # ── upper arms ────────────────────────────────────────────────────────
        # GH rotations expressed in clavicle-local frame:
        #   flexion   → rotation around clavicle-local +X  (arm swings anterior)
        #   abduction → rotation around clavicle-local +Z  (arm swings lateral)
        # For the right arm the abduction sign is negated so that positive
        # arm_right_abduction always moves the arm away from the body.
        #
        # Sign note: the arm capsule long-axis is −y; rot_x(+θ) sweeps the elbow
        # toward −z (posterior), so we negate to obtain the anatomical convention
        # where +flexion = anterior swing.
        R_gh_l = _rot_x(-self.arm_left_flexion)  @ _rot_z( self.arm_left_abduction)
        R_gh_r = _rot_x(-self.arm_right_flexion) @ _rot_z(-self.arm_right_abduction)

        # Combined world-space rotation: clavicle rotation ∘ GH rotation.
        R_arm_l = R_l @ R_gh_l
        R_arm_r = R_r @ R_gh_r

        # GH pivot world positions (GH local expressed in the clavicle frame,
        # then pushed through the same clavicle transform used for the bone mesh).
        gh_l_world = (_GH_L_LOCAL @ R_l.T) + self.chest_pos + self._clavicle_l_offset
        gh_r_world = (_GH_R_LOCAL @ R_r.T) + self.chest_pos + self._clavicle_r_offset

        # Transform arm vertices: local → world via (R_arm) then translate to GH pivot.
        self._upper_arm_l_world = (self._upper_arm_l_v_local @ R_arm_l.T) + gh_l_world
        self._upper_arm_r_world = (self._upper_arm_r_v_local @ R_arm_r.T) + gh_r_world
        self.upper_arm_l_v.from_numpy(self._upper_arm_l_world.astype(np.float32))
        self.upper_arm_r_v.from_numpy(self._upper_arm_r_world.astype(np.float32))

        # ── ribcage (rigid, parented to chest root) ────────────────────────
        if hasattr(self, '_ribcage_v_local'):
            self._ribcage_world = (self._ribcage_v_local + self.chest_pos).astype(np.float32)
            if self._ribcage_v_p_ext is not None:
                self._ribcage_v_p_ext.from_numpy(self._ribcage_world)

    # ------------------------------------------------------------------
    def set_ribcage_mesh(self, verts_world_np: np.ndarray,
                         v_p_field=None,
                         faces_np: np.ndarray | None = None):
        """Register the ribcage mesh so it tracks ``chest_pos`` each frame.

        Parameters
        ----------
        verts_world_np : (N, 3) ndarray
            Ribcage vertex positions in **world space** at the time of
            registration (current ``chest_pos`` is assumed).
        v_p_field : ti.MatrixField or None
            If provided, ``update()`` will push the recomputed world
            positions into this field each frame so the mesh renders
            correctly without the caller doing anything extra.
        faces_np : (M, 3) int32 ndarray or None
            Surface triangle face array (indices into ``verts_world_np``).
            Required for inward-raycast skin-shell generation.
        """
        self._ribcage_v_local  = (verts_world_np - self.chest_pos).astype(np.float32)
        self._ribcage_world    = verts_world_np.copy().astype(np.float32)
        self._ribcage_v_p_ext  = v_p_field   # may be None
        self._ribcage_f_np     = (np.asarray(faces_np, dtype=np.int32).reshape(-1, 3)
                                  if faces_np is not None
                                  else np.zeros((0, 3), dtype=np.int32))

    def get_ribcage_faces_np(self) -> np.ndarray:
        """Triangle face array (M, 3) for the ribcage mesh.

        Returns an empty ``(0, 3)`` array if no ribcage has been registered
        or if the ribcage was registered without a face array.
        """
        if not hasattr(self, '_ribcage_f_np'):
            return np.zeros((0, 3), dtype=np.int32)
        return self._ribcage_f_np.copy()

    def get_ribcage_verts_world_np(self) -> np.ndarray:
        """Current world-space ribcage vertex positions.

        Returns an empty ``(0, 3)`` array if no ribcage has been registered.
        """
        if not hasattr(self, '_ribcage_world'):
            return np.zeros((0, 3), dtype=np.float32)
        return self._ribcage_world.copy()

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
        return self.clavicle_l_v.to_numpy()#[self._clavicle_l_surf_idx]   # (n_surf, 3)

    def get_clavicle_right_surface_anchors_np(self):
        """World-space positions of the right clavicle surface vertices."""
        return self.clavicle_r_v.to_numpy()#[self._clavicle_r_surf_idx]   # (n_surf, 3)

    def get_fascia_left_surface_anchors_np(self):
        """World-space positions of ALL left fascia vertices (used as ligament anchors)."""
        return self._fascia_l_world.copy()   # (n_fascia, 3)

    def get_fascia_right_surface_anchors_np(self):
        """World-space positions of ALL right fascia vertices (used as ligament anchors)."""
        return self._fascia_r_world.copy()   # (n_fascia, 3)

    def get_clavicle_left_world_np(self) -> np.ndarray:
        """World-space positions of ALL left clavicle vertices."""
        return self._clavicle_l_world.copy()

    def get_clavicle_left_faces_np(self) -> np.ndarray:
        """Triangle face array (M, 3) for the left clavicle mesh."""
        return self._clavicle_l.faces

    def get_clavicle_right_faces_np(self) -> np.ndarray:
        """Triangle face array (M, 3) for the right clavicle mesh.
        Indices reference the same vertex ordering as
        ``get_clavicle_right_world_np()`` (x-mirrored world positions)."""
        return self._clavicle_r.faces

    def get_clavicle_right_world_np(self) -> np.ndarray:
        """World-space positions of ALL right clavicle vertices."""
        return self._clavicle_r_world.copy()

    def get_upper_arm_left_surface_np(self) -> np.ndarray:
        """World-space positions of ALL left upper-arm vertices."""
        return self._upper_arm_l_world.copy()

    def get_upper_arm_right_surface_np(self) -> np.ndarray:
        """World-space positions of ALL right upper-arm vertices."""
        return self._upper_arm_r_world.copy()

    def get_upper_arm_left_faces_np(self) -> np.ndarray:
        """Triangle face array (M, 3) for the left upper-arm mesh."""
        return self._upper_arm_l_f_np

    def get_upper_arm_right_faces_np(self) -> np.ndarray:
        """Triangle face array (M, 3) for the right upper-arm mesh.
        Indices address the vertex ordering of get_upper_arm_right_surface_np()."""
        return self._upper_arm_r_f_np

    # ------------------------------------------------------------------
    def reset_pose(self):
        """Restore all joint angles to anatomical neutral and recompute world positions."""
        self.clavicle_left_pitch  = self.clavicle_rest_pitch   # natural superior bow
        self.clavicle_right_pitch = self.clavicle_rest_pitch
        self.clavicle_left_yaw    = self.clavicle_rest_yaw
        self.clavicle_right_yaw   = -self.clavicle_rest_yaw
        self.arm_left_flexion     = 0.0
        self.arm_left_abduction   = 0.0
        self.arm_right_flexion    = 0.0
        self.arm_right_abduction  = 0.0
        self.update()

    # ------------------------------------------------------------------
    def get_render_draws(self, clavicle=(0.60, 0.35, 0.35),
                         ribcage=(0.55, 0.55, 0.65),
                         fascia=(0.70, 0.80, 0.60),
                         upper_arm=(0.82, 0.65, 0.55)):
        def draw_clavicle_l(scene):
            scene.mesh(self.clavicle_l_v, self.clavicle_l_f,
                       color=clavicle, two_sided=False, show_wireframe=True)
        def draw_clavicle_r(scene):
            scene.mesh(self.clavicle_r_v, self.clavicle_r_f,
                       color=clavicle, two_sided=False)
        def draw_fascia_l(scene):
            scene.mesh(self.fascia_l_v, self.fascia_l_f_ti,
                       color=fascia, two_sided=False)
        def draw_fascia_r(scene):
            scene.mesh(self.fascia_r_v, self.fascia_r_f_ti,
                       color=fascia, two_sided=False)
        def draw_upper_arm_l(scene):
            scene.mesh(self.upper_arm_l_v, self.upper_arm_l_f,
                       color=upper_arm, two_sided=False)
        def draw_upper_arm_r(scene):
            scene.mesh(self.upper_arm_r_v, self.upper_arm_r_f,
                       color=upper_arm, two_sided=False)
        return [draw_clavicle_l, draw_clavicle_r,
                draw_fascia_l,   draw_fascia_r,
                draw_upper_arm_l, draw_upper_arm_r]
