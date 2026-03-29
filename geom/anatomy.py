"""
Skeletal proxy geometry for the breast simulation.

Hierarchy:
  chest (root, static)
  └── ribcage                 – static visual geometry attached to chest
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
# Ribcage procedural geometry  –  anatomically accurate female skeleton
# ---------------------------------------------------------------------------
#
# Anthropometric sources:
#   • Clauser et al. NASA CR-1537 (1969) – female surface anthropometry
#   • Gayzik et al. J Biomech 2012 – average female thorax geometry
#   • Gray's Anatomy (41st ed.) – rib geometry and sternal attachments
#   • Loveday & Evans Clin Radiol 1989 – female thorax CT measurements
#
# Average female skeleton (~162 cm stature):
#   Thoracic inlet (T1/R1) y ≈ +0.04 m  (above our chest_pos)
#   R7 costal cartilage / xiphoid  y ≈ -0.14 m
#   Bi-costal width at R1  ≈ 110 mm (half = 55 mm per side)
#   Bi-costal width at R7  ≈ 230 mm (half = 115 mm per side)
#   Max AP thoracic depth (at ~R5)  ≈ 175 mm (half-depth ≈ 88 mm)
#
# Coordinate frame (local, origin = chest_pos):
#   x  medial(0) → lateral(+)   left side only; right is mirrored
#   y  inferior(−) → superior(+)
#   z  posterior(−) ← chest wall(0) → anterior(+)
#
# Each rib is modelled as a tube swept along a 3-D parametric curve:
#   • The curve starts at the costo-sternal joint (medial, x≈0, z≈0)
#   • sweeps laterally, posteriorly, and slightly inferiorly following
#     the natural rib angle
#   • ends at a costovertebral joint position (posterior, z ≈ -depth)
#
# Ribs 1-7 are the "true" ribs whose costal cartilages attach to the sternum.
# Ribs 8-10 attach via shared costal cartilage (not individually to sternum).
# Ribs 11-12 are floating – they end freely, lateral and slightly inferior.
#
# We model LEFT-side ribs only; the caller mirrors for the right side.

# Per-rib parameters for the LEFT side.  All dimensions in metres.
#
#   x_start   – x of the ANTERIOR (sternal/chondral) end of the rib bone
#               R1-R7: x≈0 (sternal attachment)
#               R8-R10: x≈0.02-0.04 (attach to costal margin of R7, not sternum)
#               R11-R12: x≈0.06-0.08 (floating, no anterior attachment)
#   y_start   – y of that anterior end (local, rel. to chest_pos)
#   x_lateral – x of the lateral-most point of the rib (widest point)
#   ap_depth  – full AP depth of the thorax at this rib level (z = -ap_depth
#               at the costovertebral joint on the spine)
#   y_cv      – y of the costovertebral joint (usually slightly below y_start
#               because ribs slope inferiorly from sternum to spine)
#   tube_r    – tube cross-section radius
#   n_pts     – number of curve sample points

_RIB_PARAMS = [
    # idx  x_start  y_start  x_lat   x_angle  z_angle  ap_depth  y_cv    tube_r  n_ant n_post
    #
    # z_angle = z of the rib ANGLE (angulus costae).
    # Anatomically the rib angle is the sharpest point of posterior curvature,
    # located ~60-65% laterally.  For a female thorax:
    #   R1:  rib angle ~30mm posterior to chest wall
    #   R5:  rib angle ~70mm posterior  (deepest)
    #   R12: rib angle ~35mm posterior
    #
    # ap_depth = z of costovertebral joint (where rib meets spine).
    # Spine sits a further ~50mm behind this.
    #
    # R1
    (  0,   0.000,   0.035,  0.060,   0.050,  -0.030,   0.055,   0.028,  0.006,  10, 10),
    # R2
    (  1,   0.000,   0.018,  0.080,   0.065,  -0.045,   0.072,   0.007,  0.006,  12, 12),
    # R3
    (  2,   0.000,   0.000,  0.098,   0.078,  -0.058,   0.083,  -0.010,  0.006,  14, 12),
    # R4
    (  3,   0.000,  -0.018,  0.112,   0.088,  -0.065,   0.090,  -0.030,  0.006,  14, 14),
    # R5  (widest, deepest rib angle)
    (  4,   0.000,  -0.036,  0.120,   0.094,  -0.070,   0.095,  -0.052,  0.006,  14, 14),
    # R6
    (  5,   0.000,  -0.056,  0.122,   0.096,  -0.068,   0.092,  -0.074,  0.006,  14, 14),
    # R7
    (  6,   0.000,  -0.076,  0.118,   0.092,  -0.063,   0.086,  -0.096,  0.006,  12, 14),
    # R8  – false rib
    (  7,   0.030,  -0.090,  0.114,   0.096,  -0.057,   0.080,  -0.114,  0.005,  12, 12),
    # R9
    (  8,   0.045,  -0.104,  0.108,   0.094,  -0.050,   0.073,  -0.130,  0.005,  10, 12),
    # R10
    (  9,   0.060,  -0.116,  0.098,   0.088,  -0.043,   0.064,  -0.144,  0.005,  10, 10),
    # R11  floating
    ( 10,   0.075,  -0.126,  0.084,   0.080,  -0.037,   0.054,  -0.154,  0.004,   8, 10),
    # R12  floating, shortest
    ( 11,   0.085,  -0.134,  0.066,   0.062,  -0.030,   0.042,  -0.160,  0.004,   8,  8),
]


def _rib_curve(x_start, y_start, x_lateral, x_angle, z_angle,
               ap_depth, y_cv, n_pts_ant, n_pts_post):
    """
    Two-segment rib centreline joined at the rib ANGLE (angulus costae).

    Segment 1 – Anterior: sternum → rib angle
      Departs laterally (+x), curves gently posteriorly.
      Produces the flatter anterior/lateral face of the rib.

    Segment 2 – Posterior: rib angle → costovertebral joint
      Sweeps sharply posteriorly and medially back to x=0 at the spine.
      Produces the tighter, more curved posterior section.

    C1-continuous at the join: the outgoing tangent of seg2 mirrors the
    incoming tangent of seg1, giving a smooth curve with the characteristic
    D/kidney shape of a real rib when viewed from above.
    """
    y_angle = y_start + (y_cv - y_start) * 0.35   # rib angle sits ~35% of the way down

    # ── Segment 1: anterior arc (sternum → rib angle) ─────────────────────
    q0 = np.array([x_start,   y_start,  0.0],     dtype=np.float64)
    q2 = np.array([x_angle,   y_angle,  z_angle],  dtype=np.float64)

    # Control point: overshoot laterally so the quadratic arc actually
    # reaches x_lateral as its peak.
    # z at 0.50 of z_angle gives the anterior arc enough posterior curvature
    # to produce the correct barrel shape when viewed from above.
    q1 = np.array([x_start + (x_lateral - x_start) * 1.55,
                   y_start + (y_angle - y_start) * 0.30,
                   z_angle * 0.50],               # half-way to rib angle depth
                  dtype=np.float64)

    t1 = np.linspace(0.0, 1.0, n_pts_ant)[:, None]
    seg1 = (1-t1)**2 * q0 + 2*(1-t1)*t1 * q1 + t1**2 * q2

    # ── Segment 2: posterior arc (rib angle → costovertebral joint) ────────
    r0 = q2   # same join point
    r2 = np.array([0.0,   y_cv,  -ap_depth], dtype=np.float64)

    # C1 continuity: incoming tangent at q2 from seg1 = 2*(q2 - q1).
    # Outgoing tangent of seg2 at r0 = 2*(r1 - r0).
    # For C1: r1 = r0 + (r0 - q1)  →  mirrors q1 across the rib angle.
    r1_c1 = r0 + (r0 - q1)

    # Blend r1_c1 toward the spine to get the tighter posterior sweep:
    # pull x toward 0, deepen z further, keep y interpolated.
    r1 = np.array([r1_c1[0] * 0.45,
                   r0[1] + (y_cv - r0[1]) * 0.45,
                   r0[2] + (-ap_depth - r0[2]) * 0.60],
                  dtype=np.float64)

    t2 = np.linspace(0.0, 1.0, n_pts_post)[:, None]
    seg2 = (1-t2)**2 * r0 + 2*(1-t2)*t2 * r1 + t2**2 * r2

    # Concatenate, dropping the duplicated rib-angle point at the join
    curve = np.concatenate([seg1, seg2[1:]], axis=0)
    return curve.astype(np.float32)


def _tube_mesh_along_curve(curve, radius, segs=8):
    """
    Sweep a circular cross-section of given radius along a 3-D polyline.
    Returns verts (N,3), faces (M,3).
    """
    n_pts = len(curve)
    verts = []
    faces = []

    # Build a local frame at each point using the Frenet-Serret method
    # (parallel transport to avoid twisting).
    tangents = np.zeros_like(curve)
    tangents[:-1] = curve[1:] - curve[:-1]
    tangents[-1]  = tangents[-2]
    norms = np.linalg.norm(tangents, axis=1, keepdims=True)
    norms = np.where(norms < 1e-10, 1.0, norms)
    tangents = tangents / norms

    # Seed the first normal perpendicular to the first tangent
    seed = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(tangents[0], seed)) > 0.9:
        seed = np.array([0.0, 1.0, 0.0])
    normal = seed - np.dot(seed, tangents[0]) * tangents[0]
    normal /= np.linalg.norm(normal)

    normals  = [normal]
    binormals = [np.cross(tangents[0], normal)]

    for i in range(1, n_pts):
        n_prev = normals[-1]
        t_cur  = tangents[i]
        # Parallel transport
        n_cur = n_prev - np.dot(n_prev, t_cur) * t_cur
        ln = np.linalg.norm(n_cur)
        if ln < 1e-10:
            n_cur = normals[-1]
        else:
            n_cur /= ln
        normals.append(n_cur)
        binormals.append(np.cross(t_cur, n_cur))

    for i in range(n_pts):
        n  = normals[i]
        b  = binormals[i]
        c  = curve[i]
        base = len(verts)
        for si in range(segs):
            angle = 2 * np.pi * si / segs
            verts.append(c + radius * (np.cos(angle) * n + np.sin(angle) * b))

        if i > 0:
            prev = base - segs
            for si in range(segs):
                a  = prev + si
                bv = prev + (si + 1) % segs
                cv = base + si
                dv = base + (si + 1) % segs
                faces += [[a, cv, bv], [bv, cv, dv]]

    # End caps
    def _cap(ring_start, tip, outward):
        for si in range(segs):
            a = ring_start + si
            b = ring_start + (si + 1) % segs
            if outward:
                faces.append([a, tip, b])
            else:
                faces.append([a, b, tip])

    tip0 = len(verts);  verts.append(curve[0])
    _cap(0, tip0, outward=False)
    tip1 = len(verts);  verts.append(curve[-1])
    _cap((n_pts - 1) * segs, tip1, outward=True)

    return np.array(verts, dtype=np.float32), np.array(faces, dtype=np.int32)


def _spine_verts_local():
    """
    Approximate thoracic vertebral column as a tapered capsule tube
    running along the posterior midline (x=0) from T1 to T12.

    The vertebral bodies sit ~50mm posterior to the posterior rib angle.
    Rib angle z = -ap_depth; spine z = -ap_depth - 0.050.
    The thoracic spine has a natural kyphotic curve (convex posteriorly),
    so z_spine is slightly deeper at mid-thorax than at top/bottom.
    """
    # _RIB_PARAMS columns: idx,x_start,y_start,x_lat,x_angle,z_angle,ap_depth,y_cv,tube_r,n_ant,n_post
    VERT_OFFSET = 0.050   # vertebral bodies ~50mm behind rib angle

    n_verts_col = 14
    segs = 8
    spine_r = 0.014   # ~28mm vertebral body radius

    n_ribs = len(_RIB_PARAMS)
    rib_y_cv    = np.array([p[7] for p in _RIB_PARAMS], dtype=np.float64)
    rib_ap      = np.array([p[6] for p in _RIB_PARAMS], dtype=np.float64)
    rib_z_angle = -rib_ap
    rib_z_spine = rib_z_angle - VERT_OFFSET

    verts = []
    faces = []

    for vi in range(n_verts_col):
        t = vi / (n_verts_col - 1)
        # Interpolate along the rib sequence
        rib_t = t * (n_ribs - 1)
        ri0 = int(np.floor(rib_t)); ri1 = min(ri0 + 1, n_ribs - 1)
        alpha = rib_t - ri0
        y = rib_y_cv[ri0] * (1 - alpha) + rib_y_cv[ri1] * alpha
        z = rib_z_spine[ri0] * (1 - alpha) + rib_z_spine[ri1] * alpha
        r = spine_r * (1.0 - 0.10 * t)

        base = len(verts)
        for si in range(segs):
            angle = 2 * np.pi * si / segs
            verts.append([r * np.cos(angle), y, z + r * np.sin(angle)])
        if vi > 0:
            prev = base - segs
            for si in range(segs):
                a  = prev + si;        bv = prev + (si+1) % segs
                cv = base + si;        dv = base + (si+1) % segs
                faces += [[a, cv, bv], [bv, cv, dv]]

    # end caps
    y0, z0 = rib_y_cv[0],  rib_z_spine[0]
    y1, z1 = rib_y_cv[-1], rib_z_spine[-1]
    tip0 = len(verts); verts.append([0.0, y0, z0])
    for si in range(segs):
        faces.append([si, (si+1) % segs, tip0])
    tip1 = len(verts); verts.append([0.0, y1, z1])
    last = (n_verts_col - 1) * segs
    for si in range(segs):
        faces.append([last + si, tip1, last + (si+1) % segs])

    return np.array(verts, dtype=np.float32), np.array(faces, dtype=np.int32)


def _ribcage_verts_local():
    """
    Build a full left+right ribcage from 12 individually modelled rib bones
    plus the thoracic spine, using average female anthropometry.

    Returns verts (N,3), faces (M,3)  in local chest space
    (origin = chest_pos, z=0 = anterior chest wall).
    """
    all_verts = []
    all_faces = []

    for (_idx, x_start, y_start, x_lateral, x_angle, z_angle,
         ap_depth, y_cv, tube_r, n_ant, n_post) in _RIB_PARAMS:

        curve_l = _rib_curve(x_start, y_start, x_lateral, x_angle, z_angle,
                             ap_depth, y_cv, n_ant, n_post)
        v_l, f_l = _tube_mesh_along_curve(curve_l, radius=tube_r, segs=8)

        # Mirror for right side: flip x
        curve_r = curve_l.copy(); curve_r[:, 0] *= -1
        v_r, f_r = _tube_mesh_along_curve(curve_r, radius=tube_r, segs=8)
        # Flip winding on right side (mirroring reverses handedness)
        f_r = f_r[:, ::-1]

        off_l = sum(len(v) for v in all_verts)
        all_verts.append(v_l);  all_faces.append(f_l + off_l)

        off_r = sum(len(v) for v in all_verts)
        all_verts.append(v_r);  all_faces.append(f_r + off_r)

    # Vertebral column
    v_sp, f_sp = _spine_verts_local()
    off_sp = sum(len(v) for v in all_verts)
    all_verts.append(v_sp);  all_faces.append(f_sp + off_sp)

    verts = np.concatenate(all_verts, axis=0).astype(np.float32)
    faces = np.concatenate(all_faces, axis=0).astype(np.int32)

    # ── Thoracic kyphosis: tilt the whole ribcage ~15° about x, pivoting at
    # mid-thorax (y≈-0.06 in local coords).  Negative angle: upper sternum
    # tilts anteriorly (+z), lower thorax tilts posteriorly (−z).
    kyphosis_deg = -15.0
    a = np.deg2rad(kyphosis_deg)
    c, s = np.cos(a), np.sin(a)
    Rx = np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float32)
    pivot_y = -0.06  # mid-thorax y
    verts[:, 1] -= pivot_y
    verts = (verts @ Rx.T)
    verts[:, 1] += pivot_y

    return verts, faces


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

        # ── ribcage (static, attached to chest root) ───────────────────────
        (self._ribcage_v_local,
         self._ribcage_f_np) = _ribcage_verts_local()

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

        # Ribcage (world = chest_pos + local, static)
        ribcage_world = self._ribcage_v_local + self.chest_pos
        self.ribcage_v, self.ribcage_f = _make_ti_mesh(
            ribcage_world, self._ribcage_f_np)

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
        def draw_ribcage(scene):
            scene.mesh(self.ribcage_v, self.ribcage_f,
                       color=ribcage, two_sided=True)
        def draw_fascia_l(scene):
            scene.mesh(self.fascia_l_v, self.fascia_l_f_ti,
                       color=fascia, two_sided=True)
        def draw_fascia_r(scene):
            scene.mesh(self.fascia_r_v, self.fascia_r_f_ti,
                       color=fascia, two_sided=True)
        return [draw_clavicle_l, draw_clavicle_r,
                draw_ribcage,
                draw_fascia_l, draw_fascia_r]
