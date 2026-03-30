import os

import taichi as ti
import numpy as np
import math
import random

from cons import framework, deform3d, coopers, breast
from geom import gtet, obj, anatomy, gmesh
from utils import renderer, breast_mesh_generator, parser

from PBD_Taichi.cons.breast import Breast

ti.init(arch=ti.cpu, cpu_max_num_threads=1)

# ── Breast meshes ─────────────────────────────────────────────────────────────
# Left breast: positive x  (patient's left, +x in world)
# Right breast: mirrored at negative x — reflecting the loaded verts through x=0.
# Parameters are slightly randomized per side for naturalistic asymmetry.

rng = random.Random(42)

def _rand(center, spread):
    """Return center ± spread * U(-1, 1)."""
    return center + spread * (rng.random() * 2 - 1)

# ribcage
def _load_ribcage_mesh(scale=1.0, repose=(0, 0, 0)):
    filepath = os.path.join(os.getcwd(), 'assets', 'mesh', 'ribcage_and_pelvis.obj')
    verts, faces = parser.obj_parser(filepath)
    verts = verts*scale
    verts -= verts.mean(axis=0)  # center at origin

    # filter to faces that are y>0
    vert_mask = verts[:, 1] > 0
    # faces is flat (shape [N*3]), reshape to [N, 3] for masking
    faces = faces.reshape(-1, 3)
    face_mask = vert_mask[faces].all(axis=1)
    faces = faces[face_mask]
    # flatten
    faces = faces.flatten().astype(np.int32)

    verts += repose

    masked_verts = verts[vert_mask]

    print(f"visible ribcage verts range: "
          f"x=[{masked_verts[:, 0].min():.3f}, {masked_verts[:, 0].max():.3f}] "
          f"y=[{masked_verts[:, 1].min():.3f}, {masked_verts[:, 1].max():.3f}] "
          f"z=[{masked_verts[:, 2].min():.3f}, {masked_verts[:, 2].max():.3f}]")

    return gmesh.TrianMesh(verts, faces, dim=3, rho=1.0)

ribcage = _load_ribcage_mesh(
    scale=1/50,
    repose=(-0.003, -0.14, -0.1)
)

# Left breast (patient's left, +x in world)
left = Breast.make(
    rho=1.0, scale=1.0, spread=0.5,
    radius=_rand(0.070, 0.004),
    height=_rand(0.060, 0.004),
    k=_rand(0.70, 0.05),
    target_tets=300,
)

# Right breast (mirrored through x=0)
right = Breast.make(
    rho=1.0, scale=1.0, spread=-0.5,
    radius=_rand(0.070, 0.004),
    height=_rand(0.060, 0.004),
    k=_rand(0.70, 0.05),
    target_tets=300,
)


g          = ti.Vector([0.0, -1.0, 0.0])
fps        = 60
substep    = 6
solve_step = 2
dt         = 1.0 / (fps * substep)

# ── Bounding box (covers both breasts) ───────────────────────────────────────
all_v = np.concatenate([left.mesh.v_p.to_numpy(), right.mesh.v_p.to_numpy()], axis=0)
bb_np = np.array([[all_v[:, i].min() - 0.5, all_v[:, i].max() + 0.5]
                  for i in range(3)], dtype=np.float32)
bb = ti.field(dtype=ti.f32, shape=(3, 2))
bb.from_numpy(bb_np)
box3d = obj.BoundBox3D(bound_box=bb, padding=0.01, bound_epsilon=1e-6)

# ── PBD framework + deformation – LEFT ───────────────────────────────────────
xpbd_l = framework.pbd_framework(g=g, n_vert=left.mesh.n_vert, v_p=left.mesh.v_p,
                                  dt=dt, damp=0.99, invm=left.mesh.v_invm)
deform_l = deform3d.Deform3D(n=left.mesh.n_tet, indices=left.mesh.t_i,
                              invm=left.mesh.v_invm, pos=left.mesh.v_p,
                              pos_ref=left.mesh.v_p_ref, tet_mass=left.mesh.t_mass,
                              dt=dt, hydro_alpha=1e-2, devia_alpha=1e1)
xpbd_l.add_cons(deform_l)
xpbd_l.add_collision(box3d.collision)
xpbd_l.init_rest_status()

# ── PBD framework + deformation – RIGHT ──────────────────────────────────────
xpbd_r = framework.pbd_framework(g=g, n_vert=right.mesh.n_vert, v_p=right.mesh.v_p,
                                  dt=dt, damp=0.99, invm=right.mesh.v_invm)
deform_r = deform3d.Deform3D(n=right.mesh.n_tet, indices=right.mesh.t_i,
                              invm=right.mesh.v_invm, pos=right.mesh.v_p,
                              pos_ref=right.mesh.v_p_ref, tet_mass=right.mesh.t_mass,
                              dt=dt, hydro_alpha=1e-2, devia_alpha=1e1)
xpbd_r.add_cons(deform_r)
xpbd_r.add_collision(box3d.collision)
xpbd_r.init_rest_status()

# ── Skeleton ──────────────────────────────────────────────────────────────────
skel = anatomy.Skeleton()

# ── Cooper's ligaments – LEFT ─────────────────────────────────────────────────
ligaments_l, _ = coopers.build_coopers(
    skeleton=skel, breast=left, dt=dt, alpha=1e3, pull_only=True,
    max_attach_dist=0.5, n_ligaments=90, outer_z_min=0.03, pretension=1.0,
    side='left')
xpbd_l.add_cons(ligaments_l)
ligaments_l.init_rest_status()

# ── Cooper's ligaments – RIGHT ────────────────────────────────────────────────
ligaments_r, _ = coopers.build_coopers(
    skeleton=skel, breast=right, dt=dt, alpha=1e3, pull_only=True,
    max_attach_dist=0.5, n_ligaments=90, outer_z_min=0.03, pretension=1.0,
    side='right')
xpbd_r.add_cons(ligaments_r)
ligaments_r.init_rest_status()

# ── Renderer ──────────────────────────────────────────────────────────────────
tirender = renderer.TaichiRenderer3D("Deform 3D – Cooper's Ligaments",
                                     res=(900, 900), fps=fps,
                                     cameraPos=(0.5, 0.15, 0.2),
                                     cameraLookat=(-0.4, -0.03, -0.17))

skin = (0.85, 0.65, 0.55)
tirender.add_scene_render_draw(ribcage.get_render_draw(color=(0.7, 0.7, 0.5), wireframe=False))
tirender.add_scene_render_draw(left.mesh.get_render_draw(color=skin, wireframe=True))
tirender.add_scene_render_draw(right.mesh.get_render_draw(color=skin, wireframe=False))
for draw in skel.get_render_draws():
    tirender.add_scene_render_draw(draw)
tirender.add_scene_render_draw(ligaments_l.get_render_draw())
tirender.add_scene_render_draw(ligaments_r.get_render_draw())

# ── GUI ───────────────────────────────────────────────────────────────────────
log_hydro     = [math.log10(deform_l.hydro_alpha)]
log_devia     = [math.log10(deform_l.devia_alpha)]
log_lig_alpha = [math.log10(ligaments_l.alpha)]

def gui_draw(gui):
    gui.text("── Tissue stiffness ──")
    log_hydro[0] = gui.slider_float("log10(hydro)", log_hydro[0], -3.0, 3.0)
    log_devia[0] = gui.slider_float("log10(devia)", log_devia[0], -3.0, 3.0)
    for d in (deform_l, deform_r):
        d.hydro_alpha = 10 ** log_hydro[0]
        d.devia_alpha = 10 ** log_devia[0]
    gui.text(f"  hydro={deform_l.hydro_alpha:.2e}  devia={deform_l.devia_alpha:.2e}")

    gui.text("── Cooper's ligaments ──")
    log_lig_alpha[0] = gui.slider_float("log10(lig alpha)", log_lig_alpha[0], -1.0, 6.0)
    for lig in (ligaments_l, ligaments_r):
        lig.alpha = 10 ** log_lig_alpha[0]
    gui.text(f"  alpha={ligaments_l.alpha:.2e}  n_l={ligaments_l.n}  n_r={ligaments_r.n}")

    gui.text("── Clavicle joints ──")
    skel.clavicle_left_pitch  = gui.slider_float("L pitch", skel.clavicle_left_pitch, -0.5, 0.6)
    skel.clavicle_left_yaw    = gui.slider_float("L yaw", skel.clavicle_left_yaw, -0.5, 0.5)
    skel.clavicle_right_pitch = gui.slider_float("R pitch", skel.clavicle_right_pitch, -0.5, 0.6)
    skel.clavicle_right_yaw   = gui.slider_float("R yaw", skel.clavicle_right_yaw, -0.5, 0.5)

tirender.add_gui_draw(gui_draw)

# ── Simulation control ────────────────────────────────────────────────────────
sim = {'paused': False, 'step_once': False, 'sim_rate': 1.0, 'frame': 0}

def sim_reset():
    for breast, xpbd, lig, get_anchors in [
        (left,  xpbd_l, ligaments_l, skel.get_fascia_left_surface_anchors_np),
        (right, xpbd_r, ligaments_r, skel.get_fascia_right_surface_anchors_np),
    ]:
        breast.reset()
        xpbd.v_v.fill(0)
        lig.update_anchors(get_anchors())
        xpbd.init_rest_status()
        lig.init_rest_status()
    skel.reset_pose()
    sim['frame'] = 0

def gui_draw_debug(gui):
    gui.text("── Simulation control ──")
    if gui.button("Resume" if sim['paused'] else "Pause"):
        sim['paused'] = not sim['paused']
    if gui.button("Step"):
        sim['paused'] = True;  sim['step_once'] = True
    if gui.button("Reset"):
        sim_reset()
    sim['sim_rate'] = gui.slider_float("Sim rate", sim['sim_rate'], 0.0, 2.0)
    status = 'PAUSED' if sim['paused'] else f"x{sim['sim_rate']:.2f}"
    gui.text(f"  frame={sim['frame']}  {status}")

tirender.add_gui_draw(gui_draw_debug)

# ── Main loop ─────────────────────────────────────────────────────────────────
import time as _time
_accum     = 0.0
_wall_prev = _time.time()

while tirender.window.running:
    tirender.handle_input()

    skel.update()
    ligaments_l.update_anchors(skel.get_fascia_left_surface_anchors_np())
    ligaments_r.update_anchors(skel.get_fascia_right_surface_anchors_np())

    wall_now   = _time.time()
    wall_delta = wall_now - _wall_prev
    _wall_prev = wall_now

    should_step = False
    if sim['step_once']:
        should_step = True;  sim['step_once'] = False
    elif not sim['paused']:
        _accum += wall_delta * sim['sim_rate']
        if _accum >= 1.0 / fps:
            _accum -= 1.0 / fps
            should_step = True

    if should_step:
        for xpbd, breast in [(xpbd_l, left), (xpbd_r, right)]:
            for _ in range(substep):
                xpbd.make_prediction_pinned(breast.mesh.v_invm)
                xpbd.preupdate_cons()
                for _ in range(solve_step):
                    xpbd.update_cons()
                xpbd.update_vel_pinned(breast.mesh.v_invm)
        sim['frame'] += 1

    tirender.render()

