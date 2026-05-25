import os

import taichi as ti
import numpy as np
import math
import random

from cons import framework, deform3d, coopers, breast
from cons.torso import UnifiedTorso
from geom import gtet, obj, anatomy, gmesh
from utils import renderer, breast_mesh_generator, parser

from PBD_Taichi.cons.torso import SKIN_HYDRO_ALPHA, SKIN_DEVIA_ALPHA

ti.init(arch=ti.cpu, cpu_max_num_threads=1)

# ── Ribcage mesh (static visual + optional skin anchor source) ────────────────

def _load_mesh(filepath, scale=1.0, repose=(0, 0, 0)):
    verts, faces = parser.obj_parser(filepath)
    verts = verts * scale
    verts -= verts.mean(axis=0)
    verts += repose
    return gmesh.TrianMesh(verts, faces, dim=3, rho=1.0,
                           get_edge=False, get_edgeside=False,
                           get_edgeNeib=False, get_faceedge=False)

skeleton_mesh = _load_mesh(
    filepath=os.path.join(os.getcwd(), 'assets', 'mesh', 'female_skeleton_first_anatomy_study.OBJ'),
    scale=1/10,
    repose=(0, -0.255, -0.08)
)

ribcage_mesh = _load_mesh(
    filepath=os.path.join(os.getcwd(), 'assets', 'mesh', 'ribcage.obj'),
    scale=1/49,
    repose=(-0.006, 0.04, -0.085)
)

# ── Skeleton (clavicles + upper arms) ─────────────────────────────────────────
skel = anatomy.Skeleton()

# Register the ribcage with the skeleton so it tracks chest_pos each frame.
# Skeleton stores local coords (world − chest_pos) and writes world positions
# back into ribcage_mesh.v_p on every skel.update() call.
skel.set_ribcage_mesh(ribcage_mesh.v_p.to_numpy(),
                      v_p_field=ribcage_mesh.v_p,
                      faces_np=ribcage_mesh.faces_np)

# ── Simulation parameters ─────────────────────────────────────────────────────
g          = (0.0, -9.8, 0.0)
fps        = 60
substep    = 6
solve_step = 2
dt         = 1.0 / (fps * substep)

# ── UnifiedTorso: merged breasts + skin shell ─────────────────────────────────
ribcage_verts = ribcage_mesh.v_p.to_numpy()
torso = UnifiedTorso(
    skel,
    breast_height=0.04,
    breast_radius=0.06,
    breast_k=1.0,
    breast_spread=0.6,
    breast_tilt=0.3,
    breast_target_tets=600,
    ribcage_verts_np=ribcage_verts,
    ribcage_faces_np=ribcage_mesh.faces_np,
    skin_target_n_tets=10000,
    skin_thickness=0.02,
    g=g,
    dt=dt,
    fps=fps,
    substep=substep,
)

# ── Bounding box ──────────────────────────────────────────────────────────────
all_v = torso.skin_mesh.v_p.to_numpy()
bb_np = np.array([[all_v[:, i].min() - 0.5, all_v[:, i].max() + 0.5]
                  for i in range(3)], dtype=np.float32)
bb = ti.field(dtype=ti.f32, shape=(3, 2))
bb.from_numpy(bb_np)
box3d = obj.BoundBox3D(bound_box=bb, padding=0.01, bound_epsilon=1e-6)
torso.xpbd.add_collision(box3d.collision)

# ── Renderer ──────────────────────────────────────────────────────────────────
tirender = renderer.TaichiRenderer3D("Deform 3D – Unified Torso",
                                     res=(900, 900), fps=fps,
                                     cameraPos=(0.5, 0.15, 0.2),
                                     cameraLookat=(-0.4, -0.03, -0.17))

skin_color = (0.85, 0.65, 0.55)
tirender.add_scene_render_draw(skeleton_mesh.get_render_draw(color=(0.7, 0.7, 0.5), wireframe=False))
tirender.add_scene_render_draw(ribcage_mesh.get_render_draw(color=(0.7, 0.7, 0.5), wireframe=False))
for draw in torso.get_render_draws():
    tirender.add_scene_render_draw(draw)
#for draw in torso.get_skin_draws(color=(0.1, 1.0, 0.1)):          # raycast skin surface
#    tirender.add_scene_render_draw(draw)
for draw in skel.get_render_draws():
    tirender.add_scene_render_draw(draw)
for draw in torso.get_ligament_draws():
    tirender.add_scene_render_draw(draw)
for draw in torso.get_skin_anchor_draws():   # reddish bilateral breast-skin springs
    tirender.add_scene_render_draw(draw)

# ── Surface-normal overlay (wgpu backend: yellow lines; Taichi backend: no-op) ─
if torso.skin_f_i is not None:
    _skin_fi_np = torso.skin_f_i.to_numpy().reshape(-1, 3)
    _normals_draw = tirender.make_normals_draw_callback(
        lambda: torso.skin_mesh.v_p.to_numpy().astype(np.float32),
        lambda: _skin_fi_np,
    )
    tirender.add_scene_render_draw(_normals_draw)

# ── GUI ───────────────────────────────────────────────────────────────────────
log_b_hydro   = [math.log10(torso.deform_breast.hydro_alpha)]
log_b_devia   = [math.log10(torso.deform_breast.devia_alpha)]
log_s_hydro   = [math.log10(torso.deform_skin.hydro_alpha)]  if torso.deform_skin else [math.log10(SKIN_HYDRO_ALPHA)]
log_s_devia   = [math.log10(torso.deform_skin.devia_alpha)]  if torso.deform_skin else [math.log10(SKIN_DEVIA_ALPHA)]
log_lig_alpha  = [math.log10(torso.ligaments_l.alpha)] if torso.ligaments_l else [0.0]
log_fascia_alpha = [math.log10(torso.breast_l_fascia_springs.alpha)] if torso.breast_l_fascia_springs and torso.breast_l_fascia_springs.n > 0 else [-4.0]
log_skin_alpha = [math.log10(torso.skin_anchors.alpha)] if torso.skin_anchors.n > 0 else [0.0]
log_skel_alpha = [math.log10(torso.skeleton_skin_springs.alpha)] if torso.skeleton_skin_springs.n > 0 else [0.0]
log_rc_alpha   = [math.log10(torso.ribcage_skin_springs.alpha)] if torso.ribcage_skin_springs.n > 0 else [0.0]

def gui_draw(gui):
    gui.text("-- Breast tissue stiffness --")
    log_b_hydro[0] = gui.slider_float("log10(breast hydro)", log_b_hydro[0], -3.0, 3.0)
    log_b_devia[0] = gui.slider_float("log10(breast devia)", log_b_devia[0], -3.0, 3.0)
    torso.deform_breast.hydro_alpha = 10 ** log_b_hydro[0]
    torso.deform_breast.devia_alpha = 10 ** log_b_devia[0]
    gui.text(f"  hydro={torso.deform_breast.hydro_alpha:.2e}  devia={torso.deform_breast.devia_alpha:.2e}")

    if torso.deform_skin:
        gui.text("-- Skin / subcut-fat stiffness --")
        log_s_hydro[0] = gui.slider_float("log10(skin hydro)", log_s_hydro[0], -3.0, 3.0)
        log_s_devia[0] = gui.slider_float("log10(skin devia)", log_s_devia[0], -3.0, 3.0)
        torso.deform_skin.hydro_alpha = 10 ** log_s_hydro[0]
        torso.deform_skin.devia_alpha = 10 ** log_s_devia[0]
        gui.text(f"  hydro={torso.deform_skin.hydro_alpha:.2e}  devia={torso.deform_skin.devia_alpha:.2e}")

    gui.text("-- Cooper's ligaments --")
    log_lig_alpha[0] = gui.slider_float("log10(lig)", log_lig_alpha[0], -1.0, 6.0)
    if torso.ligaments_l:
        torso.ligaments_l.alpha = 10 ** log_lig_alpha[0]
        torso.ligaments_r.alpha = 10 ** log_lig_alpha[0]
        gui.text(f"  alpha={torso.ligaments_l.alpha:.2e}")

    if torso.breast_l_fascia_springs and torso.breast_l_fascia_springs.n > 0:
        gui.text("-- Breast-base <-> fascia springs --")
        log_fascia_alpha[0] = gui.slider_float("log10(fascia)", log_fascia_alpha[0], -3.0, 6.0)
        torso.breast_l_fascia_springs.alpha = 10 ** log_fascia_alpha[0]
        torso.breast_r_fascia_springs.alpha = 10 ** log_fascia_alpha[0]
        gui.text(f"  alpha={torso.breast_l_fascia_springs.alpha:.2e}"
                 f"  n_L={torso.breast_l_fascia_springs.n}"
                 f"  n_R={torso.breast_r_fascia_springs.n}")

    if torso.skin_anchors.n > 0:
        gui.text("-- Breast-skin springs --")
        log_skin_alpha[0] = gui.slider_float("log10(breast-skin)", log_skin_alpha[0], -4.0, 6.0)
        torso.skin_anchors.alpha = 10 ** log_skin_alpha[0]
        gui.text(f"  alpha={torso.skin_anchors.alpha:.2e}  n={torso.skin_anchors.n}")

    if torso.skeleton_skin_springs.n > 0:
        gui.text("-- Skeleton springs (clav+arm) --")
        log_skel_alpha[0] = gui.slider_float("log10(skel)", log_skel_alpha[0], -4.0, 6.0)
        torso.skeleton_skin_springs.alpha = 10 ** log_skel_alpha[0]
        gui.text(f"  alpha={torso.skeleton_skin_springs.alpha:.2e}  n={torso.skeleton_skin_springs.n}")

    if torso.ribcage_skin_springs.n > 0:
        gui.text("-- Ribcage-skin springs --")
        log_rc_alpha[0] = gui.slider_float("log10(ribcage-skin)", log_rc_alpha[0], -4.0, 6.0)
        torso.ribcage_skin_springs.alpha = 10 ** log_rc_alpha[0]
        gui.text(f"  alpha={torso.ribcage_skin_springs.alpha:.2e}  n={torso.ribcage_skin_springs.n}")

    gui.text("-- Clavicle joints --")
    skel.clavicle_left_pitch  = gui.slider_float("L pitch", skel.clavicle_left_pitch, -0.5, 0.6)
    skel.clavicle_left_yaw    = gui.slider_float("L yaw", skel.clavicle_left_yaw, -0.5, 0.5)
    skel.clavicle_right_pitch = gui.slider_float("R pitch", skel.clavicle_right_pitch, -0.5, 0.6)
    skel.clavicle_right_yaw   = gui.slider_float("R yaw", skel.clavicle_right_yaw, -0.5, 0.5)

    gui.text("-- Arm joints --")
    skel.arm_left_flexion    = gui.slider_float("L flex", skel.arm_left_flexion, -1.0, 2.5)
    skel.arm_left_abduction  = gui.slider_float("L abd", skel.arm_left_abduction, -0.3, 2.5)
    skel.arm_right_flexion   = gui.slider_float("R flex", skel.arm_right_flexion, -1.0, 2.5)
    skel.arm_right_abduction = gui.slider_float("R abd", skel.arm_right_abduction, -0.3, 2.5)

tirender.add_gui_draw(gui_draw)

# ── Simulation control ────────────────────────────────────────────────────────
sim = {'paused': False, 'step_once': False, 'sim_rate': 1.0, 'frame': 0}

def sim_reset():
    torso.reset()
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

    # Update skeleton and kinematic skin anchors once per frame
    skel.update()
    torso.update_kinematic_skin()
    if torso.ligaments_l:
        torso.ligaments_l.update_anchors(skel.get_fascia_left_surface_anchors_np())
        torso.ligaments_r.update_anchors(skel.get_fascia_right_surface_anchors_np())

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
        for _ in range(substep):
            torso.xpbd.make_prediction_pinned(torso.skin_mesh.v_invm)
            torso.xpbd.preupdate_cons()
            for _ in range(solve_step):
                torso.xpbd.update_cons()
            torso.xpbd.update_vel_pinned(torso.skin_mesh.v_invm)
        sim['frame'] += 1

    tirender.render()
