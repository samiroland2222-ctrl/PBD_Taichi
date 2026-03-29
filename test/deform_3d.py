import taichi as ti
import numpy as np
import math
import random

from cons import framework, deform3d, coopers
from geom import gtet, obj, anatomy
from utils import renderer, breast_mesh_generator

ti.init(arch=ti.cpu, cpu_max_num_threads=1)

# ── Breast meshes ─────────────────────────────────────────────────────────────
# Left breast: positive x  (patient's left, +x in world)
# Right breast: mirrored at negative x — flip x by negating repose and
#               reflecting the loaded verts through x=0.
# Parameters are slightly randomized per side for naturalistic asymmetry.

def _make_breast_mesh(rho=1.0, scale=1.0, repose=(0.08, 0.0, 0.0),
                      radius=0.07, height=0.06, k=0.7, target_tets=300):
    """Generate a TetMesh directly from the procedural breast mesh generator."""
    coords, node_tags, tet_node_tags, _ = breast_mesh_generator.generate_breast_msh(
        radius=radius, height=height, k=k, target_tets=target_tets)

    # node_tags are 1-indexed; build a mapping to 0-indexed positions
    tag_to_idx = {tag: i for i, tag in enumerate(node_tags)}
    v = coords.astype(np.float32)

    # tet_node_tags are 1-indexed node tags → convert to 0-indexed
    tets = np.array([[tag_to_idx[n] for n in row] for row in tet_node_tags],
                    dtype=np.int32)

    f = gtet.extract_surface_triangles(v, tets)
    t_flat = tets.flatten().astype(np.int32)
    f_flat = f.flatten().astype(np.int32)

    return gtet.TetMesh(v=v, t=t_flat, f=f_flat,
                        rho=rho, scale=scale, repose=repose)

rng = random.Random(42)

def _rand(center, spread):
    """Return center ± spread * U(-1, 1)."""
    return center + spread * (rng.random() * 2 - 1)

# Left breast parameters (slightly randomized)
mesh_l = _make_breast_mesh(
    rho=1.0, scale=1.0, repose=(0.08, 0.0, 0.0),
    radius=_rand(0.070, 0.004),
    height=_rand(0.060, 0.004),
    k=_rand(0.70, 0.05),
    target_tets=300,
)

# Right breast parameters (independently randomized)
mesh_r = _make_breast_mesh(
    rho=1.0, scale=1.0, repose=(0.08, 0.0, 0.0),
    radius=_rand(0.070, 0.004),
    height=_rand(0.060, 0.004),
    k=_rand(0.70, 0.05),
    target_tets=300,
)

# Mirror the right breast through x=0: x → -x gives [-0.14, +0.02]
def _mirror_x(mesh):
    # Flip vertex positions
    v = mesh.v_p.to_numpy();       v[:, 0]    *= -1; mesh.v_p.from_numpy(v)
    vref = mesh.v_p_ref.to_numpy(); vref[:, 0] *= -1; mesh.v_p_ref.from_numpy(vref)
    # Negating x flips handedness → swap two tet vertices to restore positive volume
    t = mesh.t_i.to_numpy().reshape(-1, 4)
    t[:, [0, 1]] = t[:, [1, 0]]
    mesh.t_i.from_numpy(t.flatten().astype(np.int32))
    # Flip surface triangle winding so normals point outward
    f = mesh.f_i.to_numpy().reshape(-1, 3)
    f[:, [0, 1]] = f[:, [1, 0]]
    mesh.f_i.from_numpy(f.flatten().astype(np.int32))
    # Recompute mass with corrected geometry
    mesh.reset_mass(rho=1.0)

_mirror_x(mesh_r)

g          = ti.Vector([0.0, -1.0, 0.0])
fps        = 60
substep    = 6
solve_step = 2
dt         = 1.0 / (fps * substep)

# ── Bounding box (covers both breasts) ───────────────────────────────────────
all_v = np.concatenate([mesh_l.v_p.to_numpy(), mesh_r.v_p.to_numpy()], axis=0)
bb_np = np.array([[all_v[:, i].min() - 0.5, all_v[:, i].max() + 0.5]
                  for i in range(3)], dtype=np.float32)
bb = ti.field(dtype=ti.f32, shape=(3, 2))
bb.from_numpy(bb_np)
box3d = obj.BoundBox3D(bound_box=bb, padding=0.01, bound_epsilon=1e-6)

# ── PBD framework + deformation – LEFT ───────────────────────────────────────
xpbd_l = framework.pbd_framework(g=g, n_vert=mesh_l.n_vert, v_p=mesh_l.v_p,
                                  dt=dt, damp=0.99, invm=mesh_l.v_invm)
deform_l = deform3d.Deform3D(n=mesh_l.n_tet, indices=mesh_l.t_i,
                              invm=mesh_l.v_invm, pos=mesh_l.v_p,
                              pos_ref=mesh_l.v_p_ref, tet_mass=mesh_l.t_mass,
                              dt=dt, hydro_alpha=1e-2, devia_alpha=1e1)
xpbd_l.add_cons(deform_l)
xpbd_l.add_collision(box3d.collision)
xpbd_l.init_rest_status()

# ── PBD framework + deformation – RIGHT ──────────────────────────────────────
xpbd_r = framework.pbd_framework(g=g, n_vert=mesh_r.n_vert, v_p=mesh_r.v_p,
                                  dt=dt, damp=0.99, invm=mesh_r.v_invm)
deform_r = deform3d.Deform3D(n=mesh_r.n_tet, indices=mesh_r.t_i,
                              invm=mesh_r.v_invm, pos=mesh_r.v_p,
                              pos_ref=mesh_r.v_p_ref, tet_mass=mesh_r.t_mass,
                              dt=dt, hydro_alpha=1e-2, devia_alpha=1e1)
xpbd_r.add_cons(deform_r)
xpbd_r.add_collision(box3d.collision)
xpbd_r.init_rest_status()

# ── Pin base (z ≈ 0) – both breasts ──────────────────────────────────────────
def _make_base_pin(mesh):
    v = mesh.v_p_ref.to_numpy()
    z_min, z_max = v[:, 2].min(), v[:, 2].max()
    idx_np = np.where(v[:, 2] <= z_min + (z_max - z_min) * 0.02)[0].astype(np.int32)
    idx_ti = ti.field(dtype=ti.i32, shape=len(idx_np))
    idx_ti.from_numpy(idx_np)
    mesh.set_fixed_point(len(idx_np), idx_ti)
    return idx_np, idx_ti

verts_l_np,  verts_r_np  = mesh_l.v_p_ref.to_numpy(), mesh_r.v_p_ref.to_numpy()
base_idx_l_np, base_idx_l = _make_base_pin(mesh_l)
base_idx_r_np, base_idx_r = _make_base_pin(mesh_r)

# ── Skeleton ──────────────────────────────────────────────────────────────────
skel = anatomy.Skeleton()

# ── Cooper's ligaments – LEFT ─────────────────────────────────────────────────
surf_l = np.unique(mesh_l.f_i.to_numpy().reshape(-1, 3))
ligaments_l, _ = coopers.build_coopers(
    skeleton=skel, breast_pos_np=verts_l_np,
    breast_surface_idx_np=surf_l, breast_pos_field=mesh_l.v_p,
    breast_invm_field=mesh_l.v_invm, dt=dt, alpha=1e3, pull_only=True,
    max_attach_dist=0.5, n_ligaments=90, outer_z_min=0.03, pretension=1.0,
    excluded_vertex_idx=base_idx_l_np, side='left')
xpbd_l.add_cons(ligaments_l)
ligaments_l.init_rest_status()

# ── Cooper's ligaments – RIGHT ────────────────────────────────────────────────
surf_r = np.unique(mesh_r.f_i.to_numpy().reshape(-1, 3))
ligaments_r, _ = coopers.build_coopers(
    skeleton=skel, breast_pos_np=verts_r_np,
    breast_surface_idx_np=surf_r, breast_pos_field=mesh_r.v_p,
    breast_invm_field=mesh_r.v_invm, dt=dt, alpha=1e3, pull_only=True,
    max_attach_dist=0.5, n_ligaments=90, outer_z_min=0.03, pretension=1.0,
    excluded_vertex_idx=base_idx_r_np, side='right')
xpbd_r.add_cons(ligaments_r)
ligaments_r.init_rest_status()

# ── Renderer ──────────────────────────────────────────────────────────────────
tirender = renderer.TaichiRenderer3D("Deform 3D – Cooper's Ligaments",
                                     res=(900, 900), fps=fps,
                                     cameraPos=(0.5, 0.15, 0.2),
                                     cameraLookat=(-0.4, -0.03, -0.17))

skin = (0.85, 0.65, 0.55)
tirender.add_scene_render_draw(mesh_l.get_render_draw(color=skin, wireframe=False))
tirender.add_scene_render_draw(mesh_r.get_render_draw(color=skin, wireframe=False))
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

    gui.text("── Pectoral joints ──")
    skel.pec_left_pitch  = gui.slider_float("L pitch", skel.pec_left_pitch,  -0.5, 0.5)
    skel.pec_left_yaw    = gui.slider_float("L yaw",   skel.pec_left_yaw,   -0.5, 0.5)
    skel.pec_right_pitch = gui.slider_float("R pitch", skel.pec_right_pitch, -0.5, 0.5)
    skel.pec_right_yaw   = gui.slider_float("R yaw",   skel.pec_right_yaw,  -0.5, 0.5)

tirender.add_gui_draw(gui_draw)

# ── Simulation control ────────────────────────────────────────────────────────
sim = {'paused': False, 'step_once': False, 'sim_rate': 1.0, 'frame': 0}

def sim_reset():
    for mesh, xpbd, base_idx_np, base_idx_ti, lig, get_anchors in [
        (mesh_l, xpbd_l, base_idx_l_np, base_idx_l, ligaments_l,
         skel.get_pec_left_surface_anchors_np),
        (mesh_r, xpbd_r, base_idx_r_np, base_idx_r, ligaments_r,
         skel.get_pec_right_surface_anchors_np),
    ]:
        mesh.v_p.copy_from(mesh.v_p_ref)
        xpbd.v_v.fill(0)
        mesh.reset_mass(rho=1.0)
        mesh.set_fixed_point(len(base_idx_np), base_idx_ti)
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
    ligaments_l.update_anchors(skel.get_pec_left_surface_anchors_np())
    ligaments_r.update_anchors(skel.get_pec_right_surface_anchors_np())

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
        for xpbd, mesh in [(xpbd_l, mesh_l), (xpbd_r, mesh_r)]:
            for _ in range(substep):
                xpbd.make_prediction_pinned(mesh.v_invm)
                xpbd.preupdate_cons()
                for _ in range(solve_step):
                    xpbd.update_cons()
                xpbd.update_vel_pinned(mesh.v_invm)
        sim['frame'] += 1

    tirender.render()

