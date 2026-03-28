import taichi as ti
import numpy as np
import math

from cons import framework, deform3d, coopers
from geom import gtet, obj, anatomy
from utils import renderer

ti.init(arch=ti.cpu, cpu_max_num_threads=1)

# ── Breast mesh (LEFT breast, offset to +x) ─────────────────────────────────
mesh = gtet.TetMesh("assets/mesh/breast.msh",
                    rho=1.0, scale=1.0, repose=(0.06, 0.0, 0.0))

g        = ti.Vector([0.0, -1.0, 0.0])
fps      = 60
substep  = 6
solve_step = 2
dt       = 1.0 / (fps * substep)

# ── Bounding box ─────────────────────────────────────────────────────────────
bb_np = np.array([[mesh.v_p.to_numpy()[:, i].min() - 0.5,
                   mesh.v_p.to_numpy()[:, i].max() + 0.5]
                  for i in range(3)], dtype=np.float32)
bb = ti.field(dtype=ti.f32, shape=(3, 2))
bb.from_numpy(bb_np)
box3d = obj.BoundBox3D(bound_box=bb, padding=0.01, bound_epsilon=1e-6)

# ── PBD framework + deformation constraint ───────────────────────────────────
xpbd = framework.pbd_framework(g=g, n_vert=mesh.n_vert, v_p=mesh.v_p,
                               dt=dt, damp=0.99, invm=mesh.v_invm)
deform = deform3d.Deform3D(n=mesh.n_tet, indices=mesh.t_i,
                           invm=mesh.v_invm, pos=mesh.v_p,
                           pos_ref=mesh.v_p_ref, tet_mass=mesh.t_mass,
                           dt=dt, hydro_alpha=1e0, devia_alpha=1e2)
xpbd.add_cons(deform)
xpbd.add_collision(box3d.collision)
xpbd.init_rest_status()

# ── Pin base (z ≈ 0) ─────────────────────────────────────────────────────────
verts_np = mesh.v_p_ref.to_numpy()
z_min, z_max = verts_np[:, 2].min(), verts_np[:, 2].max()
base_mask = verts_np[:, 2] <= z_min + (z_max - z_min) * 0.02
base_idx_np = np.where(base_mask)[0].astype(np.int32)
base_idx = ti.field(dtype=ti.i32, shape=len(base_idx_np))
base_idx.from_numpy(base_idx_np)
mesh.set_fixed_point(len(base_idx_np), base_idx)

# ── Skeleton (clavipectoral fascia + pectorals) ───────────────────────────────
skel = anatomy.Skeleton()

# ── Cooper's ligaments ────────────────────────────────────────────────────────
# Collect unique surface vertex indices from the surface face index buffer
surf_face_idx = mesh.f_i.to_numpy().reshape(-1, 3)
surf_vert_idx = np.unique(surf_face_idx)   # sorted unique vertex indices

ligaments, _anchor_map = coopers.build_coopers(
    skeleton           = skel,
    breast_pos_np      = verts_np,
    breast_surface_idx_np = surf_vert_idx,
    breast_pos_field   = mesh.v_p,
    breast_invm_field  = mesh.v_invm,
    dt                 = dt,
    alpha              = 1e-3,
    pull_only          = True,
    max_attach_dist    = 0.5,
    n_ligaments        = 60,
    outer_z_min        = 0.005,
    pretension         = 1.0,
)
xpbd.add_cons(ligaments)
ligaments.init_rest_status()

# ── Renderer ──────────────────────────────────────────────────────────────────
tirender = renderer.TaichiRenderer3D("Deform 3D – Cooper's Ligaments",
                                     res=(900, 900), fps=fps,
                                     cameraPos=(0.0, 0.15, 0.45),
                                     cameraLookat=(0.0, 0.0, 0.05))

tirender.add_scene_render_draw(mesh.get_render_draw(color=(0.85, 0.65, 0.55),
                                                    wireframe=True))
for draw in skel.get_render_draws():
    tirender.add_scene_render_draw(draw)
tirender.add_scene_render_draw(ligaments.get_render_draw())

# ── GUI ───────────────────────────────────────────────────────────────────────
log_hydro      = [math.log10(deform.hydro_alpha)]
log_devia      = [math.log10(deform.devia_alpha)]
log_lig_alpha  = [math.log10(ligaments.alpha)]
show_ligaments = [True]

def gui_draw(gui):
    gui.text("── Tissue stiffness ──")
    log_hydro[0] = gui.slider_float("log10(hydro)",  log_hydro[0], -3.0, 3.0)
    log_devia[0] = gui.slider_float("log10(devia)",  log_devia[0], -3.0, 3.0)
    deform.hydro_alpha = 10 ** log_hydro[0]
    deform.devia_alpha = 10 ** log_devia[0]
    gui.text(f"  hydro={deform.hydro_alpha:.2e}  devia={deform.devia_alpha:.2e}")

    gui.text("── Cooper's ligaments ──")
    log_lig_alpha[0] = gui.slider_float("log10(lig alpha)", log_lig_alpha[0], -6.0, 0.0)
    ligaments.alpha = 10 ** log_lig_alpha[0]
    gui.text(f"  alpha={ligaments.alpha:.2e}  n={ligaments.n}")

    gui.text("── Pectoral joints ──")
    skel.pec_left_pitch  = gui.slider_float("L pitch", skel.pec_left_pitch,  -0.5, 0.5)
    skel.pec_left_yaw    = gui.slider_float("L yaw",   skel.pec_left_yaw,   -0.5, 0.5)
    skel.pec_right_pitch = gui.slider_float("R pitch", skel.pec_right_pitch, -0.5, 0.5)
    skel.pec_right_yaw   = gui.slider_float("R yaw",   skel.pec_right_yaw,  -0.5, 0.5)

tirender.add_gui_draw(gui_draw)

# ── Main loop ─────────────────────────────────────────────────────────────────
while tirender.window.running:
    tirender.handle_input()

    # Update skeleton world positions, then push to ligament anchors
    skel.update()
    ligaments.update_anchors(skel.get_pec_left_surface_anchors_np())

    for _ in range(substep):
        xpbd.make_prediction_pinned(mesh.v_invm)
        xpbd.preupdate_cons()
        for _ in range(solve_step):
            xpbd.update_cons()
        xpbd.update_vel_pinned(mesh.v_invm)

    tirender.render()

