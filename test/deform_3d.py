import taichi as ti
import numpy as np

from cons import framework, deform3d
from geom import gtet, obj
from utils import renderer

ti.init(arch=ti.cpu, cpu_max_num_threads=1)

mesh = gtet.TetMesh("assets/mesh/breast.msh",
                    rho=1.0,
                    scale=1.0,
                    repose=(0.0, 0.0, 0.0))

g = ti.Vector([0.0, -1.0, 0.0])
fps = 60
substep = 6
solve_step = 2
dt = 1.0 / (fps * substep)

bound_box_np = np.array([[mesh.v_p.to_numpy()[:, i].min() - 0.5,
                           mesh.v_p.to_numpy()[:, i].max() + 0.5]
                          for i in range(3)], dtype=np.float32)
bound_box = ti.field(dtype=ti.f32, shape=(3, 2))
bound_box.from_numpy(bound_box_np)
box3d = obj.BoundBox3D(bound_box=bound_box,
                       padding=0.01,
                       bound_epsilon=1e-6)

xpbd = framework.pbd_framework(g=g,
                               n_vert=mesh.n_vert,
                               v_p=mesh.v_p,
                               dt=dt,
                               damp=0.99)
deform = deform3d.Deform3D(n=mesh.n_tet,
                           indices=mesh.t_i,
                           invm=mesh.v_invm,
                           pos=mesh.v_p,
                           pos_ref=mesh.v_p_ref,
                           tet_mass=mesh.t_mass,
                           dt=dt,
                           hydro_alpha=1e-1,
                           devia_alpha=1e-1)
xpbd.add_cons(deform)
xpbd.add_collision(box3d.collision)
xpbd.init_rest_status()

tirender = renderer.TaichiRenderer3D("Deform 3D",
                                     res=(800, 800),
                                     fps=fps,
                                     cameraPos=(0.0, 0.5, 3.0),
                                     cameraLookat=(0.0, 0.0, 0.0))
tirender.add_scene_render_draw(mesh.get_render_draw(color=(0.8, 0.6, 0.5),
                                                    wireframe=True))

while tirender.window.running:
  tirender.handle_input()

  for sub in range(substep):
    xpbd.make_prediction()
    xpbd.preupdate_cons()
    for _ in range(solve_step):
      xpbd.update_cons()
    xpbd.update_vel()

  tirender.render()
