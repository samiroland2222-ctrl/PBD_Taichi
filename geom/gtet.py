import numpy as np
import taichi as ti
import meshio


def extract_surface_triangles(tets):
  """Extract surface triangles from a tetrahedral mesh.
  Surface faces are those shared by only one tetrahedron."""
  face_count = {}
  tet_faces = [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]]
  for tet in tets:
    for fi in tet_faces:
      face = tuple(sorted([tet[fi[0]], tet[fi[1]], tet[fi[2]]]))
      face_count[face] = face_count.get(face, 0) + 1
  surface = [list(f) for f, cnt in face_count.items() if cnt == 1]
  return np.array(surface, dtype=np.int32)


def read_tet_mesh(filepath):
  mesh = meshio.read(filepath)
  v = mesh.points.astype(np.float32)
  t = mesh.cells_dict['tetra'].astype(np.int32)
  f = extract_surface_triangles(t)
  n_v = v.shape[0]
  n_t = t.shape[0]
  n_f = f.shape[0]
  return n_v, n_t, n_f, v, t.flatten(), f.flatten()


@ti.data_oriented
class TetMesh:

  def __init__(self, filepath, rho=1.0, scale=1.0, repose=(0.0, 0.0, 0.0)) -> None:
    n_v, n_t, n_f, v, t, f = read_tet_mesh(filepath)
    self.n_vert = n_v
    self.n_face = n_f
    self.n_tet = n_t

    v = v * scale
    for i in range(3):
      v[:, i] += repose[i]

    self.v_p = ti.Vector.field(3, dtype=ti.f32, shape=n_v)
    self.v_p.from_numpy(v)
    self.v_p_ref = ti.Vector.field(3, dtype=ti.f32, shape=n_v)
    self.v_p_ref.copy_from(self.v_p)
    self.t_i = ti.field(dtype=ti.i32, shape=self.n_tet * 4)
    self.t_i.from_numpy(t)
    self.f_i = ti.field(dtype=ti.i32, shape=self.n_face * 3)
    self.f_i.from_numpy(f)

    self.compute_mass(rho)

  def compute_mass(self, rho: float):
    self.v_invm = ti.field(dtype=ti.f32, shape=self.n_vert)
    self.t_mass = ti.field(dtype=ti.f32, shape=self.n_tet)
    self.get_mass(rho)

  @ti.kernel
  def get_mass(self, rho: ti.f32):
    for k in range(self.n_tet):
      p1 = self.t_i[k * 4]
      p2 = self.t_i[k * 4 + 1]
      p3 = self.t_i[k * 4 + 2]
      p4 = self.t_i[k * 4 + 3]
      x1 = self.v_p[p1]
      x2 = self.v_p[p2]
      x3 = self.v_p[p3]
      x4 = self.v_p[p4]
      self.t_mass[k] = rho * ti.abs((x4 - x1).dot(
          (x2 - x1).cross(x3 - x1))) / 6.0
      self.v_invm[p1] += self.t_mass[k] / 4.0
      self.v_invm[p2] += self.t_mass[k] / 4.0
      self.v_invm[p3] += self.t_mass[k] / 4.0
      self.v_invm[p4] += self.t_mass[k] / 4.0
    for k in range(self.n_vert):
      self.v_invm[k] = 1.0 / self.v_invm[k]

  @ti.kernel
  def set_fixed_point(self, n: ti.i32, index: ti.template()):
    for k in range(n):
      self.v_invm[index[k]] = 0.0

  def get_render_draw(self, color=(0.5, 0.5, 0.5), wireframe=False):

    def render_draw(scene: ti.ui.Scene):
      scene.mesh(self.v_p, self.f_i, color=color, show_wireframe=wireframe)

    return render_draw

