import numpy as np
import taichi as ti
import meshio

try:
    from PBD_Taichi.geom.distance_field import extract_surface_triangles
except ImportError:
    from geom.distance_field import extract_surface_triangles  # noqa: F401 (re-export)


def read_tet_mesh(filepath):
  mesh = meshio.read(filepath)
  v = mesh.points.astype(np.float32)
  t = mesh.cells_dict['tetra'].astype(np.int32)
  f = extract_surface_triangles(v, t)
  n_v = v.shape[0]
  n_t = t.shape[0]
  n_f = f.shape[0]
  return n_v, n_t, n_f, v, t.flatten(), f.flatten()


def merge_numpy(*parts):
  """Merge multiple (verts, tets_flat, faces_flat) tuples into one.

  Each *part* is ``(v, t_flat, f_flat)`` where indices in *t_flat* and
  *f_flat* are 0-based within that part.  Returns the concatenated arrays
  with the second/third/… parts' indices offset by the accumulated vertex
  count.
  """
  all_v, all_t, all_f = [], [], []
  offset = 0
  for v, t, f in parts:
    all_v.append(v)
    all_t.append(t + offset)
    all_f.append(f + offset)
    offset += len(v)
  return (np.concatenate(all_v, axis=0).astype(np.float32),
          np.concatenate(all_t, axis=0).astype(np.int32),
          np.concatenate(all_f, axis=0).astype(np.int32))


@ti.data_oriented
class TetMesh:

  def __init__(self,
               filepath: str=None,
               v: np.ndarray|None=None, t: np.ndarray|None=None, f: np.ndarray|None=None,
               rho=1.0,
               scale=1.0,
               repose=(0.0, 0.0, 0.0)
  ) -> None:
    """
    Load a tet mesh from file and initialize Taichi fields for vertex positions, indices, and mass.
    :param filepath: Path to the .msh file containing the tet mesh. If provided, v, t, f are ignored.
    :param v: Vertex positions as a numpy array of shape (n_vert, 3). Ignored if filepath is provided.
    :param t: Tetrahedron vertex indices as a numpy array of shape (n_tet, 4). Ignored if filepath is provided.
    :param f: Surface triangle vertex indices as a numpy array of shape (n_face, 3). Ignored if filepath is provided.
    :param rho: Material density (used to compute mass from volume)
    """
    if filepath is not None:
      n_v, n_t, n_f, v, t, f = read_tet_mesh(filepath)
      if v is not None and t is not None and f is not None:
        raise ValueError("Provide either filepath or (v, t, f), not both")
    elif v is not None and t is not None and f is not None:
      n_v, n_t, n_f = v.shape[0], t.shape[0] // 4, f.shape[0] // 3
    else:
      raise ValueError("Either filepath or (v, t, f) must be provided")
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

  def reset_mass(self, rho: float):
    """Refill v_invm and t_mass in-place (no reallocation). Use for sim reset."""
    self.v_invm.fill(0)
    self.t_mass.fill(0)
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
      if self.v_invm[k] > 0.0:
        self.v_invm[k] = 1.0 / self.v_invm[k]
      else:
        self.v_invm[k] = 0.0  # orphan vertex — treat as immovable

  @ti.kernel
  def set_fixed_point(self, n: ti.i32, index: ti.template()):
    for k in range(n):
      self.v_invm[index[k]] = 0.0

  def get_render_draw(self, color=(0.5, 0.5, 0.5), wireframe=False):

    def render_draw(scene: ti.ui.Scene):
      scene.mesh(self.v_p, self.f_i, color=color, show_wireframe=wireframe)

    return render_draw

