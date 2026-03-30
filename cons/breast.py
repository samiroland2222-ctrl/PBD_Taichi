import math

from PBD_Taichi.geom import gtet
import taichi as ti
import numpy as np

from PBD_Taichi.utils import breast_mesh_generator


class Breast:
    """Encapsulates all per-breast data: mesh, base vertices, surface indices, and rest positions."""

    def __init__(self, mesh: gtet.TetMesh, base_idx_np: np.ndarray, top_idx_np: np.ndarray):
        self.mesh        = mesh
        self.base_idx_np = base_idx_np
        self.top_idx_np  = top_idx_np

        # ── Pin base (z ≈ 0) – both breasts ──────────────────────────────────────────
        def _make_base_pin(mesh, idx_np):
            idx_ti = ti.field(dtype=ti.i32, shape=len(idx_np))
            idx_ti.from_numpy(idx_np)
            mesh.set_fixed_point(len(idx_np), idx_ti)
            return idx_ti

        self.base_idx_ti = _make_base_pin(self.mesh, self.base_idx_np)

    @property
    def surf(self):
        return np.unique(self.mesh.f_i.to_numpy().reshape(-1, 3))

    @property
    def verts_np(self):
        return self.mesh.v_p_ref.to_numpy()

    @classmethod
    def make(cls, **kwargs) -> 'Breast':
        """Create a Breast from _make_breast_mesh keyword arguments."""
        mesh, base_idx_np, top_idx_np = _make_breast_mesh(**kwargs)
        return cls(mesh, base_idx_np, top_idx_np)

    def reset(self):
        self.mesh.v_p.copy_from(self.mesh.v_p_ref)
        self.mesh.reset_mass(rho=1.0)
        self.mesh.set_fixed_point(len(self.base_idx_np), self.base_idx_ti)


def _make_breast_mesh(rho=1.0, scale=1.0, spread=0.0, back_distance=0.125, tilt=-0.2,
                      radius=0.07, height=0.06, k=0.7, target_tets=300):
    """Generate a TetMesh directly from the procedural breast mesh generator.

    Positioning is controlled by two sequential rotations:
      1. Rotate *tilt* radians about the X-axis at the origin (forward/backward lean).
      2. Rotate *spread* radians about the Y-axis pivoting at (0, 0, -back_distance)
         (left/right spread away from the midline).
    """
    coords, node_tags, tet_node_tags, _ = breast_mesh_generator.generate_breast_msh(
        radius=radius, height=height, k=k, target_tets=target_tets)

    # node_tags are 1-indexed; build a mapping to 0-indexed positions
    tag_to_idx = {tag: i for i, tag in enumerate(node_tags)}
    v = coords.astype(np.float64)

    z_min, z_max = v[:, 2].min(), v[:, 2].max()
    base_idx = np.where(v[:, 2] <= z_min + (z_max - z_min) * 0.02)[0].astype(np.int32)
    # top_idx = indexes that are not base_idx
    all_idx = np.arange(len(node_tags))
    top_idx = np.setdiff1d(all_idx, base_idx)

    # Step 1: tilt – rotate about X-axis at origin
    cosx, sinx = math.cos(tilt), math.sin(tilt)
    rot_tilt = np.array([[1,    0,     0],
                         [0, cosx, -sinx],
                         [0, sinx,  cosx]], dtype=np.float64)
    v = v @ rot_tilt.T

    # Step 2: spread – rotate about Y-axis pivoting at (0, 0, -back_distance)
    pivot = np.array([0.0, 0.0, -back_distance], dtype=np.float64)
    cosy, siny = math.cos(spread), math.sin(spread)
    rot_spread = np.array([[ cosy, 0, siny],
                           [    0, 1,    0],
                           [-siny, 0, cosy]], dtype=np.float64)
    v = (v - pivot) @ rot_spread.T + pivot

    v = v.astype(np.float32)

    # tet_node_tags are 1-indexed node tags → convert to 0-indexed
    tets = np.array([[tag_to_idx[n] for n in row] for row in tet_node_tags],
                    dtype=np.int32)

    f = gtet.extract_surface_triangles(v, tets)
    t_flat = tets.flatten().astype(np.int32)
    f_flat = f.flatten().astype(np.int32)

    return gtet.TetMesh(v=v, t=t_flat, f=f_flat,
                        rho=rho, scale=scale), base_idx, top_idx
