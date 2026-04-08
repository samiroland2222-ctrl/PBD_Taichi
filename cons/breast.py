import math

from PBD_Taichi.cons import framework, deform3d
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

        self.xbpd = None
        self.deform = None

    def build_xpbd_deform(self, g, dt, world_bounds):
        self.xpbd = framework.pbd_framework(g=g, n_vert=self.mesh.n_vert, v_p=self.mesh.v_p,
                                       dt=dt, damp=0.99, invm=self.mesh.v_invm)
        self.deform = deform3d.Deform3D(n=self.mesh.n_tet, indices=self.mesh.t_i,
                                   invm=self.mesh.v_invm, pos=self.mesh.v_p,
                                   pos_ref=self.mesh.v_p_ref, tet_mass=self.mesh.t_mass,
                                   dt=dt, hydro_alpha=1e-2, devia_alpha=1e1)
        self.xpbd.add_cons(self.deform)
        self.xpbd.add_collision(world_bounds.collision)
        self.xpbd.init_rest_status()

    @property
    def surf(self):
        return np.unique(self.mesh.f_i.to_numpy().reshape(-1, 3))

    @property
    def verts_np(self):
        return self.mesh.v_p_ref.to_numpy()

    @classmethod
    def make_numpy(cls, rho=1.0, scale=1.0, spread=0.0, back_distance=0.125, tilt=0.2,
                   radius=0.07, height=0.06, k=0.7, k2=0.2, target_tets=300):
        """Generate breast mesh data as numpy arrays, without creating Taichi fields.

        Returns (v, t_flat, f_flat, base_idx, top_idx).
        Vertices are already scaled by *scale*.
        """
        coords, node_tags, tet_node_tags, tet_tags, base_vert_index = breast_mesh_generator.generate_breast_msh(
            radius=radius, height=height, k=k, k2=k2, target_tets=target_tets, debug_save_path="breast.msh")

        tag_to_idx = {tag: i for i, tag in enumerate(node_tags)}
        v = coords.astype(np.float64)

        base_idx = base_vert_index.astype(np.int32)
        all_idx = np.arange(len(node_tags))
        top_idx = np.setdiff1d(all_idx, base_idx)

        pivot = np.array([0.0, -0.02, -back_distance], dtype=np.float64)
        cosx, sinx = math.cos(-tilt), math.sin(-tilt)
        cosy, siny = math.cos(spread), math.sin(spread)
        rot = np.array([[cosy, sinx * siny, cosx * siny],
                        [0, cosx, -sinx],
                        [-siny, sinx * cosy, cosx * cosy]], dtype=np.float64)
        v = (v - pivot) @ rot.T + pivot

        v = (v * scale).astype(np.float32)

        tets = np.array([[tag_to_idx[n] for n in row] for row in tet_node_tags],
                        dtype=np.int32)
        f = gtet.extract_surface_triangles(v, tets)
        t_flat = tets.flatten().astype(np.int32)
        f_flat = f.flatten().astype(np.int32)
        return v, t_flat, f_flat, base_idx, top_idx

    @classmethod
    def make(cls, rho=1.0, scale=1.0, spread=0.0, back_distance=0.125, tilt=0.2,
                      radius=0.07, height=0.06, k=0.7, target_tets=300) -> 'Breast':
        """Generate a TetMesh directly from the procedural breast mesh generator.

        Positioning is controlled by two sequential rotations:
          1. Rotate *tilt* radians about the X-axis at the origin (forward/backward lean).
          2. Rotate *spread* radians about the Y-axis pivoting at (0, 0, -back_distance)
             (left/right spread away from the midline).
        """
        v, t_flat, f_flat, base_idx, top_idx = cls.make_numpy(
            rho=rho, scale=scale, spread=spread, back_distance=back_distance,
            tilt=tilt, radius=radius, height=height, k=k, target_tets=target_tets)

        # TetMesh applies scale internally, but we already scaled in make_numpy
        mesh = gtet.TetMesh(v=v, t=t_flat, f=f_flat,
                            rho=rho, scale=1.0)
        return cls(mesh, base_idx, top_idx)

    def reset(self):
        self.mesh.v_p.copy_from(self.mesh.v_p_ref)
        self.mesh.reset_mass(rho=1.0)
        self.mesh.set_fixed_point(len(self.base_idx_np), self.base_idx_ti)
        if self.xpbd is not None:
            self.xpbd.v_v.fill(0)
            self.xpbd.init_rest_status()


class BreastRegion:
    """Provides the Breast-like interface that build_coopers expects."""
    def __init__(self, mesh: gtet.TetMesh,
                 base_idx_np: np.ndarray,
                 top_idx_np: np.ndarray):
        self.mesh = mesh
        self.base_idx_np = base_idx_np
        self.top_idx_np  = top_idx_np
    @property
    def verts_np(self):
        return self.mesh.v_p_ref.to_numpy()
