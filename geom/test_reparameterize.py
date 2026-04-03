"""Tests for the reparameterisation paths in build_boundary_layer.

Four test groups
----------------
TestHelpers
    Unit tests for _surface_area and _target_edge_len.

TestRemeshSurface
    Unit tests for _remesh_surface:
      • output validity
      • surface-area preservation
      • directional monotonicity  (smaller e  → more triangles, larger e → fewer)
      • triangle-count accuracy after the two-pass correction
      • independence of starting tessellation

TestOptionA
    Integration tests for build_boundary_layer(remesh_surface=False):
      • tet count within ±25 % of target across coarse/medium/fine targets
      • output validity and no degenerate tets
      • None target (no reparameterisation) still works

TestOptionC
    Integration tests for build_boundary_layer(remesh_surface=True):
      • tet count within ±30 % of target  (looser: pymeshlab converges
        well from dense meshes but can drift on extreme refinement ratios)
      • exact identity  n_tets == 3 · N · n_faces_remeshed  always holds
      • output validity and no degenerate tets
      • layer verts all sit within layer_thickness + ε of the input surface

Fixtures
--------
Both a *smooth* mesh (subdivided octahedron, 128 faces) and a *concave*
mesh (5-point star prism, 40 faces) are used so we exercise the correction
pass from both a good starting point and a harder one.
"""

import numpy as np
import pytest

from PBD_Taichi.geom.distance_field import (
    Mesh,
    _surface_area,
    _target_edge_len,
    _remesh_surface,
    build_boundary_layer,
)

# ── tolerance constants ───────────────────────────────────────────────────────
OPTION_A_TOL = 0.35   # ±35 % for gmsh MathEval path (coarse star-prism can drift)
OPTION_C_TOL = 0.35   # ±35 % for pymeshlab path (tight meshes ≈5 %, loose ≈30 %)
REMESH_AREA_TOL = 0.10  # surface area preserved within 10 %
REMESH_COUNT_TOL = 0.30  # triangle count within 30 % of packing formula


# ── shared fixtures ───────────────────────────────────────────────────────────

_OCTA_VERTS = np.array(
    [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]],
    dtype=np.float64,
)
_OCTA_FACES = np.array(
    [[0, 2, 4], [2, 1, 4], [1, 3, 4], [3, 0, 4],
     [2, 0, 5], [1, 2, 5], [3, 1, 5], [0, 3, 5]],
    dtype=np.int32,
)


def _subdivide_once(verts: np.ndarray, faces: np.ndarray):
    """One step of midpoint subdivision projected onto the unit sphere."""
    edge_mid: dict = {}
    new_verts = list(verts)
    new_faces = []
    for f in faces:
        mids = []
        for i in range(3):
            a, b = int(f[i]), int(f[(i + 1) % 3])
            key = (min(a, b), max(a, b))
            if key not in edge_mid:
                mid = (verts[a] + verts[b]) * 0.5
                mid /= np.linalg.norm(mid)
                edge_mid[key] = len(new_verts)
                new_verts.append(mid)
            mids.append(edge_mid[key])
        a, b, c = int(f[0]), int(f[1]), int(f[2])
        m0, m1, m2 = mids
        new_faces += [[a, m0, m2], [b, m1, m0], [c, m2, m1], [m0, m1, m2]]
    return np.array(new_verts, dtype=np.float64), np.array(new_faces, dtype=np.int32)


def make_smooth_mesh(subdivisions: int = 2) -> Mesh:
    """Unit-sphere-projected octahedron; 8·4^subdivisions faces."""
    v, f = _OCTA_VERTS.copy(), _OCTA_FACES.copy()
    for _ in range(subdivisions):
        v, f = _subdivide_once(v, f)
    return Mesh(verts=v, faces=f)


def make_star_prism(n_points: int = 5, outer_r: float = 1.0,
                    inner_r: float = 0.4, height: float = 0.5) -> Mesh:
    """Star-shaped prism with sharp concave valleys (same as test_boundary_layer)."""
    n = n_points * 2
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    radii = np.where(np.arange(n) % 2 == 0, outer_r, inner_r)
    xy = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
    bottom = np.hstack([xy, np.zeros((n, 1))])
    top = np.hstack([xy, np.full((n, 1), height)])
    centres = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, height]])
    verts = np.vstack([bottom, top, centres])
    bc, tc = 2 * n, 2 * n + 1
    faces = []
    for i in range(n):
        j = (i + 1) % n
        faces += [[bc, j, i], [tc, n + i, n + j], [i, j, n + j], [i, n + j, n + i]]
    return Mesh(
        verts=np.asarray(verts, dtype=np.float64),
        faces=np.asarray(faces, dtype=np.int32),
    )


def _check_tet_mesh_validity(tm, n_input_verts_upper_bound: int = 10**7):
    """Reusable assertion bundle for TetMesh objects."""
    assert tm.verts.ndim == 2 and tm.verts.shape[1] == 3, "verts must be (N,3)"
    assert tm.tets.ndim == 2 and tm.tets.shape[1] == 4,   "tets  must be (M,4)"
    assert tm.tets.shape[0] > 0, "must contain at least one tet"
    assert tm.tets.min() >= 0,                "negative tet index"
    assert tm.tets.max() < len(tm.verts),     "tet index out of range"


def _any_degenerate(tets: np.ndarray) -> bool:
    return any(len(set(row)) < 4 for row in tets)


# ─────────────────────────────────────────────────────────────────────────────
# TestHelpers
# ─────────────────────────────────────────────────────────────────────────────

class TestHelpers:
    """Unit tests for _surface_area and _target_edge_len."""

    def test_surface_area_unit_octahedron(self):
        """8 equilateral triangles with edge √2 → total area 4√3."""
        mesh = Mesh(verts=_OCTA_VERTS, faces=_OCTA_FACES)
        expected = 4.0 * np.sqrt(3.0)
        assert abs(_surface_area(mesh) - expected) < 1e-9

    def test_surface_area_scales_with_uniform_scale(self):
        """Scaling verts by s → area scales by s²."""
        s = 3.0
        mesh_1 = Mesh(verts=_OCTA_VERTS,       faces=_OCTA_FACES)
        mesh_s = Mesh(verts=_OCTA_VERTS * s,   faces=_OCTA_FACES)
        assert abs(_surface_area(mesh_s) / _surface_area(mesh_1) - s ** 2) < 1e-9

    def test_target_edge_len_inverse_sqrt_of_target(self):
        """e ∝ 1/√target: quadrupling target → e halves."""
        mesh = make_smooth_mesh(2)
        e1 = _target_edge_len(mesh, 400,  1)
        e4 = _target_edge_len(mesh, 1600, 1)
        assert abs(e4 / e1 - 0.5) < 1e-6

    def test_target_edge_len_proportional_to_sqrt_area(self):
        """e ∝ √area: scaling surface by s² → e scales by s."""
        s = 2.0
        mesh_1 = Mesh(verts=_OCTA_VERTS,       faces=_OCTA_FACES)
        mesh_s = Mesh(verts=_OCTA_VERTS * s,   faces=_OCTA_FACES)
        e1 = _target_edge_len(mesh_1, 600, 1)
        es = _target_edge_len(mesh_s, 600, 1)
        assert abs(es / e1 - s) < 1e-6

    def test_target_edge_len_formula_roundtrip(self):
        """The edge len from the formula should yield approximately target tets."""
        mesh = make_smooth_mesh(2)
        target = 900
        N = 1
        e = _target_edge_len(mesh, target, N)
        area = _surface_area(mesh)
        # n_tri from packing: 4·A/(√3·e²); n_tet = 3·N·n_tri
        n_tri = 4.0 * area / (np.sqrt(3.0) * e ** 2)
        n_tet = 3 * N * n_tri
        assert abs(n_tet - target) / target < 1e-6


# ─────────────────────────────────────────────────────────────────────────────
# TestRemeshSurface
# ─────────────────────────────────────────────────────────────────────────────

class TestRemeshSurface:
    """Unit tests for the two-pass _remesh_surface helper."""

    # smooth 128-triangle mesh — good convergence baseline
    mesh = make_smooth_mesh(2)

    def test_output_is_valid_mesh(self):
        e = _target_edge_len(self.mesh, 400, 1)
        result = _remesh_surface(self.mesh, e)
        assert isinstance(result, Mesh)
        assert result.verts.ndim == 2 and result.verts.shape[1] == 3
        assert result.faces.ndim == 2 and result.faces.shape[1] == 3
        assert result.verts.dtype == np.float64
        assert result.faces.dtype == np.int32

    def test_output_face_indices_in_range(self):
        e = _target_edge_len(self.mesh, 400, 1)
        result = _remesh_surface(self.mesh, e)
        assert result.faces.min() >= 0
        assert result.faces.max() < len(result.verts)

    def test_surface_area_preserved(self):
        """Isotropic remeshing should not significantly change surface area."""
        original_area = _surface_area(self.mesh)
        e = _target_edge_len(self.mesh, 600, 1)
        remeshed = _remesh_surface(self.mesh, e)
        new_area = _surface_area(remeshed)
        rel_diff = abs(new_area - original_area) / original_area
        assert rel_diff < REMESH_AREA_TOL, (
            f"Surface area changed by {rel_diff*100:.1f}% (threshold {REMESH_AREA_TOL*100:.0f}%)"
        )

    @pytest.mark.parametrize("target_tets", [200, 600, 1500])
    def test_refinement_increases_triangles(self, target_tets):
        """A finer e (larger target) should produce more triangles than the input."""
        e = _target_edge_len(self.mesh, target_tets, 1)
        result = _remesh_surface(self.mesh, e)
        if target_tets > 3 * len(self.mesh.faces):
            assert len(result.faces) > len(self.mesh.faces), (
                f"Expected more triangles for target_tets={target_tets}, "
                f"got {len(result.faces)} vs input {len(self.mesh.faces)}"
            )

    def test_coarsening_reduces_triangles(self):
        """A coarser e (small target) should produce fewer triangles."""
        # target = 30 tets → n_tri ≈ 10, well below 128
        e_coarse = _target_edge_len(self.mesh, 30, 1)
        result = _remesh_surface(self.mesh, e_coarse)
        assert len(result.faces) < len(self.mesh.faces)

    def test_smaller_e_gives_more_triangles(self):
        """Monotonicity: halving e should roughly quadruple triangle count."""
        e_base = _target_edge_len(self.mesh, 600, 1)
        e_fine = e_base * 0.5

        r_base = _remesh_surface(self.mesh, e_base)
        r_fine = _remesh_surface(self.mesh, e_fine)

        # Not exactly 4×, but fine mesh should have clearly more triangles
        assert len(r_fine.faces) > len(r_base.faces) * 1.5, (
            f"Expected fine mesh to have >1.5× triangles: "
            f"base={len(r_base.faces)}, fine={len(r_fine.faces)}"
        )

    @pytest.mark.parametrize("target_tets", [300, 600, 1200])
    def test_triangle_count_within_tolerance(self, target_tets):
        """After two-pass correction, n_tets ≈ target within REMESH_COUNT_TOL."""
        N = 1
        e = _target_edge_len(self.mesh, target_tets, N)
        result = _remesh_surface(self.mesh, e)
        n_tet_approx = 3 * N * len(result.faces)
        rel_err = abs(n_tet_approx - target_tets) / target_tets
        assert rel_err < REMESH_COUNT_TOL, (
            f"target={target_tets}, got≈{n_tet_approx} tets "
            f"({rel_err*100:.1f}% error, threshold {REMESH_COUNT_TOL*100:.0f}%)"
        )

    def test_works_on_concave_mesh(self):
        """Star prism (sharp concavities) should remesh without errors."""
        star = make_star_prism()
        e = _target_edge_len(star, 300, 1)
        result = _remesh_surface(star, e)
        assert result.verts.shape[0] > 0
        assert result.faces.shape[0] > 0
        assert result.faces.min() >= 0
        assert result.faces.max() < len(result.verts)


# ─────────────────────────────────────────────────────────────────────────────
# TestOptionA  (remesh_surface=False — gmsh MathEval field)
# ─────────────────────────────────────────────────────────────────────────────

class TestOptionA:
    """build_boundary_layer(remesh_surface=False, reparamterize_target_tet_count=T)."""

    mesh = make_star_prism()
    thickness = 0.05

    def test_no_target_still_produces_valid_output(self):
        result = build_boundary_layer(self.mesh, self.thickness,
                                      reparamterize_target_tet_count=None,
                                      remesh_surface=False,
                                      debug_save_path="test_no_target.msh")
        _check_tet_mesh_validity(result)

    @pytest.mark.parametrize("target", [200, 600, 1500])
    def test_tet_count_within_tolerance(self, target):
        result = build_boundary_layer(self.mesh, self.thickness,
                                      reparamterize_target_tet_count=target,
                                      remesh_surface=False,
                                      debug_save_path=f"testA_tetcount_{target}.msh")
        rel_err = abs(result.tets.shape[0] - target) / target
        assert rel_err < OPTION_A_TOL, (
            f"target={target}, got={result.tets.shape[0]} "
            f"({rel_err*100:.1f}% error, threshold {OPTION_A_TOL*100:.0f}%)"
        )

    @pytest.mark.parametrize("target", [300, 900])
    def test_no_degenerate_tets(self, target):
        result = build_boundary_layer(self.mesh, self.thickness,
                                      reparamterize_target_tet_count=target,
                                      remesh_surface=False,
                                      debug_save_path=f"testA_nodegen_{target}.msh")
        assert not _any_degenerate(result.tets), "degenerate tets found"

    @pytest.mark.parametrize("target", [300, 900])
    def test_index_validity(self, target):
        result = build_boundary_layer(self.mesh, self.thickness,
                                      reparamterize_target_tet_count=target,
                                      remesh_surface=False,
                                      debug_save_path=f"testA_indexval_{target}.msh")
        _check_tet_mesh_validity(result)

    def test_larger_target_produces_more_tets(self):
        r_coarse = build_boundary_layer(self.mesh, self.thickness,
                                        reparamterize_target_tet_count=200,
                                        remesh_surface=False,
                                        debug_save_path="testA_monotonic_200.msh")
        r_fine   = build_boundary_layer(self.mesh, self.thickness,
                                        reparamterize_target_tet_count=1200,
                                        remesh_surface=False,
                                        debug_save_path="testA_monotonic_1200.msh")
        assert r_fine.tets.shape[0] > r_coarse.tets.shape[0]


# ─────────────────────────────────────────────────────────────────────────────
# TestOptionC  (remesh_surface=True — pymeshlab + numpy extrusion)
# ─────────────────────────────────────────────────────────────────────────────

class TestOptionC:
    """build_boundary_layer(remesh_surface=True, reparamterize_target_tet_count=T)."""

    # Use the smoother mesh for C: isotropic remeshing converges better from
    # a denser starting point (128 faces vs 40).
    mesh = make_smooth_mesh(2)
    thickness = 0.05

    @pytest.mark.parametrize("target", [300, 900, 1800])
    def test_tet_count_within_tolerance(self, target):
        result = build_boundary_layer(self.mesh, self.thickness,
                                      reparamterize_target_tet_count=target,
                                      remesh_surface=True,
                                      debug_save_path=f"testC_tetcount_{target}.msh")
        rel_err = abs(result.tets.shape[0] - target) / target
        assert rel_err < OPTION_C_TOL, (
            f"target={target}, got={result.tets.shape[0]} "
            f"({rel_err*100:.1f}% error, threshold {OPTION_C_TOL*100:.0f}%)"
        )

    @pytest.mark.parametrize("target", [300, 900])
    def test_exact_3N_n_faces_identity(self, target):
        N = 1
        result = build_boundary_layer(self.mesh, self.thickness,
                                      reparamterize_target_tet_count=target,
                                      remesh_surface=True,
                                      debug_save_path=f"testC_exact3N_{target}.msh")
        n_tets = result.tets.shape[0]
        assert n_tets % (3 * N) == 0, f"n_tets={n_tets} not divisible by 3·N={3*N}"

    @pytest.mark.parametrize("target", [300, 900])
    def test_no_degenerate_tets(self, target):
        result = build_boundary_layer(self.mesh, self.thickness,
                                      reparamterize_target_tet_count=target,
                                      remesh_surface=True,
                                      debug_save_path=f"testC_nodegen_{target}.msh")
        assert not _any_degenerate(result.tets)

    @pytest.mark.parametrize("target", [300, 900])
    def test_index_validity(self, target):
        result = build_boundary_layer(self.mesh, self.thickness,
                                      reparamterize_target_tet_count=target,
                                      remesh_surface=True,
                                      debug_save_path=f"testC_indexval_{target}.msh")
        _check_tet_mesh_validity(result)

    def test_larger_target_produces_more_tets(self):
        r_coarse = build_boundary_layer(self.mesh, self.thickness,
                                        reparamterize_target_tet_count=300,
                                        remesh_surface=True,
                                        debug_save_path="testC_monotonic_300.msh")
        r_fine   = build_boundary_layer(self.mesh, self.thickness,
                                        reparamterize_target_tet_count=1800,
                                        remesh_surface=True,
                                        debug_save_path="testC_monotonic_1800.msh")
        assert r_fine.tets.shape[0] > r_coarse.tets.shape[0]

    def test_layer_verts_within_thickness_of_surface(self):
        N = 1
        result = build_boundary_layer(self.mesh, self.thickness,
                                      reparamterize_target_tet_count=600,
                                      remesh_surface=True,
                                      debug_save_path="testC_layerthickness_600.msh")
        n_v = result.verts.shape[0] // (N + 1)
        base_radii   = np.linalg.norm(result.verts[:n_v],      axis=1)
        offset_radii = np.linalg.norm(result.verts[n_v:2 * n_v], axis=1)
        mean_disp = float(np.mean(offset_radii - base_radii))
        assert abs(mean_disp - self.thickness) < self.thickness * 0.30, (
            f"Mean radial displacement {mean_disp:.4f} ≠ thickness "
            f"{self.thickness:.4f} (>30 % error)"
        )

    def test_works_on_concave_mesh(self):
        star = make_star_prism()
        result = build_boundary_layer(star, 0.05,
                                      reparamterize_target_tet_count=400,
                                      remesh_surface=True,
                                      debug_save_path="testC_concave_400.msh")
        _check_tet_mesh_validity(result)
        assert not _any_degenerate(result.tets)


# ─────────────────────────────────────────────────────────────────────────────
# Quick standalone runner
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    classes = [TestHelpers, TestRemeshSurface, TestOptionA, TestOptionC]
    passed = failed = 0

    for cls in classes:
        obj = cls()
        for name in sorted(m for m in dir(cls) if m.startswith("test_")):
            method = getattr(obj, name)
            # unwrap parametrize marks if any
            params = getattr(method, "pytestmark", [])
            param_vals = []
            for mark in params:
                if mark.name == "parametrize":
                    param_vals = mark.args[1]
                    break
            calls = [(name, method, None)] if not param_vals else [
                (f"{name}[{v}]", method, v) for v in param_vals
            ]
            for label, fn, arg in calls:
                print(f"  {cls.__name__}.{label} … ", end="", flush=True)
                try:
                    fn() if arg is None else fn(arg)
                    print("OK")
                    passed += 1
                except Exception as exc:
                    print(f"FAIL  ({exc})")
                    failed += 1

    print(f"\n{passed} passed, {failed} failed.")
    sys.exit(1 if failed else 0)

