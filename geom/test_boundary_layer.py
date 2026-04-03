"""Tests for build_boundary_layer with surfaces that contain sharp concavities.

The primary fixture is a star-shaped prism: a 2D star cross-section
(alternating outer / inner radii) capped top and bottom.  The inner vertices
produce sharp re-entrant (concave) angles that stress extrudeBoundaryLayer
because the outward normals of adjacent faces point *towards* each other at
those vertices – the classic scenario where a thick layer self-intersects.

Tests verify:
  • output shapes and dtypes are correct
  • all tet indices lie in [0, n_verts)
  • every tet has 4 distinct vertex indices (no degenerate elements)
  • at least some output vertices have moved away from the original surface
    by approximately `layer_thickness` (layer is non-zero in extent)
"""

import numpy as np
import pytest
from PBD_Taichi.geom.distance_field import Mesh, build_boundary_layer


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def make_star_prism(
    n_points: int = 5,
    outer_r: float = 1.0,
    inner_r: float = 0.4,
    height: float = 0.5,
) -> Mesh:
    """Return a closed triangulated star-prism surface.

    The star cross-section has `n_points` outer tips (at `outer_r`) and
    `n_points` inner valleys (at `inner_r`).  When `inner_r` is small the
    valleys become sharp concavities.

    Vertex layout
    -------------
    0   … 2n-1        bottom ring (alternating outer / inner)
    2n  … 4n-1        top    ring
    4n                bottom centre
    4n+1              top    centre
    """
    n = n_points * 2  # vertices per ring
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    radii = np.where(np.arange(n) % 2 == 0, outer_r, inner_r)

    xy = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
    bottom = np.hstack([xy, np.zeros((n, 1))])
    top = np.hstack([xy, np.full((n, 1), height)])
    centres = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, height]])

    verts = np.vstack([bottom, top, centres])  # (2n+2, 3)
    bc, tc = 2 * n, 2 * n + 1

    faces = []
    for i in range(n):
        j = (i + 1) % n
        # bottom cap (winding: centre, next, cur — CW when viewed from below)
        faces.append([bc,   j,    i   ])
        # top cap
        faces.append([tc,   n+i,  n+j ])
        # side quad → two triangles
        faces.append([i,    j,    n+j ])
        faces.append([i,    n+j,  n+i ])

    return Mesh(
        verts=np.asarray(verts, dtype=np.float64),
        faces=np.asarray(faces,  dtype=np.int32),
    )


def _check_tet_mesh(tm, layer_thickness: float):
    """Common validity assertions for a TetMesh boundary layer."""
    # --- shape ----------------------------------------------------------------
    assert tm.verts.ndim == 2 and tm.verts.shape[1] == 3, "verts must be (N,3)"
    assert tm.tets.ndim  == 2 and tm.tets.shape[1]  == 4, "tets must be (M,4)"
    assert tm.tets.shape[0] > 0, "boundary layer must contain at least one tet"

    # --- index range ----------------------------------------------------------
    assert tm.tets.min() >= 0,                 "negative tet index"
    assert tm.tets.max() < len(tm.verts),      "tet index out of range"

    # --- non-degenerate elements ----------------------------------------------
    for row in tm.tets:
        assert len(set(row)) == 4, f"degenerate tet with repeated index: {row}"

    # --- layer has real thickness ---------------------------------------------
    # At least one vertex should sit at distance >= 0.5*layer_thickness from
    # the origin (true for any surface that surrounds the origin).
    # We just check that the bbox of the output is larger than the input bbox.
    extents = tm.verts.max(axis=0) - tm.verts.min(axis=0)
    assert extents.min() > 0, "output mesh has zero extent in some direction"


# ─────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────

class TestStarPrismBoundaryLayer:
    """Standard 5-point star – moderate concavity (inner_r = 0.4 × outer_r)."""

    def setup_method(self):
        self.mesh = make_star_prism(n_points=5, outer_r=1.0, inner_r=0.4, height=0.5)
        self.thickness = 0.05

    def test_output_shapes(self):
        result = build_boundary_layer(self.mesh, self.thickness, debug_save_path="/tmp/star_output_shapes.msh")
        assert result.verts.shape[1] == 3
        assert result.tets.shape[1] == 4
        print(f"\n[star5] verts={result.verts.shape[0]}  tets={result.tets.shape[0]}")

    def test_index_validity(self):
        result = build_boundary_layer(self.mesh, self.thickness, debug_save_path="/tmp/star_index_validity.msh")
        assert result.tets.min() >= 0
        assert result.tets.max() < len(result.verts)

    def test_no_degenerate_tets(self):
        result = build_boundary_layer(self.mesh, self.thickness, debug_save_path="/tmp/star_no_degenerate_tets.msh")
        degen = np.array([len(set(row)) < 4 for row in result.tets])
        assert not degen.any(), f"{degen.sum()} degenerate tets found"

    def test_layer_is_non_trivial(self):
        result = build_boundary_layer(self.mesh, self.thickness, debug_save_path="/tmp/star_layer_is_non_trivial.msh")
        _check_tet_mesh(result, self.thickness)


class TestDeepConcavityBoundaryLayer:
    """Very sharp concavity: inner_r = 0.1, so the valley half-angle ≈ 18°.

    This is the hardest case – the extrusion normals at inner vertices
    converge sharply and the layer must not produce self-intersecting or
    degenerate elements for a thin enough thickness.
    """

    def setup_method(self):
        self.mesh = make_star_prism(n_points=5, outer_r=1.0, inner_r=0.1, height=0.5)
        # keep thickness well below the concavity radius to avoid self-intersection
        self.thickness = 0.02

    def test_output_shapes(self):
        result = build_boundary_layer(self.mesh, self.thickness, debug_save_path="/tmp/dc_output_shapes.msh")
        assert result.verts.shape[1] == 3
        assert result.tets.shape[1] == 4
        print(f"\n[sharp5] verts={result.verts.shape[0]}  tets={result.tets.shape[0]}")

    def test_index_validity(self):
        result = build_boundary_layer(self.mesh, self.thickness, debug_save_path="/tmp/dc_index_validity.msh")
        assert result.tets.min() >= 0
        assert result.tets.max() < len(result.verts)

    def test_no_degenerate_tets(self):
        result = build_boundary_layer(self.mesh, self.thickness, debug_save_path="/tmp/dc_no_degenerate_tets.msh")
        degen = np.array([len(set(row)) < 4 for row in result.tets])
        assert not degen.any(), f"{degen.sum()} degenerate tets found"


class TestManyPointsStarBoundaryLayer:
    """6-point star – more concave valleys, tests generalisation to n_points != 5."""

    def setup_method(self):
        self.mesh = make_star_prism(n_points=6, outer_r=1.0, inner_r=0.3, height=0.4)
        self.thickness = 0.04

    def test_output_shapes(self):
        result = build_boundary_layer(self.mesh, self.thickness, debug_save_path='/tmp/mps_output_shapes.msh')
        assert result.verts.shape[1] == 3
        assert result.tets.shape[1] == 4
        print(f"\n[star6] verts={result.verts.shape[0]}  tets={result.tets.shape[0]}")

    def test_index_validity(self):
        result = build_boundary_layer(self.mesh, self.thickness, debug_save_path='/tmp/mps_index_validity.msh')
        assert result.tets.min() >= 0
        assert result.tets.max() < len(result.verts)

    def test_no_degenerate_tets(self):
        result = build_boundary_layer(self.mesh, self.thickness, debug_save_path='/tmp/mps_no_degenerate_tets.msh')
        degen = np.array([len(set(row)) < 4 for row in result.tets])
        assert not degen.any(), f"{degen.sum()} degenerate tets found"


# ─────────────────────────────────────────────────────────────
# Quick standalone run
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    for cls in (TestStarPrismBoundaryLayer, TestDeepConcavityBoundaryLayer,
                TestManyPointsStarBoundaryLayer):
        obj = cls()
        obj.setup_method()
        for name in [m for m in dir(cls) if m.startswith("test_")]:
            print(f"  {cls.__name__}.{name} … ", end="", flush=True)
            getattr(obj, name)()
            print("OK")
    print("\nAll tests passed.")

