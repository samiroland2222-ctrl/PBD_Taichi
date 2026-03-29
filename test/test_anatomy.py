"""
Unit tests for geom/anatomy.py covering:
  - Phase 1: Ribcage geometry generation and world-space transform
  - Phase 2: Clavipectoral fascia surface generation, kinematic binding,
             and anchor getters
"""

import numpy as np
import taichi as ti
import pytest

ti.init(arch=ti.cpu, cpu_max_num_threads=1)

from geom import anatomy
from geom.anatomy import (
    _ribcage_verts_local,
    _fascia_verts_local,
    Skeleton,
)


# ───────────────────────────────────────────────────────────────────────────
# Phase 1 – Ribcage
# ───────────────────────────────────────────────────────────────────────────

class TestRibcageGeometry:
    """_ribcage_verts_local returns valid geometry."""

    def test_returns_arrays(self):
        verts, faces = _ribcage_verts_local()
        assert isinstance(verts, np.ndarray)
        assert isinstance(faces, np.ndarray)

    def test_vertex_shape(self):
        verts, _ = _ribcage_verts_local(n_ribs=7, segs=12)
        assert verts.ndim == 2
        assert verts.shape[1] == 3
        assert len(verts) == 7 * 12   # n_ribs * segs

    def test_face_shape(self):
        _, faces = _ribcage_verts_local(n_ribs=7)
        # At least some faces generated (n_ribs-1 rows of segs quads)
        assert faces.shape[1] == 3
        assert len(faces) > 0

    def test_z_nonnegative(self):
        """Ribcage vertices should not extend behind the chest wall (z < 0)."""
        verts, _ = _ribcage_verts_local()
        assert np.all(verts[:, 2] >= -1e-6), \
            "Some ribcage verts have z < 0 (behind chest wall)"

    def test_skeleton_ribcage_world_transform(self):
        """Ribcage world positions = local + chest_pos."""
        chest_pos = np.array([0.0, 0.08, 0.0], dtype=np.float32)
        skel = Skeleton(chest_pos=chest_pos.tolist())

        local_v, _ = _ribcage_verts_local()
        expected_world = local_v + chest_pos

        actual_world = skel.ribcage_v.to_numpy()
        np.testing.assert_allclose(actual_world, expected_world, atol=1e-5,
                                   err_msg="Ribcage world verts != local + chest_pos")

    def test_skeleton_ribcage_static_after_clavicle_move(self):
        """Moving clavicle pitch/yaw must NOT change the ribcage positions."""
        skel = Skeleton()
        before = skel.ribcage_v.to_numpy().copy()
        skel.clavicle_left_pitch = 0.4
        skel.clavicle_right_yaw  = 0.3
        skel.update()
        after = skel.ribcage_v.to_numpy()
        np.testing.assert_allclose(before, after, atol=1e-6,
                                   err_msg="Ribcage should be static – unchanged by clavicle rotation")


# ───────────────────────────────────────────────────────────────────────────
# Phase 2 – Fascia
# ───────────────────────────────────────────────────────────────────────────

class TestFasciaGeometry:
    """_fascia_verts_local returns a valid quad grid."""

    def test_returns_arrays(self):
        verts, faces, top_idx, bot_idx = _fascia_verts_local()
        assert isinstance(verts, np.ndarray)
        assert isinstance(faces, np.ndarray)

    def test_vertex_count(self):
        rows, cols = 5, 6
        verts, _, top_idx, bot_idx = _fascia_verts_local(rows=rows, cols=cols)
        assert len(verts) == rows * cols
        assert len(top_idx) == cols
        assert len(bot_idx) == cols

    def test_face_count(self):
        rows, cols = 5, 6
        _, faces, _, _ = _fascia_verts_local(rows=rows, cols=cols)
        # Each quad cell → 2 triangles; (rows-1)*(cols-1) quads
        assert len(faces) == 2 * (rows - 1) * (cols - 1)

    def test_top_row_indices_are_row0(self):
        rows, cols = 5, 6
        verts, _, top_idx, _ = _fascia_verts_local(rows=rows, cols=cols)
        expected = np.arange(cols)
        np.testing.assert_array_equal(top_idx, expected)

    def test_bottom_row_indices_are_last_row(self):
        rows, cols = 5, 6
        verts, _, _, bot_idx = _fascia_verts_local(rows=rows, cols=cols)
        expected = np.arange((rows - 1) * cols, rows * cols)
        np.testing.assert_array_equal(bot_idx, expected)


class TestFasciaKinematics:
    """Skeleton fascia deforms correctly with clavicle motion."""

    def _make_skel(self, **kwargs):
        return Skeleton(**kwargs)

    def test_fascia_left_anchor_shape(self):
        skel = self._make_skel()
        anchors = skel.get_fascia_left_surface_anchors_np()
        assert anchors.ndim == 2
        assert anchors.shape[1] == 3

    def test_fascia_right_anchor_shape(self):
        skel = self._make_skel()
        anchors = skel.get_fascia_right_surface_anchors_np()
        assert anchors.ndim == 2
        assert anchors.shape[1] == 3

    def test_bottom_edge_static_left(self):
        """Left fascia bottom row must not move when clavicle pitch changes."""
        skel = self._make_skel()
        bot_idx = skel._fascia_l_bot_idx
        before = skel.get_fascia_left_surface_anchors_np()[bot_idx].copy()

        skel.clavicle_left_pitch = 0.4
        skel.clavicle_left_yaw   = 0.3
        skel.update()
        after = skel.get_fascia_left_surface_anchors_np()[bot_idx]

        np.testing.assert_allclose(before, after, atol=1e-5,
                                   err_msg="Left fascia bottom edge must remain fixed")

    def test_bottom_edge_static_right(self):
        """Right fascia bottom row must not move when clavicle pitch changes."""
        skel = self._make_skel()
        bot_idx = skel._fascia_r_bot_idx
        before = skel.get_fascia_right_surface_anchors_np()[bot_idx].copy()

        skel.clavicle_right_pitch = 0.4
        skel.clavicle_right_yaw   = 0.3
        skel.update()
        after = skel.get_fascia_right_surface_anchors_np()[bot_idx]

        np.testing.assert_allclose(before, after, atol=1e-5,
                                   err_msg="Right fascia bottom edge must remain fixed")

    def test_top_edge_moves_with_clavicle_left(self):
        """Left fascia top row should change when clavicle rotates."""
        skel = self._make_skel()
        top_idx = skel._fascia_l_top_idx
        before = skel.get_fascia_left_surface_anchors_np()[top_idx].copy()

        skel.clavicle_left_pitch = 0.4
        skel.update()
        after = skel.get_fascia_left_surface_anchors_np()[top_idx]

        assert not np.allclose(before, after, atol=1e-4), \
            "Left fascia top edge should move with clavicle rotation"

    def test_top_edge_moves_with_clavicle_right(self):
        """Right fascia top row should change when clavicle rotates."""
        skel = self._make_skel()
        top_idx = skel._fascia_r_top_idx
        before = skel.get_fascia_right_surface_anchors_np()[top_idx].copy()

        skel.clavicle_right_pitch = 0.4
        skel.update()
        after = skel.get_fascia_right_surface_anchors_np()[top_idx]

        assert not np.allclose(before, after, atol=1e-4), \
            "Right fascia top edge should move with clavicle rotation"

    def test_interior_rows_interpolated(self):
        """Interior fascia rows should lie between top and bottom in y."""
        skel = self._make_skel()
        anchors = skel.get_fascia_left_surface_anchors_np()

        cols = len(skel._fascia_l_top_idx)
        rows = len(anchors) // cols

        top_y = anchors[skel._fascia_l_top_idx, 1].mean()
        bot_y = anchors[skel._fascia_l_bot_idx, 1].mean()

        for ri in range(1, rows - 1):
            row_y = anchors[ri * cols:(ri + 1) * cols, 1].mean()
            assert min(top_y, bot_y) - 1e-5 <= row_y <= max(top_y, bot_y) + 1e-5, \
                f"Interior row {ri} y={row_y:.4f} not between top={top_y:.4f} and bot={bot_y:.4f}"

    def test_reset_pose_restores_fascia(self):
        """reset_pose() must restore fascia to its default positions."""
        skel = self._make_skel()
        default_l = skel.get_fascia_left_surface_anchors_np().copy()
        default_r = skel.get_fascia_right_surface_anchors_np().copy()

        skel.clavicle_left_pitch  = 0.4
        skel.clavicle_right_yaw   = 0.3
        skel.update()

        skel.reset_pose()
        restored_l = skel.get_fascia_left_surface_anchors_np()
        restored_r = skel.get_fascia_right_surface_anchors_np()

        np.testing.assert_allclose(default_l, restored_l, atol=1e-5,
                                   err_msg="Left fascia not restored after reset_pose")
        np.testing.assert_allclose(default_r, restored_r, atol=1e-5,
                                   err_msg="Right fascia not restored after reset_pose")

    def test_render_draws_include_ribcage_and_fascia(self):
        """get_render_draws() should return 5 callables (2 clavicles + ribcage + 2 fascia)."""
        skel = self._make_skel()
        draws = skel.get_render_draws()
        assert len(draws) == 5, f"Expected 5 render draws, got {len(draws)}"
        for d in draws:
            assert callable(d)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

