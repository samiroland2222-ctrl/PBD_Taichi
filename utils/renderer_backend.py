"""Backend-neutral draw-scene protocol for render-draw callbacks.

Both the Taichi ``_SceneProxy`` (utils/renderer.py) and the wgpu
``WgpuSceneProxy`` (utils/renderer_wgpu.py) implement the ``DrawScene``
protocol so that render-draw callbacks produced by geometry objects are
fully backend-agnostic.

Geometry objects should type-annotate their scene argument as
``DrawScene`` (or leave it un-typed) rather than ``ti.ui.Scene``.

Example
-------
::

    from utils.renderer_backend import DrawScene

    def my_draw(scene: DrawScene) -> None:
        scene.mesh(v_p, f_i, color=(0.7, 0.3, 0.1))
        scene.lines(verts, 2.0, idx, color=(1.0, 1.0, 1.0))
        scene.particles(pts, 0.02, color=(0.0, 0.7, 0.7))
"""

from __future__ import annotations

from typing import Tuple

# 3-component colour tuple, values in [0, 1].
Color3 = Tuple[float, float, float]


class DrawScene:
    """Minimal protocol understood by render-draw callbacks.

    Concrete implementations:
    * ``_SceneProxy`` in ``utils/renderer.py``          – Taichi UI backend
    * ``WgpuSceneProxy`` in ``utils/renderer_wgpu.py``  – pygfx/wgpu backend
    """

    def mesh(
        self,
        vertices,
        indices,
        *,
        color: Color3 = (0.5, 0.5, 0.5),
        show_wireframe: bool = False,
        two_sided: bool = False,
    ) -> None:
        """Draw a triangle mesh.

        Parameters
        ----------
        vertices:
            ``ti.MatrixField`` *or* ``np.ndarray`` of shape ``(N, 3)`` float32.
        indices:
            ``ti.Field`` *or* ``np.ndarray`` of shape ``(3*M,)`` int32 – flat
            triangle list.
        color:
            Uniform RGB colour in [0, 1].
        show_wireframe:
            If ``True``, render in wireframe mode.
        two_sided:
            If ``True``, light both front and back faces.
        """
        raise NotImplementedError

    def lines(
        self,
        vertices,
        width: float,
        indices,
        *,
        color: Color3 = (0.5, 0.5, 0.5),
    ) -> None:
        """Draw indexed line segments.

        Parameters
        ----------
        vertices:
            ``ti.MatrixField`` *or* ``np.ndarray`` of shape ``(N, 3)`` float32.
        width:
            Line width in screen pixels (backend-dependent semantics).
        indices:
            ``ti.Field`` *or* ``np.ndarray`` of shape ``(2*S,)`` int32 – flat
            pairs of vertex indices, one pair per segment.
        color:
            Uniform RGB colour in [0, 1].
        """
        raise NotImplementedError

    def particles(
        self,
        vertices,
        radius: float,
        per_vertex_color=None,
        *,
        color: Color3 = (0.0, 0.7, 0.7),
    ) -> None:
        """Draw point-sphere particles.

        Parameters
        ----------
        vertices:
            ``ti.MatrixField`` *or* ``np.ndarray`` of shape ``(N, 3)`` float32.
        radius:
            World-space sphere radius.
        per_vertex_color:
            Optional per-vertex colour buffer: ``ti.MatrixField`` or
            ``np.ndarray`` of shape ``(N, 3)`` float32. If *None*, the
            uniform ``color`` is used.
        color:
            Uniform RGB colour used when *per_vertex_color* is ``None``.
        """
        raise NotImplementedError

