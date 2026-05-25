"""Shared 3-D geometry utilities.

Functions here are intentionally stateless and dependency-light so that they
can be reused across different pipeline modules without circular imports.
"""

import numpy as np


def vertex_normals_trimesh(verts: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Per-vertex outward normals for a triangle mesh.

    Each vertex normal is the area-weighted average of the normals of all
    triangles that share that vertex, then unit-normalised.

    Parameters
    ----------
    verts : (N, 3) float array
    faces : (M, 3) int array, 0-indexed

    Returns
    -------
    normals : (N, 3) float array, unit length
    """
    n = np.zeros_like(verts)
    a = verts[faces[:, 0]]
    b = verts[faces[:, 1]]
    c = verts[faces[:, 2]]
    fn = np.cross(b - a, c - a)        # 2×area-weighted face normals
    np.add.at(n, faces[:, 0], fn)
    np.add.at(n, faces[:, 1], fn)
    np.add.at(n, faces[:, 2], fn)
    norms = np.linalg.norm(n, axis=1, keepdims=True)
    return np.where(norms > 0, n / norms, 0.0)

