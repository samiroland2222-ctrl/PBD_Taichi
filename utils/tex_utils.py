"""Lightweight numpy ↔ pygfx texture helpers."""
from __future__ import annotations

import numpy as np
import pygfx


def np_to_texture(
    arr: np.ndarray,
    srgb: bool = False,
) -> pygfx.Texture:
    """Upload a numpy image array as a pygfx Texture.

    Parameters
    ----------
    arr : (H, W), (H, W, 1), (H, W, 3) or (H, W, 4) uint8
        Image data.  Values must already be in [0, 255].
    srgb : bool
        Unused currently; reserved for future format selection.

    Returns
    -------
    pygfx.Texture
        Texture object ready to be assigned to a material property.
    """
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    arr = np.ascontiguousarray(arr)
    return pygfx.Texture(arr, dim=2)


def float01_to_u8(arr: np.ndarray) -> np.ndarray:
    """Map float array in [0, 1] → uint8 [0, 255], clipped."""
    return (np.clip(arr, 0.0, 1.0) * 255.0).astype(np.uint8)

