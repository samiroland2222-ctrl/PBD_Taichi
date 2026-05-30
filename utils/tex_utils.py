"""Lightweight numpy ↔ pygfx texture helpers."""
from __future__ import annotations

import numpy as np
import pygfx


def linear_to_srgb(x: np.ndarray) -> np.ndarray:
    """Convert linear light values [0,1] to sRGB-encoded values [0,1].

    pygfx (colorspace="srgb", the default) decodes textures from sRGB to
    linear in the shader.  Colour and emissive textures must therefore be
    stored as sRGB-encoded bytes so that the decode brings them back to the
    intended linear values.

    Non-colour data (normals, roughness, AO) must be uploaded with
    colorspace="physical" via np_to_texture(..., colorspace="physical") to
    skip the decode entirely.
    """
    x = np.clip(x, 0.0, 1.0).astype(np.float32)
    return np.where(
        x <= 0.0031308,
        x * 12.92,
        1.055 * np.power(x, 1.0 / 2.4) - 0.055,
    ).astype(np.float32)


def np_to_texture(
    arr: np.ndarray,
    colorspace: str = "srgb",
) -> pygfx.Texture:
    """Upload a numpy image array as a pygfx Texture.

    Parameters
    ----------
    arr : (H, W), (H, W, 1), (H, W, 3) or (H, W, 4) uint8
        Image data.  Values must already be in [0, 255].
    colorspace : str
        ``"srgb"``     — colour / emissive maps (pygfx decodes sRGB→linear).
        ``"physical"`` — non-colour data (normals, roughness, AO, height);
                         skips the sRGB decode so values reach the shader
                         unchanged.

    Returns
    -------
    pygfx.Texture
        Texture object ready to be assigned to a material property.
    """
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    arr = np.ascontiguousarray(arr)
    return pygfx.Texture(arr, dim=2, colorspace=colorspace)


def float01_to_u8(arr: np.ndarray) -> np.ndarray:
    """Map float array in [0, 1] → uint8 [0, 255], clipped (no gamma)."""
    return (np.clip(arr, 0.0, 1.0) * 255.0).astype(np.uint8)


def linear_to_srgb_u8(arr: np.ndarray) -> np.ndarray:
    """Convert linear [0,1] float → sRGB-encoded uint8 [0,255].

    Use this for colour and emissive data that will be uploaded via
    np_to_texture() with the default colorspace="srgb".  The sRGB encode
    here cancels the sRGB decode in the shader, so the shader sees the
    original linear values.
    """
    return float01_to_u8(linear_to_srgb(arr))


