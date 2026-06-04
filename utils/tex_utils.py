"""Lightweight numpy ↔ pygfx texture helpers.

COLOR-SPACE CONTRACT
====================
pygfx reads every texture in the shader and applies an sRGB→linear decode
**unless** the texture was uploaded with ``colorspace="physical"``.

This means the caller is responsible for choosing the right encode:

  COLOR / EMISSIVE maps  (albedo, SSS glow)
  ────────────────────────────────────────────
  · Data lives in LINEAR float space inside Python.
  · Must be gamma-encoded before upload: call ``lin_f_to_srgb_u8()``.
  · Upload with the default ``colorspace="srgb"`` (or omit the kwarg).
  · Shader decode then cancels the encode → shader sees original linear values.

  DATA / NON-COLOR maps  (normals, roughness, AO, height, thickness)
  ────────────────────────────────────────────
  · Values are not colors — gamma encoding would corrupt them.
  · Store as raw linear: call ``lin_f_to_lin_u8()``.
  · Upload with ``colorspace="physical"`` to skip the sRGB decode.
  · Shader sees the stored values unchanged.

APPS HUNGARIAN NAMING USED IN THIS CODEBASE
============================================
  lin_*   — float [0,1] array in LINEAR light space (physical units)
  srgb_*  — array in sRGB-encoded space (perceptual, gamma-curved)
  *_f     — float32 dtype
  *_u8    — uint8 dtype [0, 255]

  Combined:
    lin_f         →  linear float [0,1]        (intermediate computation)
    srgb_u8       →  sRGB uint8 [0,255]        → upload colorspace="srgb"
    lin_u8        →  linear uint8 [0,255]      → upload colorspace="physical"

PYGFX MATERIAL COLOR PROPERTIES
================================
pygfx ``MeshStandardMaterial`` / ``MeshPhysicalMaterial`` color uniforms
(``color``, ``emissive``, ``sheen_color``, light ``color``, …) accept
linear RGB tuples — they are **not** sRGB-encoded.  Always specify
material color tuples in LINEAR space.
"""
from __future__ import annotations

import numpy as np
import pygfx


def linear_to_srgb(lin_f: np.ndarray) -> np.ndarray:
    """Convert linear light values [0,1] to sRGB-encoded values [0,1].

    The IEC 61966-2-1 piecewise transfer function (identical to the
    "power-2.2" approximation for values > 0.003).

    Parameters
    ----------
    lin_f : array_like, float32
        Linear light values in [0, 1].

    Returns
    -------
    np.ndarray
        sRGB-encoded floats in [0, 1].  Pass to ``lin_f_to_lin_u8`` if you
        need uint8 (which bypasses the second gamma step); or pass directly
        to ``lin_f_to_srgb_u8`` which does both in one call.
    """
    x = np.clip(lin_f, 0.0, 1.0).astype(np.float32)
    return np.where(
        x <= 0.0031308,
        x * 12.92,
        1.055 * np.power(x, 1.0 / 2.4) - 0.055,
    ).astype(np.float32)


def lin_f_to_lin_u8(lin_f: np.ndarray) -> np.ndarray:
    """Map LINEAR float [0,1] → LINEAR uint8 [0,255] (no gamma encode).

    Use for NON-COLOR data (normals, roughness, AO, height, thickness).
    Upload the result with ``np_to_texture(arr, colorspace="physical")``
    so the shader receives the values unchanged.

    Formerly named ``float01_to_u8``.
    """
    return (np.clip(lin_f, 0.0, 1.0) * 255.0).astype(np.uint8)


# Legacy alias — kept so old call-sites that haven't been updated yet still
# compile.  New code should use the descriptive name ``lin_f_to_lin_u8``.
float01_to_u8 = lin_f_to_lin_u8


def lin_f_to_srgb_u8(lin_f: np.ndarray) -> np.ndarray:
    """Convert LINEAR float [0,1] → sRGB-encoded uint8 [0,255].

    Use for COLOR / EMISSIVE data that will be uploaded via
    ``np_to_texture(arr, colorspace="srgb")`` (the default).

    The sRGB encode here cancels the sRGB decode applied by the pygfx shader,
    so the shader ultimately sees the original linear values.

    Formerly named ``linear_to_srgb_u8``.
    """
    return lin_f_to_lin_u8(linear_to_srgb(lin_f))


# Legacy alias — kept for backward compatibility.
linear_to_srgb_u8 = lin_f_to_srgb_u8


def np_to_texture(
    arr: np.ndarray,
    colorspace: str = "srgb",
) -> pygfx.Texture:
    """Upload a numpy image array as a pygfx Texture.

    Parameters
    ----------
    arr : uint8 ndarray of shape (H, W), (H, W, 1), (H, W, 3), or (H, W, 4)
        Image data in [0, 255].  Must already be encoded for the chosen
        colorspace (see module docstring).
    colorspace : str
        ``"srgb"``      — colour / emissive maps.  pygfx decodes sRGB→linear
                          in the shader.  Pass ``srgb_u8`` arrays here (made
                          with ``lin_f_to_srgb_u8``).
        ``"physical"``  — non-colour data (normals, roughness, AO, etc.).
                          Skips the sRGB decode; shader sees raw stored values.
                          Pass ``lin_u8`` arrays here (made with
                          ``lin_f_to_lin_u8``).

    Returns
    -------
    pygfx.Texture
        Ready to assign to a material map property.
    """
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    arr = np.ascontiguousarray(arr)
    return pygfx.Texture(arr, dim=2, colorspace=colorspace)

