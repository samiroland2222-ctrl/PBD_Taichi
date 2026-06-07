"""
Procedural areola and nipple texture painting.

Patches the areola and nipple pigmentation into an existing ``SkinTextures``
object that was produced by ``skin_texture_gen.generate_skin_textures()``.

Pipeline
--------
1. Find all atlas texels whose **world-space position** falls within the areola
   radius (in 3-D Euclidean distance, not UV-distance → no seam artefacts).
2. Build a radial soft mask (Gaussian falloff at the areola border).
3. Blend areola/nipple colour using the same Beer–Lambert spectral model as
   the rest of the skin but with elevated melanin concentration.
4. Boost roughness and add a radial normal-map dome for the areola mound.
5. Rebuild the affected GPU textures in-place.

Usage
-----
    from PBD_Taichi.utils.nipple_texture import paint_nipple_areola

    # After generating the skin textures …
    skin_tex = generate_skin_textures(verts, faces)

    # Patch in the nipple (world_pos_map comes from _rasterize_atlas internals;
    # for the high-level API we bake it via the helper below)
    paint_nipple_areola(
        skin_tex,
        world_pos_map  = world_pos_map,   # (H, W, 3) float32
        mask           = atlas_mask,       # (H, W) bool
        nipple_center  = apex_world,       # (3,) float32
        areola_radius  = 0.022,
        nipple_radius  = 0.007,
        melanin_amount = 0.08,             # base skin melanin (from generate_skin_textures call)
        haemo_amount   = 0.55,
        oxygenation    = 1.0,
    )

COLOR-SPACE CONTRACT (matches skin_texture_gen.py)
───────────────────────────────────────────────────
All intermediate arrays are LINEAR float [0,1].
`albedo_srgb_u8` is sRGB-encoded uint8 → uploaded with colorspace="srgb".
`normal_lin_u8` / `roughness_lin_u8` are LINEAR uint8 → colorspace="physical".
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from PBD_Taichi.utils.skin_texture_gen import SkinTextures, spectral_skin_base_color
from PBD_Taichi.utils.tex_utils import (
    lin_f_to_lin_u8, lin_f_to_srgb_u8, np_to_texture,
)


# ──────────────────────────────────────────────────────────────────────────────
# Public constants (tuneable)
# ──────────────────────────────────────────────────────────────────────────────

# Melanin boost factor for areola / nipple relative to the surrounding skin.
# Tuned for fair/light skin (melanin_amount ≈ 0.03–0.05): keeps the areola a
# soft dusty-rose rather than medium-brown, and the nipple a deeper rose-pink.
_AREOLA_MELANIN_BOOST = 1.8   # areola: ~1.8× darker in melanin (subtle on fair skin)
_NIPPLE_MELANIN_BOOST = 2.8   # nipple centre: ~2.8× darker

# Haemoglobin boost (areola is notably more pink/flushed than surrounding skin)
_AREOLA_HAEMO_BOOST   = 1.8   # raised from 1.3 → warmer pink on light skin

# Roughness values for areola / nipple regions (overrides skin roughness)
_AREOLA_ROUGHNESS     = 0.72  # slightly rougher than surrounding skin (Montgomery glands)
_NIPPLE_ROUGHNESS     = 0.80  # keratinised tip — noticeably rougher

# Normal-map dome amplitude: adds a subtle mound shape to the areola
_AREOLA_DOME_STRENGTH = 0.18  # 0 = flat, 1 = very pronounced mound

# Gaussian edge falloff: sigma as a fraction of the areola radius (in 3-D metres)
_AREOLA_EDGE_SIGMA_FRAC = 0.30
_NIPPLE_EDGE_SIGMA_FRAC = 0.25


# ──────────────────────────────────────────────────────────────────────────────
# Main entry point
# ──────────────────────────────────────────────────────────────────────────────

def paint_nipple_areola(
    skin_tex:       SkinTextures,
    world_pos_map:  np.ndarray,          # (H, W, 3) float32 — baked world positions
    mask:           np.ndarray,          # (H, W) bool — atlas coverage mask
    nipple_center:  np.ndarray,          # (3,) float32 — apex world position
    areola_radius:  float = 0.022,       # metres
    nipple_radius:  float = 0.007,       # metres
    # Skin model parameters (must match the values used in generate_skin_textures)
    melanin_amount: float = 0.08,
    haemo_amount:   float = 0.55,
    oxygenation:    float = 1.0,
    # Overrideable tint multipliers
    areola_melanin_boost: float = _AREOLA_MELANIN_BOOST,
    nipple_melanin_boost: float = _NIPPLE_MELANIN_BOOST,
    areola_haemo_boost:   float = _AREOLA_HAEMO_BOOST,
    areola_roughness:     float = _AREOLA_ROUGHNESS,
    nipple_roughness:     float = _NIPPLE_ROUGHNESS,
    areola_dome_strength: float = _AREOLA_DOME_STRENGTH,
    rebuild_gpu_textures: bool  = True,
    verbose: bool = False,
) -> None:
    """Paint areola and nipple pigmentation into *skin_tex* in-place.

    Mutates the following ``SkinTextures`` fields:
      - ``albedo_srgb_u8``    (and ``albedo_tex`` if rebuild_gpu_textures)
      - ``roughness_lin_u8``  (and ``roughness_tex``)
      - ``normal_lin_u8``     (and ``normal_tex``)

    Parameters
    ----------
    skin_tex      : SkinTextures returned by generate_skin_textures().
    world_pos_map : (H, W, 3) float32 — world-space position per atlas texel.
                    Obtain this from _rasterize_atlas() or bake_world_pos().
    mask          : (H, W) bool — True where the atlas has valid geometry.
    nipple_center : (3,) float32 — world position of the nipple/areola centre
                    (use find_apex_frame() from nipple.py).
    areola_radius : Outer radius of the pigmented areola disc (metres).
    nipple_radius : Radius of the nipple base cylinder (metres).
    melanin_amount: Background skin melanin (must match skin_texture_gen call).
    haemo_amount  : Background skin haemoglobin (must match skin_texture_gen call).
    oxygenation   : Background oxygenation (must match skin_texture_gen call).
    rebuild_gpu_textures : If True, uploads the mutated maps to pygfx Textures.
    """
    H, W = world_pos_map.shape[:2]
    center = np.asarray(nipple_center, dtype=np.float32)

    # ── 1. Compute world-space radial distance for every atlas texel ─────────
    rows, cols = np.where(mask)
    if len(rows) == 0:
        return

    wp = world_pos_map[rows, cols]   # (N, 3)
    dist = np.linalg.norm(wp - center[np.newaxis, :], axis=1)  # (N,) metres

    if verbose:
        print(f"  [nipple_tex] areola dist range: {dist.min():.4f}–{dist.max():.4f} m "
              f"(target areola r={areola_radius:.4f}, nipple r={nipple_radius:.4f})")

    # ── 2. Soft radial masks ─────────────────────────────────────────────────
    areola_sigma = areola_radius * _AREOLA_EDGE_SIGMA_FRAC
    nipple_sigma  = nipple_radius  * _NIPPLE_EDGE_SIGMA_FRAC

    # Areola mask: 1 inside, Gaussian falloff outside
    areola_mask_flat = _radial_soft_mask(dist, areola_radius, areola_sigma)
    # Nipple mask: 1 inside nipple_radius, Gaussian falloff
    nipple_mask_flat  = _radial_soft_mask(dist, nipple_radius,  nipple_sigma)

    # ── 3. Compute target colours ─────────────────────────────────────────────
    # Areola colour (elevated melanin)
    areola_color = spectral_skin_base_color(
        melanin_amount=np.clip(melanin_amount * areola_melanin_boost, 0.0, 1.0),
        haemo_amount  =np.clip(haemo_amount   * areola_haemo_boost,   0.0, 1.0),
        oxygenation   =oxygenation,
    )  # (3,) linear RGB

    # Nipple colour (higher melanin still)
    nipple_color = spectral_skin_base_color(
        melanin_amount=np.clip(melanin_amount * nipple_melanin_boost, 0.0, 1.0),
        haemo_amount  =np.clip(haemo_amount   * areola_haemo_boost,   0.0, 1.0),
        oxygenation   =oxygenation,
    )  # (3,) linear RGB

    # ── 4. Decode current albedo LINEAR float ────────────────────────────────
    # albedo_srgb_u8 is sRGB-encoded; decode to linear float for blending
    albedo_srgb_u8 = skin_tex.albedo_srgb_u8.astype(np.float32) / 255.0
    albedo_lin_f   = _srgb_to_lin(albedo_srgb_u8[..., :3])  # (H, W, 3) linear

    # ── 5. Blend areola then nipple colour into albedo ───────────────────────
    # Areola pass: blend over the existing skin colour
    albedo_lin_f = _apply_color_patch(
        albedo_lin_f, rows, cols, areola_mask_flat, areola_color)
    # Nipple pass: blend the deeper nipple colour on top
    albedo_lin_f = _apply_color_patch(
        albedo_lin_f, rows, cols, nipple_mask_flat, nipple_color)

    # Re-encode to sRGB uint8 + alpha
    alpha_ch = skin_tex.albedo_srgb_u8[:, :, 3:4]
    new_albedo_srgb_u8 = np.concatenate(
        [lin_f_to_srgb_u8(albedo_lin_f), alpha_ch], axis=2)
    skin_tex.albedo_srgb_u8[:] = new_albedo_srgb_u8

    # ── 6. Roughness patch ────────────────────────────────────────────────────
    rough_lin_u8 = skin_tex.roughness_lin_u8.copy()  # (H, W, 3)
    # roughness is in GREEN channel (G), normalised [0,255]
    rough_g = rough_lin_u8[:, :, 1].astype(np.float32) / 255.0  # (H, W) in [0,1]

    # Areola roughness
    rough_g = _blend_scalar_patch(rough_g, rows, cols, areola_mask_flat, areola_roughness)
    # Nipple roughness
    rough_g = _blend_scalar_patch(rough_g, rows, cols, nipple_mask_flat, nipple_roughness)

    np.clip(rough_g, 0.0, 1.0, out=rough_g)
    rough_lin_u8[:, :, 1] = (rough_g * 255.0).clip(0, 255).astype(np.uint8)
    skin_tex.roughness_lin_u8[:] = rough_lin_u8

    # ── 7. Normal map dome (areola mound) ────────────────────────────────────
    _add_areola_dome_normal(
        skin_tex, world_pos_map, mask, rows, cols,
        center, areola_radius, areola_mask_flat,
        strength=areola_dome_strength,
    )

    # ── 8. Rebuild GPU textures ───────────────────────────────────────────────
    if rebuild_gpu_textures:
        skin_tex.albedo_tex    = np_to_texture(skin_tex.albedo_srgb_u8,   colorspace="srgb")
        skin_tex.roughness_tex = np_to_texture(skin_tex.roughness_lin_u8, colorspace="physical")
        skin_tex.normal_tex    = np_to_texture(skin_tex.normal_lin_u8,    colorspace="physical")

    if verbose:
        print(f"  [nipple_tex] done — areola α_max={areola_mask_flat.max():.3f}, "
              f"nipple α_max={nipple_mask_flat.max():.3f}")


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _radial_soft_mask(
    dist:   np.ndarray,   # (N,) — distance to centre (metres)
    radius: float,        # hard edge radius
    sigma:  float,        # Gaussian falloff width outside the hard edge
) -> np.ndarray:
    """Soft radial mask in [0, 1].

    = 1.0  for dist ≤ radius
    = Gaussian falloff for dist > radius, zero at dist = radius + 3*sigma
    """
    inside  = dist <= radius
    outside = ~inside
    alpha   = np.where(inside, 1.0, 0.0).astype(np.float32)
    if sigma > 1e-9:
        excess = np.maximum(dist[outside] - radius, 0.0)
        alpha[outside] = np.exp(-0.5 * (excess / sigma) ** 2).astype(np.float32)
    return alpha


def _srgb_to_lin(srgb: np.ndarray) -> np.ndarray:
    """Inverse sRGB gamma: sRGB float [0,1] → linear float [0,1]."""
    lin = np.where(srgb <= 0.04045,
                   srgb / 12.92,
                   ((srgb + 0.055) / 1.055) ** 2.4)
    return lin.astype(np.float32)


def _apply_color_patch(
    albedo_lin_f: np.ndarray,      # (H, W, 3) float32 — current albedo (linear)
    rows: np.ndarray,               # masked pixel row indices
    cols: np.ndarray,               # masked pixel col indices
    mask_flat: np.ndarray,          # (N,) float32 — alpha per masked pixel
    color: np.ndarray,              # (3,) float32 — target linear RGB
) -> np.ndarray:
    """Alpha-blend *color* over albedo at the masked pixels.

    Returns a new (H, W, 3) array (the input is not modified).
    """
    out = albedo_lin_f.copy()
    α   = mask_flat[:, np.newaxis]                    # (N, 1)
    out[rows, cols] = (1.0 - α) * out[rows, cols] + α * color[np.newaxis, :]
    return out


def _blend_scalar_patch(
    arr_2d:    np.ndarray,   # (H, W) float32
    rows:      np.ndarray,
    cols:      np.ndarray,
    mask_flat: np.ndarray,   # (N,) float32
    target:    float,
) -> np.ndarray:
    """Alpha-blend scalar *target* into arr_2d at masked pixels."""
    out = arr_2d.copy()
    out[rows, cols] = (1.0 - mask_flat) * out[rows, cols] + mask_flat * target
    return out


def _add_areola_dome_normal(
    skin_tex:       SkinTextures,
    world_pos_map:  np.ndarray,      # (H, W, 3)
    mask:           np.ndarray,      # (H, W) bool
    rows:           np.ndarray,
    cols:           np.ndarray,
    center:         np.ndarray,      # (3,) float32 — nipple apex world pos
    areola_radius:  float,
    areola_mask_flat: np.ndarray,    # (N,) float32
    strength:       float,
) -> None:
    """Add a gentle dome-shaped bump to the normal map over the areola region.

    Computes a radial gradient (pointing outward from the areola centre) and
    blends it into the existing tangent-space normal map using UDN compositing.
    This gives the areola a subtle raised-mound appearance.

    Mutates ``skin_tex.normal_lin_u8`` in-place.
    """
    if strength <= 0.0:
        return

    H, W = world_pos_map.shape[:2]

    # Current normal map: (H, W, 4) linear uint8 → decode to float [-1,1]
    normal_f = skin_tex.normal_lin_u8[:, :, :3].astype(np.float32) / 255.0  # [0,1]
    nx = normal_f[:, :, 0] * 2.0 - 1.0
    ny = normal_f[:, :, 1] * 2.0 - 1.0
    nz = normal_f[:, :, 2] * 2.0 - 1.0

    # Compute radial displacement vector at each masked pixel (projected onto
    # the image plane of the normal map — approximated as 2-D outward gradient).
    wp   = world_pos_map[rows, cols]                          # (N, 3)
    dr   = wp - center[np.newaxis, :]                         # (N, 3) radial
    dist = np.linalg.norm(dr, axis=1, keepdims=True) + 1e-9
    dr_n = (dr / dist)                                        # (N, 3) unit radial

    # We need the radial vector projected onto the local tangent frame.
    # Simplified: use the XZ components (tangent plane, ignoring local frame
    # rotations) scaled by the dome slope.  The dome height ≈ strength * r_frac.
    r_frac = np.clip((dist[:, 0] / areola_radius), 0.0, 1.0)  # (N,) normalised radius
    # Dome profile: maximum slope at ~0.7 of the radius, zero at centre and edge
    dome_slope = strength * 4.0 * r_frac * (1.0 - r_frac)    # (N,)

    # Perturb tangent XY toward outward radial direction, scaled by dome slope
    # (dr_n[:, 0] ~ t̂ component, dr_n[:, 2] ~ b̂ component — approximate)
    perturb_x = dr_n[:, 0] * dome_slope  # along tangent
    perturb_y = dr_n[:, 2] * dome_slope  # along bitangent

    # Alpha blend the dome perturbation, weighted by areola mask
    α = areola_mask_flat  # (N,)

    # UDN blend: add perturbation to existing XY, keep Z from base
    base_nx_flat = nx[rows, cols]
    base_ny_flat = ny[rows, cols]
    base_nz_flat = nz[rows, cols]

    blended_nx = base_nx_flat + α * perturb_x
    blended_ny = base_ny_flat + α * perturb_y
    blended_nz = base_nz_flat

    n_len = np.sqrt(blended_nx**2 + blended_ny**2 + blended_nz**2) + 1e-8
    blended_nx /= n_len
    blended_ny /= n_len
    blended_nz /= n_len

    nx[rows, cols] = blended_nx
    ny[rows, cols] = blended_ny
    nz[rows, cols] = blended_nz

    # Re-encode to [0,1] → uint8 (linear)
    enc = np.stack([
        np.clip(nx * 0.5 + 0.5, 0.0, 1.0),
        np.clip(ny * 0.5 + 0.5, 0.0, 1.0),
        np.clip(nz * 0.5 + 0.5, 0.0, 1.0),
    ], axis=2).astype(np.float32)

    new_normal = skin_tex.normal_lin_u8.copy()
    new_normal[:, :, :3] = lin_f_to_lin_u8(enc)
    # Reset pixels outside mask to flat normal (0.5, 0.5, 1.0) → (128, 128, 255)
    not_mask = ~mask
    new_normal[not_mask, 0] = 128
    new_normal[not_mask, 1] = 128
    new_normal[not_mask, 2] = 255
    skin_tex.normal_lin_u8[:] = new_normal


# ──────────────────────────────────────────────────────────────────────────────
# Convenience wrapper: bake world-pos atlas from mesh + SkinTextures UVs
# ──────────────────────────────────────────────────────────────────────────────

def bake_world_pos_map(
    verts: np.ndarray,         # (V_orig, 3) float32  — original mesh vertices
    skin_tex: SkinTextures,
) -> tuple[np.ndarray, np.ndarray]:
    """Rasterize world-space position into the atlas.

    This is a stripped-down version of ``_rasterize_atlas`` from
    ``skin_texture_gen.py`` that only bakes position (no normals/tangents).
    Useful when you want to call ``paint_nipple_areola`` after the fact
    without storing the full rasterizer output.

    Returns
    -------
    world_pos_map : (H, W, 3) float32
    mask          : (H, W) bool
    """
    from PBD_Taichi.utils.skin_texture_gen import _rasterize_atlas, _vertex_normals

    uvs      = skin_tex.uvs        # (V_uv, 2)
    indices  = skin_tex.indices    # (F, 3) indexed into V_uv space
    vmapping = skin_tex.vmapping   # (V_uv,) → original vertex index

    v = np.asarray(verts, dtype=np.float32)

    # Recover original-vertex-indexed faces via vmapping so that vertex normals
    # are computed in original-vertex space (vmapping may have split seam verts).
    orig_faces = vmapping[indices]  # (F, 3) — original vertex indices
    vn = _vertex_normals(v, orig_faces)  # (V_orig, 3)

    # Infer atlas resolution from the stored albedo texture shape.
    H, W = skin_tex.albedo_srgb_u8.shape[:2]

    world_pos, _, _, _, mask = _rasterize_atlas(uvs, indices, v, vn, vmapping, H, W)
    return world_pos, mask



