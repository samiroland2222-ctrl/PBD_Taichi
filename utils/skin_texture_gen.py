"""Procedural PBR skin texture generation for auto-generated skin meshes.

Pipeline
--------
1.  UV atlas via xatlas (call once per mesh topology; ~1–3 s).
2.  Rasterize: bake per-texel world-space position, surface normal, and
    tangent frame from the mesh geometry.
3.  Procedural layers (world/tangent-space → no UV-seam artefacts on
    fine detail; UV-space gaussian fields for large-scale tone):
      Height    Voronoi F1 pore dimples + fBm micro-texture
      Normal    Sobel gradient of height → tangent-space RGB encoding
      Albedo    base tone · Perlin variation · freckles · pore tint
      Roughness base (dewy young skin) + pore boost − dewy-patch reduction
      AO        horizon-based micro-AO baked from height map
      Emissive  warm SSS glow modulated by noise (+ optional thickness)
4.  Upload all maps as pygfx textures and create a pre-wired
    MeshStandardMaterial.

Usage
-----
    from PBD_Taichi.utils.skin_texture_gen import generate_skin_textures

    skin_tex = generate_skin_textures(verts_np, faces_np, resolution=2048)
    # skin_tex.material   – ready-made pygfx.MeshStandardMaterial
    # skin_tex.uvs        – (V_new, 2) float32 UV coords
    # skin_tex.indices    – (F, 3) int32 remapped face indices
    # skin_tex.vmapping   – (V_new,) int32  new→old vertex map
"""
from __future__ import annotations

import dataclasses
import math
import time
from typing import Optional

import numpy as np
import pygfx
from scipy.ndimage import gaussian_filter, maximum_filter

from PBD_Taichi.utils.tex_utils import float01_to_u8, np_to_texture


# ══════════════════════════════════════════════════════════════════════════════
# Public result type
# ══════════════════════════════════════════════════════════════════════════════

@dataclasses.dataclass
class SkinTextures:
    """All outputs of the procedural skin generator."""

    # ── UV parameterization (renderer uses these every frame) ──────────────
    uvs:      np.ndarray   # (V_new, 2) float32 — UV coords per UV-vertex
    indices:  np.ndarray   # (F, 3) int32        — face indices into V_new
    vmapping: np.ndarray   # (V_new,) int32      — UV-vertex → original vertex

    # ── Raw numpy maps (uint8; also usable for save / debug) ──────────────
    albedo_np:    np.ndarray   # (H, W, 4) RGBA
    normal_np:    np.ndarray   # (H, W, 4) RGBA
    roughness_np: np.ndarray   # (H, W, 1)
    ao_np:        np.ndarray   # (H, W, 1)
    height_np:    np.ndarray   # (H, W, 1)
    emissive_np:  np.ndarray   # (H, W, 4) RGBA

    # ── pygfx textures ─────────────────────────────────────────────────────
    albedo_tex:    pygfx.Texture
    normal_tex:    pygfx.Texture
    roughness_tex: pygfx.Texture
    ao_tex:        pygfx.Texture
    emissive_tex:  pygfx.Texture

    # ── pre-wired PBR material ─────────────────────────────────────────────
    material: pygfx.MeshStandardMaterial

    def save(self, directory: str, prefix: str = "skin") -> None:
        """Write all maps as PNG files into *directory*."""
        from PIL import Image
        import os
        os.makedirs(directory, exist_ok=True)

        def _save(arr, name):
            path = os.path.join(directory, f"{prefix}_{name}.png")
            if arr.shape[-1] == 1:
                Image.fromarray(arr[:, :, 0], mode="L").save(path)
            elif arr.shape[-1] == 3:
                Image.fromarray(arr[:, :, :3], mode="RGB").save(path)
            else:
                Image.fromarray(arr, mode="RGBA").save(path)
            print(f"  saved {path}")

        _save(self.albedo_np,    "albedo")
        _save(self.normal_np[:, :, :3],   "normal")
        _save(self.roughness_np, "roughness")
        _save(self.ao_np,        "ao")
        _save(self.height_np,    "height")
        _save(self.emissive_np,  "emissive")


# ══════════════════════════════════════════════════════════════════════════════
# Main entry point
# ══════════════════════════════════════════════════════════════════════════════

def generate_skin_textures(
    verts: np.ndarray,                       # (V, 3) float64
    faces: np.ndarray,                       # (F, 3) int32
    resolution: int = 2048,
    seed: int = 42,
    # ── appearance knobs ─────────────────────────────────────────────────
    pore_cell_size: float = 0.0009,          # pore spacing in mesh units (~0.9 mm)
    freckle_density: float = 0.5,            # 0 = none, 1 = many
    base_roughness: float = 0.28,            # young/dewy skin baseline
    dewy_intensity: float = 0.12,            # roughness reduction in sebum patches
    emissive_intensity: float = 0.10,        # SSS glow multiplier in material
    normal_strength: float = 0.65,           # height→normal bump amplitude
    # ── optional inputs ──────────────────────────────────────────────────
    thickness_per_vertex: Optional[np.ndarray] = None,  # (V,) — shell thickness
    sun_dir: Optional[np.ndarray] = None,               # (3,) — sun direction
    verbose: bool = True,
) -> SkinTextures:
    """Generate a full PBR skin texture set for an arbitrary triangle mesh.

    Returns a :class:`SkinTextures` with pre-built pygfx textures and a
    ready-to-use :class:`pygfx.MeshStandardMaterial`.
    """
    rng = np.random.default_rng(seed)
    H = W = resolution

    if sun_dir is None:
        sun_dir = np.array([0.1, 1.0, 0.3], dtype=np.float64)
    sun_dir = sun_dir / (np.linalg.norm(sun_dir) + 1e-12)

    # ── 1. UV atlas ──────────────────────────────────────────────────────────
    t0 = time.time()
    vmapping, uv_indices, uvs = _generate_uv_atlas(verts, faces, resolution)
    if verbose:
        print(f"  [skin_tex] UV atlas  {time.time()-t0:.1f}s  "
              f"({len(vmapping)} UV-verts, {len(uv_indices)} faces)")

    # ── 2. Rasterize ─────────────────────────────────────────────────────────
    t0 = time.time()
    vnormals = _vertex_normals(verts, faces)
    world_pos, world_nrm, tangent_map, bitan_map, mask = _rasterize_atlas(
        uvs, uv_indices, verts, vnormals, vmapping, H, W,
    )
    if verbose:
        coverage = mask.sum() / (H * W) * 100
        print(f"  [skin_tex] rasterize {time.time()-t0:.1f}s  "
              f"(coverage {coverage:.1f}%)")

    # convenience: flat views of masked pixels
    mp = mask  # alias

    # ── 3. Height map ────────────────────────────────────────────────────────
    t0 = time.time()
    height_f = _gen_height(world_pos, tangent_map, bitan_map, mp,
                           pore_cell_size, H, W, rng)
    if verbose:
        print(f"  [skin_tex] height    {time.time()-t0:.1f}s")

    # ── 4. Normal map ────────────────────────────────────────────────────────
    normal_f  = _gen_normal_from_height(height_f, mp, normal_strength)

    # ── 5. Albedo ────────────────────────────────────────────────────────────
    t0 = time.time()
    albedo_f  = _gen_albedo(world_pos, world_nrm, tangent_map, bitan_map,
                             mp, pore_cell_size, height_f,
                             freckle_density, sun_dir, H, W, rng)
    if verbose:
        print(f"  [skin_tex] albedo    {time.time()-t0:.1f}s")

    # ── 6. Roughness ─────────────────────────────────────────────────────────
    rough_f   = _gen_roughness(height_f, mp, base_roughness, dewy_intensity,
                               H, W, rng)

    # ── 7. AO ────────────────────────────────────────────────────────────────
    ao_f      = _gen_ao(height_f, mp)

    # ── 8. Emissive (SSS approximation) ──────────────────────────────────────
    emissive_f = _gen_emissive(world_pos, mp, thickness_per_vertex,
                                vmapping, H, W, rng)

    # ── 9. Dilate all maps into unmapped atlas space ──────────────────────────
    # xatlas typically leaves 40-60 % of the atlas unpopulated.  Without
    # dilation, bilinear sampling at UV-chart edges bleeds in black pixels
    # and the mesh looks splotchy.  Dilate every map outward so the entire
    # atlas is filled with plausible values.
    if verbose and not mp.all():
        cov = mp.sum() / mp.size * 100
        print(f"  [skin_tex] dilating (coverage {cov:.1f}% → 100%) …")

    albedo_f   = _dilate_texture(albedo_f,   mp)
    normal_f   = _dilate_texture(normal_f,   mp)
    rough_f    = _dilate_texture(rough_f[:, :, np.newaxis], mp)[:, :, 0]
    ao_f       = _dilate_texture(ao_f[:, :, np.newaxis],    mp)[:, :, 0]
    height_f   = _dilate_texture(height_f[:, :, np.newaxis], mp)[:, :, 0]
    emissive_f = _dilate_texture(emissive_f, mp)

    # ── 10. Encode to uint8 ───────────────────────────────────────────────────
    def rgba_u8(rgb_f: np.ndarray, alpha: float = 1.0) -> np.ndarray:
        H_, W_, C = rgb_f.shape
        alpha_ch = np.full((H_, W_, 1), alpha * 255, dtype=np.uint8)
        return np.concatenate([float01_to_u8(rgb_f), alpha_ch], axis=2)

    albedo_np    = rgba_u8(albedo_f)
    normal_np    = rgba_u8(normal_f)
    roughness_np = float01_to_u8(rough_f[..., np.newaxis])
    ao_np        = float01_to_u8(ao_f[..., np.newaxis])
    height_np    = float01_to_u8(height_f[..., np.newaxis])
    emissive_np  = rgba_u8(emissive_f)

    # ── 11. Upload textures ───────────────────────────────────────────────────
    albedo_tex    = np_to_texture(albedo_np)
    normal_tex    = np_to_texture(normal_np)
    roughness_tex = np_to_texture(roughness_np)
    ao_tex        = np_to_texture(ao_np)
    emissive_tex  = np_to_texture(emissive_np)

    # ── 12. Build material ────────────────────────────────────────────────────
    material = pygfx.MeshStandardMaterial(
        # Albedo
        map=albedo_tex,
        color=(1.0, 1.0, 1.0),
        # Geometry / surface
        normal_map=normal_tex,
        normal_scale=(normal_strength * 0.5, normal_strength * 0.5),
        roughness=1.0,           # map provides per-texel value
        roughness_map=roughness_tex,
        metalness=0.0,           # skin is dielectric
        # AO
        ao_map=ao_tex,
        # SSS approximation via emissive
        emissive=(1.0, 1.0, 1.0),
        emissive_map=emissive_tex,
        emissive_intensity=emissive_intensity,
        side="both",
    )

    if verbose:
        print("  [skin_tex] done.")

    return SkinTextures(
        uvs=uvs, indices=uv_indices, vmapping=vmapping,
        albedo_np=albedo_np, normal_np=normal_np,
        roughness_np=roughness_np, ao_np=ao_np, height_np=height_np,
        emissive_np=emissive_np,
        albedo_tex=albedo_tex, normal_tex=normal_tex,
        roughness_tex=roughness_tex, ao_tex=ao_tex,
        emissive_tex=emissive_tex,
        material=material,
    )


# ══════════════════════════════════════════════════════════════════════════════
# UV Atlas
# ══════════════════════════════════════════════════════════════════════════════

def _generate_uv_atlas(
    verts: np.ndarray,
    faces: np.ndarray,
    resolution: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (vmapping, new_indices, uvs) from xatlas."""
    import xatlas

    atlas = xatlas.Atlas()
    atlas.add_mesh(
        verts.astype(np.float32),
        faces.astype(np.uint32),
    )

    chart_opts = xatlas.ChartOptions()
    chart_opts.max_cost = 8.0          # high tolerance → fewer charts → better coverage

    pack_opts = xatlas.PackOptions()
    pack_opts.resolution = resolution
    pack_opts.padding    = 2           # minimal padding → more usable space
    pack_opts.bilinear   = True

    atlas.generate(chart_options=chart_opts, pack_options=pack_opts)

    vmapping, new_indices, uvs = atlas[0]
    return (
        vmapping.astype(np.int32),
        new_indices.astype(np.int32),
        uvs.astype(np.float32),
    )


# ══════════════════════════════════════════════════════════════════════════════
# Rasterizer
# ══════════════════════════════════════════════════════════════════════════════

def _vertex_normals(verts: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Area-weighted per-vertex normals, unit-normalised."""
    nv = len(verts)
    nrm = np.zeros((nv, 3), dtype=np.float64)
    v0 = verts[faces[:, 0]]; v1 = verts[faces[:, 1]]; v2 = verts[faces[:, 2]]
    fn = np.cross(v1 - v0, v2 - v0)   # (F, 3) — magnitude = 2 × area
    np.add.at(nrm, faces[:, 0], fn)
    np.add.at(nrm, faces[:, 1], fn)
    np.add.at(nrm, faces[:, 2], fn)
    n = np.linalg.norm(nrm, axis=1, keepdims=True)
    return np.where(n > 1e-12, nrm / n, nrm).astype(np.float32)


def _rasterize_atlas(
    uvs_new:    np.ndarray,   # (V_new, 2) float32 UVs in [0,1]²
    idx_new:    np.ndarray,   # (F, 3) int32
    verts_orig: np.ndarray,   # (V, 3)
    vnormals:   np.ndarray,   # (V, 3)
    vmapping:   np.ndarray,   # (V_new,) int32
    H: int, W: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """For each atlas texel, bake world-space position, normal, and
    tangent frame.  Returns (world_pos, world_nrm, tangent, bitangent, mask).
    """
    # xatlas UV convention: (0,0) = top-left, same as wgpu.
    # Convert UV → pixel: col = u*(W-1),  row = v*(H-1)  (NO Y-flip).
    uvs_px = np.empty_like(uvs_new)
    uvs_px[:, 0] = uvs_new[:, 0] * (W - 1)
    uvs_px[:, 1] = uvs_new[:, 1] * (H - 1)   # top-left → no flip

    # Remap 3D data to UV-vertex order
    verts3d  = verts_orig[vmapping].astype(np.float32)
    normals3d = vnormals[vmapping].astype(np.float32)

    world_pos = np.zeros((H, W, 3), np.float32)
    world_nrm = np.zeros((H, W, 3), np.float32)
    tan_out   = np.zeros((H, W, 3), np.float32)
    btan_out  = np.zeros((H, W, 3), np.float32)
    mask      = np.zeros((H, W), bool)

    for fi in range(len(idx_new)):
        i0, i1, i2 = idx_new[fi]

        p0 = uvs_px[i0]; p1 = uvs_px[i1]; p2 = uvs_px[i2]
        v0 = verts3d[i0]; v1 = verts3d[i1]; v2 = verts3d[i2]
        n0 = normals3d[i0]; n1 = normals3d[i1]; n2 = normals3d[i2]

        # ── Tangent frame from UV mapping ──────────────────────────────────
        dp1x = float(p1[0] - p0[0]); dp1y = float(p1[1] - p0[1])
        dp2x = float(p2[0] - p0[0]); dp2y = float(p2[1] - p0[1])
        dv1 = v1 - v0; dv2 = v2 - v0
        det = dp1x * dp2y - dp1y * dp2x
        if abs(det) < 1e-8:
            continue
        inv = 1.0 / det
        tan  = np.float32( dp2y * inv) * dv1 - np.float32(dp1y * inv) * dv2
        btan = np.float32(-dp2x * inv) * dv1 + np.float32(dp1x * inv) * dv2
        tn = math.sqrt(float(tan[0]**2 + tan[1]**2 + tan[2]**2))
        bn = math.sqrt(float(btan[0]**2 + btan[1]**2 + btan[2]**2))
        if tn < 1e-10 or bn < 1e-10:
            continue
        tan  = tan  / np.float32(tn)
        btan = btan / np.float32(bn)

        # ── Bounding box ────────────────────────────────────────────────────
        x0_ = max(0, int(math.floor(min(p0[0], p1[0], p2[0]))) - 1)
        x1_ = min(W - 1, int(math.ceil(max(p0[0], p1[0], p2[0]))) + 1)
        y0_ = max(0, int(math.floor(min(p0[1], p1[1], p2[1]))) - 1)
        y1_ = min(H - 1, int(math.ceil(max(p0[1], p1[1], p2[1]))) + 1)
        if x0_ > x1_ or y0_ > y1_:
            continue

        gx = np.arange(x0_, x1_ + 1, dtype=np.float32)
        gy = np.arange(y0_, y1_ + 1, dtype=np.float32)
        GX, GY = np.meshgrid(gx, gy)   # (Ny, Nx)

        # ── Edge functions (barycentric) ────────────────────────────────────
        # w_i = signed area of sub-triangle opposite vertex i
        def _e(ax, ay, bx, by):
            return (float(bx) - float(ax)) * (GY - float(ay)) \
                 - (float(by) - float(ay)) * (GX - float(ax))

        w0 = _e(p1[0], p1[1], p2[0], p2[1])   # sub-area opposite p0
        w1 = _e(p2[0], p2[1], p0[0], p0[1])   # sub-area opposite p1
        w2 = _e(p0[0], p0[1], p1[0], p1[1])   # sub-area opposite p2

        area2 = float(
            (p1[0] - p0[0]) * (p2[1] - p0[1])
            - (p1[1] - p0[1]) * (p2[0] - p0[0])
        )
        if abs(area2) < 1e-8:
            continue

        # Inside: all w same sign as area2 (tol = 0.5 px for anti-alias padding)
        eps = 0.5
        if area2 > 0:
            inside = (w0 >= -eps) & (w1 >= -eps) & (w2 >= -eps)
        else:
            inside = (w0 <= eps) & (w1 <= eps) & (w2 <= eps)

        rows, cols = np.where(inside)
        if len(rows) == 0:
            continue

        b0_ = (w0[rows, cols] / area2).astype(np.float32)[:, np.newaxis]
        b1_ = (w1[rows, cols] / area2).astype(np.float32)[:, np.newaxis]
        b2_ = (w2[rows, cols] / area2).astype(np.float32)[:, np.newaxis]

        ry_ = rows + y0_
        cx_ = cols + x0_

        world_pos[ry_, cx_] = v0 * b0_ + v1 * b1_ + v2 * b2_
        wn = n0 * b0_ + n1 * b1_ + n2 * b2_
        wn_l = np.linalg.norm(wn, axis=1, keepdims=True)
        world_nrm[ry_, cx_] = np.where(wn_l > 1e-8, wn / wn_l, wn)
        tan_out[ry_, cx_]   = tan
        btan_out[ry_, cx_]  = btan
        mask[ry_, cx_]      = True

    return world_pos, world_nrm, tan_out, btan_out, mask


# ══════════════════════════════════════════════════════════════════════════════
# Texture dilation
# ══════════════════════════════════════════════════════════════════════════════

def _dilate_texture(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Fill every unmapped texel with the value of the nearest mapped texel.

    Uses ``scipy.ndimage.distance_transform_edt`` with ``return_indices`` for
    an O(H×W) single-pass nearest-neighbour fill.  This prevents black seam
    bleeding at UV chart edges when bilinear filtering is used.

    Parameters
    ----------
    img  : (H, W) or (H, W, C) float array — texture data.
    mask : (H, W) bool — True where img has valid data.

    Returns
    -------
    (H, W[, C]) float array with unmapped pixels filled in.
    """
    from scipy.ndimage import distance_transform_edt

    if mask.all():
        return img   # already fully covered

    # Find for every pixel the coordinates of the nearest mask=True pixel
    _, (nearest_row, nearest_col) = distance_transform_edt(~mask, return_indices=True)

    result = img.copy()
    fill = ~mask
    result[fill] = img[nearest_row[fill], nearest_col[fill]]
    return result


# ══════════════════════════════════════════════════════════════════════════════
# Procedural noise helpers
# ══════════════════════════════════════════════════════════════════════════════

def _uv_smooth_noise(H: int, W: int, sigma_px: float, rng) -> np.ndarray:
    """Gaussian-smoothed white noise in [0, 1], shape (H, W)."""
    n = rng.standard_normal((H, W))
    n = gaussian_filter(n, sigma=sigma_px)
    lo, hi = n.min(), n.max()
    if hi - lo < 1e-10:
        return np.zeros((H, W), np.float32)
    return ((n - lo) / (hi - lo)).astype(np.float32)


def _voronoi_f1(u: np.ndarray, v: np.ndarray, jitter: float = 0.8) -> np.ndarray:
    """Vectorised 2-D F1 Voronoi distance.

    Uses the "sin hash" trick (Shadertoy convention) on integer cell
    coordinates: fast, seamless, no table look-up.

    Parameters
    ----------
    u, v : (N,) flat arrays — coordinates in *cell* units
           (divide world coords by ``cell_size`` before calling).
    jitter : float in [0, 1] — 0 → perfect grid, 1 → maximum randomness.

    Returns
    -------
    f1 : (N,) float32 — distance to nearest Voronoi site, in [0, ~0.75].
    """
    u = u.astype(np.float64)
    v = v.astype(np.float64)
    iu = np.floor(u)
    iv = np.floor(v)
    fu = u - iu
    fv = v - iv

    min_d2 = np.full(len(u), 1e10, dtype=np.float64)

    for di in range(-2, 3):
        for dj in range(-2, 3):
            ni = iu + di   # cell integer coords
            nj = iv + dj

            # Shadertoy-style hash: frac(sin(dot(n, prime)) * big_number)
            rx = ni * 127.1 + nj * 311.7
            rx = np.sin(rx) * 43758.5453
            rx -= np.floor(rx)            # frac → [0, 1]

            ry = ni * 269.5 + nj * 183.3
            ry = np.sin(ry) * 43758.5453
            ry -= np.floor(ry)

            # Jitter: point sits at (di + jittered_offset - fu, ...)
            rx = 0.5 + jitter * (rx - 0.5)
            ry = 0.5 + jitter * (ry - 0.5)

            dx = di - fu + rx
            dy = dj - fv + ry
            d2 = dx * dx + dy * dy
            np.minimum(min_d2, d2, out=min_d2)

    return np.sqrt(min_d2).astype(np.float32)


def _fbm_opensimplex(
    u: np.ndarray, v: np.ndarray,
    octaves: int = 4, lacunarity: float = 2.0, gain: float = 0.5,
    seed: int = 0,
) -> np.ndarray:
    """fBm using opensimplex, evaluated at flat arrays (u, v).

    Returns values in approximately [−1, 1].
    """
    import opensimplex as osx
    osx.seed(seed)

    amp = 1.0
    freq = 1.0
    total = np.zeros(len(u), dtype=np.float32)
    norm = 0.0

    for _ in range(octaves):
        vals = np.array(
            [osx.noise2(float(u[i] * freq), float(v[i] * freq))
             for i in range(len(u))],
            dtype=np.float32,
        )
        total += amp * vals
        norm  += amp
        amp  *= gain
        freq *= lacunarity

    return total / (norm + 1e-12)


# ══════════════════════════════════════════════════════════════════════════════
# Height map
# ══════════════════════════════════════════════════════════════════════════════

def _gen_height(
    world_pos:   np.ndarray,   # (H, W, 3)
    tangent_map: np.ndarray,   # (H, W, 3)
    bitan_map:   np.ndarray,   # (H, W, 3)
    mask:        np.ndarray,   # (H, W) bool
    pore_cell_size: float,
    H: int, W: int,
    rng,
) -> np.ndarray:
    """Height map in [0, 1]. Shape (H, W).

    Layers:
    - Pore dimples   : F1 Voronoi in tangent-plane space (0 = pore centre, 1 = rim)
    - Micro-texture  : low-amplitude fBm (opensimplex) evaluated at tangent UV
    """
    height = np.zeros((H, W), np.float32)

    if not mask.any():
        return height

    # Flat arrays of masked pixels
    rows, cols = np.where(mask)

    wp = world_pos[rows, cols]    # (N, 3)
    T  = tangent_map[rows, cols]  # (N, 3)
    B  = bitan_map[rows, cols]    # (N, 3)

    # Tangent-plane coordinates (world units)
    u_w = (wp * T).sum(axis=1)
    v_w = (wp * B).sum(axis=1)

    # ── Pore layer ──────────────────────────────────────────────────────────
    u_cell = u_w / pore_cell_size
    v_cell = v_w / pore_cell_size
    f1 = _voronoi_f1(u_cell, v_cell, jitter=0.82)
    # Normalise to [0, 1]; rims at ~0.58 (typical F1 max for jitter 0.8)
    f1 = np.clip(f1 / 0.60, 0.0, 1.0)
    # Sharpen: dimple is a sharp dip, not a broad bowl
    pore_h = f1 ** 2.0    # concave — deeper at pore centre (f1 ≈ 0)

    # ── Micro-texture layer (opensimplex fBm in tangent space) ──────────────
    freq_base = 1.0 / (pore_cell_size * 3.0)   # ~3× pore frequency
    u_n = u_w * freq_base
    v_n = v_w * freq_base

    # Limit sample count to avoid excessive runtime for large meshes
    N = len(rows)
    MAX_NOISE_SAMPLES = 200_000
    if N > MAX_NOISE_SAMPLES:
        # Subsample, then interpolate via sparse-to-dense
        idx_sub = rng.choice(N, MAX_NOISE_SAMPLES, replace=False)
        micro_sub = _fbm_opensimplex(u_n[idx_sub], v_n[idx_sub], octaves=3)
        # Scatter into full-size array and blur to fill gaps
        micro_full_flat = np.zeros(N, np.float32)
        micro_full_flat[idx_sub] = micro_sub
        # Re-scatter into 2D and blur
        micro_2d = np.zeros((H, W), np.float32)
        np.add.at(micro_2d, (rows, cols), micro_full_flat)
        from scipy.ndimage import gaussian_filter
        micro_2d = gaussian_filter(micro_2d, sigma=2.0)
        micro = micro_2d[rows, cols]
    else:
        micro = _fbm_opensimplex(u_n, v_n, octaves=3)

    micro = (micro * 0.5 + 0.5).astype(np.float32)   # [0, 1]

    # ── Combine ─────────────────────────────────────────────────────────────
    h_pixels = 0.72 * pore_h + 0.28 * micro

    height[rows, cols] = h_pixels

    # Dilate into unmasked border pixels so Sobel gradient doesn't glitch at seams
    from scipy.ndimage import uniform_filter
    height_filled = uniform_filter(height, size=3)
    height[~mask] = 0.5

    return height


# ══════════════════════════════════════════════════════════════════════════════
# Normal map
# ══════════════════════════════════════════════════════════════════════════════

def _gen_normal_from_height(
    height:   np.ndarray,   # (H, W)
    mask:     np.ndarray,   # (H, W)
    strength: float,
) -> np.ndarray:
    """Tangent-space normal map from height gradient.

    Encodes as RGB float [0, 1] → uint8 later.
    Standard wgpu/DirectX convention: R=X(right), G=Y(down), B=Z(up from surface).
    """
    H, W = height.shape
    # Central-difference gradient
    dHdy, dHdx = np.gradient(height)

    # Scale: strength controls bump amplitude
    sx = -dHdx * strength * W
    sy =  dHdy * strength * H   # +Y = down in wgpu tangent space; flip sign vs OpenGL

    sz = np.ones((H, W), np.float32)

    nrm_len = np.sqrt(sx**2 + sy**2 + 1.0)
    nx = sx / nrm_len
    ny = sy / nrm_len
    nz = sz / nrm_len

    # Encode [−1, 1] → [0, 1]; background initialized to flat (0.5, 0.5, 1.0)
    out = np.stack([
        (nx * 0.5 + 0.5),
        (ny * 0.5 + 0.5),
        (nz * 0.5 + 0.5),
    ], axis=2).astype(np.float32)

    # Flat normal for unmasked pixels (will be replaced by dilation but good fallback)
    flat = np.array([0.5, 0.5, 1.0], np.float32)
    out[~mask] = flat

    return out   # (H, W, 3)


# ══════════════════════════════════════════════════════════════════════════════
# Albedo
# ══════════════════════════════════════════════════════════════════════════════

# Base skin tone (linear sRGB, fair warm-healthy skin)
_SKIN_BASE_LIN = np.array([0.765, 0.544, 0.445], np.float32)

# Freckle colour (warm brown, darkened)
_FRECKLE_COLOR = np.array([0.42, 0.27, 0.18], np.float32)

# Subtle blemish colour (rose/pink tint)
_BLEMISH_COLOR = np.array([0.82, 0.48, 0.43], np.float32)


def _gen_albedo(
    world_pos:   np.ndarray,
    world_nrm:   np.ndarray,
    tangent_map: np.ndarray,
    bitan_map:   np.ndarray,
    mask:        np.ndarray,
    pore_cell_size: float,
    height:      np.ndarray,
    freckle_density: float,
    sun_dir:     np.ndarray,
    H: int, W: int,
    rng,
) -> np.ndarray:
    """Layered albedo map. Returns (H, W, 3) float [0, 1]."""
    # Pre-fill entire image with base skin tone so empty atlas space is skin-coloured.
    albedo = np.tile(_SKIN_BASE_LIN, (H, W, 1)).astype(np.float32)
    if not mask.any():
        return albedo

    rows, cols = np.where(mask)
    N = len(rows)

    # ── Base tone ────────────────────────────────────────────────────────────
    layer = np.tile(_SKIN_BASE_LIN, (N, 1))   # (N, 3)

    # ── Large-scale tone variation (~50 mm) ──────────────────────────────────
    var_map = _uv_smooth_noise(H, W, sigma_px=H * 0.025, rng=rng)  # ~2.5% of image
    var = var_map[rows, cols]
    layer += (var - 0.5)[:, np.newaxis] * 0.06   # ±3 % variation

    # ── Sun-exposure gradient (upward-facing areas slightly warmer/darker) ───
    facing = np.clip((world_nrm[rows, cols] * sun_dir).sum(axis=1), 0.0, 1.0)
    sun_tint = np.array([0.04, 0.015, -0.01], np.float32)
    layer += facing[:, np.newaxis] * sun_tint

    # ── Pore colour: very faint darkening at pore centres ───────────────────
    wp = world_pos[rows, cols]
    T  = tangent_map[rows, cols]
    B  = bitan_map[rows, cols]
    u_cell = (wp * T).sum(axis=1) / pore_cell_size
    v_cell = (wp * B).sum(axis=1) / pore_cell_size
    f1 = _voronoi_f1(u_cell, v_cell, jitter=0.82)
    f1 = np.clip(f1 / 0.60, 0.0, 1.0)
    pore_dark = (1.0 - f1 ** 3) * 0.07    # subtle darkening at centre
    layer -= pore_dark[:, np.newaxis]

    # ── Freckles ─────────────────────────────────────────────────────────────
    if freckle_density > 0.0:
        n_freckles = int(freckle_density * 35)
        _apply_freckles(albedo, layer, rows, cols, world_pos, world_nrm,
                        sun_dir, n_freckles, mask, H, W, rng)
    else:
        albedo[rows, cols] = layer

    albedo[rows, cols] = np.clip(albedo[rows, cols], 0.0, 1.0)

    # Merge freckles if they were written separately
    # (already in albedo[rows, cols] from _apply_freckles or layer assignment)

    # ── 1–2 micro-blemishes (very faint rose patches) ───────────────────────
    _apply_blemishes(albedo, mask, H, W, rng)

    # ── Final clamp ──────────────────────────────────────────────────────────
    np.clip(albedo, 0.0, 1.0, out=albedo)
    return albedo


def _apply_freckles(
    albedo: np.ndarray,
    base_layer: np.ndarray,   # (N, 3) already built base for masked pixels
    rows: np.ndarray,
    cols: np.ndarray,
    world_pos: np.ndarray,
    world_nrm: np.ndarray,
    sun_dir: np.ndarray,
    n_seeds: int,
    surf_mask: np.ndarray,    # (H, W) bool
    H: int, W: int,
    rng,
) -> None:
    """Write freckles into *albedo* using Poisson-disk seeds in UV space."""
    N = len(rows)

    # Seed positions in UV space, weighted toward sun-facing
    facing = np.clip((world_nrm[rows, cols] * sun_dir).sum(axis=1), 0.0, 1.0) ** 2
    prob = facing / (facing.sum() + 1e-12)

    # Sample seed pixel indices
    seed_idx = rng.choice(N, size=min(n_seeds * 4, N), replace=False, p=prob)

    # Poisson-disk filtering in image space (reject if too close to existing seeds)
    min_dist_px = H * 0.025    # ~25 mm for a human-scale mesh → well-spaced freckles
    accepted = []
    seed_uv   = []
    for si in seed_idx:
        r, c = rows[si], cols[si]
        ok = True
        for (pr, pc) in seed_uv:
            if (r - pr)**2 + (c - pc)**2 < min_dist_px**2:
                ok = False
                break
        if ok:
            accepted.append(si)
            seed_uv.append((r, c))
        if len(accepted) >= n_seeds:
            break

    # Write base layer first
    albedo[rows, cols] = base_layer

    # Paint each freckle as a small Gaussian blob (physically ~1–3 mm diameter)
    for si in accepted:
        sr, sc = rows[si], cols[si]
        # Radius in pixels: aim for ~1–2.5 mm on a mesh measured in metres.
        # Heuristic: freckle spans ~0.002 m; relate to image density later.
        # Use 0.5–1.2 % of H as pixel radius (sub-pixel at low H → scales up nicely).
        radius_px = rng.uniform(H * 0.005, H * 0.012)   # 5-12 px at 1024
        opacity   = rng.uniform(0.18, 0.42)
        # Slight colour variation between freckles
        brightness = rng.uniform(0.85, 1.15)
        color = np.clip(_FRECKLE_COLOR * brightness, 0.0, 1.0)

        # Bounding box
        r0 = max(0, int(sr - radius_px * 3))
        r1 = min(H - 1, int(sr + radius_px * 3))
        c0 = max(0, int(sc - radius_px * 3))
        c1 = min(W - 1, int(sc + radius_px * 3))

        gr = np.arange(r0, r1 + 1)
        gc = np.arange(c0, c1 + 1)
        GR, GC = np.meshgrid(gr, gc, indexing='ij')
        d2 = (GR - sr)**2 + (GC - sc)**2
        alpha = opacity * np.exp(-d2 / (2 * radius_px**2)).astype(np.float32)
        mf = surf_mask[r0:r1+1, c0:c1+1]
        albedo[r0:r1+1, c0:c1+1] = (
            albedo[r0:r1+1, c0:c1+1] * (1 - alpha[:, :, np.newaxis])
            + color * alpha[:, :, np.newaxis]
        ) * mf[:, :, np.newaxis] + albedo[r0:r1+1, c0:c1+1] * (~mf)[:, :, np.newaxis]


def _apply_blemishes(
    albedo: np.ndarray,
    mask: np.ndarray,
    H: int, W: int,
    rng,
) -> None:
    """1–2 very faint pink blemish patches."""
    rows_m, cols_m = np.where(mask)
    if len(rows_m) == 0:
        return
    n_blem = rng.integers(1, 3)
    for _ in range(n_blem):
        idx   = rng.integers(0, len(rows_m))
        sr, sc = rows_m[idx], cols_m[idx]
        radius_px = rng.uniform(H * 0.010, H * 0.020)
        opacity = rng.uniform(0.04, 0.10)

        r0 = max(0, int(sr - radius_px * 2.5))
        r1 = min(H - 1, int(sr + radius_px * 2.5))
        c0 = max(0, int(sc - radius_px * 2.5))
        c1 = min(W - 1, int(sc + radius_px * 2.5))

        gr = np.arange(r0, r1 + 1)
        gc = np.arange(c0, c1 + 1)
        GR, GC = np.meshgrid(gr, gc, indexing='ij')
        d2 = (GR - sr)**2 + (GC - sc)**2
        alpha = opacity * np.exp(-d2 / (2 * radius_px**2)).astype(np.float32)
        mf = mask[r0:r1+1, c0:c1+1]
        albedo[r0:r1+1, c0:c1+1] = (
            albedo[r0:r1+1, c0:c1+1] * (1 - alpha[:, :, np.newaxis])
            + _BLEMISH_COLOR * alpha[:, :, np.newaxis]
        ) * mf[:, :, np.newaxis] + albedo[r0:r1+1, c0:c1+1] * (~mf)[:, :, np.newaxis]


# ══════════════════════════════════════════════════════════════════════════════
# Roughness
# ══════════════════════════════════════════════════════════════════════════════

def _gen_roughness(
    height: np.ndarray,   # (H, W)
    mask:   np.ndarray,
    base:   float,
    dewy:   float,
    H: int, W: int,
    rng,
) -> np.ndarray:
    """Roughness map in [0, 1]. Shape (H, W)."""
    rough = np.full((H, W), base, np.float32)
    inverted_height = 1.0 - height
    pore_mask = np.clip(inverted_height * 1.3, 0.0, 1.0) ** 3   # only very-low pores
    rough += pore_mask * 0.12

    # Dewy patches: smooth low-frequency noise → lower roughness
    dewy_map = _uv_smooth_noise(H, W, sigma_px=H * 0.06, rng=rng)
    rough -= dewy_map * dewy

    # Micro variation
    micro_rough = _uv_smooth_noise(H, W, sigma_px=2.0, rng=rng)
    rough += (micro_rough - 0.5) * 0.04

    rough[~mask] = base
    np.clip(rough, 0.12, 0.55, out=rough)
    return rough


# ══════════════════════════════════════════════════════════════════════════════
# Ambient Occlusion
# ══════════════════════════════════════════════════════════════════════════════

def _gen_ao(height: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Micro-AO baked from the height map.

    Pixels in local minima (pore centres) get darker AO.  Uses a fast
    maximum-filter approach: AO = 1 − (local_max − height) × strength.
    """
    from scipy.ndimage import maximum_filter
    pore_radius_px = max(3, int(height.shape[0] / 256))
    local_max = maximum_filter(height, size=pore_radius_px * 2 + 1)
    ao = 1.0 - np.clip((local_max - height) * 1.8, 0.0, 1.0)
    ao[~mask] = 1.0
    return ao.astype(np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# Emissive (SSS approximation)
# ══════════════════════════════════════════════════════════════════════════════

# Warm-red SSS colour (dermis-scattered light)
_SSS_COLOR = np.array([0.85, 0.32, 0.20], np.float32)


def _gen_emissive(
    world_pos:          np.ndarray,    # (H, W, 3)
    mask:               np.ndarray,
    thickness_per_vert: Optional[np.ndarray],   # (V_orig,) optional
    vmapping:           np.ndarray,
    H: int, W: int,
    rng,
) -> np.ndarray:
    """Warm SSS glow map.  Returns (H, W, 3) float [0, 1]."""
    emissive = np.zeros((H, W, 3), np.float32)
    if not mask.any():
        return emissive

    # Modulation: low-frequency noise
    mod = _uv_smooth_noise(H, W, sigma_px=H * 0.07, rng=rng)  # (H, W) [0,1]
    mod = 0.5 + mod * 0.5    # [0.5, 1.0] — ensures always some glow

    emissive[mask] = _SSS_COLOR * mod[mask, np.newaxis]

    return np.clip(emissive, 0.0, 1.0)

