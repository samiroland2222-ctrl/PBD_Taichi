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
from scipy.ndimage import gaussian_filter

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

    # ------------------------------------------------------------------
    def save_with_overlay(self, directory: str, prefix: str = "skin",
                          line_color: tuple = (255, 220, 0),
                          line_alpha: int = 180,
                          line_width: int = 1) -> None:
        """Write all maps as PNG files with the UV mesh wireframe overlaid.

        Each triangle edge of the UV atlas is drawn in *line_color* so you
        can verify that the texture content aligns with the mesh geometry.

        Parameters
        ----------
        directory  : Output directory (created if absent).
        prefix     : Filename prefix (default ``"skin"``).
        line_color : RGB tuple for the wireframe lines (default yellow).
        line_alpha : Opacity 0-255 of the overlay lines (default 180).
        line_width : Width of the triangle edge lines in pixels (default 1).
        """
        _save_overlay_maps(
            directory, prefix,
            uvs=self.uvs, indices=self.indices,
            maps=[
                (self.albedo_np,           "albedo"),
                (self.normal_np[:, :, :3], "normal"),
                (self.roughness_np,        "roughness"),
                (self.ao_np,               "ao"),
                (self.height_np,           "height"),
                (self.emissive_np,         "emissive"),
            ],
            verbose=True,
            line_color=line_color,
            line_alpha=line_alpha,
            line_width=line_width,
        )


# ══════════════════════════════════════════════════════════════════════════════
# Main entry point
# ══════════════════════════════════════════════════════════════════════════════

def generate_skin_textures(
    verts: np.ndarray,                       # (V, 3) float64
    faces: np.ndarray,                       # (F, 3) int32
    resolution: int = 4096,
    seed: int = 42,
    # ── appearance knobs ─────────────────────────────────────────────────
    pore_cell_size: float = 0.0,             # pore spacing (0 = auto from bbox)
    freckle_density: float = 0.5,            # 0 = none, 1 = many
    base_roughness: float = 0.55,            # skin roughness baseline (0=mirror, 1=diffuse)
    dewy_intensity: float = 0.10,            # roughness reduction in sebum patches
    emissive_intensity: float = 0.10,        # SSS glow multiplier in material
    normal_strength: float = 0.45,           # height→normal bump amplitude (0=flat, 1=strong)
    # ── optional inputs ──────────────────────────────────────────────────
    thickness_per_vertex: Optional[np.ndarray] = None,  # (V,) — shell thickness
    sun_dir: Optional[np.ndarray] = None,               # (3,) — sun direction
    verbose: bool = True,
    debug_save_dir: Optional[str] = None,    # if set, save atlas PNG files here
) -> SkinTextures:
    """Generate a full PBR skin texture set for an arbitrary triangle mesh.

    Returns a :class:`SkinTextures` with pre-built pygfx textures and a
    ready-to-use :class:`pygfx.MeshStandardMaterial`.

    ``pore_cell_size`` controls pore spacing in mesh units.  The default (0)
    auto-scales it to ~1/80 of the mesh bounding-box diagonal so pores are
    always a few atlas pixels wide regardless of mesh scale.
    """
    rng = np.random.default_rng(seed)
    H = W = resolution

    # ── Auto-scale pore cell size ─────────────────────────────────────────────
    if pore_cell_size <= 0.0:
        bbox_diag = float(np.linalg.norm(verts.max(axis=0) - verts.min(axis=0)))
        # Target: pore ≈ 1/180 of bbox diagonal — gives fine micro-texture
        # (smaller divisor = larger pores; 80 was too coarse, 180 gives ~5.7 px/pore at 1024)
        pore_cell_size = max(bbox_diag / 180.0, 1e-6)
        if verbose:
            print(f"  [skin_tex] pore_cell_size auto → {pore_cell_size:.5f} "
                  f"(bbox diag {bbox_diag:.4f})")

    if sun_dir is None:
        sun_dir = np.array([0.1, 1.0, 0.3], dtype=np.float64)
    sun_dir = sun_dir / (np.linalg.norm(sun_dir) + 1e-12)

    # ── 1. UV atlas ──────────────────────────────────────────────────────────
    t0 = time.time()
    vmapping, uv_indices, uvs = _generate_uv_atlas(verts, faces, resolution,
                                                    verbose=verbose)
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
    t0 = time.time()
    normal_f  = _gen_normal_from_height(height_f, mp, normal_strength)
    if verbose:
        print(f"  [skin_tex] normal    {time.time()-t0:.1f}s")

    # ── 5. Albedo ────────────────────────────────────────────────────────────
    t0 = time.time()
    albedo_f  = _gen_albedo(world_pos, world_nrm, tangent_map, bitan_map,
                             mp, pore_cell_size, height_f,
                             freckle_density, sun_dir, H, W, rng)
    if verbose:
        print(f"  [skin_tex] albedo    {time.time()-t0:.1f}s")

    # ── 6. Roughness ─────────────────────────────────────────────────────────
    t0 = time.time()
    rough_f   = _gen_roughness(height_f, mp, base_roughness, dewy_intensity,
                               H, W, rng)
    if verbose:
        print(f"  [skin_tex] roughness {time.time()-t0:.1f}s")

    # ── 7. AO ────────────────────────────────────────────────────────────────
    t0 = time.time()
    ao_f      = _gen_ao(height_f, mp)
    if verbose:
        print(f"  [skin_tex] ao        {time.time()-t0:.1f}s")

    # ── 8. Emissive (SSS approximation) ──────────────────────────────────────
    t0 = time.time()
    emissive_f = _gen_emissive(world_pos, mp, thickness_per_vertex,
                                vmapping, H, W, rng)
    if verbose:
        print(f"  [skin_tex] emissive  {time.time()-t0:.1f}s")

    # ── 9. Dilate all maps into unmapped atlas space ──────────────────────────
    # xatlas typically leaves 40-60 % of the atlas unpopulated.  Without
    # dilation, bilinear sampling at UV-chart edges bleeds in black pixels
    # and the mesh looks splotchy.  Compute the EDT nearest-neighbour map
    # ONCE and share it across all six textures.
    if verbose and not mp.all():
        cov = mp.sum() / mp.size * 100
        print(f"  [skin_tex] dilating (coverage {cov:.1f}% → 100%) …")

    if mp.all():
        nearest_row, nearest_col = None, None   # no-op path
    else:
        from scipy.ndimage import distance_transform_edt
        _, (nearest_row, nearest_col) = distance_transform_edt(~mp, return_indices=True)

    def _dilate(arr):
        if nearest_row is None:
            return arr
        out = arr.copy()
        fill = ~mp
        out[fill] = arr[nearest_row[fill], nearest_col[fill]]
        return out

    albedo_f   = _dilate(albedo_f)
    normal_f   = _dilate(normal_f)
    rough_f    = _dilate(rough_f[:, :, np.newaxis])[:, :, 0]
    ao_f       = _dilate(ao_f[:, :, np.newaxis])[:, :, 0]
    height_f   = _dilate(height_f[:, :, np.newaxis])[:, :, 0]
    emissive_f = _dilate(emissive_f)

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

    # ── 12. Optional debug save ───────────────────────────────────────────────
    if debug_save_dir is not None:
        import os
        from PIL import Image as _PIL
        os.makedirs(debug_save_dir, exist_ok=True)
        _PIL.fromarray(albedo_np[:,:,:3]).save(os.path.join(debug_save_dir, "skin_albedo.png"))
        _PIL.fromarray(normal_np[:,:,:3]).save(os.path.join(debug_save_dir, "skin_normal.png"))
        _PIL.fromarray(roughness_np[:,:,0]).save(os.path.join(debug_save_dir, "skin_roughness.png"))
        _PIL.fromarray(ao_np[:,:,0]).save(os.path.join(debug_save_dir, "skin_ao.png"))
        _PIL.fromarray(height_np[:,:,0]).save(os.path.join(debug_save_dir, "skin_height.png"))
        _PIL.fromarray(emissive_np[:,:,:3]).save(os.path.join(debug_save_dir, "skin_emissive.png"))
        if verbose:
            print(f"  [skin_tex] debug maps saved to {debug_save_dir}/")
        # Also save mesh-overlay versions for seam/coverage inspection
        _save_overlay_maps(
            debug_save_dir, "skin",
            uvs=uvs, indices=uv_indices,
            maps=[
                (albedo_np,            "albedo"),
                (normal_np[:,:,:3],    "normal"),
                (roughness_np,         "roughness"),
                (ao_np,                "ao"),
                (height_np,            "height"),
                (emissive_np,          "emissive"),
            ],
            verbose=verbose,
        )

    # ── 13. Build material ────────────────────────────────────────────────────
    # Use clamp wrapping on all maps so UV-atlas chart boundaries never bleed
    # into the opposite side of the atlas (which repeat-wrapping would allow).
    def _atlas_map(tex, **kw):
        return pygfx.TextureMap(tex, wrap="clamp", **kw)

    material = pygfx.MeshStandardMaterial(
        # Albedo
        map=_atlas_map(albedo_tex),
        color=(1.0, 1.0, 1.0),
        # Geometry / surface
        normal_map=_atlas_map(normal_tex),
        normal_scale=(normal_strength, normal_strength),   # baked intensity already calibrated
        roughness=1.0,           # map provides per-texel value
        roughness_map=_atlas_map(roughness_tex),
        metalness=0.0,           # skin is dielectric
        # AO
        ao_map=_atlas_map(ao_tex),
        # SSS approximation via emissive
        emissive=(1.0, 1.0, 1.0),
        emissive_map=_atlas_map(emissive_tex),
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
# UV Atlas  —  single-chart Tutte harmonic embedding + xatlas packing fallback
# ══════════════════════════════════════════════════════════════════════════════

def _mesh_boundary_loop(
    faces: np.ndarray, n_verts: int,
) -> np.ndarray | None:
    """Return the ordered boundary vertex loop, or None if mesh is closed.

    Boundary edges are those that appear in only one triangle (no opposing
    half-edge).  Returns the longest loop so we handle meshes with multiple
    boundaries (holes) by picking the outer one.
    """
    from collections import defaultdict

    # Half-edge existence map  (a→b means triangle has directed edge a→b)
    fwd: dict[tuple[int, int], bool] = {}
    for tri in faces:
        for k in range(3):
            a, b = int(tri[k]), int(tri[(k + 1) % 3])
            fwd[(a, b)] = True

    # Boundary: half-edges whose opposite (b→a) does NOT exist
    bnd_next: dict[int, int] = {}      # a → b  (boundary adjacency)
    for (a, b) in fwd:
        if (b, a) not in fwd:
            bnd_next[a] = b

    if not bnd_next:
        return None   # closed mesh

    # Walk all boundary loops, keep the longest
    visited: set[int] = set()
    longest: list[int] = []

    for start in list(bnd_next.keys()):
        if start in visited:
            continue
        loop: list[int] = []
        cur = start
        while cur not in visited:
            visited.add(cur)
            loop.append(cur)
            nxt = bnd_next.get(cur)
            if nxt is None or nxt == start:
                break
            cur = nxt
        if len(loop) > len(longest):
            longest = loop

    return np.array(longest, dtype=np.int32) if len(longest) >= 3 else None


def _tutte_uv(
    verts: np.ndarray,    # (V, 3)
    faces: np.ndarray,    # (F, 3) int
    boundary: np.ndarray, # (B,) int  — ordered boundary vertex indices
) -> np.ndarray:
    """Tutte barycentric embedding — one-chart UV map for disk-topology meshes.

    Boundary vertices are pinned to a unit circle (arc-length parameterised).
    Interior vertices are set to the harmonic mean of their neighbours by
    solving the graph-Laplacian system.  Tutte's theorem guarantees the result
    is bijective (no foldovers) for any mesh with convex boundary.

    Returns (V, 2) float32 UVs in [pad, 1-pad] (already padded for the atlas).
    """
    from scipy.sparse import lil_matrix, csr_matrix
    from scipy.sparse.linalg import spsolve

    V = len(verts)
    B = len(boundary)

    # ── 1. Map boundary → unit circle (arc-length parameterised) ────────────
    bv = verts[boundary]                                    # (B, 3)
    edge_len = np.linalg.norm(
        np.roll(bv, -1, axis=0) - bv, axis=1)              # (B,)
    cum = np.concatenate([[0.0], np.cumsum(edge_len)])
    angles = cum[:-1] / (cum[-1] + 1e-12) * 2.0 * np.pi
    bnd_uv = np.stack([np.cos(angles), np.sin(angles)], axis=1)  # (B, 2)

    # ── 2. Build uniform graph Laplacian ─────────────────────────────────────
    # L[i,i] = degree(i),  L[i,j] = -1 if (i,j) is an edge, else 0
    adj: dict[int, set[int]] = {i: set() for i in range(V)}
    for tri in faces:
        for k in range(3):
            a, b = int(tri[k]), int(tri[(k + 1) % 3])
            adj[a].add(b)
            adj[b].add(a)

    L = lil_matrix((V, V), dtype=np.float64)
    for i in range(V):
        nbrs = adj[i]
        L[i, i] = float(len(nbrs))
        for j in nbrs:
            L[i, j] = -1.0
    L = csr_matrix(L)

    # ── 3. Partition: boundary (fixed) ↔ interior (unknown) ─────────────────
    is_bnd = np.zeros(V, bool)
    is_bnd[boundary] = True
    int_idx = np.where(~is_bnd)[0]   # interior vertex global indices
    bnd_idx = boundary                # boundary vertex global indices (ordered)

    uv = np.zeros((V, 2), np.float64)
    uv[bnd_idx] = bnd_uv

    if len(int_idx) > 0:
        L_ii = L[int_idx][:, int_idx]   # interior × interior
        L_ib = L[int_idx][:, bnd_idx]   # interior × boundary

        # Solve  L_ii * uv_int = -L_ib * uv_bnd
        rhs = -(L_ib @ bnd_uv)           # (I, 2) float64
        uv_int = spsolve(L_ii, rhs)      # (I, 2)
        uv[int_idx] = uv_int

    # ── 4. Normalise: unit circle → [pad, 1-pad] ────────────────────────────
    # Scale uniformly (preserve aspect ratio) so the largest dimension fills
    # the available atlas space with a 4-pixel border.
    lo = uv.min(axis=0); hi = uv.max(axis=0)
    span = (hi - lo).max() + 1e-12
    uv = (uv - lo) / span                   # [0, max_span]; square aspect
    # Re-centre
    centre_off = (1.0 - (hi - lo) / span) * 0.5
    uv += centre_off
    return uv.astype(np.float32)             # [0, 1]


def _generate_uv_atlas(
    verts: np.ndarray,
    faces: np.ndarray,
    resolution: int,
    verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate UV atlas.

    Strategy (in order of preference):
    1.  Open mesh (has boundary loop) → **Tutte harmonic embedding**:
        one island, seams only along the mesh boundary, zero per-triangle
        fragmentation.
    2.  Closed mesh → **spherical projection**:
        one island, seam only on the "back face" (opposite to dominant
        outward normal), minimal vertex splitting.
    3.  Last resort → xatlas fallback (degenerate/non-manifold topology).

    Returns (vmapping, new_indices, uvs) with the same semantics as before.
    For the Tutte and spherical paths vmapping contains no or very few splits.
    """
    # ── Try single-chart Tutte embedding (open meshes) ────────────────────
    bnd = _mesh_boundary_loop(faces, len(verts))

    if bnd is not None:
        if verbose:
            print(f"  [skin_tex] UV: Tutte embedding "
                  f"({len(verts)} verts, boundary loop {len(bnd)})")
        try:
            uvs = _tutte_uv(verts, faces, bnd)
            if np.all(np.isfinite(uvs)):
                pad = 4.0 / resolution
                uvs = uvs * (1.0 - 2.0 * pad) + pad
                vmapping = np.arange(len(verts), dtype=np.int32)
                return vmapping, faces.astype(np.int32), uvs
            if verbose:
                print("  [skin_tex] Tutte UV had non-finite values; "
                      "trying spherical projection")
        except Exception as e:
            if verbose:
                print(f"  [skin_tex] Tutte UV failed ({e}); "
                      "trying spherical projection")

    # ── Single-chart spherical projection (open or closed meshes) ────────
    if verbose:
        msg = "closed mesh" if bnd is None else "Tutte failed"
        print(f"  [skin_tex] UV: spherical projection ({msg})")
    try:
        vmapping, new_faces, uvs = _spherical_atlas(verts, faces, resolution)
        if np.all(np.isfinite(uvs)):
            return vmapping, new_faces, uvs
        if verbose:
            print("  [skin_tex] spherical UV had non-finite values; "
                  "falling back to xatlas")
    except Exception as e:
        if verbose:
            print(f"  [skin_tex] spherical UV failed ({e}); falling back to xatlas")

    # ── Last resort: xatlas ───────────────────────────────────────────────
    if verbose:
        print("  [skin_tex] UV: xatlas (fallback)")
    return _xatlas_atlas(verts, faces, resolution)


def _spherical_atlas(
    verts: np.ndarray,   # (V, 3)
    faces: np.ndarray,   # (F, 3) int
    resolution: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Single-chart spherical (equirectangular) UV projection.

    Works for both open and closed meshes.  The seam is placed on the "back"
    of the mesh — the half opposite to the area-weighted dominant outward
    normal — so it is hidden from the main viewing direction.

    Vertices that straddle the seam (u ≈ 0 ↔ u ≈ 1) are duplicated; all
    other vertices are shared across adjacent triangles (no per-triangle
    splitting).

    Returns (vmapping, new_faces, uvs).
    """
    V = len(verts)
    centroid = verts.mean(axis=0)

    # ── 1. Dominant outward normal (area-weighted mean face normal) ───────
    v0 = verts[faces[:, 0]]; v1 = verts[faces[:, 1]]; v2 = verts[faces[:, 2]]
    fn    = np.cross(v1 - v0, v2 - v0)                           # (F, 3)
    areas = np.linalg.norm(fn, axis=1, keepdims=True) + 1e-12
    dominant = (fn / areas).mean(axis=0)
    dn = np.linalg.norm(dominant)
    if dn < 1e-8:
        dominant = np.array([0.0, 0.0, 1.0])
    else:
        dominant /= dn

    # ── 2. Orthonormal frame: Z = dominant, X = "azimuth 0" ───────────────
    Z = dominant
    hint = np.array([0.0, 1.0, 0.0]) if abs(Z[0]) < 0.9 \
        else np.array([0.0, 0.0, 1.0])
    X = np.cross(hint, Z);  X /= (np.linalg.norm(X) + 1e-12)
    Y = np.cross(Z, X)

    # ── 3. Spherical coords for every vertex ──────────────────────────────
    dirs = verts - centroid
    dirs_n = dirs / (np.linalg.norm(dirs, axis=1, keepdims=True) + 1e-12)

    dx = (dirs_n * X).sum(axis=1)
    dy = (dirs_n * Y).sum(axis=1)
    dz = np.clip((dirs_n * Z).sum(axis=1), -1.0, 1.0)

    azimuth   = np.arctan2(dy, dx)       # [-π, π]
    elevation = np.arcsin(dz)             # [-π/2, π/2]

    # u : 0 = "back" (azimuth = -π), 0.5 = "front" (azimuth = 0), seam at u=0/1
    u = (azimuth / (2.0 * np.pi) + 0.5) % 1.0    # [0, 1)
    v_uv = (elevation + np.pi * 0.5) / np.pi       # [0, 1]

    # ── 4. Detect seam-crossing triangles ─────────────────────────────────
    u_tri   = u[faces]                              # (F, 3)
    u_range = u_tri.max(axis=1) - u_tri.min(axis=1)
    seam_tris = np.where(u_range > 0.5)[0]

    # For each seam triangle, vertices with u < 0.5 are on the "left" side
    # and need to be duplicated with u_new = u + 1.0 so the triangle spans
    # [u~1 … u~1+something] rather than jumping across the atlas.
    seam_vert_set: set[int] = set()
    for fi in seam_tris:
        for k in range(3):
            vi = int(faces[fi, k])
            if u[vi] < 0.5:
                seam_vert_set.add(vi)

    # ── 5. Duplicate seam vertices ────────────────────────────────────────
    seam_vert_list = sorted(seam_vert_set)
    n_seam = len(seam_vert_list)

    old_to_dup: dict[int, int] = {v: V + i for i, v in enumerate(seam_vert_list)}

    # Extended arrays
    u_ext   = np.concatenate([u,    u[seam_vert_list] + 1.0])  # dup → u+1
    v_ext   = np.concatenate([v_uv, v_uv[seam_vert_list]])
    vmapping = np.concatenate([
        np.arange(V, dtype=np.int32),
        np.array(seam_vert_list, dtype=np.int32),
    ])

    # ── 6. Rebuild faces: seam triangles use duplicates ───────────────────
    new_faces = faces.copy().astype(np.int32)
    for fi in seam_tris:
        for k in range(3):
            vi = int(faces[fi, k])
            if vi in old_to_dup:
                new_faces[fi, k] = old_to_dup[vi]

    # ── 7. Clamp u to [0, 1] and apply atlas border padding ───────────────
    u_ext = np.clip(u_ext, 0.0, 1.0)
    v_ext = np.clip(v_ext, 0.0, 1.0)
    pad   = 4.0 / resolution
    u_ext = u_ext * (1.0 - 2.0 * pad) + pad
    v_ext = v_ext * (1.0 - 2.0 * pad) + pad

    uvs = np.stack([u_ext, v_ext], axis=1).astype(np.float32)
    return vmapping, new_faces, uvs


def _xatlas_atlas(
    verts: np.ndarray,
    faces: np.ndarray,
    resolution: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """xatlas UV atlas with aggressive chart-merging (closed mesh fallback)."""
    import xatlas

    atlas = xatlas.Atlas()
    atlas.add_mesh(verts.astype(np.float32), faces.astype(np.uint32))

    chart_opts = xatlas.ChartOptions()
    chart_opts.max_cost = 1e6   # effectively unlimited — merge everything possible
    # Disable quality-based split weights so only topology drives chart boundaries
    for attr in ("normal_deviation_weight", "roundness_weight",
                 "straightness_weight", "normal_seam_weight",
                 "texture_seam_weight"):
        try:
            setattr(chart_opts, attr, 0.0)
        except AttributeError:
            pass

    pack_opts = xatlas.PackOptions()
    pack_opts.resolution = resolution
    pack_opts.padding    = 4
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
    """Gaussian-smoothed white noise in [0, 1], shape (H, W).

    For large sigma (> 8 px) we generate at a downsampled resolution and zoom
    back up — visually identical (the blur destroys sub-sample detail anyway)
    but runs ~scale² times faster.  At 4 K with sigma=287 this cuts runtime
    from ~20 s to < 0.1 s.
    """
    from scipy.ndimage import zoom as _zoom
    if sigma_px > 8.0:
        # Work at the smallest size where sigma ≥ 4 px (enough blur fidelity)
        scale = max(1, int(sigma_px / 4.0))
        Hs = max(16, (H + scale - 1) // scale)
        Ws = max(16, (W + scale - 1) // scale)
        n = rng.standard_normal((Hs, Ws)).astype(np.float32)
        n = gaussian_filter(n, sigma=sigma_px / scale)
        # Upsample to full atlas size (bilinear, prefilter=False avoids ringing)
        n = _zoom(n, (H / Hs, W / Ws), order=1, prefilter=False)
        n = n[:H, :W]     # trim any rounding overshoot
    else:
        n = rng.standard_normal((H, W)).astype(np.float32)
        n = gaussian_filter(n, sigma=sigma_px)
    lo, hi = float(n.min()), float(n.max())
    if hi - lo < 1e-10:
        return np.zeros((H, W), np.float32)
    return ((n - lo) / (hi - lo)).astype(np.float32)


def _voronoi_f1(u: np.ndarray, v: np.ndarray, jitter: float = 0.8) -> np.ndarray:
    """Vectorised 2-D F1 Voronoi distance (kept for reference; prefer 3-D version)."""
    u = u.astype(np.float64)
    v = v.astype(np.float64)
    iu = np.floor(u); iv = np.floor(v)
    fu = u - iu;      fv = v - iv
    min_d2 = np.full(len(u), 1e10, dtype=np.float64)
    for di in range(-2, 3):
        for dj in range(-2, 3):
            ni = iu + di; nj = iv + dj
            rx = np.sin(ni * 127.1 + nj * 311.7) * 43758.5453; rx -= np.floor(rx)
            ry = np.sin(ni * 269.5 + nj * 183.3) * 43758.5453; ry -= np.floor(ry)
            rx = 0.5 + jitter * (rx - 0.5); ry = 0.5 + jitter * (ry - 0.5)
            dx = di - fu + rx; dy = dj - fv + ry
            np.minimum(min_d2, dx*dx + dy*dy, out=min_d2)
    return np.sqrt(min_d2).astype(np.float32)


def _voronoi_f1_3d(x: np.ndarray, y: np.ndarray, z: np.ndarray,
                   jitter: float = 0.8) -> np.ndarray:
    """Vectorised 3-D F1 Voronoi distance — seamless on any triangle mesh.

    Evaluating noise in world-space 3-D coords means the pattern is
    continuous at every triangle/chart boundary regardless of UV layout.

    Parameters
    ----------
    x, y, z : (N,) flat arrays — world-space coordinates in *cell* units
              (divide by ``cell_size`` before calling).
    jitter  : float in [0, 1].

    Returns
    -------
    f1 : (N,) float32 — distance to nearest Voronoi site.
    """
    x = x.astype(np.float64); y = y.astype(np.float64); z = z.astype(np.float64)
    ix = np.floor(x); iy = np.floor(y); iz = np.floor(z)
    fx = x - ix;      fy = y - iy;      fz = z - iz
    min_d2 = np.full(len(x), 1e10, dtype=np.float64)

    for di in range(-1, 2):
        for dj in range(-1, 2):
            for dk in range(-1, 2):
                ni = ix + di; nj = iy + dj; nk = iz + dk
                # Three independent hashes per axis
                hx = np.sin(ni * 127.1 + nj * 311.7 + nk *  74.7) * 43758.5453
                hx -= np.floor(hx)
                hy = np.sin(ni * 269.5 + nj * 183.3 + nk * 246.1) * 43758.5453
                hy -= np.floor(hy)
                hz = np.sin(ni * 113.5 + nj * 271.9 + nk * 124.6) * 43758.5453
                hz -= np.floor(hz)
                rx = 0.5 + jitter * (hx - 0.5)
                ry = 0.5 + jitter * (hy - 0.5)
                rz = 0.5 + jitter * (hz - 0.5)
                dx = di - fx + rx; dy = dj - fy + ry; dz = dk - fz + rz
                np.minimum(min_d2, dx*dx + dy*dy + dz*dz, out=min_d2)

    return np.sqrt(min_d2).astype(np.float32)


def _fbm_opensimplex(
    u: np.ndarray, v: np.ndarray,
    octaves: int = 4, lacunarity: float = 2.0, gain: float = 0.5,
    seed: int = 0,
) -> np.ndarray:
    """fBm using opensimplex, evaluated at flat arrays (u, v).  Returns ≈[−1,1]."""
    import opensimplex as osx
    osx.seed(seed)
    amp = 1.0; freq = 1.0
    total = np.zeros(len(u), dtype=np.float32)
    norm = 0.0
    for _ in range(octaves):
        vals = np.array([osx.noise2(float(u[i]*freq), float(v[i]*freq))
                         for i in range(len(u))], dtype=np.float32)
        total += amp * vals; norm += amp; amp *= gain; freq *= lacunarity
    return total / (norm + 1e-12)


def _fbm_opensimplex_3d(
    x: np.ndarray, y: np.ndarray, z: np.ndarray,
    octaves: int = 4, lacunarity: float = 2.0, gain: float = 0.5,
    seed: int = 0,
) -> np.ndarray:
    """fBm using opensimplex 3-D noise — seamless across UV chart seams.

    Note: ``opensimplex.noise3array`` is a *grid* API (x, y, z are axis
    vectors, not point lists) and can't be used here.  We call the scalar
    ``noise3`` in a tight Python loop which is acceptable because callers
    always cap the point count at ≤ 200 K before calling this function.

    Returns values in approximately [−1, 1].
    """
    import opensimplex as osx
    osx.seed(seed)
    amp = 1.0; freq = 1.0
    total = np.zeros(len(x), dtype=np.float32)
    norm = 0.0
    xf = np.asarray(x, dtype=np.float64)
    yf = np.asarray(y, dtype=np.float64)
    zf = np.asarray(z, dtype=np.float64)
    for _ in range(octaves):
        vals = np.array(
            [osx.noise3(xf[i]*freq, yf[i]*freq, zf[i]*freq)
             for i in range(len(xf))],
            dtype=np.float32,
        )
        total += amp * vals; norm += amp; amp *= gain; freq *= lacunarity
    return total / (norm + 1e-12)


# ══════════════════════════════════════════════════════════════════════════════
# Height map
# ══════════════════════════════════════════════════════════════════════════════

def _gen_height(
    world_pos:   np.ndarray,   # (H, W, 3)
    tangent_map: np.ndarray,   # (H, W, 3)  — unused now (kept for API compat)
    bitan_map:   np.ndarray,   # (H, W, 3)  — unused now
    mask:        np.ndarray,   # (H, W) bool
    pore_cell_size: float,
    H: int, W: int,
    rng,
) -> np.ndarray:
    """Height map in [0, 1]. Shape (H, W).

    Layers:
    - Pore dimples   : 3-D F1 Voronoi in world space (seamless across UV seams)
    - Micro-texture  : 3-D fBm opensimplex evaluated at world-space coords
    """
    height = np.zeros((H, W), np.float32)

    if not mask.any():
        return height

    # Flat arrays of masked pixels
    rows, cols = np.where(mask)
    wp = world_pos[rows, cols]    # (N, 3)

    # World-space cell coordinates (no tangent-frame dependency → seamless)
    x_c = wp[:, 0] / pore_cell_size
    y_c = wp[:, 1] / pore_cell_size
    z_c = wp[:, 2] / pore_cell_size

    # Budget: cap samples at 200 K regardless of resolution.
    # At 4 K with ~11 M masked pixels the blur sigma will be ~22 px,
    # which softens the pore grid into fine micro-texture — exactly what
    # smooth dewy skin should look like.  Keeping the cap fixed avoids
    # the O(N_pixels) Voronoi loop that would take ~2 minutes at 4 K.
    MAX_SAMPLES = 200_000

    N = len(rows)

    # ── Pore layer ──────────────────────────────────────────────────────────
    if N > MAX_SAMPLES:
        idx_v = rng.choice(N, MAX_SAMPLES, replace=False)
        f1_sub = _voronoi_f1_3d(x_c[idx_v], y_c[idx_v], z_c[idx_v], jitter=0.82)
        f1_sparse = np.zeros(N, np.float32)
        f1_sparse[idx_v] = f1_sub
        f1_2d = np.zeros((H, W), np.float32)
        np.add.at(f1_2d, (rows, cols), f1_sparse)
        v_sigma = max(2.0, np.sqrt(float(N) / MAX_SAMPLES) * 2.0)
        f1_2d = gaussian_filter(f1_2d, sigma=v_sigma)
        f1 = f1_2d[rows, cols]
    else:
        f1 = _voronoi_f1_3d(x_c, y_c, z_c, jitter=0.82)
    # 3-D Voronoi has typical max ~0.87 (cube diagonal / 2≈0.87)
    f1 = np.clip(f1 / 0.75, 0.0, 1.0)
    # Gentle bowl shape — avoid sharp craters; smooth skin has rounded pores
    pore_h = f1 ** 1.2    # was 2.0 (sharp craters) → 1.2 (gentle concavity)

    # ── Micro-texture layer (opensimplex fBm in world space) ──────────────
    freq_base = 1.0 / (pore_cell_size * 3.0)   # ~3× pore frequency

    if N > MAX_SAMPLES:
        # Subsample, then blur back to full coverage
        idx_sub = rng.choice(N, MAX_SAMPLES, replace=False)
        micro_sub = _fbm_opensimplex_3d(
            wp[idx_sub, 0] * freq_base,
            wp[idx_sub, 1] * freq_base,
            wp[idx_sub, 2] * freq_base,
            octaves=3,
        )
        micro_full_flat = np.zeros(N, np.float32)
        micro_full_flat[idx_sub] = micro_sub
        micro_2d = np.zeros((H, W), np.float32)
        np.add.at(micro_2d, (rows, cols), micro_full_flat)
        m_sigma = max(2.0, np.sqrt(float(N) / MAX_SAMPLES) * 1.5)
        micro_2d = gaussian_filter(micro_2d, sigma=m_sigma)
        micro = micro_2d[rows, cols]
    else:
        micro = _fbm_opensimplex_3d(
            wp[:, 0] * freq_base,
            wp[:, 1] * freq_base,
            wp[:, 2] * freq_base,
            octaves=3,
        )

    micro = (micro * 0.5 + 0.5).astype(np.float32)   # [0, 1]

    # ── Combine ─────────────────────────────────────────────────────────────
    # Dewy young skin: mostly smooth micro-texture, barely-visible pores.
    # Weight pores at 25% and fBm micro-grain at 75% → no obvious crater grid.
    h_pixels = 0.25 * pore_h + 0.75 * micro

    height[rows, cols] = h_pixels

    # Pre-fill border pixels so Sobel gradient doesn't glitch at seams
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

    The gradient is computed only within atlas chart interiors: boundary pixels
    (mask neighbours that are outside) fall back to the flat (0, 0, 1) normal
    so that Sobel doesn't cross-pollinate neighbouring chart regions and create
    dark lines along UV seam edges.

    ``strength`` is a [0, 1] user knob.  Internally it is mapped to a moderate
    slope scale (~10–25° at pore edges for strength ≈ 0.5) by multiplying by a
    fixed constant.  The old ``* W`` formula scaled with resolution (W ≈ 1024)
    giving preposterously steep slopes (≈ 89°); this version uses a fixed
    multiplier of ~6 that is independent of atlas resolution.
    """
    H, W = height.shape

    # Erode the mask by 1 pixel so gradient is only computed where ALL
    # neighbours are valid chart texels → no bleed across seam gaps.
    from scipy.ndimage import binary_erosion
    inner_mask = binary_erosion(mask, iterations=1)

    # Neutral height outside mask so boundary gradient is zero.
    h_safe = height.copy()
    h_safe[~mask] = 0.5

    dHdy, dHdx = np.gradient(h_safe)

    # ``np.gradient`` returns [height per pixel].  Typical gradient at a pore
    # edge spanning ~10 px with height amplitude ~0.8 ≈ 0.08 / px.
    # We want a physical deflection angle of ~5–20° for natural skin:
    #   tan(10°) = 0.18  →  scale = 0.18 / 0.08 ≈ 2.2
    #   tan(20°) = 0.36  →  scale = 0.36 / 0.08 ≈ 4.5
    # Map strength [0, 1] to bump_scale [0, 6] so strength=0.5 ≈ 14° at pore edge.
    bump_scale = strength * 6.0              # ← was: strength * W  (~665 → 89°!)

    sx = -dHdx * bump_scale
    sy =  dHdy * bump_scale    # +Y = down in wgpu tangent space

    sz = np.ones((H, W), np.float32)

    nrm_len = np.sqrt(sx**2 + sy**2 + 1.0)
    nx = sx / nrm_len
    ny = sy / nrm_len
    nz = sz / nrm_len

    # Encode [−1, 1] → [0, 1]
    out = np.stack([
        (nx * 0.5 + 0.5),
        (ny * 0.5 + 0.5),
        (nz * 0.5 + 0.5),
    ], axis=2).astype(np.float32)

    # Force boundary & unmapped pixels back to flat normal
    flat = np.array([0.5, 0.5, 1.0], np.float32)
    out[~inner_mask] = flat

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
    tangent_map: np.ndarray,   # unused — kept for API compat
    bitan_map:   np.ndarray,   # unused — kept for API compat
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

    # ── Pore colour: very faint darkening at pore centres (3-D world space) ──
    wp = world_pos[rows, cols]   # (N, 3)
    x_c = wp[:, 0] / pore_cell_size
    y_c = wp[:, 1] / pore_cell_size
    z_c = wp[:, 2] / pore_cell_size
    MAX_SAMPLES = 200_000
    if N > MAX_SAMPLES:
        idx_v = rng.choice(N, MAX_SAMPLES, replace=False)
        f1_sub = _voronoi_f1_3d(x_c[idx_v], y_c[idx_v], z_c[idx_v], jitter=0.82)
        f1_sparse = np.zeros(N, np.float32)
        f1_sparse[idx_v] = f1_sub
        f1_2d_a = np.zeros((H, W), np.float32)
        np.add.at(f1_2d_a, (rows, cols), f1_sparse)
        a_sigma = max(2.0, np.sqrt(float(N) / MAX_SAMPLES) * 2.5)
        f1_2d_a = gaussian_filter(f1_2d_a, sigma=a_sigma)
        f1 = f1_2d_a[rows, cols]
    else:
        f1 = _voronoi_f1_3d(x_c, y_c, z_c, jitter=0.82)
    f1 = np.clip(f1 / 0.75, 0.0, 1.0)
    pore_dark = (1.0 - f1 ** 3) * 0.04    # very faint darkening at pore centre (was 0.07)
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
    """Roughness map in [0, 1]. Shape (H, W).

    Skin roughness for a PBR renderer:
    - 0.0 = perfect mirror  → definitely not skin
    - 0.45–0.55 = "dewy/sebum" patches (slight sheen, still clearly not plastic)
    - 0.55–0.70 = normal skin surface (diffuse with a hint of gloss)
    - 0.70–0.80 = dry / matte areas

    The old range [0.12, 0.55] produced plastic-like specular highlights.
    New range [0.45, 0.80] keeps the surface clearly organic.
    """
    rough = np.full((H, W), base, np.float32)

    # Pore centres are slightly rougher (sebum pooling vs rim)
    inverted_height = 1.0 - height
    pore_mask = np.clip(inverted_height * 1.3, 0.0, 1.0) ** 2
    rough += pore_mask * 0.05   # was 0.12 — very subtle variation

    # Dewy / sebum patches: smooth low-freq noise → lower roughness
    dewy_map = _uv_smooth_noise(H, W, sigma_px=H * 0.06, rng=rng)
    rough -= dewy_map * dewy

    # Micro variation
    micro_rough = _uv_smooth_noise(H, W, sigma_px=2.0, rng=rng)
    rough += (micro_rough - 0.5) * 0.03

    rough[~mask] = base
    np.clip(rough, 0.45, 0.80, out=rough)   # was [0.12, 0.55]
    return rough


# ══════════════════════════════════════════════════════════════════════════════
# Ambient Occlusion
# ══════════════════════════════════════════════════════════════════════════════

def _gen_ao(height: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Micro-AO baked from the height map.

    Pixels in local minima (pore centres) get a subtle darkening.  Max
    darkening is capped at 15% (AO ≥ 0.85) so pore centres look slightly
    shadowed rather than pitch-black.  The old ``* 1.8`` coefficient could
    produce AO = 0 (fully black) at the bottom of any pore.
    """
    from scipy.ndimage import maximum_filter
    pore_radius_px = max(3, int(height.shape[0] / 256))
    local_max = maximum_filter(height, size=pore_radius_px * 2 + 1)
    # strength 0.5 → max 15% darkening  (was 1.8 → 100% = pitch black craters)
    ao = 1.0 - np.clip((local_max - height) * 0.5, 0.0, 0.15)
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


def _save_overlay_maps(
    directory: str,
    prefix: str,
    uvs: np.ndarray,       # (V_new, 2) float32
    indices: np.ndarray,   # (F, 3) int32
    maps: list[tuple[np.ndarray, str]],
    verbose: bool = True,
    line_color: tuple = (255, 220, 0),
    line_alpha: int = 180,
    line_width: int = 1,
) -> None:
    """Save each map in *maps* as ``<prefix>_<name>_mesh.png`` with the UV
    wireframe drawn on top in semi-transparent yellow.
    """
    from PIL import Image, ImageDraw
    import os
    os.makedirs(directory, exist_ok=True)

    # Derive H, W from the first map
    H, W = maps[0][0].shape[:2]
    rgba_line = (*line_color, line_alpha)   # semi-transparent yellow

    # UV [0,1] → pixel coords (col = u*(W-1),  row = v*(H-1))
    uvs_px = np.empty_like(uvs)
    uvs_px[:, 0] = uvs[:, 0] * (W - 1)   # col
    uvs_px[:, 1] = uvs[:, 1] * (H - 1)   # row

    # Build the shared overlay layer (transparent, same size for all maps)
    overlay_base = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay_base, "RGBA")
    for tri in indices:
        pts = [
            (float(uvs_px[tri[0], 0]), float(uvs_px[tri[0], 1])),
            (float(uvs_px[tri[1], 0]), float(uvs_px[tri[1], 1])),
            (float(uvs_px[tri[2], 0]), float(uvs_px[tri[2], 1])),
        ]
        for j in range(3):
            draw.line([pts[j], pts[(j + 1) % 3]], fill=rgba_line, width=line_width)

    def _to_pil_rgba(arr: np.ndarray) -> Image.Image:
        if arr.ndim == 2 or (arr.ndim == 3 and arr.shape[-1] == 1):
            ch = arr[:, :, 0] if arr.ndim == 3 else arr
            return Image.fromarray(ch, mode="L").convert("RGBA")
        if arr.shape[-1] == 3:
            return Image.fromarray(arr[:, :, :3], mode="RGB").convert("RGBA")
        return Image.fromarray(arr, mode="RGBA")

    for arr, name in maps:
        img = Image.alpha_composite(_to_pil_rgba(arr), overlay_base)
        path = os.path.join(directory, f"{prefix}_{name}_mesh.png")
        img.save(path)
        if verbose:
            print(f"  saved {path}")

