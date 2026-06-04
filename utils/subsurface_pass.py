"""
subsurface_pass.py
------------------
Skin screen-space subsurface scattering (SSS) post-processing pass for pygfx.

Algorithm
~~~~~~~~~
Implements a *separable Gaussian blur* with **per-channel radii** to
approximate the skin diffusion profile described by Jimenez et al. (2015):

    • Red channel  – σ_r  (largest spread, ~3–5 mm in real skin)
    • Green channel – σ_g
    • Blue channel  – σ_b (smallest spread)

The pass executes three full-screen stages per frame:

    1. H-blur  : ``color_tex`` → ``_temp_h``
       Horizontal separable Gaussian with per-channel weights.

    2. V-blur  : ``_temp_h``  → ``_temp_v``
       Vertical separable Gaussian with the same weights.

    3. Composite: mix(``color_tex``, ``_temp_v``, strength) → ``target_tex``
       Specular highlights are protected by a luminance mask so they are
       not smeared by the blur.

Usage
~~~~~
::
    from PBD_Taichi.utils.subsurface_pass import SkinSSSPass

    sss = SkinSSSPass(sss_strength=0.30, sigma_r=5.0, sigma_g=3.0, sigma_b=1.5)
    renderer.effect_passes = (sss,)

    # Tune at runtime (e.g. from a GUI slider):
    sss.sss_strength = 0.25
    sss.enabled = False   # bypass pass without removing it
"""

from __future__ import annotations

import numpy as np
import wgpu

from pygfx.renderers.wgpu.engine.effectpasses import (
    EffectPass,
    FullQuadPass,
)
from pygfx.renderers.wgpu.engine.shared import get_shared


# ──────────────────────────────────────────────────────────────────────────────
# WGSL shaders
# ──────────────────────────────────────────────────────────────────────────────

# Separable Gaussian blur with per-channel σ.
# Uniform layout (u_effect):
#   sigma_r      f32 – Gaussian σ in pixels for the R channel
#   sigma_g      f32 – Gaussian σ for G
#   sigma_b      f32 – Gaussian σ for B
#   is_horizontal i32 – 1 = horizontal pass, 0 = vertical pass
#   half_taps    i32 – kernel half-width in pixels (= ceil(3 * sigma_r))
#
# Inputs (auto-declared by FullQuadPass._create_pipeline):
#   colorTex     – source texture (binding 2)
#   texSampler   – linear sampler (binding 1)
_BLUR_WGSL = """
@fragment
fn fs_main(varyings: Varyings) -> @location(0) vec4<f32> {
    let tc      = varyings.texCoord;
    let ts      = vec2<f32>(textureDimensions(colorTex));

    // One-pixel step in the blur direction
    var step: vec2<f32>;
    if (u_effect.is_horizontal != 0) {
        step = vec2<f32>(1.0 / ts.x, 0.0);
    } else {
        step = vec2<f32>(0.0, 1.0 / ts.y);
    }

    let half_n = u_effect.half_taps;
    let sr     = u_effect.sigma_r;
    let sg     = u_effect.sigma_g;
    let sb     = u_effect.sigma_b;

    // Accumulate weighted sum per channel
    var rgb_sum     = vec3<f32>(0.0);
    var rgb_weights = vec3<f32>(0.0);

    // Dynamic loop – wgpu supports uniform-driven bounds
    for (var i: i32 = -half_n; i <= half_n; i++) {
        let d  = f32(i);
        let d2 = d * d;

        // Per-channel Gaussian weights  w = exp(-d² / 2σ²)
        let wr = exp(-d2 / (2.0 * sr * sr));
        let wg = exp(-d2 / (2.0 * sg * sg));
        let wb = exp(-d2 / (2.0 * sb * sb));

        let s = textureSample(colorTex, texSampler, tc + step * d).rgb;

        rgb_sum     += s * vec3<f32>(wr, wg, wb);
        rgb_weights += vec3<f32>(wr, wg, wb);
    }

    return vec4<f32>(rgb_sum / rgb_weights, 1.0);
}
"""

# Composite: mix original + blurred, protect specular highlights.
# Uniform layout (u_effect):
#   time         f32 – (inherited from EffectPass, unused here)
#   sss_strength f32 – blend factor [0, 1]
#   highlight_lo f32 – luminance below which full SSS is applied (default 0.55)
#   highlight_hi f32 – luminance above which SSS is zeroed (default 0.80)
#
# Inputs:
#   colorTex    – original rendered frame (binding 2)
#   blurredTex  – V-blurred SSS result   (binding 3)
_COMPOSITE_WGSL = """
@fragment
fn fs_main(varyings: Varyings) -> @location(0) vec4<f32> {
    let tc       = varyings.texCoord;
    let original = textureSample(colorTex,   texSampler, tc);
    let blurred  = textureSample(blurredTex, texSampler, tc);

    // Luminance of original pixel (BT.709 coefficients)
    let lum = dot(original.rgb, vec3<f32>(0.2126, 0.7152, 0.0722));

    // Ramp that protects bright specular highlights from being blurred
    // 0.0 → full SSS applied; 1.0 → no SSS (specular region)
    let hi_lo = u_effect.highlight_lo;
    let hi_hi = u_effect.highlight_hi;
    let specular_mask = saturate((lum - hi_lo) / max(hi_hi - hi_lo, 0.001));

    let strength = u_effect.sss_strength * (1.0 - specular_mask);

    // Blend toward SSS-blurred result
    let result = mix(original.rgb, blurred.rgb, strength);
    return vec4<f32>(result, original.a);
}
"""


# ──────────────────────────────────────────────────────────────────────────────
# Inner blur pass (used for both H and V)
# ──────────────────────────────────────────────────────────────────────────────

class _BlurPass(FullQuadPass):
    """Horizontal or vertical separable Gaussian blur with per-channel σ."""

    uniform_type = dict(
        sigma_r      = "f4",
        sigma_g      = "f4",
        sigma_b      = "f4",
        is_horizontal = "i4",
        half_taps    = "i4",
    )

    wgsl = _BLUR_WGSL


# ──────────────────────────────────────────────────────────────────────────────
# SkinSSSPass
# ──────────────────────────────────────────────────────────────────────────────

class SkinSSSPass(EffectPass):
    """Screen-space skin SSS via separable Gaussian blur.

    Parameters
    ----------
    sss_strength : float
        Blend factor between original (0.0) and blurred (1.0).  Default 0.30.
        Values around 0.20–0.40 look realistic; >0.6 looks milky.
    sigma_r : float
        Gaussian σ in screen pixels for the Red channel.  Default 5.0.
        Red spreads farthest (haemoglobin absorption is low in red).
    sigma_g : float
        Gaussian σ for the Green channel.  Default 3.0.
    sigma_b : float
        Gaussian σ for the Blue channel.  Default 1.5.
        Blue spreads least (Rayleigh-like scattering, quickly absorbed).
    highlight_lo : float
        Luminance threshold below which SSS is applied at full strength.
        Default 0.55.
    highlight_hi : float
        Luminance threshold above which SSS is completely suppressed
        (protects specular highlights).  Default 0.80.
    """

    uniform_type = dict(
        EffectPass.uniform_type,   # time: f4
        sss_strength  = "f4",
        highlight_lo  = "f4",
        highlight_hi  = "f4",
    )

    # The composite is run by the EffectPass machinery; WGSL reads
    # both 'colorTex' (original) and 'blurredTex' (V-blur output).
    wgsl = _COMPOSITE_WGSL

    # ------------------------------------------------------------------ init

    def __init__(
        self,
        sss_strength : float = 0.30,
        sigma_r      : float = 5.0,
        sigma_g      : float = 3.0,
        sigma_b      : float = 1.5,
        highlight_lo : float = 0.55,
        highlight_hi : float = 0.80,
    ) -> None:
        super().__init__()

        # Parameters (validated via properties)
        self.sigma_r       = sigma_r
        self.sigma_g       = sigma_g
        self.sigma_b       = sigma_b
        self.sss_strength  = sss_strength
        self.highlight_lo  = highlight_lo
        self.highlight_hi  = highlight_hi

        # Shared H+V blur sub-pass
        self._blur_pass = _BlurPass()

        # Ping-pong intermediate textures (allocated lazily)
        self._temp_h      : wgpu.GPUTexture | None      = None
        self._view_h      : wgpu.GPUTextureView | None  = None
        self._temp_v      : wgpu.GPUTexture | None      = None
        self._view_v      : wgpu.GPUTextureView | None  = None
        self._last_fmt    : str  | None = None
        self._last_size   : tuple| None = None

    # ------------------------------------------------------------------ properties

    @property
    def sss_strength(self) -> float:
        return float(self._uniform_data["sss_strength"])

    @sss_strength.setter
    def sss_strength(self, v: float) -> None:
        self._uniform_data["sss_strength"] = float(np.clip(v, 0.0, 1.0))

    @property
    def sigma_r(self) -> float:
        return self._sigma_r

    @sigma_r.setter
    def sigma_r(self, v: float) -> None:
        self._sigma_r = max(0.5, float(v))

    @property
    def sigma_g(self) -> float:
        return self._sigma_g

    @sigma_g.setter
    def sigma_g(self, v: float) -> None:
        self._sigma_g = max(0.5, float(v))

    @property
    def sigma_b(self) -> float:
        return self._sigma_b

    @sigma_b.setter
    def sigma_b(self, v: float) -> None:
        self._sigma_b = max(0.5, float(v))

    @property
    def highlight_lo(self) -> float:
        return float(self._uniform_data["highlight_lo"])

    @highlight_lo.setter
    def highlight_lo(self, v: float) -> None:
        self._uniform_data["highlight_lo"] = float(np.clip(v, 0.0, 1.0))

    @property
    def highlight_hi(self) -> float:
        return float(self._uniform_data["highlight_hi"])

    @highlight_hi.setter
    def highlight_hi(self, v: float) -> None:
        self._uniform_data["highlight_hi"] = float(np.clip(v, 0.0, 1.0))

    # ------------------------------------------------------------------ internals

    def _ensure_temps(self, source_tex: wgpu.GPUTexture) -> None:
        """Allocate (or reallocate) ping-pong textures to match the source."""
        w, h = source_tex.size[0], source_tex.size[1]
        fmt  = source_tex.format

        # Skip if already the right size + format
        if self._last_size == (w, h) and self._last_fmt == fmt:
            return

        device = get_shared().device
        usage  = (wgpu.TextureUsage.TEXTURE_BINDING |
                  wgpu.TextureUsage.RENDER_ATTACHMENT)

        self._temp_h = device.create_texture(
            size=(w, h, 1), format=fmt, usage=usage)
        self._view_h = self._temp_h.create_view()

        self._temp_v = device.create_texture(
            size=(w, h, 1), format=fmt, usage=usage)
        self._view_v = self._temp_v.create_view()

        self._last_size = (w, h)
        self._last_fmt  = fmt

    def _set_blur_uniforms(self, is_horizontal: int) -> None:
        """Write σ / direction into the blur sub-pass uniform buffer."""
        sigma_r   = self._sigma_r
        sigma_g   = self._sigma_g
        sigma_b   = self._sigma_b
        half_taps = int(np.ceil(2.5 * sigma_r))   # 2.5σ ≈ 99.4 % of Gaussian mass

        bp = self._blur_pass
        bp._uniform_data["sigma_r"]       = sigma_r
        bp._uniform_data["sigma_g"]       = sigma_g
        bp._uniform_data["sigma_b"]       = sigma_b
        bp._uniform_data["is_horizontal"] = is_horizontal
        bp._uniform_data["half_taps"]     = half_taps

    # ------------------------------------------------------------------ render

    def render(
        self,
        command_encoder,
        color_tex  : wgpu.GPUTextureView,
        depth_tex  : wgpu.GPUTextureView | None,
        target_tex : wgpu.GPUTextureView,
    ) -> None:
        """Execute H-blur → V-blur → composite.

        Parameters
        ----------
        command_encoder
            Active wgpu command encoder (provided by pygfx).
        color_tex
            The rendered scene (output of the previous effect pass or pygfx
            renderer). Used as both the blur source and the composite original.
        depth_tex
            Depth texture (not used; present for API compatibility).
        target_tex
            Final output texture view (canvas surface or next pass input).
        """
        if not self._enabled:
            # Bypass: copy original → target by compositing with strength=0.
            # Temporarily zero the strength, then restore it so the user's
            # setting is not lost.
            import time as _time
            saved_strength = float(self._uniform_data["sss_strength"])
            self._uniform_data["time"]         = _time.perf_counter()
            self._uniform_data["sss_strength"] = 0.0
            super(EffectPass, self).render(
                command_encoder,
                colorTex   = color_tex,
                blurredTex = color_tex,   # same src, strength=0 → pass-through
                targetTex  = target_tex,
            )
            self._uniform_data["sss_strength"] = saved_strength
            return

        # ── Ensure ping-pong textures exist ──────────────────────────────
        self._ensure_temps(color_tex.texture)

        # ── Pass 1: Horizontal blur  color_tex → _view_h ─────────────────
        self._set_blur_uniforms(is_horizontal=1)
        self._blur_pass.render(
            command_encoder,
            colorTex  = color_tex,
            targetTex = self._view_h,
        )

        # ── Pass 2: Vertical blur  _view_h → _view_v ─────────────────────
        self._set_blur_uniforms(is_horizontal=0)
        self._blur_pass.render(
            command_encoder,
            colorTex  = self._view_h,
            targetTex = self._view_v,
        )

        # ── Pass 3: Composite  mix(original, blurred) → target ───────────
        # Set composite uniforms (sss_strength/highlight_lo/hi already set via
        # properties; set 'time' manually since we bypass EffectPass.render()).
        import time as _time
        self._uniform_data["time"] = _time.perf_counter()

        # Call FullQuadPass.render() directly (bypassing EffectPass.render()
        # which would only pass colorTex without blurredTex).
        super(EffectPass, self).render(
            command_encoder,
            colorTex   = color_tex,      # original (source binding 2)
            blurredTex = self._view_v,   # fully blurred (source binding 3)
            targetTex  = target_tex,     # output (render attachment)
        )





