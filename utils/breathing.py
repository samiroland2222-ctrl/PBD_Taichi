"""
BreathingController – sinusoidal breathing animation for the torso simulation.

Two anatomical regions are driven each cycle:
  Chest   – ribcage expands up (+y) and out (+z) on inhale; collapses on exhale.
  Abdomen – anterior lower-torso skin protrudes out (+z, slight −y) on inhale
            as the diaphragm descends; retracts on exhale.

The clavicles and shoulders track the top of the ribcage (sternoclavicular joint)
so they rise/forward-tilt on inhale and fall/retract on exhale.

Waveform
--------
The breathing cycle follows the standard asymmetric lung-volume curve::

    LV(t) = 0.5 * (1 − cos(2π * t / T))

remapped through a piecewise linear ramp so inhale occupies ``inhale_fraction``
of the full period and exhale fills the remainder.  A cubic smoothstep removes
the kink at the midpoint.

Usage
-----
    # 1. Create once (after skeleton and torso are built)
    params = BreathingParams(rate_bpm=15.0, chest_amplitude=0.008)
    breath_ctrl = BreathingController(params)

    # 2. Snapshot rest pose (call ONCE after skel.set_ribcage_mesh)
    breath_ctrl.snapshot_ribcage_rest(skel)

    # 3. Per-frame, BEFORE skel.update():
    if breathing_on:
        breath_ctrl.apply_to_ribcage(skel, sim_time)

    # 4. Per-frame, AFTER torso.update_kinematic_skin():
    breath_ctrl.apply_abdomen(torso, sim_time)
"""
from __future__ import annotations

import dataclasses
import numpy as np


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class BreathingParams:
    """All tuneable parameters for the breathing animation."""

    rate_bpm:            float = 15.0   # breaths per minute (normal rest: 12–20)
    chest_amplitude:     float = 0.008  # peak rib-cage expansion [m]  (~8 mm)
    abdomen_amplitude:   float = 0.012  # peak abdominal protrusion [m] (~12 mm)
    inhale_fraction:     float = 0.40   # fraction of cycle spent inhaling (40 %)

    # Chest direction: rib cage rises slightly (+y) and expands anteriorly (+z)
    chest_direction:     tuple = (0.0, 0.6, 0.4)

    # Abdomen direction: belly protrudes anteriorly (+z), pushed slightly inferior (−y)
    abdomen_direction:   tuple = (0.0, -0.15, 0.85)

    # Shoulder / clavicle rise on inhale: sternoclavicular joint tracks ribcage top.
    # shoulder_rise_rad is an additional clavicle-pitch rotation (superior elevation).
    shoulder_rise_rad:   float = float(np.deg2rad(3.0))   # ≈ 3° peak elevation


# ---------------------------------------------------------------------------
# Controller
# ---------------------------------------------------------------------------

class BreathingController:
    """Drives the skeleton ribcage, clavicles, and abdominal skin springs each frame.

    Attributes
    ----------
    params  : BreathingParams
    enabled : bool  – set to False to freeze breathing (ribcage returned to rest)
    """

    def __init__(self, params: BreathingParams | None = None):
        self.params  = params if params is not None else BreathingParams()
        self.enabled = True

        # Populated by snapshot_ribcage_rest()
        self._ribcage_v_local_rest: np.ndarray | None = None   # (N, 3) float32
        self._rib_weights:          np.ndarray | None = None   # (N,)   float32

        # Clavicle rest state (snapshotted once)
        self._clavicle_l_offset_rest:  np.ndarray | None = None  # (3,) float32
        self._clavicle_r_offset_rest:  np.ndarray | None = None  # (3,) float32
        self._clavicle_pitch_l_rest:   float | None      = None
        self._clavicle_pitch_r_rest:   float | None      = None
        # Weight of the SC joint in the ribcage weight scheme (≈1 for top of sternum)
        self._sc_weight: float = 1.0

    # ------------------------------------------------------------------
    # One-time setup
    # ------------------------------------------------------------------

    def snapshot_ribcage_rest(self, skel) -> None:
        """Capture the ribcage rest shape and clavicle rest pose.

        Call ONCE after ``skel.set_ribcage_mesh()``.  The controller then
        uses these snapshots as the baseline for each frame's offset.
        """
        if not hasattr(skel, '_ribcage_v_local'):
            return
        self._ribcage_v_local_rest = skel._ribcage_v_local.copy().astype(np.float32)

        # Weight: 1 at the superior (highest y) ribcage verts, 0 at the
        # inferior end.  Upper ribs expand more than lower ribs on inhale.
        y     = self._ribcage_v_local_rest[:, 1]
        y_min = float(y.min())
        y_rng = max(float(y.max()) - y_min, 1e-6)
        self._rib_weights = np.clip((y - y_min) / y_rng, 0.0, 1.0).astype(np.float32)

        # ── Snapshot clavicle rest offsets and pitch ──────────────────────
        if hasattr(skel, '_clavicle_l_offset'):
            self._clavicle_l_offset_rest = skel._clavicle_l_offset.copy().astype(np.float32)
            self._clavicle_r_offset_rest = skel._clavicle_r_offset.copy().astype(np.float32)
        if hasattr(skel, 'clavicle_left_pitch'):
            self._clavicle_pitch_l_rest = float(skel.clavicle_left_pitch)
            self._clavicle_pitch_r_rest = float(skel.clavicle_right_pitch)

        # Compute the weight that the SC joint y-level would receive.
        # The SC joint sits at clavicle_l_offset[1] in ribcage-local coords.
        # This drives how much the clavicle offset tracks ribcage expansion.
        if self._clavicle_l_offset_rest is not None:
            sc_y_local = float(self._clavicle_l_offset_rest[1])
            self._sc_weight = float(np.clip((sc_y_local - y_min) / y_rng, 0.0, 1.0))
        else:
            self._sc_weight = 1.0   # assume top of ribcage

    # ------------------------------------------------------------------
    # Waveform
    # ------------------------------------------------------------------

    @staticmethod
    def _smoothstep(t: float) -> float:
        """Cubic Hermite: 3t² − 2t³, t ∈ [0, 1]."""
        t = float(np.clip(t, 0.0, 1.0))
        return t * t * (3.0 - 2.0 * t)

    def breath_phase(self, t: float) -> float:
        """Return the breath phase ∈ [0, 1] at simulation time *t* [s].

        0 = end-exhale (lungs empty), peaks at 1.0 at end-of-inhale, then
        returns to 0 over the exhale portion of the cycle.
        """
        p      = self.params
        period = 60.0 / max(float(p.rate_bpm), 1e-3)
        t_norm = float(t) % period / period      # [0, 1) within one cycle
        if t_norm < p.inhale_fraction:
            raw = t_norm / max(p.inhale_fraction, 1e-6)
        else:
            raw = 1.0 - (t_norm - p.inhale_fraction) / max(1.0 - p.inhale_fraction, 1e-6)
        return self._smoothstep(raw)

    # ------------------------------------------------------------------
    # Per-frame: chest / ribcage
    # ------------------------------------------------------------------

    def apply_to_ribcage(self, skel, t: float) -> None:
        """Offset ``skel._ribcage_v_local`` to simulate chest expansion, and
        track the clavicles / shoulders so they rise with the sternum.

        **Must be called BEFORE** ``skel.update()`` each frame.

        When ``enabled`` is False the ribcage and clavicles are restored to
        their rest shapes.
        """
        if self._ribcage_v_local_rest is None or not hasattr(skel, '_ribcage_v_local'):
            return

        if not self.enabled:
            skel._ribcage_v_local = self._ribcage_v_local_rest.copy()
            # Restore clavicle offsets and pitch
            if self._clavicle_l_offset_rest is not None:
                skel._clavicle_l_offset = self._clavicle_l_offset_rest.copy()
                skel._clavicle_r_offset = self._clavicle_r_offset_rest.copy()
            if self._clavicle_pitch_l_rest is not None:
                skel.clavicle_left_pitch  = self._clavicle_pitch_l_rest
                skel.clavicle_right_pitch = self._clavicle_pitch_r_rest
            return

        p     = self.params
        phase = self.breath_phase(t)

        d = np.array(p.chest_direction, dtype=np.float32)
        n = float(np.linalg.norm(d))
        if n > 1e-8:
            d /= n

        # Per-vert ribcage offset: upper ribs (weight=1) expand by full amplitude,
        # lower ribs (weight≈0) barely move.
        disp   = phase * float(p.chest_amplitude) * d          # (3,)
        offset = (self._rib_weights[:, None] * disp[None, :])  # (N, 3)
        skel._ribcage_v_local = (self._ribcage_v_local_rest + offset).astype(np.float32)

        # ── Drive clavicle / SC joint to track the sternum ────────────────
        # The sternoclavicular (SC) joint sits at the top of the sternum.
        # As the ribcage expands (d direction), the SC joint moves by the
        # same displacement weighted by its position in the ribcage weight
        # scheme (_sc_weight ≈ 1 for top ribs).
        if self._clavicle_l_offset_rest is None:
            return

        sc_disp = phase * float(p.chest_amplitude) * d * self._sc_weight  # (3,)

        # Update the medial (sternal) pivot offsets for both clavicles.
        # The x-component of _clavicle_*_offset encodes medial position;
        # only y (superior) and z (anterior) components of sc_disp are
        # anatomically meaningful here.  The x-component of d is 0 (symmetric
        # chest direction) so sc_disp[0] ≈ 0 — safe to add to both sides.
        skel._clavicle_l_offset = (self._clavicle_l_offset_rest + sc_disp).astype(np.float32)
        skel._clavicle_r_offset = (self._clavicle_r_offset_rest + sc_disp).astype(np.float32)

        # Additional superior pitch elevation of the clavicle on inhale
        if self._clavicle_pitch_l_rest is not None:
            pitch_delta = phase * float(p.shoulder_rise_rad)
            skel.clavicle_left_pitch  = self._clavicle_pitch_l_rest  + pitch_delta
            skel.clavicle_right_pitch = self._clavicle_pitch_r_rest + pitch_delta

    # ------------------------------------------------------------------
    # Per-frame: abdomen
    # ------------------------------------------------------------------

    def apply_abdomen(self, torso, t: float) -> None:
        """Push abdominal skin spring targets outward for the current phase.

        Requires ``torso.abdomen_skin_springs`` (a
        ``KinematicSkinSpringConstraint``) and
        ``torso._abdomen_rest_positions`` (created during
        ``UnifiedTorso.__init__``).

        **Must be called AFTER** ``torso.update_kinematic_skin()`` each frame.
        """
        springs = getattr(torso, 'abdomen_skin_springs',    None)
        rest_np = getattr(torso, '_abdomen_rest_positions', None)
        if springs is None or rest_np is None or springs.n == 0:
            return

        if not self.enabled:
            springs.update_targets(rest_np.astype(np.float32))
            return

        p     = self.params
        phase = self.breath_phase(t)

        d = np.array(p.abdomen_direction, dtype=np.float32)
        n = float(np.linalg.norm(d))
        if n > 1e-8:
            d /= n

        disp = phase * float(p.abdomen_amplitude) * d          # (3,)

        # Optional per-vert weight (lower → more abdominal → more movement)
        weights = getattr(torso, '_abdomen_weights', None)
        if weights is not None:
            new_targets = rest_np + weights[:, None] * disp[None, :]
        else:
            new_targets = rest_np + disp[None, :]

        springs.update_targets(new_targets.astype(np.float32))

