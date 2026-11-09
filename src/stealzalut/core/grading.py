"""User adjustments applied on top of a fitted colour transfer.

The fitted :class:`~stealzalut.core.color_transfer.TransferModel` gives,
for every node of the LUT grid, a target colour.  This module expresses
that as a displacement in Oklab (``Δ = target − input``) and lets the user
dose it:

* **Luminance / chroma strength** — scale ``ΔL`` and ``Δab`` separately
  ("borrow the colours" vs "borrow the contrast").
* **Preserve neutrals** — cancel ``Δab`` on near-grey inputs (keeps white
  balance).
* **Skin protection** — cancel ``Δab`` on skin-tone inputs.  Lightness is
  deliberately left alone: holding skin lightness while neighbouring colours
  darken creates folds (brighter input → darker output).
* **Mix** — global amount of the whole grade.
* **Smoothing** — Gaussian smoothing of the displacement over the grid.

Because ``ΔL`` is only ever scaled by a constant, these adjustments cannot
fold a monotonic transfer.  Folds are still measured and, if adjustments
ever add some (per-zone transfers will), the displacement is smoothed
automatically (see :mod:`stealzalut.core.regularize`).

All of this works for every transfer method, and the result is still a
single standard ``.cube`` grid.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from stealzalut.core.color_transfer import TransferModel
from stealzalut.core.colorspace import gamut_compress, oklab_to_oklch, oklab_to_srgb, srgb_to_oklab
from stealzalut.core.lut_engine import generate_neutral_grid
from stealzalut.core.regularize import fold_fraction, smooth_field

F64 = NDArray[np.float64]

#: Extra folded fraction (above the raw transfer's own) that triggers auto-smoothing.
MAX_EXTRA_FOLDS = 0.005
#: Smoothing used for the first automatic attempt when the user set none.
AUTO_SMOOTHING_START = 0.5
AUTO_SMOOTHING_STEPS = 3

# Skin-tone window in Oklch (calibrated on light to dark skin samples:
# h ≈ 37°–59°, C ≈ 0.06–0.10, L ≈ 0.38–0.89).  Full weight inside the inner
# bounds, fading to zero at the outer bounds.
SKIN_HUE_CENTER = 57.5
SKIN_HUE_FULL = 20.0  # half-width with full weight (37.5°–77.5°)
SKIN_HUE_FADE = 30.0  # half-width where the weight reaches 0 (27.5°–87.5°)
SKIN_CHROMA = (0.015, 0.035, 0.15, 0.20)  # fade-in start/end, fade-out start/end
SKIN_LIGHTNESS = (0.25, 0.35, 0.92, 0.98)

# Near-neutral window: full protection below C = 0.02, none above C = 0.06.
NEUTRAL_CHROMA = (0.02, 0.06)


@dataclass(frozen=True, slots=True)
class GradingParams:
    """User adjustments.  The defaults reproduce the raw transfer exactly."""

    luminance_strength: float = 1.0
    chroma_strength: float = 1.0
    preserve_neutrals: float = 0.0
    skin_protection: float = 0.0
    mix: float = 1.0
    smoothing: float = 0.0

    @property
    def is_passthrough(self) -> bool:
        """True when the adjustments leave the raw transfer untouched."""
        return self == GradingParams()


@dataclass(frozen=True, slots=True)
class GridResult:
    """A baked grid plus diagnostics for the UI.

    Attributes:
        grid: Final LUT grid ``(size³, 3)`` in ``[0, 1]``.
        fold_fraction: Folded fraction of the final grid (see
            :func:`~stealzalut.core.regularize.fold_fraction`).
        smoothing_used: Smoothing actually applied (may exceed the user's).
        auto_smoothed: True when smoothing was raised automatically.
    """

    grid: F64
    fold_fraction: float
    smoothing_used: float
    auto_smoothed: bool


# ── Masks ────────────────────────────────────────────────────────────────────


def smoothstep(x: F64, edge0: float, edge1: float) -> F64:
    """Smooth 0→1 transition between *edge0* and *edge1*."""
    t = np.clip((x - edge0) / (edge1 - edge0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _window(x: F64, bounds: tuple[float, float, float, float]) -> F64:
    in0, in1, out0, out1 = bounds
    return smoothstep(x, in0, in1) * (1.0 - smoothstep(x, out0, out1))


def neutral_mask(lch: F64) -> F64:
    """Weight in ``[0, 1]``: 1 for greys, 0 for clearly coloured inputs."""
    return 1.0 - smoothstep(lch[:, 1], *NEUTRAL_CHROMA)


def skin_mask(lch: F64) -> F64:
    """Weight in ``[0, 1]``: 1 for typical skin tones, 0 far from them."""
    hue_dist = np.abs((lch[:, 2] - SKIN_HUE_CENTER + 180.0) % 360.0 - 180.0)
    hue_w = 1.0 - smoothstep(hue_dist, SKIN_HUE_FULL, SKIN_HUE_FADE)
    return hue_w * _window(lch[:, 1], SKIN_CHROMA) * _window(lch[:, 0], SKIN_LIGHTNESS)


# ── Grid construction ────────────────────────────────────────────────────────


def _sigma_cells(smoothing: float, size: int) -> float:
    """Smoothing slider value → Gaussian sigma in cells (resolution independent)."""
    return smoothing * (size - 1) / 32.0


def adjust_displacement(lab_in: F64, lab_target: F64, params: GradingParams) -> F64:
    """Apply strengths, protections and mix to the Oklab displacement."""
    delta = lab_target - lab_in
    delta[:, 0] *= params.luminance_strength
    delta[:, 1:] *= params.chroma_strength

    lch_in = oklab_to_oklch(lab_in)
    keep_neutral = params.preserve_neutrals * neutral_mask(lch_in)
    keep_skin = params.skin_protection * skin_mask(lch_in)
    delta[:, 1:] *= ((1.0 - keep_neutral) * (1.0 - keep_skin))[:, None]

    delta *= params.mix
    return delta


def build_grid(
    model: TransferModel,
    size: int,
    params: GradingParams | None = None,
) -> GridResult:
    """Bake *model* onto a *size*³ grid with the user's adjustments.

    With default :class:`GradingParams` the grid is exactly ``model.bake(size)``
    for plain transfers.  Blended transfers (zones) always go through the
    fold check, using their :meth:`~TransferModel.fold_reference` as baseline.

    Args:
        model: Fitted transfer model.
        size: LUT resolution per axis.
        params: Adjustments (defaults: none).

    Returns:
        The grid and its diagnostics.
    """
    params = params or GradingParams()
    base = generate_neutral_grid(size)
    raw = model.apply(base)
    lab_raw = srgb_to_oklab(raw)

    baseline = model.fold_reference()
    if baseline is model:
        raw_folds = fold_fraction(lab_raw[:, 0], size)
        if params.is_passthrough:
            return GridResult(raw, raw_folds, 0.0, auto_smoothed=False)
    else:
        raw_folds = fold_fraction(srgb_to_oklab(baseline.apply(base))[:, 0], size)

    lab_in = srgb_to_oklab(base)
    delta = adjust_displacement(lab_in, lab_raw, params)

    smoothing = params.smoothing
    auto = False
    for attempt in range(AUTO_SMOOTHING_STEPS + 1):
        lab_out = lab_in + smooth_field(delta, size, _sigma_cells(smoothing, size))
        folds = fold_fraction(lab_out[:, 0], size)
        if folds <= raw_folds + MAX_EXTRA_FOLDS or attempt == AUTO_SMOOTHING_STEPS:
            break
        smoothing = max(smoothing * 2.0, AUTO_SMOOTHING_START)
        auto = True

    grid = oklab_to_srgb(gamut_compress(lab_out))
    return GridResult(grid, folds, smoothing, auto_smoothed=auto)
