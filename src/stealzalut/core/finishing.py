"""Finishing tools applied after the transfer, in Oklab / Oklch.

* **Colour wheels** (shadows / midtones / highlights): a lightness offset
  and a tint, weighted by a Bernstein basis on L
  (``(1−L)²``, ``2L(1−L)``, ``L²``, summing to 1) — the lift / gamma /
  gain a Resolve user expects, without the instabilities of ASC-CDL.
* **Tone curve** on L only (monotone cubic, never oscillates): hue and
  chroma are untouched, so no colour shift.
* **Per-colour bands** (hue vs hue / saturation / lightness): Gaussian
  bumps around a hue, faded out on greys whose hue means nothing.
* **Saturation / vibrance**: vibrance boosts dull colours more than
  saturated ones.

Every setting is a value in ``[-1, 1]`` (``0`` = neutral), so a recipe
stays readable and the UI maps sliders directly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import cast

import numpy as np
from numpy.typing import NDArray

from stealzalut.core.colorspace import oklab_to_oklch, oklch_to_oklab, srgb_to_oklab

F64 = NDArray[np.float64]

#: Oklab lightness offset of a wheel at ``lightness = ±1``.
WHEEL_LIGHTNESS_RANGE = 0.15
#: Oklab chroma of a wheel's tint at ``amount = 1``.
WHEEL_TINT_MAX = 0.05
#: Hue rotation (degrees) of a colour band at ``hue = ±1``.
BAND_HUE_RANGE = 30.0
#: Lightness offset of a colour band at ``lightness = ±1``.
BAND_LIGHTNESS_RANGE = 0.1
#: Gaussian width (degrees) of the colour bands.
BAND_WIDTH = 22.0
#: Chroma below which hue-based tools fade out (greys).
GREY_CHROMA = (0.01, 0.04)
#: Chroma above which vibrance has no effect.
VIBRANCE_CHROMA = 0.15

IDENTITY_CURVE: tuple[tuple[float, float], ...] = ((0.0, 0.0), (1.0, 1.0))

#: Reference colours of the per-colour bands (name, sRGB 0–255).
_BAND_COLOURS: tuple[tuple[str, tuple[int, int, int]], ...] = (
    ("Rouges", (230, 30, 40)),
    ("Oranges", (245, 130, 30)),
    ("Jaunes", (240, 215, 40)),
    ("Verts", (60, 180, 60)),
    ("Cyans", (30, 200, 210)),
    ("Bleus", (40, 90, 230)),
    ("Violets", (140, 60, 220)),
    ("Magentas", (225, 50, 170)),
)


def _band_hue(rgb: tuple[int, int, int]) -> float:
    lab = srgb_to_oklab(np.array([rgb], dtype=np.float64) / 255.0)
    return round(float(oklab_to_oklch(lab)[0, 2]), 1)


#: (name, centre hue in degrees, display colour) of the per-colour bands.
BANDS: tuple[tuple[str, float, tuple[int, int, int]], ...] = tuple(
    (name, _band_hue(rgb), rgb) for name, rgb in _BAND_COLOURS
)


@dataclass(frozen=True, slots=True)
class Wheel:
    """One colour wheel.

    Attributes:
        hue: Direction of the tint, degrees (Oklch hue).
        amount: Strength of the tint, ``[0, 1]``.
        lightness: Lightness offset, ``[-1, 1]``.
    """

    hue: float = 0.0
    amount: float = 0.0
    lightness: float = 0.0

    @property
    def is_neutral(self) -> bool:
        return self.amount == 0.0 and self.lightness == 0.0

    @property
    def tint_ab(self) -> F64:
        """Tint as an Oklab ``(a, b)`` offset."""
        rad = np.deg2rad(self.hue)
        return self.amount * WHEEL_TINT_MAX * np.array([np.cos(rad), np.sin(rad)])


@dataclass(frozen=True, slots=True)
class HueBand:
    """Adjustment of the colours around one hue.

    Attributes:
        center: Centre hue, degrees.
        hue: Hue rotation, ``[-1, 1]`` (× :data:`BAND_HUE_RANGE`).
        saturation: Chroma change, ``[-1, 1]`` (factor ``1 + saturation``).
        lightness: Lightness offset, ``[-1, 1]`` (× :data:`BAND_LIGHTNESS_RANGE`).
        width: Gaussian width, degrees.
    """

    center: float
    hue: float = 0.0
    saturation: float = 0.0
    lightness: float = 0.0
    width: float = BAND_WIDTH

    @property
    def is_neutral(self) -> bool:
        return self.hue == 0.0 and self.saturation == 0.0 and self.lightness == 0.0


def default_bands() -> tuple[HueBand, ...]:
    """The eight neutral per-colour bands (reds … magentas)."""
    return tuple(HueBand(center=hue) for _, hue, _ in BANDS)


@dataclass(frozen=True, slots=True)
class Finishing:
    """All finishing settings; the defaults change nothing."""

    shadows: Wheel = Wheel()
    midtones: Wheel = Wheel()
    highlights: Wheel = Wheel()
    tone_curve: tuple[tuple[float, float], ...] = IDENTITY_CURVE
    bands: tuple[HueBand, ...] = field(default_factory=default_bands)
    saturation: float = 0.0
    vibrance: float = 0.0

    @property
    def is_identity(self) -> bool:
        return (
            self.shadows.is_neutral
            and self.midtones.is_neutral
            and self.highlights.is_neutral
            and curve_is_identity(self.tone_curve)
            and all(band.is_neutral for band in self.bands)
            and self.saturation == 0.0
            and self.vibrance == 0.0
        )


# ── Tone curve ───────────────────────────────────────────────────────────────


def normalize_curve(points: tuple[tuple[float, float], ...]) -> tuple[tuple[float, float], ...]:
    """Sort, clip to ``[0, 1]``, keep one point per x, make y non-decreasing
    and include both ends."""
    pts = sorted((min(max(x, 0.0), 1.0), min(max(y, 0.0), 1.0)) for x, y in points)
    unique: list[tuple[float, float]] = []
    for x, y in pts:
        if unique and abs(unique[-1][0] - x) < 1e-6:
            unique[-1] = (x, y)
        else:
            unique.append((x, y))
    if not unique or unique[0][0] > 0.0:
        unique.insert(0, (0.0, unique[0][1] if unique else 0.0))
    if unique[-1][0] < 1.0:
        unique.append((1.0, unique[-1][1]))
    monotone: list[tuple[float, float]] = []
    for x, y in unique:
        monotone.append((x, max(y, monotone[-1][1]) if monotone else y))
    return tuple(monotone)


def curve_is_identity(points: tuple[tuple[float, float], ...]) -> bool:
    return all(abs(x - y) < 1e-9 for x, y in points)


def eval_curve(points: tuple[tuple[float, float], ...], x: F64) -> F64:
    """Monotone cubic (Fritsch–Carlson / PCHIP) through *points*.

    Monotone points give a monotone curve with no overshoot; outside
    ``[x0, xn]`` the end values are held.
    """
    pts = normalize_curve(points)
    xs = np.array([p[0] for p in pts])
    ys = np.array([p[1] for p in pts])
    if len(xs) == 2:
        return cast("F64", np.interp(x, xs, ys))

    h = np.diff(xs)
    delta = np.diff(ys) / h
    slopes = np.zeros_like(ys)
    # Interior: weighted harmonic mean, zero at extrema / flat parts
    for k in range(1, len(xs) - 1):
        if delta[k - 1] * delta[k] > 0:
            w1 = 2 * h[k] + h[k - 1]
            w2 = h[k] + 2 * h[k - 1]
            slopes[k] = (w1 + w2) / (w1 / delta[k - 1] + w2 / delta[k])
    slopes[0] = _end_slope(h[0], h[1], delta[0], delta[1])
    slopes[-1] = _end_slope(h[-1], h[-2], delta[-1], delta[-2])

    xc = np.clip(x, xs[0], xs[-1])
    seg = np.clip(np.searchsorted(xs, xc, side="right") - 1, 0, len(xs) - 2)
    t = (xc - xs[seg]) / h[seg]
    h00 = (1 + 2 * t) * (1 - t) ** 2
    h10 = t * (1 - t) ** 2
    h01 = t * t * (3 - 2 * t)
    h11 = t * t * (t - 1)
    out = (
        h00 * ys[seg]
        + h10 * h[seg] * slopes[seg]
        + h01 * ys[seg + 1]
        + h11 * h[seg] * slopes[seg + 1]
    )
    return cast("F64", out)


def _end_slope(h0: float, h1: float, d0: float, d1: float) -> float:
    """One-sided three-point slope, shape-preserving (as in PCHIP)."""
    slope = ((2 * h0 + h1) * d0 - h0 * d1) / (h0 + h1)
    if np.sign(slope) != np.sign(d0):
        return 0.0
    if np.sign(d0) != np.sign(d1) and abs(slope) > abs(3 * d0):
        return 3 * d0
    return float(slope)


# ── Application ──────────────────────────────────────────────────────────────


def _smoothstep(x: F64, edge0: float, edge1: float) -> F64:
    t = np.clip((x - edge0) / (edge1 - edge0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def apply_finishing(lab: F64, finishing: Finishing) -> F64:
    """Apply *finishing* to Oklab colours ``(N, 3)``; returns new colours."""
    if finishing.is_identity:
        return lab.copy()
    out = lab.copy()

    # 1. Tone curve (L only)
    if not curve_is_identity(finishing.tone_curve):
        out[:, 0] = eval_curve(finishing.tone_curve, out[:, 0])

    # 2. Wheels: Bernstein weights on the (curved) lightness
    lightness = np.clip(out[:, 0], 0.0, 1.0)
    weights = (
        (finishing.shadows, (1.0 - lightness) ** 2),
        (finishing.midtones, 2.0 * lightness * (1.0 - lightness)),
        (finishing.highlights, lightness**2),
    )
    for wheel, w in weights:
        if wheel.is_neutral:
            continue
        out[:, 0] += w * wheel.lightness * WHEEL_LIGHTNESS_RANGE
        out[:, 1:] += w[:, None] * wheel.tint_ab[None, :]

    # 3. Per-colour bands and 4. saturation / vibrance, in Oklch
    active = [band for band in finishing.bands if not band.is_neutral]
    if active or finishing.saturation != 0.0 or finishing.vibrance != 0.0:
        lch = oklab_to_oklch(out)
        colourful = _smoothstep(lch[:, 1], *GREY_CHROMA)
        hue_shift = np.zeros(len(lch))
        chroma_factor = np.ones(len(lch))
        light_shift = np.zeros(len(lch))
        for band in active:
            dist = (lch[:, 2] - band.center + 180.0) % 360.0 - 180.0
            w = np.exp(-0.5 * (dist / band.width) ** 2) * colourful
            hue_shift += w * band.hue * BAND_HUE_RANGE
            chroma_factor *= 1.0 + w * band.saturation
            light_shift += w * band.lightness * BAND_LIGHTNESS_RANGE
        lch[:, 2] = (lch[:, 2] + hue_shift) % 360.0
        lch[:, 0] += light_shift
        chroma = lch[:, 1] * chroma_factor * (1.0 + finishing.saturation)
        if finishing.vibrance != 0.0:
            dull = 1.0 - _smoothstep(chroma, 0.0, VIBRANCE_CHROMA)
            chroma = chroma * (1.0 + finishing.vibrance * dull)
        lch[:, 1] = np.maximum(chroma, 0.0)
        out = oklch_to_oklab(lch)
    return out


# ── Serialisation (recipes) ──────────────────────────────────────────────────


def finishing_to_dict(finishing: Finishing) -> dict[str, object]:
    """JSON-friendly form of *finishing* (neutral bands omitted)."""

    def wheel(w: Wheel) -> dict[str, float]:
        return {"hue": w.hue, "amount": w.amount, "lightness": w.lightness}

    return {
        "shadows": wheel(finishing.shadows),
        "midtones": wheel(finishing.midtones),
        "highlights": wheel(finishing.highlights),
        "tone_curve": [list(p) for p in finishing.tone_curve],
        "bands": [
            {
                "center": b.center,
                "hue": b.hue,
                "saturation": b.saturation,
                "lightness": b.lightness,
                "width": b.width,
            }
            for b in finishing.bands
            if not b.is_neutral
        ],
        "saturation": finishing.saturation,
        "vibrance": finishing.vibrance,
    }


def with_band(finishing: Finishing, band: HueBand) -> Finishing:
    """*finishing* where the band with *band*'s centre is replaced (or added)."""
    bands = list(finishing.bands)
    for i, existing in enumerate(bands):
        if abs(existing.center - band.center) < 1e-6:
            bands[i] = band
            break
    else:
        bands.append(band)
    return Finishing(
        shadows=finishing.shadows,
        midtones=finishing.midtones,
        highlights=finishing.highlights,
        tone_curve=finishing.tone_curve,
        bands=tuple(bands),
        saturation=finishing.saturation,
        vibrance=finishing.vibrance,
    )
