"""Look transfer: the reference read as a rendering, not as a palette.

The zones method copies the reference's colours zone by zone.  What one
recognises in a reference (and in a film print) is rather a *rendering*:

1. a **tone curve** (dense blacks, where the mid-tones sit, how bright the
   lights stay);
2. a **cast that changes with tone** (green shadows, mauve highlights);
3. an **amount of colour that changes with tone**;
4. how far apart the hues are pushed (**separation**).

This model learns the four from the pair of images:

* **Tone.**  A monotone curve through matched lightness quantiles of the
  bulk of both images, plus their white points.  Bright light sources are
  matched by level, not by share: a street with many signs must not turn
  grey because the reference has few.
* **Colour, conditionally on tone.**  For each tone band, the source's
  chroma distribution ``(a, b)`` is transported onto the chroma distribution
  the reference has *at the tone the band is sent to* (sliced optimal
  transport).  The target colour therefore exists at the output lightness,
  so the gamut compression has little left to trim.  The band's mean
  displacement is its **cast**; the remainder spreads the source's small
  hue differences over the reference's palette — its weight is the
  **separation**.

Lightness only depends on input lightness, through a monotone curve: the
grid cannot fold.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import cast

import numpy as np
from numpy.typing import NDArray

from stealzalut.core.color_transfer import TransferModel
from stealzalut.core.colorspace import gamut_compress, oklab_to_srgb, srgb_to_oklab
from stealzalut.core.finishing import eval_curve
from stealzalut.core.palette import analysis_oklab, colour_stats

F64 = NDArray[np.float64]

#: Source quantiles whose lightness is matched to the reference's (the bulk).
TONE_QUANTILES: tuple[float, ...] = (0.01, 0.05, 0.15, 0.30, 0.50, 0.70, 0.85)
#: Quantile of the white point (light sources).
WHITE_QUANTILE = 0.999
#: Bounds on the slope of the tone curve between two knots.
TONE_SLOPE: tuple[float, float] = (0.35, 3.0)
#: Minimum lightness gap between two knots of the tone curve.
TONE_MIN_GAP = 0.01
#: Samples of the tone curve stored in the model.
TONE_SAMPLES = 129

#: Tone bands of the chroma transport.
BANDS = 12
#: Chroma samples drawn per band and per image.
BAND_SAMPLES = 3000
#: Minimum width (Oklab L) of a band's Gaussian window.
BAND_MIN_WIDTH = 0.02
#: Share of the reference below which a band's transport is not trusted.
MIN_BAND_SHARE = 0.004
#: Projections of the sliced transport.
SLICES = 48
#: Smoothing of casts and fields across neighbouring bands (in bands).
BAND_SMOOTHING = 0.7

#: Companded chroma ``p = ab / (|ab| + COMPAND)``: the lattice is fine near
#: grey, where most pixels of a photograph sit, and coarse for vivid colours.
COMPAND = 0.05
#: Nodes of the chroma lattice per axis, covering companded ``[-1, 1]``.
AB_CELLS = 49
#: Smoothing of the residual field (lattice cells).
SIGMA_FINE = 1.5
#: Far-field smoothing (lattice cells): what colours absent from the source get.
SIGMA_COARSE = 6.0
#: Weight of the far field against the local field (samples per cell).
FIELD_PRIOR = 2.0
#: Pulls the far field to zero where the source has no colour at all.
FAR_PRIOR = 0.05

#: Chroma range over which a source colour counts as vivid (as in grading).
VIVID_CHROMA: tuple[float, float] = (0.04, 0.12)
#: Protection of vivid source colours when the reference has a single hue.
MONOCHROME_VIVID_PROTECTION = 0.8
#: Range of the weight of a reference zone (0 = ignored, 1 = normal).
FOCUS_RANGE: tuple[float, float] = (0.0, 4.0)
#: Range of the separation setting.
SEPARATION_RANGE: tuple[float, float] = (0.0, 1.5)

#: Share of coloured source pixels below which there is little to separate.
LOW_COLOUR_SHARE = 0.03
#: Trust below which a tone band counts as not covered by the reference.
WEAK_BAND_TRUST = 0.5

_GOLDEN = 0.6180339887498949


def _smoothstep(x: F64, edge0: float, edge1: float) -> F64:
    t = np.clip((x - edge0) / (edge1 - edge0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _analysis_lab(image: NDArray[np.uint8]) -> F64:
    """Oklab pixels ``(N, 3)`` of the analysis copy of *image* (unclipped if known)."""
    return analysis_oklab(image)[0]


def weighted_quantile(values: F64, weights: F64, u: F64) -> F64:
    """Quantiles *u* of *values* under non-negative *weights* (mid-point rule).

    Values with a zero weight are ignored.
    """
    kept = weights > 0.0
    values, weights = values[kept], weights[kept]
    order = np.argsort(values, kind="stable")
    sorted_w = weights[order]
    cdf = np.cumsum(sorted_w)
    return np.interp(u, (cdf - 0.5 * sorted_w) / cdf[-1], values[order])


# ── Tone ─────────────────────────────────────────────────────────────────────


def fit_tone_curve(
    source_l: F64,
    reference_l: F64,
    reference_weights: F64 | None = None,
) -> tuple[F64, F64]:
    """Monotone tone curve from source lightness to reference lightness.

    Knots: black ``(0, 0)``, the matched quantiles :data:`TONE_QUANTILES`,
    the white points (:data:`WHITE_QUANTILE`), then ``x = 1``.  Each knot is
    clipped so that the slope from the previous one stays within
    :data:`TONE_SLOPE`; knots closer than :data:`TONE_MIN_GAP` are skipped.
    The curve is the monotone cubic through the knots.

    Returns:
        ``(x, y)``: :data:`TONE_SAMPLES` regular samples of the curve.
    """
    u = np.array([*TONE_QUANTILES, WHITE_QUANTILE])
    weights = np.ones_like(reference_l) if reference_weights is None else reference_weights
    q_src = np.quantile(source_l, u)
    q_ref = weighted_quantile(reference_l, weights, u)

    xs, ys = [0.0], [0.0]
    lo_slope, hi_slope = TONE_SLOPE
    for x_k, y_k in zip(q_src.tolist(), q_ref.tolist(), strict=True):
        gap = x_k - xs[-1]
        if gap <= TONE_MIN_GAP:
            continue
        xs.append(x_k)
        ys.append(float(np.clip(y_k, ys[-1] + lo_slope * gap, ys[-1] + hi_slope * gap)))
    if xs[-1] < 1.0 - 1e-6:
        ys.append(min(1.0, ys[-1] + (1.0 - xs[-1])))
        xs.append(1.0)

    x: F64 = np.linspace(0.0, 1.0, TONE_SAMPLES, dtype=np.float64)
    y = eval_curve(tuple(zip(xs, ys, strict=True)), x)
    return x, np.clip(y, 0.0, 1.0)


# ── Chroma transport ─────────────────────────────────────────────────────────


def _resample(points: F64, weights: F64, count: int) -> F64:
    """*count* points drawn from *points* proportionally to *weights* (systematic)."""
    cdf = np.cumsum(weights)
    picks = np.searchsorted(cdf / cdf[-1], (np.arange(count) + 0.5) / count)
    return points[np.clip(picks, 0, len(points) - 1)]


def sliced_transport(source: F64, target: F64, slices: int = SLICES) -> F64:
    """Move *source* points ``(n, 2)`` onto the distribution of *target* ``(n, 2)``.

    Iterative distribution transfer: along each direction of a fixed
    sequence, the projections of the moving points are matched rank for rank
    to the target's.  Both arrays must hold the same number of points.

    Returns:
        The moved points, in the order of *source*.
    """
    moved = source.copy()
    for k in range(slices):
        angle = np.pi * ((k * _GOLDEN) % 1.0)
        direction = np.array([np.cos(angle), np.sin(angle)])
        projected = moved @ direction
        goal = np.sort(target @ direction)
        order = np.argsort(projected, kind="stable")
        shift = np.empty_like(projected)
        shift[order] = goal - projected[order]
        moved += shift[:, None] * direction[None, :]
    return moved


def _blur(volume: F64, sigma: float, axes: tuple[int, ...]) -> F64:
    """Gaussian blur along *axes* with replicate padding."""
    radius = max(1, int(np.ceil(3.0 * sigma)))
    offsets = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-(offsets**2) / (2.0 * sigma**2))
    kernel /= kernel.sum()
    out = volume
    for axis in axes:
        pad = [(0, 0)] * out.ndim
        pad[axis] = (radius, radius)
        padded = np.pad(out, pad, mode="edge")
        acc = np.zeros_like(out)
        length = out.shape[axis]
        for i, weight in enumerate(kernel):
            acc += weight * np.take(padded, np.arange(i, i + length), axis=axis)
        out = acc
    return out


def lattice_position(ab: F64) -> F64:
    """Continuous lattice coordinates ``(N, 2)`` of chroma values ``(N, 2)``."""
    companded = ab / (np.hypot(ab[:, 0], ab[:, 1]) + COMPAND)[:, None]
    return cast("F64", np.clip((companded + 1.0) * 0.5 * (AB_CELLS - 1), 0.0, AB_CELLS - 1.0))


def _splat(points: F64, values: F64) -> tuple[F64, F64]:
    """Bilinear accumulation of *values* ``(n, 2)`` at chroma *points*."""
    pos = np.minimum(lattice_position(points), AB_CELLS - 1.0 - 1e-9)
    i0 = np.floor(pos).astype(np.int64)
    frac = pos - i0
    total = np.zeros((AB_CELLS, AB_CELLS, 2))
    count = np.zeros((AB_CELLS, AB_CELLS))
    for da, wa in ((0, 1.0 - frac[:, 0]), (1, frac[:, 0])):
        for db, wb in ((0, 1.0 - frac[:, 1]), (1, frac[:, 1])):
            w = wa * wb
            np.add.at(count, (i0[:, 0] + da, i0[:, 1] + db), w)
            np.add.at(total, (i0[:, 0] + da, i0[:, 1] + db), w[:, None] * values)
    return total, count


# ── Model ────────────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True, eq=False)
class LookTransferModel(TransferModel):
    """Tone curve, cast per tone and residual chroma field.

    Attributes:
        tone_x: Input lightness samples of the tone curve, ``(TONE_SAMPLES,)``.
        tone_y: Output lightness at *tone_x*, non-decreasing.
        band_lightness: Source lightness of the band centres, ``(BANDS,)``,
            increasing.
        cast: Mean chroma displacement of each band, ``(BANDS, 2)``.
        field: Residual chroma displacement on the lattice,
            ``(BANDS, AB_CELLS, AB_CELLS, 2)``.
        band_trust: How much each band's transport was trusted, ``(BANDS,)``
            in ``[0, 1]`` (low when the reference has almost no pixel at the
            band's tone).
        separation: Weight of the residual field (0 = cast only).
        protect_vivid: Share of the displacement withheld from vivid source
            colours.
    """

    tone_x: F64
    tone_y: F64
    band_lightness: F64
    cast: F64
    field: F64
    band_trust: F64
    separation: float = 1.0
    protect_vivid: float = 0.0

    @property
    def method(self) -> str:
        return "look"

    def with_separation(self, separation: float) -> LookTransferModel:
        """Same look, different separation (no refit)."""
        lo, hi = SEPARATION_RANGE
        return replace(self, separation=float(np.clip(separation, lo, hi)))

    def with_protect_vivid(self, protect_vivid: float) -> LookTransferModel:
        """Same look, different protection of vivid source colours."""
        return replace(self, protect_vivid=float(np.clip(protect_vivid, 0.0, 1.0)))

    def tone(self, lightness: F64) -> F64:
        """Output lightness for input lightness values ``(N,)``."""
        return np.interp(lightness, self.tone_x, self.tone_y)

    def chroma_shift(self, lab: F64) -> F64:
        """Chroma displacement ``(N, 2)`` for Oklab colours *lab* ``(N, 3)``."""
        index = np.arange(len(self.band_lightness), dtype=np.float64)
        band = np.interp(lab[:, 0], self.band_lightness, index)
        coords = np.column_stack([band, lattice_position(lab[:, 1:])])
        upper = np.array(self.field.shape[:3]) - 2
        i0 = np.clip(np.floor(coords).astype(np.int64), 0, upper)
        frac = coords - i0
        residual = np.zeros((lab.shape[0], 2))
        for d0, w0 in ((0, 1.0 - frac[:, 0]), (1, frac[:, 0])):
            for d1, w1 in ((0, 1.0 - frac[:, 1]), (1, frac[:, 1])):
                for d2, w2 in ((0, 1.0 - frac[:, 2]), (1, frac[:, 2])):
                    residual += (w0 * w1 * w2)[:, None] * self.field[
                        i0[:, 0] + d0, i0[:, 1] + d1, i0[:, 2] + d2
                    ]
        band_cast = np.column_stack([np.interp(band, index, self.cast[:, i]) for i in range(2)])
        shift: F64 = band_cast + self.separation * residual
        if self.protect_vivid > 0.0:
            chroma = np.hypot(lab[:, 1], lab[:, 2])
            shift *= (1.0 - self.protect_vivid * _smoothstep(chroma, *VIVID_CHROMA))[:, None]
        return shift

    def apply_lab(self, lab: F64) -> F64:
        out = np.empty_like(lab)
        out[:, 0] = self.tone(lab[:, 0])
        out[:, 1:] = lab[:, 1:] + self.chroma_shift(lab)
        return out

    def apply(self, colors: F64) -> F64:
        return oklab_to_srgb(gamut_compress(self.apply_lab(srgb_to_oklab(colors))))


def fit_look(
    source: NDArray[np.uint8],
    reference: NDArray[np.uint8],
    reference_weights: F64 | None = None,
) -> LookTransferModel:
    """Fit the look model on a source / reference pair (RGB uint8).

    Args:
        source: Source image, **RGB** uint8.
        reference: Reference image, **RGB** uint8.
        reference_weights: Optional non-negative weight per pixel of the
            reference's analysis copy (``create_thumbnail(reference,
            ANALYSIS_MAX_SIZE)``, flattened row by row): what the look is
            read from.  ``None`` = every pixel counts the same.

    Returns:
        The fitted model.  Vivid source colours are protected
        (:data:`MONOCHROME_VIVID_PROTECTION`) when the reference has a single
        dominant hue, so that it does not act as a uniform filter.

    Raises:
        ValueError: If *reference_weights* has the wrong length or no
            positive weight.
    """
    lab_s, lab_r = _analysis_lab(source), _analysis_lab(reference)
    if reference_weights is None:
        weights: F64 = np.ones(len(lab_r))
    else:
        weights = np.asarray(reference_weights, dtype=np.float64).ravel()
        if weights.shape[0] != lab_r.shape[0] or weights.min() < 0.0 or weights.sum() <= 0.0:
            raise ValueError("reference_weights must match the analysis copy and be positive")

    tone_x, tone_y = fit_tone_curve(lab_s[:, 0], lab_r[:, 0], weights)

    centres = np.asarray(
        np.quantile(lab_s[:, 0], (np.arange(BANDS) + 0.5) / BANDS), dtype=np.float64
    )
    band_l: F64 = np.maximum.accumulate(centres + 1e-6 * np.arange(BANDS))
    target_l = np.interp(band_l, tone_x, tone_y)
    width_s = np.maximum(np.gradient(band_l), BAND_MIN_WIDTH)
    width_r = np.maximum(np.gradient(target_l), BAND_MIN_WIDTH)

    field = np.zeros((BANDS, AB_CELLS, AB_CELLS, 2))
    casts = np.zeros((BANDS, 2))
    trust = np.zeros(BANDS)
    for m in range(BANDS):
        w_s = np.exp(-0.5 * ((lab_s[:, 0] - band_l[m]) / width_s[m]) ** 2)
        w_r = weights * np.exp(-0.5 * ((lab_r[:, 0] - target_l[m]) / width_r[m]) ** 2)
        share = float(w_r.sum() / weights.sum())
        trust[m] = min(1.0, share / MIN_BAND_SHARE) ** 2
        if share <= 0.0:
            continue
        points = _resample(lab_s[:, 1:], w_s, BAND_SAMPLES)
        goal = _resample(lab_r[:, 1:], w_r, BAND_SAMPLES)
        move = trust[m] * (sliced_transport(points, goal) - points)
        casts[m] = move.mean(axis=0)
        total, count = _splat(points, move - casts[m])
        far = _blur(total, SIGMA_COARSE, (0, 1)) / (
            _blur(count, SIGMA_COARSE, (0, 1)) + FAR_PRIOR
        )[..., None]
        local_total = _blur(total, SIGMA_FINE, (0, 1))
        local_count = _blur(count, SIGMA_FINE, (0, 1))
        field[m] = (local_total + FIELD_PRIOR * far) / (local_count + FIELD_PRIOR)[..., None]

    monochrome = colour_stats(reference).is_monochrome
    return LookTransferModel(
        tone_x=tone_x,
        tone_y=tone_y,
        band_lightness=band_l,
        cast=_blur(casts, BAND_SMOOTHING, (0,)),
        field=_blur(field, BAND_SMOOTHING, (0,)),
        band_trust=trust,
        protect_vivid=MONOCHROME_VIVID_PROTECTION if monochrome else 0.0,
    )


# ── Reading only part of the reference ───────────────────────────────────────


def zone_weights(labels: NDArray[np.int32], focus: Mapping[int, float]) -> F64:
    """Per-pixel weights of the reference from weights per zone.

    Args:
        labels: Zone id of every pixel of the reference's analysis copy
            (:attr:`~stealzalut.core.palette.ZonePalette.labels`).
        focus: Weight per zone id; zones left out weigh 1.

    Returns:
        ``labels.size`` weights, row by row, as :func:`fit_look` expects.

    Raises:
        ValueError: If a weight is outside :data:`FOCUS_RANGE`.
    """
    lo, hi = FOCUS_RANGE
    weights: F64 = np.ones(labels.size, dtype=np.float64)
    flat = labels.ravel()
    for zone_id, weight in focus.items():
        if not lo <= weight <= hi:
            raise ValueError(f"zone weight {weight} outside [{lo}, {hi}]")
        weights[flat == zone_id] = weight
    return weights


# ── Diagnostics ──────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class LookDiagnostics:
    """What limits a look on this pair, for the UI to show.

    Attributes:
        source_colourful: Share of coloured pixels in the source.
        weak_bands: Tone bands (indices) the reference barely covers: their
            colours are left almost untouched.
        low_colour: The source is nearly grey: separation has little to
            work with.
    """

    source_colourful: float
    weak_bands: tuple[int, ...]
    low_colour: bool


def look_diagnostics(model: LookTransferModel, source: NDArray[np.uint8]) -> LookDiagnostics:
    """Measure the known limits of *model* on its *source* image (RGB uint8)."""
    colourful = colour_stats(source).colourful
    weak = tuple(int(m) for m in np.flatnonzero(model.band_trust < WEAK_BAND_TRUST))
    return LookDiagnostics(colourful, weak, colourful < LOW_COLOUR_SHARE)
