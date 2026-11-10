"""Prototype of the "look" transfer: tone curve + tone-conditional chroma transport.

Not production code.  It validates the design of docs/approach-v3/DESIGN.md on
the real cases.  The model is a plain ``TransferModel`` so it goes through the
real baking pipeline (``build_grid``) and the real trilinear LUT.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from stealzalut.core.color_transfer import TransferModel
from stealzalut.core.finishing import eval_curve
from stealzalut.core.colorspace import gamut_compress, oklab_to_srgb, srgb_to_oklab
from stealzalut.utils.image_io import create_thumbnail

F64 = NDArray[np.float64]

FIT_SIZE = 256  # longest edge of the analysed copies
TONE_KNOTS = 65  # quantiles of the tone curve
TONE_SLOPE = (0.35, 3.0)  # bounds on the slope of the tone curve
BANDS = 12  # tone bands of the chroma transport
BAND_SAMPLES = 3000  # points per band and per image
SLICES = 48  # sliced-transport iterations
COMPAND = 0.05  # companded chroma p = ab / (|ab| + COMPAND): fine cells near grey
AB_CELLS = 49  # nodes per companded chroma axis, covering [-1, 1]
SIGMA_FINE = 1.5  # smoothing of the residual field (cells)
SIGMA_COARSE = 6.0  # far-field smoothing (cells)
PRIOR = 2.0  # weight of the far field (samples per cell)
FAR_PRIOR = 0.05  # pulls the far field to zero where the source has no colour
VIVID_CHROMA = (0.04, 0.12)


def _smoothstep(x: F64, e0: float, e1: float) -> F64:
    t = np.clip((x - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _compand(ab: F64) -> F64:
    return ab / (np.hypot(ab[:, 0], ab[:, 1]) + COMPAND)[:, None]


def _lab(image: NDArray[np.uint8]) -> F64:
    small = create_thumbnail(image, FIT_SIZE)
    return srgb_to_oklab(small.reshape(-1, 3).astype(np.float64) / 255.0)


TONE_QUANTILES = (0.01, 0.05, 0.15, 0.30, 0.50, 0.70, 0.85)  # the bulk of the picture
WHITE_QUANTILE = 0.999  # light sources: their level is matched, not their share
MIN_BAND_SHARE = 0.004  # reference share below which a tone band is not trusted


def _wquantile(values: F64, weights: F64, u: F64) -> F64:
    order = np.argsort(values, kind="stable")
    cdf = np.cumsum(weights[order])
    return np.interp(u, (cdf - 0.5 * weights[order]) / cdf[-1], values[order])


def fit_tone_curve(l_src: F64, l_ref: F64, w_ref: F64) -> tuple[F64, F64]:
    """Monotone curve through matched quantiles of the bulk + the white point."""
    u = np.array([*TONE_QUANTILES, WHITE_QUANTILE])
    qs = np.quantile(l_src, u)
    qr = _wquantile(l_ref, w_ref, u)
    xs, ys = [0.0], [0.0]
    for x_k, y_k in zip(qs, qr, strict=True):
        if x_k > xs[-1] + 0.01:
            lo = ys[-1] + TONE_SLOPE[0] * (x_k - xs[-1])
            hi = ys[-1] + TONE_SLOPE[1] * (x_k - xs[-1])
            xs.append(float(x_k))
            ys.append(float(np.clip(y_k, lo, hi)))
    if xs[-1] < 1.0 - 1e-6:
        ys.append(min(1.0, ys[-1] + (1.0 - xs[-1])))
        xs.append(1.0)
    x = np.linspace(0.0, 1.0, 129)
    y = eval_curve(tuple(zip(xs, ys, strict=True)), x)
    return x, np.clip(y, 0.0, 1.0)


def _resample(points: F64, weights: F64, n: int, rng: np.random.Generator) -> F64:
    cdf = np.cumsum(weights)
    picks = np.searchsorted(cdf / cdf[-1], (np.arange(n) + rng.random()) / n)
    return points[np.clip(picks, 0, len(points) - 1)]


def sliced_transport(src: F64, ref: F64, slices: int = SLICES) -> F64:
    """Move *src* points ``(n, 2)`` onto the distribution of *ref* ``(n, 2)``."""
    moved = src.copy()
    for k in range(slices):
        angle = np.pi * ((k * 0.6180339887498949) % 1.0)
        direction = np.array([np.cos(angle), np.sin(angle)])
        p_src = moved @ direction
        p_ref = np.sort(ref @ direction)
        order = np.argsort(p_src, kind="stable")
        shift = np.empty_like(p_src)
        shift[order] = p_ref - p_src[order]
        moved += shift[:, None] * direction[None, :]
    return moved


def _blur(volume: F64, sigma: float, axes: tuple[int, ...]) -> F64:
    radius = max(1, int(np.ceil(3.0 * sigma)))
    xk = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-(xk**2) / (2.0 * sigma**2))
    kernel /= kernel.sum()
    out = volume
    for axis in axes:
        pad = [(0, 0)] * out.ndim
        pad[axis] = (radius, radius)
        padded = np.pad(out, pad, mode="edge")
        acc = np.zeros_like(out)
        n = out.shape[axis]
        for i, w in enumerate(kernel):
            acc += w * np.take(padded, np.arange(i, i + n), axis=axis)
        out = acc
    return out


def _lattice_pos(ab: F64) -> F64:
    return np.clip((_compand(ab) + 1.0) * 0.5 * (AB_CELLS - 1), 0.0, AB_CELLS - 1 - 1e-9)


def _splat(points: F64, values: F64) -> tuple[F64, F64]:
    """Bilinear splat of *values* ``(n, 2)`` at chroma *points* onto the lattice."""
    pos = _lattice_pos(points)
    i0 = np.floor(pos).astype(np.int64)
    f = pos - i0
    num = np.zeros((AB_CELLS, AB_CELLS, 2))
    cnt = np.zeros((AB_CELLS, AB_CELLS))
    for da, wa in ((0, 1.0 - f[:, 0]), (1, f[:, 0])):
        for db, wb in ((0, 1.0 - f[:, 1]), (1, f[:, 1])):
            w = wa * wb
            np.add.at(cnt, (i0[:, 0] + da, i0[:, 1] + db), w)
            np.add.at(num, (i0[:, 0] + da, i0[:, 1] + db), w[:, None] * values)
    return num, cnt


@dataclass(frozen=True, slots=True, eq=False)
class LookModel(TransferModel):
    """Tone curve + cast per tone + residual chroma field (tone band, a, b)."""

    tone_x: F64
    tone_y: F64
    band_l: F64  # (BANDS,) source lightness of the band centres
    cast: F64  # (BANDS, 2) mean chroma displacement of each band
    field: F64  # (BANDS, AB_CELLS, AB_CELLS, 2) residual chroma displacement
    separation: float = 1.0
    protect_vivid: float = 0.0

    @property
    def method(self) -> str:
        return "look"

    def chroma_shift(self, lab: F64) -> F64:
        idx = np.arange(len(self.band_l), dtype=np.float64)
        band = np.interp(lab[:, 0], self.band_l, idx)
        coords = np.column_stack([band, _lattice_pos(lab[:, 1:])])
        shape = np.array(self.field.shape[:3])
        i0 = np.clip(np.floor(coords).astype(np.int64), 0, shape - 2)
        f = coords - i0
        out = np.zeros((lab.shape[0], 2))
        for d0, w0 in ((0, 1.0 - f[:, 0]), (1, f[:, 0])):
            for d1, w1 in ((0, 1.0 - f[:, 1]), (1, f[:, 1])):
                for d2, w2 in ((0, 1.0 - f[:, 2]), (1, f[:, 2])):
                    out += (w0 * w1 * w2)[:, None] * self.field[
                        i0[:, 0] + d0, i0[:, 1] + d1, i0[:, 2] + d2
                    ]
        cast = np.column_stack([np.interp(band, idx, self.cast[:, i]) for i in range(2)])
        return cast + self.separation * out

    def apply(self, colors: F64) -> F64:
        lab = srgb_to_oklab(colors)
        shift = self.chroma_shift(lab)
        if self.protect_vivid > 0.0:
            chroma = np.hypot(lab[:, 1], lab[:, 2])
            shift *= (1.0 - self.protect_vivid * _smoothstep(chroma, *VIVID_CHROMA))[:, None]
        out = np.empty_like(lab)
        out[:, 0] = np.interp(lab[:, 0], self.tone_x, self.tone_y)
        out[:, 1:] = lab[:, 1:] + shift
        return oklab_to_srgb(gamut_compress(out))


def fit_look(
    source: NDArray[np.uint8],
    reference: NDArray[np.uint8],
    protect_vivid: float = 0.0,
    separation: float = 1.0,
    sigma_fine: float = SIGMA_FINE,
    bands: int = BANDS,
    ref_weights: F64 | None = None,
) -> LookModel:
    lab_s, lab_r = _lab(source), _lab(reference)
    w_all = np.ones(len(lab_r)) if ref_weights is None else ref_weights
    tone_x, tone_y = fit_tone_curve(lab_s[:, 0], lab_r[:, 0], w_all)
    rng = np.random.default_rng(0)

    band_l = np.quantile(lab_s[:, 0], (np.arange(bands) + 0.5) / bands)
    band_l = np.maximum.accumulate(band_l + 1e-6 * np.arange(bands))
    target_l = np.interp(band_l, tone_x, tone_y)
    spacing_s, spacing_r = np.gradient(band_l), np.gradient(target_l)

    field = np.zeros((bands, AB_CELLS, AB_CELLS, 2))
    casts = np.zeros((bands, 2))
    for m in range(bands):
        width_s = max(float(spacing_s[m]), 0.02)
        width_r = max(float(spacing_r[m]), 0.02)
        w_s = np.exp(-0.5 * ((lab_s[:, 0] - band_l[m]) / width_s) ** 2)
        w_r = w_all * np.exp(-0.5 * ((lab_r[:, 0] - target_l[m]) / width_r) ** 2)
        trust = float(w_r.sum() / w_all.sum()) / MIN_BAND_SHARE
        trust = min(1.0, trust) ** 2
        pts_s = _resample(lab_s[:, 1:], w_s, BAND_SAMPLES, rng)
        pts_r = _resample(lab_r[:, 1:], w_r, BAND_SAMPLES, rng)
        moved = pts_s + trust * (sliced_transport(pts_s, pts_r) - pts_s)
        casts[m] = (moved - pts_s).mean(axis=0)
        num, cnt = _splat(pts_s, moved - pts_s - casts[m])
        far = _blur(num, SIGMA_COARSE, (0, 1)) / (_blur(cnt, SIGMA_COARSE, (0, 1)) + FAR_PRIOR)[
            ..., None
        ]
        num_f, cnt_f = _blur(num, sigma_fine, (0, 1)), _blur(cnt, sigma_fine, (0, 1))
        field[m] = (num_f + PRIOR * far) / (cnt_f + PRIOR)[..., None]
    field = _blur(field, 0.7, (0,))
    casts = _blur(casts, 0.7, (0,))
    return LookModel(tone_x, tone_y, band_l, casts, field, separation, protect_vivid)
