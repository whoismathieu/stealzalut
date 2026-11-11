"""Colour-space conversions: sRGB ↔ linear RGB ↔ Oklab ↔ Oklch.

Oklab (Björn Ottosson, 2020) is a perceptual colour space where
``L`` is perceived lightness and ``(a, b)`` carry chroma.  Working in
Oklab lets the transfer engine treat lightness and colour separately,
which a gamma-encoded sRGB space cannot do.

All functions operate on flat arrays of shape ``(N, 3)`` in float64.
sRGB values are expected in ``[0, 1]``.

.. note::
    Hue is undefined for near-neutral colours (``C ≈ 0``).  Any
    hue-driven operation must be weighted by chroma, e.g.
    ``smoothstep(C, 0.01, 0.04)``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import cast

import numpy as np
from numpy.typing import NDArray

F64 = NDArray[np.float64]

# ── Oklab matrices (reference values from the original publication) ─────────

_M1: F64 = np.array(
    [
        [0.4122214708, 0.5363325363, 0.0514459929],
        [0.2119034982, 0.6806995451, 0.1073969566],
        [0.0883024619, 0.2817188376, 0.6299787005],
    ]
)
_M2: F64 = np.array(
    [
        [0.2104542553, 0.7936177850, -0.0040720468],
        [1.9779984951, -2.4285922050, 0.4505937099],
        [0.0259040371, 0.7827717662, -0.8086757660],
    ]
)
_M2_INV: F64 = np.array(
    [
        [1.0, 0.3963377774, 0.2158037573],
        [1.0, -0.1055613458, -0.0638541728],
        [1.0, -0.0894841775, -1.2914855480],
    ]
)
_M1_INV: F64 = np.array(
    [
        [4.0767416621, -3.3077115913, 0.2309699292],
        [-1.2684380046, 2.6097574011, -0.3413193965],
        [-0.0041960863, -0.7034186147, 1.7076147010],
    ]
)

_GAMUT_EPS = 1e-6


# ── sRGB transfer function ───────────────────────────────────────────────────


def srgb_to_linear(c: F64) -> F64:
    """Decode sRGB gamma: encoded ``[0, 1]`` → linear light."""
    return cast("F64", np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4))


def linear_to_srgb(c: F64) -> F64:
    """Encode linear light → sRGB gamma.  Negative inputs are clamped to 0."""
    c = np.clip(c, 0.0, None)
    return cast(
        "F64",
        np.where(c <= 0.0031308, 12.92 * c, 1.055 * np.power(c, 1.0 / 2.4) - 0.055),
    )


# ── Oklab ────────────────────────────────────────────────────────────────────


def linear_to_oklab(rgb_linear: F64) -> F64:
    """Linear RGB ``(N, 3)`` → Oklab ``(N, 3)``."""
    lms = rgb_linear @ _M1.T
    return cast("F64", np.cbrt(lms) @ _M2.T)


def oklab_to_linear(lab: F64) -> F64:
    """Oklab ``(N, 3)`` → linear RGB ``(N, 3)`` (may fall outside ``[0, 1]``)."""
    lms: F64 = (lab @ _M2_INV.T) ** 3
    return lms @ _M1_INV.T


def srgb_to_oklab(rgb: F64) -> F64:
    """sRGB ``(N, 3)`` in ``[0, 1]`` → Oklab ``(N, 3)``."""
    return linear_to_oklab(srgb_to_linear(rgb))


def oklab_to_srgb(lab: F64) -> F64:
    """Oklab ``(N, 3)`` → sRGB ``(N, 3)`` clamped to ``[0, 1]``.

    Out-of-gamut colours are clipped per channel, which can shift hue.
    Call :func:`gamut_compress` first to avoid that.
    """
    return cast("F64", np.clip(linear_to_srgb(oklab_to_linear(lab)), 0.0, 1.0))


# ── Oklch (polar form) ───────────────────────────────────────────────────────


def oklab_to_oklch(lab: F64) -> F64:
    """Oklab → Oklch: ``(L, C, h)`` with hue ``h`` in degrees ``[0, 360)``."""
    chroma = np.hypot(lab[:, 1], lab[:, 2])
    hue = np.degrees(np.arctan2(lab[:, 2], lab[:, 1])) % 360.0
    return np.stack([lab[:, 0], chroma, hue], axis=1)


def oklch_to_oklab(lch: F64) -> F64:
    """Oklch ``(L, C, h°)`` → Oklab."""
    hue = np.radians(lch[:, 2])
    return np.stack([lch[:, 0], lch[:, 1] * np.cos(hue), lch[:, 1] * np.sin(hue)], axis=1)


# ── Gamut mapping ────────────────────────────────────────────────────────────


def _in_gamut(rgb_linear: F64) -> NDArray[np.bool_]:
    return cast(
        "NDArray[np.bool_]",
        np.all((rgb_linear >= -_GAMUT_EPS) & (rgb_linear <= 1.0 + _GAMUT_EPS), axis=1),
    )


def gamut_compress(
    lab: F64,
    iters: int = 12,
    to_linear: Callable[[F64], F64] | None = None,
) -> F64:
    """Pull out-of-gamut Oklab colours back into sRGB by reducing chroma.

    Lightness is clamped to ``[0, 1]`` and hue is preserved; only the
    chroma of offending colours is scaled down (binary search), which
    avoids the hue shifts produced by naive per-channel clipping.

    Args:
        lab: Oklab colours ``(N, 3)``.
        iters: Binary-search iterations (12 → precision ≈ 2.4e-4).
        to_linear: Oklab → linear RGB of the target space (default: sRGB).
            The gamut is the unit cube of those linear values.

    Returns:
        A new ``(N, 3)`` array whose colours map inside the sRGB gamut.
    """
    result = lab.copy()
    result[:, 0] = np.clip(result[:, 0], 0.0, 1.0)
    convert = oklab_to_linear if to_linear is None else to_linear
    out = ~_in_gamut(convert(result))
    if not out.any():
        return result

    lightness = result[out, :1]
    ab = result[out, 1:]
    lo: F64 = np.zeros(len(lightness))
    hi: F64 = np.ones(len(lightness))
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        candidate = np.concatenate([lightness, ab * mid[:, None]], axis=1)
        ok = _in_gamut(convert(candidate))
        lo = np.where(ok, mid, lo)
        hi = np.where(ok, hi, mid)
    result[out, 1:] = ab * lo[:, None]
    return result
