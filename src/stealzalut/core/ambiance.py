"""Ambiance transfer: the reference read as a colour cast per tone.

The zones method copies the reference's *palette*: zone by zone, colours
take the reference's colours.  When the reference holds a single colour
(a purple-lit night, a teal grade…), every target has that colour and the
result can only be a uniform filter.

A colourist reads such a reference as a **mood**: "dark tones purple,
lights warm white".  This method does the same:

1. Lightness follows the global Oklab transfer (the reference's tone).
2. The reference gives a **tint curve**: its mean colour at each lightness
   level (kernel average over the reference's pixels).  Since the output
   tone follows the reference, each output colour gets the tint the
   reference has at that tone: a dark purple sky stays dark purple, its
   light sources stay warm white.
3. The tint is **added** to the source's own colours, which are only
   muted a little so the cast reads, instead of being replaced.
4. Vivid source colours (neons, signs, lights) are protected: they keep
   their hue, so the picture keeps its colour variety.

The global "Intensité couleur" slider doses the cast like any transfer.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from stealzalut.core.color_transfer import OklabTransferModel, TransferModel, fit_transfer
from stealzalut.core.colorspace import gamut_compress, oklab_to_srgb, srgb_to_oklab
from stealzalut.core.palette import analysis_oklab

F64 = NDArray[np.float64]

#: Share of the source's own chroma removed under the cast.
MUTE = 0.35
#: How much vivid source colours are spared (cast and muting).
VIVID_PROTECTION = 0.8
#: Chroma range over which a source colour becomes "vivid".
VIVID_CHROMA = (0.04, 0.12)
#: Lightness samples of the tint curve.
CURVE_SAMPLES = 64
#: Width (Oklab L) of the kernel averaging reference colours per tone.
CURVE_BANDWIDTH = 0.05
#: Pseudo-count pulling sparse tones towards the reference's overall tint.
CURVE_PRIOR = 25.0


def _smoothstep(x: F64, edge0: float, edge1: float) -> F64:
    t = np.clip((x - edge0) / (edge1 - edge0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def curve_lightness() -> F64:
    """Lightness levels at which the tint curve is sampled."""
    levels: F64 = np.linspace(0.0, 1.0, CURVE_SAMPLES, dtype=np.float64)
    return levels


def reference_tint_curve(reference_rgb: NDArray[np.uint8]) -> F64:
    """Mean Oklab ``(a, b)`` of the reference at each lightness level.

    Gaussian kernel average over the reference's pixels; tones the
    reference barely has (e.g. above its brightest pixels) fall back to its
    overall mean colour.

    Returns:
        ``(CURVE_SAMPLES, 2)`` array, sampled at :func:`curve_lightness`.
    """
    lab = analysis_oklab(reference_rgb)[0]
    levels = curve_lightness()
    weights = np.exp(-0.5 * ((levels[:, None] - lab[None, :, 0]) / CURVE_BANDWIDTH) ** 2)
    overall = lab[:, 1:].mean(axis=0)
    sums = weights @ lab[:, 1:] + CURVE_PRIOR * overall
    curve: F64 = sums / (weights.sum(axis=1) + CURVE_PRIOR)[:, None]
    return curve


@dataclass(frozen=True, slots=True, eq=False)
class AmbianceTransferModel(TransferModel):
    """Reference tone + colour cast by tone, added to the source's colours."""

    tint_curve: F64
    global_model: OklabTransferModel
    mute: float = MUTE
    protect_vivid: float = VIVID_PROTECTION

    @property
    def method(self) -> str:
        return "ambiance"

    def tint(self, lightness: F64) -> F64:
        """Cast ``(a, b)`` for output lightness values ``(N,)``."""
        levels = curve_lightness()
        return np.column_stack(
            [np.interp(lightness, levels, self.tint_curve[:, i]) for i in range(2)]
        )

    def apply(self, colors: F64) -> F64:
        lab = srgb_to_oklab(colors)
        lightness = srgb_to_oklab(self.global_model.apply(colors))[:, 0]
        chroma = np.hypot(lab[:, 1], lab[:, 2])
        free = 1.0 - self.protect_vivid * _smoothstep(chroma, *VIVID_CHROMA)
        ab = lab[:, 1:] * (1.0 - self.mute * free)[:, None] + free[:, None] * self.tint(lightness)
        return oklab_to_srgb(gamut_compress(np.column_stack([lightness, ab])))


def fit_ambiance(
    source: NDArray[np.uint8],
    reference: NDArray[np.uint8],
) -> AmbianceTransferModel:
    """Fit the ambiance model on a source / reference pair (RGB uint8)."""
    global_model = fit_transfer(source, reference, method="oklab")
    assert isinstance(global_model, OklabTransferModel)
    return AmbianceTransferModel(
        tint_curve=reference_tint_curve(reference), global_model=global_model
    )
