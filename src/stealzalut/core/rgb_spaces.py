"""RGB colour spaces a ``.cube`` can be made for.

A LUT is only right in the encoding it was computed for.  The engine
works in Oklab, which is not bounded by any RGB gamut; a
:class:`RGBSpace` says how the nodes of a grid are to be read (which
primaries, which transfer curve) and written back.

Matrices are derived from the published chromaticities of each space, not
typed in; spaces whose white point is D50 (ProPhoto) are adapted to D65
with the Bradford transform.  All spaces here are **display-referred**:
log and scene-referred encodings are out of scope.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from stealzalut.core.colorspace import (
    linear_to_oklab,
    linear_to_srgb,
    oklab_to_linear,
    srgb_to_linear,
)

F64 = NDArray[np.float64]

_D65 = (0.3127, 0.3290)
_D50 = (0.3457, 0.3585)
_BRADFORD: F64 = np.array(
    [
        [0.8951, 0.2664, -0.1614],
        [-0.7502, 1.7135, 0.0367],
        [0.0389, -0.0685, 1.0296],
    ]
)


def _xy_to_xyz(xy: tuple[float, float]) -> F64:
    x, y = xy
    return np.array([x / y, 1.0, (1.0 - x - y) / y])


def _rgb_to_xyz(primaries: tuple[tuple[float, float], ...], white: tuple[float, float]) -> F64:
    """Matrix from linear RGB to XYZ (relative to the space's own white)."""
    columns = np.column_stack([_xy_to_xyz(xy) for xy in primaries])
    scale = np.linalg.solve(columns, _xy_to_xyz(white))
    matrix: F64 = np.asarray(columns * scale[None, :], dtype=np.float64)
    return matrix


def _adaptation(source_white: tuple[float, float], target_white: tuple[float, float]) -> F64:
    """Bradford chromatic adaptation matrix between two white points."""
    source = _BRADFORD @ _xy_to_xyz(source_white)
    target = _BRADFORD @ _xy_to_xyz(target_white)
    return np.linalg.inv(_BRADFORD) @ np.diag(target / source) @ _BRADFORD


_SRGB_PRIMARIES = ((0.64, 0.33), (0.30, 0.60), (0.15, 0.06))
_XYZ_TO_SRGB: F64 = np.asarray(
    np.linalg.inv(_rgb_to_xyz(_SRGB_PRIMARIES, _D65)), dtype=np.float64
)


def _to_linear_srgb(
    primaries: tuple[tuple[float, float], ...], white: tuple[float, float]
) -> F64:
    matrix = _rgb_to_xyz(primaries, white)
    if white != _D65:
        matrix = _adaptation(white, _D65) @ matrix
    return _XYZ_TO_SRGB @ matrix


def _gamma(exponent: float) -> tuple[Callable[[F64], F64], Callable[[F64], F64]]:
    def decode(c: F64) -> F64:
        return np.power(np.clip(c, 0.0, None), exponent)

    def encode(c: F64) -> F64:
        return np.power(np.clip(c, 0.0, None), 1.0 / exponent)

    return decode, encode


def _prophoto_decode(c: F64) -> F64:
    c = np.clip(c, 0.0, None)
    return np.where(c < 16.0 / 512.0, c / 16.0, np.power(c, 1.8))


def _prophoto_encode(c: F64) -> F64:
    c = np.clip(c, 0.0, None)
    return np.where(c < 1.0 / 512.0, 16.0 * c, np.power(c, 1.0 / 1.8))


@dataclass(frozen=True, slots=True, eq=False)
class RGBSpace:
    """One RGB encoding: primaries (as a matrix to linear sRGB) and transfer curve.

    Attributes:
        key: Stable identifier (settings, file names).
        label: Name shown to the user.
        host_name: What the space is called in a host's dialog.
        to_linear_srgb: 3×3 matrix, linear RGB of this space → linear sRGB.
        decode: Encoded values → linear light.
        encode: Linear light → encoded values (negative input clamped to 0).
        wide: True when the gamut holds colours a sRGB screen cannot show.
    """

    key: str
    label: str
    host_name: str
    to_linear_srgb: F64
    decode: Callable[[F64], F64]
    encode: Callable[[F64], F64]
    wide: bool

    def to_oklab(self, rgb: F64) -> F64:
        """Encoded RGB ``(N, 3)`` in ``[0, 1]`` → Oklab ``(N, 3)``."""
        return linear_to_oklab(self.decode(rgb) @ self.to_linear_srgb.T)

    def oklab_to_linear(self, lab: F64) -> F64:
        """Oklab → linear RGB of this space (outside ``[0, 1]`` when out of gamut)."""
        return oklab_to_linear(lab) @ np.linalg.inv(self.to_linear_srgb).T

    def from_oklab(self, lab: F64) -> F64:
        """Oklab → encoded RGB clamped to ``[0, 1]`` (compress the gamut first)."""
        return np.clip(self.encode(self.oklab_to_linear(lab)), 0.0, 1.0)


_IDENTITY: F64 = np.eye(3)
_P3 = _to_linear_srgb(((0.680, 0.320), (0.265, 0.690), (0.150, 0.060)), _D65)
_ADOBE = _to_linear_srgb(((0.64, 0.33), (0.21, 0.71), (0.15, 0.06)), _D65)
_PROPHOTO = _to_linear_srgb(((0.7347, 0.2653), (0.1596, 0.8404), (0.0366, 0.0001)), _D50)
_ADOBE_CURVE = _gamma(563.0 / 256.0)
_REC709_CURVE = _gamma(2.4)

SRGB = RGBSpace("srgb", "sRGB", "sRGB", _IDENTITY, srgb_to_linear, linear_to_srgb, False)
DISPLAY_P3 = RGBSpace(
    "display-p3", "Display P3", "Display P3", _P3, srgb_to_linear, linear_to_srgb, True
)
ADOBE_RGB = RGBSpace(
    "adobe-rgb", "Adobe RGB (1998)", "Adobe RGB", _ADOBE, _ADOBE_CURVE[0], _ADOBE_CURVE[1], True
)
PROPHOTO_RGB = RGBSpace(
    "prophoto", "ProPhoto RGB", "ProPhoto RGB", _PROPHOTO, _prophoto_decode, _prophoto_encode, True
)
REC709 = RGBSpace(
    "rec709-g24",
    "Rec.709, gamma 2,4",
    "Rec.709 Gamma 2.4",
    _IDENTITY,
    _REC709_CURVE[0],
    _REC709_CURVE[1],
    False,
)

#: Spaces offered for the exported ``.cube``, in menu order.
CUBE_SPACES: tuple[RGBSpace, ...] = (SRGB, DISPLAY_P3, ADOBE_RGB, PROPHOTO_RGB, REC709)


def space_by_key(key: str) -> RGBSpace:
    """The space named *key*.

    Raises:
        ValueError: If no space has this key.
    """
    for space in CUBE_SPACES:
        if space.key == key:
            return space
    raise ValueError(f"Unknown colour space '{key}'")
