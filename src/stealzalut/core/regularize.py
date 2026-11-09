"""Regularisation of 3D LUT grids.

Blending several colour transforms (protections today, per-zone transfers
later) can create banding or *folds* — a brighter input colour mapped to a
darker output.  This module smooths displacement fields over the 3D grid
and measures how folded a grid is.

Grids follow the ``.cube`` ordering (R fastest, B slowest), so a flat
``(N³, k)`` array reshapes to ``(N, N, N, k)`` indexed ``[b, g, r]``.
"""

from __future__ import annotations

from typing import cast

import numpy as np
from numpy.typing import NDArray

F64 = NDArray[np.float64]

#: A lightness drop larger than this between neighbouring nodes is a fold.
FOLD_TOLERANCE = 1e-3


def _gaussian_kernel(sigma: float) -> F64:
    radius = max(1, int(np.ceil(3.0 * sigma)))
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-(x**2) / (2.0 * sigma**2))
    return cast("F64", kernel / kernel.sum())


def smooth_field(field: F64, size: int, sigma_cells: float) -> F64:
    """Separable Gaussian blur of a per-node field over the 3D grid.

    Args:
        field: Values at each grid node, shape ``(size³, k)``.
        size: Grid resolution per axis.
        sigma_cells: Standard deviation in grid cells.  ``<= 0`` returns a copy.

    Returns:
        The smoothed field, same shape.  Edges use replicate padding, so a
        constant field is left unchanged.
    """
    if sigma_cells <= 0.0:
        return field.copy()
    channels = field.shape[1]
    volume = field.reshape(size, size, size, channels)
    kernel = _gaussian_kernel(sigma_cells)
    radius = (len(kernel) - 1) // 2
    for axis in range(3):
        pad = [(0, 0)] * 4
        pad[axis] = (radius, radius)
        padded = np.pad(volume, pad, mode="edge")
        acc = np.zeros_like(volume)
        for i, weight in enumerate(kernel):
            acc += weight * np.take(padded, np.arange(i, i + size), axis=axis)
        volume = acc
    return cast("F64", volume.reshape(-1, channels))


def fold_fraction(lightness: F64, size: int, tolerance: float = FOLD_TOLERANCE) -> float:
    """Fraction of grid edges along which output lightness decreases.

    Increasing R, G or B always increases input lightness, so a well-behaved
    grade keeps output lightness non-decreasing along every axis.

    Args:
        lightness: Output lightness (e.g. Oklab ``L``) per node, shape ``(size³,)``.
        size: Grid resolution per axis.
        tolerance: Drops smaller than this are ignored (numerical noise).

    Returns:
        Violations divided by the number of neighbouring node pairs, in ``[0, 1]``.
    """
    volume = lightness.reshape(size, size, size)
    violations = sum(int((np.diff(volume, axis=axis) < -tolerance).sum()) for axis in range(3))
    total = 3 * size * size * (size - 1)
    return violations / total
