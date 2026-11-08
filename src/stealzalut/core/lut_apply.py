"""Apply a 3D LUT grid to an image.

Used for the live preview so that what the user sees is exactly what
the exported ``.cube`` file will produce (same grid, same resolution,
trilinear interpolation like most host applications).
"""

from __future__ import annotations

from typing import cast

import numpy as np
from numpy.typing import NDArray

from stealzalut.core.lut_engine import validate_grid_shape


def apply_lut(
    image_rgb: NDArray[np.uint8],
    grid: NDArray[np.float64],
    size: int,
) -> NDArray[np.uint8]:
    """Apply a LUT grid to an image using trilinear interpolation.

    Args:
        image_rgb: Image, **RGB** uint8, shape ``(H, W, 3)``.
        grid: LUT grid ``(size³, 3)`` in ``[0, 1]``, ``.cube`` ordering
            (R fastest, B slowest — see :mod:`stealzalut.core.lut_engine`).
        size: Grid resolution per axis.

    Returns:
        Graded image, **RGB** uint8, same shape as *image_rgb*.

    Raises:
        ValueError: If *grid* does not match *size*.
    """
    validate_grid_shape(grid, size)
    lut = grid.reshape(size, size, size, 3)  # indexed [b, g, r]

    coords = image_rgb.reshape(-1, 3).astype(np.float64) / 255.0 * (size - 1)
    i0 = np.clip(np.floor(coords).astype(np.int64), 0, size - 2)
    frac = coords - i0
    r0, g0, b0 = i0[:, 0], i0[:, 1], i0[:, 2]
    fr, fg, fb = frac[:, 0:1], frac[:, 1:2], frac[:, 2:3]

    out = np.zeros_like(coords)
    for db, wb in ((0, 1.0 - fb), (1, fb)):
        for dg, wg in ((0, 1.0 - fg), (1, fg)):
            for dr, wr in ((0, 1.0 - fr), (1, fr)):
                out += wb * wg * wr * lut[b0 + db, g0 + dg, r0 + dr]

    result = (np.clip(out, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    return cast("NDArray[np.uint8]", result.reshape(image_rgb.shape))
