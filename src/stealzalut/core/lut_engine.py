"""Dynamic 3D LUT grid generation.

A 3D LUT (Look-Up Table) maps every possible input colour to an output
colour via a discretised sampling of the RGB cube.  This module generates
the **neutral (identity) grid** — a uniform lattice in [0, 1]³ — that
will later be deformed by the colour transfer engine.

Grid layout follows the ``.cube`` file convention:

* **R** varies fastest (innermost loop).
* **G** varies next.
* **B** varies slowest (outermost loop).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

# Resolutions supported by the application.
VALID_LUT_SIZES: frozenset[int] = frozenset({17, 33, 64})


def generate_neutral_grid(size: int) -> NDArray[np.float64]:
    """Generate a neutral (identity) 3D LUT grid.

    Creates a uniform sampling of the RGB colour cube where each output
    colour equals its input — i.e. an identity transform.

    Args:
        size: Grid resolution **per axis** (typically 17, 33, or 64).
              The total number of entries is ``size³``.

    Returns:
        Flat grid of shape ``(size³, 3)`` with values in [0.0, 1.0].

    Raises:
        ValueError: If *size* is not in :data:`VALID_LUT_SIZES`.

    Examples:
        >>> grid = generate_neutral_grid(17)
        >>> grid.shape
        (4913, 3)
        >>> grid[0]       # First entry: R=0, G=0, B=0  (black)
        array([0., 0., 0.])
        >>> grid[-1]      # Last entry:  R=1, G=1, B=1  (white)
        array([1., 1., 1.])
    """
    if size not in VALID_LUT_SIZES:
        raise ValueError(
            f"LUT size must be one of {sorted(VALID_LUT_SIZES)}, got {size}"
        )

    axis = np.linspace(0.0, 1.0, size, dtype=np.float64)

    # meshgrid with indexing='ij': first arg varies slowest.
    # Args order (B, G, R) → B slowest, R fastest  ← .cube convention.
    b_grid, g_grid, r_grid = np.meshgrid(axis, axis, axis, indexing="ij")

    grid: NDArray[np.float64] = np.stack(
        [r_grid.ravel(), g_grid.ravel(), b_grid.ravel()],
        axis=-1,
    )
    return grid


def validate_grid_shape(grid: NDArray[np.float64], size: int) -> None:
    """Assert that *grid* has the expected shape for *size*.

    Args:
        grid: LUT grid array to validate.
        size: Expected resolution per axis.

    Raises:
        ValueError: If ``grid.shape != (size³, 3)``.
    """
    expected_shape = (size**3, 3)
    if grid.shape != expected_shape:
        raise ValueError(
            f"Grid shape {grid.shape} does not match expected "
            f"{expected_shape} for LUT size {size}"
        )

