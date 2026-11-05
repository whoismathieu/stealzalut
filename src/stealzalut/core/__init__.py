"""Core business logic — pure Python, no Qt dependencies.

This package contains all colour-science computations and file I/O.
It is fully testable with ``pytest`` without instantiating a
``QApplication``.
"""

from stealzalut.core.color_transfer import (
    METHOD_LABELS,
    SUPPORTED_METHODS,
    AffineTransform,
    ColorTransferEngine,
)
from stealzalut.core.lut_engine import VALID_LUT_SIZES, generate_neutral_grid
from stealzalut.core.lut_export import write_cube_file

__all__ = [
    "METHOD_LABELS",
    "SUPPORTED_METHODS",
    "VALID_LUT_SIZES",
    "AffineTransform",
    "ColorTransferEngine",
    "generate_neutral_grid",
    "write_cube_file",
]

