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
from stealzalut.core.xmp_export import (
    b85_decode_le,
    b85_encode_le,
    build_xmp_table_payload,
    cube_to_xmp,
    read_cube_file,
    write_xmp_profile,
)

__all__ = [
    "METHOD_LABELS",
    "SUPPORTED_METHODS",
    "VALID_LUT_SIZES",
    "AffineTransform",
    "ColorTransferEngine",
    "b85_decode_le",
    "b85_encode_le",
    "build_xmp_table_payload",
    "cube_to_xmp",
    "generate_neutral_grid",
    "read_cube_file",
    "write_cube_file",
    "write_xmp_profile",
]
