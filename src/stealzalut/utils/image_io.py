"""Image I/O utilities built on OpenCV and NumPy.

All public functions in this module work with NumPy arrays and never
depend on Qt, keeping them usable in headless / test environments.
"""

from __future__ import annotations

from pathlib import Path
from typing import cast

import cv2
import numpy as np
from numpy.typing import NDArray

# ── Type aliases ─────────────────────────────────────────────────────────────
ImageArray = NDArray[np.uint8]
"""HWC image in uint8 [0, 255]."""

FloatImage = NDArray[np.float64]
"""HWC image in float64 [0.0, 1.0]."""

# ── Constants ────────────────────────────────────────────────────────────────
SUPPORTED_EXTENSIONS: frozenset[str] = frozenset(
    {".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp", ".webp"}
)


# ── Loading ──────────────────────────────────────────────────────────────────


def load_image_bgr(path: str | Path) -> ImageArray:
    """Load an image from disk as a **BGR** uint8 NumPy array.

    Args:
        path: Path to the image file.

    Returns:
        BGR image of shape ``(H, W, 3)``, dtype ``uint8``.

    Raises:
        FileNotFoundError: If *path* does not exist.
        ValueError: If the extension is unsupported or decoding fails.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Image file not found: {path}")
    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported image format '{path.suffix}'. "
            f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Failed to decode image: {path}")
    return cast("ImageArray", image)


# ── Colour-space helpers ─────────────────────────────────────────────────────


def bgr_to_rgb(image: ImageArray) -> ImageArray:
    """Convert a BGR image to RGB (or vice-versa — the op is symmetric)."""
    return cast("ImageArray", cv2.cvtColor(image, cv2.COLOR_BGR2RGB))


def rgb_to_bgr(image: ImageArray) -> ImageArray:
    """Convert an RGB image to BGR."""
    return cast("ImageArray", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))


# ── Normalisation ────────────────────────────────────────────────────────────


def normalize_to_float(image: ImageArray) -> FloatImage:
    """Convert uint8 ``[0, 255]`` → float64 ``[0.0, 1.0]``."""
    return image.astype(np.float64) / 255.0


def float_to_uint8(image: FloatImage) -> ImageArray:
    """Convert float64 ``[0.0, 1.0]`` → uint8 ``[0, 255]``, clamped."""
    return np.clip(image * 255.0, 0.0, 255.0).astype(np.uint8)


# ── Thumbnailing ─────────────────────────────────────────────────────────────


def create_thumbnail(image: ImageArray, max_size: int = 512) -> ImageArray:
    """Resize an image so its longest edge is at most *max_size* px.

    Uses ``INTER_AREA`` interpolation for clean down-scaling.

    Args:
        image: Input image ``(H, W, 3)``.
        max_size: Maximum dimension in pixels.

    Returns:
        Down-scaled copy, or an unchanged copy if already small enough.
    """
    h, w = image.shape[:2]
    if max(h, w) <= max_size:
        return image.copy()

    scale = max_size / max(h, w)
    new_w = int(w * scale)
    new_h = int(h * scale)
    return cast("ImageArray", cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA))

