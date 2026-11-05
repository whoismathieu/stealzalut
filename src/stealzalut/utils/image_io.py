"""Image I/O utilities built on OpenCV, Pillow, pillow-heif, rawpy, and NumPy.

All public functions in this module work with NumPy arrays and never
depend on Qt, keeping them usable in headless / test environments.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import cast

import cv2
import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

# ── Type aliases ─────────────────────────────────────────────────────────────
ImageArray = NDArray[np.uint8]
"""HWC image in uint8 [0, 255]."""

FloatImage = NDArray[np.float64]
"""HWC image in float64 [0.0, 1.0]."""

# ── Constants ────────────────────────────────────────────────────────────────
HEIC_EXTENSIONS: frozenset[str] = frozenset({".heic", ".heif", ".hif"})
RAW_EXTENSIONS: frozenset[str] = frozenset(
    {
        ".dng",  # Adobe / Leica / Apple ProRAW
        ".cr2",  # Canon RAW 2
        ".cr3",  # Canon RAW 3
        ".crw",  # Canon original RAW
        ".nef",  # Nikon Electronic Format
        ".nrw",  # Nikon Coolpix RAW
        ".arw",  # Sony Alpha RAW
        ".srf",  # Sony RAW format
        ".sr2",  # Sony RAW format 2
        ".raf",  # Fujifilm RAW
        ".rw2",  # Panasonic Lumix RAW
        ".raw",  # Leica / Panasonic / Generic RAW
        ".orf",  # Olympus / OM System RAW
        ".pef",  # Pentax Electronic Format
        ".ptx",  # Pentax RAW
        ".rwl",  # Leica RAW
        ".3fr",  # Hasselblad RAW
        ".fff",  # Hasselblad RAW
        ".iiq",  # Phase One RAW
        ".mrw",  # Konica Minolta RAW
        ".x3f",  # Sigma Foveon RAW
    }
)
STANDARD_EXTENSIONS: frozenset[str] = frozenset(
    {".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp", ".webp"}
)

SUPPORTED_EXTENSIONS: frozenset[str] = frozenset(
    STANDARD_EXTENSIONS | HEIC_EXTENSIONS | RAW_EXTENSIONS
)


# ── Format-specific decoders ────────────────────────────────────────────────


def _load_heic_bgr(path: Path) -> ImageArray:
    """Load a HEIC/HEIF image as BGR uint8 NumPy array with orientation correction."""
    try:
        import pillow_heif
        from PIL import Image, ImageOps

        pillow_heif.register_heif_opener()
        with Image.open(path) as img:
            transposed = ImageOps.exif_transpose(img)
            rgb = np.array(transposed.convert("RGB"), dtype=np.uint8)
            return cast("ImageArray", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    except Exception as exc:
        raise ValueError(f"Failed to decode HEIC image {path}: {exc}") from exc


def _load_raw_bgr(path: Path) -> ImageArray:
    """Load a camera RAW/DNG image as BGR uint8 NumPy array."""
    try:
        import rawpy

        with rawpy.imread(str(path)) as raw:
            color_space_srgb = rawpy.ColorSpace.sRGB  # type: ignore[attr-defined]
            try:
                rgb = raw.postprocess(
                    use_camera_wb=True,
                    output_color=color_space_srgb,
                    output_bps=8,
                )
            except Exception:
                rgb = raw.postprocess(
                    use_auto_wb=True,
                    output_color=color_space_srgb,
                    output_bps=8,
                )
            return cast("ImageArray", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    except Exception as raw_exc:
        logger.debug("rawpy failed to decode %s (%s), falling back to Pillow", path, raw_exc)
        try:
            from PIL import Image, ImageOps

            with Image.open(path) as img:
                transposed = ImageOps.exif_transpose(img)
                rgb = np.array(transposed.convert("RGB"), dtype=np.uint8)
                return cast("ImageArray", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        except Exception as exc:
            raise ValueError(f"Failed to decode RAW image {path}: {exc}") from exc


def _load_standard_bgr(path: Path) -> ImageArray:
    """Load standard image formats using OpenCV with unicode & Pillow fallbacks."""
    # First attempt: standard cv2 imread
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is not None:
        return cast("ImageArray", image)

    # Second attempt: cv2 imdecode from byte buffer (handles accented / non-ASCII Windows paths)
    try:
        buf = np.fromfile(str(path), dtype=np.uint8)
        image = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if image is not None:
            return cast("ImageArray", image)
    except Exception:
        pass

    # Third attempt: Pillow fallback
    try:
        from PIL import Image, ImageOps

        with Image.open(path) as img:
            transposed = ImageOps.exif_transpose(img)
            rgb = np.array(transposed.convert("RGB"), dtype=np.uint8)
            return cast("ImageArray", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    except Exception as exc:
        raise ValueError(f"Failed to decode image: {path}") from exc


# ── Loading ──────────────────────────────────────────────────────────────────


def load_image_bgr(path: str | Path) -> ImageArray:
    """Load an image from disk as a **BGR** uint8 NumPy array.

    Supports:
    - Standard raster formats (.jpg, .png, .tiff, .webp, .bmp, etc.)
    - Apple / Mobile HEIC/HEIF (.heic, .heif, .hif)
    - Camera RAW & DNG (.cr2, .cr3, .nef, .arw, .raf, .rw2, .orf, .pef, .dng, etc.)

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

    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported image format '{path.suffix}'. "
            f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )

    if suffix in HEIC_EXTENSIONS:
        return _load_heic_bgr(path)
    if suffix in RAW_EXTENSIONS:
        return _load_raw_bgr(path)
    return _load_standard_bgr(path)


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

