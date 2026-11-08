"""Image I/O utilities built on OpenCV, Pillow, pillow-heif, rawpy, and NumPy.

All public functions in this module work with NumPy arrays and never
depend on Qt, keeping them usable in headless / test environments.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, cast

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


def _read_dng_baseline_exposure(path: Path) -> float:
    """Read the Adobe/DNG BaselineExposure tag (50730) if present.

    Returns exposure shift in EV (stops), e.g. 1.0 = +1 stop. Defaults to 0.0.
    """
    if path.suffix.lower() != ".dng":
        return 0.0
    try:
        from PIL import Image

        with Image.open(path) as img:
            tags = getattr(img, "tag_v2", None) or getattr(img, "tag", None)
            if tags and 50730 in tags:
                val = tags[50730]
                if isinstance(val, (int, float)):
                    return float(val)
                if (
                    hasattr(val, "numerator")
                    and hasattr(val, "denominator")
                    and val.denominator != 0
                ):
                    return float(val.numerator) / float(val.denominator)
                if isinstance(val, (tuple, list)) and len(val) >= 2 and val[1] != 0:
                    return float(val[0]) / float(val[1])
    except Exception as exc:
        logger.debug("Could not read BaselineExposure from %s: %s", path, exc)
    return 0.0


def _try_extract_embedded_preview(raw: Any) -> ImageArray | None:
    """Extract embedded camera-rendered JPEG preview if available and high-resolution."""
    try:
        import io

        import rawpy
        from PIL import Image, ImageOps

        if not hasattr(raw, "extract_thumb"):
            return None

        thumb = raw.extract_thumb()
        thumb_format_jpeg = rawpy.ThumbFormat.JPEG  # type: ignore[attr-defined]
        if getattr(thumb, "format", None) == thumb_format_jpeg and hasattr(thumb, "data"):
            with Image.open(io.BytesIO(thumb.data)) as img:
                transposed = ImageOps.exif_transpose(img)
                rgb = np.array(transposed.convert("RGB"), dtype=np.uint8)
                h, w = rgb.shape[:2]
                # If preview is high-resolution (longest edge >= 1024), use camera's processed image
                if max(h, w) >= 1024:
                    logger.info("Using embedded camera preview JPEG (%dx%d)", w, h)
                    return cast("ImageArray", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    except Exception as exc:
        logger.debug("Embedded preview extraction skipped: %s", exc)
    return None


def enhance_raw_photographic(image_bgr: ImageArray) -> ImageArray:
    """Enhance demosaiced raw image with photographic exposure, tone curve & vibrance.

    Raw sensor data demosaiced without a camera profile is typically linear-ish,
    underexposed (low midtone reflectance), flat in contrast, and desaturated.
    This function:
      1. Adaptively lifts underexposed midtones towards photographic 18% gray target
         using smooth power gamma without blowing highlights.
      2. Applies a photographic contrast S-curve (filmic smoothstep blend) to give punch.
      3. Enhances color vibrance in HSV space so colors match developed photographic intensity.
    """
    if image_bgr.size == 0 or image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        return image_bgr

    img_f = image_bgr.astype(np.float32) / 255.0

    # 1. Luminance analysis (Rec.709 coefficients)
    lum = 0.0722 * img_f[:, :, 0] + 0.7152 * img_f[:, :, 1] + 0.2126 * img_f[:, :, 2]
    med_lum = float(np.median(lum))

    # Standard photographic target for middle gray in sRGB is ~0.38 - 0.42
    if 0.005 < med_lum < 0.35:
        target_mid = 0.40
        gamma = float(np.clip(np.log(target_mid) / np.log(med_lum), 0.50, 1.0))
        img_f = np.power(img_f, gamma)

    # 2. Photographic S-curve (smooth contrast curve: 35% S-curve, 65% linear)
    s_curve = img_f * img_f * (3.0 - 2.0 * img_f)
    img_f = 0.65 * img_f + 0.35 * s_curve

    # 3. Vibrance boost in HSV space
    bgr_temp = np.clip(img_f * 255.0, 0, 255).astype(np.uint8)
    hsv = cv2.cvtColor(bgr_temp, cv2.COLOR_BGR2HSV).astype(np.float32)
    sat = hsv[:, :, 1] / 255.0
    sat_boosted = sat + 0.35 * sat * (1.0 - sat)
    hsv[:, :, 1] = np.clip(sat_boosted * 255.0, 0, 255)

    return cast("ImageArray", cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR))


def _load_raw_bgr(path: Path, auto_develop: bool = True) -> ImageArray:
    """Load a camera RAW/DNG image as BGR uint8 NumPy array with photographic intensity."""
    try:
        import rawpy

        with rawpy.imread(str(path)) as raw:
            # 1. Try camera-embedded high-resolution preview JPEG
            if auto_develop:
                embedded = _try_extract_embedded_preview(raw)
                if embedded is not None:
                    return embedded

            # 2. Demosaic via LibRaw with baseline exposure compensation
            color_space_srgb = rawpy.ColorSpace.sRGB  # type: ignore[attr-defined]
            baseline_ev = _read_dng_baseline_exposure(path)
            exp_shift = float(np.clip(2.0 ** baseline_ev, 1.0, 8.0)) if baseline_ev > 0.0 else 1.0

            try:
                rgb = raw.postprocess(
                    use_camera_wb=True,
                    output_color=color_space_srgb,
                    output_bps=8,
                    no_auto_bright=False,
                    auto_bright_thr=0.005,
                    exp_shift=exp_shift,
                    exp_preserve_highlights=0.5,
                )
            except Exception:
                try:
                    rgb = raw.postprocess(
                        use_auto_wb=True,
                        output_color=color_space_srgb,
                        output_bps=8,
                        no_auto_bright=False,
                        auto_bright_thr=0.005,
                        exp_shift=exp_shift,
                        exp_preserve_highlights=0.5,
                    )
                except Exception:
                    rgb = raw.postprocess(
                        use_auto_wb=True,
                        output_color=color_space_srgb,
                        output_bps=8,
                    )

            bgr = cast("ImageArray", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            if auto_develop:
                bgr = enhance_raw_photographic(bgr)
            return bgr
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


def load_image_bgr(path: str | Path, auto_develop_raw: bool = True) -> ImageArray:
    """Load an image from disk as a **BGR** uint8 NumPy array.

    Supports:
    - Standard raster formats (.jpg, .png, .tiff, .webp, .bmp, etc.)
    - Apple / Mobile HEIC/HEIF (.heic, .heif, .hif)
    - Camera RAW & DNG (.cr2, .cr3, .nef, .arw, .raf, .rw2, .orf, .pef, .dng, etc.)
      with automatic photographic tone-mapping and exposure enhancement.

    Args:
        path: Path to the image file.
        auto_develop_raw: If True, applies photographic tone curves, exposure compensation,
            and vibrance enhancement to camera RAW / DNG files so they display with natural
            brightness and contrast rather than appearing flat and dark.

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
        return _load_raw_bgr(path, auto_develop=auto_develop_raw)
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

