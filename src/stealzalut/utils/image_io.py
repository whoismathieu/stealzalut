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


# ── Pictures with their unclipped colours ────────────────────────────────────

#: Longest edge of the analysis copy (must equal ``palette.ANALYSIS_MAX_SIZE``).
WIDE_ANALYSIS_SIZE = 256

#: Values of :attr:`WideImage.rendering`.
RENDERING_FILE = "file"
RENDERING_NEUTRAL_RAW = "neutral-raw"
RENDERING_CAMERA_PREVIEW = "camera-preview"


class WideImage(np.ndarray[Any, np.dtype[np.uint8]]):
    """An sRGB uint8 picture that also remembers its colours before clipping.

    The array itself is what is displayed: RGB, sRGB, 8 bits.  Converting a
    Display P3 picture to that loses every colour sRGB cannot hold (a phone's
    neons, for a start).  ``analysis_lab`` keeps them: the Oklab colours of
    the analysis copy, computed from the file's own colour space, without
    clipping.  The engine reads the look from those.

    Attributes:
        analysis_lab: Oklab ``(h, w, 3)`` of the analysis copy (same
            geometry as ``create_thumbnail(image, WIDE_ANALYSIS_SIZE)``), or
            ``None`` when the file is plain sRGB.  Arrays derived from this
            one (slices, reshapes, thumbnails) do **not** inherit it.
        source_space: Name of the colour space the file was in.
        native: The same picture, same shape, as the file encodes it: RGB
            uint8 **in the file's own space** (``native_space``), or
            ``None`` when the file is plain sRGB.  A LUT is applied to
            these, so that colours sRGB cannot hold enter it intact; sRGB
            8 bits is only made afterwards, for the screen or a file.
        native_space: The :class:`~stealzalut.core.rgb_spaces.RGBSpace` of
            ``native`` (``None`` without ``native``).
        rendering: Where the colours come from: :data:`RENDERING_FILE` (the
            file's own pixels), :data:`RENDERING_NEUTRAL_RAW` (a RAW file
            developed here, see :func:`develop_raw_linear`) or
            :data:`RENDERING_CAMERA_PREVIEW` (the JPEG the camera embedded in
            a RAW file, with the camera's own look).
    """

    analysis_lab: NDArray[np.float64] | None
    source_space: str
    native: NDArray[np.uint8] | None
    native_space: Any
    rendering: str

    def __new__(
        cls,
        srgb: NDArray[np.uint8],
        analysis_lab: NDArray[np.float64] | None = None,
        source_space: str = "sRGB",
        native: NDArray[np.uint8] | None = None,
        native_space: Any = None,
        rendering: str = "file",
    ) -> WideImage:
        obj = np.asarray(srgb, dtype=np.uint8).view(cls)
        obj.analysis_lab = analysis_lab
        obj.source_space = source_space
        obj.native = native
        obj.native_space = native_space if native is not None else None
        obj.rendering = rendering
        return obj

    def __array_finalize__(self, obj: Any) -> None:
        # A view or a copy is another set of pixels: the analysis no longer fits
        self.analysis_lab = None
        self.source_space = "sRGB"
        self.native = None
        self.native_space = None
        self.rendering = RENDERING_FILE


def _profile_description(img: Any) -> str:
    """Lower-case description of a Pillow image's ICC profile ("" if none)."""
    icc = img.info.get("icc_profile")
    if not icc or img.mode not in ("RGB", "RGBA"):
        return ""
    try:
        import io

        from PIL import ImageCms

        profile = ImageCms.ImageCmsProfile(io.BytesIO(icc))
        return str(ImageCms.getProfileDescription(profile)).lower()
    except Exception:
        return ""


def _known_space(description: str) -> Any:
    """The RGB space a profile description names, or ``None`` if not a known wide one.

    Profiles of the same space go by many names ("Display P3", "DCI-P3 D65",
    "AdobeRGB1998", "ROMM RGB"…): the match is a case-insensitive substring.
    """
    description = description.lower()
    from stealzalut.core.rgb_spaces import ADOBE_RGB, DISPLAY_P3, PROPHOTO_RGB

    if "p3" in description:
        return DISPLAY_P3
    if any(name in description for name in ("adobe", "clayrgb", "oprgb")):
        return ADOBE_RGB
    if "prophoto" in description or "romm" in description:
        return PROPHOTO_RGB
    return None


def _open_for_analysis(path: Path) -> Any:
    """Pillow image of *path* in its own colour space (embedded preview for RAW files)."""
    from PIL import Image

    suffix = path.suffix.lower()
    if suffix in RAW_EXTENSIONS:
        import io

        import rawpy

        with rawpy.imread(str(path)) as raw:
            thumb = raw.extract_thumb()
        if getattr(thumb, "format", None) != rawpy.ThumbFormat.JPEG:  # type: ignore[attr-defined]
            return None
        return Image.open(io.BytesIO(bytes(thumb.data)))
    if suffix in HEIC_EXTENSIONS:
        import pillow_heif

        pillow_heif.register_heif_opener()
    return Image.open(path)


def _native_pixels(path: Path) -> tuple[NDArray[np.uint8], Any] | None:
    """The picture as its file encodes it: RGB uint8 in its own space, and that space.

    ``None`` when the file is sRGB, has no profile, or names a space the
    application does not know: the sRGB picture is then all there is.
    """
    try:
        from PIL import ImageOps

        img = _open_for_analysis(path)
        if img is None:
            return None
        with img:
            space = _known_space(_profile_description(img))
            if space is None:
                return None
            pixels = np.array(ImageOps.exif_transpose(img).convert("RGB"), dtype=np.uint8)
        return pixels, space
    except Exception as exc:
        logger.debug("Native pixels of %s skipped: %s", path, exc)
        return None


def _analysis_of(
    pixels: NDArray[np.uint8], space: Any, shape: tuple[int, int]
) -> NDArray[np.float64]:
    """Unclipped Oklab analysis copy ``(h, w, 3)`` of *pixels* in *space*."""
    small = cv2.resize(pixels, (shape[1], shape[0]), interpolation=cv2.INTER_AREA)
    lab = space.to_oklab(small.reshape(-1, 3).astype(np.float64) / 255.0)
    return cast("NDArray[np.float64]", lab.reshape(shape[0], shape[1], 3))


def load_image_rgb(path: str | Path, auto_develop_raw: bool = True) -> WideImage:
    """Load an image as **RGB** sRGB uint8, with its unclipped colours for the engine.

    Same files and errors as :func:`load_image_bgr`.  When the file is in a
    wider space than sRGB (Display P3, Adobe RGB, ProPhoto), the result
    carries ``analysis_lab`` and ``native``; see :class:`WideImage`.

    A RAW / DNG file is developed here on a neutral base
    (:func:`develop_raw_linear`) and comes back with its pixels in Display P3.
    With ``auto_develop_raw=False``, or when the file cannot be demosaiced,
    the JPEG the camera embedded in it is used instead.
    """
    path = Path(path)
    rendering = RENDERING_FILE
    if path.suffix.lower() in RAW_EXTENSIONS and path.exists():
        if auto_develop_raw:
            developed = _develop_raw_file(path)
            if developed is not None:
                return developed
        preview = _raw_preview_bgr(path)
        if preview is not None:
            rgb = bgr_to_rgb(preview)
            rendering = RENDERING_CAMERA_PREVIEW
        else:
            # No usable preview: develop after all, then whatever Pillow reads
            developed = None if auto_develop_raw else _develop_raw_file(path)
            if developed is not None:
                return developed
            rgb = bgr_to_rgb(_load_pillow_raw_bgr(path))
    else:
        rgb = bgr_to_rgb(load_image_bgr(path, auto_develop_raw=auto_develop_raw))
    small_shape = _analysis_shape(rgb.shape[0], rgb.shape[1])
    found = _native_pixels(path)
    if found is None:
        return WideImage(rgb, rendering=rendering)
    native, space = found
    analysis = _analysis_of(native, space, small_shape)
    if native.shape != rgb.shape:
        # Not the picture that is displayed: its colours still inform the engine.
        return WideImage(rgb, analysis, str(space.label), rendering=rendering)
    return WideImage(rgb, analysis, str(space.label), native, space, rendering)


def _analysis_shape(height: int, width: int) -> tuple[int, int]:
    """``(h, w)`` of the analysis copy: what :func:`create_thumbnail` gives."""
    longest = max(height, width)
    if longest <= WIDE_ANALYSIS_SIZE:
        return height, width
    scale = WIDE_ANALYSIS_SIZE / longest
    return int(height * scale), int(width * scale)


# ── Colour management ────────────────────────────────────────────────────────


def _pil_to_srgb(img: Any) -> Any:
    """Convert a Pillow image carrying a non-sRGB ICC profile to sRGB.

    Phones save their pictures (and the previews embedded in their DNG
    files) in Display P3.  Reading those values as sRGB mutes every colour
    and shifts hues, and the engine works in sRGB.  Images without a
    profile, or already in sRGB, are returned untouched.
    """
    icc = img.info.get("icc_profile")
    if not icc or img.mode not in ("RGB", "RGBA"):
        return img
    try:
        import io

        from PIL import ImageCms

        source = ImageCms.ImageCmsProfile(io.BytesIO(icc))
        if "srgb" in ImageCms.getProfileDescription(source).lower():
            return img
        converted = ImageCms.profileToProfile(
            img.convert("RGB"), source, ImageCms.createProfile("sRGB"), outputMode="RGB"
        )
        return img if converted is None else converted
    except Exception as exc:
        logger.debug("ICC conversion to sRGB skipped: %s", exc)
        return img


def _has_wide_gamut_profile(path: Path) -> bool:
    """True when the file carries an ICC profile that is not sRGB (header read only)."""
    try:
        import io

        from PIL import Image, ImageCms

        with Image.open(path) as img:
            icc = img.info.get("icc_profile")
            if not icc or img.mode not in ("RGB", "RGBA"):
                return False
            description = ImageCms.getProfileDescription(ImageCms.ImageCmsProfile(io.BytesIO(icc)))
            return "srgb" not in description.lower()
    except Exception:
        return False


# ── Format-specific decoders ────────────────────────────────────────────────


def _load_heic_bgr(path: Path) -> ImageArray:
    """Load a HEIC/HEIF image as BGR uint8 NumPy array with orientation correction."""
    try:
        import pillow_heif
        from PIL import Image, ImageOps

        pillow_heif.register_heif_opener()
        with Image.open(path) as img:
            transposed = _pil_to_srgb(ImageOps.exif_transpose(img))
            rgb = np.array(transposed.convert("RGB"), dtype=np.uint8)
            return cast("ImageArray", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    except Exception as exc:
        raise ValueError(f"Failed to decode HEIC image {path}: {exc}") from exc


# ── RAW / DNG: neutral development ───────────────────────────────────────────
#
# A LUT is computed from what the source looks like *before* it.  For a RAW
# file that is the host's base rendering (Adobe Standard, sliders at zero),
# not the camera's JPEG, and never something that depends on the picture's
# histogram: every picture goes through the same fixed operations.

#: DNG tags read by :func:`read_dng_tags`.
_TAG_SUB_IFDS = 330
_TAG_BASELINE_EXPOSURE = 50730
_TAG_PROFILE_GAIN_TABLE_MAP = 52525

#: BaselineExposure outside this range (EV) is taken for a broken tag.
BASELINE_EXPOSURE_RANGE = (-4.0, 4.0)
#: Linear value above which a positive exposure rolls off instead of clipping.
EXPOSURE_KNEE = 0.8

#: Default base tone curve of Adobe Camera Raw ("ACR3"), linear in → linear out,
#: sampled every 1/128 from ``dng_tone_curve_acr3_default`` of the Adobe DNG
#: SDK (``dng_render.cpp``, 1025 entries there).  A profile without a curve of
#: its own, Adobe Standard included, is rendered with it.
ACR3_BASE_CURVE: NDArray[np.float64] = np.array(
    [
        0.00000, 0.00623, 0.01485, 0.02643, 0.04002, 0.05483, 0.07057, 0.08710,
        0.10433, 0.12218, 0.14061, 0.15956, 0.17901, 0.19893, 0.21929, 0.23963,
        0.25961, 0.27918, 0.29833, 0.31704, 0.33531, 0.35313, 0.37050, 0.38742,
        0.40389, 0.41993, 0.43554, 0.45073, 0.46551, 0.47989, 0.49387, 0.50747,
        0.52069, 0.53356, 0.54607, 0.55824, 0.57007, 0.58159, 0.59281, 0.60373,
        0.61436, 0.62472, 0.63482, 0.64466, 0.65426, 0.66362, 0.67275, 0.68165,
        0.69035, 0.69883, 0.70711, 0.71520, 0.72309, 0.73081, 0.73834, 0.74570,
        0.75289, 0.75992, 0.76679, 0.77350, 0.78006, 0.78647, 0.79274, 0.79887,
        0.80486, 0.81072, 0.81645, 0.82205, 0.82753, 0.83288, 0.83812, 0.84324,
        0.84825, 0.85315, 0.85794, 0.86262, 0.86720, 0.87168, 0.87606, 0.88034,
        0.88453, 0.88862, 0.89262, 0.89653, 0.90035, 0.90408, 0.90773, 0.91130,
        0.91478, 0.91818, 0.92150, 0.92475, 0.92792, 0.93101, 0.93403, 0.93698,
        0.93986, 0.94266, 0.94540, 0.94807, 0.95067, 0.95320, 0.95567, 0.95808,
        0.96042, 0.96271, 0.96493, 0.96709, 0.96919, 0.97123, 0.97322, 0.97515,
        0.97702, 0.97884, 0.98061, 0.98232, 0.98398, 0.98558, 0.98714, 0.98864,
        0.99009, 0.99150, 0.99285, 0.99416, 0.99542, 0.99664, 0.99780, 0.99892,
        1.00000,
    ]
)  # fmt: skip
_CURVE_X: NDArray[np.float64] = np.linspace(0.0, 1.0, ACR3_BASE_CURVE.size, dtype=np.float64)

#: Rows developed at a time (bounds the memory of a 50-megapixel file).
_DEVELOP_ROWS = 256

_TIFF_TYPES = {1: "B", 3: "H", 4: "I", 5: "II", 7: "B", 8: "h", 9: "i", 10: "ii", 11: "f", 12: "d"}


def read_dng_tags(path: Path) -> tuple[float, bool]:
    """BaselineExposure (EV) and presence of a ProfileGainTableMap in a DNG file.

    The main directory and its sub-directories are read: the gain table map of
    an Apple ProRAW file sits with the raw pixels, not in the first directory.
    Any other file, or a file that cannot be parsed, gives ``(0.0, False)``.
    """
    if path.suffix.lower() != ".dng":
        return 0.0, False
    import struct

    baseline, gain_map = 0.0, False
    try:
        with path.open("rb") as handle:
            order = {b"II": "<", b"MM": ">"}.get(handle.read(2))
            if order is None or struct.unpack(order + "H", handle.read(2))[0] != 42:
                return 0.0, False

            def directory(offset: int) -> dict[int, tuple[int, int, bytes]]:
                handle.seek(offset)
                (count,) = struct.unpack(order + "H", handle.read(2))
                entries = {}
                for _ in range(min(count, 512)):
                    tag, kind, number, raw = struct.unpack(order + "HHI4s", handle.read(12))
                    entries[tag] = (kind, number, raw)
                return entries

            def values(entry: tuple[int, int, bytes], limit: int = 16) -> tuple[float, ...]:
                kind, number, raw = entry
                code = _TIFF_TYPES.get(kind)
                if code is None:
                    return ()
                size = struct.calcsize(order + code)
                number = min(number, limit)
                if size * number <= 4:
                    data = raw[: size * number]
                else:
                    handle.seek(struct.unpack(order + "I", raw)[0])
                    data = handle.read(size * number)
                flat = struct.unpack(order + code * number, data)
                if kind in (5, 10):
                    return tuple(
                        flat[i] / flat[i + 1] if flat[i + 1] else 0.0
                        for i in range(0, len(flat), 2)
                    )
                return tuple(float(v) for v in flat)

            (first,) = struct.unpack(order + "I", handle.read(4))
            main = directory(first)
            subs = [int(v) for v in values(main[_TAG_SUB_IFDS])] if _TAG_SUB_IFDS in main else []
            for entries in [main, *[directory(offset) for offset in subs]]:
                if _TAG_PROFILE_GAIN_TABLE_MAP in entries:
                    gain_map = True
                if _TAG_BASELINE_EXPOSURE in entries:
                    found = values(entries[_TAG_BASELINE_EXPOSURE])
                    if found and np.isfinite(found[0]):
                        baseline = float(found[0])
    except Exception as exc:
        logger.debug("Could not read the DNG tags of %s: %s", path, exc)
        return 0.0, False
    low, high = BASELINE_EXPOSURE_RANGE
    if not low <= baseline <= high:
        baseline = 0.0
    return baseline, gain_map


def _read_dng_baseline_exposure(path: Path) -> float:
    """The DNG BaselineExposure tag (50730) in EV; 0.0 when absent or not a DNG."""
    return read_dng_tags(path)[0]


def exposure_curve(value: NDArray[np.float64], exposure_ev: float) -> NDArray[np.float64]:
    """Apply *exposure_ev* stops to linear values in ``[0, 1]``, staying in ``[0, 1]``.

    A fixed function of the value, increasing, with 0 → 0 and 1 → 1 for any
    non-zero exposure:

    - positive: a plain gain ``2 ** ev`` up to :data:`EXPOSURE_KNEE`, then a
      shoulder with the same slope at the knee that reaches 1 for the brightest
      input.  A DNG whose BaselineExposure is large (+1.6 EV on Apple ProRAW,
      which counts on its gain table map to pull the highlights back) keeps
      its highlights instead of losing everything above ``2 ** -ev``;
    - negative: the darkening of the DNG SDK (``dng_function_exposure_tone``),
      a slope ``2 ** ev`` up to 0.25 and a parabola that still maps 1 to 1.
    """
    value = np.clip(value, 0.0, 1.0)
    if exposure_ev > 0.0:
        gain = 2.0**exposure_ev
        lifted = value * gain
        knee = EXPOSURE_KNEE
        power = (gain - knee) / (1.0 - knee)
        over = np.clip((lifted - knee) / (gain - knee), 0.0, 1.0)
        shoulder = knee + (1.0 - knee) * (1.0 - (1.0 - over) ** power)
        return cast("NDArray[np.float64]", np.where(lifted <= knee, lifted, shoulder))
    if exposure_ev < 0.0:
        slope = 2.0**exposure_ev
        a = 16.0 / 9.0 * (1.0 - slope)
        b = slope - 0.5 * a
        c = 1.0 - a - b
        curved = (a * value + b) * value + c
        return cast("NDArray[np.float64]", np.where(value <= 0.25, value * slope, curved))
    return value


def base_curve(value: NDArray[np.float64]) -> NDArray[np.float64]:
    """Adobe's default base tone curve on linear values in ``[0, 1]``."""
    return np.interp(value, _CURVE_X, ACR3_BASE_CURVE)


def develop_raw_linear(
    linear_prophoto: NDArray[np.float64], baseline_exposure: float = 0.0
) -> NDArray[np.float64]:
    """Neutral development of demosaiced pixels: linear ProPhoto in, linear ProPhoto out.

    Two fixed steps, the same for every picture (nothing is measured on it):

    1. the file's BaselineExposure (:func:`exposure_curve`), as a scale of the
       three channels by what it does to the largest one, so that hue and
       saturation are untouched;
    2. Adobe's default base curve (:func:`base_curve`) with the DNG rule for
       curves: the largest and the smallest channel go through the curve, the
       middle one keeps its relative position between them.  Hue is constant
       in the working space; there is no saturation or vibrance step.

    Args:
        linear_prophoto: ``(N, 3)`` linear ProPhoto RGB, scene white near 1.
        baseline_exposure: Stops, from the DNG tag (0 for other RAW files).

    Returns:
        ``(N, 3)`` linear ProPhoto RGB in ``[0, 1]``, ready to be encoded.
    """
    rgb = np.clip(np.asarray(linear_prophoto, dtype=np.float64), 0.0, 1.0)
    largest = rgb.max(axis=1, keepdims=True)
    if baseline_exposure != 0.0:
        exposed = exposure_curve(largest, baseline_exposure)
        rgb = rgb * np.divide(exposed, largest, out=np.ones_like(largest), where=largest > 0.0)
        largest = exposed
    smallest = rgb.min(axis=1, keepdims=True)
    top, bottom = base_curve(largest), base_curve(smallest)
    span = largest - smallest
    position = np.divide(rgb - smallest, span, out=np.zeros_like(rgb), where=span > 0.0)
    return cast("NDArray[np.float64]", bottom + position * (top - bottom))


def _demosaic_linear(raw: Any) -> NDArray[np.uint16]:
    """Demosaic *raw* (an open rawpy file) to 16-bit linear ProPhoto RGB.

    White balance as shot; no automatic brightening, no tone curve.
    """
    import rawpy

    settings: dict[str, Any] = {
        "output_color": rawpy.ColorSpace.ProPhoto,  # type: ignore[attr-defined]
        "output_bps": 16,
        "gamma": (1.0, 1.0),
        "no_auto_bright": True,
    }
    try:
        linear = raw.postprocess(use_camera_wb=True, **settings)
    except Exception as exc:
        logger.debug("No camera white balance (%s): using LibRaw's automatic one", exc)
        linear = raw.postprocess(use_auto_wb=True, **settings)
    if not isinstance(linear, np.ndarray) or linear.ndim != 3 or linear.shape[2] != 3:
        raise ValueError("LibRaw did not return an RGB picture")
    if linear.dtype != np.uint16:
        raise ValueError(f"LibRaw returned {linear.dtype} pixels, expected 16 bits")
    return linear


def _develop_rows(
    rows: NDArray[np.uint16], baseline_exposure: float
) -> NDArray[np.float32]:
    """:func:`develop_raw_linear` on 16-bit pixels, through tables (fast path).

    The exposure and the curve are functions of one value: they are tabulated
    on the 65 536 values a channel takes.  Same result as the reference
    function to within the 16-bit step.
    """
    levels = np.arange(65536, dtype=np.float64) / 65535.0
    exposed = exposure_curve(levels, baseline_exposure)
    ratio = np.divide(exposed, levels, out=np.ones_like(levels), where=levels > 0.0)
    curve = base_curve(levels).astype(np.float32)
    top_table = base_curve(exposed).astype(np.float32)

    largest = rows.max(axis=2)
    smallest = rows.min(axis=2)
    top = top_table[largest]
    low = np.rint(smallest * ratio.astype(np.float32)[largest]).astype(np.uint16)
    bottom = curve[low]
    span = (largest - smallest).astype(np.float32)
    scale = np.divide(top - bottom, span, out=np.zeros_like(span), where=span > 0.0)
    offset = (rows - smallest[..., None]).astype(np.float32)
    return cast("NDArray[np.float32]", bottom[..., None] + offset * scale[..., None])


def develop_raw_pixels(linear16: NDArray[np.uint16], baseline_exposure: float = 0.0) -> WideImage:
    """A :class:`WideImage` from 16-bit linear ProPhoto pixels (see :func:`develop_raw_linear`).

    The displayed array is sRGB; ``native`` holds the same pixels in Display
    P3; ``analysis_lab`` is computed from the developed floating-point values,
    before any gamut is clipped (ProPhoto holds colours neither sRGB nor
    Display P3 can).
    """
    from stealzalut.core.colorspace import linear_to_oklab
    from stealzalut.core.rgb_spaces import DISPLAY_P3, PROPHOTO_RGB, SRGB

    height, width = linear16.shape[:2]
    to_srgb = np.asarray(PROPHOTO_RGB.to_linear_srgb, dtype=np.float64)
    to_p3 = np.linalg.inv(np.asarray(DISPLAY_P3.to_linear_srgb, dtype=np.float64)) @ to_srgb
    # sRGB and Display P3 share one transfer curve: one table encodes both
    steps = 65535
    encode = np.rint(SRGB.encode(np.arange(steps + 1, dtype=np.float64) / steps) * 255.0).astype(
        np.uint8
    )

    def encoded(developed: NDArray[np.float32], matrix: NDArray[np.float64]) -> NDArray[np.uint8]:
        linear = np.clip(developed @ matrix.T.astype(np.float32), 0.0, 1.0)
        return encode[np.rint(linear * steps).astype(np.uint16)]

    # The analysis copy is an average of the *developed* pixels.  Reducing the
    # picture first and developing it afterwards would average the light before
    # the curve: one bright lamp then lifts its dark neighbours, and the engine
    # would read a brighter picture than the one the LUT is applied to.  Each
    # band is therefore reduced once developed, in a cube-root (perceptual)
    # encoding, by a whole factor; the final size is reached at the end.
    small_h, small_w = _analysis_shape(height, width)
    factor = 1
    while factor < 16 and height // (2 * factor) >= small_h and width // (2 * factor) >= small_w:
        factor *= 2

    srgb = np.empty((height, width, 3), dtype=np.uint8)
    native = np.empty((height, width, 3), dtype=np.uint8)
    reduced: list[NDArray[Any]] = []
    for start in range(0, height, _DEVELOP_ROWS):
        developed = _develop_rows(linear16[start : start + _DEVELOP_ROWS], baseline_exposure)
        srgb[start : start + _DEVELOP_ROWS] = encoded(developed, to_srgb)
        native[start : start + _DEVELOP_ROWS] = encoded(developed, to_p3)
        perceptual: NDArray[Any] = np.cbrt(developed)
        if factor > 1:
            rows = max(1, round(developed.shape[0] / factor))
            perceptual = cv2.resize(
                perceptual, (width // factor, rows), interpolation=cv2.INTER_AREA
            )
        reduced.append(perceptual)
    small = np.vstack(reduced)
    if small.shape[:2] != (small_h, small_w):
        small = cv2.resize(small, (small_w, small_h), interpolation=cv2.INTER_AREA)
    linear = small.reshape(-1, 3).astype(np.float64) ** 3
    analysis = linear_to_oklab(linear @ to_srgb.T).reshape(small_h, small_w, 3)
    return WideImage(
        srgb, analysis, str(DISPLAY_P3.label), native, DISPLAY_P3, RENDERING_NEUTRAL_RAW
    )


def _develop_raw_file(path: Path) -> WideImage | None:
    """Develop the RAW file at *path*, or ``None`` when LibRaw cannot demosaic it."""
    try:
        import rawpy

        with rawpy.imread(str(path)) as raw:
            linear = _demosaic_linear(raw)
    except Exception as exc:
        logger.info("Cannot develop %s (%s): falling back to its embedded preview", path, exc)
        return None
    baseline, gain_map = read_dng_tags(path)
    logger.info(
        "Developed %s: %dx%d, BaselineExposure %+.2f EV%s",
        path.name,
        linear.shape[1],
        linear.shape[0],
        baseline,
        " (gain table map ignored, as outside the file's own profile)" if gain_map else "",
    )
    return develop_raw_pixels(linear, baseline)


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
                transposed = _pil_to_srgb(ImageOps.exif_transpose(img))
                rgb = np.array(transposed.convert("RGB"), dtype=np.uint8)
                h, w = rgb.shape[:2]
                # A small thumbnail is not a picture to compute a LUT from
                if max(h, w) >= 1024:
                    logger.info("Using embedded camera preview JPEG (%dx%d)", w, h)
                    return cast("ImageArray", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    except Exception as exc:
        logger.debug("Embedded preview extraction skipped: %s", exc)
    return None


def _raw_preview_bgr(path: Path) -> ImageArray | None:
    """The JPEG the camera embedded in the RAW file at *path*, as BGR (``None`` if none)."""
    try:
        import rawpy

        with rawpy.imread(str(path)) as raw:
            return _try_extract_embedded_preview(raw)
    except Exception as exc:
        logger.debug("rawpy cannot open %s: %s", path, exc)
        return None


def _load_raw_bgr(path: Path, auto_develop: bool = True) -> ImageArray:
    """Load a camera RAW/DNG image as BGR uint8.

    In order: the neutral development (when *auto_develop*), the JPEG embedded
    by the camera, the neutral development when it was not asked for but there
    is no preview, and Pillow for the DNG files LibRaw does not read.
    """
    developed = _develop_raw_file(path) if auto_develop else None
    if developed is not None:
        return rgb_to_bgr(np.asarray(developed))
    preview = _raw_preview_bgr(path)
    if preview is not None:
        return preview
    if not auto_develop:
        developed = _develop_raw_file(path)
        if developed is not None:
            return rgb_to_bgr(np.asarray(developed))
    return _load_pillow_raw_bgr(path)


def _load_pillow_raw_bgr(path: Path) -> ImageArray:
    """Last resort for a DNG file LibRaw does not read: its first picture, through Pillow."""
    try:
        from PIL import Image, ImageOps

        with Image.open(path) as img:
            transposed = _pil_to_srgb(ImageOps.exif_transpose(img))
            rgb = np.array(transposed.convert("RGB"), dtype=np.uint8)
            return cast("ImageArray", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    except Exception as exc:
        raise ValueError(f"Failed to decode RAW image {path}: {exc}") from exc


def _load_standard_bgr(path: Path) -> ImageArray:
    """Load standard image formats using OpenCV with unicode & Pillow fallbacks."""
    # OpenCV ignores ICC profiles: a Display P3 file goes through Pillow (below)
    if not _has_wide_gamut_profile(path):
        # First attempt: standard cv2 imread
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is not None:
            return cast("ImageArray", image)

        # Second attempt: cv2 imdecode from byte buffer (accented / non-ASCII Windows paths)
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
            transposed = _pil_to_srgb(ImageOps.exif_transpose(img))
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
    - Camera RAW & DNG (.cr2, .cr3, .nef, .arw, .raf, .rw2, .orf, .pef, .dng, etc.),
      developed on a neutral base (see :func:`develop_raw_linear`).

    Args:
        path: Path to the image file.
        auto_develop_raw: If True, a RAW / DNG file is demosaiced and developed here
            with fixed settings, the same for every picture.  If False (or when the
            file cannot be demosaiced), the JPEG embedded by the camera is used,
            with the camera's own look.

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

