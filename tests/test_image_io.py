"""Unit tests for Image I/O utilities, including HEIC and DNG support."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pillow_heif
import pytest
from numpy.typing import NDArray
from PIL import Image

from stealzalut.core.rgb_spaces import DISPLAY_P3, PROPHOTO_RGB, SRGB
from stealzalut.utils.image_io import (
    ACR3_BASE_CURVE,
    EXPOSURE_KNEE,
    HEIC_EXTENSIONS,
    RAW_EXTENSIONS,
    RENDERING_CAMERA_PREVIEW,
    RENDERING_FILE,
    RENDERING_NEUTRAL_RAW,
    SUPPORTED_EXTENSIONS,
    _read_dng_baseline_exposure,
    base_curve,
    bgr_to_rgb,
    create_thumbnail,
    develop_raw_linear,
    develop_raw_pixels,
    exposure_curve,
    float_to_uint8,
    load_image_bgr,
    load_image_rgb,
    normalize_to_float,
    read_dng_tags,
    rgb_to_bgr,
)


@pytest.fixture()
def sample_bgr_image() -> NDArray[np.uint8]:
    """Create a 40x50 BGR test image."""
    img = np.zeros((40, 50, 3), dtype=np.uint8)
    img[:, :, 0] = 200  # Blue
    img[:, :, 1] = 100  # Green
    img[:, :, 2] = 50   # Red
    return img


@pytest.fixture()
def sample_heic_file(tmp_path: Path) -> Path:
    """Create a valid temporary .heic file on disk."""
    pillow_heif.register_heif_opener()
    heic_path = tmp_path / "test_photo.heic"
    # Create an RGB image: 60x80 with distinct colors
    img = Image.new("RGB", (80, 60), color=(12, 34, 56))
    img.save(heic_path, format="HEIF")
    return heic_path


# ── Format constants ─────────────────────────────────────────────────────────


class TestSupportedExtensions:
    """Verify supported image extensions list."""

    def test_contains_heic_variants(self) -> None:
        assert ".heic" in SUPPORTED_EXTENSIONS
        assert ".heif" in SUPPORTED_EXTENSIONS
        assert ".hif" in SUPPORTED_EXTENSIONS
        assert ".heic" in HEIC_EXTENSIONS

    def test_contains_camera_raw_formats(self) -> None:
        raw_samples = [
            ".dng", ".cr2", ".cr3", ".nef", ".arw",
            ".raf", ".rw2", ".orf", ".pef", ".raw",
        ]
        for ext in raw_samples:
            assert ext in SUPPORTED_EXTENSIONS
            assert ext in RAW_EXTENSIONS

    def test_contains_standard_formats(self) -> None:
        for ext in [".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp", ".webp"]:
            assert ext in SUPPORTED_EXTENSIONS


# ── HEIC loading tests ───────────────────────────────────────────────────────


class TestHeicLoading:
    """Tests for Apple HEIC / HEIF format loading."""

    def test_load_valid_heic(self, sample_heic_file: Path) -> None:
        bgr = load_image_bgr(sample_heic_file)
        assert bgr.dtype == np.uint8
        assert bgr.shape == (60, 80, 3)
        # Expected RGB (12, 34, 56) -> BGR (56, 34, 12)
        # Allow slight compression tolerance (+/- 2)
        np.testing.assert_allclose(bgr[10, 10], [56, 34, 12], atol=3)

    def test_load_heif_extension(self, sample_heic_file: Path, tmp_path: Path) -> None:
        heif_path = tmp_path / "renamed.heif"
        heif_path.write_bytes(sample_heic_file.read_bytes())
        bgr = load_image_bgr(heif_path)
        assert bgr.shape == (60, 80, 3)

    def test_load_corrupt_heic_raises_value_error(self, tmp_path: Path) -> None:
        corrupt = tmp_path / "corrupt.heic"
        corrupt.write_bytes(b"invalid heic data header")
        with pytest.raises(ValueError, match="Failed to decode HEIC image"):
            load_image_bgr(corrupt)


# ── Camera RAW & DNG loading tests ──────────────────────────────────────────


def _mock_raw(postprocess: object) -> MagicMock:
    """An open rawpy file whose demosaicing gives (or raises) *postprocess*."""
    raw = MagicMock()
    if isinstance(postprocess, (list, Exception)):
        raw.postprocess.side_effect = postprocess
    else:
        raw.postprocess.return_value = postprocess
    raw.__enter__.return_value = raw
    raw.__exit__.return_value = None
    return raw


def _embedded_jpeg(size: tuple[int, int] = (1200, 1600)) -> MagicMock:
    import io

    import rawpy

    data = io.BytesIO()
    Image.new("RGB", size, color=(180, 120, 60)).save(data, format="JPEG")
    thumb = MagicMock()
    thumb.format = rawpy.ThumbFormat.JPEG  # type: ignore[attr-defined]
    thumb.data = data.getvalue()
    return thumb


def _linear16(height: int = 100, width: int = 150, value: int = 9000) -> NDArray[np.uint16]:
    return np.full((height, width, 3), value, dtype=np.uint16)


class TestCameraRawLoading:
    """Tests for Camera RAW (Canon, Nikon, Sony, Fuji...) & DNG format loading."""

    @pytest.mark.parametrize("ext", [".dng", ".cr2", ".cr3", ".nef", ".arw", ".raf", ".rw2"])
    def test_load_raw_formats_with_rawpy(self, tmp_path: Path, ext: str) -> None:
        raw_path = tmp_path / f"sample{ext}"
        raw_path.write_bytes(b"dummy raw content")
        mock_raw = _mock_raw(_linear16())

        with patch("rawpy.imread", return_value=mock_raw):
            bgr = load_image_bgr(raw_path)

        assert bgr.shape == (100, 150, 3)
        assert bgr.dtype == np.uint8
        mock_raw.postprocess.assert_called_once()

    def test_the_raw_file_is_demosaiced_linear_wide_and_untouched(self, tmp_path: Path) -> None:
        """No automatic brightening, no tone curve, 16 bits, a wide gamut."""
        import rawpy

        raw_path = tmp_path / "sample.nef"
        raw_path.write_bytes(b"dummy raw content")
        mock_raw = _mock_raw(_linear16())
        with patch("rawpy.imread", return_value=mock_raw):
            load_image_rgb(raw_path)
        settings = mock_raw.postprocess.call_args.kwargs
        assert settings == {
            "use_camera_wb": True,
            "output_color": rawpy.ColorSpace.ProPhoto,  # type: ignore[attr-defined]
            "output_bps": 16,
            "gamma": (1.0, 1.0),
            "no_auto_bright": True,
        }

    def test_load_raw_fallback_auto_wb(self, tmp_path: Path) -> None:
        raw_path = tmp_path / "sample_nowb.cr2"
        raw_path.write_bytes(b"dummy raw content")
        mock_raw = _mock_raw([RuntimeError("No camera WB"), _linear16(40, 50)])

        with patch("rawpy.imread", return_value=mock_raw):
            bgr = load_image_bgr(raw_path)

        assert bgr.shape == (40, 50, 3)
        assert mock_raw.postprocess.call_count == 2
        assert mock_raw.postprocess.call_args.kwargs["use_auto_wb"] is True

    def test_load_dng_pillow_fallback_for_linear_dng(self, tmp_path: Path) -> None:
        dng_path = tmp_path / "linear.dng"
        img = Image.new("RGB", (30, 40), color=(100, 150, 200))
        img.save(dng_path, format="TIFF")

        with patch("rawpy.imread", side_effect=RuntimeError("Unsupported DNG compression")):
            bgr = load_image_bgr(dng_path)
            wide = load_image_rgb(dng_path)

        assert bgr.shape == (40, 30, 3)
        assert bgr[0, 0, 0] == 200  # Blue channel matches RGB (100, 150, 200)
        assert wide.rendering == RENDERING_FILE and tuple(wide[0, 0]) == (100, 150, 200)

    def test_load_corrupt_raw_raises_value_error(self, tmp_path: Path) -> None:
        corrupt = tmp_path / "corrupt.nef"
        corrupt.write_bytes(b"not a valid raw")
        with pytest.raises(ValueError, match="Failed to decode RAW image"):
            load_image_bgr(corrupt)
        with pytest.raises(ValueError, match="Failed to decode RAW image"):
            load_image_rgb(corrupt)

    def test_a_missing_raw_file_is_reported_as_missing(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_image_rgb(tmp_path / "nowhere.dng")

    def test_the_embedded_preview_is_not_the_default(self, tmp_path: Path) -> None:
        """The camera's JPEG carries the camera's look: the file is developed instead."""
        raw_path = tmp_path / "sample_preview.cr3"
        raw_path.write_bytes(b"dummy raw content")
        mock_raw = _mock_raw(_linear16(60, 80))
        mock_raw.extract_thumb.return_value = _embedded_jpeg()

        with patch("rawpy.imread", return_value=mock_raw):
            image = load_image_rgb(raw_path)

        assert image.shape == (60, 80, 3)
        assert image.rendering == RENDERING_NEUTRAL_RAW
        mock_raw.extract_thumb.assert_not_called()

    def test_the_embedded_preview_is_the_fallback(self, tmp_path: Path) -> None:
        raw_path = tmp_path / "sample_preview.cr3"
        raw_path.write_bytes(b"dummy raw content")
        mock_raw = _mock_raw(RuntimeError("cannot demosaic"))
        mock_raw.extract_thumb.return_value = _embedded_jpeg()

        with patch("rawpy.imread", return_value=mock_raw):
            image = load_image_rgb(raw_path)
            bgr = load_image_bgr(raw_path)

        assert image.shape == (1600, 1200, 3) == bgr.shape
        assert image.rendering == RENDERING_CAMERA_PREVIEW

    def test_the_embedded_preview_can_be_asked_for(self, tmp_path: Path) -> None:
        raw_path = tmp_path / "sample_preview.cr3"
        raw_path.write_bytes(b"dummy raw content")
        mock_raw = _mock_raw(_linear16(60, 80))
        mock_raw.extract_thumb.return_value = _embedded_jpeg()

        with patch("rawpy.imread", return_value=mock_raw):
            image = load_image_rgb(raw_path, auto_develop_raw=False)
            bgr = load_image_bgr(raw_path, auto_develop_raw=False)

        assert image.shape == (1600, 1200, 3) == bgr.shape
        assert image.rendering == RENDERING_CAMERA_PREVIEW
        mock_raw.postprocess.assert_not_called()

    def test_a_tiny_thumbnail_is_not_used_as_the_picture(self, tmp_path: Path) -> None:
        raw_path = tmp_path / "sample_thumb.cr2"
        raw_path.write_bytes(b"dummy raw content")
        mock_raw = _mock_raw(_linear16(60, 80))
        mock_raw.extract_thumb.return_value = _embedded_jpeg((160, 120))

        with patch("rawpy.imread", return_value=mock_raw):
            image = load_image_rgb(raw_path, auto_develop_raw=False)

        assert image.shape == (60, 80, 3)  # developed after all
        assert image.rendering == RENDERING_NEUTRAL_RAW


# ── Neutral RAW development ──────────────────────────────────────────────────


def _lab(linear_prophoto: NDArray[np.float64]) -> NDArray[np.float64]:
    encoded = PROPHOTO_RGB.encode(np.clip(linear_prophoto, 0.0, 1.0))
    return PROPHOTO_RGB.to_oklab(encoded)


class TestNeutralRawDevelopment:
    """The development is a fixed function of each pixel, hue-preserving, wide gamut."""

    def test_the_base_curve_is_adobe_s(self) -> None:
        assert ACR3_BASE_CURVE.shape == (129,)
        assert ACR3_BASE_CURVE[0] == 0.0 and ACR3_BASE_CURVE[-1] == 1.0
        assert np.all(np.diff(ACR3_BASE_CURVE) > 0.0)
        # A raw mid-grey around 10 % comes out near 20 %, one stop brighter
        assert base_curve(np.array([13 / 128]))[0] == pytest.approx(0.19893)
        assert base_curve(np.array([0.5]))[0] == pytest.approx(0.80486)

    def test_black_white_and_greys_stay_neutral(self) -> None:
        greys = np.repeat(np.linspace(0.0, 1.0, 33)[:, None], 3, axis=1)
        out = develop_raw_linear(greys)
        assert np.allclose(out[0], 0.0) and np.allclose(out[-1], 1.0)
        assert np.allclose(out[:, 0], out[:, 1]) and np.allclose(out[:, 1], out[:, 2])
        assert np.all(np.diff(out[:, 0]) > 0.0)

    def test_nothing_depends_on_the_rest_of_the_picture(self) -> None:
        """A night scene is not brightened: a pixel is developed the same
        whether its neighbours are dark or bright (no auto-exposure)."""
        rng = np.random.default_rng(3)
        pixel = np.array([[0.012, 0.010, 0.008]])
        night = np.vstack([pixel, rng.uniform(0.0, 0.02, (500, 3))])
        day = np.vstack([pixel, rng.uniform(0.3, 1.0, (500, 3))])
        alone = develop_raw_linear(pixel)[0]
        assert np.array_equal(develop_raw_linear(night)[0], alone)
        assert np.array_equal(develop_raw_linear(day)[0], alone)
        # ... and a dark picture stays dark: lightness well under a mid-grey
        assert _lab(develop_raw_linear(night))[:, 0].mean() < 0.30

    def test_the_curve_keeps_hue_and_adds_no_saturation_step(self) -> None:
        rng = np.random.default_rng(5)
        rgb = rng.uniform(0.02, 0.9, (400, 3))
        out = develop_raw_linear(rgb)

        def position(values: NDArray[np.float64]) -> NDArray[np.float64]:
            ordered = np.sort(values, axis=1)
            return (ordered[:, 1] - ordered[:, 0]) / (ordered[:, 2] - ordered[:, 0])

        # DNG rule: the middle channel keeps its place between the two others,
        # and the channels keep their order
        assert np.allclose(position(out), position(rgb), atol=1e-9)
        assert np.array_equal(np.argsort(out, axis=1), np.argsort(rgb, axis=1))
        # the extreme channels are exactly the curve of what they were
        assert np.allclose(out.max(axis=1), base_curve(rgb.max(axis=1)))
        assert np.allclose(out.min(axis=1), base_curve(rgb.min(axis=1)))

    def test_exposure_is_a_gain_then_a_shoulder(self) -> None:
        values = np.linspace(0.0, 1.0, 2001)
        for stops in (0.4, 1.0, 1.6152, 3.0):
            gain = 2.0**stops
            out = exposure_curve(values, stops)
            linear = values * gain <= EXPOSURE_KNEE
            assert np.allclose(out[linear], values[linear] * gain)
            assert np.all(np.diff(out) >= 0.0) and out[-1] == pytest.approx(1.0)
            assert out.max() <= 1.0
            # no step and no kink at the knee: the slope there is still the gain
            just_over = np.array([EXPOSURE_KNEE / gain, EXPOSURE_KNEE / gain + 1e-6])
            slope = np.diff(exposure_curve(just_over, stops))[0] / 1e-6
            assert slope == pytest.approx(gain, rel=1e-3)

    def test_a_large_baseline_exposure_does_not_erase_the_highlights(self) -> None:
        """Apple ProRAW asks for +1.6 EV: a plain gain would clip everything
        above 33 % of the range to white."""
        bright = np.array([0.34, 0.40, 0.50, 0.70])
        plain = np.minimum(bright * 2.0**1.6152, 1.0)
        assert np.all(plain == 1.0)
        kept = exposure_curve(bright, 1.6152)
        assert np.all(np.diff(kept) > 0.0) and kept[0] < 0.95

    def test_negative_exposure_darkens_and_keeps_white(self) -> None:
        values = np.linspace(0.0, 1.0, 501)
        out = exposure_curve(values, -1.0)
        assert np.allclose(out[values <= 0.25], values[values <= 0.25] * 0.5)
        assert np.all(np.diff(out) > 0.0) and out[-1] == pytest.approx(1.0)
        assert np.array_equal(exposure_curve(values, 0.0), values)

    def test_exposure_scales_the_three_channels_alike(self) -> None:
        rgb = np.array([[0.10, 0.05, 0.02], [0.30, 0.20, 0.25]])
        plain, lifted = develop_raw_linear(rgb), develop_raw_linear(rgb, 1.0)
        assert np.all(lifted > plain)
        # below the knee, +1 EV is exactly twice the light before the curve
        assert np.allclose(lifted[0], develop_raw_linear(rgb[:1] * 2.0)[0])

    def test_the_wide_image_holds_the_three_views(self) -> None:
        rng = np.random.default_rng(11)
        linear = rng.integers(0, 65535, (40, 60, 3), dtype=np.uint16)
        image = develop_raw_pixels(linear, 0.5)

        assert image.dtype == np.uint8 and image.shape == (40, 60, 3)
        assert image.rendering == RENDERING_NEUTRAL_RAW
        assert image.native is not None and image.native.shape == image.shape
        assert image.native_space is DISPLAY_P3 and image.source_space == DISPLAY_P3.label
        assert image.analysis_lab is not None and image.analysis_lab.shape == (40, 60, 3)

        reference = develop_raw_linear(linear.reshape(-1, 3) / 65535.0, 0.5)
        to_srgb = PROPHOTO_RGB.to_linear_srgb
        # sRGB for the screen and Display P3 for the LUT: the same development,
        # to within one 8-bit step of the reference
        srgb = SRGB.encode(np.clip(reference @ to_srgb.T, 0.0, 1.0)) * 255.0
        assert np.abs(np.asarray(image).reshape(-1, 3) - srgb).max() <= 1.0
        to_p3 = np.linalg.inv(DISPLAY_P3.to_linear_srgb) @ to_srgb
        p3 = DISPLAY_P3.encode(np.clip(reference @ to_p3.T, 0.0, 1.0)) * 255.0
        assert np.abs(image.native.reshape(-1, 3) - p3).max() <= 1.0
        # the analysis is the reference itself, in Oklab, with nothing clipped
        lab = PROPHOTO_RGB.to_oklab(PROPHOTO_RGB.encode(reference))
        assert np.allclose(image.analysis_lab.reshape(-1, 3), lab, atol=2e-3)

    def test_the_analysis_keeps_colours_no_screen_space_holds(self) -> None:
        green = np.zeros((20, 20, 3), dtype=np.uint16)
        green[..., 1] = 40000  # pure ProPhoto green: outside sRGB and Display P3
        image = develop_raw_pixels(green)
        assert image.analysis_lab is not None and image.native is not None
        wide = np.hypot(image.analysis_lab[..., 1], image.analysis_lab[..., 2]).mean()
        in_p3 = DISPLAY_P3.to_oklab(image.native.reshape(-1, 3) / 255.0)
        in_srgb = SRGB.to_oklab(np.asarray(image).reshape(-1, 3) / 255.0)
        chroma_p3 = np.hypot(in_p3[:, 1], in_p3[:, 2]).mean()
        assert wide > chroma_p3 + 0.02
        assert chroma_p3 > np.hypot(in_srgb[:, 1], in_srgb[:, 2]).mean()

    def test_the_analysis_copy_matches_the_pixels_of_a_large_picture(self) -> None:
        """Developed before it is reduced: a lamp does not lift its dark
        neighbours, and the engine reads the picture the LUT is applied to."""
        rng = np.random.default_rng(13)
        # A night scene: dark, nearly neutral, with small bright lights
        light = rng.uniform(0.01, 0.05, (600, 800, 1)) * rng.uniform(0.9, 1.1, (600, 800, 3))
        lamps = rng.uniform(0.0, 1.0, (600, 800)) > 0.95
        light[lamps] = rng.uniform(0.5, 1.0, (int(lamps.sum()), 1))
        linear = (light * 65535).astype(np.uint16)
        image = develop_raw_pixels(linear, 1.0)
        assert image.analysis_lab is not None and image.native is not None
        assert image.analysis_lab.shape == create_thumbnail(np.asarray(image), 256).shape
        small = create_thumbnail(image.native, 256).reshape(-1, 3) / 255.0
        shown = DISPLAY_P3.to_oklab(small)[:, 0].mean()
        assert image.analysis_lab[..., 0].mean() == pytest.approx(shown, abs=0.01)

    def test_the_baseline_exposure_of_the_file_is_applied(self, tmp_path: Path) -> None:
        path = tmp_path / "exposed.dng"
        path.write_bytes(_tiny_dng("<", baseline=(1, 1), gain_map=True))
        with patch("rawpy.imread", return_value=_mock_raw(_linear16(30, 30, 3000))):
            exposed = load_image_rgb(path)
        path.write_bytes(_tiny_dng("<", baseline=(0, 1), gain_map=False))
        with patch("rawpy.imread", return_value=_mock_raw(_linear16(30, 30, 3000))):
            plain = load_image_rgb(path)
        with patch("rawpy.imread", return_value=_mock_raw(_linear16(30, 30, 6000))):
            twice = load_image_rgb(path)
        assert int(exposed[0, 0, 0]) > int(plain[0, 0, 0])
        assert np.array_equal(np.asarray(exposed), np.asarray(twice))


# ── DNG Baseline Exposure tests ──────────────────────────────────────────────


class TestDngBaselineExposure:
    """Tests for reading Adobe DNG BaselineExposure metadata tag."""

    def test_read_baseline_exposure_from_dng(self, tmp_path: Path) -> None:
        from PIL import TiffImagePlugin

        dng_path = tmp_path / "test_baseline.dng"
        img = Image.new("RGB", (20, 20), color=(50, 50, 50))
        ifd = TiffImagePlugin.ImageFileDirectory_v2()
        ifd[50730] = 1.75  # BaselineExposure
        img.save(dng_path, format="TIFF", tiffinfo=ifd)

        ev = _read_dng_baseline_exposure(dng_path)
        assert abs(ev - 1.75) < 1e-4

    def test_read_baseline_exposure_non_dng(self, tmp_path: Path) -> None:
        jpg_path = tmp_path / "test.jpg"
        jpg_path.write_bytes(b"dummy")
        assert _read_dng_baseline_exposure(jpg_path) == 0.0

    @pytest.mark.parametrize("order", ["<", ">"])
    def test_tags_are_read_in_the_sub_directories(self, tmp_path: Path, order: str) -> None:
        """An Apple ProRAW file keeps its gain table map with the raw pixels."""
        path = tmp_path / "proraw.dng"
        path.write_bytes(_tiny_dng(order, baseline=(16152, 10000), gain_map=True))
        baseline, gain_map = read_dng_tags(path)
        assert baseline == pytest.approx(1.6152) and gain_map is True

        path.write_bytes(_tiny_dng(order, baseline=(-1, 2), gain_map=False))
        assert read_dng_tags(path) == (-0.5, False)

    def test_an_absurd_or_unreadable_tag_counts_for_nothing(self, tmp_path: Path) -> None:
        path = tmp_path / "odd.dng"
        path.write_bytes(_tiny_dng("<", baseline=(40, 1), gain_map=False))
        assert read_dng_tags(path) == (0.0, False)
        path.write_bytes(b"II*\x00garbage")
        assert read_dng_tags(path) == (0.0, False)
        path.write_bytes(b"")
        assert read_dng_tags(path) == (0.0, False)


def _tiny_dng(order: str, baseline: tuple[int, int], gain_map: bool) -> bytes:
    """A TIFF skeleton: main directory (SubIFDs, BaselineExposure) and one sub-directory."""
    import struct

    def entry(tag: int, kind: int, count: int, value: bytes) -> bytes:
        return struct.pack(order + "HHI", tag, kind, count) + value.ljust(4, b"\0")

    main_at = 8
    main_size = 2 + 2 * 12 + 4
    rational_at = main_at + main_size
    sub_at = rational_at + 8
    header = (b"II" if order == "<" else b"MM") + struct.pack(order + "HI", 42, main_at)
    main = (
        struct.pack(order + "H", 2)
        + entry(330, 4, 1, struct.pack(order + "I", sub_at))
        + entry(50730, 10, 1, struct.pack(order + "I", rational_at))
        + struct.pack(order + "I", 0)
    )
    rational = struct.pack(order + "ii", *baseline)
    tags = [entry(256, 4, 1, struct.pack(order + "I", 64))]
    if gain_map:
        tags.append(entry(52525, 7, 4, b"\x00\x00\x00\x30"))
    sub = struct.pack(order + "H", len(tags)) + b"".join(tags) + struct.pack(order + "I", 0)
    return header + main + rational + sub


# ── General Image I/O tests ──────────────────────────────────────────────────


class TestImageIoGeneral:
    """General error handling and helper tests."""

    def test_missing_file_raises_filenotfound(self) -> None:
        with pytest.raises(FileNotFoundError):
            load_image_bgr("nonexistent_image_12345.jpg")

    def test_unsupported_format_raises_value_error(self, tmp_path: Path) -> None:
        txt_path = tmp_path / "readme.txt"
        txt_path.write_text("not an image")
        with pytest.raises(ValueError, match="Unsupported image format"):
            load_image_bgr(txt_path)

    def test_load_standard_png(self, tmp_path: Path, sample_bgr_image: NDArray[np.uint8]) -> None:
        png_path = tmp_path / "image.png"
        cv2.imwrite(str(png_path), sample_bgr_image)
        loaded = load_image_bgr(png_path)
        np.testing.assert_array_equal(loaded, sample_bgr_image)

    def test_bgr_rgb_roundtrip(self, sample_bgr_image: NDArray[np.uint8]) -> None:
        rgb = bgr_to_rgb(sample_bgr_image)
        recovered_bgr = rgb_to_bgr(rgb)
        np.testing.assert_array_equal(recovered_bgr, sample_bgr_image)

    def test_normalize_and_float_to_uint8(self, sample_bgr_image: NDArray[np.uint8]) -> None:
        flt = normalize_to_float(sample_bgr_image)
        assert flt.dtype == np.float64
        assert flt.min() >= 0.0 and flt.max() <= 1.0
        uint8_arr = float_to_uint8(flt)
        np.testing.assert_array_equal(uint8_arr, sample_bgr_image)

    def test_create_thumbnail_preserves_aspect_ratio(
        self, sample_bgr_image: NDArray[np.uint8]
    ) -> None:
        thumb = create_thumbnail(sample_bgr_image, max_size=20)
        assert max(thumb.shape[:2]) <= 20
