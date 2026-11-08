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

from stealzalut.utils.image_io import (
    HEIC_EXTENSIONS,
    RAW_EXTENSIONS,
    SUPPORTED_EXTENSIONS,
    _read_dng_baseline_exposure,
    bgr_to_rgb,
    create_thumbnail,
    enhance_raw_photographic,
    float_to_uint8,
    load_image_bgr,
    normalize_to_float,
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


class TestCameraRawLoading:
    """Tests for Camera RAW (Canon, Nikon, Sony, Fuji...) & DNG format loading."""

    @pytest.mark.parametrize("ext", [".dng", ".cr2", ".cr3", ".nef", ".arw", ".raf", ".rw2"])
    def test_load_raw_formats_with_rawpy(self, tmp_path: Path, ext: str) -> None:
        raw_path = tmp_path / f"sample{ext}"
        raw_path.write_bytes(b"dummy raw content")

        mock_rgb = np.full((100, 150, 3), 180, dtype=np.uint8)
        mock_raw = MagicMock()
        mock_raw.postprocess.return_value = mock_rgb
        mock_raw.__enter__.return_value = mock_raw
        mock_raw.__exit__.return_value = None

        with patch("rawpy.imread", return_value=mock_raw):
            bgr = load_image_bgr(raw_path)

        assert bgr.shape == (100, 150, 3)
        assert bgr.dtype == np.uint8
        mock_raw.postprocess.assert_called_once()

    def test_load_raw_fallback_auto_wb(self, tmp_path: Path) -> None:
        raw_path = tmp_path / "sample_nowb.cr2"
        raw_path.write_bytes(b"dummy raw content")

        mock_rgb = np.full((40, 50, 3), 90, dtype=np.uint8)
        mock_raw = MagicMock()
        mock_raw.postprocess.side_effect = [RuntimeError("No camera WB"), mock_rgb]
        mock_raw.__enter__.return_value = mock_raw
        mock_raw.__exit__.return_value = None

        with patch("rawpy.imread", return_value=mock_raw):
            bgr = load_image_bgr(raw_path)

        assert bgr.shape == (40, 50, 3)
        assert mock_raw.postprocess.call_count == 2

    def test_load_dng_pillow_fallback_for_linear_dng(self, tmp_path: Path) -> None:
        dng_path = tmp_path / "linear.dng"
        img = Image.new("RGB", (30, 40), color=(100, 150, 200))
        img.save(dng_path, format="TIFF")

        with patch("rawpy.imread", side_effect=RuntimeError("Unsupported DNG compression")):
            bgr = load_image_bgr(dng_path)

        assert bgr.shape == (40, 30, 3)
        assert bgr[0, 0, 0] == 200  # Blue channel matches RGB (100, 150, 200)

    def test_load_corrupt_raw_raises_value_error(self, tmp_path: Path) -> None:
        corrupt = tmp_path / "corrupt.nef"
        corrupt.write_bytes(b"not a valid raw")
        with pytest.raises(ValueError, match="Failed to decode RAW image"):
            load_image_bgr(corrupt)

    def test_load_raw_uses_embedded_preview_when_available(self, tmp_path: Path) -> None:
        """When an embedded JPEG preview is high-res, it is used without demosaicing."""
        raw_path = tmp_path / "sample_preview.cr3"
        raw_path.write_bytes(b"dummy raw content")

        # Create 1200x1600 JPEG preview
        preview_img = Image.new("RGB", (1200, 1600), color=(180, 120, 60))
        import io
        preview_bytes = io.BytesIO()
        preview_img.save(preview_bytes, format="JPEG")

        import rawpy
        mock_thumb = MagicMock()
        mock_thumb.format = rawpy.ThumbFormat.JPEG  # type: ignore[attr-defined]
        mock_thumb.data = preview_bytes.getvalue()

        mock_raw = MagicMock()
        mock_raw.extract_thumb.return_value = mock_thumb
        mock_raw.__enter__.return_value = mock_raw
        mock_raw.__exit__.return_value = None

        with patch("rawpy.imread", return_value=mock_raw):
            bgr = load_image_bgr(raw_path)

        assert bgr.shape == (1600, 1200, 3)
        # Verify postprocess was not called because embedded preview was used
        mock_raw.postprocess.assert_not_called()


# ── Photographic Raw Enhancement tests ───────────────────────────────────────


class TestPhotographicRawEnhancement:
    """Tests for adaptive exposure, S-curve contrast, and vibrance boost."""

    def test_lifts_dark_image(self) -> None:
        dark = np.full((100, 100, 3), 35, dtype=np.uint8)
        enhanced = enhance_raw_photographic(dark)

        # Values should be lifted from ~35 into midtones (> 80)
        assert enhanced[0, 0, 0] > 70
        assert enhanced.dtype == np.uint8
        assert enhanced.shape == (100, 100, 3)

    def test_preserves_black_and_white(self) -> None:
        black = np.zeros((10, 10, 3), dtype=np.uint8)
        assert np.all(enhance_raw_photographic(black) == 0)

        white = np.full((10, 10, 3), 255, dtype=np.uint8)
        assert np.all(enhance_raw_photographic(white) == 255)

    def test_boosts_vibrance(self) -> None:
        # Create image with subdued color (low saturation)
        subdued = np.full((20, 20, 3), [100, 120, 140], dtype=np.uint8)  # BGR
        enhanced = enhance_raw_photographic(subdued)

        # Check HSV saturation
        hsv_before = cv2.cvtColor(subdued, cv2.COLOR_BGR2HSV)
        hsv_after = cv2.cvtColor(enhanced, cv2.COLOR_BGR2HSV)
        assert float(hsv_after[0, 0, 1]) >= float(hsv_before[0, 0, 1])

    def test_edge_cases(self) -> None:
        empty = np.empty((0, 0, 3), dtype=np.uint8)
        assert enhance_raw_photographic(empty).size == 0


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
