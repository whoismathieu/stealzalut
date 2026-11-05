"""Tests for stealzalut.gui — ImageDropZone, PreviewPanel, and MainWindow."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest
from PySide6.QtWidgets import QApplication

from stealzalut.config import Settings
from stealzalut.gui.main_window import MainWindow
from stealzalut.gui.widgets.image_drop_zone import ImageDropZone
from stealzalut.gui.widgets.preview_panel import PreviewPanel

# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture(scope="session")
def qapp() -> QApplication:
    """Create a single QApplication for the entire test session."""
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    return app


@pytest.fixture()
def sample_image_path(tmp_path: Path) -> Path:
    """Create a temporary valid PNG image on disk."""
    path = tmp_path / "sample.png"
    # Create 64x64 BGR image
    img = np.full((64, 64, 3), 120, dtype=np.uint8)
    cv2.imwrite(str(path), img)
    return path


@pytest.fixture()
def default_settings() -> Settings:
    return Settings(
        lut_default_size=33,
        preview_max_size=256,
        app_theme="dark",
    )


# ── ImageDropZone tests ──────────────────────────────────────────────────────


class TestImageDropZone:
    """Tests for ImageDropZone widget."""

    def test_initial_state(self, qapp: QApplication) -> None:
        zone = ImageDropZone()
        assert not zone.has_image
        assert zone.image_rgb is None
        assert zone.image_path is None

    def test_load_valid_image(self, qapp: QApplication, sample_image_path: Path) -> None:
        zone = ImageDropZone()
        loaded_events: list[Any] = []
        zone.image_loaded.connect(loaded_events.append)

        success = zone.load_from_path(sample_image_path)

        assert success is True
        assert zone.has_image is True
        assert zone.image_path == sample_image_path
        assert zone.image_rgb is not None
        assert zone.image_rgb.shape == (64, 64, 3)
        assert len(loaded_events) == 1
        assert isinstance(loaded_events[0], np.ndarray)

    def test_load_nonexistent_image(self, qapp: QApplication, tmp_path: Path) -> None:
        zone = ImageDropZone()
        success = zone.load_from_path(tmp_path / "does_not_exist.png")
        assert success is False
        assert not zone.has_image

    def test_load_heic_image(self, qapp: QApplication, tmp_path: Path) -> None:
        import pillow_heif
        from PIL import Image

        pillow_heif.register_heif_opener()
        heic_path = tmp_path / "photo.heic"
        img = Image.new("RGB", (50, 60), color=(10, 20, 30))
        img.save(heic_path, format="HEIF")

        zone = ImageDropZone()
        assert zone.load_from_path(heic_path) is True
        assert zone.has_image is True
        assert zone.image_rgb is not None
        assert zone.image_rgb.shape == (60, 50, 3)

    def test_load_dng_image(self, qapp: QApplication, tmp_path: Path) -> None:
        from unittest.mock import MagicMock, patch

        dng_path = tmp_path / "photo.dng"
        dng_path.write_bytes(b"mock raw bytes")

        mock_rgb = np.full((70, 90, 3), 150, dtype=np.uint8)
        mock_raw = MagicMock()
        mock_raw.postprocess.return_value = mock_rgb
        mock_raw.__enter__.return_value = mock_raw
        mock_raw.__exit__.return_value = None

        zone = ImageDropZone()
        with patch("rawpy.imread", return_value=mock_raw):
            assert zone.load_from_path(dng_path) is True
        assert zone.has_image is True
        assert zone.image_rgb is not None
        assert zone.image_rgb.shape == (70, 90, 3)

    def test_load_camera_raw_cr2_image(self, qapp: QApplication, tmp_path: Path) -> None:
        from unittest.mock import MagicMock, patch

        cr2_path = tmp_path / "photo.cr2"
        cr2_path.write_bytes(b"mock cr2 bytes")

        mock_rgb = np.full((70, 90, 3), 150, dtype=np.uint8)
        mock_raw = MagicMock()
        mock_raw.postprocess.return_value = mock_rgb
        mock_raw.__enter__.return_value = mock_raw
        mock_raw.__exit__.return_value = None

        zone = ImageDropZone()
        with patch("rawpy.imread", return_value=mock_raw):
            assert zone.load_from_path(cr2_path) is True
        assert zone.has_image is True
        assert zone.image_rgb is not None

    def test_clear_image(self, qapp: QApplication, sample_image_path: Path) -> None:
        zone = ImageDropZone()
        cleared_events: list[bool] = []
        zone.image_cleared.connect(lambda: cleared_events.append(True))

        zone.load_from_path(sample_image_path)
        assert zone.has_image is True

        zone.clear_image()
        assert not zone.has_image
        assert zone.image_rgb is None
        assert zone.image_path is None
        assert len(cleared_events) == 1


# ── PreviewPanel tests ───────────────────────────────────────────────────────


class TestPreviewPanel:
    """Tests for PreviewPanel widget."""

    def test_initial_state(self, qapp: QApplication) -> None:
        panel = PreviewPanel(title="Test Preview")
        assert panel._pixmap is None

    def test_set_image(self, qapp: QApplication) -> None:
        panel = PreviewPanel(title="Test Preview")
        sample_rgb = np.zeros((100, 100, 3), dtype=np.uint8)
        panel.set_image(sample_rgb)
        assert panel._pixmap is not None
        assert not panel._pixmap.isNull()

    def test_clear(self, qapp: QApplication) -> None:
        panel = PreviewPanel()
        sample_rgb = np.zeros((50, 50, 3), dtype=np.uint8)
        panel.set_image(sample_rgb)
        assert panel._pixmap is not None

        panel.clear()
        assert panel._pixmap is None


# ── MainWindow tests ─────────────────────────────────────────────────────────


class TestMainWindow:
    """Tests for MainWindow."""

    def test_initial_controls_state(self, qapp: QApplication, default_settings: Settings) -> None:
        window = MainWindow(settings=default_settings)

        # Export button disabled initially
        assert window._export_btn.isEnabled() is False

        # Size combo box populated and default selected
        assert window._size_combo.count() == 3
        sizes = [window._size_combo.itemData(i) for i in range(window._size_combo.count())]
        assert sizes == [17, 33, 64]
        assert window._size_combo.currentData() == 33

        # Algorithm combo box populated with all 7 methods and default selected
        assert window._algo_combo.count() == 7
        algos = [window._algo_combo.itemData(i) for i in range(window._algo_combo.count())]
        assert "mkl" in algos
        assert "reinhard" in algos
        assert "hm" in algos
        assert window._algo_combo.currentData() == "mkl"

    def test_export_button_enables_when_both_images_loaded(
        self,
        qapp: QApplication,
        default_settings: Settings,
        sample_image_path: Path,
    ) -> None:
        window = MainWindow(settings=default_settings)
        try:
            assert window._export_btn.isEnabled() is False

            # Load only source
            window._source_zone.load_from_path(sample_image_path)
            assert window._export_btn.isEnabled() is False

            # Load reference
            window._reference_zone.load_from_path(sample_image_path)
            assert window._export_btn.isEnabled() is True
        finally:
            window.stop_workers()

    def test_clear_all_resets_state(
        self,
        qapp: QApplication,
        default_settings: Settings,
        sample_image_path: Path,
    ) -> None:
        window = MainWindow(settings=default_settings)
        try:
            window._source_zone.load_from_path(sample_image_path)
            window._reference_zone.load_from_path(sample_image_path)
            assert window._export_btn.isEnabled() is True

            # Click Clear All
            window._clear_btn.click()

            assert window._source_zone.has_image is False
            assert window._reference_zone.has_image is False
            assert window._export_btn.isEnabled() is False
            assert window._progress_bar.value() == 0
        finally:
            window.stop_workers()

    def test_algo_change_triggers_preview_when_images_loaded(
        self,
        qapp: QApplication,
        default_settings: Settings,
        sample_image_path: Path,
    ) -> None:
        from unittest.mock import patch

        window = MainWindow(settings=default_settings)
        try:
            window._source_zone.load_from_path(sample_image_path)
            window._reference_zone.load_from_path(sample_image_path)

            with patch.object(window, "_run_transfer_preview") as mock_preview:
                reinhard_idx = window._algo_combo.findData("reinhard")
                assert reinhard_idx >= 0
                window._algo_combo.setCurrentIndex(reinhard_idx)
                mock_preview.assert_called_once()
        finally:
            window.stop_workers()
