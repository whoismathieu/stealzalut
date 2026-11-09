"""Tests for stealzalut.gui — ImageDropZone, PreviewPanel, and MainWindow."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, cast

import cv2
import numpy as np
import pytest
from PySide6.QtGui import QGuiApplication
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
    return cast("QApplication", app)


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

    def test_path_hidden_when_empty(self, qapp: QApplication) -> None:
        zone = ImageDropZone()
        assert zone.path_text == ""
        assert zone._path_row.isHidden()

    def test_path_shown_after_load(self, qapp: QApplication, sample_image_path: Path) -> None:
        zone = ImageDropZone()
        zone.load_from_path(sample_image_path)

        assert zone.path_text == str(sample_image_path.absolute())
        assert not zone._path_row.isHidden()
        assert zone._path_field.isReadOnly()

    def test_path_hidden_after_clear(self, qapp: QApplication, sample_image_path: Path) -> None:
        zone = ImageDropZone()
        zone.load_from_path(sample_image_path)
        zone.clear_image()

        assert zone.path_text == ""
        assert zone._path_row.isHidden()

    def test_failed_load_keeps_previous_path(
        self, qapp: QApplication, sample_image_path: Path, tmp_path: Path
    ) -> None:
        zone = ImageDropZone()
        zone.load_from_path(sample_image_path)
        zone.load_from_path(tmp_path / "missing.png")

        assert zone.path_text == str(sample_image_path.absolute())

    def test_copy_button_copies_path(self, qapp: QApplication, sample_image_path: Path) -> None:
        zone = ImageDropZone()
        zone.load_from_path(sample_image_path)

        zone._copy_btn.click()

        assert QGuiApplication.clipboard().text() == str(sample_image_path.absolute())
        assert zone._copy_btn.text() == "Copié ✓"


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

        # Algorithm combo box populated with all 8 methods and default selected
        assert window._algo_combo.count() == 8
        algos = [window._algo_combo.itemData(i) for i in range(window._algo_combo.count())]
        assert "oklab" in algos
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

    def test_superseded_preview_does_not_crash(
        self,
        qapp: QApplication,
        default_settings: Settings,
        sample_image_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Re-dropping an image while a preview runs must not destroy the thread.

        Regression: the running QThread lost its last reference and Qt aborted
        with "QThread: Destroyed while thread is still running".
        """
        import gc
        import time
        from unittest.mock import patch

        import stealzalut.gui.main_window as main_window_module
        from stealzalut.workers.transfer_worker import TransferWorker

        class SlowTransferWorker(TransferWorker):
            def run(self) -> None:
                time.sleep(0.4)
                super().run()

        monkeypatch.setattr(main_window_module, "TransferWorker", SlowTransferWorker)

        window = MainWindow(settings=default_settings)
        try:
            with patch.object(window, "_run_transfer_preview"):
                window._source_zone.load_from_path(sample_image_path)
                window._reference_zone.load_from_path(sample_image_path)

            with patch.object(window._preview, "set_image") as set_image:
                window._run_transfer_preview()
                first = window._transfer_worker
                window._run_transfer_preview()  # supersedes the running one
                gc.collect()

                assert first is not None and first.isRunning()
                assert window._stale_transfer_workers == [first]

                deadline = time.monotonic() + 15
                while time.monotonic() < deadline and (
                    window._stale_transfer_workers
                    or (window._transfer_worker and window._transfer_worker.isRunning())
                    or not set_image.called
                ):
                    qapp.processEvents()
                    time.sleep(0.01)

                assert window._stale_transfer_workers == []
                set_image.assert_called_once()  # the stale result was ignored
        finally:
            window.stop_workers()



# ── AdjustmentsPanel tests ───────────────────────────────────────────────────


class TestAdjustmentsPanel:
    def test_defaults_are_passthrough(self, qapp: QApplication) -> None:
        from stealzalut.gui.widgets.adjustments_panel import AdjustmentsPanel

        assert AdjustmentsPanel().params().is_passthrough

    def test_set_params_roundtrip_and_reset(self, qapp: QApplication) -> None:
        from stealzalut.core.grading import GradingParams
        from stealzalut.gui.widgets.adjustments_panel import AdjustmentsPanel

        panel = AdjustmentsPanel()
        wanted = GradingParams(
            luminance_strength=0.4,
            chroma_strength=0.9,
            preserve_neutrals=0.5,
            skin_protection=0.75,
            mix=0.8,
            smoothing=1.25,
        )
        panel.set_params(wanted)
        assert panel.params() == wanted
        panel.reset()
        assert panel.params().is_passthrough

    def test_change_is_debounced(self, qapp: QApplication) -> None:
        import time

        from stealzalut.gui.widgets.adjustments_panel import DEBOUNCE_MS, AdjustmentsPanel

        panel = AdjustmentsPanel()
        emitted: list[Any] = []
        panel.params_changed.connect(emitted.append)
        slider = panel._sliders["mix"]
        for value in (90, 80, 70):
            slider.setValue(value)
        assert emitted == []  # nothing before the debounce delay

        deadline = time.monotonic() + DEBOUNCE_MS / 1000 + 2
        while not emitted and time.monotonic() < deadline:
            qapp.processEvents()
            time.sleep(0.01)
        assert len(emitted) == 1
        assert emitted[0].mix == pytest.approx(0.7)


class TestModelCache:
    def test_second_preview_reuses_fitted_model(
        self,
        qapp: QApplication,
        default_settings: Settings,
        sample_image_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Moving a slider must not re-read the images (no refit)."""
        import time
        from unittest.mock import patch

        from stealzalut.core.color_transfer import ColorTransferEngine

        window = MainWindow(settings=default_settings)
        try:
            with patch.object(window, "_run_transfer_preview"):
                window._source_zone.load_from_path(sample_image_path)
                window._reference_zone.load_from_path(sample_image_path)

            calls: list[str] = []
            original_fit = ColorTransferEngine.fit

            def counting_fit(self: Any, *args: Any, **kwargs: Any) -> Any:
                calls.append("fit")
                return original_fit(self, *args, **kwargs)

            monkeypatch.setattr(ColorTransferEngine, "fit", counting_fit)

            def run_and_wait() -> None:
                window._run_transfer_preview()
                worker = window._transfer_worker
                deadline = time.monotonic() + 15
                while worker is not None and time.monotonic() < deadline and (
                    worker.isRunning() or not worker.isFinished()
                ):
                    qapp.processEvents()
                    time.sleep(0.01)
                for _ in range(20):
                    qapp.processEvents()

            run_and_wait()
            assert calls == ["fit"]
            assert window._model_cache is not None

            window._adjustments._sliders["mix"].setValue(50)
            run_and_wait()
            assert calls == ["fit"]  # reused, not refitted

            window._model_cache = None  # e.g. a new image was dropped
            run_and_wait()
            assert calls == ["fit", "fit"]
        finally:
            window.stop_workers()
