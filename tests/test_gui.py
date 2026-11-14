"""Tests for stealzalut.gui — ImageDropZone, PreviewPanel, and MainWindow."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, cast
from unittest.mock import patch

import cv2
import numpy as np
import pytest
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import QApplication, QLabel

from stealzalut.config import Settings
from stealzalut.core.look import LookTransferModel
from stealzalut.core.recipe import parse_recipe
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

        mock_rgb = np.full((70, 90, 3), 9000, dtype=np.uint16)  # 16-bit linear
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

        mock_rgb = np.full((70, 90, 3), 9000, dtype=np.uint16)  # 16-bit linear
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

        # Algorithm combo box populated with all 11 methods and default selected
        assert window._algo_combo.count() == 11
        algos = [window._algo_combo.itemData(i) for i in range(window._algo_combo.count())]
        assert "oklab" in algos
        assert "zones" in algos
        assert "ambiance" in algos
        assert "mkl" in algos
        assert "reinhard" in algos
        assert "hm" in algos
        assert window._algo_combo.currentData() == "look"

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
                # A fast global method: the cache is what is tested here
                window._algo_combo.setCurrentIndex(window._algo_combo.findData("mkl"))
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


class TestSeparationSlider:
    """The separation slider of the look method."""

    def test_separation_slider_only_visible_for_look(
        self, qapp: QApplication, default_settings: Settings
    ) -> None:
        window = MainWindow(settings=default_settings)
        try:
            window._algo_combo.setCurrentIndex(window._algo_combo.findData("mkl"))
            assert window._separation_slider.isHidden()
            window._algo_combo.setCurrentIndex(window._algo_combo.findData("look"))
            assert not window._separation_slider.isHidden()
            assert not window._separation_label.isHidden()
            window._algo_combo.setCurrentIndex(window._algo_combo.findData("oklab"))
            assert window._separation_slider.isHidden()
        finally:
            window.stop_workers()

    def test_separation_slider_changes_the_model(
        self, qapp: QApplication, default_settings: Settings
    ) -> None:
        from stealzalut.core.look import LookTransferModel, fit_look

        rng = np.random.default_rng(0)
        src = rng.integers(0, 255, (48, 48, 3), dtype=np.uint8)
        ref = rng.integers(0, 255, (48, 48, 3), dtype=np.uint8)
        window = MainWindow(settings=default_settings)
        try:
            window._algo_combo.setCurrentIndex(window._algo_combo.findData("look"))
            window._separation_slider.setValue(50)
            assert window._separation_value.text() == "0.50"
            model = window._effective_model(fit_look(src, ref))
            assert isinstance(model, LookTransferModel)
            assert model.separation == 0.5
        finally:
            window.stop_workers()

    def test_separation_is_undoable(
        self, qapp: QApplication, default_settings: Settings
    ) -> None:
        window = MainWindow(settings=default_settings)
        try:
            window._algo_combo.setCurrentIndex(window._algo_combo.findData("look"))
            window._separation_slider.setValue(130)
            assert window._history.current.separation == 1.3
            window.undo()
            assert window._history.current.separation == 1.0
            assert window._separation_slider.value() == 100
        finally:
            window.stop_workers()

    def test_locality_slider_only_visible_for_look(
        self, qapp: QApplication, default_settings: Settings
    ) -> None:
        window = MainWindow(settings=default_settings)
        try:
            window._algo_combo.setCurrentIndex(window._algo_combo.findData("mkl"))
            assert window._locality_slider.isHidden() and window._locality_label.isHidden()
            window._algo_combo.setCurrentIndex(window._algo_combo.findData("look"))
            assert not window._locality_slider.isHidden()
            assert window._locality_slider.value() == 0  # off unless asked for
            assert window._locality_value.text() == "0 %"
        finally:
            window.stop_workers()

    def test_locality_slider_changes_the_model(
        self, qapp: QApplication, default_settings: Settings
    ) -> None:
        from stealzalut.core.look import REACH_RANGE, LookTransferModel, fit_look

        rng = np.random.default_rng(0)
        src = rng.integers(0, 255, (48, 48, 3), dtype=np.uint8)
        ref = rng.integers(0, 255, (48, 48, 3), dtype=np.uint8)
        window = MainWindow(settings=default_settings)
        try:
            window._algo_combo.setCurrentIndex(window._algo_combo.findData("look"))
            fitted = fit_look(src, ref)
            plain = window._effective_model(fitted)
            assert isinstance(plain, LookTransferModel) and plain.reach is None
            window._locality_slider.setValue(100)
            assert window._locality_value.text() == "100 %"
            model = window._effective_model(fitted)
            assert isinstance(model, LookTransferModel)
            assert model.reach == pytest.approx(REACH_RANGE[1])
        finally:
            window.stop_workers()

    def test_locality_is_undoable(self, qapp: QApplication, default_settings: Settings) -> None:
        window = MainWindow(settings=default_settings)
        try:
            window._algo_combo.setCurrentIndex(window._algo_combo.findData("look"))
            window._locality_slider.setValue(40)
            assert window._history.current.locality == 0.4
            window.undo()
            assert window._history.current.locality == 0.0
            assert window._locality_slider.value() == 0
            window.redo()
            assert window._locality_slider.value() == 40
        finally:
            window.stop_workers()


def _write_rgb(path: Path, rgb: np.ndarray) -> Path:
    cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    return path


def _look_window(qapp: QApplication, settings: Settings, source: Path, reference: Path) -> Any:
    """A window in the look method, once the model of the pair is fitted."""
    window = MainWindow(settings=settings)
    window._algo_combo.setCurrentIndex(window._algo_combo.findData("look"))
    window._source_zone.load_from_path(source)
    window._reference_zone.load_from_path(reference)
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline and not (
        window._look_diagnostics is not None
        and not (window._transfer_worker and window._transfer_worker.isRunning())
    ):
        qapp.processEvents()
        time.sleep(0.01)
    return window


class TestLookBanner:
    """The look method tells its limits instead of hiding them."""

    @staticmethod
    def _grey(tmp_path: Path) -> Path:
        rng = np.random.default_rng(0)
        img = np.full((96, 128, 3), 110.0)
        img[:, 64:] = 50.0
        noisy = np.clip(img + rng.normal(0, 2, img.shape), 0, 255).astype(np.uint8)
        return _write_rgb(tmp_path / "grey.png", noisy)

    @staticmethod
    def _coloured(tmp_path: Path, name: str, tint: tuple[int, int, int]) -> Path:
        rng = np.random.default_rng(1)
        img = np.zeros((96, 128, 3))
        img[:48] = tint
        img[48:] = (200, 60, 40)
        img[40:56, 40:90] = (40, 90, 220)
        return _write_rgb(
            tmp_path / name, np.clip(img + rng.normal(0, 3, img.shape), 0, 255).astype(np.uint8)
        )

    def test_look_banner_shows_low_colour(
        self, qapp: QApplication, default_settings: Settings, tmp_path: Path
    ) -> None:
        reference = self._coloured(tmp_path, "r.png", (30, 120, 60))
        window = _look_window(qapp, default_settings, self._grey(tmp_path), reference)
        try:
            assert not window._mode_hint.isHidden()
            assert window._mode_hint_label.text() == (
                "La photo a peu de couleurs : la séparation aura peu d'effet."
            )
            assert window._mode_hint_btn.isHidden()
            window._dismiss_mode_hint()
            assert window._mode_hint.isHidden()
        finally:
            window.stop_workers()

    def test_look_banner_is_quiet_on_a_coloured_source(
        self, qapp: QApplication, default_settings: Settings, tmp_path: Path
    ) -> None:
        window = _look_window(
            qapp,
            default_settings,
            self._coloured(tmp_path, "s.png", (60, 60, 160)),
            self._coloured(tmp_path, "r.png", (30, 120, 60)),
        )
        try:
            assert window._look_diagnostics is not None
            assert not window._look_diagnostics.low_colour
            assert window._mode_hint.isHidden()
        finally:
            window.stop_workers()

    def test_leaving_look_restores_the_ambiance_button(
        self, qapp: QApplication, default_settings: Settings, tmp_path: Path
    ) -> None:
        reference = self._coloured(tmp_path, "r.png", (30, 120, 60))
        window = _look_window(qapp, default_settings, self._grey(tmp_path), reference)
        try:
            with patch.object(window, "_run_transfer_preview"):
                window._algo_combo.setCurrentIndex(window._algo_combo.findData("oklab"))
            assert window._look_diagnostics is None
            assert window._mode_hint.isHidden()
            assert not window._mode_hint_btn.isHidden()
        finally:
            window.stop_workers()


class TestLookFocus:
    """Choosing which zones of the reference the look method reads."""

    def _window(self, qapp: QApplication, settings: Settings, tmp_path: Path) -> Any:
        banner = TestLookBanner
        return _look_window(
            qapp,
            settings,
            banner._coloured(tmp_path, "s.png", (60, 60, 160)),
            banner._coloured(tmp_path, "r.png", (30, 120, 60)),
        )

    def test_focus_button_only_visible_for_look(
        self, qapp: QApplication, default_settings: Settings
    ) -> None:
        window = MainWindow(settings=default_settings)
        try:
            window._algo_combo.setCurrentIndex(window._algo_combo.findData("mkl"))
            assert window._focus_btn.isHidden()
            window._algo_combo.setCurrentIndex(window._algo_combo.findData("look"))
            assert not window._focus_btn.isHidden()
            assert window._focus_btn.text() == "Référence ▾"
        finally:
            window.stop_workers()

    def test_ignoring_a_zone_refits_with_weights(
        self, qapp: QApplication, default_settings: Settings, tmp_path: Path
    ) -> None:
        window = self._window(qapp, default_settings, tmp_path)
        try:
            assert window._reference_weights() is None
            key_before = window._method_key()
            palette = window._reference_zones()
            zone_id = palette.zones[0].id
            with patch.object(window, "_run_transfer_preview"):
                window._set_focus(zone_id, 0.0)
            assert window._history.current.focus == ((zone_id, 0.0),)
            assert window._focus_btn.text() == "Référence (1) ▾"
            assert window._method_key() != key_before  # the cached model is not reused
            weights = window._reference_weights()
            assert weights.shape == (palette.labels.size,)
            assert np.array_equal(weights == 0.0, palette.labels.ravel() == zone_id)

            window.undo()
            assert window._history.current.focus == ()
            assert window._focus_btn.text() == "Référence ▾"
            assert window._method_key() == key_before
        finally:
            window.stop_workers()

    def test_normal_weight_removes_the_zone_from_the_focus(
        self, qapp: QApplication, default_settings: Settings, tmp_path: Path
    ) -> None:
        window = self._window(qapp, default_settings, tmp_path)
        try:
            with patch.object(window, "_run_transfer_preview"):
                window._set_focus(1, 2.0)
                window._set_focus(2, 0.0)
                assert window._history.current.focus == ((1, 2.0), (2, 0.0))
                window._set_focus(1, 1.0)
                assert window._history.current.focus == ((2, 0.0),)
                window._clear_focus()
            assert window._history.current.focus == ()
        finally:
            window.stop_workers()

    def test_focus_menu_lists_the_reference_zones(
        self, qapp: QApplication, default_settings: Settings, tmp_path: Path
    ) -> None:
        window = self._window(qapp, default_settings, tmp_path)
        try:
            with patch.object(window, "_run_transfer_preview"):
                window._set_focus(1, 0.0)
            window._open_focus_menu()
            window._focus_menu.hide()
            titles = [a.text() for a in window._focus_menu.actions() if a.menu() is not None]
            assert len(titles) == len(window._reference_zones().zones)
            assert titles[0].startswith("R1 — ") and titles[0].endswith(" · ignorée")
            assert window._focus_menu.actions()[-1].text() == "Tout remettre au poids normal"
        finally:
            window.stop_workers()

    def test_a_new_reference_drops_the_focus(
        self, qapp: QApplication, default_settings: Settings, tmp_path: Path
    ) -> None:
        window = self._window(qapp, default_settings, tmp_path)
        try:
            with patch.object(window, "_run_transfer_preview"):
                window._set_focus(1, 0.0)
                other = TestLookBanner._coloured(tmp_path, "r2.png", (120, 30, 120))
                window._reference_zone.load_from_path(other)
            assert window._history.current.focus == ()
            assert window._reference_palette is None
            assert window._focus_btn.text() == "Référence ▾"
        finally:
            window.stop_workers()


def _look_recipe(**look: Any) -> dict[str, Any]:
    return {"format": "stealzalut.recipe", "version": 2, "mode": "look", "look": look}


class TestWindowLookRecipe:
    """Recipes of the look method: import, export, switching method."""

    def _window(self, qapp: QApplication, settings: Settings, tmp_path: Path) -> Any:
        return _look_window(
            qapp,
            settings,
            TestLookBanner._coloured(tmp_path, "s.png", (60, 60, 160)),
            TestLookBanner._coloured(tmp_path, "r.png", (30, 120, 60)),
        )

    def test_recipe_button_only_visible_for_look_recipe(
        self, qapp: QApplication, default_settings: Settings, tmp_path: Path
    ) -> None:
        window = self._window(qapp, default_settings, tmp_path)
        try:
            assert not window._look_recipe_btn.isHidden()
            with patch.object(window, "_run_transfer_preview"):
                window._algo_combo.setCurrentIndex(window._algo_combo.findData("oklab"))
            assert window._look_recipe_btn.isHidden()
        finally:
            window.stop_workers()

    def test_apply_look_recipe(
        self, qapp: QApplication, default_settings: Settings, tmp_path: Path
    ) -> None:
        window = self._window(qapp, default_settings, tmp_path)
        try:
            data = _look_recipe(separation=1.3, protect_vivid=0.4, focus={"R1": 0})
            data["global"] = {"mix": 0.8}
            data["finishing"] = {"vibrance": 0.3}
            data["name"] = "Essai"
            with patch.object(window, "_run_transfer_preview"):
                assert window.apply_recipe(parse_recipe(data))
            state = window._history.current
            assert state.separation == 1.3 and state.protect_vivid == 0.4
            assert state.focus == ((1, 0.0),) and state.params.mix == 0.8
            assert state.finishing.vibrance == 0.3
            assert window._separation_slider.value() == 130
            assert window._focus_btn.text() == "Référence (1) ▾"
            assert "Recette « Essai » appliquée en mode Rendu." in window._status_note
            model = window._effective_model(window._model_cache[3])
            assert isinstance(model, LookTransferModel)
            assert model.separation == 1.3 and model.protect_vivid == 0.4

            window.undo()
            assert window._history.current.separation == 1.0
            assert window._history.current.focus == ()
        finally:
            window.stop_workers()

    def test_look_recipe_switches_to_the_look_method(
        self, qapp: QApplication, default_settings: Settings, tmp_path: Path
    ) -> None:
        window = self._window(qapp, default_settings, tmp_path)
        try:
            with patch.object(window, "_run_transfer_preview"):
                window._algo_combo.setCurrentIndex(window._algo_combo.findData("oklab"))
                assert window.apply_recipe(parse_recipe(_look_recipe(separation=0.4)))
            assert window._current_method == "look"
            assert window._history.current.separation == 0.4
        finally:
            window.stop_workers()

    def test_focus_of_another_pair_is_not_applied(
        self, qapp: QApplication, default_settings: Settings, tmp_path: Path
    ) -> None:
        window = self._window(qapp, default_settings, tmp_path)
        try:
            data = _look_recipe(separation=0.7, focus={"R1": 0})
            data["analysis_id"] = "00000000"
            with patch.object(window, "_run_transfer_preview"):
                assert window.apply_recipe(parse_recipe(data))
            assert window._history.current.separation == 0.7
            assert window._history.current.focus == ()
            assert "Poids des zones de référence non repris" in window._status_note
        finally:
            window.stop_workers()

    def test_unknown_focus_zone_is_refused(
        self, qapp: QApplication, default_settings: Settings, tmp_path: Path
    ) -> None:
        window = self._window(qapp, default_settings, tmp_path)
        try:
            recipe = parse_recipe(_look_recipe(separation=0.7, focus={"R99": 0}))
            with patch("stealzalut.gui.main_window.QMessageBox.critical") as critical:
                assert not window.apply_recipe(recipe)
            assert "R99" in critical.call_args.args[2]
            assert window._history.current.separation == 1.0
        finally:
            window.stop_workers()

    def test_zones_recipe_in_look_mode_is_explained(
        self, qapp: QApplication, default_settings: Settings, tmp_path: Path
    ) -> None:
        window = self._window(qapp, default_settings, tmp_path)
        try:
            recipe = parse_recipe({"format": "stealzalut.recipe", "version": 2, "mode": "zones"})
            with patch("stealzalut.gui.main_window.QMessageBox.information") as information:
                assert not window.apply_recipe(recipe)
            assert "Zones" in information.call_args.args[2]
        finally:
            window.stop_workers()

    def test_export_then_import_gives_the_same_state(
        self, qapp: QApplication, default_settings: Settings, tmp_path: Path
    ) -> None:
        window = self._window(qapp, default_settings, tmp_path)
        try:
            target = tmp_path / "look.szl"
            with patch.object(window, "_run_transfer_preview"):
                window._separation_slider.setValue(60)
                window._set_focus(2, 2.0)
                saved = window._history.current
                with patch(
                    "stealzalut.gui.main_window.QFileDialog.getSaveFileName",
                    return_value=(str(target), ""),
                ):
                    window._on_export_recipe()
                written = json.loads(target.read_text(encoding="utf-8"))
                assert written["mode"] == "look"
                assert written["look"] == {"separation": 0.6, "focus": {"R2": 2.0}}
                assert written["analysis_id"] == window._look_analysis_id()

                window._separation_slider.setValue(100)
                window._clear_focus()
                assert window.apply_recipe(parse_recipe(target.read_text(encoding="utf-8")))
            assert window._history.current == saved
        finally:
            window.stop_workers()


class TestWindowLookPack:
    """« Préparer pour mon IA » in the look method: the pack shows the render."""

    def test_prepare_look_pack(
        self,
        qapp: QApplication,
        default_settings: Settings,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from PySide6.QtWidgets import QFileDialog, QMessageBox

        window = _look_window(
            qapp,
            default_settings,
            TestLookBanner._coloured(tmp_path, "src.png", (60, 60, 160)),
            TestLookBanner._coloured(tmp_path, "ref.png", (30, 120, 60)),
        )
        try:
            deadline = time.monotonic() + 15
            while time.monotonic() < deadline and window._preview.image_rgb is None:
                qapp.processEvents()
                time.sleep(0.01)
            out = tmp_path / "packs"
            out.mkdir()
            monkeypatch.setattr(QFileDialog, "getExistingDirectory", lambda *a, **k: str(out))
            monkeypatch.setattr(QMessageBox, "exec", lambda self: 0)
            with patch.object(window, "_run_transfer_preview"):
                window._separation_slider.setValue(120)
            window._on_prepare_look_pack()

            folder = out / "src_x_ref_stealzalut_look_pack"
            assert (folder / "3_current_render.jpg").exists()
            prompt = (folder / "PROMPT.md").read_text(encoding="utf-8")
            assert QGuiApplication.clipboard().text() == prompt
            assert "actuellement 1.2" in prompt
            assert window._look_analysis_id() in prompt

            # The model answer of the prompt imports as it is
            with patch.object(window, "_run_transfer_preview"):
                assert window.apply_recipe(parse_recipe(prompt))
            assert window._history.current.protect_vivid == 0.5
        finally:
            window.stop_workers()

    def test_look_pack_needs_a_render(
        self, qapp: QApplication, default_settings: Settings
    ) -> None:
        window = MainWindow(settings=default_settings)
        try:
            window._algo_combo.setCurrentIndex(window._algo_combo.findData("look"))
            with patch("stealzalut.gui.main_window.QMessageBox.information") as information:
                window._on_prepare_look_pack()
            assert information.call_args.args[1] == "Rendu pas encore prêt"
        finally:
            window.stop_workers()


class TestSourceFidelity:
    """What the preview shows is what can be exported, and RAW sources are flagged."""

    def test_raw_source_is_flagged(
        self,
        qapp: QApplication,
        default_settings: Settings,
        sample_image_path: Path,
        tmp_path: Path,
    ) -> None:
        """A RAW source says how it was rendered, and how far that can be trusted."""
        from unittest.mock import MagicMock

        from stealzalut.gui.main_window import _RAW_DEVELOPED_HINT, _RAW_PREVIEW_HINT

        window = MainWindow(settings=default_settings)
        label = window._source_hint.findChild(QLabel, "modeHintLabel")
        try:
            with patch.object(window, "_run_transfer_preview"):
                window._source_zone.load_from_path(sample_image_path)
            assert window._source_hint.isHidden()  # a PNG is what it looks like

            # A RAW file read through the camera's JPEG: the camera's look, not the host's
            window._source_zone._image_path = Path("IMG_0001.DNG")
            window._update_source_hint()
            assert not window._source_hint.isHidden()
            assert label.text() == _RAW_PREVIEW_HINT
            assert "JPEG intégré par l'appareil" in label.text()

            # A RAW file developed here: neutral, but not the host's exact rendering
            raw_path = tmp_path / "photo.dng"
            raw_path.write_bytes(b"mock raw bytes")
            mock_raw = MagicMock()
            mock_raw.postprocess.return_value = np.full((70, 90, 3), 9000, dtype=np.uint16)
            mock_raw.__enter__.return_value = mock_raw
            mock_raw.__exit__.return_value = None
            with (
                patch("rawpy.imread", return_value=mock_raw),
                patch.object(window, "_run_transfer_preview"),
            ):
                assert window._source_zone.load_from_path(raw_path)
            assert not window._source_hint.isHidden()
            assert label.text() == _RAW_DEVELOPED_HINT
            assert "développé ici sur une base neutre" in label.text()
            assert "l'écart n'a pas été mesuré" in label.text()
            assert window._source_zone.image_rgb.native_space.key == "display-p3"

            window._source_zone.clear_image()
            assert window._source_hint.isHidden()
        finally:
            window.stop_workers()

    def test_export_image_writes_the_preview_at_full_size(
        self,
        qapp: QApplication,
        default_settings: Settings,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from PySide6.QtWidgets import QFileDialog

        from stealzalut.core.grading import build_grid
        from stealzalut.core.lut_apply import apply_lut

        window = _look_window(
            qapp,
            default_settings,
            TestLookBanner._coloured(tmp_path, "s.png", (60, 60, 160)),
            TestLookBanner._coloured(tmp_path, "r.png", (30, 120, 60)),
        )
        try:
            assert window._export_image_btn.isEnabled()
            target = tmp_path / "graded.png"
            monkeypatch.setattr(
                QFileDialog, "getSaveFileName", lambda *a, **k: (str(target), "PNG (*.png)")
            )
            with patch.object(window, "_run_transfer_preview"):
                window._separation_slider.setValue(130)
            window._on_export_image_clicked()

            written = cv2.cvtColor(cv2.imread(str(target)), cv2.COLOR_BGR2RGB)
            source = window._source_zone.image_rgb
            model = window._effective_model(window._model_cache[3])
            assert model.separation == 1.3
            grid = build_grid(model, 33, window._adjustments.params()).grid
            assert written.shape == source.shape
            assert np.array_equal(written, apply_lut(source, grid, 33))
        finally:
            window.stop_workers()

    def test_export_image_adds_a_missing_extension(
        self,
        qapp: QApplication,
        default_settings: Settings,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from PySide6.QtWidgets import QFileDialog

        window = _look_window(
            qapp,
            default_settings,
            TestLookBanner._coloured(tmp_path, "s.png", (60, 60, 160)),
            TestLookBanner._coloured(tmp_path, "r.png", (30, 120, 60)),
        )
        try:
            monkeypatch.setattr(
                QFileDialog,
                "getSaveFileName",
                lambda *a, **k: (str(tmp_path / "graded"), "TIFF (*.tif)"),
            )
            window._on_export_image_clicked()
            assert (tmp_path / "graded.tif").exists()
        finally:
            window.stop_workers()

    def test_export_image_needs_a_render(
        self, qapp: QApplication, default_settings: Settings, sample_image_path: Path
    ) -> None:
        window = MainWindow(settings=default_settings)
        try:
            assert not window._export_image_btn.isEnabled()
            with patch.object(window, "_run_transfer_preview"):
                window._source_zone.load_from_path(sample_image_path)
                window._reference_zone.load_from_path(sample_image_path)
            with patch("stealzalut.gui.main_window.QMessageBox.information") as information:
                window._on_export_image_clicked()
            assert information.call_args.args[1] == "Rendu pas encore prêt"
        finally:
            window.stop_workers()


class TestCubeSpace:
    """The colour space the exported .cube is made for."""

    def test_space_menu_defaults_to_srgb(
        self, qapp: QApplication, default_settings: Settings
    ) -> None:
        from stealzalut.core.rgb_spaces import CUBE_SPACES, SRGB

        window = MainWindow(settings=default_settings)
        try:
            combo = window._space_combo
            assert [combo.itemData(i) for i in range(combo.count())] == [
                space.key for space in CUBE_SPACES
            ]
            assert window._cube_space() is SRGB
            assert window._cube_suffix() == ""
            combo.setCurrentIndex(combo.findData("display-p3"))
            assert window._cube_space().label == "Display P3"
            assert window._cube_suffix() == "_display-p3"
        finally:
            window.stop_workers()

    def test_export_uses_the_chosen_space(
        self,
        qapp: QApplication,
        default_settings: Settings,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from PySide6.QtWidgets import QMessageBox

        from stealzalut.core.grading import build_grid
        from stealzalut.core.rgb_spaces import DISPLAY_P3

        window = _look_window(
            qapp,
            default_settings,
            TestLookBanner._coloured(tmp_path, "s.png", (60, 60, 160)),
            TestLookBanner._coloured(tmp_path, "r.png", (30, 120, 60)),
        )
        try:
            shown: list[str] = []
            monkeypatch.setattr(
                QMessageBox, "information", lambda parent, title, text: shown.append(text)
            )
            window._space_combo.setCurrentIndex(window._space_combo.findData("display-p3"))
            target = tmp_path / "p3.cube"
            window._run_export(target)
            deadline = time.monotonic() + 20
            while time.monotonic() < deadline and not shown:
                qapp.processEvents()
                time.sleep(0.01)

            assert "Display P3" in shown[0] and "plus large que sRGB" in shown[0]
            rows = [
                [float(v) for v in line.split()]
                for line in target.read_text().splitlines()
                if line and (line[0].isdigit() or line[0] == "-")
            ]
            model = window._effective_model(window._model_cache[3])
            expected = build_grid(model, 33, window._adjustments.params(), None, DISPLAY_P3).grid
            assert np.allclose(np.array(rows), expected, atol=1e-6)
            srgb_grid = build_grid(model, 33, window._adjustments.params()).grid
            assert not np.allclose(np.array(rows), srgb_grid, atol=1e-3)  # really another grid
        finally:
            window.stop_workers()
