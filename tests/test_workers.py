"""Tests for stealzalut.workers — QThread workers for transfer & export.

These tests require a ``QApplication`` instance (provided by the
``qapp`` fixture) because ``QThread`` needs an active event loop.

Signals emitted from worker threads are queued connections — they
require the main-thread event loop to be pumping.  We use
``QEventLoop`` to block until the worker's ``finished`` signal fires,
which ensures all queued signals are processed.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
from PySide6.QtCore import QEventLoop, QTimer
from PySide6.QtWidgets import QApplication

from stealzalut.workers.export_worker import ExportWorker
from stealzalut.workers.transfer_worker import TransferWorker

# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture(scope="session")
def qapp() -> QApplication:
    """Create a single QApplication for the entire test session."""
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    return cast("QApplication", app)


def _make_synthetic_image(
    mean: tuple[float, float, float],
    std: float,
    h: int = 32,
    w: int = 32,
    seed: int = 42,
) -> np.ndarray:
    """Create a small synthetic RGB uint8 image."""
    rng = np.random.default_rng(seed)
    img = rng.normal(loc=mean, scale=std, size=(h, w, 3))
    return np.clip(img, 0, 255).astype(np.uint8)


@pytest.fixture()
def source_image() -> np.ndarray:
    return _make_synthetic_image(mean=(180, 100, 80), std=30, seed=1)


@pytest.fixture()
def reference_image() -> np.ndarray:
    return _make_synthetic_image(mean=(80, 120, 200), std=25, seed=2)


# ── Helpers ──────────────────────────────────────────────────────────────────


def _run_worker_with_event_loop(
    worker: Any,
    timeout_ms: int = 30000,
) -> None:
    """Start a worker and pump the event loop until it finishes.

    Uses ``QEventLoop`` so that queued signal-slot connections from the
    worker thread are properly delivered to the main thread.
    """
    loop = QEventLoop()

    # When the worker thread finishes, quit the event loop
    worker.finished.connect(loop.quit)

    # Safety timeout to avoid hanging indefinitely
    timer = QTimer()
    timer.setSingleShot(True)
    timer.timeout.connect(loop.quit)
    timer.start(timeout_ms)

    worker.start()
    loop.exec()

    if timer.isActive():
        timer.stop()
    else:
        worker.terminate()
        worker.wait(1000)
        pytest.fail(f"Worker {worker.__class__.__name__} timed out after {timeout_ms}ms")


# ── TransferWorker tests ─────────────────────────────────────────────────────


class TestTransferWorker:
    """Tests for TransferWorker."""

    def test_produces_result(
        self,
        qapp: QApplication,
        source_image: np.ndarray,
        reference_image: np.ndarray,
    ) -> None:
        """The worker must emit result_ready with a valid image."""
        results: list[Any] = []
        errors: list[str] = []

        worker = TransferWorker(source_image, reference_image)
        worker.result_ready.connect(results.append)
        worker.error.connect(errors.append)

        _run_worker_with_event_loop(worker)

        assert len(errors) == 0, f"Worker emitted error: {errors}"
        assert len(results) == 1
        result = results[0]
        assert isinstance(result, np.ndarray)
        assert result.shape == source_image.shape
        assert result.dtype == np.uint8

    def test_emits_progress(
        self,
        qapp: QApplication,
        source_image: np.ndarray,
        reference_image: np.ndarray,
    ) -> None:
        """The worker must emit progress signals reaching 100."""
        progress_values: list[int] = []

        worker = TransferWorker(source_image, reference_image)
        worker.progress.connect(progress_values.append)

        _run_worker_with_event_loop(worker)

        assert len(progress_values) > 0
        assert progress_values[-1] == 100

    def test_result_differs_from_source(
        self,
        qapp: QApplication,
        source_image: np.ndarray,
        reference_image: np.ndarray,
    ) -> None:
        """The transferred image should differ from the source."""
        results: list[Any] = []

        worker = TransferWorker(source_image, reference_image)
        worker.result_ready.connect(results.append)

        _run_worker_with_event_loop(worker)

        assert len(results) == 1
        assert not np.array_equal(results[0], source_image)


# ── ExportWorker tests ───────────────────────────────────────────────────────


class TestExportWorker:
    """Tests for ExportWorker."""

    def test_creates_cube_file(
        self,
        qapp: QApplication,
        source_image: np.ndarray,
        reference_image: np.ndarray,
        tmp_path: Path,
    ) -> None:
        """The worker must write a valid .cube file."""
        output = tmp_path / "test_export.cube"
        results: list[str] = []
        errors: list[str] = []

        worker = ExportWorker(
            source_rgb=source_image,
            reference_rgb=reference_image,
            lut_size=17,
            output_path=output,
        )
        worker.result_ready.connect(results.append)
        worker.error.connect(errors.append)

        _run_worker_with_event_loop(worker)

        assert len(errors) == 0, f"Worker emitted error: {errors}"
        assert len(results) == 1

        result_path = Path(results[0])
        assert result_path.exists()
        assert result_path.suffix == ".cube"

        # Verify file content
        content = result_path.read_text(encoding="utf-8")
        assert "LUT_3D_SIZE 17" in content
        assert 'TITLE "StealZaLut"' in content

    def test_emits_progress_reaching_100(
        self,
        qapp: QApplication,
        source_image: np.ndarray,
        reference_image: np.ndarray,
        tmp_path: Path,
    ) -> None:
        """The worker must emit progress signals from 0 to 100."""
        progress_values: list[int] = []

        worker = ExportWorker(
            source_rgb=source_image,
            reference_rgb=reference_image,
            lut_size=17,
            output_path=tmp_path / "progress_test.cube",
        )
        worker.progress.connect(progress_values.append)

        _run_worker_with_event_loop(worker)

        assert len(progress_values) > 0
        assert progress_values[0] == 0
        assert progress_values[-1] == 100
        # Progress should be monotonically non-decreasing
        for i in range(1, len(progress_values)):
            assert progress_values[i] >= progress_values[i - 1]

    def test_cube_file_has_correct_line_count(
        self,
        qapp: QApplication,
        source_image: np.ndarray,
        reference_image: np.ndarray,
        tmp_path: Path,
    ) -> None:
        """The exported .cube file must have exactly size³ data lines."""
        size = 17
        output = tmp_path / "lines_test.cube"
        results: list[str] = []

        worker = ExportWorker(
            source_rgb=source_image,
            reference_rgb=reference_image,
            lut_size=size,
            output_path=output,
        )
        worker.result_ready.connect(results.append)

        _run_worker_with_event_loop(worker)

        assert len(results) == 1
        content = Path(results[0]).read_text(encoding="utf-8")
        lines = content.strip().split("\n")
        # Header: TITLE, LUT_3D_SIZE, DOMAIN_MIN, DOMAIN_MAX, blank line = 5 lines
        header_lines = 5
        data_lines = len(lines) - header_lines
        assert data_lines == size**3

    def test_custom_title(
        self,
        qapp: QApplication,
        source_image: np.ndarray,
        reference_image: np.ndarray,
        tmp_path: Path,
    ) -> None:
        """The worker must accept a custom LUT title."""
        results: list[str] = []

        worker = ExportWorker(
            source_rgb=source_image,
            reference_rgb=reference_image,
            lut_size=17,
            output_path=tmp_path / "titled.cube",
            title="MyCustomGrade",
        )
        worker.result_ready.connect(results.append)

        _run_worker_with_event_loop(worker)

        assert len(results) == 1
        content = Path(results[0]).read_text(encoding="utf-8")
        assert 'TITLE "MyCustomGrade"' in content
