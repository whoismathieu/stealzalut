"""Main application window.

Orchestrates the GUI layout and connects user interactions to the
background workers.  The window never performs heavy computation
directly — all matrix operations run in QThread workers.
"""

from __future__ import annotations

import logging
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSplitter,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

from stealzalut.config import Settings
from stealzalut.core.color_transfer import METHOD_LABELS
from stealzalut.core.lut_engine import VALID_LUT_SIZES
from stealzalut.gui.widgets.image_drop_zone import ImageDropZone
from stealzalut.gui.widgets.preview_panel import PreviewPanel
from stealzalut.workers.export_worker import ExportWorker
from stealzalut.workers.transfer_worker import TransferWorker

logger = logging.getLogger(__name__)


class MainWindow(QMainWindow):
    """StealZaLut main application window.

    Layout::

        ┌──────────────────────────────────────────────────┐
        │  Source Image  │  Reference Image  │   Preview   │
        │  (drop zone)   │  (drop zone)      │   (result)  │
        ├──────────────────────────────────────────────────┤
        │  LUT Size: [17 ▼ 33 | 64]  │  [Export .cube]    │
        ├──────────────────────────────────────────────────┤
        │  ████████████████░░░░░░░░  45%                   │
        └──────────────────────────────────────────────────┘
    """

    def __init__(self, settings: Settings, parent: Any = None) -> None:
        super().__init__(parent)
        self._settings = settings
        self._transfer_worker: TransferWorker | None = None
        self._export_worker: ExportWorker | None = None
        # Superseded preview workers still running; kept alive until they end.
        self._stale_transfer_workers: list[TransferWorker] = []

        self.setWindowTitle("StealZaLut — Colour LUT Generator")
        self.setMinimumSize(900, 550)
        self.resize(1100, 650)

        self._build_ui()
        self._connect_signals()
        self._update_export_button_state()

        logger.info("MainWindow initialized")

    # ── UI Construction ──────────────────────────────────────────────────

    def _build_ui(self) -> None:
        """Build the complete widget tree."""
        central = QWidget()
        self.setCentralWidget(central)
        root_layout = QVBoxLayout(central)
        root_layout.setSpacing(8)

        # ── Top: Image panels (source | reference | preview) ─────────
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Source drop zone
        source_group = QGroupBox("Source Image")
        source_layout = QVBoxLayout(source_group)
        self._source_zone = ImageDropZone(
            label_text="Drop SOURCE image\nor click to browse",
            max_preview=self._settings.preview_max_size,
        )
        source_layout.addWidget(self._source_zone)
        splitter.addWidget(source_group)

        # Reference drop zone
        ref_group = QGroupBox("Reference Image")
        ref_layout = QVBoxLayout(ref_group)
        self._reference_zone = ImageDropZone(
            label_text="Drop REFERENCE image\nor click to browse",
            max_preview=self._settings.preview_max_size,
        )
        ref_layout.addWidget(self._reference_zone)
        splitter.addWidget(ref_group)

        # Preview panel
        preview_group = QGroupBox("Transfer Preview")
        preview_layout = QVBoxLayout(preview_group)
        self._preview = PreviewPanel(title="")
        preview_layout.addWidget(self._preview)
        splitter.addWidget(preview_group)

        # Equal widths
        splitter.setSizes([350, 350, 350])
        root_layout.addWidget(splitter, stretch=1)

        # ── Middle: Controls (Algorithm + LUT size + Export button) ─
        controls_layout = QHBoxLayout()
        controls_layout.setSpacing(12)

        # Algorithm selector
        algo_label = QLabel("Algorithm:")
        algo_label.setObjectName("controlLabel")
        controls_layout.addWidget(algo_label)

        self._algo_combo = QComboBox()
        self._algo_combo.setObjectName("algoCombo")
        for method_key, method_name in METHOD_LABELS.items():
            self._algo_combo.addItem(method_name, userData=method_key)
        default_algo_idx = self._algo_combo.findData(self._settings.color_transfer_method)
        if default_algo_idx >= 0:
            self._algo_combo.setCurrentIndex(default_algo_idx)
        controls_layout.addWidget(self._algo_combo)

        # LUT size selector
        size_label = QLabel("LUT Size:")
        size_label.setObjectName("controlLabel")
        controls_layout.addWidget(size_label)

        self._size_combo = QComboBox()
        self._size_combo.setObjectName("lutSizeCombo")
        for size in sorted(VALID_LUT_SIZES):
            label = f"{size}  ({size**3:,} points)"
            self._size_combo.addItem(label, userData=size)
        # Set default from settings
        default_idx = self._size_combo.findData(self._settings.lut_default_size)
        if default_idx >= 0:
            self._size_combo.setCurrentIndex(default_idx)
        controls_layout.addWidget(self._size_combo)

        controls_layout.addStretch()

        # Clear button
        self._clear_btn = QPushButton("Clear All")
        self._clear_btn.setObjectName("clearButton")
        controls_layout.addWidget(self._clear_btn)

        # Export button (.cube)
        self._export_btn = QPushButton("Export .cube")
        self._export_btn.setObjectName("exportButton")
        self._export_btn.setEnabled(False)
        self._export_btn.setToolTip("Exporter le fichier 3D LUT (.cube)")
        controls_layout.addWidget(self._export_btn)

        root_layout.addLayout(controls_layout)

        # ── Bottom: Progress bar ─────────────────────────────────────
        self._progress_bar = QProgressBar()
        self._progress_bar.setRange(0, 100)
        self._progress_bar.setValue(0)
        self._progress_bar.setTextVisible(True)
        self._progress_bar.setObjectName("mainProgressBar")
        root_layout.addWidget(self._progress_bar)

        # Status bar
        self._status_bar = QStatusBar()
        self.setStatusBar(self._status_bar)
        self._status_bar.showMessage("Ready — drop a source and reference image to begin.")

    # ── Signal wiring ────────────────────────────────────────────────────

    def _connect_signals(self) -> None:
        """Wire up all signal-slot connections."""
        # Image loading → auto-preview
        self._source_zone.image_loaded.connect(self._on_image_changed)
        self._reference_zone.image_loaded.connect(self._on_image_changed)

        # Image cleared
        self._source_zone.image_cleared.connect(self._on_image_cleared)
        self._reference_zone.image_cleared.connect(self._on_image_cleared)

        # Buttons
        self._export_btn.clicked.connect(self._on_export_clicked)
        self._clear_btn.clicked.connect(self._on_clear_all)

        # Algorithm or LUT size changed → the previewed grid changes too
        self._algo_combo.currentIndexChanged.connect(self._on_algo_changed)
        self._size_combo.currentIndexChanged.connect(self._on_algo_changed)

    # ── Slots ────────────────────────────────────────────────────────────

    def _on_algo_changed(self, _index: int) -> None:
        """Called when the user changes the algorithm or the LUT size."""
        if self._source_zone.has_image and self._reference_zone.has_image:
            self._run_transfer_preview()

    def _on_image_changed(self, _image: Any) -> None:
        """Called when either source or reference image is loaded."""
        self._update_export_button_state()

        if self._source_zone.has_image and self._reference_zone.has_image:
            self._run_transfer_preview()

    def _on_image_cleared(self) -> None:
        """Called when an image is removed."""
        self._update_export_button_state()
        self._preview.clear()
        self._progress_bar.setValue(0)

    def _on_clear_all(self) -> None:
        """Clear both images and the preview."""
        self._source_zone.clear_image()
        self._reference_zone.clear_image()
        self._preview.clear()
        self._progress_bar.setValue(0)
        self._status_bar.showMessage("Cleared — ready for new images.")

    def _on_export_clicked(self) -> None:
        """Open a save dialog and start the .cube export worker."""
        if not self._source_zone.has_image or not self._reference_zone.has_image:
            return

        # Determine default filename
        default_name = "stealzalut_export.cube"
        if self._source_zone.image_path and self._reference_zone.image_path:
            src_stem = self._source_zone.image_path.stem
            ref_stem = self._reference_zone.image_path.stem
            default_name = f"{src_stem}_x_{ref_stem}.cube"

        default_dir = str(self._settings.lut_output_dir)
        default_path = str(Path(default_dir) / default_name)

        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export 3D LUT (.cube)",
            default_path,
            "Cube LUT (*.cube)",
        )
        if not path:
            return

        self._run_export(Path(path))

    # ── Transfer preview ─────────────────────────────────────────────────

    def _run_transfer_preview(self) -> None:
        """Launch the TransferWorker for a live preview."""
        source = self._source_zone.image_rgb
        reference = self._reference_zone.image_rgb
        if source is None or reference is None:
            return

        method: str = str(self._algo_combo.currentData())
        lut_size: int = self._size_combo.currentData()

        # A newer request supersedes any preview still being computed
        self._retire_transfer_worker()

        self._status_bar.showMessage(f"Computing colour transfer preview ({method})…")
        self._progress_bar.setValue(0)

        self._transfer_worker = TransferWorker(
            source, reference, method=method, lut_size=lut_size
        )
        self._transfer_worker.progress.connect(self._on_transfer_progress)
        self._transfer_worker.result_ready.connect(self._on_transfer_result)
        self._transfer_worker.error.connect(self._on_worker_error)
        self._transfer_worker.start()

    def _retire_transfer_worker(self) -> None:
        """Detach the current preview worker without blocking the GUI.

        Its signals are disconnected so a stale result can never overwrite
        a newer preview.  If it is still running, it stays referenced until
        it finishes: a ``QThread`` garbage-collected while running aborts
        the whole application ("QThread: Destroyed while thread is still
        running").
        """
        old = self._transfer_worker
        self._transfer_worker = None
        if old is None:
            return

        old.progress.disconnect(self._on_transfer_progress)
        old.result_ready.disconnect(self._on_transfer_result)
        old.error.disconnect(self._on_worker_error)

        if old.isRunning():
            self._stale_transfer_workers.append(old)
            old.finished.connect(partial(self._forget_stale_worker, old))

    def _forget_stale_worker(self, worker: TransferWorker) -> None:
        """Release a superseded preview worker once its thread has ended."""
        worker.wait()  # ``finished`` fires just before the thread fully exits
        if worker in self._stale_transfer_workers:
            self._stale_transfer_workers.remove(worker)

    def _on_transfer_progress(self, pct: int) -> None:
        self._progress_bar.setValue(pct)

    def _on_transfer_result(self, result: NDArray[np.uint8]) -> None:
        self._preview.set_image(result)
        self._progress_bar.setValue(100)
        self._status_bar.showMessage("Preview ready — click 'Export .cube' to save.")
        logger.info("Transfer preview displayed")

    # ── Export ────────────────────────────────────────────────────────────

    def _run_export(self, output_path: Path) -> None:
        """Launch the ExportWorker to generate the .cube file."""
        source = self._source_zone.image_rgb
        reference = self._reference_zone.image_rgb
        if source is None or reference is None:
            return

        lut_size: int = self._size_combo.currentData()
        method: str = str(self._algo_combo.currentData())

        # Disable export during processing
        self._export_btn.setEnabled(False)
        self._status_bar.showMessage(
            f"Exporting 3D LUT ({method}) to {output_path.name}…"
        )
        self._progress_bar.setValue(0)

        title = output_path.stem
        self._export_worker = ExportWorker(
            source_rgb=source,
            reference_rgb=reference,
            lut_size=lut_size,
            output_path=output_path,
            title=title,
            method=method,
        )
        self._export_worker.progress.connect(self._on_export_progress)
        self._export_worker.result_ready.connect(self._on_export_result)
        self._export_worker.error.connect(self._on_worker_error)
        self._export_worker.finished.connect(self._on_export_finished)
        self._export_worker.start()

    def _on_export_progress(self, pct: int) -> None:
        self._progress_bar.setValue(pct)

    def _on_export_result(self, path_str: str) -> None:
        self._progress_bar.setValue(100)
        path = Path(path_str)
        self._status_bar.showMessage(f"Exported: {path.name}")
        logger.info("Exported to %s", path)

        msg = (
            f"3D LUT saved to:\n{path}\n\n"
            f"You can import this .cube file in Lightroom, "
            f"DaVinci Resolve, Premiere Pro, Photoshop, or any compatible application."
        )

        QMessageBox.information(
            self,
            "Export Complete",
            msg,
        )

    def _on_export_finished(self) -> None:
        """Re-enable the export button when the worker finishes."""
        self._update_export_button_state()

    # ── Error handling ───────────────────────────────────────────────────

    def _on_worker_error(self, message: str) -> None:
        """Display an error message from a worker."""
        self._progress_bar.setValue(0)
        self._status_bar.showMessage(f"Error: {message}")
        logger.error("Worker error: %s", message)

        QMessageBox.critical(self, "Error", message)
        self._update_export_button_state()

    # ── Helpers ──────────────────────────────────────────────────────────

    def _update_export_button_state(self) -> None:
        """Enable the export button only when both images are loaded."""
        both_loaded = self._source_zone.has_image and self._reference_zone.has_image
        not_exporting = (
            self._export_worker is None or not self._export_worker.isRunning()
        )
        is_enabled = both_loaded and not_exporting
        self._export_btn.setEnabled(is_enabled)

    def stop_workers(self, timeout_ms: int = 3000) -> None:
        """Wait for any running background workers to finish."""
        workers = [self._transfer_worker, self._export_worker, *self._stale_transfer_workers]
        for worker in workers:
            if worker is not None and worker.isRunning():
                worker.wait(timeout_ms)

    def closeEvent(self, event: Any) -> None:
        """Ensure background threads are safely terminated before closing."""
        self.stop_workers()
        super().closeEvent(event)

