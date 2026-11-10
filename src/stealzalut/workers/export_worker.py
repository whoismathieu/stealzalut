"""QThread worker for 3D LUT generation and .cube export.

Runs the full LUT pipeline in a background thread:
  1. Fit the transfer model on source/reference (image statistics, once),
     unless the preview's fitted model is passed in.
  2. Generate a neutral 3D grid at the requested resolution.
  3. Apply the model and the user's adjustments (strengths, protections,
     mix, smoothing) — the same code path as the preview.
  4. Write the resulting grid to a ``.cube`` file on disk.

Progress is reported granularly across all four phases via the
``progress`` signal so the UI can display a smooth progress bar.
"""

from __future__ import annotations

import logging
import traceback
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from PySide6.QtCore import QThread, Signal

from stealzalut.core.color_transfer import ColorTransferEngine, TransferModel
from stealzalut.core.finishing import Finishing
from stealzalut.core.grading import GradingParams, build_grid
from stealzalut.core.lut_export import write_cube_file

logger = logging.getLogger(__name__)


# ── Progress allocation across phases ────────────────────────────────────────
# Phase 1: Transfer model fitting         →  0 – 20 %
# Phase 2: Neutral grid generation         → 20 – 30 %
# Phase 3: Grid baking (apply transform)  → 30 – 60 %
# Phase 4: .cube file writing             → 60 – 100 %


class ExportWorker(QThread):
    """Background worker that generates and exports a 3D LUT ``.cube`` file.

    Signals:
        progress (int): Percentage complete (0–100).
        result_ready (str): Absolute path to the exported ``.cube`` file.
        error (str): Human-readable error message if the export fails.

    Usage::

        worker = ExportWorker(
            source_rgb=source,
            reference_rgb=reference,
            lut_size=33,
            output_path=Path("~/Documents/my_lut.cube"),
        )
        worker.result_ready.connect(on_export_done)
        worker.progress.connect(progress_bar.setValue)
        worker.error.connect(on_error)
        worker.start()
    """

    progress = Signal(int)
    result_ready = Signal(str)
    error = Signal(str)

    def __init__(
        self,
        source_rgb: NDArray[np.uint8],
        reference_rgb: NDArray[np.uint8],
        lut_size: int,
        output_path: str | Path,
        title: str = "StealZaLut",
        group: str = "StealZaLut",
        method: str = "mkl",
        params: GradingParams | None = None,
        finishing: Finishing | None = None,
        model: TransferModel | None = None,
        parent: Any = None,
    ) -> None:
        super().__init__(parent)
        self._params = params or GradingParams()
        self._finishing = finishing
        self._model = model  # reuse the preview's fitted model when available
        self._source = source_rgb
        self._reference = reference_rgb
        self._lut_size = lut_size
        self._output_path = Path(output_path)
        self._title = title
        self._group = group
        self._method = method
        self._engine = ColorTransferEngine(method=method)

    def run(self) -> None:
        """Execute the full LUT generation pipeline in the worker thread."""
        try:
            total_points = self._lut_size ** 3
            logger.info(
                "ExportWorker started — LUT size %d (%d points), method: %s, output: %s",
                self._lut_size,
                total_points,
                self._method,
                self._output_path,
            )

            # ── Phase 1: Prepare color transfer parameters (0–20 %) ──────
            self.progress.emit(0)
            model = self._model
            if model is None:
                logger.debug("Phase 1/4: Analyzing color distributions for %s…", self._method)
                model = self._engine.fit(self._source, self._reference, method=self._method)
            self.progress.emit(20)

            # ── Phases 2–3: Bake transfer + adjustments onto the grid (20–60 %)
            logger.debug(
                "Phases 2-3/4: Baking %s transform and adjustments onto %d grid points…",
                self._method,
                total_points,
            )
            self.progress.emit(30)
            grid_result = build_grid(model, self._lut_size, self._params, self._finishing)
            baked_grid = grid_result.grid
            if grid_result.auto_smoothed:
                logger.info(
                    "Grid auto-smoothed to %.2f to remove folds", grid_result.smoothing_used
                )
            self.progress.emit(60)

            # ── Phase 4: Write output .cube file (60–100 %) ──────────────
            def _write_progress(pct: int) -> None:
                """Map write progress [0, 100] → [60, 100]."""
                mapped = 60 + int(pct * 0.4)
                self.progress.emit(min(mapped, 100))

            logger.debug("Phase 4/4: Writing .cube file…")
            result_path = write_cube_file(
                path=self._output_path,
                grid=baked_grid,
                size=self._lut_size,
                title=self._title,
                progress_callback=_write_progress,
            )

            self.progress.emit(100)
            logger.info("ExportWorker completed — file: %s", result_path)
            self.result_ready.emit(str(result_path))

        except Exception as exc:
            msg = f"LUT export failed: {exc}"
            logger.error(msg)
            logger.debug(traceback.format_exc())
            self.error.emit(msg)
