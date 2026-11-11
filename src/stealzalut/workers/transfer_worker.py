"""QThread worker for colour transfer preview computation.

Builds the same 3D LUT grid that the export would produce (transfer model
+ user adjustments) and applies it to a down-scaled copy of the source
image, so the preview shows exactly what the exported ``.cube`` file will
do.  A previously fitted model can be passed in: moving a slider then only
rebuilds the grid, which takes a fraction of a second.  Runs in a background thread so
the GUI remains responsive; communicates exclusively via Qt Signals.
"""

from __future__ import annotations

import logging
import traceback
from collections.abc import Callable
from typing import Any

import numpy as np
from numpy.typing import NDArray
from PySide6.QtCore import QThread, Signal

from stealzalut.core.color_transfer import ColorTransferEngine, TransferModel
from stealzalut.core.finishing import Finishing
from stealzalut.core.grading import GradingParams, build_grid
from stealzalut.core.lut_apply import apply_lut
from stealzalut.core.palette import DEFAULT_ZONE_COUNT
from stealzalut.utils.image_io import create_thumbnail

logger = logging.getLogger(__name__)

#: Longest edge (px) of the image the preview LUT is applied to.
PREVIEW_RENDER_MAX_SIZE = 1280


class TransferWorker(QThread):
    """Background worker that renders the LUT preview.

    Signals:
        progress (int): Percentage complete (0–100).
        model_ready (object): The fitted :class:`TransferModel`, emitted only
            when it was computed by this worker (not when passed in).  It is
            the raw fit; *prepare* (if given) is applied after this signal,
            before baking (e.g. the zone separation chosen by the user).
        grid_ready (object): The :class:`GridResult` (grid + diagnostics).
        result_ready (object): The graded RGB uint8 preview image (ndarray).
        error (str): Human-readable error message if the computation fails.

    Usage::

        worker = TransferWorker(source_rgb, reference_rgb, lut_size=33)
        worker.result_ready.connect(on_preview_ready)
        worker.error.connect(on_error)
        worker.start()
    """

    progress = Signal(int)
    model_ready = Signal(object)
    grid_ready = Signal(object)
    result_ready = Signal(object)
    error = Signal(str)

    def __init__(
        self,
        source: NDArray[np.uint8],
        reference: NDArray[np.uint8],
        method: str = "mkl",
        lut_size: int = 33,
        render_max_size: int = PREVIEW_RENDER_MAX_SIZE,
        params: GradingParams | None = None,
        finishing: Finishing | None = None,
        model: TransferModel | None = None,
        zone_count: int = DEFAULT_ZONE_COUNT,
        prepare: Callable[[TransferModel], TransferModel] | None = None,
        reference_weights: NDArray[np.float64] | None = None,
        parent: Any = None,
    ) -> None:
        super().__init__(parent)
        self._source = source
        self._reference = reference
        self._method = method
        self._lut_size = lut_size
        self._render_max_size = render_max_size
        self._params = params or GradingParams()
        self._finishing = finishing
        self._model = model
        self._zone_count = zone_count
        self._prepare = prepare
        self._reference_weights = reference_weights
        self._engine = ColorTransferEngine(method=method)

    def run(self) -> None:
        """Fit the transfer, bake the grid and apply it to the preview image."""
        try:
            logger.info(
                "TransferWorker started — source %s, reference %s, method: %s, LUT %d",
                self._source.shape,
                self._reference.shape,
                self._method,
                self._lut_size,
            )
            self.progress.emit(10)

            model = self._model
            if model is None:
                model = self._fit()
                self.model_ready.emit(model)
                if self._prepare is not None:
                    model = self._prepare(model)
            self.progress.emit(60)

            grid_result = build_grid(model, self._lut_size, self._params, self._finishing)
            self.grid_ready.emit(grid_result)
            self.progress.emit(75)

            preview_src = create_thumbnail(self._source, max_size=self._render_max_size)
            result = apply_lut(preview_src, grid_result.grid, self._lut_size)
            self.progress.emit(95)

            logger.info("TransferWorker completed — result %s", result.shape)
            self.progress.emit(100)
            self.result_ready.emit(result)

        except Exception as exc:
            msg = f"Colour transfer failed: {exc}"
            logger.error(msg)
            logger.debug(traceback.format_exc())
            self.error.emit(msg)

    def _fit(self) -> TransferModel:
        if self._method == "zones":
            from stealzalut.core.zone_transfer import fit_zone_transfer

            return fit_zone_transfer(self._source, self._reference, self._zone_count)
        if self._method == "look":
            from stealzalut.core.look import fit_look

            return fit_look(self._source, self._reference, self._reference_weights)
        return self._engine.fit(self._source, self._reference, method=self._method)
