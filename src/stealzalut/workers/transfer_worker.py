"""QThread worker for colour transfer preview computation.

Runs the MKL colour transfer in a background thread so the GUI
remains responsive during heavy matrix operations.  Communicates
results back to the main thread exclusively via Qt Signals.
"""

from __future__ import annotations

import logging
import traceback
from typing import Any

import numpy as np
from numpy.typing import NDArray
from PySide6.QtCore import QThread, Signal

from stealzalut.core.color_transfer import ColorTransferEngine

logger = logging.getLogger(__name__)


class TransferWorker(QThread):
    """Background worker that applies MKL colour transfer.

    Signals:
        progress (int): Percentage complete (0–100).
        result_ready (object): The transferred RGB uint8 image (ndarray).
        error (str): Human-readable error message if the computation fails.

    Usage::

        worker = TransferWorker(source_rgb, reference_rgb)
        worker.result_ready.connect(on_preview_ready)
        worker.error.connect(on_error)
        worker.start()
    """

    progress = Signal(int)
    result_ready = Signal(object)
    error = Signal(str)

    def __init__(
        self,
        source: NDArray[np.uint8],
        reference: NDArray[np.uint8],
        parent: Any = None,
    ) -> None:
        super().__init__(parent)
        self._source = source
        self._reference = reference
        self._engine = ColorTransferEngine()

    def run(self) -> None:
        """Execute the colour transfer in the worker thread."""
        try:
            logger.info(
                "TransferWorker started — source %s, reference %s",
                self._source.shape,
                self._reference.shape,
            )
            self.progress.emit(10)

            # Compute the transferred image via color-matcher
            result = self._engine.transfer_image(
                self._source,
                self._reference,
            )
            self.progress.emit(90)

            logger.info("TransferWorker completed — result %s", result.shape)
            self.progress.emit(100)
            self.result_ready.emit(result)

        except Exception as exc:
            msg = f"Colour transfer failed: {exc}"
            logger.error(msg)
            logger.debug(traceback.format_exc())
            self.error.emit(msg)
