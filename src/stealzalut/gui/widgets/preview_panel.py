"""Image preview panel.

Displays an image (typically the colour-transfer result) scaled to
fit the available space, with zoom and pan (see
:class:`~stealzalut.gui.widgets.image_view.ImageView`).  Also supports a
simple "empty" state with a placeholder message.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from numpy.typing import NDArray
from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget

from stealzalut.gui.overlay import highlight_zone
from stealzalut.gui.widgets.image_view import ImageView

logger = logging.getLogger(__name__)


class PreviewPanel(QWidget):
    """Widget displaying an image preview with auto-scaling.

    Usage::

        panel = PreviewPanel("Transfer Preview")
        panel.set_image(rgb_ndarray)
        panel.clear()
    """

    def __init__(
        self,
        title: str = "Preview",
        parent: Any = None,
    ) -> None:
        super().__init__(parent)

        self._title = title
        self._image_rgb: NDArray[np.uint8] | None = None
        self._highlight: NDArray[np.bool_] | None = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        # Title label
        self._title_label = QLabel(title)
        self._title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._title_label.setObjectName("previewTitle")
        layout.addWidget(self._title_label)

        # Image label
        self._image_label = ImageView("No preview available")
        self._image_label.setMinimumSize(200, 160)
        self._image_label.setObjectName("previewImage")
        layout.addWidget(self._image_label)

        self.setObjectName("previewPanel")

    # ── Public API ───────────────────────────────────────────────────────

    def set_image(self, image_rgb: NDArray[np.uint8]) -> None:
        """Display an RGB uint8 image, scaled to fit the widget.

        Args:
            image_rgb: RGB image of shape ``(H, W, 3)``, dtype ``uint8``.
        """
        self._image_rgb = np.ascontiguousarray(image_rgb)
        shown = self._image_rgb
        if self._highlight is not None:
            shown = highlight_zone(shown, self._highlight)
        self._show_rgb(shown)
        logger.debug("Preview updated: %dx%d", image_rgb.shape[1], image_rgb.shape[0])

    def set_highlight(self, mask: NDArray[np.bool_] | None) -> None:
        """Highlight a colour zone of the source on the preview (``None`` = off).

        The highlight is kept when a new preview image arrives.
        """
        self._highlight = mask
        if self._image_rgb is None:
            return
        shown = self._image_rgb if mask is None else highlight_zone(self._image_rgb, mask)
        self._show_rgb(shown)

    @property
    def image_rgb(self) -> NDArray[np.uint8] | None:
        """The current preview image (without highlight), or ``None``."""
        return self._image_rgb

    @property
    def view(self) -> ImageView:
        """The zoomable display of the preview."""
        return self._image_label

    @property
    def _pixmap(self) -> QPixmap | None:
        return self._image_label.pixmap_shown

    def clear(self) -> None:
        """Clear the preview and show the placeholder text."""
        self._image_rgb = None
        self._highlight = None
        self._image_label.set_rgb(None)

    # ── Internal ─────────────────────────────────────────────────────────

    def _show_rgb(self, rgb: NDArray[np.uint8]) -> None:
        self._image_label.set_rgb(rgb)
