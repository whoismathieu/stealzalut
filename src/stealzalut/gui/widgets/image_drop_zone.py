"""Drag-and-drop image widget.

Displays a placeholder label when empty, or a scaled thumbnail when
an image has been loaded (via drag-and-drop or file dialog).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QDragEnterEvent, QDropEvent, QImage, QMouseEvent, QPixmap
from PySide6.QtWidgets import QFileDialog, QLabel, QSizePolicy, QVBoxLayout, QWidget

from stealzalut.utils.image_io import (
    SUPPORTED_EXTENSIONS,
    bgr_to_rgb,
    create_thumbnail,
    load_image_bgr,
)

logger = logging.getLogger(__name__)

_RAW_FILTER_EXTS = (
    "*.cr2 *.cr3 *.nef *.arw *.raf *.rw2 *.orf *.pef *.dng *.raw "
    "*.crw *.nrw *.srf *.sr2 *.rwl *.3fr *.fff *.iiq *.mrw *.x3f"
)

_FILTER = (
    "All Supported Images ({all_ext});;"
    f"Camera RAW & DNG ({_RAW_FILTER_EXTS});;"
    "Standard Images (*.jpg *.jpeg *.png *.tiff *.tif *.bmp *.webp);;"
    "Apple HEIC / HEIF (*.heic *.heif *.hif);;"
    "All Files (*.*)"
).format(all_ext=" ".join(f"*{ext}" for ext in sorted(SUPPORTED_EXTENSIONS)))


class ImageDropZone(QWidget):
    """Widget accepting image files via drag-and-drop or click-to-browse.

    Signals:
        image_loaded (object): Emitted with the loaded RGB uint8 ndarray
            when a valid image is successfully loaded.
        image_cleared: Emitted when the image is removed.

    The widget stores the **full-resolution RGB image** internally and
    only displays a down-scaled thumbnail.
    """

    image_loaded = Signal(object)
    image_cleared = Signal()

    def __init__(
        self,
        label_text: str = "Drop image here\nor click to browse",
        max_preview: int = 280,
        parent: Any = None,
    ) -> None:
        super().__init__(parent)

        self._label_text = label_text
        self._max_preview = max_preview
        self._image_rgb: NDArray[np.uint8] | None = None
        self._image_path: Path | None = None

        # Accept drops
        self.setAcceptDrops(True)

        # Layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self._label = QLabel(label_text)
        self._label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._label.setWordWrap(True)
        self._label.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )
        self._label.setMinimumSize(260, 200)
        self._label.setObjectName("dropZoneLabel")
        layout.addWidget(self._label)

        self.setObjectName("imageDropZone")

    # ── Public API ───────────────────────────────────────────────────────

    @property
    def image_rgb(self) -> NDArray[np.uint8] | None:
        """The full-resolution RGB image, or ``None`` if empty."""
        return self._image_rgb

    @property
    def image_path(self) -> Path | None:
        """Path of the loaded image file, or ``None``."""
        return self._image_path

    @property
    def has_image(self) -> bool:
        """Whether an image is currently loaded."""
        return self._image_rgb is not None

    def clear_image(self) -> None:
        """Remove the loaded image and reset to placeholder."""
        self._image_rgb = None
        self._image_path = None
        self._label.setPixmap(QPixmap())
        self._label.setText(self._label_text)
        self.image_cleared.emit()

    def load_from_path(self, path: str | Path) -> bool:
        """Load an image from *path*.

        Returns:
            ``True`` if the image was loaded successfully.
        """
        path = Path(path)
        try:
            bgr = load_image_bgr(path)
            rgb = bgr_to_rgb(bgr)
            self._image_rgb = rgb
            self._image_path = path
            self._update_thumbnail(bgr)
            self.image_loaded.emit(rgb)
            logger.info("Loaded image: %s (%s)", path.name, rgb.shape)
            return True
        except (FileNotFoundError, ValueError) as exc:
            logger.warning("Failed to load image %s: %s", path, exc)
            return False

    # ── Drag-and-drop ────────────────────────────────────────────────────

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        if event.mimeData().hasUrls():
            # Accept if at least one URL looks like a supported image
            for url in event.mimeData().urls():
                if url.isLocalFile():
                    suffix = Path(url.toLocalFile()).suffix.lower()
                    if suffix in SUPPORTED_EXTENSIONS:
                        event.acceptProposedAction()
                        return
        event.ignore()

    def dropEvent(self, event: QDropEvent) -> None:
        for url in event.mimeData().urls():
            if url.isLocalFile():
                path = Path(url.toLocalFile())
                if path.suffix.lower() in SUPPORTED_EXTENSIONS and self.load_from_path(path):
                    event.acceptProposedAction()
                    return
        event.ignore()

    # ── Click to browse ──────────────────────────────────────────────────

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._open_file_dialog()
        super().mousePressEvent(event)

    def _open_file_dialog(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Image",
            "",
            _FILTER,
        )
        if path:
            self.load_from_path(path)

    # ── Internal ─────────────────────────────────────────────────────────

    def _update_thumbnail(self, bgr_image: NDArray[np.uint8]) -> None:
        """Generate and display a thumbnail from the BGR image."""
        thumb = create_thumbnail(bgr_image, self._max_preview)
        # Convert BGR → RGB for Qt
        rgb = cv2.cvtColor(thumb, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        qt_image = QImage(
            rgb.data.tobytes(),
            w,
            h,
            bytes_per_line,
            QImage.Format.Format_RGB888,
        )
        pixmap = QPixmap.fromImage(qt_image)
        self._label.setText("")
        self._label.setPixmap(pixmap)

