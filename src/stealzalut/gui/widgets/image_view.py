"""Image display that can be zoomed and panned.

The image is fitted to the widget; the wheel zooms around the pointer, a
drag moves the zoomed image, a right click comes back to the whole image.
Two views can be kept on the same area (source and preview) through
:attr:`ImageView.view_changed` and :meth:`ImageView.set_view`.

It is a ``QLabel`` so that the stylesheet frames it and draws its
placeholder text; the image itself is painted by hand.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QImage, QMouseEvent, QPainter, QPaintEvent, QPixmap, QWheelEvent
from PySide6.QtWidgets import QLabel, QSizePolicy

#: Zoom range, 1 being the whole image fitted to the widget.
ZOOM_RANGE = (1.0, 16.0)
#: Zoom factor of one wheel notch.
ZOOM_STEP = 1.25
#: Pointer travel (px) from which a press is a drag, not a click.
DRAG_THRESHOLD = 4

ZOOM_TOOLTIP = "Molette : zoomer · glisser : déplacer · clic droit : image entière"


class ImageView(QLabel):
    """Shows an RGB image fitted to the widget, with zoom and pan.

    Signals:
        clicked: A left click that was not a drag.
        view_changed (float, float, float): Zoom and centre (fractions of the
            image width and height) after the user zoomed or panned.
    """

    clicked = Signal()
    view_changed = Signal(float, float, float)

    def __init__(self, placeholder: str = "", parent: Any = None) -> None:
        super().__init__(placeholder, parent)
        self._placeholder = placeholder
        self._pixmap: QPixmap | None = None
        self._zoom = 1.0
        self._centre = (0.5, 0.5)
        self._press: QPointF | None = None
        self._dragged = False
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setWordWrap(True)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    # ── Public API ───────────────────────────────────────────────────────

    @property
    def pixmap_shown(self) -> QPixmap | None:
        """The image on display, or ``None``."""
        return self._pixmap

    @property
    def zoom(self) -> float:
        """Current zoom (1 = whole image)."""
        return self._zoom

    @property
    def centre(self) -> tuple[float, float]:
        """Point of the image at the centre of the widget, as fractions."""
        return self._centre

    def set_rgb(self, rgb: NDArray[np.uint8] | None) -> None:
        """Show *rgb* (``(H, W, 3)`` uint8), or the placeholder when ``None``.

        The zoom is kept when the new image has the size of the previous one
        (a refreshed preview, a highlighted zone), and reset otherwise.
        """
        if rgb is None:
            self._pixmap = None
            self.setText(self._placeholder)
            self.setToolTip("")
            self.reset_view()
            return
        h, w, ch = rgb.shape
        image = QImage(rgb.data.tobytes(), w, h, ch * w, QImage.Format.Format_RGB888)
        same_size = self._pixmap is not None and self._pixmap.size() == image.size()
        self._pixmap = QPixmap.fromImage(image)
        self.setText("")
        self.setToolTip(ZOOM_TOOLTIP)
        if same_size:
            self.update()
        else:
            self.reset_view()

    def reset_view(self) -> None:
        """Come back to the whole image."""
        self.set_view(1.0, 0.5, 0.5)

    def set_view(self, zoom: float, cx: float, cy: float) -> None:
        """Set zoom and centre without emitting :attr:`view_changed`."""
        self._zoom = float(np.clip(zoom, *ZOOM_RANGE))
        self._centre = self._clamped(cx, cy)
        self.update()

    # ── Geometry ─────────────────────────────────────────────────────────

    def _scale(self) -> float:
        """Widget pixels per image pixel at the current zoom."""
        assert self._pixmap is not None
        area = self.contentsRect()
        fit = min(area.width() / self._pixmap.width(), area.height() / self._pixmap.height())
        return max(fit, 1e-6) * self._zoom

    def _clamped(self, cx: float, cy: float) -> tuple[float, float]:
        """Centre kept so that the image never leaves an empty margin."""
        if self._pixmap is None:
            return (0.5, 0.5)
        scale = self._scale()
        area = self.contentsRect()
        out = []
        for value, size, room in (
            (cx, self._pixmap.width() * scale, area.width()),
            (cy, self._pixmap.height() * scale, area.height()),
        ):
            half = room / (2.0 * size) if size > room else 0.5
            out.append(float(np.clip(value, half, 1.0 - half)))
        return (out[0], out[1])

    def _target(self) -> QRectF:
        """Where the whole image lands, in widget coordinates."""
        assert self._pixmap is not None
        scale = self._scale()
        width, height = self._pixmap.width() * scale, self._pixmap.height() * scale
        middle = QRectF(self.contentsRect()).center()
        cx, cy = self._centre
        return QRectF(middle.x() - cx * width, middle.y() - cy * height, width, height)

    # ── Events ───────────────────────────────────────────────────────────

    def paintEvent(self, event: QPaintEvent) -> None:
        super().paintEvent(event)  # frame, background and placeholder text
        if self._pixmap is None:
            return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        painter.setClipRect(self.contentsRect())
        painter.drawPixmap(self._target(), self._pixmap, QRectF(self._pixmap.rect()))
        painter.end()

    def resizeEvent(self, event: Any) -> None:
        super().resizeEvent(event)
        self._centre = self._clamped(*self._centre)

    def wheelEvent(self, event: QWheelEvent) -> None:
        if self._pixmap is None or event.angleDelta().y() == 0:
            super().wheelEvent(event)
            return
        notches = event.angleDelta().y() / 120.0
        zoom = float(np.clip(self._zoom * ZOOM_STEP**notches, *ZOOM_RANGE))
        if zoom != self._zoom:
            # Keep the image point under the pointer where it is
            before = self._target()
            pointer = event.position()
            fx = (pointer.x() - before.x()) / before.width()
            fy = (pointer.y() - before.y()) / before.height()
            self._zoom = zoom
            after = self._target()
            middle = QRectF(self.contentsRect()).center()
            cx = fx + (middle.x() - pointer.x()) / after.width()
            cy = fy + (middle.y() - pointer.y()) / after.height()
            self._centre = self._clamped(cx, cy)
            self.update()
            self.view_changed.emit(self._zoom, *self._centre)
        event.accept()

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._press = event.position()
            self._dragged = False
        elif event.button() == Qt.MouseButton.RightButton and self._zoom > 1.0:
            self.reset_view()
            self.view_changed.emit(self._zoom, *self._centre)
        event.accept()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._press is None or self._pixmap is None:
            return
        move = event.position() - self._press
        if not self._dragged and move.manhattanLength() < DRAG_THRESHOLD:
            return
        self._dragged = True
        if self._zoom <= 1.0:
            return
        target = self._target()
        cx = self._centre[0] - move.x() / target.width()
        cy = self._centre[1] - move.y() / target.height()
        self._centre = self._clamped(cx, cy)
        self._press = event.position()
        self.update()
        self.view_changed.emit(self._zoom, *self._centre)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton and self._press is not None:
            self._press = None
            if not self._dragged:
                self.clicked.emit()
        event.accept()
