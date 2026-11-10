"""Finishing panel: colour wheels, tone curve, per-colour bands, saturation.

Every edit is emitted at once as a new :class:`Finishing` with a *merge
key* (one drag = one undo step); the window debounces the preview.  The
panel holds no colour logic beyond drawing: the maths live in
:mod:`stealzalut.core.finishing`.
"""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Any

import numpy as np
from numpy.typing import NDArray
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import (
    QBrush,
    QColor,
    QImage,
    QMouseEvent,
    QPainter,
    QPainterPath,
    QPaintEvent,
    QPalette,
    QPen,
)
from PySide6.QtWidgets import (
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from stealzalut.core.colorspace import gamut_compress, oklab_to_srgb, oklch_to_oklab
from stealzalut.core.finishing import (
    BANDS,
    IDENTITY_CURVE,
    Finishing,
    HueBand,
    Wheel,
    eval_curve,
    normalize_curve,
    with_band,
)

_SLIDER_STEPS = 100
#: Wheel tints below this amount snap to "no tint".
_WHEEL_SNAP = 0.04
_WHEEL_DISPLAY_L = 0.72
_WHEEL_DISPLAY_C = 0.13

_WHEEL_NAMES: tuple[tuple[str, str, str], ...] = (
    ("shadows", "Ombres", "Teinte et luminosité des tons sombres (lift)."),
    ("midtones", "Tons moyens", "Teinte et luminosité des tons moyens (gamma)."),
    ("highlights", "Hautes lumières", "Teinte et luminosité des tons clairs (gain)."),
)


def _slider(minimum: int, maximum: int, value: int, tooltip: str, name: str) -> QSlider:
    slider = QSlider(Qt.Orientation.Horizontal)
    slider.setObjectName(name)
    slider.setRange(minimum, maximum)
    slider.setValue(value)
    slider.setToolTip(tooltip)
    return slider


# ── Colour wheel ─────────────────────────────────────────────────────────────


_WHEEL_CACHE: dict[int, QImage] = {}


def wheel_image(size: int) -> QImage:
    """Hue/chroma disc in Oklch (angle = hue, radius = chroma), RGBA."""
    if size in _WHEEL_CACHE:
        return _WHEEL_CACHE[size]
    radius = size / 2.0
    ys, xs = np.mgrid[0:size, 0:size].astype(np.float64)
    dx = (xs + 0.5 - radius) / radius
    dy = -(ys + 0.5 - radius) / radius
    dist = np.hypot(dx, dy)
    hue = np.degrees(np.arctan2(dy, dx)) % 360.0
    lch = np.stack(
        [np.full(dist.size, _WHEEL_DISPLAY_L), np.clip(dist, 0, 1).ravel() * _WHEEL_DISPLAY_C,
         hue.ravel()],
        axis=1,
    )
    rgb = oklab_to_srgb(gamut_compress(oklch_to_oklab(lch)))
    alpha = np.clip((1.0 - dist) * radius, 0.0, 1.0).ravel()
    rgba = np.concatenate([rgb, alpha[:, None]], axis=1)
    data = np.ascontiguousarray((rgba * 255.0 + 0.5).astype(np.uint8).reshape(size, size, 4))
    image = QImage(data.data, size, size, size * 4, QImage.Format.Format_RGBA8888).copy()
    _WHEEL_CACHE[size] = image
    return image


class ColorWheel(QWidget):
    """Hue/amount picker: drag the dot, double-click to remove the tint.

    Signals:
        tint_changed (float, float): ``(hue in degrees, amount in [0, 1])``.
    """

    tint_changed = Signal(float, float)

    def __init__(self, diameter: int = 104, parent: Any = None) -> None:
        super().__init__(parent)
        self._diameter = diameter
        self._hue = 0.0
        self._amount = 0.0
        self.setFixedSize(diameter + 8, diameter + 8)
        self.setCursor(Qt.CursorShape.CrossCursor)
        self.setToolTip("Glisse le point vers une couleur · double-clic : sans teinte")

    @property
    def tint(self) -> tuple[float, float]:
        return self._hue, self._amount

    def set_tint(self, hue: float, amount: float) -> None:
        self._hue, self._amount = hue % 360.0, min(max(amount, 0.0), 1.0)
        self.update()

    def _center(self) -> QPointF:
        return QPointF(self.width() / 2.0, self.height() / 2.0)

    def tint_at(self, pos: QPointF) -> tuple[float, float]:
        """Tint under *pos* (widget coordinates)."""
        radius = self._diameter / 2.0
        dx = (pos.x() - self._center().x()) / radius
        dy = -(pos.y() - self._center().y()) / radius
        amount = min(math.hypot(dx, dy), 1.0)
        if amount < _WHEEL_SNAP:
            return self._hue, 0.0
        return math.degrees(math.atan2(dy, dx)) % 360.0, amount

    def point_for(self, hue: float, amount: float) -> QPointF:
        radius = self._diameter / 2.0 * amount
        rad = math.radians(hue)
        c = self._center()
        return QPointF(c.x() + radius * math.cos(rad), c.y() - radius * math.sin(rad))

    def paintEvent(self, _event: QPaintEvent) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        offset = (self.width() - self._diameter) / 2.0
        p.drawImage(QPointF(offset, offset), wheel_image(self._diameter))
        c = self._center()
        p.setPen(QPen(QColor(0, 0, 0, 90), 1.0))
        p.drawLine(QPointF(c.x() - 5, c.y()), QPointF(c.x() + 5, c.y()))
        p.drawLine(QPointF(c.x(), c.y() - 5), QPointF(c.x(), c.y() + 5))
        dot = self.point_for(self._hue, self._amount)
        p.setPen(QPen(QColor(20, 20, 24), 2.0))
        p.setBrush(QBrush(QColor(255, 255, 255)))
        p.drawEllipse(dot, 5.5, 5.5)

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._pick(event.position())

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if event.buttons() & Qt.MouseButton.LeftButton:
            self._pick(event.position())

    def mouseDoubleClickEvent(self, event: QMouseEvent) -> None:
        self.set_tint(self._hue, 0.0)
        self.tint_changed.emit(self._hue, 0.0)

    def _pick(self, pos: QPointF) -> None:
        hue, amount = self.tint_at(pos)
        if (hue, amount) != (self._hue, self._amount):
            self.set_tint(hue, amount)
            self.tint_changed.emit(hue, amount)


# ── Tone curve ───────────────────────────────────────────────────────────────


class ToneCurveEditor(QWidget):
    """Lightness curve: drag points, click to add, double-click a point to remove.

    Signals:
        curve_changed (object): The new curve (tuple of ``(x, y)`` points).
    """

    curve_changed = Signal(object)

    MARGIN = 8
    HIT_RADIUS = 8.0

    def __init__(self, parent: Any = None) -> None:
        super().__init__(parent)
        self.setObjectName("toneCurveEditor")
        self.setMinimumSize(170, 170)
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.setFixedSize(180, 180)
        self.setToolTip(
            "Courbe de luminosité (n'altère ni la teinte ni la saturation).\n"
            "Clic : ajouter un point · glisser : déplacer · double-clic sur un point : le retirer"
        )
        self._points: list[tuple[float, float]] = list(IDENTITY_CURVE)
        self._drag: int | None = None

    @property
    def curve(self) -> tuple[tuple[float, float], ...]:
        return tuple(self._points)

    def set_curve(self, points: tuple[tuple[float, float], ...]) -> None:
        self._points = list(normalize_curve(points))
        self._drag = None
        self.update()

    # Geometry
    def _area(self) -> QRectF:
        m = self.MARGIN
        return QRectF(m, m, self.width() - 2 * m, self.height() - 2 * m)

    def to_widget(self, x: float, y: float) -> QPointF:
        a = self._area()
        return QPointF(a.left() + x * a.width(), a.bottom() - y * a.height())

    def to_curve(self, pos: QPointF) -> tuple[float, float]:
        a = self._area()
        x = (pos.x() - a.left()) / a.width()
        y = (a.bottom() - pos.y()) / a.height()
        return min(max(x, 0.0), 1.0), min(max(y, 0.0), 1.0)

    def _hit(self, pos: QPointF) -> int | None:
        for i, (x, y) in enumerate(self._points):
            if (self.to_widget(x, y) - pos).manhattanLength() <= self.HIT_RADIUS * 1.4:
                return i
        return None

    def paintEvent(self, _event: QPaintEvent) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        a = self._area()
        p.fillRect(self.rect(), self.palette().color(QPalette.ColorRole.Base))
        grid = QPen(QColor(120, 125, 135, 70), 1.0)
        p.setPen(grid)
        for i in range(1, 4):
            x = a.left() + a.width() * i / 4
            y = a.top() + a.height() * i / 4
            p.drawLine(QPointF(x, a.top()), QPointF(x, a.bottom()))
            p.drawLine(QPointF(a.left(), y), QPointF(a.right(), y))
        p.setPen(QPen(QColor(120, 125, 135, 140), 1.0, Qt.PenStyle.DashLine))
        p.drawLine(a.bottomLeft(), a.topRight())

        xs: NDArray[np.float64] = np.linspace(0.0, 1.0, 96, dtype=np.float64)
        ys = eval_curve(tuple(self._points), xs)
        path = QPainterPath(self.to_widget(float(xs[0]), float(ys[0])))
        for x, y in zip(xs[1:], ys[1:], strict=True):
            path.lineTo(self.to_widget(float(x), float(y)))
        p.setPen(QPen(QColor(236, 238, 242), 2.0))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawPath(path)

        p.setBrush(QBrush(QColor(79, 128, 255)))
        p.setPen(QPen(QColor(255, 255, 255), 1.5))
        for x, y in self._points:
            p.drawEllipse(self.to_widget(x, y), 4.5, 4.5)

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() != Qt.MouseButton.LeftButton:
            return
        pos = event.position()
        hit = self._hit(pos)
        if hit is None:
            x, _ = self.to_curve(pos)
            if 0.02 < x < 0.98:
                y = float(eval_curve(tuple(self._points), np.array([x]))[0])
                self._points.append((x, y))
                self._points.sort()
                hit = self._points.index((x, y))
        self._drag = hit
        if hit is not None:
            self._move_point(hit, pos)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._drag is not None and event.buttons() & Qt.MouseButton.LeftButton:
            self._move_point(self._drag, event.position())

    def mouseReleaseEvent(self, _event: QMouseEvent) -> None:
        self._drag = None

    def mouseDoubleClickEvent(self, event: QMouseEvent) -> None:
        hit = self._hit(event.position())
        if hit is not None and 0 < hit < len(self._points) - 1:
            del self._points[hit]
            self._drag = None
            self.update()
            self.curve_changed.emit(self.curve)

    def _move_point(self, index: int, pos: QPointF) -> None:
        x, y = self.to_curve(pos)
        last = len(self._points) - 1
        if index == 0:
            x = 0.0
        elif index == last:
            x = 1.0
        else:
            x = min(max(x, self._points[index - 1][0] + 0.02), self._points[index + 1][0] - 0.02)
        # Keep the curve monotone: a point stays between its neighbours' heights
        if index > 0:
            y = max(y, self._points[index - 1][1])
        if index < last:
            y = min(y, self._points[index + 1][1])
        if self._points[index] != (x, y):
            self._points[index] = (x, y)
            self.update()
            self.curve_changed.emit(self.curve)


# ── Panel ────────────────────────────────────────────────────────────────────


class FinishingPanel(QWidget):
    """All finishing tools in one row.

    Signals:
        finishing_changed (object, str): The new :class:`Finishing` and a
            merge key for the undo history.
    """

    finishing_changed = Signal(object, str)

    def __init__(self, parent: Any = None) -> None:
        super().__init__(parent)
        self.setObjectName("finishingPanel")
        self._finishing = Finishing()

        root = QHBoxLayout(self)
        root.setContentsMargins(8, 4, 8, 4)
        root.setSpacing(18)

        # Wheels + saturation / vibrance
        left = QVBoxLayout()
        wheels_row = QHBoxLayout()
        wheels_row.setSpacing(10)
        self._wheels: dict[str, ColorWheel] = {}
        self._wheel_lightness: dict[str, QSlider] = {}
        for key, wheel_name, tooltip in _WHEEL_NAMES:
            column = QVBoxLayout()
            column.setSpacing(2)
            title = QLabel(wheel_name)
            title.setObjectName("adjustLabel")
            title.setAlignment(Qt.AlignmentFlag.AlignCenter)
            title.setToolTip(tooltip)
            wheel = ColorWheel()
            wheel.setObjectName(f"wheel_{key}")
            wheel.tint_changed.connect(lambda h, a, k=key: self._on_tint(k, h, a))
            light = _slider(
                -_SLIDER_STEPS,
                _SLIDER_STEPS,
                0,
                f"Luminosité — {wheel_name.lower()}",
                f"wheelLight_{key}",
            )
            light.valueChanged.connect(lambda v, k=key: self._on_wheel_lightness(k, v))
            column.addWidget(title)
            column.addWidget(wheel, alignment=Qt.AlignmentFlag.AlignHCenter)
            column.addWidget(light)
            wheels_row.addLayout(column)
            self._wheels[key] = wheel
            self._wheel_lightness[key] = light
        left.addLayout(wheels_row)

        globals_row = QGridLayout()
        globals_row.setHorizontalSpacing(8)
        self._saturation = _slider(
            -_SLIDER_STEPS, _SLIDER_STEPS, 0, "Saturation de toutes les couleurs", "finSaturation"
        )
        self._vibrance = _slider(
            -_SLIDER_STEPS,
            _SLIDER_STEPS,
            0,
            "Vibrance : sature surtout les couleurs ternes, épargne celles déjà vives",
            "finVibrance",
        )
        self._saturation.valueChanged.connect(lambda v: self._on_global("saturation", v))
        self._vibrance.valueChanged.connect(lambda v: self._on_global("vibrance", v))
        for row, (name, slider) in enumerate(
            (("Saturation", self._saturation), ("Vibrance", self._vibrance))
        ):
            label = QLabel(name)
            label.setObjectName("adjustLabel")
            globals_row.addWidget(label, row, 0)
            globals_row.addWidget(slider, row, 1)
        globals_row.setColumnStretch(1, 1)
        left.addLayout(globals_row)
        root.addLayout(left)

        # Tone curve
        middle = QVBoxLayout()
        middle.setSpacing(2)
        curve_title = QLabel("Courbe de luminosité")
        curve_title.setObjectName("adjustLabel")
        self._curve = ToneCurveEditor()
        self._curve.curve_changed.connect(self._on_curve)
        reset_curve = QPushButton("Courbe neutre")
        reset_curve.setObjectName("resetCurveButton")
        reset_curve.clicked.connect(lambda: self._on_curve(IDENTITY_CURVE, reset=True))
        middle.addWidget(curve_title)
        middle.addWidget(self._curve)
        middle.addWidget(reset_curve, alignment=Qt.AlignmentFlag.AlignLeft)
        middle.addStretch()
        root.addLayout(middle)

        # Per-colour bands
        right = QGridLayout()
        right.setHorizontalSpacing(8)
        right.setVerticalSpacing(1)
        for col, heading in enumerate(("Teinte", "Saturation", "Luminosité"), start=2):
            header = QLabel(heading)
            header.setObjectName("adjustLabel")
            header.setAlignment(Qt.AlignmentFlag.AlignCenter)
            right.addWidget(header, 0, col)
        self._band_sliders: dict[tuple[int, str], QSlider] = {}
        for row, (name, center, rgb) in enumerate(BANDS, start=1):
            swatch = QLabel()
            swatch.setFixedSize(12, 12)
            swatch.setStyleSheet(
                f"background-color: rgb{rgb}; border-radius: 6px; border: 1px solid rgba(0,0,0,90);"
            )
            label = QLabel(name)
            label.setObjectName("adjustLabel")
            right.addWidget(swatch, row, 0)
            right.addWidget(label, row, 1)
            for col, field in enumerate(("hue", "saturation", "lightness"), start=2):
                slider = _slider(
                    -_SLIDER_STEPS,
                    _SLIDER_STEPS,
                    0,
                    f"{name} : {('teinte', 'saturation', 'luminosité')[col - 2]}",
                    f"band_{row - 1}_{field}",
                )
                slider.setMinimumWidth(90)
                slider.valueChanged.connect(
                    lambda v, c=center, f=field, i=row - 1: self._on_band(i, c, f, v)
                )
                right.addWidget(slider, row, col)
                self._band_sliders[(row - 1, field)] = slider
        reset_all = QPushButton("Réinitialiser la finition")
        reset_all.setObjectName("resetFinishingButton")
        reset_all.clicked.connect(self.reset)
        right.addWidget(reset_all, len(BANDS) + 1, 0, 1, 5, Qt.AlignmentFlag.AlignRight)
        root.addLayout(right, stretch=1)

    # ── Public API ───────────────────────────────────────────────────────

    def finishing(self) -> Finishing:
        return self._finishing

    def set_finishing(self, finishing: Finishing) -> None:
        """Show *finishing* without emitting anything."""
        self._finishing = finishing
        sliders: list[QSlider] = [
            *self._wheel_lightness.values(),
            self._saturation,
            self._vibrance,
            *self._band_sliders.values(),
        ]
        for slider in sliders:
            slider.blockSignals(True)
        try:
            for key, wheel in self._wheels.items():
                value: Wheel = getattr(finishing, key)
                wheel.set_tint(value.hue, value.amount)
                self._wheel_lightness[key].setValue(round(value.lightness * _SLIDER_STEPS))
            self._saturation.setValue(round(finishing.saturation * _SLIDER_STEPS))
            self._vibrance.setValue(round(finishing.vibrance * _SLIDER_STEPS))
            self._curve.set_curve(finishing.tone_curve)
            for i, (_, center, _) in enumerate(BANDS):
                band = self._band(finishing, center)
                for field in ("hue", "saturation", "lightness"):
                    self._band_sliders[(i, field)].setValue(
                        round(getattr(band, field) * _SLIDER_STEPS)
                    )
        finally:
            for slider in sliders:
                slider.blockSignals(False)

    def reset(self) -> None:
        """Back to no finishing (one undo step)."""
        self.set_finishing(Finishing())
        self.finishing_changed.emit(self._finishing, "")

    # ── Internal ─────────────────────────────────────────────────────────

    @staticmethod
    def _band(finishing: Finishing, center: float) -> HueBand:
        for band in finishing.bands:
            if abs(band.center - center) < 1e-6:
                return band
        return HueBand(center=center)

    def _emit(self, finishing: Finishing, merge_key: str) -> None:
        if finishing == self._finishing:
            return
        self._finishing = finishing
        self.finishing_changed.emit(finishing, merge_key)

    def _on_tint(self, key: str, hue: float, amount: float) -> None:
        wheel = replace(getattr(self._finishing, key), hue=hue, amount=amount)
        self._emit(replace(self._finishing, **{key: wheel}), f"wheel_{key}")

    def _on_wheel_lightness(self, key: str, value: int) -> None:
        wheel = replace(getattr(self._finishing, key), lightness=value / _SLIDER_STEPS)
        self._emit(replace(self._finishing, **{key: wheel}), f"wheel_light_{key}")

    def _on_global(self, field: str, value: int) -> None:
        amount = value / _SLIDER_STEPS
        if field == "saturation":
            self._emit(replace(self._finishing, saturation=amount), field)
        else:
            self._emit(replace(self._finishing, vibrance=amount), field)

    def _on_curve(self, points: tuple[tuple[float, float], ...], reset: bool = False) -> None:
        if reset:
            self._curve.set_curve(points)
        curve = normalize_curve(points)
        self._emit(replace(self._finishing, tone_curve=curve), "" if reset else "curve")

    def _on_band(self, index: int, center: float, field: str, value: int) -> None:
        band = replace(self._band(self._finishing, center), **{field: value / _SLIDER_STEPS})
        self._emit(with_band(self._finishing, band), f"band_{index}_{field}")
