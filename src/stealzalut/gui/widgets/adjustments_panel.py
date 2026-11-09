"""Adjustment sliders applied on top of the colour transfer.

Emits a :class:`~stealzalut.core.grading.GradingParams` whenever the user
changes a slider (debounced so dragging does not flood the workers).
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import QGridLayout, QLabel, QPushButton, QSlider, QWidget

from stealzalut.core.grading import GradingParams

#: Delay between the last slider move and the recomputation.
DEBOUNCE_MS = 150


@dataclass(frozen=True, slots=True)
class _SliderSpec:
    field: str  # GradingParams attribute
    label: str
    maximum: float  # slider range is [0, maximum]
    tooltip: str


_SPECS: tuple[_SliderSpec, ...] = (
    _SliderSpec(
        "luminance_strength",
        "Intensité luminance",
        1.0,
        "Part du contraste et de la luminosité de la référence appliquée.\n"
        "Baisser quand les deux images n'ont pas la même lumière : "
        "on garde les couleurs sans écraser le contraste de la source.",
    ),
    _SliderSpec(
        "chroma_strength",
        "Intensité couleur",
        1.0,
        "Part des couleurs (teinte et saturation) de la référence appliquée.",
    ),
    _SliderSpec(
        "preserve_neutrals",
        "Préserver les neutres",
        1.0,
        "Garde les gris et les blancs neutres (balance des blancs conservée).",
    ),
    _SliderSpec(
        "skin_protection",
        "Protection peau",
        1.0,
        "Garde la teinte et la saturation naturelles des tons chair\n"
        "(leur luminosité suit le reste de l'image, pour éviter les cassures).\n"
        "Touche aussi un peu les oranges et beiges proches de la peau.",
    ),
    _SliderSpec(
        "mix",
        "Mix",
        1.0,
        "Quantité globale de l'effet (0 = image d'origine).",
    ),
    _SliderSpec(
        "smoothing",
        "Lissage",
        3.0,
        "Lisse la LUT pour éviter cassures et dégradés en escalier.\n"
        "Il est augmenté automatiquement si les réglages créent des inversions.",
    ),
)

_STEPS_PER_UNIT = 100


class AdjustmentsPanel(QWidget):
    """Grid of labelled sliders bound to :class:`GradingParams`.

    Signals:
        params_changed (object): The new :class:`GradingParams`, debounced.
    """

    params_changed = Signal(object)

    def __init__(self, parent: Any = None) -> None:
        super().__init__(parent)
        self.setObjectName("adjustmentsPanel")

        self._sliders: dict[str, QSlider] = {}
        self._value_labels: dict[str, QLabel] = {}

        self._debounce = QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.setInterval(DEBOUNCE_MS)
        self._debounce.timeout.connect(self._emit_params)

        layout = QGridLayout(self)
        layout.setContentsMargins(8, 4, 8, 4)
        layout.setHorizontalSpacing(10)
        layout.setVerticalSpacing(4)

        defaults = GradingParams()
        columns = 2  # two slider columns
        for i, spec in enumerate(_SPECS):
            row, col = divmod(i, columns)
            base = col * 3

            name = QLabel(spec.label)
            name.setObjectName("adjustLabel")
            name.setToolTip(spec.tooltip)

            slider = QSlider(Qt.Orientation.Horizontal)
            slider.setObjectName(f"slider_{spec.field}")
            slider.setRange(0, round(spec.maximum * _STEPS_PER_UNIT))
            slider.setValue(round(getattr(defaults, spec.field) * _STEPS_PER_UNIT))
            slider.setToolTip(spec.tooltip)
            slider.valueChanged.connect(self._on_slider_moved)

            value = QLabel()
            value.setObjectName("adjustValue")
            value.setMinimumWidth(40)
            value.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

            layout.addWidget(name, row, base)
            layout.addWidget(slider, row, base + 1)
            layout.addWidget(value, row, base + 2)
            layout.setColumnStretch(base + 1, 1)

            self._sliders[spec.field] = slider
            self._value_labels[spec.field] = value
            self._refresh_value_label(spec)

        self._reset_btn = QPushButton("Réinitialiser")
        self._reset_btn.setObjectName("resetAdjustmentsButton")
        self._reset_btn.setToolTip("Remettre tous les réglages par défaut (transfert brut)")
        self._reset_btn.clicked.connect(self.reset)
        last_row = (len(_SPECS) + columns - 1) // columns
        layout.addWidget(self._reset_btn, last_row, 0, 1, columns * 3, Qt.AlignmentFlag.AlignRight)

    # ── Public API ───────────────────────────────────────────────────────

    def params(self) -> GradingParams:
        """Current slider values as :class:`GradingParams`."""
        values = {
            spec.field: self._sliders[spec.field].value() / _STEPS_PER_UNIT for spec in _SPECS
        }
        return GradingParams(**values)

    def set_params(self, params: GradingParams) -> None:
        """Move the sliders to *params* (emits one debounced change)."""
        for f in fields(GradingParams):
            slider = self._sliders[f.name]
            slider.setValue(round(getattr(params, f.name) * _STEPS_PER_UNIT))

    def reset(self) -> None:
        """Restore the defaults (raw transfer, no adjustment)."""
        self.set_params(GradingParams())

    # ── Internal ─────────────────────────────────────────────────────────

    def _on_slider_moved(self, _value: int) -> None:
        for spec in _SPECS:
            self._refresh_value_label(spec)
        self._debounce.start()

    def _refresh_value_label(self, spec: _SliderSpec) -> None:
        raw = self._sliders[spec.field].value()
        text = f"{raw} %" if spec.maximum == 1.0 else f"{raw / _STEPS_PER_UNIT:.2f}"
        self._value_labels[spec.field].setText(text)

    def _emit_params(self) -> None:
        self.params_changed.emit(self.params())
