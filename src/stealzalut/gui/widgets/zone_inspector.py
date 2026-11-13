"""Inspector of the selected source zone.

Shows and edits one :class:`~stealzalut.core.correspondence.Link`: lock,
strengths of the transfer, manual retouch, target weights.  Every edit is emitted as
a new link (with a *merge key* so that dragging a slider forms a single
undo step); the window puts it in the table and calls :meth:`set_zone`
back with the result.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from stealzalut.core.correspondence import (
    RETOUCH_RANGE,
    Level,
    Link,
    link_without_target,
    set_target_weight,
)
from stealzalut.core.palette import Zone, ZonePalette
from stealzalut.core.zone_transfer import retouch_from_intent
from stealzalut.gui.zone_text import ORIGIN_WORDS, RETOUCH_KINDS, pct, signed_pct

_STRENGTHS: tuple[tuple[str, str, str], ...] = (
    ("strength", "Force", "Quantité de transfert pour cette zone (0 = inchangée)."),
    ("chroma_strength", "Couleur", "Part de la couleur de la cible prise par cette zone."),
    (
        "luminance_strength",
        "Lumière cible",
        "0 = la luminosité suit le reste de l'image (sans cassure).\n"
        "100 = la zone prend la luminosité de sa cible : peut créer des\n"
        "cassures si les zones changent d'ordre de luminosité.",
    ),
)
_STRENGTH_DEFAULTS = {"strength": 1.0, "chroma_strength": 1.0, "luminance_strength": 0.0}
_KEEP_TONE_TOOLTIP = (
    "0 = la zone s'assombrit ou s'éclaircit avec le reste de l'image.\n"
    "100 = la zone garde la luminosité qu'elle a dans ta photo.\n"
    "Utile pour une peau ou un sujet devenu trop sombre ; sa couleur suit toujours la cible."
)
#: Slider units for a retouch of 1.
_RETOUCH_STEPS = 100


def _swatch(hex_color: str, size: int = 16) -> QLabel:
    label = QLabel()
    label.setFixedSize(size, size)
    label.setStyleSheet(
        f"background-color: {hex_color}; border: 1px solid rgba(0,0,0,90); border-radius: 3px;"
    )
    return label


class ZoneInspector(QWidget):
    """Settings of one source zone.

    Signals:
        link_changed (object, str): The edited :class:`Link` and a merge key
            (``""`` for a discrete edit that is its own undo step).
    """

    link_changed = Signal(object, str)

    def __init__(self, parent: Any = None) -> None:
        super().__init__(parent)
        self.setObjectName("zoneInspector")
        self.setMinimumWidth(300)

        self._link: Link | None = None
        self._auto_link: Link | None = None
        self._reference: ZonePalette | None = None
        self._targets_key: tuple[int, tuple[int, ...]] | None = None

        root = QVBoxLayout(self)
        root.setContentsMargins(8, 4, 8, 4)
        root.setSpacing(6)

        # Header: swatch + title
        header = QHBoxLayout()
        self._header_swatch = _swatch("#000000", 20)
        self._title = QLabel()
        self._title.setObjectName("inspectorTitle")
        header.addWidget(self._header_swatch)
        header.addWidget(self._title, stretch=1)
        root.addLayout(header)

        self._hint = QLabel("Clique sur une zone source pour la régler.")
        self._hint.setObjectName("inspectorHint")
        self._hint.setWordWrap(True)
        root.addWidget(self._hint)

        self._body = QWidget()
        body = QVBoxLayout(self._body)
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(6)
        root.addWidget(self._body)

        self._lock = QCheckBox("Verrouiller (garder cette zone telle quelle)")
        self._lock.toggled.connect(self._on_lock)
        body.addWidget(self._lock)

        grid = QGridLayout()
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(3)
        self._strength_sliders: dict[str, QSlider] = {}
        self._strength_values: dict[str, QLabel] = {}
        row = 0
        for field, label, tooltip in _STRENGTHS:
            slider = QSlider(Qt.Orientation.Horizontal)
            slider.setObjectName(f"inspector_{field}")
            slider.setRange(0, 100)
            slider.setToolTip(tooltip)
            slider.valueChanged.connect(lambda v, f=field: self._on_strength(f, v))
            value = QLabel()
            value.setMinimumWidth(42)
            value.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            name = QLabel(label)
            name.setToolTip(tooltip)
            grid.addWidget(name, row, 0)
            grid.addWidget(slider, row, 1)
            grid.addWidget(value, row, 2)
            self._strength_sliders[field] = slider
            self._strength_values[field] = value
            row += 1

        retouch_title = QLabel("Retouche de la zone")
        retouch_title.setObjectName("inspectorSection")
        retouch_title.setToolTip(
            "Ces réglages s'ajoutent au transfert et ne sont pas atténués par les\n"
            "réglages globaux (intensités, protections, mix)."
        )
        grid.addWidget(retouch_title, row, 0, 1, 3)
        row += 1

        def add_row(name: str, label: str, tooltip: str, low: int, high: int) -> QSlider:
            nonlocal row
            slider = QSlider(Qt.Orientation.Horizontal)
            slider.setObjectName(name)
            slider.setRange(low, high)
            slider.setToolTip(tooltip)
            value = QLabel()
            value.setMinimumWidth(42)
            value.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            title = QLabel(label)
            title.setToolTip(tooltip)
            grid.addWidget(title, row, 0)
            grid.addWidget(slider, row, 1)
            grid.addWidget(value, row, 2)
            self._retouch_values[name] = value
            row += 1
            return slider

        self._retouch_values: dict[str, QLabel] = {}
        self._keep_tone = add_row(
            "inspector_keep_tone", "Garder ma lumière", _KEEP_TONE_TOOLTIP, 0, 100
        )
        self._keep_tone.valueChanged.connect(self._on_keep_tone)
        self._retouch_sliders: dict[str, QSlider] = {}
        low, high = (round(bound * _RETOUCH_STEPS) for bound in RETOUCH_RANGE)
        for field, _intent, label, _up, _down, tooltip in RETOUCH_KINDS:
            slider = add_row(f"retouch_{field}", label, tooltip, low, high)
            slider.valueChanged.connect(lambda v, f=field: self._on_retouch(f, v))
            self._retouch_sliders[field] = slider
        grid.setColumnStretch(1, 1)
        body.addLayout(grid)

        targets_title = QLabel("Cibles dans la référence")
        targets_title.setObjectName("inspectorSection")
        body.addWidget(targets_title)
        self._targets_box = QVBoxLayout()
        self._targets_box.setSpacing(3)
        body.addLayout(self._targets_box)
        self._target_rows: dict[int, tuple[QSlider, QLabel, QToolButton]] = {}

        self._rationale = QLabel()
        self._rationale.setObjectName("inspectorRationale")
        self._rationale.setWordWrap(True)
        body.addWidget(self._rationale)

        self._revert = QPushButton("Revenir à l'auto")
        self._revert.setObjectName("revertZoneButton")
        self._revert.clicked.connect(self._on_revert)
        body.addWidget(self._revert, alignment=Qt.AlignmentFlag.AlignLeft)

        root.addStretch()
        self.set_zone(None, None, None, None)

    # ── Public API ───────────────────────────────────────────────────────

    @property
    def link(self) -> Link | None:
        return self._link

    def set_zone(
        self,
        link: Link | None,
        zone: Zone | None,
        reference: ZonePalette | None,
        auto_link: Link | None,
    ) -> None:
        """Show *link* (``None`` = nothing selected)."""
        self._link, self._auto_link, self._reference = link, auto_link, reference
        if link is None or zone is None or reference is None:
            self._targets_key = None
            self._title.setText("Aucune zone sélectionnée")
            self._header_swatch.hide()
            self._hint.show()
            self._body.hide()
            self._fit_height()
            return

        self._hint.hide()
        self._body.show()
        self._header_swatch.show()
        self._header_swatch.setStyleSheet(
            f"background-color: {zone.hex}; border: 1px solid rgba(0,0,0,90); border-radius: 3px;"
        )
        self._title.setText(
            f"<b>S{zone.id}</b> · {pct(zone.area)} de l'image · réglée {ORIGIN_WORDS[link.origin]}"
        )

        key = (zone.id, tuple(sorted(link.targets)))
        if key != self._targets_key:
            self._rebuild_targets(link, reference)
            self._targets_key = key
        self._sync(link)
        self._fit_height()

    # ── Internal ─────────────────────────────────────────────────────────

    def _fit_height(self) -> None:
        """Never get squeezed below the content height (the parent scrolls instead)."""
        layout = self.layout()
        if layout is not None:
            layout.activate()
            self.setMinimumHeight(layout.sizeHint().height())

    def _sync(self, link: Link) -> None:
        """Show *link*'s values without emitting anything."""
        widgets: list[QWidget] = [
            self._lock,
            *self._strength_sliders.values(),
            self._keep_tone,
            *self._retouch_sliders.values(),
            *(row[0] for row in self._target_rows.values()),
        ]
        for w in widgets:
            w.blockSignals(True)
        try:
            self._lock.setChecked(link.locked)
            for field, slider in self._strength_sliders.items():
                value = getattr(link, field)
                value = _STRENGTH_DEFAULTS[field] if value is None else value
                slider.setValue(round(value * 100))
                slider.setEnabled(not link.locked)
                self._strength_values[field].setText(pct(value))
            self._strength_sliders["chroma_strength"].setEnabled(
                not link.locked and bool(link.targets)
            )
            self._strength_sliders["luminance_strength"].setEnabled(
                not link.locked and bool(link.targets)
            )
            self._keep_tone.setValue(round(link.keep_tone * 100))
            self._keep_tone.setEnabled(not link.locked)
            self._retouch_values["inspector_keep_tone"].setText(pct(link.keep_tone))
            for field, slider in self._retouch_sliders.items():
                value = self._retouch_value(link, field)
                slider.setValue(round(value * _RETOUCH_STEPS))
                slider.setEnabled(not link.locked)
                self._retouch_values[f"retouch_{field}"].setText(signed_pct(value))
            for ref_id, (slider, value, remove) in self._target_rows.items():
                weight = link.targets.get(ref_id, 0.0)
                slider.setValue(round(weight * 100))
                slider.setEnabled(not link.locked and len(link.targets) > 1)
                remove.setEnabled(len(link.targets) > 1)
                value.setText(pct(weight))
        finally:
            for w in widgets:
                w.blockSignals(False)

        self._rationale.setText(f"<i>Note : {link.rationale}</i>" if link.rationale else "")
        self._rationale.setVisible(bool(link.rationale))
        self._revert.setEnabled(self._auto_link is not None and link != self._auto_link)

    def _rebuild_targets(self, link: Link, reference: ZonePalette) -> None:
        while self._targets_box.count():
            item = self._targets_box.takeAt(0)
            widget = item.widget() if item is not None else None
            if widget is not None:
                widget.setParent(None)  # gone from the screen now, not at the next loop
                widget.deleteLater()
        self._target_rows = {}
        if not link.targets:
            empty = QLabel("Aucune : la zone est gardée telle quelle.")
            self._targets_box.addWidget(empty)
            return
        for ref_id in sorted(link.targets):
            row = QWidget()
            layout = QHBoxLayout(row)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.setSpacing(6)
            layout.addWidget(_swatch(reference.zone(ref_id).hex))
            layout.addWidget(QLabel(f"R{ref_id}"))
            slider = QSlider(Qt.Orientation.Horizontal)
            slider.setObjectName(f"target_weight_{ref_id}")
            slider.setRange(1, 99)
            slider.setToolTip("Part de cette cible dans la couleur de la zone")
            slider.valueChanged.connect(lambda v, r=ref_id: self._on_weight(r, v))
            value = QLabel()
            value.setMinimumWidth(42)
            value.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            remove = QToolButton()
            remove.setText("×")
            remove.setObjectName(f"remove_target_{ref_id}")
            remove.setToolTip("Retirer cette cible")
            remove.clicked.connect(lambda _checked=False, r=ref_id: self._on_remove(r))
            layout.addWidget(slider, stretch=1)
            layout.addWidget(value)
            layout.addWidget(remove)
            self._targets_box.addWidget(row)
            self._target_rows[ref_id] = (slider, value, remove)

    def _emit(self, link: Link, merge_key: str = "") -> None:
        if self._link is None or link == self._link:
            return
        self._link = link
        self.link_changed.emit(link, merge_key)

    def _on_lock(self, checked: bool) -> None:
        if self._link is not None:
            self._emit(replace(self._link, locked=checked, origin="user"))

    def _on_strength(self, field: str, value: int) -> None:
        if self._link is None:
            return
        self._strength_values[field].setText(f"{value} %")
        fraction = value / 100.0
        if field == "strength":
            new = replace(self._link, strength=fraction, origin="user")
        elif field == "chroma_strength":
            new = replace(self._link, chroma_strength=fraction, origin="user")
        else:
            new = replace(self._link, luminance_strength=fraction, origin="user")
        self._emit(new, f"{field}:{self._link.source_zone}")

    @staticmethod
    def _retouch_value(link: Link, field: str) -> float:
        """Slider position of *field*: the retouch plus what a level of an
        older recipe amounts to."""
        low, high = RETOUCH_RANGE
        legacy = getattr(retouch_from_intent(link.intent), field)
        return float(min(max(getattr(link.retouch, field) + legacy, low), high))

    def _on_keep_tone(self, value: int) -> None:
        if self._link is None:
            return
        self._retouch_values["inspector_keep_tone"].setText(f"{value} %")
        self._emit(
            replace(self._link, keep_tone=value / 100.0, origin="user"),
            f"keep_tone:{self._link.source_zone}",
        )

    def _on_retouch(self, field: str, value: int) -> None:
        if self._link is None:
            return
        fraction = value / _RETOUCH_STEPS
        self._retouch_values[f"retouch_{field}"].setText(signed_pct(fraction))
        # The slider now says everything: the level it replaces is dropped
        intent_field = next(k[1] for k in RETOUCH_KINDS if k[0] == field)
        neutral: Level = "0"
        self._emit(
            replace(
                self._link,
                retouch=replace(self._link.retouch, **{field: fraction}),
                intent=replace(self._link.intent, **{intent_field: neutral}),
                origin="user",
            ),
            f"retouch_{field}:{self._link.source_zone}",
        )

    def _on_weight(self, ref_id: int, value: int) -> None:
        if self._link is None:
            return
        self._emit(
            set_target_weight(self._link, ref_id, value / 100.0),
            f"weight_{ref_id}:{self._link.source_zone}",
        )

    def _on_remove(self, ref_id: int) -> None:
        if self._link is not None:
            self._emit(link_without_target(self._link, ref_id))

    def _on_revert(self) -> None:
        if self._auto_link is not None:
            self._emit(self._auto_link)
