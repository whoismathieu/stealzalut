"""Correspondence editor: which reference zones colour each source zone.

Source zones are drawn as swatches on the left, reference zones on the
right (height ∝ area).  Curves link each source zone to its targets
(thickness ∝ weight, colour = who set the link: grey auto, white user,
violet AI).

Interactions:

* hover a swatch → :attr:`zone_hovered` (the window highlights it on the images);
* click a source swatch → :attr:`zone_selected` (shown in the inspector);
* drag a source swatch onto a reference swatch → replace its target
  (with Shift: add a target);
* right-click a source swatch → lock / back to auto / remove a target.

The widget only edits the table (:mod:`stealzalut.core.correspondence`)
and emits :attr:`table_changed`; it contains no colour logic.
"""

from __future__ import annotations

from typing import Any

from PySide6.QtCore import QEvent, QPointF, QRectF, Qt, Signal
from PySide6.QtGui import (
    QBrush,
    QColor,
    QContextMenuEvent,
    QFont,
    QHelpEvent,
    QMouseEvent,
    QPainter,
    QPainterPath,
    QPaintEvent,
    QPalette,
    QPen,
)
from PySide6.QtWidgets import QApplication, QMenu, QSizePolicy, QToolTip, QWidget

from stealzalut.core.correspondence import (
    CorrespondenceTable,
    Origin,
    remove_target,
    retarget,
    revert_to_auto,
    set_locked,
)
from stealzalut.core.palette import Zone, ZonePalette
from stealzalut.core.zone_transfer import Conflict
from stealzalut.gui.zone_text import pct, reference_tooltip, source_tooltip

SOURCE = "source"
REFERENCE = "reference"

MIN_SWATCH_H = 20
SWATCH_GAP = 4
SWATCH_W = 112
MARGIN = 10
HEADER_H = 22

ORIGIN_COLORS: dict[Origin, QColor] = {
    "auto": QColor(138, 145, 158),
    "user": QColor(236, 238, 242),
    "ai": QColor(167, 139, 250),
}
ACCENT = QColor(79, 128, 255)
CONFLICT_COLOR = QColor(255, 159, 67)


def column_heights(
    areas: list[float],
    height: float,
    min_h: float = MIN_SWATCH_H,
    gap: float = SWATCH_GAP,
) -> list[float]:
    """Swatch heights: at least *min_h*, the remaining space ∝ area."""
    k = len(areas)
    if k == 0:
        return []
    available = max(height - gap * (k - 1), k * min_h)
    extra = available - k * min_h
    total = sum(areas) or 1.0
    return [min_h + a / total * extra for a in areas]


def min_height_for(zone_count: int) -> int:
    """Smallest widget height that shows *zone_count* swatches."""
    return int(2 * MARGIN + HEADER_H + zone_count * MIN_SWATCH_H + (zone_count - 1) * SWATCH_GAP)


class CorrespondenceView(QWidget):
    """Two columns of zone swatches linked by curves.

    Signals:
        table_changed (object): The edited :class:`CorrespondenceTable`.
        zone_selected (int): Source zone clicked (0 = none).
        zone_hovered (str, int): ``(side, zone_id)`` under the mouse;
            ``("", 0)`` when the mouse leaves every swatch.
    """

    table_changed = Signal(object)
    zone_selected = Signal(int)
    zone_hovered = Signal(str, int)

    def __init__(self, parent: Any = None) -> None:
        super().__init__(parent)
        self.setObjectName("correspondenceView")
        self.setMouseTracking(True)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMinimumSize(380, 160)

        self._source: ZonePalette | None = None
        self._reference: ZonePalette | None = None
        self._table: CorrespondenceTable | None = None
        self._auto: CorrespondenceTable | None = None
        self._conflict_zones: set[int] = set()
        self._selected = 0
        self._hover: tuple[str, int] | None = None
        self._placeholder = "Charge une source et une référence pour voir leurs zones."

        self._drag_from = 0
        self._press_pos: QPointF | None = None
        self._drag_pos: QPointF | None = None

    # ── Public API ───────────────────────────────────────────────────────

    def set_data(
        self,
        source: ZonePalette,
        reference: ZonePalette,
        table: CorrespondenceTable,
        auto_table: CorrespondenceTable,
    ) -> None:
        """Show new zones (new images or zone count)."""
        self._source, self._reference = source, reference
        self._table, self._auto = table, auto_table
        self._selected = 0
        self._hover = None
        self._conflict_zones = set()
        self.setMinimumHeight(min_height_for(max(len(source.zones), len(reference.zones))))
        self.update()

    def set_table(self, table: CorrespondenceTable) -> None:
        """Show *table* (after an undo or an edit made elsewhere)."""
        self._table = table
        self.update()

    def set_conflicts(self, conflicts: list[Conflict]) -> None:
        self._conflict_zones = {z for c in conflicts for z in (c.zone_a, c.zone_b)}
        self.update()

    def clear(self, message: str | None = None) -> None:
        """Remove the zones and show *message*."""
        self._source = self._reference = None
        self._table = self._auto = None
        self._selected = 0
        self._hover = None
        self._conflict_zones = set()
        if message is not None:
            self._placeholder = message
        self.setMinimumHeight(160)
        self.update()

    def select_zone(self, zone_id: int, emit: bool = False) -> None:
        self._selected = zone_id
        self.update()
        if emit:
            self.zone_selected.emit(zone_id)

    @property
    def table(self) -> CorrespondenceTable | None:
        return self._table

    @property
    def selected_zone(self) -> int:
        return self._selected

    @property
    def conflict_zones(self) -> set[int]:
        return set(self._conflict_zones)

    # ── Geometry ─────────────────────────────────────────────────────────

    def _column(self, palette: ZonePalette, x: float) -> dict[int, QRectF]:
        top = MARGIN + HEADER_H
        heights = column_heights([z.area for z in palette.zones], self.height() - top - MARGIN)
        rects: dict[int, QRectF] = {}
        y = float(top)
        for zone, h in zip(palette.zones, heights, strict=True):
            rects[zone.id] = QRectF(x, y, SWATCH_W, h)
            y += h + SWATCH_GAP
        return rects

    def swatch_rects(self) -> tuple[dict[int, QRectF], dict[int, QRectF]]:
        """``(source rects, reference rects)`` keyed by zone id."""
        if self._source is None or self._reference is None:
            return {}, {}
        return (
            self._column(self._source, MARGIN),
            self._column(self._reference, self.width() - MARGIN - SWATCH_W),
        )

    def zone_at(self, pos: QPointF) -> tuple[str, int] | None:
        """``(side, zone_id)`` of the swatch under *pos*, if any."""
        src_rects, ref_rects = self.swatch_rects()
        for side, rects in ((SOURCE, src_rects), (REFERENCE, ref_rects)):
            for zone_id, rect in rects.items():
                if rect.contains(pos):
                    return side, zone_id
        return None

    # ── Painting ─────────────────────────────────────────────────────────

    def paintEvent(self, _event: QPaintEvent) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.fillRect(self.rect(), self.palette().color(QPalette.ColorRole.Base))
        text_color = self.palette().color(QPalette.ColorRole.Text)

        if self._source is None or self._reference is None or self._table is None:
            p.setPen(QColor(138, 145, 158))
            p.drawText(
                self.rect().adjusted(20, 0, -20, 0),
                Qt.AlignmentFlag.AlignCenter | Qt.TextFlag.TextWordWrap,
                self._placeholder,
            )
            return

        src_rects, ref_rects = self.swatch_rects()

        # Column headers
        header_font = QFont(self.font())
        header_font.setPointSizeF(header_font.pointSizeF() * 0.9)
        p.setFont(header_font)
        p.setPen(QColor(157, 165, 180))
        p.drawText(QRectF(MARGIN, MARGIN, SWATCH_W, HEADER_H - 4), Qt.AlignmentFlag.AlignLeft,
                   "Source")
        p.drawText(
            QRectF(self.width() - MARGIN - SWATCH_W, MARGIN, SWATCH_W, HEADER_H - 4),
            Qt.AlignmentFlag.AlignRight,
            "Référence",
        )

        # Links (focused ones drawn last, on top)
        focus = self._focus()
        links = sorted(
            self._table.links, key=lambda link: link.source_zone in focus.sources
        )
        for link in links:
            if link.locked or link.source_zone not in src_rects:
                continue
            for ref_id, weight in link.targets.items():
                if ref_id not in ref_rects:
                    continue
                focused = not focus.active or (
                    link.source_zone in focus.sources or ref_id in focus.references
                )
                self._draw_link(
                    p,
                    src_rects[link.source_zone],
                    ref_rects[ref_id],
                    weight,
                    ORIGIN_COLORS[link.origin],
                    alpha=(0.35 + 0.65 * link.strength) * (1.0 if focused else 0.25),
                    dashed=link.strength == 0.0,
                )

        # Swatches
        for zone in self._source.zones:
            link = self._table.link(zone.id)
            self._draw_swatch(
                p,
                src_rects[zone.id],
                zone,
                f"S{zone.id}",
                selected=zone.id == self._selected,
                hovered=self._hover == (SOURCE, zone.id),
                conflict=zone.id in self._conflict_zones,
                locked=link.locked,
                marked=not link.intent.is_neutral or link.strength < 1.0,
            )
        for zone in self._reference.zones:
            self._draw_swatch(
                p,
                ref_rects[zone.id],
                zone,
                f"R{zone.id}",
                selected=False,
                hovered=self._hover == (REFERENCE, zone.id),
                conflict=False,
                locked=False,
                marked=False,
            )

        # Link being dragged
        if self._drag_pos is not None and self._drag_from in src_rects:
            start = src_rects[self._drag_from]
            target = self.zone_at(self._drag_pos)
            end_rect = (
                ref_rects[target[1]]
                if target is not None and target[0] == REFERENCE
                else QRectF(self._drag_pos, self._drag_pos)
            )
            self._draw_link(p, start, end_rect, 0.5, text_color, alpha=0.9, dashed=False)

    def _focus(self) -> _Focus:
        if self._table is None:
            return _Focus()
        if self._drag_pos is not None:
            return _Focus(sources={self._drag_from})
        if self._hover is not None:
            side, zone_id = self._hover
            if side == SOURCE:
                return _Focus(sources={zone_id})
            return _Focus(references={zone_id})
        if self._selected:
            return _Focus(sources={self._selected})
        return _Focus()

    @staticmethod
    def _draw_link(
        p: QPainter,
        a: QRectF,
        b: QRectF,
        weight: float,
        color: QColor,
        alpha: float,
        dashed: bool,
    ) -> None:
        start = QPointF(a.right(), a.center().y())
        end = QPointF(b.left(), b.center().y())
        dx = max((end.x() - start.x()) * 0.5, 20.0)
        path = QPainterPath(start)
        path.cubicTo(start + QPointF(dx, 0), end - QPointF(dx, 0), end)
        c = QColor(color)
        c.setAlphaF(max(0.0, min(1.0, alpha)))
        pen = QPen(c, 1.5 + 6.0 * weight)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        if dashed:
            pen.setStyle(Qt.PenStyle.DashLine)
        p.setPen(pen)
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawPath(path)

    def _draw_swatch(
        self,
        p: QPainter,
        rect: QRectF,
        zone: Zone,
        name: str,
        *,
        selected: bool,
        hovered: bool,
        conflict: bool,
        locked: bool,
        marked: bool,
    ) -> None:
        fill = QColor(zone.hex)
        if selected:
            border = QPen(ACCENT, 3.0)
        elif conflict:
            border = QPen(CONFLICT_COLOR, 2.5)
        elif hovered:
            border = QPen(QColor(255, 255, 255), 2.0)
        else:
            border = QPen(QColor(0, 0, 0, 90), 1.0)
        p.setPen(border)
        p.setBrush(QBrush(fill))
        p.drawRoundedRect(rect.adjusted(1, 1, -1, -1), 4, 4)

        ink = QColor(20, 20, 24) if fill.lightnessF() > 0.6 else QColor(245, 246, 248)
        p.setPen(ink)
        text_rect = rect.adjusted(8, 0, -8, 0)
        bold = QFont(self.font())
        bold.setBold(True)
        p.setFont(bold)
        p.drawText(text_rect, Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, name)
        p.setFont(self.font())
        p.drawText(
            text_rect, Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter, pct(zone.area)
        )

        if locked:
            self._draw_padlock(p, QPointF(rect.left() + 38, rect.center().y()), ink)
        if marked:
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(ACCENT)
            p.drawEllipse(QPointF(rect.right() - 4, rect.top() + 4), 3.5, 3.5)
        if conflict:
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(CONFLICT_COLOR)
            p.drawEllipse(QPointF(rect.left() - 0.5, rect.center().y()), 4, 4)

    @staticmethod
    def _draw_padlock(p: QPainter, center: QPointF, ink: QColor) -> None:
        body = QRectF(center.x() - 5, center.y() - 1, 10, 7)
        p.setPen(QPen(ink, 1.6))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawArc(QRectF(center.x() - 3.5, center.y() - 7, 7, 9), 0, 180 * 16)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(ink)
        p.drawRoundedRect(body, 1.5, 1.5)

    # ── Mouse ────────────────────────────────────────────────────────────

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            hit = self.zone_at(event.position())
            if hit is not None and hit[0] == SOURCE:
                self._drag_from = hit[1]
                self._press_pos = event.position()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        pos = event.position()
        if self._drag_from and self._press_pos is not None:
            moved = (pos - self._press_pos).manhattanLength()
            if self._drag_pos is not None or moved >= QApplication.startDragDistance():
                self._drag_pos = pos
                self.setCursor(Qt.CursorShape.ClosedHandCursor)
                self.update()
        self._set_hover(self.zone_at(pos))
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton and self._drag_from:
            source_zone = self._drag_from
            if self._drag_pos is not None:
                hit = self.zone_at(event.position())
                if hit is not None and hit[0] == REFERENCE and self._table is not None:
                    add = bool(event.modifiers() & Qt.KeyboardModifier.ShiftModifier)
                    self._emit_table(retarget(self._table, source_zone, hit[1], add=add))
                self.select_zone(source_zone, emit=True)
            else:
                self.select_zone(source_zone, emit=True)
            self._end_drag()
        super().mouseReleaseEvent(event)

    def leaveEvent(self, event: QEvent) -> None:
        self._set_hover(None)
        super().leaveEvent(event)

    def contextMenuEvent(self, event: QContextMenuEvent) -> None:
        hit = self.zone_at(QPointF(event.pos()))
        if hit is None or hit[0] != SOURCE:
            return
        self.select_zone(hit[1], emit=True)
        self.build_context_menu(hit[1]).exec(event.globalPos())

    def event(self, event: QEvent) -> bool:
        if event.type() == QEvent.Type.ToolTip and isinstance(event, QHelpEvent):
            text = self._tooltip(QPointF(event.pos()))
            if text:
                QToolTip.showText(event.globalPos(), text, self)
            else:
                QToolTip.hideText()
                event.ignore()
            return True
        return super().event(event)

    def build_context_menu(self, zone_id: int) -> QMenu:
        """Right-click menu of source zone *zone_id*."""
        menu = QMenu(self)
        if self._table is None or self._auto is None:
            return menu
        table, auto = self._table, self._auto
        link = table.link(zone_id)

        locked = link.locked
        lock = menu.addAction("Déverrouiller" if locked else "Verrouiller (garder telle quelle)")
        lock.triggered.connect(lambda: self._emit_table(set_locked(table, zone_id, not locked)))

        back = menu.addAction("Revenir à l'auto")
        back.setEnabled(link != auto.link(zone_id))
        back.triggered.connect(lambda: self._emit_table(revert_to_auto(table, auto, zone_id)))

        remove_menu = menu.addMenu("Retirer une cible")
        remove_menu.setEnabled(len(link.targets) > 1)
        for ref_id, weight in sorted(link.targets.items()):
            action = remove_menu.addAction(f"R{ref_id} ({pct(weight)})")
            action.triggered.connect(
                lambda _checked=False, r=ref_id: self._emit_table(remove_target(table, zone_id, r))
            )
        return menu

    # ── Internal ─────────────────────────────────────────────────────────

    def _emit_table(self, table: CorrespondenceTable) -> None:
        if table == self._table:
            return
        self._table = table
        self.update()
        self.table_changed.emit(table)

    def _set_hover(self, hit: tuple[str, int] | None) -> None:
        if hit == self._hover:
            return
        self._hover = hit
        self.update()
        self.zone_hovered.emit(*(hit if hit is not None else ("", 0)))

    def _end_drag(self) -> None:
        self._drag_from = 0
        self._press_pos = None
        self._drag_pos = None
        self.unsetCursor()
        self.update()

    def _tooltip(self, pos: QPointF) -> str:
        hit = self.zone_at(pos)
        if hit is None or self._table is None or self._source is None or self._reference is None:
            return ""
        side, zone_id = hit
        if side == SOURCE:
            return source_tooltip(self._source.zone(zone_id), self._table.link(zone_id))
        used_by = [
            link.source_zone
            for link in self._table.links
            if zone_id in link.targets and not link.locked
        ]
        return reference_tooltip(self._reference.zone(zone_id), used_by)


class _Focus:
    """Zones whose links are emphasised (the others are dimmed)."""

    def __init__(self, sources: set[int] | None = None, references: set[int] | None = None):
        self.sources = sources or set()
        self.references = references or set()

    @property
    def active(self) -> bool:
        return bool(self.sources or self.references)
