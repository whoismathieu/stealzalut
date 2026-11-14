"""Layout helpers for a window that is resized.

:class:`ResponsiveRow` keeps a toolbar readable when the window gets narrow;
:class:`SplitterSection` folds a section of a vertical splitter down to its
header.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from PySide6.QtCore import QEvent, QObject, QSize
from PySide6.QtWidgets import (
    QBoxLayout,
    QHBoxLayout,
    QLayout,
    QSplitter,
    QToolButton,
    QWidget,
)

_QWIDGETSIZE_MAX = 16777215

_EXPANDED, _COLLAPSED = "▾", "▸"


class ResponsiveRow(QWidget):
    """Two groups of controls on one row, or on two when the row is too narrow.

    Without it, a window narrower than what the row needs squeezes the
    controls until they overlap.  The right group stays aligned to the right.
    """

    def __init__(self, spacing: int = 12, parent: Any = None) -> None:
        super().__init__(parent)
        self._left = QWidget()
        self.left = QHBoxLayout(self._left)
        self._right = QWidget()
        self.right = QHBoxLayout(self._right)
        for layout in (self.left, self.right):
            layout.setContentsMargins(0, 0, 0, 0)
            layout.setSpacing(spacing)
        self.right.addStretch()

        self._box = QBoxLayout(QBoxLayout.Direction.LeftToRight, self)
        self._box.setContentsMargins(0, 0, 0, 0)
        # The row decides its own minimum (see minimumSizeHint), not its layout
        self._box.setSizeConstraint(QLayout.SizeConstraint.SetNoConstraint)
        self._box.setSpacing(spacing)
        self._box.addWidget(self._left, stretch=1)
        self._box.addWidget(self._right)

    @property
    def stacked(self) -> bool:
        """Whether the two groups are on two rows."""
        return self._box.direction() == QBoxLayout.Direction.TopToBottom

    def needed_width(self) -> int:
        """Width under which the two groups no longer fit on one row."""
        return (
            self._left.minimumSizeHint().width()
            + self._right.minimumSizeHint().width()
            + self._box.spacing()
        )

    def minimumSizeHint(self) -> QSize:
        """The row never asks for more width than its two groups stacked."""
        hint = super().minimumSizeHint()
        left, right = self._left.minimumSizeHint(), self._right.minimumSizeHint()
        return QSize(max(left.width(), right.width()), hint.height())

    def resizeEvent(self, event: Any) -> None:
        super().resizeEvent(event)
        wanted = (
            QBoxLayout.Direction.TopToBottom
            if self.width() < self.needed_width()
            else QBoxLayout.Direction.LeftToRight
        )
        if wanted != self._box.direction():
            self._box.setDirection(wanted)


def collapse_button(tooltip: str) -> QToolButton:
    """The small arrow that folds a section."""
    button = QToolButton()
    button.setObjectName("collapseButton")
    button.setText(_EXPANDED)
    button.setCheckable(True)
    button.setToolTip(tooltip)
    return button


class SplitterSection(QObject):
    """Folds a widget of a vertical splitter down to its header.

    Args:
        splitter: The splitter holding *widget*.
        widget: The section.
        body: What disappears when the section is folded.
        button: The toggle (checked = folded).
        header_height: Height of the section once folded.
    """

    def __init__(
        self,
        splitter: QSplitter,
        widget: QWidget,
        body: QWidget,
        button: QToolButton,
        header_height: Callable[[], int],
    ) -> None:
        super().__init__(widget)
        self._splitter = splitter
        self._widget = widget
        self._body = body
        self._button = button
        self._header_height = header_height
        self._sizes: list[int] | None = None
        button.toggled.connect(self.set_collapsed)
        widget.installEventFilter(self)

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:
        """Keep a folded section at the height of its header when that changes
        (a toolbar that wraps on two rows in a narrower window)."""
        if (
            watched is self._widget
            and self.collapsed
            and event.type() in (QEvent.Type.Resize, QEvent.Type.LayoutRequest)
        ):
            height = self._header_height()
            if height != self._widget.maximumHeight():
                self._widget.setFixedHeight(height)
        return False

    @property
    def collapsed(self) -> bool:
        return self._button.isChecked()

    def set_collapsed(self, collapsed: bool) -> None:
        """Fold or unfold the section; the other sections take or give the room."""
        if self._button.isChecked() != collapsed:
            self._button.setChecked(collapsed)  # comes back here through the signal
            return
        self._button.setText(_COLLAPSED if collapsed else _EXPANDED)
        if collapsed:
            self._sizes = self._splitter.sizes()
            self._body.hide()
            self._widget.setFixedHeight(self._header_height())
            return
        self._body.show()
        self._widget.setMinimumHeight(0)
        self._widget.setMaximumHeight(_QWIDGETSIZE_MAX)
        if self._sizes is not None:
            self._splitter.setSizes(self._sizes)
