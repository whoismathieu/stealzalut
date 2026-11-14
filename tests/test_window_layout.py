"""The window when it is resized: toolbars, zoom on the images, folded sections."""

from __future__ import annotations

import itertools
import sys
from typing import cast

import numpy as np
import pytest
from PySide6.QtCore import QPoint, QPointF, QRect, Qt
from PySide6.QtGui import QWheelEvent
from PySide6.QtTest import QTest
from PySide6.QtWidgets import (
    QAbstractButton,
    QApplication,
    QComboBox,
    QLabel,
    QPushButton,
    QSlider,
    QSpinBox,
    QWidget,
)

from stealzalut.config import Settings
from stealzalut.gui.main_window import MainWindow
from stealzalut.gui.widgets.image_drop_zone import ImageDropZone
from stealzalut.gui.widgets.image_view import ImageView
from stealzalut.gui.widgets.responsive import ResponsiveRow

_CONTROLS = (QAbstractButton, QComboBox, QLabel, QSlider, QSpinBox)


@pytest.fixture(scope="session")
def qapp() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    return cast("QApplication", app)


@pytest.fixture()
def window(qapp: QApplication) -> MainWindow:
    window = MainWindow(settings=Settings(lut_default_size=33, preview_max_size=256))
    window.show()
    yield window
    window.stop_workers()
    window.close()


def _settle(qapp: QApplication) -> None:
    for _ in range(30):
        qapp.processEvents()


def _problems(row: QWidget) -> list[str]:
    """Controls of *row* that overlap another one or stick out of it."""
    shown = [
        child
        for child in row.findChildren(QWidget)
        if isinstance(child, _CONTROLS) and child.isVisible()
    ]
    rects = [(c, QRect(c.mapTo(row, QPoint(0, 0)), c.size())) for c in shown]
    found = [
        f"{a.objectName()} overlaps {b.objectName()}"
        for (a, ra), (b, rb) in itertools.combinations(rects, 2)
        if ra.intersects(rb)
    ]
    found += [f"{c.objectName()} sticks out" for c, r in rects if not row.rect().contains(r)]
    found += [
        f"{c.objectName()} is squeezed"
        for c in shown
        if c.width() < c.minimumSizeHint().width() or c.height() < c.minimumSizeHint().height()
    ]
    return found


def _wheel(view: ImageView, notches: int, at: QPointF) -> None:
    event = QWheelEvent(
        at,
        QPointF(view.mapToGlobal(at.toPoint())),
        QPoint(0, 0),
        QPoint(0, 120 * notches),
        Qt.MouseButton.NoButton,
        Qt.KeyboardModifier.NoModifier,
        Qt.ScrollPhase.NoScrollPhase,
        False,
    )
    QApplication.sendEvent(view, event)


def _picture(height: int = 60, width: int = 90) -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.integers(0, 255, (height, width, 3), dtype=np.uint8)


class TestToolbars:
    @pytest.mark.parametrize("method", ["look", "zones", "mkl"])
    @pytest.mark.parametrize("width", [900, 1000, 1150, 1280, 1600])
    def test_no_control_overlaps_at_any_width(
        self, qapp: QApplication, window: MainWindow, method: str, width: int
    ) -> None:
        window._algo_combo.setCurrentIndex(window._algo_combo.findData(method))
        window.resize(width, 800)
        _settle(qapp)
        assert window.width() == width
        assert _problems(window._controls_row) == []
        if method == "zones":
            assert _problems(window._zones_row) == []

    def test_the_window_can_hold_its_content_at_its_smallest(
        self, qapp: QApplication, window: MainWindow
    ) -> None:
        window._algo_combo.setCurrentIndex(window._algo_combo.findData("zones"))
        _settle(qapp)
        central = window.centralWidget()
        assert central.minimumSizeHint().width() <= window.minimumWidth()

    def test_one_row_at_1280_two_when_narrow(self, qapp: QApplication, window: MainWindow) -> None:
        window.resize(1280, 800)
        _settle(qapp)
        assert not window._controls_row.stacked
        window.resize(900, 800)
        _settle(qapp)
        assert window._controls_row.stacked
        window.resize(1280, 800)
        _settle(qapp)
        assert not window._controls_row.stacked

    def test_a_row_stacks_under_the_width_it_needs(self, qapp: QApplication) -> None:
        row = ResponsiveRow()
        row.left.addWidget(QPushButton("one rather long label"))
        row.right.addWidget(QPushButton("another rather long label"))
        row.show()
        row.resize(row.needed_width() + 40, 80)
        _settle(qapp)
        assert not row.stacked
        row.resize(row.needed_width() - 40, 80)
        _settle(qapp)
        assert row.stacked and _problems(row) == []
        assert row.minimumSizeHint().width() < row.needed_width()


class TestZoom:
    def _view(self, qapp: QApplication) -> ImageView:
        view = ImageView("empty")
        view.resize(300, 200)
        view.show()
        view.set_rgb(_picture())
        _settle(qapp)
        return view

    def test_the_wheel_zooms_around_the_pointer(self, qapp: QApplication) -> None:
        view = self._view(qapp)
        assert view.zoom == 1.0
        _wheel(view, 4, QPointF(280, 180))  # bottom right
        assert view.zoom == pytest.approx(1.25**4)
        assert view.centre[0] > 0.5 and view.centre[1] > 0.5
        _wheel(view, -20, QPointF(10, 10))
        assert view.zoom == 1.0 and view.centre == (0.5, 0.5)

    def test_the_zoom_stops_at_its_bounds(self, qapp: QApplication) -> None:
        view = self._view(qapp)
        _wheel(view, 60, QPointF(150, 100))
        assert view.zoom == 16.0

    def test_a_drag_moves_the_zoomed_image_and_is_not_a_click(self, qapp: QApplication) -> None:
        view = self._view(qapp)
        clicks: list[bool] = []
        view.clicked.connect(lambda: clicks.append(True))
        _wheel(view, 6, QPointF(150, 100))
        before = view.centre
        QTest.mousePress(view, Qt.MouseButton.LeftButton, pos=QPoint(150, 100))
        QTest.mouseMove(view, QPoint(110, 80))
        QTest.mouseRelease(view, Qt.MouseButton.LeftButton, pos=QPoint(110, 80))
        assert view.centre[0] > before[0] and view.centre[1] > before[1]
        assert clicks == []
        QTest.mouseClick(view, Qt.MouseButton.LeftButton, pos=QPoint(150, 100))
        assert clicks == [True]

    def test_a_right_click_shows_the_whole_image(self, qapp: QApplication) -> None:
        view = self._view(qapp)
        _wheel(view, 6, QPointF(20, 20))
        QTest.mouseClick(view, Qt.MouseButton.RightButton, pos=QPoint(150, 100))
        assert view.zoom == 1.0 and view.centre == (0.5, 0.5)

    def test_a_refreshed_image_keeps_the_zoom_a_new_one_resets_it(
        self, qapp: QApplication
    ) -> None:
        view = self._view(qapp)
        _wheel(view, 5, QPointF(40, 40))
        zoom, centre = view.zoom, view.centre
        view.set_rgb(_picture())  # the preview after a slider moved
        assert (view.zoom, view.centre) == (zoom, centre)
        view.set_rgb(_picture(80, 50))  # another picture
        assert view.zoom == 1.0
        view.set_rgb(None)
        assert view.pixmap_shown is None and view.text() == "empty"

    def test_the_zoomed_image_always_fills_the_view(self, qapp: QApplication) -> None:
        view = self._view(qapp)
        _wheel(view, 8, QPointF(0, 0))  # top-left corner
        target, area = view._target(), view.contentsRect()
        assert target.left() <= area.left() and target.top() <= area.top()
        assert target.right() >= area.right() and target.bottom() >= area.bottom()

    def test_a_click_on_a_drop_zone_still_opens_the_file_dialog(
        self, qapp: QApplication, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        zone = ImageDropZone()
        zone.resize(320, 260)
        zone.show()
        opened: list[bool] = []
        monkeypatch.setattr(zone, "_open_file_dialog", lambda: opened.append(True))
        zone.view.clicked.disconnect()
        zone.view.clicked.connect(zone._open_file_dialog)
        QTest.mouseClick(zone.view, Qt.MouseButton.LeftButton, pos=QPoint(100, 100))
        assert opened == [True]

    def test_source_and_preview_show_the_same_area(
        self, qapp: QApplication, window: MainWindow
    ) -> None:
        source, preview = window._source_zone.view, window._preview.view
        source.set_rgb(_picture())
        preview.set_rgb(_picture())
        _settle(qapp)
        _wheel(preview, 6, QPointF(30, 30))
        assert source.zoom == preview.zoom > 1.0
        _wheel(source, -1, QPointF(30, 30))
        assert preview.zoom == source.zoom
        assert window._reference_zone.view.zoom == 1.0


class TestFoldedSections:
    def test_the_tabs_fold_to_their_bar_and_come_back(
        self, qapp: QApplication, window: MainWindow
    ) -> None:
        window.resize(1280, 900)
        _settle(qapp)
        tabs = window._bottom_tabs
        open_height, images_height = tabs.height(), window._main_splitter.sizes()[0]
        window._tabs_collapse_btn.click()
        _settle(qapp)
        assert tabs.height() == tabs.tabBar().sizeHint().height() < 60
        assert not window._adjustments.isVisible()
        assert window._main_splitter.sizes()[0] > images_height  # the images take the room
        window._tabs_collapse_btn.click()
        _settle(qapp)
        assert window._adjustments.isVisible() and tabs.height() == open_height

    def test_choosing_a_tab_unfolds_the_section(
        self, qapp: QApplication, window: MainWindow
    ) -> None:
        window._tabs_section.set_collapsed(True)
        _settle(qapp)
        bar = window._bottom_tabs.tabBar()
        QTest.mouseClick(bar, Qt.MouseButton.LeftButton, pos=bar.tabRect(1).center())
        _settle(qapp)
        assert not window._tabs_section.collapsed
        assert window._finishing_panel.isVisible()

    def test_the_zones_fold_to_their_toolbar(self, qapp: QApplication, window: MainWindow) -> None:
        window._algo_combo.setCurrentIndex(window._algo_combo.findData("zones"))
        window.resize(1280, 900)
        _settle(qapp)
        window._corr_collapse_btn.click()
        _settle(qapp)
        assert not window._corr_body.isVisible()
        assert window._zone_count_spin.isVisible()
        assert _problems(window._zones_row) == []
        folded = window._corr_group.height()
        assert folded < 110
        # Too narrow for one row: the toolbar wraps and the folded section follows it
        window._tau_slider.setMinimumWidth(400)
        window.resize(900, 900)
        _settle(qapp)
        assert window._zones_row.stacked
        assert window._corr_group.height() > folded
        assert _problems(window._zones_row) == []
        window._corr_collapse_btn.click()
        _settle(qapp)
        assert window._corr_body.isVisible() and window._corr_group.height() > 150

    def test_the_bottom_panels_can_be_made_smaller_than_their_content(
        self, qapp: QApplication, window: MainWindow
    ) -> None:
        assert window._bottom_tabs.minimumSizeHint().height() < 120
        assert window._bottom_tabs.minimumSizeHint().width() < 500
