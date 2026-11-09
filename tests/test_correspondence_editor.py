"""Tests for the correspondence editor (phase 2b): table edits, history,
zone highlight, editor widgets and their wiring in the main window."""

from __future__ import annotations

import json
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest
from PySide6.QtCore import QEvent, QPointF, Qt
from PySide6.QtGui import QMouseEvent
from PySide6.QtWidgets import QApplication, QMessageBox

from stealzalut.config import Settings
from stealzalut.core.correspondence import (
    CorrespondenceTable,
    Intent,
    Link,
    link_without_target,
    remove_target,
    retarget,
    revert_to_auto,
    set_locked,
    set_target_weight,
)
from stealzalut.core.grading import GradingParams
from stealzalut.core.matching import match_zones
from stealzalut.core.palette import ZonePalette, analyze_zones
from stealzalut.gui.history import MERGE_WINDOW_S, EditState, History
from stealzalut.gui.overlay import DIM_FACTOR, highlight_zone
from stealzalut.gui.widgets.correspondence_view import (
    MIN_SWATCH_H,
    REFERENCE,
    SOURCE,
    CorrespondenceView,
    column_heights,
)
from stealzalut.gui.widgets.zone_inspector import ZoneInspector


@pytest.fixture(scope="session")
def qapp() -> QApplication:
    app = QApplication.instance()
    if not isinstance(app, QApplication):
        app = QApplication(sys.argv)
    return app


def _stripes(colors: list[tuple[int, int, int]], seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    img = np.zeros((90, 60 * len(colors), 3), dtype=np.float64)
    for i, c in enumerate(colors):
        img[:, i * 60 : (i + 1) * 60] = c
    img += rng.normal(0, 3, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


SOURCE_IMG = _stripes([(30, 70, 30), (120, 170, 230), (224, 172, 140)])
REFERENCE_IMG = _stripes([(30, 110, 120), (215, 180, 120), (255, 150, 70)], seed=1)


@pytest.fixture(scope="module")
def palettes() -> tuple[ZonePalette, ZonePalette, CorrespondenceTable]:
    src = analyze_zones(SOURCE_IMG, 3)
    ref = analyze_zones(REFERENCE_IMG, 3)
    return src, ref, CorrespondenceTable.from_matching(match_zones(src, ref))


# ── Table edits ──────────────────────────────────────────────────────────────


class TestTableEdits:
    TABLE = CorrespondenceTable.from_matching({1: {1: 1.0}, 2: {2: 0.6, 3: 0.4}, 3: {3: 1.0}})

    def test_retarget_replaces(self) -> None:
        new = retarget(self.TABLE, 2, 1)
        assert new.link(2).targets == {1: 1.0}
        assert new.link(2).origin == "user"
        assert new.link(1) == self.TABLE.link(1)

    def test_retarget_add_shares_weight(self) -> None:
        new = retarget(self.TABLE, 2, 1, add=True)
        targets = new.link(2).targets
        assert targets[1] == pytest.approx(1 / 3)
        assert targets[2] / targets[3] == pytest.approx(0.6 / 0.4)
        assert sum(targets.values()) == pytest.approx(1.0)
        assert retarget(new, 2, 1, add=True) == new  # already a target

    def test_retarget_unlocks_and_clears_rationale(self) -> None:
        table = self.TABLE.with_link(Link(1, {1: 1.0}, locked=True, origin="ai", rationale="x"))
        link = retarget(table, 1, 2).link(1)
        assert not link.locked and link.rationale == "" and link.origin == "user"

    def test_remove_target(self) -> None:
        new = remove_target(self.TABLE, 2, 2)
        assert new.link(2).targets == {3: pytest.approx(1.0)}
        assert remove_target(new, 2, 3) == new  # last target stays
        assert link_without_target(self.TABLE.link(1), 1) is self.TABLE.link(1)

    def test_set_target_weight_keeps_proportions(self) -> None:
        link = Link(1, {1: 0.5, 2: 0.3, 3: 0.2})
        new = set_target_weight(link, 1, 0.8)
        assert new.targets[1] == pytest.approx(0.8)
        assert new.targets[2] == pytest.approx(0.12)
        assert new.targets[3] == pytest.approx(0.08)
        assert set_target_weight(Link(1, {1: 1.0}), 1, 0.3).targets == {1: 1.0}

    def test_lock_and_revert(self) -> None:
        locked = set_locked(self.TABLE, 3, True)
        assert locked.link(3).locked
        assert locked.has_manual_links
        assert not self.TABLE.has_manual_links
        assert revert_to_auto(locked, self.TABLE, 3) == self.TABLE

    def test_is_compatible(self) -> None:
        assert self.TABLE.is_compatible((1, 2, 3), (1, 2, 3))
        assert not self.TABLE.is_compatible((1, 2, 3), (1, 2))
        assert not self.TABLE.is_compatible((1, 2), (1, 2, 3))


# ── History ──────────────────────────────────────────────────────────────────


class _Clock:
    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t


def _state(mix: float, tau: float = 1.0) -> EditState:
    return EditState(None, GradingParams(mix=mix), tau)


class TestHistory:
    def test_undo_redo(self) -> None:
        h = History(_state(1.0))
        assert h.push(_state(0.5))
        assert h.push(_state(0.2))
        assert h.undo() == _state(0.5)
        assert h.undo() == _state(1.0)
        assert h.undo() is None
        assert h.redo() == _state(0.5)
        h.push(_state(0.9))  # a new edit drops the redo branch
        assert not h.can_redo

    def test_equal_state_is_ignored(self) -> None:
        h = History(_state(1.0))
        assert not h.push(_state(1.0))
        assert not h.can_undo

    def test_merge_window(self) -> None:
        clock = _Clock()
        h = History(_state(1.0), clock=clock)
        for mix in (0.9, 0.8, 0.7):
            clock.t += 0.1
            h.push(_state(mix), "params")
        assert h.undo() == _state(1.0)  # one gesture = one undo step
        h.redo()
        clock.t += MERGE_WINDOW_S + 0.5
        h.push(_state(0.3), "params")
        assert h.undo() == _state(0.7)  # a pause starts a new step

    def test_different_keys_are_not_merged(self) -> None:
        h = History(_state(1.0), clock=_Clock())
        h.push(_state(0.5), "params")
        h.push(_state(0.5, tau=2.0), "tau")
        assert h.undo() == _state(0.5)

    def test_limit_and_reset(self) -> None:
        h = History(_state(0.0), limit=3)
        for i in range(1, 6):
            h.push(_state(i / 10))
        count = 0
        while h.undo() is not None:
            count += 1
        assert count == 3
        h.reset(_state(0.42))
        assert h.current == _state(0.42) and not h.can_undo and not h.can_redo


# ── Overlay and layout ───────────────────────────────────────────────────────


class TestOverlay:
    def test_highlight_dims_outside_only(self) -> None:
        img = np.full((100, 200, 3), 200, dtype=np.uint8)
        mask = np.zeros((25, 50), dtype=bool)
        mask[:, :25] = True  # left half
        out = highlight_zone(img, mask)
        assert out.shape == img.shape
        assert np.all(out[50, 50] == 200)  # inside, away from the outline
        assert out[50, 150, 0] == pytest.approx(200 * DIM_FACTOR, abs=1)
        assert not np.shares_memory(out, img)


def test_column_heights() -> None:
    heights = column_heights([0.5, 0.3, 0.2], 300)
    assert sum(heights) + 2 * 4 == pytest.approx(300, abs=1)
    assert heights[0] > heights[1] > heights[2] >= MIN_SWATCH_H
    tiny = column_heights([0.9, 0.1], 10)
    assert tiny == [MIN_SWATCH_H, MIN_SWATCH_H]


# ── CorrespondenceView ───────────────────────────────────────────────────────


def _mouse(
    view: CorrespondenceView,
    kind: QEvent.Type,
    pos: QPointF,
    modifiers: Qt.KeyboardModifier = Qt.KeyboardModifier.NoModifier,
) -> None:
    buttons = (
        Qt.MouseButton.NoButton
        if kind == QEvent.Type.MouseButtonRelease
        else Qt.MouseButton.LeftButton
    )
    event = QMouseEvent(
        kind, pos, view.mapToGlobal(pos), Qt.MouseButton.LeftButton, buttons, modifiers
    )
    QApplication.sendEvent(view, event)


@pytest.fixture()
def view(
    qapp: QApplication, palettes: tuple[ZonePalette, ZonePalette, CorrespondenceTable]
) -> CorrespondenceView:
    src, ref, table = palettes
    v = CorrespondenceView()
    v.resize(600, 400)
    v.set_data(src, ref, table, table)
    return v


def _drag(
    view: CorrespondenceView,
    source_zone: int,
    reference_zone: int,
    modifiers: Qt.KeyboardModifier = Qt.KeyboardModifier.NoModifier,
) -> None:
    src_rects, ref_rects = view.swatch_rects()
    start = src_rects[source_zone].center()
    end = ref_rects[reference_zone].center()
    _mouse(view, QEvent.Type.MouseButtonPress, start)
    _mouse(view, QEvent.Type.MouseMove, (start + end) / 2)
    _mouse(view, QEvent.Type.MouseMove, end)
    _mouse(view, QEvent.Type.MouseButtonRelease, end, modifiers)


class TestCorrespondenceView:
    def test_hit_testing(self, view: CorrespondenceView) -> None:
        src_rects, ref_rects = view.swatch_rects()
        assert len(src_rects) == 3 and len(ref_rects) == 3
        assert view.zone_at(src_rects[2].center()) == (SOURCE, 2)
        assert view.zone_at(ref_rects[3].center()) == (REFERENCE, 3)
        assert view.zone_at(QPointF(300, 5)) is None

    def test_drag_replaces_target(self, view: CorrespondenceView) -> None:
        emitted: list[CorrespondenceTable] = []
        view.table_changed.connect(emitted.append)
        selected: list[int] = []
        view.zone_selected.connect(selected.append)
        current = view.table
        assert current is not None
        new_target = next(r for r in (1, 2, 3) if r not in current.link(1).targets)
        _drag(view, 1, new_target)
        assert len(emitted) == 1
        assert emitted[0].link(1).targets == {new_target: 1.0}
        assert selected == [1]

    def test_shift_drag_adds_target(self, view: CorrespondenceView) -> None:
        current = view.table
        assert current is not None
        new_target = next(r for r in (1, 2, 3) if r not in current.link(2).targets)
        emitted: list[CorrespondenceTable] = []
        view.table_changed.connect(emitted.append)
        _drag(view, 2, new_target, Qt.KeyboardModifier.ShiftModifier)
        assert set(emitted[0].link(2).targets) == set(current.link(2).targets) | {new_target}

    def test_click_selects_without_editing(self, view: CorrespondenceView) -> None:
        emitted: list[Any] = []
        view.table_changed.connect(emitted.append)
        center = view.swatch_rects()[0][3].center()
        _mouse(view, QEvent.Type.MouseButtonPress, center)
        _mouse(view, QEvent.Type.MouseButtonRelease, center)
        assert view.selected_zone == 3
        assert emitted == []

    def test_hover_signal(self, view: CorrespondenceView) -> None:
        hovered: list[tuple[str, int]] = []
        view.zone_hovered.connect(lambda side, z: hovered.append((side, z)))
        ref_center = view.swatch_rects()[1][2].center()
        move = QMouseEvent(
            QEvent.Type.MouseMove,
            ref_center,
            view.mapToGlobal(ref_center),
            Qt.MouseButton.NoButton,
            Qt.MouseButton.NoButton,
            Qt.KeyboardModifier.NoModifier,
        )
        QApplication.sendEvent(view, move)
        QApplication.sendEvent(view, QEvent(QEvent.Type.Leave))
        assert hovered == [(REFERENCE, 2), ("", 0)]

    def test_context_menu_actions(self, view: CorrespondenceView) -> None:
        emitted: list[CorrespondenceTable] = []
        view.table_changed.connect(emitted.append)
        menu = view.build_context_menu(1)
        actions = {a.text(): a for a in menu.actions()}
        assert not actions["Revenir à l'auto"].isEnabled()  # still automatic
        actions["Verrouiller (garder telle quelle)"].trigger()
        assert emitted[-1].link(1).locked

        menu = view.build_context_menu(1)
        actions = {a.text(): a for a in menu.actions()}
        assert actions["Revenir à l'auto"].isEnabled()
        actions["Revenir à l'auto"].trigger()
        assert not emitted[-1].link(1).locked

    def test_conflicts_and_clear(self, view: CorrespondenceView) -> None:
        from stealzalut.core.zone_transfer import Conflict

        view.set_conflicts([Conflict(1, 2, 0.6, 0.1)])
        assert view.conflict_zones == {1, 2}
        view.grab()  # paints without error
        view.clear("vide")
        assert view.table is None and view.swatch_rects() == ({}, {})
        view.grab()


# ── ZoneInspector ────────────────────────────────────────────────────────────


class TestZoneInspector:
    def _inspector(
        self, palettes: tuple[ZonePalette, ZonePalette, CorrespondenceTable], link: Link
    ) -> tuple[ZoneInspector, list[tuple[Link, str]]]:
        src, ref, table = palettes
        inspector = ZoneInspector()
        emitted: list[tuple[Link, str]] = []
        inspector.link_changed.connect(lambda link, key: emitted.append((link, key)))
        inspector.set_zone(link, src.zone(link.source_zone), ref, table.link(link.source_zone))
        return inspector, emitted

    def test_empty_state(self, qapp: QApplication) -> None:
        inspector = ZoneInspector()
        assert inspector.link is None
        assert inspector._body.isHidden()

    def test_strength_and_intent(
        self,
        qapp: QApplication,
        palettes: tuple[ZonePalette, ZonePalette, CorrespondenceTable],
    ) -> None:
        inspector, emitted = self._inspector(palettes, palettes[2].link(1))
        assert emitted == []  # showing a link emits nothing
        inspector._strength_sliders["strength"].setValue(40)
        link, key = emitted[-1]
        assert link.strength == pytest.approx(0.4) and link.origin == "user"
        assert key == "strength:1"
        inspector._intent_sliders["warmth"].setValue(2)
        assert emitted[-1][0].intent == Intent(warmth="+medium")
        assert emitted[-1][0].strength == pytest.approx(0.4)  # earlier edit kept

    def test_target_weights(
        self,
        qapp: QApplication,
        palettes: tuple[ZonePalette, ZonePalette, CorrespondenceTable],
    ) -> None:
        inspector, emitted = self._inspector(palettes, Link(2, {1: 0.5, 3: 0.5}))
        inspector._target_rows[1][0].setValue(80)
        targets = emitted[-1][0].targets
        assert targets[1] == pytest.approx(0.8) and targets[3] == pytest.approx(0.2)
        inspector._target_rows[3][2].click()  # remove R3
        assert emitted[-1][0].targets == {1: pytest.approx(1.0)}

        # Showing the new link rebuilds the rows: the removed target is gone at once
        src, ref, table = palettes
        inspector.set_zone(emitted[-1][0], src.zone(2), ref, table.link(2))
        assert set(inspector._target_rows) == {1}
        assert inspector._targets_box.count() == 1

    def test_lock_and_revert(
        self,
        qapp: QApplication,
        palettes: tuple[ZonePalette, ZonePalette, CorrespondenceTable],
    ) -> None:
        auto = palettes[2].link(3)
        inspector, emitted = self._inspector(palettes, auto)
        assert not inspector._revert.isEnabled()
        inspector._lock.setChecked(True)
        assert emitted[-1][0].locked
        src, ref, _ = palettes
        inspector.set_zone(emitted[-1][0], src.zone(3), ref, auto)
        assert inspector._revert.isEnabled()
        assert not inspector._strength_sliders["strength"].isEnabled()
        inspector._revert.click()
        assert emitted[-1][0] == auto


# ── Main window integration ──────────────────────────────────────────────────


@pytest.fixture()
def image_paths(tmp_path: Path) -> tuple[Path, Path]:
    src, ref = tmp_path / "src.png", tmp_path / "ref.png"
    cv2.imwrite(str(src), cv2.cvtColor(SOURCE_IMG, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(ref), cv2.cvtColor(REFERENCE_IMG, cv2.COLOR_RGB2BGR))
    return src, ref


def _wait(qapp: QApplication, condition: Callable[[], bool], timeout: float = 20.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline and not condition():
        qapp.processEvents()
        time.sleep(0.01)
    assert condition()


def _zones_settings() -> Settings:
    return Settings(
        lut_output_dir=Path.home() / "Documents" / "LUTs",
        lut_default_size=17,
        color_transfer_method="zones",
        app_theme="dark",
        log_level="WARNING",
        log_file=None,
    )


class TestMainWindowZones:
    def _window(self, qapp: QApplication, image_paths: tuple[Path, Path]) -> Any:
        from stealzalut.gui.main_window import MainWindow

        window = MainWindow(settings=_zones_settings())
        window._zone_count_spin.setValue(3)
        window._source_zone.load_from_path(image_paths[0])
        window._reference_zone.load_from_path(image_paths[1])
        _wait(
            qapp,
            lambda: window._zone_model is not None
            and window._preview.image_rgb is not None
            and not (window._transfer_worker and window._transfer_worker.isRunning()),
        )
        return window

    def test_editor_shows_zones_and_undo_works(
        self, qapp: QApplication, image_paths: tuple[Path, Path]
    ) -> None:
        window = self._window(qapp, image_paths)
        try:
            assert not window._corr_group.isHidden()
            assert len(window._zone_model.source.zones) == 3
            assert not window._undo_btn.isEnabled()

            auto = window._history.current.table
            new_target = next(r for r in (1, 2, 3) if r not in auto.link(1).targets)
            window._corr_view._emit_table(retarget(auto, 1, new_target))
            assert window._history.current.table.link(1).targets == {new_target: 1.0}
            assert window._undo_btn.isEnabled()

            # The edit reaches the preview (debounced re-bake)
            _wait(qapp, lambda: window._transfer_worker is not None)
            effective = window._effective_model(window._zone_model)
            assert effective.table.link(1).targets == {new_target: 1.0}

            window.undo()
            assert window._history.current.table == auto
            assert window._corr_view.table == auto
            assert window._redo_btn.isEnabled()
            window.redo()
            assert window._history.current.table.link(1).targets == {new_target: 1.0}
        finally:
            window.stop_workers()

    def test_hover_highlights_images(
        self, qapp: QApplication, image_paths: tuple[Path, Path]
    ) -> None:
        window = self._window(qapp, image_paths)
        try:
            calls: dict[str, Any] = {}
            window._source_zone.set_highlight = lambda m: calls.__setitem__("src", m)
            window._reference_zone.set_highlight = lambda m: calls.__setitem__("ref", m)
            window._preview.set_highlight = lambda m: calls.__setitem__("preview", m)
            window._on_zone_hovered(SOURCE, 2)
            assert calls["src"] is not None and calls["src"].any()
            assert calls["preview"] is calls["src"]
            assert calls["ref"] is None
            window._on_zone_hovered("", 0)
            assert calls["src"] is None and calls["ref"] is None
        finally:
            window.stop_workers()

    def test_zone_count_change_asks_when_edited(
        self,
        qapp: QApplication,
        image_paths: tuple[Path, Path],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        window = self._window(qapp, image_paths)
        try:
            table = window._history.current.table
            window._corr_view._emit_table(set_locked(table, 1, True))
            asked: list[str] = []

            def refuse(*args: Any, **kwargs: Any) -> QMessageBox.StandardButton:
                asked.append(args[1])
                return QMessageBox.StandardButton.No

            monkeypatch.setattr(QMessageBox, "question", refuse)
            window._zone_count_spin.setValue(4)
            assert asked == ["Correspondances modifiées"]
            assert window._zone_count_spin.value() == 3  # change refused
            assert window._history.current.table.link(1).locked  # edits kept
        finally:
            window.stop_workers()

    def test_panel_hidden_for_other_methods(
        self, qapp: QApplication, image_paths: tuple[Path, Path]
    ) -> None:
        window = self._window(qapp, image_paths)
        try:
            window._algo_combo.setCurrentIndex(window._algo_combo.findData("oklab"))
            assert window._corr_group.isHidden()
            assert window._zone_model is None
            assert window._history.current.table is None
        finally:
            window.stop_workers()


# ── Phase 3: recipes and AI pack in the window ───────────────────────────────


def _answer(analysis: str, target: int) -> str:
    recipe = {
        "format": "stealzalut.recipe",
        "version": 2,
        "analysis_id": analysis,
        "name": "Essai",
        "global": {"mix": 0.8},
        "links": [{"source_zone": 1, "targets": {str(target): 1}, "rationale": "Feuillage"}],
    }
    return "Analyse : S1 est le feuillage.\n```json\n" + json.dumps(recipe) + "\n```"


class TestRecipesInWindow:
    _window = TestMainWindowZones._window

    def test_buttons_need_zones(self, qapp: QApplication) -> None:
        from stealzalut.gui.main_window import MainWindow

        window = MainWindow(settings=_zones_settings())
        assert not window._ai_pack_btn.isEnabled()
        assert not window._import_btn.isEnabled()
        assert not window._export_recipe_btn.isEnabled()

    def test_apply_ai_answer_is_one_undo_step(
        self, qapp: QApplication, image_paths: tuple[Path, Path]
    ) -> None:
        from stealzalut.core.recipe import parse_recipe

        window = self._window(qapp, image_paths)
        try:
            assert window._ai_pack_btn.isEnabled()
            auto = window._history.current.table
            target = next(r for r in (1, 2, 3) if r not in auto.link(1).targets)
            answer = _answer(window._zone_model.analysis_id, target)
            assert window.apply_recipe(parse_recipe(answer))

            link = window._history.current.table.link(1)
            assert link.targets == {target: 1.0}
            assert link.origin == "ai" and link.rationale == "Feuillage"
            assert window._adjustments.params().mix == pytest.approx(0.8)
            assert window._corr_view.table == window._history.current.table

            window.undo()
            assert window._history.current.table == auto
            assert window._adjustments.params().mix == pytest.approx(1.0)
        finally:
            window.stop_workers()

    def test_recipe_of_other_analysis_asks(
        self,
        qapp: QApplication,
        image_paths: tuple[Path, Path],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from stealzalut.core.recipe import parse_recipe

        window = self._window(qapp, image_paths)
        try:
            auto = window._history.current.table
            target = next(r for r in (1, 2, 3) if r not in auto.link(1).targets)
            recipe = parse_recipe(_answer("00000000", target))

            monkeypatch.setattr(window, "_ask_other_analysis", lambda _r: "cancel")
            assert not window.apply_recipe(recipe)
            assert window._history.current.table == auto

            monkeypatch.setattr(window, "_ask_other_analysis", lambda _r: "apply")
            assert window.apply_recipe(recipe)
            assert window._history.current.table.link(1).targets == {target: 1.0}
        finally:
            window.stop_workers()

    def test_unknown_zone_is_reported(
        self,
        qapp: QApplication,
        image_paths: tuple[Path, Path],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from stealzalut.core.recipe import parse_recipe

        window = self._window(qapp, image_paths)
        try:
            errors: list[str] = []
            monkeypatch.setattr(QMessageBox, "critical", lambda *a, **k: errors.append(a[2]))
            assert not window.apply_recipe(parse_recipe(_answer(window._zone_model.analysis_id, 9)))
            assert "R9 n'existe pas" in errors[0]
        finally:
            window.stop_workers()

    def test_export_then_import_restores_work(
        self,
        qapp: QApplication,
        image_paths: tuple[Path, Path],
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from PySide6.QtWidgets import QFileDialog

        window = self._window(qapp, image_paths)
        try:
            edited = set_locked(window._history.current.table, 2, True)
            window._corr_view._emit_table(edited)
            path = tmp_path / "mon look.szl"
            monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *a, **k: (str(path), ""))
            window._on_export_recipe()
            assert path.exists()

            window._on_all_auto()
            assert not window._history.current.table.link(2).locked

            monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *a, **k: (str(path), ""))
            window._on_open_recipe()
            assert window._history.current.table == edited
        finally:
            window.stop_workers()

    def test_prepare_pack(
        self,
        qapp: QApplication,
        image_paths: tuple[Path, Path],
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from PySide6.QtGui import QGuiApplication
        from PySide6.QtWidgets import QFileDialog

        window = self._window(qapp, image_paths)
        try:
            monkeypatch.setattr(QFileDialog, "getExistingDirectory", lambda *a, **k: str(tmp_path))
            monkeypatch.setattr(QMessageBox, "exec", lambda self: 0)
            window._on_prepare_ai_pack()
            folder = tmp_path / "src_x_ref_stealzalut_pack"
            assert (folder / "PROMPT.md").exists() and (folder / "analysis.json").exists()
            assert window._zone_model.analysis_id in QGuiApplication.clipboard().text()
        finally:
            window.stop_workers()


class TestPasteDialog:
    def test_error_keeps_dialog_open(self, qapp: QApplication) -> None:
        from stealzalut.gui.widgets.paste_recipe_dialog import PasteRecipeDialog

        dialog = PasteRecipeDialog()
        dialog.set_text("Je ne sais pas.")
        dialog._on_accept()
        assert dialog.recipe is None
        assert dialog.result() != dialog.DialogCode.Accepted
        assert "Aucun bloc JSON" in dialog._error.text()

        dialog.set_text('```json\n{"format": "stealzalut.recipe", "version": 2, "links": []}\n```')
        dialog._on_accept()
        assert dialog.recipe is not None
        assert dialog.result() == dialog.DialogCode.Accepted
