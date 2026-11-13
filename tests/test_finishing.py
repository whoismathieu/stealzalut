"""Tests for stealzalut.core.finishing and its use in grids, recipes and the GUI."""

from __future__ import annotations

import sys
from dataclasses import replace

import numpy as np
import pytest
from PySide6.QtCore import QEvent, QPointF, Qt
from PySide6.QtGui import QMouseEvent
from PySide6.QtWidgets import QApplication

from stealzalut.core.color_transfer import fit_transfer
from stealzalut.core.colorspace import oklab_to_oklch, oklch_to_oklab, srgb_to_oklab
from stealzalut.core.correspondence import CorrespondenceTable
from stealzalut.core.finishing import (
    BANDS,
    IDENTITY_CURVE,
    Finishing,
    HueBand,
    Wheel,
    apply_finishing,
    eval_curve,
    normalize_curve,
    with_band,
)
from stealzalut.core.grading import GradingParams, build_grid
from stealzalut.core.lut_engine import generate_neutral_grid
from stealzalut.core.recipe import (
    RecipeError,
    apply_recipe,
    dump_recipe,
    parse_recipe,
    recipe_to_dict,
)

GRID = srgb_to_oklab(generate_neutral_grid(17))


def _lch(lightness: float, chroma: float, hue: float) -> np.ndarray:
    return oklch_to_oklab(np.array([[lightness, chroma, hue]]))


# ── Core ─────────────────────────────────────────────────────────────────────


class TestCurve:
    def test_identity_and_endpoints(self) -> None:
        x = np.linspace(0, 1, 21)
        np.testing.assert_allclose(eval_curve(IDENTITY_CURVE, x), x)

    def test_passes_through_points_and_is_monotone(self) -> None:
        points = ((0.0, 0.0), (0.25, 0.1), (0.5, 0.55), (0.75, 0.9), (1.0, 1.0))
        x = np.linspace(0, 1, 501)
        y = eval_curve(points, x)
        assert np.all(np.diff(y) >= -1e-12)  # monotone
        assert y.min() >= 0.0 and y.max() <= 1.0  # no overshoot
        for px, py in points:
            assert float(eval_curve(points, np.array([px]))[0]) == pytest.approx(py)

    def test_flat_segment_stays_flat(self) -> None:
        points = ((0.0, 0.2), (0.4, 0.2), (1.0, 1.0))
        y = eval_curve(points, np.linspace(0, 0.4, 20))
        np.testing.assert_allclose(y, 0.2, atol=1e-12)

    def test_normalize(self) -> None:
        assert normalize_curve(((0.6, 0.4), (0.3, 0.5))) == (
            (0.0, 0.5),
            (0.3, 0.5),
            (0.6, 0.5),  # raised: the curve can't go down
            (1.0, 0.5),
        )
        assert normalize_curve(((-1.0, 0.0), (2.0, 3.0))) == ((0.0, 0.0), (1.0, 1.0))


class TestApply:
    def test_identity(self) -> None:
        np.testing.assert_array_equal(apply_finishing(GRID, Finishing()), GRID)

    def test_tone_curve_keeps_hue_and_chroma(self) -> None:
        f = Finishing(tone_curve=((0.0, 0.0), (0.5, 0.65), (1.0, 1.0)))
        before, after = oklab_to_oklch(GRID), oklab_to_oklch(apply_finishing(GRID, f))
        np.testing.assert_allclose(after[:, 1], before[:, 1], atol=1e-9)
        coloured = before[:, 1] > 0.02
        np.testing.assert_allclose(after[coloured, 2], before[coloured, 2], atol=1e-6)
        assert np.all(after[:, 0] >= before[:, 0] - 1e-9)  # brightened

    def test_wheels_target_their_range(self) -> None:
        dark, bright = _lch(0.15, 0.0, 0.0), _lch(0.9, 0.0, 0.0)
        f = Finishing(shadows=Wheel(hue=250.0, amount=1.0, lightness=0.5))
        d_dark = apply_finishing(dark, f) - dark
        d_bright = apply_finishing(bright, f) - bright
        assert d_dark[0, 0] > 5 * d_bright[0, 0] > 0
        assert np.linalg.norm(d_dark[0, 1:]) > 5 * np.linalg.norm(d_bright[0, 1:])
        assert d_dark[0, 2] < 0  # hue 250° ≈ blue: b decreases

    def test_band_affects_its_colours_only(self) -> None:
        green_hue = BANDS[3][1]
        f = with_band(Finishing(), HueBand(center=green_hue, saturation=-0.8))
        green, red = _lch(0.6, 0.12, green_hue), _lch(0.6, 0.12, BANDS[0][1])
        grey = _lch(0.6, 0.0, 0.0)
        assert oklab_to_oklch(apply_finishing(green, f))[0, 1] < 0.05
        assert oklab_to_oklch(apply_finishing(red, f))[0, 1] == pytest.approx(0.12, abs=1e-6)
        np.testing.assert_allclose(apply_finishing(grey, f), grey, atol=1e-12)

    def test_band_hue_rotation(self) -> None:
        center = BANDS[5][1]
        f = with_band(Finishing(), HueBand(center=center, hue=0.5))
        out = oklab_to_oklch(apply_finishing(_lch(0.5, 0.15, center), f))
        assert out[0, 2] == pytest.approx(center + 15.0, abs=0.1)

    def test_saturation_and_vibrance(self) -> None:
        dull, vivid = _lch(0.6, 0.03, 120.0), _lch(0.6, 0.2, 120.0)
        sat = Finishing(saturation=0.5)
        assert oklab_to_oklch(apply_finishing(dull, sat))[0, 1] == pytest.approx(0.045)
        vib = Finishing(vibrance=1.0)
        gain_dull = oklab_to_oklch(apply_finishing(dull, vib))[0, 1] / 0.03
        gain_vivid = oklab_to_oklch(apply_finishing(vivid, vib))[0, 1] / 0.2
        assert gain_dull > 1.5 and gain_vivid == pytest.approx(1.0)

    def test_with_band_replaces_by_centre(self) -> None:
        f = with_band(Finishing(), HueBand(center=BANDS[2][1], lightness=0.3))
        assert len(f.bands) == len(BANDS)
        assert not f.is_identity


# ── Pipeline and recipes ─────────────────────────────────────────────────────


def _img(color: tuple[int, int, int], seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    base = np.full((40, 40, 3), color, dtype=np.float64) + rng.normal(0, 25, (40, 40, 3))
    return np.clip(base, 0, 255).astype(np.uint8)


class TestPipeline:
    def test_identity_finishing_keeps_raw_bake(self) -> None:
        m = fit_transfer(_img((180, 100, 80), 1), _img((70, 120, 200), 2), "oklab")
        result = build_grid(m, 17, GradingParams(), Finishing())
        np.testing.assert_allclose(result.grid, m.bake(17), atol=1e-5)

    def test_finishing_is_applied_last(self) -> None:
        m = fit_transfer(_img((180, 100, 80), 1), _img((70, 120, 200), 2), "oklab")
        f = Finishing(saturation=-1.0)  # fully desaturated output
        grid = build_grid(m, 17, GradingParams(), f).grid
        assert np.all(np.isfinite(grid)) and grid.min() >= 0.0 and grid.max() <= 1.0
        chroma = oklab_to_oklch(srgb_to_oklab(grid))[:, 1]
        assert chroma.max() < 1e-3


class TestRecipeFinishing:
    AUTO = CorrespondenceTable.from_matching({1: {1: 1.0}})

    def test_round_trip(self) -> None:
        f = Finishing(
            shadows=Wheel(hue=250.0, amount=0.4, lightness=-0.2),
            tone_curve=((0.0, 0.05), (0.5, 0.55), (1.0, 0.95)),
            saturation=0.1,
            vibrance=-0.3,
        )
        f = with_band(f, HueBand(center=BANDS[3][1], hue=-0.2, saturation=-0.5))
        text = dump_recipe(recipe_to_dict(self.AUTO, GradingParams(), 1.0, "x", 3, finishing=f))
        recipe = parse_recipe(text)
        assert recipe.finishing == f
        assert apply_recipe(recipe, self.AUTO, (1,)) == self.AUTO

    def test_null_and_identity(self) -> None:
        assert parse_recipe({"finishing": None}).finishing is None
        data = recipe_to_dict(self.AUTO, GradingParams(), 1.0, "x", 3, finishing=Finishing())
        assert data["finishing"] is None

    @pytest.mark.parametrize(
        ("finishing", "message"),
        [
            ({"saturation": 3}, "finishing.saturation"),
            ({"shadows": {"amount": -1}}, "finishing.shadows.amount"),
            ({"tone_curve": [[0, 0]]}, "au moins 2 points"),
            ({"tone_curve": [[0, 0], [1]]}, "point n°2"),
            ({"bands": [{"hue": 0.5}]}, "center"),
            ([], "objet"),
        ],
    )
    def test_errors(self, finishing: object, message: str) -> None:
        with pytest.raises(RecipeError, match=message):
            parse_recipe({"finishing": finishing})


# ── GUI ──────────────────────────────────────────────────────────────────────


@pytest.fixture(scope="session")
def qapp() -> QApplication:
    app = QApplication.instance()
    if not isinstance(app, QApplication):
        app = QApplication(sys.argv)
    return app


def _press_move(widget: object, start: QPointF, end: QPointF) -> None:
    from PySide6.QtWidgets import QWidget

    assert isinstance(widget, QWidget)
    for kind, pos, buttons in (
        (QEvent.Type.MouseButtonPress, start, Qt.MouseButton.LeftButton),
        (QEvent.Type.MouseMove, end, Qt.MouseButton.LeftButton),
        (QEvent.Type.MouseButtonRelease, end, Qt.MouseButton.NoButton),
    ):
        event = QMouseEvent(
            kind, pos, widget.mapToGlobal(pos), Qt.MouseButton.LeftButton, buttons,
            Qt.KeyboardModifier.NoModifier,
        )
        QApplication.sendEvent(widget, event)


class TestFinishingWidgets:
    def test_wheel_mapping(self, qapp: QApplication) -> None:
        from stealzalut.gui.widgets.finishing_panel import ColorWheel

        wheel = ColorWheel()
        hue, amount = wheel.tint_at(wheel.point_for(250.0, 0.6))
        assert hue == pytest.approx(250.0, abs=0.5) and amount == pytest.approx(0.6, abs=0.01)
        center = QPointF(wheel.width() / 2, wheel.height() / 2)
        assert wheel.tint_at(center)[1] == 0.0  # snaps to no tint

    def test_wheel_drag_emits(self, qapp: QApplication) -> None:
        from stealzalut.gui.widgets.finishing_panel import ColorWheel

        wheel = ColorWheel()
        emitted: list[tuple[float, float]] = []
        wheel.tint_changed.connect(lambda h, a: emitted.append((h, a)))
        center = QPointF(wheel.width() / 2, wheel.height() / 2)
        _press_move(wheel, center + QPointF(10, 0), wheel.point_for(90.0, 0.8))
        assert emitted[-1][0] == pytest.approx(90.0, abs=1.0)

    def test_curve_editor_drag_stays_monotone(self, qapp: QApplication) -> None:
        from stealzalut.gui.widgets.finishing_panel import ToneCurveEditor

        editor = ToneCurveEditor()
        emitted: list[tuple[tuple[float, float], ...]] = []
        editor.curve_changed.connect(emitted.append)
        start = editor.to_widget(0.5, 0.5)  # empty spot on the diagonal: adds a point
        _press_move(editor, start, editor.to_widget(0.5, 0.8))
        curve = emitted[-1]
        assert len(curve) == 3 and curve[1][1] == pytest.approx(0.8, abs=0.02)
        # Dragging the end point below the middle one is clamped (curve stays monotone)
        _press_move(editor, editor.to_widget(1.0, 1.0), editor.to_widget(1.0, 0.1))
        assert emitted[-1][2][1] >= emitted[-1][1][1]

    def test_panel_round_trip(self, qapp: QApplication) -> None:
        from stealzalut.gui.widgets.finishing_panel import FinishingPanel

        panel = FinishingPanel()
        emitted: list[tuple[Finishing, str]] = []
        panel.finishing_changed.connect(lambda f, key: emitted.append((f, key)))
        f = with_band(
            Finishing(midtones=Wheel(hue=40.0, amount=0.3, lightness=0.25), vibrance=0.5),
            HueBand(center=BANDS[1][1], saturation=0.4),
        )
        panel.set_finishing(f)
        assert emitted == [] and panel.finishing() == f
        panel._saturation.setValue(-30)
        assert emitted[-1][0] == replace(f, saturation=-0.3)
        assert emitted[-1][1] == "saturation"
        panel._band_sliders[(1, "hue")].setValue(20)
        assert emitted[-1][1] == "band_1_hue"
        panel.reset()
        assert emitted[-1] == (Finishing(), "")


class TestFinishingInWindow:
    def test_edit_is_undoable(self, qapp: QApplication) -> None:
        from pathlib import Path

        from stealzalut.config import Settings
        from stealzalut.gui.main_window import MainWindow

        settings = Settings(lut_output_dir=Path.home(), log_level="WARNING", log_file=None)
        window = MainWindow(settings=settings)
        panel = window._finishing_panel
        panel._vibrance.setValue(40)
        assert window._history.current.finishing.vibrance == pytest.approx(0.4)
        assert window._undo_btn.isEnabled()

        window.undo()
        assert window._history.current.finishing == Finishing()
        assert panel.finishing() == Finishing()
        assert panel._vibrance.value() == 0

        window.redo()
        assert panel._vibrance.value() == 40
        window.stop_workers()


class TestPartialFinishing:
    USER = Finishing(
        shadows=Wheel(hue=200.0, amount=0.3),
        tone_curve=((0.0, 0.0), (0.5, 0.4), (1.0, 1.0)),
        saturation=-0.2,
    )

    def test_ai_gives_only_saturation_and_vibrance(self) -> None:
        from stealzalut.core.recipe import merge_finishing

        recipe = parse_recipe({"finishing": {"saturation": 0.15, "vibrance": 0.5}})
        assert recipe.finishing_keys == {"saturation", "vibrance"}
        merged = merge_finishing(recipe, self.USER)
        assert (merged.saturation, merged.vibrance) == (0.15, 0.5)
        assert merged.shadows == self.USER.shadows  # the user's wheels survive
        assert merged.tone_curve == self.USER.tone_curve

    def test_null_keeps_current_and_full_recipe_replaces(self) -> None:
        from stealzalut.core.recipe import merge_finishing

        assert merge_finishing(parse_recipe({"finishing": None}), self.USER) == self.USER
        exported = Finishing(vibrance=0.2)
        data = recipe_to_dict(
            CorrespondenceTable.from_matching({1: {1: 1.0}}),
            GradingParams(), 1.0, "x", 3, finishing=exported,
        )
        assert merge_finishing(parse_recipe(data), self.USER) == exported
