"""Tests for the ambiance mode (monochrome references) and its wiring."""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest
from PySide6.QtWidgets import QApplication

from stealzalut.core.ambiance import AmbianceTransferModel, fit_ambiance, reference_tint_curve
from stealzalut.core.color_transfer import fit_transfer
from stealzalut.core.colorspace import oklab_to_oklch, oklab_to_srgb, oklch_to_oklab, srgb_to_oklab
from stealzalut.core.grading import GradingParams, adjust_displacement, build_grid
from stealzalut.core.palette import colour_stats
from stealzalut.core.recipe import RecipeError, parse_recipe, recipe_to_dict
from stealzalut.gui.zone_text import dominant_colour_name


def _lch_rgb(lightness: float, chroma: float, hue: float) -> np.ndarray:
    lab = oklch_to_oklab(np.array([[lightness, chroma, hue]]))
    return np.clip(oklab_to_srgb(lab)[0] * 255.0 + 0.5, 0, 255).astype(np.uint8)


def _purple_night() -> np.ndarray:
    """Purple sky over a nearly black ground, a small warm light (RGB)."""
    img = np.zeros((200, 160, 3), dtype=np.uint8)
    rng = np.random.default_rng(0)
    for y in range(140):
        img[y] = _lch_rgb(0.22 + 0.15 * y / 140, 0.15, 298.0)
    img[140:] = _lch_rgb(0.12, 0.02, 310.0)
    img[60:70, 70:90] = _lch_rgb(0.9, 0.05, 70.0)  # warm light source
    noise = rng.normal(0, 2, img.shape)
    return np.clip(img + noise, 0, 255).astype(np.uint8)


def _city() -> np.ndarray:
    """Grey buildings, a red and a green neon, a dark street (RGB)."""
    img = np.full((160, 240, 3), 70, dtype=np.uint8)
    img[:40] = 25
    img[60:90, 30:60] = (230, 30, 40)  # red neon
    img[60:90, 150:180] = (40, 200, 60)  # green neon
    img[120:] = 45
    return img


class TestColourStats:
    def test_monochrome_reference(self) -> None:
        stats = colour_stats(_purple_night())
        assert stats.is_monochrome and stats.concentration > 0.9
        assert dominant_colour_name(stats.hue) == "dominante violette"

    def test_varied_and_grey_images(self) -> None:
        varied = np.zeros((60, 240, 3), dtype=np.uint8)
        for i, color in enumerate([(220, 40, 40), (40, 200, 60), (40, 80, 230), (230, 200, 40)]):
            varied[:, i * 60 : (i + 1) * 60] = color
        assert not colour_stats(varied).is_monochrome
        grey = colour_stats(np.full((40, 40, 3), 128, dtype=np.uint8))
        assert grey.concentration == 0.0 and not grey.is_monochrome


class TestAmbianceModel:
    @pytest.fixture(scope="class")
    def model(self) -> AmbianceTransferModel:
        m = fit_transfer(_city(), _purple_night(), "ambiance")
        assert isinstance(m, AmbianceTransferModel)
        return m

    def test_tint_curve_follows_tone(self) -> None:
        curve_model = fit_ambiance(_city(), _purple_night())
        dark = oklab_to_oklch(np.array([[0.3, *curve_model.tint(np.array([0.3]))[0]]]))[0]
        assert dark[1] > 0.08 and abs(dark[2] - 298.0) < 15.0  # purple where the sky is
        assert reference_tint_curve(_purple_night()).shape == (64, 2)

    def test_greys_take_the_cast(self, model: AmbianceTransferModel) -> None:
        grey = np.array([[0.3, 0.3, 0.3]])
        out = oklab_to_oklch(srgb_to_oklab(model.apply(grey)))[0]
        assert out[1] > 0.05 and abs(out[2] - 298.0) < 20.0

    def test_vivid_colours_keep_their_hue(self, model: AmbianceTransferModel) -> None:
        red = np.array([[0.9, 0.12, 0.15]])
        before = oklab_to_oklch(srgb_to_oklab(red))[0]
        after = oklab_to_oklch(srgb_to_oklab(model.apply(red)))[0]
        assert abs((after[2] - before[2] + 180) % 360 - 180) < 12.0
        assert after[1] > 0.6 * before[1]

    def test_lightness_follows_reference_tone(self, model: AmbianceTransferModel) -> None:
        colors = np.random.default_rng(1).random((200, 3))
        out = srgb_to_oklab(model.apply(colors))[:, 0]
        expected = srgb_to_oklab(model.global_model.apply(colors))[:, 0]
        np.testing.assert_allclose(out, expected, atol=0.02)  # gamut compression aside

    def test_grid_has_no_folds(self, model: AmbianceTransferModel) -> None:
        result = build_grid(model, 33)
        assert result.fold_fraction < 0.001 and not result.auto_smoothed


def test_preserve_vivid_slider() -> None:
    lab_in = srgb_to_oklab(np.array([[0.9, 0.12, 0.15], [0.4, 0.4, 0.4]]))
    target = lab_in + np.array([0.0, 0.05, -0.08])
    full = adjust_displacement(lab_in, target, GradingParams())
    kept = adjust_displacement(lab_in, target, GradingParams(preserve_vivid=1.0))
    assert np.linalg.norm(kept[0, 1:]) < 0.1 * np.linalg.norm(full[0, 1:])  # vivid red kept
    np.testing.assert_allclose(kept[1], full[1])  # grey unaffected


class TestRecipeMode:
    def test_mode_parsing(self) -> None:
        assert parse_recipe({"mode": "ambiance"}).mode == "ambiance"
        assert parse_recipe({}).mode is None
        with pytest.raises(RecipeError, match="mode"):
            parse_recipe({"mode": "filtre"})

    def test_export_states_zones(self) -> None:
        from stealzalut.core.correspondence import CorrespondenceTable

        data = recipe_to_dict(
            CorrespondenceTable.from_matching({1: {1: 1.0}}), GradingParams(), 1.0, "x", 3
        )
        assert data["mode"] == "zones"
        assert parse_recipe(data).mode == "zones"


def test_analysis_and_prompt_suggest_ambiance() -> None:
    from stealzalut.core.ai_pack import analysis_dict, build_prompt
    from stealzalut.core.zone_transfer import fit_zone_transfer

    model = fit_zone_transfer(_city(), _purple_night(), 4)
    data = analysis_dict(model, (240, 160), (160, 200), 4)
    assert data["suggested_mode"] == "ambiance"
    assert data["reference"]["monochrome"] is True
    assert data["reference"]["hue_concentration"] > 0.9
    prompt = build_prompt(model)
    assert 'le mode "ambiance" est presque toujours le bon choix' in prompt
    assert '"mode": "zones|ambiance"' in prompt


# ── Window ───────────────────────────────────────────────────────────────────


@pytest.fixture(scope="session")
def qapp() -> QApplication:
    app = QApplication.instance()
    if not isinstance(app, QApplication):
        app = QApplication(sys.argv)
    return app


def _write(path: Path, rgb: np.ndarray) -> Path:
    cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    return path


def _window(qapp: QApplication, tmp_path: Path) -> Any:
    from stealzalut.config import Settings
    from stealzalut.gui.main_window import MainWindow

    settings = Settings(
        lut_output_dir=tmp_path, lut_default_size=17, color_transfer_method="zones",
        log_level="WARNING", log_file=None,
    )
    window = MainWindow(settings=settings)
    window._zone_count_spin.setValue(4)
    window._source_zone.load_from_path(_write(tmp_path / "city.png", _city()))
    window._reference_zone.load_from_path(_write(tmp_path / "purple.png", _purple_night()))
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline and not (
        window._zone_model is not None
        and not (window._transfer_worker and window._transfer_worker.isRunning())
    ):
        qapp.processEvents()
        time.sleep(0.01)
    return window


class TestWindowAmbiance:
    def test_hint_suggests_and_switches(self, qapp: QApplication, tmp_path: Path) -> None:
        window = _window(qapp, tmp_path)
        try:
            assert not window._mode_hint.isHidden()
            assert "dominante violette" in window._mode_hint_label.text()
            window._mode_hint_btn.click()
            assert window._current_method == "ambiance"
            assert window._mode_hint.isHidden()
            assert window._corr_group.isHidden()
        finally:
            window.stop_workers()

    def test_hint_can_be_dismissed(self, qapp: QApplication, tmp_path: Path) -> None:
        window = _window(qapp, tmp_path)
        try:
            window._dismiss_mode_hint()
            assert window._mode_hint.isHidden()
        finally:
            window.stop_workers()

    def test_ambiance_recipe_switches_mode(self, qapp: QApplication, tmp_path: Path) -> None:
        window = _window(qapp, tmp_path)
        try:
            recipe = parse_recipe(
                {
                    "mode": "ambiance",
                    "name": "Nuit violette",
                    "global": {"chroma_strength": 0.8, "preserve_vivid": 0.5},
                    "links": [],
                    "finishing": {"vibrance": 0.3},
                }
            )
            assert window.apply_recipe(recipe)
            assert window._current_method == "ambiance"
            state = window._history.current
            assert state.table is None
            assert state.params.chroma_strength == pytest.approx(0.8)
            assert state.params.preserve_vivid == pytest.approx(0.5)
            assert state.finishing.vibrance == pytest.approx(0.3)
            assert "mode Ambiance" in window._status_note
        finally:
            window.stop_workers()
