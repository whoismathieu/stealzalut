"""Tests for stealzalut.core.grading — strengths, protections, mix, smoothing."""

from __future__ import annotations

import numpy as np
import pytest

from stealzalut.core.color_transfer import SUPPORTED_METHODS, fit_transfer
from stealzalut.core.colorspace import oklab_to_oklch, srgb_to_oklab
from stealzalut.core.grading import (
    GradingParams,
    adjust_displacement,
    build_grid,
    neutral_mask,
    skin_mask,
)
from stealzalut.core.lut_engine import generate_neutral_grid

SKIN_TONES = [(224, 172, 140), (141, 85, 36), (255, 205, 190), (90, 56, 37), (232, 190, 172)]
NOT_SKIN = [(40, 160, 60), (40, 80, 200), (128, 128, 128), (220, 30, 30), (120, 40, 160)]


def _lch(colors: list[tuple[int, int, int]]) -> np.ndarray:
    return oklab_to_oklch(srgb_to_oklab(np.array(colors, dtype=np.float64) / 255.0))


def _image(mean: tuple[float, float, float], std: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.clip(rng.normal(mean, std, (64, 64, 3)), 0, 255).astype(np.uint8)


@pytest.fixture()
def model():  # type: ignore[no-untyped-def]
    return fit_transfer(_image((180, 100, 80), 40, 1), _image((70, 120, 200), 35, 2), "oklab")


class TestMasks:
    def test_skin_tones_are_protected(self) -> None:
        assert np.all(skin_mask(_lch(SKIN_TONES)) > 0.8)

    def test_other_colours_are_not_skin(self) -> None:
        assert np.all(skin_mask(_lch(NOT_SKIN)) == 0.0)

    def test_neutral_mask(self) -> None:
        greys = [(0, 0, 0), (60, 60, 60), (128, 128, 128), (250, 250, 250)]
        assert np.all(neutral_mask(_lch(greys)) == pytest.approx(1.0))
        assert np.all(neutral_mask(_lch([(40, 160, 60), (220, 30, 30)])) == 0.0)


class TestGradingParams:
    def test_defaults_are_passthrough(self) -> None:
        assert GradingParams().is_passthrough
        assert not GradingParams(mix=0.5).is_passthrough


class TestAdjustDisplacement:
    def test_strengths_scale_axes_separately(self) -> None:
        lab_in = srgb_to_oklab(generate_neutral_grid(17))
        target = lab_in + np.array([0.1, 0.02, -0.03])
        delta = adjust_displacement(
            lab_in, target, GradingParams(luminance_strength=0.5, chroma_strength=0.0)
        )
        np.testing.assert_allclose(delta[:, 0], 0.05)
        np.testing.assert_allclose(delta[:, 1:], 0.0)

    def test_preserve_neutrals_keeps_greys_grey(self) -> None:
        g = np.linspace(0.05, 0.95, 10)
        lab_in = srgb_to_oklab(np.stack([g, g, g], axis=1))
        target = lab_in + np.array([0.0, 0.05, 0.05])
        delta = adjust_displacement(lab_in, target, GradingParams(preserve_neutrals=1.0))
        np.testing.assert_allclose(delta[:, 1:], 0.0, atol=1e-9)

    def test_skin_protection_damps_skin(self) -> None:
        lab_in = srgb_to_oklab(np.array(SKIN_TONES, dtype=np.float64) / 255.0)
        target = lab_in + np.array([0.1, 0.05, -0.05])
        delta = adjust_displacement(lab_in, target, GradingParams(skin_protection=1.0))
        assert np.all(np.abs(delta[:, 1:]) < 0.05 * 0.2 + 1e-12)
        np.testing.assert_allclose(delta[:, 0], 0.1)  # lightness is never held back

    def test_mix_scales_everything(self) -> None:
        lab_in = srgb_to_oklab(generate_neutral_grid(17))
        target = lab_in + np.array([0.1, 0.02, -0.03])
        delta = adjust_displacement(lab_in, target, GradingParams(mix=0.25))
        np.testing.assert_allclose(delta, np.tile([0.025, 0.005, -0.0075], (len(delta), 1)))


class TestBuildGrid:
    @pytest.mark.parametrize("method", [m for m in SUPPORTED_METHODS if m != "zones"])
    def test_passthrough_equals_raw_bake(self, method: str) -> None:
        """Default adjustments keep the exported .cube identical to the raw transfer."""
        m = fit_transfer(_image((180, 100, 80), 40, 1), _image((70, 120, 200), 35, 2), method)
        result = build_grid(m, 17)
        np.testing.assert_array_equal(result.grid, m.bake(17))
        assert not result.auto_smoothed

    def test_zones_default_matches_raw_bake(self) -> None:
        """Zones always go through the fold check, but the result is unchanged."""
        m = fit_transfer(_image((180, 100, 80), 40, 1), _image((70, 120, 200), 35, 2), "zones")
        result = build_grid(m, 17)
        np.testing.assert_allclose(result.grid, m.bake(17), atol=1e-5)
        assert not result.auto_smoothed

    @pytest.mark.parametrize(
        "params",
        [
            GradingParams(mix=0.0),
            GradingParams(luminance_strength=0.0, chroma_strength=0.0),
        ],
    )
    def test_no_effect_gives_identity(self, model, params: GradingParams) -> None:  # type: ignore[no-untyped-def]
        result = build_grid(model, 17, params)
        np.testing.assert_allclose(result.grid, generate_neutral_grid(17), atol=1e-5)

    def test_grid_is_valid(self, model) -> None:  # type: ignore[no-untyped-def]
        params = GradingParams(
            luminance_strength=0.4, preserve_neutrals=0.7, skin_protection=0.8, smoothing=1.0
        )
        result = build_grid(model, 33, params)
        assert result.grid.shape == (33**3, 3)
        assert result.grid.min() >= 0.0 and result.grid.max() <= 1.0
        assert result.smoothing_used == pytest.approx(1.0)

    @pytest.mark.parametrize(
        "params",
        [
            GradingParams(skin_protection=1.0),
            GradingParams(preserve_neutrals=1.0),
            GradingParams(luminance_strength=0.3, chroma_strength=0.6, mix=0.7),
            GradingParams(skin_protection=1.0, preserve_neutrals=1.0, luminance_strength=0.5),
        ],
    )
    @pytest.mark.parametrize("darkening", [0.5, 0.15])
    def test_adjustments_never_add_folds(self, params: GradingParams, darkening: float) -> None:
        """Strong darkening + protections must not fold a monotonic transfer.

        Regression: holding skin lightness while its neighbours darkened
        folded ~5 % of the grid.
        """
        result = build_grid(_Scale(darkening), 33, params)  # type: ignore[arg-type]
        assert result.fold_fraction == 0.0
        assert not result.auto_smoothed

    def test_auto_smoothing_when_adjustments_fold(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """If adjustments ever add folds, the displacement is smoothed automatically."""
        import stealzalut.core.grading as grading

        original = grading.adjust_displacement

        def rippled(lab_in: np.ndarray, lab_target: np.ndarray, params: GradingParams):  # type: ignore[no-untyped-def]
            delta = original(lab_in, lab_target, params)
            delta[:, 0] += 0.04 * np.sin(lab_in[:, 0] * 40.0)  # high-frequency ripple
            return delta

        monkeypatch.setattr(grading, "adjust_displacement", rippled)
        raw = build_grid(_Scale(1.0), 33)  # type: ignore[arg-type]
        dosed = build_grid(_Scale(1.0), 33, GradingParams(mix=0.9))  # type: ignore[arg-type]
        assert dosed.auto_smoothed
        assert dosed.smoothing_used >= grading.AUTO_SMOOTHING_START
        assert dosed.fold_fraction <= raw.fold_fraction + grading.MAX_EXTRA_FOLDS


class _Scale:
    """Minimal monotonic transfer: multiplies every colour by *k*."""

    method = "scale"

    def __init__(self, k: float) -> None:
        self.k = k

    def apply(self, colors: np.ndarray) -> np.ndarray:
        return np.asarray(np.clip(colors * self.k, 0.0, 1.0))

    def bake(self, size: int) -> np.ndarray:
        return self.apply(generate_neutral_grid(size))

    def fold_reference(self) -> _Scale:
        return self
