"""Tests for stealzalut.core.regularize — grid smoothing and fold detection."""

from __future__ import annotations

import numpy as np
import pytest

from stealzalut.core.colorspace import srgb_to_oklab
from stealzalut.core.lut_engine import generate_neutral_grid
from stealzalut.core.regularize import fold_fraction, smooth_field


class TestSmoothField:
    def test_constant_field_is_unchanged(self) -> None:
        field = np.full((17**3, 3), 0.1)
        np.testing.assert_allclose(smooth_field(field, 17, 1.5), field, atol=1e-12)

    def test_linear_field_is_unchanged_inside(self) -> None:
        """A Gaussian blur preserves linear functions away from the borders."""
        grid = generate_neutral_grid(17)
        smoothed = smooth_field(grid, 17, 1.0).reshape(17, 17, 17, 3)
        inner = grid.reshape(17, 17, 17, 3)[4:-4, 4:-4, 4:-4]
        np.testing.assert_allclose(smoothed[4:-4, 4:-4, 4:-4], inner, atol=1e-12)

    def test_zero_sigma_returns_copy(self) -> None:
        field = np.random.default_rng(0).random((17**3, 3))
        out = smooth_field(field, 17, 0.0)
        np.testing.assert_array_equal(out, field)
        assert out is not field

    def test_spike_is_spread(self) -> None:
        field = np.zeros((17**3, 1))
        field[17**3 // 2] = 1.0
        out = smooth_field(field, 17, 1.0)
        assert out.max() < 0.2
        assert out.sum() == pytest.approx(1.0)


class TestFoldFraction:
    def test_identity_has_no_folds(self) -> None:
        lightness = srgb_to_oklab(generate_neutral_grid(17))[:, 0]
        assert fold_fraction(lightness, 17) == 0.0

    def test_inverted_grid_is_mostly_folded(self) -> None:
        """Every edge drops; only drops below the tolerance are not counted."""
        lightness = srgb_to_oklab(1.0 - generate_neutral_grid(17))[:, 0]
        assert fold_fraction(lightness, 17) > 0.9
        assert fold_fraction(lightness, 17, tolerance=0.0) == pytest.approx(1.0)

    def test_tolerance_ignores_tiny_drops(self) -> None:
        lightness = srgb_to_oklab(generate_neutral_grid(17))[:, 0]
        noisy = lightness - 1e-5 * (np.arange(lightness.size) % 2)
        assert fold_fraction(noisy, 17) == 0.0
