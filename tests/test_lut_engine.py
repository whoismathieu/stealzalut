"""Tests for stealzalut.core.lut_engine — neutral grid generation."""

from __future__ import annotations

import numpy as np
import pytest

from stealzalut.core.lut_engine import (
    VALID_LUT_SIZES,
    generate_neutral_grid,
    validate_grid_shape,
)


class TestGenerateNeutralGrid:
    """Tests for generate_neutral_grid()."""

    @pytest.mark.parametrize("size", sorted(VALID_LUT_SIZES))
    def test_shape(self, size: int) -> None:
        """Grid shape must be (size³, 3)."""
        grid = generate_neutral_grid(size)
        assert grid.shape == (size**3, 3)

    @pytest.mark.parametrize("size", sorted(VALID_LUT_SIZES))
    def test_dtype(self, size: int) -> None:
        """Grid values must be float64."""
        grid = generate_neutral_grid(size)
        assert grid.dtype == np.float64

    @pytest.mark.parametrize("size", sorted(VALID_LUT_SIZES))
    def test_bounds(self, size: int) -> None:
        """All values must lie in [0.0, 1.0]."""
        grid = generate_neutral_grid(size)
        assert grid.min() >= 0.0
        assert grid.max() <= 1.0

    def test_first_entry_is_black(self) -> None:
        """First entry (B=0, G=0, R=0) → (0, 0, 0)."""
        grid = generate_neutral_grid(17)
        np.testing.assert_array_almost_equal(grid[0], [0.0, 0.0, 0.0])

    def test_last_entry_is_white(self) -> None:
        """Last entry (B=1, G=1, R=1) → (1, 1, 1)."""
        grid = generate_neutral_grid(17)
        np.testing.assert_array_almost_equal(grid[-1], [1.0, 1.0, 1.0])

    def test_r_varies_fastest(self) -> None:
        """In .cube order, the R channel should increment in the first N entries."""
        size = 17
        grid = generate_neutral_grid(size)
        # The first `size` entries have B=0, G=0, R = 0..1
        first_slice = grid[:size]
        r_values = first_slice[:, 0]
        expected = np.linspace(0.0, 1.0, size)
        np.testing.assert_array_almost_equal(r_values, expected)
        # G and B should all be zero
        np.testing.assert_array_almost_equal(first_slice[:, 1], 0.0)
        np.testing.assert_array_almost_equal(first_slice[:, 2], 0.0)

    def test_invalid_size_raises(self) -> None:
        """Unsupported sizes must raise ValueError."""
        with pytest.raises(ValueError, match="LUT size must be one of"):
            generate_neutral_grid(10)

    def test_invalid_size_25_raises(self) -> None:
        """Size 25 is not in the valid set."""
        with pytest.raises(ValueError):
            generate_neutral_grid(25)


class TestValidateGridShape:
    """Tests for validate_grid_shape()."""

    def test_valid_shape_passes(self) -> None:
        """No exception for a correctly shaped grid."""
        grid = np.zeros((17**3, 3), dtype=np.float64)
        validate_grid_shape(grid, 17)  # should not raise

    def test_wrong_shape_raises(self) -> None:
        """Mismatched shape must raise ValueError."""
        grid = np.zeros((100, 3), dtype=np.float64)
        with pytest.raises(ValueError, match="does not match expected"):
            validate_grid_shape(grid, 17)

