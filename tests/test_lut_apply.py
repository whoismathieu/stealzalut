"""Tests for stealzalut.core.lut_apply — applying a LUT grid to an image."""

from __future__ import annotations

import numpy as np
import pytest

from stealzalut.core.lut_apply import apply_lut
from stealzalut.core.lut_engine import generate_neutral_grid


@pytest.fixture()
def image() -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.integers(0, 256, size=(40, 50, 3), dtype=np.uint8)


class TestApplyLut:
    @pytest.mark.parametrize("size", [17, 33, 64])
    def test_identity_grid_keeps_image(self, image: np.ndarray, size: int) -> None:
        out = apply_lut(image, generate_neutral_grid(size), size)
        assert out.dtype == np.uint8
        assert out.shape == image.shape
        assert np.abs(out.astype(int) - image.astype(int)).max() <= 1

    def test_constant_grid_gives_constant_image(self, image: np.ndarray) -> None:
        grid = np.tile([0.2, 0.4, 0.6], (17**3, 1))
        out = apply_lut(image, grid, 17)
        assert np.all(out == np.array([51, 102, 153], dtype=np.uint8))

    def test_channel_swap(self, image: np.ndarray) -> None:
        """A grid mapping (r, g, b) → (b, g, r) swaps red and blue (checks axis order)."""
        grid = generate_neutral_grid(17)[:, ::-1].copy()
        out = apply_lut(image, grid, 17)
        assert np.abs(out.astype(int) - image[..., ::-1].astype(int)).max() <= 1

    def test_linear_transform_between_nodes(self) -> None:
        """A linear grid is reproduced exactly by trilinear interpolation."""
        grid = generate_neutral_grid(17) * 0.5
        img = np.array([[[255, 128, 7]]], dtype=np.uint8)
        out = apply_lut(img, grid, 17)
        np.testing.assert_array_equal(out[0, 0], [128, 64, 4])

    def test_rejects_wrong_grid_shape(self, image: np.ndarray) -> None:
        with pytest.raises(ValueError):
            apply_lut(image, generate_neutral_grid(17), 33)
