"""Tests for stealzalut.core.colorspace — sRGB / Oklab / Oklch conversions."""

from __future__ import annotations

import numpy as np
import pytest

from stealzalut.core.colorspace import (
    gamut_compress,
    linear_to_srgb,
    oklab_to_linear,
    oklab_to_oklch,
    oklab_to_srgb,
    oklch_to_oklab,
    srgb_to_linear,
    srgb_to_oklab,
)
from stealzalut.core.lut_engine import generate_neutral_grid


class TestSrgbTransfer:
    def test_roundtrip(self) -> None:
        x = np.linspace(0.0, 1.0, 1001)
        np.testing.assert_allclose(linear_to_srgb(srgb_to_linear(x)), x, atol=1e-12)

    def test_endpoints(self) -> None:
        np.testing.assert_allclose(srgb_to_linear(np.array([0.0, 1.0])), [0.0, 1.0])

    def test_mid_grey(self) -> None:
        """sRGB 0.5 ≈ 21.4 % linear light."""
        assert srgb_to_linear(np.array([0.5]))[0] == pytest.approx(0.2140, abs=1e-4)

    def test_negative_linear_is_clamped(self) -> None:
        assert linear_to_srgb(np.array([-0.2]))[0] == 0.0


class TestOklab:
    def test_roundtrip_on_lut_grid(self) -> None:
        grid = generate_neutral_grid(33)
        np.testing.assert_allclose(oklab_to_srgb(srgb_to_oklab(grid)), grid, atol=1e-5)

    def test_white_and_black(self) -> None:
        lab = srgb_to_oklab(np.array([[1.0, 1.0, 1.0], [0.0, 0.0, 0.0]]))
        np.testing.assert_allclose(lab[0], [1.0, 0.0, 0.0], atol=1e-6)
        np.testing.assert_allclose(lab[1], [0.0, 0.0, 0.0], atol=1e-9)

    def test_greys_are_neutral(self) -> None:
        g = np.linspace(0.0, 1.0, 11)
        lab = srgb_to_oklab(np.stack([g, g, g], axis=1))
        np.testing.assert_allclose(lab[:, 1:], 0.0, atol=1e-6)
        assert np.all(np.diff(lab[:, 0]) > 0)

    def test_known_value_red(self) -> None:
        """Pure sRGB red ≈ Oklab (0.6280, 0.2249, 0.1258) (published value)."""
        lab = srgb_to_oklab(np.array([[1.0, 0.0, 0.0]]))[0]
        np.testing.assert_allclose(lab, [0.6280, 0.2249, 0.1258], atol=2e-4)


class TestOklch:
    def test_roundtrip(self) -> None:
        lab = srgb_to_oklab(generate_neutral_grid(17))
        np.testing.assert_allclose(oklch_to_oklab(oklab_to_oklch(lab)), lab, atol=1e-12)

    def test_ranges(self) -> None:
        lch = oklab_to_oklch(srgb_to_oklab(generate_neutral_grid(17)))
        assert lch[:, 2].min() >= 0.0 and lch[:, 2].max() < 360.0
        assert lch[:, 1].min() >= 0.0

    def test_skin_tone_hue(self) -> None:
        """A typical skin tone sits around h ≈ 50° in Oklch."""
        lch = oklab_to_oklch(srgb_to_oklab(np.array([[224, 172, 140]]) / 255.0))[0]
        assert 40.0 < lch[2] < 65.0


class TestGamutCompress:
    def test_in_gamut_is_unchanged(self) -> None:
        lab = srgb_to_oklab(generate_neutral_grid(17))
        np.testing.assert_allclose(gamut_compress(lab), lab, atol=1e-12)

    def test_out_of_gamut_is_brought_back(self) -> None:
        lab = srgb_to_oklab(generate_neutral_grid(17))
        lab[:, 1:] *= 3.0  # push chroma far outside sRGB
        lin = oklab_to_linear(gamut_compress(lab))
        assert lin.min() >= -1e-5 and lin.max() <= 1.0 + 1e-5

    def test_preserves_lightness_and_hue(self) -> None:
        lab = np.array([[0.6, 0.4, 0.2]])  # very saturated, out of gamut
        out = gamut_compress(lab)
        assert out[0, 0] == pytest.approx(0.6)
        lch_in, lch_out = oklab_to_oklch(lab)[0], oklab_to_oklch(out)[0]
        assert lch_out[2] == pytest.approx(lch_in[2], abs=1e-9)
        assert lch_out[1] < lch_in[1]

    def test_lightness_is_clamped(self) -> None:
        out = gamut_compress(np.array([[1.3, 0.0, 0.0], [-0.1, 0.0, 0.0]]))
        np.testing.assert_allclose(out[:, 0], [1.0, 0.0])
