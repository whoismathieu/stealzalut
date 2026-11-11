"""Tests for the colour space of the exported ``.cube``.

The key property: a grid made for another space must describe **the same
grade** as the sRGB grid the preview shows, for every colour both can hold.
"""

from __future__ import annotations

import numpy as np
import pytest

from stealzalut.core.color_transfer import SUPPORTED_METHODS, fit_transfer
from stealzalut.core.colorspace import gamut_compress, oklab_to_srgb, srgb_to_oklab
from stealzalut.core.finishing import Finishing
from stealzalut.core.grading import GradingParams, build_grid
from stealzalut.core.look import fit_look
from stealzalut.core.lut_apply import apply_lut
from stealzalut.core.rgb_spaces import (
    ADOBE_RGB,
    CUBE_SPACES,
    DISPLAY_P3,
    PROPHOTO_RGB,
    REC709,
    SRGB,
    RGBSpace,
    space_by_key,
)
from tests.test_look import _green_dusk, _neutral_street

WIDE = [space for space in CUBE_SPACES if space is not SRGB]


def _colours(seed: int = 0, count: int = 2000) -> np.ndarray:
    return np.random.default_rng(seed).uniform(0.0, 1.0, (count, 3))


# ── Conversions ──────────────────────────────────────────────────────────────


def test_srgb_space_is_the_existing_conversion() -> None:
    colours = _colours()
    assert np.allclose(SRGB.to_oklab(colours), srgb_to_oklab(colours), atol=1e-12)
    lab = srgb_to_oklab(colours)
    assert np.allclose(SRGB.from_oklab(lab), oklab_to_srgb(lab), atol=1e-12)


@pytest.mark.parametrize("space", CUBE_SPACES, ids=lambda s: s.key)
def test_round_trip(space: RGBSpace) -> None:
    colours = _colours()
    error = np.abs(space.from_oklab(space.to_oklab(colours)) - colours)
    assert np.quantile(error, 0.99) < 1e-5
    assert error.max() < 1e-3  # pure gamma curves lose a little next to black


@pytest.mark.parametrize("space", CUBE_SPACES, ids=lambda s: s.key)
def test_white_and_black_are_neutral(space: RGBSpace) -> None:
    lab = space.to_oklab(np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [0.5, 0.5, 0.5]]))
    assert lab[0, 0] == pytest.approx(0.0, abs=1e-6)
    assert lab[1, 0] == pytest.approx(1.0, abs=2e-3)
    assert np.abs(lab[:, 1:]).max() < 2e-3  # greys have no colour in any space


def test_known_primaries() -> None:
    """Published values: where the pure reds of wider spaces fall in linear sRGB."""
    red = np.array([[1.0, 0.0, 0.0]])
    assert np.allclose(red @ DISPLAY_P3.to_linear_srgb.T, [[1.2249, -0.0420, -0.0197]], atol=2e-3)
    assert np.allclose(red @ ADOBE_RGB.to_linear_srgb.T, [[1.3984, 0.0, 0.0]], atol=2e-3)
    green = np.array([[0.0, 1.0, 0.0]])
    assert np.allclose(green @ ADOBE_RGB.to_linear_srgb.T, [[-0.3984, 1.0, -0.0429]], atol=2e-3)
    assert (red @ PROPHOTO_RGB.to_linear_srgb.T)[0, 0] > 2.0  # far outside sRGB


def test_wide_spaces_hold_colours_srgb_cannot() -> None:
    for space in (DISPLAY_P3, ADOBE_RGB, PROPHOTO_RGB):
        assert space.wide
        lab = space.to_oklab(np.array([[0.0, 1.0, 0.0]]))  # the space's pure green
        chroma = float(np.hypot(lab[0, 1], lab[0, 2]))
        srgb_green = srgb_to_oklab(np.array([[0.0, 1.0, 0.0]]))
        assert chroma > float(np.hypot(srgb_green[0, 1], srgb_green[0, 2])) + 0.01
    assert not SRGB.wide and not REC709.wide


def test_rec709_differs_from_srgb_only_by_its_curve() -> None:
    grey = np.array([[0.5, 0.5, 0.5]])
    assert REC709.to_oklab(grey)[0, 0] < SRGB.to_oklab(grey)[0, 0]  # gamma 2.4 is darker
    assert np.array_equal(REC709.to_linear_srgb, SRGB.to_linear_srgb)


def test_space_by_key() -> None:
    assert space_by_key("display-p3") is DISPLAY_P3
    assert len({space.key for space in CUBE_SPACES}) == len(CUBE_SPACES)
    with pytest.raises(ValueError, match="Unknown colour space"):
        space_by_key("cmyk")


# ── Gamut ────────────────────────────────────────────────────────────────────


def test_gamut_compress_respects_the_target_space() -> None:
    p3_red = DISPLAY_P3.to_oklab(np.array([[1.0, 0.0, 0.0]]))
    kept = gamut_compress(p3_red, to_linear=DISPLAY_P3.oklab_to_linear)
    assert np.allclose(kept, p3_red, atol=1e-3)  # inside P3: untouched
    squeezed = gamut_compress(p3_red)  # towards sRGB: chroma reduced, hue kept
    assert np.hypot(*squeezed[0, 1:]) < np.hypot(*p3_red[0, 1:]) - 0.01
    assert np.arctan2(squeezed[0, 2], squeezed[0, 1]) == pytest.approx(
        np.arctan2(p3_red[0, 2], p3_red[0, 1]), abs=1e-6
    )


def test_gamut_compress_default_is_unchanged() -> None:
    lab = srgb_to_oklab(_colours())
    lab[:, 1:] *= 1.6
    assert np.array_equal(gamut_compress(lab), gamut_compress(lab, to_linear=None))


# ── Models outside sRGB ──────────────────────────────────────────────────────


@pytest.mark.parametrize("method", SUPPORTED_METHODS)
def test_apply_lab_agrees_with_apply_inside_srgb(method: str) -> None:
    """Inside sRGB, the Oklab entry point is the same grade as ``apply``."""
    model = fit_transfer(_neutral_street(), _green_dusk(), method=method)
    colours = _colours(1, 500) * 0.8 + 0.1
    expected = srgb_to_oklab(model.apply(colours))
    got = srgb_to_oklab(oklab_to_srgb(gamut_compress(model.apply_lab(srgb_to_oklab(colours)))))
    assert np.abs(got - expected).mean() < 5e-3


def test_look_handles_colours_outside_srgb() -> None:
    model = fit_look(_neutral_street(), _green_dusk())
    lab = PROPHOTO_RGB.to_oklab(np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]))
    out = model.apply_lab(lab)
    assert np.isfinite(out).all()
    assert np.allclose(out[:, 0], model.tone(lab[:, 0]))  # tone only depends on lightness


# ── Grids ────────────────────────────────────────────────────────────────────


def test_default_space_is_srgb_and_unchanged() -> None:
    model = fit_look(_neutral_street(), _green_dusk())
    params = GradingParams(mix=0.8, preserve_vivid=0.3)
    plain = build_grid(model, 17, params)
    explicit = build_grid(model, 17, params, None, SRGB)
    assert np.array_equal(plain.grid, explicit.grid)


@pytest.mark.parametrize("space", WIDE, ids=lambda s: s.key)
@pytest.mark.parametrize("method", ["look", "zones", "oklab"])
def test_other_space_is_the_same_grade(space: RGBSpace, method: str) -> None:
    """A picture graded through the grid of another space matches the preview."""
    source = _neutral_street()
    model = fit_transfer(source, _green_dusk(), method=method)
    params = GradingParams(chroma_strength=0.9)
    finishing = Finishing(vibrance=0.2)
    srgb_result = build_grid(model, 33, params, finishing)
    other_result = build_grid(model, 33, params, finishing, space)
    assert other_result.grid.shape == srgb_result.grid.shape
    assert other_result.grid.min() >= 0.0 and other_result.grid.max() <= 1.0

    flat = source.reshape(-1, 3) / 255.0
    expected = apply_lut(source, srgb_result.grid, 33).reshape(-1, 3) / 255.0

    # The same picture, encoded in the other space, through the other grid
    encoded = space.from_oklab(srgb_to_oklab(flat))
    as_image = np.clip(encoded * 255.0 + 0.5, 0, 255).astype(np.uint8).reshape(source.shape)
    graded = apply_lut(as_image, other_result.grid, 33).reshape(-1, 3) / 255.0
    back = oklab_to_srgb(gamut_compress(space.to_oklab(graded)))

    error = np.abs(srgb_to_oklab(back) - srgb_to_oklab(expected))
    assert error.mean() < 0.006  # 8-bit rounding and interpolation in two encodings
    assert np.quantile(error, 0.99) < 0.03


def test_wide_grid_keeps_colours_the_srgb_grid_clips() -> None:
    """Identity-like grade: a P3 grid holds the P3 red, the sRGB grid cannot."""
    img = _neutral_street()
    model = fit_look(img, img)  # near identity
    grid = build_grid(model, 17, None, None, DISPLAY_P3).grid.reshape(17, 17, 17, 3)
    red_out = grid[0, 0, 16]  # node (r=1, g=0, b=0) of the P3 grid
    lab = DISPLAY_P3.to_oklab(red_out[None, :])
    srgb_red = srgb_to_oklab(np.array([[1.0, 0.0, 0.0]]))
    assert np.hypot(*lab[0, 1:]) > np.hypot(*srgb_red[0, 1:])


def test_grid_in_another_space_does_not_fold() -> None:
    """The look's tone is monotone in lightness, whatever the space of the grid.

    The fold measure assumes that raising R, G or B raises lightness.  That
    holds for real colours; ProPhoto's primaries are not real colours, so its
    *neutral* grid already "folds" by that measure.  The grade must not add any.
    """
    from stealzalut.core.lut_engine import generate_neutral_grid
    from stealzalut.core.regularize import fold_fraction

    model = fit_look(_neutral_street(), _green_dusk())
    for space in WIDE:
        own = fold_fraction(space.to_oklab(generate_neutral_grid(17))[:, 0], 17)
        assert build_grid(model, 17, None, None, space).fold_fraction <= own + 0.01
        if space is not PROPHOTO_RGB:
            assert own == 0.0
