"""Colours outside sRGB must survive every step, from the file to the LUT and back."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from stealzalut.core.ambiance import AmbianceTransferModel
from stealzalut.core.color_transfer import SUPPORTED_METHODS, OklabTransferModel, fit_transfer
from stealzalut.core.colorspace import gamut_compress, oklab_to_linear, oklab_to_srgb, srgb_to_oklab
from stealzalut.core.grading import GradingParams, build_grid
from stealzalut.core.look_pack import look_gaps
from stealzalut.core.lut_apply import (
    apply_lut,
    apply_lut_full,
    render_lut,
    render_lut_lab,
    source_pixels,
)
from stealzalut.core.lut_engine import generate_neutral_grid
from stealzalut.core.rgb_spaces import ADOBE_RGB, DISPLAY_P3, SRGB
from stealzalut.core.zone_transfer import ZoneTransferModel, find_conflicts, fit_zone_transfer
from stealzalut.utils.image_io import WideImage, create_thumbnail, load_image_rgb
from stealzalut.workers.transfer_worker import TransferWorker
from tests.test_wide_images import _as_p3, _icc, _neon
from tests.test_zone_transfer import REFERENCE, SOURCE

#: The corners of Display P3 and two of its edges: none of them fits in sRGB.
P3_ONLY = np.array(
    [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.6], [0.0, 1.0, 0.7]]
)
WIDE_LAB = DISPLAY_P3.to_oklab(P3_ONLY)


def _outside_srgb(lab: np.ndarray) -> np.ndarray:
    linear = oklab_to_linear(lab)
    return np.asarray((linear.min(axis=1) < -1e-6) | (linear.max(axis=1) > 1.0 + 1e-6))


def test_the_probe_colours_are_outside_srgb() -> None:
    assert _outside_srgb(WIDE_LAB).all()


# ── Models: apply_lab is native, without a detour through sRGB ───────────────


@pytest.fixture(scope="module")
def oklab_model() -> OklabTransferModel:
    model = fit_transfer(SOURCE, REFERENCE, method="oklab")
    assert isinstance(model, OklabTransferModel)
    return model


@pytest.fixture(scope="module")
def zone_model() -> ZoneTransferModel:
    return fit_zone_transfer(SOURCE, REFERENCE, zone_count=3)


def test_oklab_apply_lab_is_the_affine_map_everywhere(oklab_model: OklabTransferModel) -> None:
    out = oklab_model.apply_lab(WIDE_LAB)
    m = oklab_model
    expected_l = m.ref_mean[0] + m.lightness_scale * (WIDE_LAB[:, 0] - m.src_mean[0])
    expected_ab = m.ref_mean[1:] + (WIDE_LAB[:, 1:] - m.src_mean[1:]) @ m.chroma_matrix.T
    np.testing.assert_allclose(out[:, 0], expected_l, atol=1e-12)
    np.testing.assert_allclose(out[:, 1:], expected_ab, atol=1e-12)


def test_oklab_apply_lab_tells_wide_colours_from_their_srgb_clipping(
    oklab_model: OklabTransferModel,
) -> None:
    """The inherited default moved a wide colour like its sRGB neighbour: an affine map does not."""
    clipped = gamut_compress(WIDE_LAB)
    wide_out, clipped_out = oklab_model.apply_lab(WIDE_LAB), oklab_model.apply_lab(clipped)
    expected = (WIDE_LAB[:, 1:] - clipped[:, 1:]) @ oklab_model.chroma_matrix.T
    np.testing.assert_allclose(wide_out[:, 1:] - clipped_out[:, 1:], expected, atol=1e-12)
    assert np.abs(expected).max() > 0.01


@pytest.mark.parametrize("method", ["oklab", "ambiance", "look", "zones"])
def test_apply_is_apply_lab_brought_into_srgb(method: str) -> None:
    model = fit_transfer(SOURCE, REFERENCE, method=method)
    colours = generate_neutral_grid(17)
    expected = oklab_to_srgb(gamut_compress(model.apply_lab(srgb_to_oklab(colours))))
    np.testing.assert_allclose(model.apply(colours), expected, atol=1e-12)


def test_zone_global_part_is_not_clipped_to_srgb(zone_model: ZoneTransferModel) -> None:
    """Lightness always follows the global transfer: it must be that of the wide colour."""
    delta = zone_model.displacement(WIDE_LAB)
    expected = zone_model.global_model.apply_lab(WIDE_LAB) - WIDE_LAB
    np.testing.assert_allclose(delta[:, 0], expected[:, 0], atol=1e-9)


def test_zone_displacement_is_continuous_across_the_srgb_boundary(
    zone_model: ZoneTransferModel,
) -> None:
    inside = gamut_compress(WIDE_LAB)
    just_outside = inside + 1e-4 * (WIDE_LAB - inside)
    step = zone_model.displacement(just_outside) - zone_model.displacement(inside)
    assert np.abs(step).max() < 1e-3


def test_find_conflicts_still_runs(zone_model: ZoneTransferModel) -> None:
    assert isinstance(find_conflicts(zone_model), list)


def test_ambiance_apply_lab_takes_the_wide_lightness() -> None:
    model = fit_transfer(SOURCE, REFERENCE, method="ambiance")
    assert isinstance(model, AmbianceTransferModel)
    out = model.apply_lab(WIDE_LAB)
    expected = np.clip(model.global_model.apply_lab(WIDE_LAB)[:, 0], 0.0, 1.0)
    np.testing.assert_allclose(out[:, 0], expected, atol=1e-12)
    # The result may itself be outside sRGB: nothing was compressed on the way
    plain = replace_protection(model).apply_lab(WIDE_LAB)
    assert _outside_srgb(plain).any()


def replace_protection(model: AmbianceTransferModel) -> AmbianceTransferModel:
    from dataclasses import replace

    return replace(model, mute=0.0, protect_vivid=1.0)


# ── build_grid: one sequence for every space ─────────────────────────────────


@pytest.mark.parametrize("method", SUPPORTED_METHODS)
@pytest.mark.parametrize("space", [SRGB, DISPLAY_P3, ADOBE_RGB])
def test_plain_grid_is_apply_lab_encoded_once(method: str, space: object) -> None:
    model = fit_transfer(SOURCE, REFERENCE, method=method)
    lab_in = space.to_oklab(generate_neutral_grid(17))  # type: ignore[attr-defined]
    grid = build_grid(model, 17, None, None, space).grid  # type: ignore[arg-type]
    if model.fold_reference() is not model:
        assert grid.min() >= -1e-9 and grid.max() <= 1.0 + 1e-9
        return
    expected = space.from_oklab(  # type: ignore[attr-defined]
        gamut_compress(model.apply_lab(lab_in), to_linear=space.oklab_to_linear)  # type: ignore[attr-defined]
    )
    np.testing.assert_allclose(grid, expected, atol=1e-9)


@pytest.mark.parametrize("space", [SRGB, DISPLAY_P3])
def test_adjustments_act_before_the_gamut_limit(
    oklab_model: OklabTransferModel, space: object
) -> None:
    """Half the mix is half the *unclipped* displacement, in sRGB as in any other space."""
    from dataclasses import replace

    strong = replace(oklab_model, chroma_matrix=oklab_model.chroma_matrix * 4.0)
    lab_in = space.to_oklab(generate_neutral_grid(17))  # type: ignore[attr-defined]
    grid = build_grid(strong, 17, GradingParams(mix=0.5), None, space).grid  # type: ignore[arg-type]
    half = lab_in + 0.5 * (strong.apply_lab(lab_in) - lab_in)
    expected = space.from_oklab(gamut_compress(half, to_linear=space.oklab_to_linear))  # type: ignore[attr-defined]
    np.testing.assert_allclose(grid, expected, atol=1e-6)


# ── Reading: the file's own pixels are kept ──────────────────────────────────


def test_p3_file_keeps_its_native_pixels(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "p3.png"
    Image.fromarray(_neon()).save(path, icc_profile=_icc())
    _as_p3(monkeypatch)
    image = load_image_rgb(path)
    assert image.native is not None and image.native_space is DISPLAY_P3
    assert image.native.shape == image.shape
    assert np.array_equal(image.native, _neon())  # the file's values, untouched


def test_plain_file_has_no_native_pixels(tmp_path: Path) -> None:
    path = tmp_path / "plain.png"
    Image.fromarray(_neon()).save(path)
    image = load_image_rgb(path)
    assert image.native is None and image.native_space is None


def test_derived_arrays_drop_the_native_pixels() -> None:
    image = WideImage(_neon(), None, "Display P3", _neon(), DISPLAY_P3)
    assert image.native is not None
    assert getattr(create_thumbnail(image, 32), "native", None) is None
    assert image[:10].native is None


def test_source_pixels_prefers_the_native_ones() -> None:
    native = _neon()
    image = WideImage(np.zeros_like(native), None, "Display P3", native, DISPLAY_P3)
    pixels, space = source_pixels(image)
    assert space is DISPLAY_P3 and np.array_equal(pixels, native)
    small, space = source_pixels(image, 40)
    assert space is DISPLAY_P3 and max(small.shape[:2]) == 40
    plain, space = source_pixels(native)
    assert space is SRGB and np.array_equal(plain, native)


# ── Applying a LUT to a picture in its own space ─────────────────────────────


def _halving_grid(space: object, size: int = 17) -> np.ndarray:
    """A LUT of *space* that halves chroma: its output for a wide input fits in sRGB."""
    lab = space.to_oklab(generate_neutral_grid(size))  # type: ignore[attr-defined]
    return np.asarray(space.from_oklab(lab * np.array([1.0, 0.5, 0.5])))  # type: ignore[attr-defined]


def test_srgb_pixels_through_an_srgb_lut_are_unchanged_behaviour() -> None:
    image = _neon()
    grid = _halving_grid(SRGB)
    assert np.array_equal(render_lut(image, SRGB, grid, 17, SRGB), apply_lut_full(image, grid, 17))
    assert np.array_equal(render_lut(image, SRGB, grid, 17, SRGB), apply_lut(image, grid, 17))


def test_wide_colours_enter_the_lut_intact() -> None:
    """A P3 red through a P3 LUT gives what the LUT says for *that* red, not for sRGB red."""
    pixels = np.zeros((2, 2, 3), dtype=np.uint8)
    pixels[..., 0] = 255
    out = render_lut(pixels, DISPLAY_P3, _halving_grid(DISPLAY_P3, 33), 33, DISPLAY_P3)
    wide_red = DISPLAY_P3.to_oklab(np.array([[1.0, 0.0, 0.0]]))
    expected = oklab_to_srgb(wide_red * np.array([1.0, 0.5, 0.5]))[0] * 255.0
    clipped = oklab_to_srgb(gamut_compress(wide_red) * np.array([1.0, 0.5, 0.5]))[0] * 255.0
    assert np.abs(out[0, 0] - expected).max() <= 2.0
    assert np.abs(expected - clipped).max() > 4.0  # the two answers are told apart


def test_an_srgb_cube_clips_wide_pixels_at_its_entrance() -> None:
    """Truthful preview: an sRGB .cube cannot see more than sRGB, in the host either."""
    pixels = np.zeros((2, 2, 3), dtype=np.uint8)
    pixels[..., 0] = 255
    identity = generate_neutral_grid(17)
    out = render_lut(pixels, DISPLAY_P3, identity, 17, SRGB)
    assert tuple(out[0, 0]) == (255, 0, 0)


def test_render_lut_shows_wide_results_inside_srgb() -> None:
    pixels = np.zeros((300, 4, 3), dtype=np.uint8)  # more rows than one band
    pixels[..., 1] = 255
    out = render_lut(pixels, DISPLAY_P3, generate_neutral_grid(17), 17, DISPLAY_P3)
    assert out.shape == pixels.shape and out.dtype == np.uint8
    assert (out == out[0, 0]).all()
    lab = render_lut_lab(pixels[:4], DISPLAY_P3, generate_neutral_grid(17), 17, DISPLAY_P3)
    assert lab.shape == (4, 4, 3) and _outside_srgb(lab.reshape(-1, 3)).all()


def test_render_lut_between_two_wide_spaces() -> None:
    pixels = np.full((2, 2, 3), 128, dtype=np.uint8)
    out = render_lut(pixels, ADOBE_RGB, generate_neutral_grid(17), 17, DISPLAY_P3)
    assert np.abs(out.astype(int) - 128).max() <= 1  # a grey is a grey in every space


# ── Preview worker ───────────────────────────────────────────────────────────


def _run(worker: TransferWorker) -> np.ndarray:
    results: list[np.ndarray] = []
    worker.result_ready.connect(results.append)
    worker.run()
    assert len(results) == 1
    return results[0]


def test_preview_applies_the_lut_to_the_native_pixels() -> None:
    native = _neon()
    srgb = oklab_to_srgb(gamut_compress(DISPLAY_P3.to_oklab(native.reshape(-1, 3) / 255.0)))
    shown = (srgb * 255.0 + 0.5).astype(np.uint8).reshape(native.shape)
    source = WideImage(shown, None, "Display P3", native, DISPLAY_P3)
    worker = TransferWorker(source, REFERENCE, method="oklab", lut_size=17, space=DISPLAY_P3)
    result = _run(worker)
    model = fit_transfer(source, REFERENCE, method="oklab")
    grid = build_grid(model, 17, None, None, DISPLAY_P3).grid
    assert np.array_equal(result, render_lut(native, DISPLAY_P3, grid, 17, DISPLAY_P3))
    assert getattr(result, "analysis_lab", None) is not None


def test_plain_preview_is_a_plain_array() -> None:
    result = _run(TransferWorker(SOURCE, REFERENCE, method="oklab", lut_size=17))
    assert getattr(result, "analysis_lab", None) is None


# ── AI pack: gaps measured on the unclipped colours ──────────────────────────


def test_gaps_use_the_unclipped_colours() -> None:
    grey = np.full((40, 40, 3), 128, dtype=np.uint8)
    vivid = np.tile(DISPLAY_P3.to_oklab(np.array([[1.0, 0.0, 0.0]])), (40 * 40, 1))
    wide = WideImage(grey, vivid.reshape(40, 40, 3), "Display P3")
    gaps = look_gaps(wide, grey)
    expected = float(np.hypot(vivid[0, 1], vivid[0, 2]))
    assert gaps["render"]["chroma_mean"] == pytest.approx(expected, abs=1e-3)
    assert gaps["reference"]["chroma_mean"] == pytest.approx(0.0, abs=1e-3)
    # More chroma than any sRGB red: the clipped picture could not have said so
    srgb_red = srgb_to_oklab(np.array([[1.0, 0.0, 0.0]]))[0]
    assert gaps["render"]["chroma_mean"] > float(np.hypot(srgb_red[1], srgb_red[2])) + 0.01
