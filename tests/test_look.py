"""Tests for the look transfer (tone curve + tone-conditional chroma transport)."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from stealzalut.core.color_transfer import SUPPORTED_METHODS, fit_transfer
from stealzalut.core.colorspace import oklab_to_srgb, oklch_to_oklab, srgb_to_oklab
from stealzalut.core.grading import build_grid
from stealzalut.core.look import (
    AB_CELLS,
    BANDS,
    MONOCHROME_VIVID_PROTECTION,
    REACH_RANGE,
    SEPARATION_RANGE,
    STRETCH_LIMITS,
    TONE_REACH_RATIO,
    TONE_SAMPLES,
    LookTransferModel,
    fit_look,
    fit_stretch,
    fit_tone_curve,
    lattice_position,
    look_diagnostics,
    sliced_transport,
    weighted_quantile,
    zone_weights,
)
from stealzalut.core.lut_apply import apply_lut
from stealzalut.core.palette import ANALYSIS_MAX_SIZE, analyze_zones
from stealzalut.utils.image_io import create_thumbnail

U8 = np.ndarray


def _lch_rgb(lightness: float, chroma: float, hue: float) -> U8:
    lab = oklch_to_oklab(np.array([[lightness, chroma, hue]]))
    return np.clip(oklab_to_srgb(lab)[0] * 255.0 + 0.5, 0, 255).astype(np.uint8)


def _noisy(img: U8, seed: int) -> U8:
    rng = np.random.default_rng(seed)
    return np.clip(img + rng.normal(0, 2, img.shape), 0, 255).astype(np.uint8)


def _neutral_street() -> U8:
    """Greyish night street: lifted shadows, many white signs, two neons (RGB)."""
    img = np.zeros((160, 200, 3), dtype=np.uint8)
    for y in range(160):
        img[y] = _lch_rgb(0.22 + 0.35 * y / 160, 0.015, 250.0)
    img[20:60, 20:60] = _lch_rgb(0.95, 0.01, 90.0)  # white signs (10 %)
    img[80:100, 120:150] = _lch_rgb(0.62, 0.20, 25.0)  # red neon
    img[110:125, 30:60] = _lch_rgb(0.70, 0.14, 250.0)  # blue neon
    return _noisy(img, 1)


def _green_dusk() -> U8:
    """Reference: green-black shadows, mauve sky, a few bright lights (RGB)."""
    img = np.zeros((200, 160, 3), dtype=np.uint8)
    for y in range(200):
        img[y] = _lch_rgb(0.10 + 0.30 * y / 200, 0.06, 160.0)
    img[:50, 60:120] = _lch_rgb(0.60, 0.07, 330.0)  # mauve sky
    img[100:106, 40:60] = _lch_rgb(0.92, 0.03, 100.0)  # small light
    img[120:135, 100:130] = _lch_rgb(0.55, 0.20, 25.0)  # red sign
    return _noisy(img, 2)


def _purple_night() -> U8:
    """Reference with a single hue: purple sky, dark ground, one warm light."""
    img = np.zeros((200, 160, 3), dtype=np.uint8)
    for y in range(140):
        img[y] = _lch_rgb(0.22 + 0.15 * y / 140, 0.15, 298.0)
    img[140:] = _lch_rgb(0.12, 0.02, 310.0)
    img[60:70, 70:90] = _lch_rgb(0.9, 0.05, 70.0)
    return _noisy(img, 3)


def _render(model: LookTransferModel, image: U8) -> np.ndarray:
    result = build_grid(model, 33)
    out = apply_lut(image, result.grid, 33)
    return srgb_to_oklab(out.reshape(-1, 3) / 255.0)


@pytest.fixture(scope="module")
def green_model() -> LookTransferModel:
    return fit_look(_neutral_street(), _green_dusk())


# ── Tone ─────────────────────────────────────────────────────────────────────


def test_weighted_quantile_matches_numpy_for_equal_weights() -> None:
    values = np.linspace(0.0, 1.0, 1001)
    u = np.array([0.1, 0.5, 0.9])
    got = weighted_quantile(values, np.ones_like(values), u)
    assert np.allclose(got, np.quantile(values, u), atol=1e-3)


def test_weighted_quantile_ignores_zero_weights() -> None:
    values = np.array([0.0, 0.1, 0.2, 0.9, 1.0])
    weights = np.array([1.0, 1.0, 1.0, 0.0, 0.0])
    assert weighted_quantile(values, weights, np.array([0.999]))[0] <= 0.2 + 1e-9


def test_tone_curve_is_monotone_and_sampled() -> None:
    rng = np.random.default_rng(0)
    x, y = fit_tone_curve(rng.uniform(0.2, 0.95, 5000), rng.uniform(0.05, 0.6, 5000))
    assert x.shape == y.shape == (TONE_SAMPLES,)
    assert x[0] == 0.0 and x[-1] == 1.0
    assert np.all(np.diff(y) >= -1e-12)
    assert y[0] == 0.0 and y[-1] <= 1.0


def test_tone_curve_matches_the_median() -> None:
    rng = np.random.default_rng(0)
    src, ref = rng.uniform(0.2, 0.95, 5000), rng.uniform(0.05, 0.6, 5000)
    x, y = fit_tone_curve(src, ref)
    assert np.interp(np.median(src), x, y) == pytest.approx(np.median(ref), abs=0.02)


def test_tone_curve_keeps_light_sources_bright() -> None:
    """Many white signs in the source, few in the reference: whites stay bright."""
    rng = np.random.default_rng(0)
    src = np.concatenate([rng.uniform(0.2, 0.6, 9000), np.full(1000, 0.95)])
    ref = np.concatenate([rng.uniform(0.1, 0.4, 9950), np.full(50, 0.92)])
    x, y = fit_tone_curve(src, ref)
    assert np.interp(0.95, x, y) > 0.85


def test_tone_curve_of_identical_images_is_identity() -> None:
    rng = np.random.default_rng(0)
    values = rng.uniform(0.0, 1.0, 20000)
    x, y = fit_tone_curve(values, values)
    assert np.max(np.abs(y - x)) < 0.02


# ── Transport ────────────────────────────────────────────────────────────────


def test_sliced_transport_reaches_the_target_distribution() -> None:
    rng = np.random.default_rng(0)
    source = rng.normal(0.0, 0.01, (2000, 2))
    target = rng.normal(0.0, 0.03, (2000, 2)) + np.array([-0.05, 0.02])
    moved = sliced_transport(source, target)
    assert np.allclose(moved.mean(axis=0), target.mean(axis=0), atol=2e-3)
    assert np.allclose(moved.std(axis=0), target.std(axis=0), rtol=0.1)


def test_sliced_transport_keeps_order_along_a_line() -> None:
    source = np.column_stack([np.linspace(-0.01, 0.01, 500), np.zeros(500)])
    target = np.column_stack([np.linspace(-0.05, 0.05, 500), np.zeros(500)])
    moved = sliced_transport(source, target)
    assert np.all(np.diff(moved[:, 0]) > 0)


def test_lattice_position_is_finer_near_grey() -> None:
    pos = lattice_position(np.array([[0.0, 0.0], [0.01, 0.0], [0.30, 0.0], [0.31, 0.0]]))
    centre = (AB_CELLS - 1) / 2
    assert pos[0] == pytest.approx([centre, centre])
    assert pos[1, 0] - pos[0, 0] > 5 * (pos[3, 0] - pos[2, 0])
    assert pos.min() >= 0.0 and pos.max() <= AB_CELLS - 1


# ── Model ────────────────────────────────────────────────────────────────────


def test_model_shapes(green_model: LookTransferModel) -> None:
    assert green_model.method == "look"
    assert green_model.cast.shape == (BANDS, 2)
    assert green_model.field.shape == (BANDS, AB_CELLS, AB_CELLS, 2)
    assert green_model.band_trust.shape == (BANDS,)
    assert np.all(np.diff(green_model.band_lightness) > 0)


def test_fit_is_deterministic() -> None:
    a = fit_look(_neutral_street(), _green_dusk())
    b = fit_look(_neutral_street(), _green_dusk())
    assert np.array_equal(a.field, b.field) and np.array_equal(a.tone_y, b.tone_y)


def test_grid_never_folds(green_model: LookTransferModel) -> None:
    for size in (17, 33):
        assert build_grid(green_model, size).fold_fraction == 0.0


def test_shadows_take_the_reference_cast(green_model: LookTransferModel) -> None:
    src = _neutral_street()
    before = srgb_to_oklab(src.reshape(-1, 3) / 255.0)
    after = _render(green_model, src)
    ref = srgb_to_oklab(_green_dusk().reshape(-1, 3) / 255.0)
    dark = after[:, 0] < 0.35
    ref_dark = ref[:, 0] < 0.35
    assert before[before[:, 0] < 0.35, 1].mean() > -0.01  # the source's shadows are grey
    assert after[dark, 1].mean() == pytest.approx(ref[ref_dark, 1].mean(), abs=0.012)
    assert after[dark, 1].mean() < -0.03  # clearly green


def test_tone_follows_the_reference_but_whites_stay_bright(
    green_model: LookTransferModel,
) -> None:
    src = _neutral_street()
    after = _render(green_model, src)
    ref = srgb_to_oklab(_green_dusk().reshape(-1, 3) / 255.0)
    assert np.median(after[:, 0]) == pytest.approx(np.median(ref[:, 0]), abs=0.03)
    assert np.quantile(after[:, 0], 0.97) > 0.8


def test_identical_images_give_a_near_identity() -> None:
    img = _green_dusk()
    model = fit_look(img, img)
    before = srgb_to_oklab(img.reshape(-1, 3) / 255.0)
    after = _render(model, img)
    assert np.abs(after - before).mean() < 0.01


def test_monochrome_reference_protects_vivid_colours() -> None:
    src = _neutral_street()
    model = fit_look(src, _purple_night())
    assert model.protect_vivid == MONOCHROME_VIVID_PROTECTION
    lab_in = srgb_to_oklab(np.array([_lch_rgb(0.62, 0.20, 25.0)], dtype=np.float64) / 255.0)
    lab_out = srgb_to_oklab(model.apply(oklab_to_srgb(lab_in)))
    hue_in = np.degrees(np.arctan2(lab_in[0, 2], lab_in[0, 1]))
    hue_out = np.degrees(np.arctan2(lab_out[0, 2], lab_out[0, 1]))
    assert abs(hue_out - hue_in) < 25.0  # the red neon stays red
    grey = srgb_to_oklab(model.apply(np.array([[0.3, 0.3, 0.3]])))[0]
    assert np.hypot(grey[1], grey[2]) > 0.04  # greys take the cast
    assert grey[1] > 0.0 and grey[2] < 0.0  # and it is purple


def test_varied_reference_does_not_protect_vivid(green_model: LookTransferModel) -> None:
    assert green_model.protect_vivid == 0.0


def test_separation_zero_leaves_only_the_cast(green_model: LookTransferModel) -> None:
    flat = green_model.with_separation(0.0)
    lab = np.array([[0.3, 0.0, 0.0], [0.3, 0.02, 0.0], [0.3, -0.02, 0.01]])
    shift = flat.chroma_shift(lab)
    assert np.allclose(shift, shift[0], atol=1e-12)
    spread = green_model.chroma_shift(lab)
    assert not np.allclose(spread, spread[0], atol=1e-4)


def test_separation_is_clamped(green_model: LookTransferModel) -> None:
    assert green_model.with_separation(9.0).separation == SEPARATION_RANGE[1]
    assert green_model.with_separation(-1.0).separation == SEPARATION_RANGE[0]
    assert green_model.with_protect_vivid(2.0).protect_vivid == 1.0


def test_reference_weights_change_what_is_read() -> None:
    src, ref = _neutral_street(), _green_dusk()
    lab_r = srgb_to_oklab(ref.reshape(-1, 3) / 255.0)
    no_sky = np.where(lab_r[:, 1] > 0.02, 0.0, 1.0)  # drop the mauve / red pixels
    plain = fit_look(src, ref)
    weighted = fit_look(src, ref, reference_weights=no_sky)
    assert weighted.cast[-3:, 0].mean() < plain.cast[-3:, 0].mean() - 0.005


def test_reference_weights_are_validated() -> None:
    src, ref = _neutral_street(), _green_dusk()
    with pytest.raises(ValueError, match="reference_weights"):
        fit_look(src, ref, reference_weights=np.ones(10))
    with pytest.raises(ValueError, match="reference_weights"):
        fit_look(src, ref, reference_weights=np.zeros(ref.shape[0] * ref.shape[1]))


def test_band_trust_is_a_share(green_model: LookTransferModel) -> None:
    assert green_model.band_trust.min() >= 0.0 and green_model.band_trust.max() <= 1.0


def test_registered_as_a_method() -> None:
    assert "look" in SUPPORTED_METHODS
    model = fit_transfer(_neutral_street(), _green_dusk(), method="look")
    assert isinstance(model, LookTransferModel)
    out = model.apply(np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [0.5, 0.2, 0.7]]))
    assert out.shape == (3, 3) and out.min() >= 0.0 and out.max() <= 1.0


# ── Diagnostics ──────────────────────────────────────────────────────────────


def test_diagnostics_flag_a_grey_source() -> None:
    grey = _noisy(np.full((120, 160, 3), 110, dtype=np.uint8), 4)
    grey[:, 80:] = _noisy(np.full((120, 80, 3), 60, dtype=np.uint8), 5)
    diagnostics = look_diagnostics(fit_look(grey, _green_dusk()), grey)
    assert diagnostics.low_colour
    assert diagnostics.source_colourful < 0.01


def test_diagnostics_are_quiet_on_a_coloured_source(green_model: LookTransferModel) -> None:
    diagnostics = look_diagnostics(green_model, _neutral_street())
    assert not diagnostics.low_colour
    assert diagnostics.source_colourful > 0.03
    assert diagnostics.weak_bands == ()


def test_diagnostics_list_untrusted_bands(green_model: LookTransferModel) -> None:
    from dataclasses import replace

    trust = green_model.band_trust.copy()
    trust[[0, 11]] = 0.1
    diagnostics = look_diagnostics(replace(green_model, band_trust=trust), _neutral_street())
    assert diagnostics.weak_bands == (0, 11)


# ── Zone weights ─────────────────────────────────────────────────────────────


def test_zone_weights_default_is_one() -> None:
    labels = np.array([[1, 2], [3, 3]], dtype=np.int32)
    assert np.array_equal(zone_weights(labels, {}), np.ones(4))


def test_zone_weights_follow_the_labels() -> None:
    labels = np.array([[1, 2], [3, 3]], dtype=np.int32)
    assert np.array_equal(zone_weights(labels, {2: 0.0, 3: 2.0}), [1.0, 0.0, 2.0, 2.0])


def test_zone_weights_rejects_out_of_range() -> None:
    labels = np.ones((2, 2), dtype=np.int32)
    with pytest.raises(ValueError, match="zone weight"):
        zone_weights(labels, {1: 5.0})
    with pytest.raises(ValueError, match="zone weight"):
        zone_weights(labels, {1: -0.1})


def test_labels_match_the_analysis_copy() -> None:
    """Zone labels and fit_look must describe the same pixels."""
    ref = np.repeat(np.repeat(_green_dusk(), 3, axis=0), 3, axis=1)  # larger than the copy
    palette = analyze_zones(ref, 6)
    copy = create_thumbnail(ref, ANALYSIS_MAX_SIZE)
    assert palette.labels.shape == copy.shape[:2]
    weights = zone_weights(palette.labels, {palette.zones[-1].id: 0.0})
    model = fit_look(_neutral_street(), ref, reference_weights=weights)
    assert isinstance(model, LookTransferModel)


# ── Linear chroma map (saturation, colour contrast) ──────────────────────────


def _cloud(seed: int = 0, spread: float = 0.05) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.asarray(rng.normal(0.0, spread, (2000, 2)) + np.array([0.02, -0.01]))


def test_stretch_recovers_a_saturation_gain() -> None:
    points = _cloud()
    centre = points.mean(axis=0)
    stretch, found = fit_stretch(points, 0.35 * (points - centre))
    np.testing.assert_allclose(found, centre)
    np.testing.assert_allclose(stretch, 0.35 * np.eye(2), atol=0.02)


def test_stretch_recovers_a_hue_rotation() -> None:
    points = _cloud(1)
    angle = np.deg2rad(12.0)
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    centred = points - points.mean(axis=0)
    stretch, _ = fit_stretch(points, centred @ rotation.T - centred)
    np.testing.assert_allclose(np.eye(2) + stretch, rotation, atol=0.02)


def test_stretch_is_bounded() -> None:
    points = _cloud(2)
    centred = points - points.mean(axis=0)
    for gain in (8.0, -0.95):
        stretch, _ = fit_stretch(points, gain * centred)
        singular = np.linalg.svd(np.eye(2) + stretch, compute_uv=False)
        assert singular.max() <= STRETCH_LIMITS[1] + 1e-9
        assert singular.min() >= STRETCH_LIMITS[0] - 1e-9


def test_stretch_reads_nothing_from_a_grey_band() -> None:
    points = _cloud(3, spread=0.001)
    stretch, _ = fit_stretch(points, 3.0 * (points - points.mean(axis=0)))
    assert np.abs(stretch).max() < 0.15  # asked for 3.0


def test_model_carries_the_stretch(green_model: LookTransferModel) -> None:
    assert green_model.stretch.shape == (BANDS, 2, 2)
    assert green_model.pivot.shape == (BANDS, 2)


def test_a_saturation_grade_is_recovered() -> None:
    """Reference = the source with more chroma: every colour gains the same share."""
    from stealzalut.devtools.ground_truth import as_picture

    rng = np.random.default_rng(5)
    lab = np.dstack([rng.uniform(0.15, 0.85, (96, 96)), rng.normal(0.0, 0.035, (96, 96, 2))])
    graded = lab * np.array([1.0, 1.35, 1.35])
    model = fit_look(as_picture(lab), as_picture(graded))
    flat, truth = lab.reshape(-1, 3), graded.reshape(-1, 3)
    out = model.apply_lab(flat)
    error = np.hypot(*(out[:, 1:] - truth[:, 1:]).T).mean()
    nothing = np.hypot(*(flat[:, 1:] - truth[:, 1:]).T).mean()
    assert error < 0.15 * nothing
    # A colour more vivid than any in the picture gains chroma too: the lattice
    # alone left it behind
    probe = np.array([[0.5, 0.12, 0.10]])
    gain = np.hypot(*model.apply_lab(probe)[0, 1:]) / np.hypot(*probe[0, 1:])
    assert 1.2 < gain < 1.5


# ── Protection of vivid colours spares the reference's own hue ───────────────


def test_monochrome_protection_remembers_the_reference_hue() -> None:
    model = fit_look(_neutral_street(), _purple_night())
    assert model.protect_hue is not None
    assert abs((model.protect_hue - 298.0 + 180.0) % 360.0 - 180.0) < 15.0


def test_protection_spares_vivid_colours_of_the_reference_hue() -> None:
    """A vivid purple under a purple reference is the look itself: it must follow it."""
    model = fit_look(_neutral_street(), _purple_night())
    assert model.protect_hue is not None
    free = replace(model, protect_vivid=0.0)
    same_hue = oklch_to_oklab(np.array([[0.4, 0.16, model.protect_hue]]))
    other_hue = oklch_to_oklab(np.array([[0.4, 0.16, model.protect_hue + 120.0]]))
    np.testing.assert_allclose(model.chroma_shift(same_hue), free.chroma_shift(same_hue))
    kept = np.hypot(*model.chroma_shift(other_hue)[0]) / np.hypot(*free.chroma_shift(other_hue)[0])
    assert kept == pytest.approx(1.0 - MONOCHROME_VIVID_PROTECTION, abs=0.02)


def test_protection_without_a_hue_covers_every_vivid_colour(
    green_model: LookTransferModel,
) -> None:
    """A protection asked for by hand (recipe) on a varied reference protects all hues."""
    protected = green_model.with_protect_vivid(0.5)
    assert protected.protect_hue is None
    for hue in (20.0, 140.0, 260.0):
        lab = oklch_to_oklab(np.array([[0.4, 0.16, hue]]))
        np.testing.assert_allclose(
            protected.chroma_shift(lab), 0.5 * green_model.chroma_shift(lab), atol=1e-12
        )


def test_a_saturated_copy_of_a_monochrome_picture_is_recovered() -> None:
    """The protection used to withhold exactly the colours such a grade moves most."""
    from stealzalut.devtools.ground_truth import as_picture, bench_image

    source = bench_image("night", _purple_night())
    graded = source.lab * np.array([1.0, 1.25, 1.25])
    model = fit_look(as_picture(source.lab), as_picture(graded))
    assert model.protect_vivid > 0.0  # the reference is monochrome
    out = model.apply_lab(source.flat)
    truth = graded.reshape(-1, 3)
    error = np.hypot(*(out[:, 1:] - truth[:, 1:]).T).mean()
    nothing = np.hypot(*(source.flat[:, 1:] - truth[:, 1:]).T).mean()
    assert error < 0.3 * nothing


# ── Locality: how far a colour may be moved ──────────────────────────────────


def test_no_locality_is_the_plain_model(green_model: LookTransferModel) -> None:
    assert green_model.reach is None
    assert green_model.with_locality(0.0).reach is None
    lab = srgb_to_oklab(np.random.default_rng(0).random((200, 3)))
    np.testing.assert_array_equal(
        green_model.with_locality(0.0).apply_lab(lab), green_model.apply_lab(lab)
    )


def test_locality_sets_the_reach(green_model: LookTransferModel) -> None:
    far, near = REACH_RANGE
    assert green_model.with_locality(1.0).reach == pytest.approx(near)
    assert green_model.with_locality(7.0).reach == pytest.approx(near)  # clamped
    assert far > green_model.with_locality(0.3).reach > green_model.with_locality(0.6).reach > near
    assert green_model.with_locality(0.5).with_locality(0.0).reach is None


def test_reach_bounds_the_displacement(green_model: LookTransferModel) -> None:
    lab = srgb_to_oklab(np.random.default_rng(1).random((3000, 3)))
    plain = green_model.apply_lab(lab) - lab
    assert np.abs(plain[:, 0]).max() > 0.1  # the look itself moves tones a lot
    for locality in (0.4, 1.0):
        local = green_model.with_locality(locality)
        assert local.reach is not None
        moved = local.apply_lab(lab) - lab
        assert np.abs(moved[:, 0]).max() <= TONE_REACH_RATIO * local.reach + 1e-12
        assert np.hypot(moved[:, 1], moved[:, 2]).max() <= local.reach + 1e-12
        # Same direction as the plain look, never further
        assert (np.abs(moved[:, 0]) <= np.abs(plain[:, 0]) + 1e-12).all()
        assert (moved[:, 0] * plain[:, 0] >= -1e-15).all()


def test_small_displacements_pass_almost_unchanged(green_model: LookTransferModel) -> None:
    """A grade moves colours a little: the limit must not eat it."""
    local = replace(green_model, reach=0.2)
    lab = srgb_to_oklab(np.random.default_rng(2).random((3000, 3)))
    plain, kept = green_model.chroma_shift(lab), local.chroma_shift(lab)
    small = np.hypot(plain[:, 0], plain[:, 1]) < 0.03
    assert small.any()
    np.testing.assert_allclose(kept[small], plain[small], atol=3e-4)


@pytest.mark.parametrize("locality", [0.2, 0.6, 1.0])
def test_tone_stays_increasing_with_locality(
    green_model: LookTransferModel, locality: float
) -> None:
    lightness = np.linspace(0.0, 1.0, 513)
    assert (np.diff(green_model.with_locality(locality).tone(lightness)) > 0.0).all()


@pytest.mark.parametrize("locality", [0.3, 1.0])
def test_grid_never_folds_with_locality(green_model: LookTransferModel, locality: float) -> None:
    assert build_grid(green_model.with_locality(locality), 33).fold_fraction == 0.0


def test_locality_keeps_the_picture_closer(green_model: LookTransferModel) -> None:
    src = _neutral_street()
    lab = srgb_to_oklab(src.reshape(-1, 3) / 255.0)
    distance = [
        float(np.linalg.norm(green_model.with_locality(t).apply_lab(lab) - lab, axis=1).mean())
        for t in (0.0, 0.5, 1.0)
    ]
    assert distance[0] > distance[1] > distance[2] > 0.0
