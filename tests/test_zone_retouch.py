"""Tests for the manual retouch of a zone: it must do what the sliders say."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from stealzalut.core.ai_pack import (
    RENDER_FILE,
    analysis_dict,
    build_prompt,
    measures_table,
    write_pack,
    zone_measures,
)
from stealzalut.core.colorspace import (
    gamut_compress,
    oklab_to_oklch,
    oklab_to_srgb,
    srgb_to_oklab,
)
from stealzalut.core.correspondence import Intent, Link, Retouch
from stealzalut.core.grading import GradingParams, build_grid, skin_mask
from stealzalut.core.lut_apply import apply_lut_colors
from stealzalut.core.recipe import (
    RecipeError,
    apply_recipe,
    dump_recipe,
    parse_recipe,
    recipe_to_dict,
)
from stealzalut.core.zone_transfer import (
    EXPOSURE_GAMMA,
    RETOUCH_EXPOSURE,
    ToneGlobalModel,
    ZoneTransferModel,
    _exposure_lift,
    fit_zone_transfer,
    retouch_from_intent,
    zone_retouch,
)

#: A dark brown skin (Oklab L ≈ 0.25).
SKIN = (50, 32, 24)
SIZE = 33


def _scene(seed: int = 0) -> np.ndarray:
    """A bright wall, foliage, a sky, and a dark skin covering 3 % of the picture."""
    rng = np.random.default_rng(seed)
    image = np.zeros((100, 200, 3), dtype=np.float64)
    image[:, :80] = (200, 190, 170)
    image[:, 80:140] = (40, 90, 45)
    image[:, 140:] = (110, 160, 225)
    image[40:70, 90:110] = SKIN  # 30 × 20 px of 20 000 = 3 %
    image += rng.normal(0.0, 3.0, image.shape)
    return np.clip(image, 0, 255).astype(np.uint8)


def _dark_reference() -> np.ndarray:
    rng = np.random.default_rng(1)
    image = np.zeros((100, 200, 3), dtype=np.float64)
    image[:, :90] = (5, 9, 11)
    image[:, 90:150] = (90, 60, 30)
    image[:, 150:] = (200, 110, 50)
    image += rng.normal(0.0, 3.0, image.shape)
    return np.clip(image, 0, 255).astype(np.uint8)


@pytest.fixture(scope="module")
def model() -> ZoneTransferModel:
    return fit_zone_transfer(_scene(), _dark_reference(), zone_count=4)


def _skin_lab() -> np.ndarray:
    return srgb_to_oklab(np.array([SKIN], dtype=np.float64) / 255.0)


def _skin_zone(model: ZoneTransferModel) -> int:
    lab = _skin_lab()[0]
    return min(model.source.zones, key=lambda z: float(np.linalg.norm(z.mean - lab))).id


def _with(model: ZoneTransferModel, zone: int, **changes: object) -> ZoneTransferModel:
    link = replace(model.table.link(zone), origin="user", **changes)  # type: ignore[arg-type]
    return model.with_table(model.table.with_link(link))


def _through_lut(model: ZoneTransferModel, params: GradingParams | None = None) -> np.ndarray:
    """Oklab colour of the skin after the baked LUT."""
    result = build_grid(model, SIZE, params)
    rgb = np.array([SKIN], dtype=np.float64) / 255.0
    return srgb_to_oklab(apply_lut_colors(rgb, result.grid, SIZE))[0]


# ── The scene is what the tests say it is ────────────────────────────────────


def test_the_skin_is_a_small_dark_zone(model: ZoneTransferModel) -> None:
    zone = model.source.zone(_skin_zone(model))
    assert zone.area == pytest.approx(0.03, abs=0.01)
    assert zone.mean[0] == pytest.approx(0.25, abs=0.03)


# ── Exposure ─────────────────────────────────────────────────────────────────


def test_exposure_lifts_a_dark_skin_through_the_lut(model: ZoneTransferModel) -> None:
    """At least +0.12 of lightness, whatever the global adjustments say."""
    zone = _skin_zone(model)
    lifted = _with(model, zone, retouch=Retouch(exposure=1.0))
    for params in (
        None,
        GradingParams(skin_protection=1.0),
        GradingParams(luminance_strength=0.3),
        GradingParams(skin_protection=1.0, preserve_neutrals=1.0, mix=0.4),
    ):
        gain = _through_lut(lifted, params)[0] - _through_lut(model, params)[0]
        assert gain >= 0.12, (params, gain)


def test_retouch_does_not_trigger_automatic_smoothing(model: ZoneTransferModel) -> None:
    """The retouch comes after the fold control: it is not blurred into the
    neighbouring zones, however strong it is."""
    plain = build_grid(model, SIZE)
    zone = _skin_zone(model)
    lifted = build_grid(_with(model, zone, retouch=Retouch(exposure=1.0), keep_tone=1.0), SIZE)
    assert lifted.auto_smoothed == plain.auto_smoothed
    assert lifted.smoothing_used == plain.smoothing_used


def test_exposure_darkens_too(model: ZoneTransferModel) -> None:
    zone = _skin_zone(model)
    base = _through_lut(_with(model, zone, keep_tone=1.0))
    darker = _through_lut(_with(model, zone, keep_tone=1.0, retouch=Retouch(exposure=-1.0)))
    assert darker[0] < base[0] - 0.05


@pytest.mark.parametrize("exposure", [-1.0, -0.4, 0.3, 1.0])
def test_exposure_keeps_the_order_of_tones(exposure: float) -> None:
    lightness = np.linspace(0.0, 1.0, 501)
    out = lightness + _exposure_lift(lightness, exposure)
    assert np.all(np.diff(out) > 0.0)
    assert out[0] == pytest.approx(0.0) and out[-1] == pytest.approx(1.0)
    assert np.abs(out - lightness).max() == pytest.approx(RETOUCH_EXPOSURE * abs(exposure), rel=0.6)


def test_full_exposure_reaches_its_stated_range() -> None:
    lightness = np.linspace(0.0, 1.0, 2001)
    assert _exposure_lift(lightness, 1.0).max() == pytest.approx(RETOUCH_EXPOSURE, abs=0.002)
    assert _exposure_lift(lightness, -1.0).min() == pytest.approx(-RETOUCH_EXPOSURE, abs=0.002)
    assert 0.0 < EXPOSURE_GAMMA < 1.0


def test_a_lifted_dark_colour_gets_chroma_back() -> None:
    lab = _skin_lab()
    link = Link(source_zone=1, retouch=Retouch(exposure=1.0))
    moved = lab + zone_retouch(lab, lab, link)
    assert moved[0, 0] > lab[0, 0] + 0.15
    assert oklab_to_oklch(moved)[0, 1] > oklab_to_oklch(lab)[0, 1]
    # A bright colour is lifted without any chroma change
    bright = np.array([[0.7, 0.03, 0.04]])
    moved = bright + zone_retouch(bright, bright, link)
    np.testing.assert_allclose(moved[0, 1:], bright[0, 1:], atol=1e-12)


# ── Keeping the zone's own lightness ─────────────────────────────────────────


def test_keep_tone_brings_the_source_lightness_back(model: ZoneTransferModel) -> None:
    zone = _skin_zone(model)
    source_l = _skin_lab()[0, 0]
    darkened = _through_lut(model)[0]
    assert darkened < source_l - 0.02  # the dark reference darkens the skin
    kept = _through_lut(_with(model, zone, keep_tone=1.0))[0]
    half = _through_lut(_with(model, zone, keep_tone=0.5))[0]
    assert abs(kept - source_l) < 0.35 * abs(darkened - source_l)
    assert darkened < half < kept


def test_keep_tone_leaves_the_colour_to_the_transfer() -> None:
    lab_in = np.array([[0.30, 0.02, 0.03]])
    lab_now = np.array([[0.15, 0.05, -0.02]])
    moved = lab_now + zone_retouch(lab_in, lab_now, Link(source_zone=1, keep_tone=1.0))
    np.testing.assert_allclose(moved[0], [0.30, 0.05, -0.02], atol=1e-12)


# ── A small zone answers every slider ────────────────────────────────────────


def test_a_small_zone_follows_every_slider(model: ZoneTransferModel) -> None:
    """The zone covers 3 % of the picture and still moves as the sliders say."""
    zone = _skin_zone(model)
    lab = _skin_lab()
    base = model.apply_lab(lab)[0]

    def moved(**retouch: float) -> np.ndarray:
        return _with(model, zone, retouch=Retouch(**retouch)).apply_lab(lab)[0]

    chroma = float(np.hypot(base[1], base[2]))
    more, less = moved(saturation=1.0), moved(saturation=-1.0)
    assert np.hypot(more[1], more[2]) > 1.6 * chroma
    assert np.hypot(less[1], less[2]) < 0.45 * chroma
    assert moved(warmth=1.0)[2] > base[2] + 0.06  # b: towards yellow
    assert moved(warmth=-1.0)[2] < base[2] - 0.06
    assert moved(tint=1.0)[1] > base[1] + 0.045  # a: towards magenta
    assert moved(tint=-1.0)[1] < base[1] - 0.045
    assert moved(exposure=1.0)[0] > base[0] + 0.12


@pytest.mark.parametrize(
    "retouch",
    [Retouch(saturation=-1.0), Retouch(warmth=-1.0), Retouch(tint=1.0), Retouch(exposure=0.6)],
)
def test_the_lut_carries_the_retouch(model: ZoneTransferModel, retouch: Retouch) -> None:
    """What the engine computes for the zone is what the baked grid gives,
    up to the colours a display can show (a dark colour holds little chroma)."""
    zone = _skin_zone(model)
    edited = _with(model, zone, retouch=retouch)
    engine = edited.apply_lab(_skin_lab())
    shown = srgb_to_oklab(oklab_to_srgb(gamut_compress(engine)))[0]
    np.testing.assert_allclose(_through_lut(edited), shown, atol=0.012)
    assert np.linalg.norm(_through_lut(edited) - _through_lut(model)) > 0.015


def test_retouch_weighs_a_small_zone_frankly(model: ZoneTransferModel) -> None:
    """Without the area prior, the zone owns its own colours."""
    zone = _skin_zone(model)
    index = 1 + [z.id for z in model.source.zones].index(zone)
    assert model._weights(_skin_lab(), area_prior=False)[0, index] > 0.9


def test_other_zones_are_left_alone(model: ZoneTransferModel) -> None:
    zone = _skin_zone(model)
    retouched = _with(model, zone, retouch=Retouch(exposure=1.0, warmth=1.0))
    for other in model.source.zones:
        if other.id == zone or abs(other.mean[0] - model.source.zone(zone).mean[0]) < 0.15:
            continue
        lab = other.mean[None, :]
        np.testing.assert_allclose(retouched.apply_lab(lab), model.apply_lab(lab), atol=2e-3)


# ── What the global adjustments do not touch ─────────────────────────────────


def test_retouch_is_not_dosed_by_the_global_adjustments(model: ZoneTransferModel) -> None:
    """A zone told to keep its lightness and take a given exposure ends at
    the same lightness whatever the global sliders do to the transfer."""
    zone = _skin_zone(model)
    set_by_hand = _with(model, zone, keep_tone=1.0, retouch=Retouch(exposure=0.5))
    results = [
        _through_lut(set_by_hand, params)[0]
        for params in (
            None,
            GradingParams(mix=0.2),
            GradingParams(luminance_strength=0.0),
            GradingParams(skin_protection=1.0, preserve_neutrals=1.0),
        )
    ]
    assert max(results) - min(results) < 0.02
    assert min(results) > _skin_lab()[0, 0] + 0.08


def test_no_retouch_changes_nothing(model: ZoneTransferModel) -> None:
    lab = srgb_to_oklab(np.random.default_rng(0).random((200, 3)))
    assert model.retouch_lab(lab, lab) is None
    np.testing.assert_array_equal(model.apply_lab(lab), model.transfer_lab(lab))


def test_a_locked_zone_ignores_its_retouch(model: ZoneTransferModel) -> None:
    zone = _skin_zone(model)
    locked = _with(model, zone, locked=True, retouch=Retouch(exposure=1.0), keep_tone=1.0)
    assert not locked.table.link(zone).is_retouched
    assert locked.retouch_lab(_skin_lab(), _skin_lab()) is None


# ── Levels of older recipes ──────────────────────────────────────────────────


def test_levels_keep_their_effect_and_add_up() -> None:
    lab = np.array([[0.5, 0.02, 0.03]])
    level = Link(source_zone=1, intent=Intent(lightness="+large", tint="+medium"))
    moved = zone_retouch(lab, lab, level)[0]
    assert moved[0] == pytest.approx(0.10) and moved[1] == pytest.approx(0.016)
    both = replace(level, retouch=Retouch(tint=0.5))
    assert zone_retouch(lab, lab, both)[0, 1] == pytest.approx(0.016 + 0.03)


def test_slider_position_of_a_level() -> None:
    position = retouch_from_intent(
        Intent(lightness="-large", saturation="+large", warmth="+large", tint="-large")
    )
    assert position.exposure == pytest.approx(-0.10 / RETOUCH_EXPOSURE)
    assert position.saturation == pytest.approx(0.5)  # ×1.4 of a ×1.8 travel
    assert position.warmth == pytest.approx(0.5) and position.tint == pytest.approx(-0.5)
    assert retouch_from_intent(Intent()) == Retouch()


# ── Global tone of the zones method ──────────────────────────────────────────


def test_global_tone_is_a_monotone_curve(model: ZoneTransferModel) -> None:
    tone = model.global_model
    assert isinstance(tone, ToneGlobalModel) and model.fold_reference() is tone
    assert np.all(np.diff(tone.tone_y) >= 0.0)
    grey = np.column_stack([np.linspace(-0.05, 1.05, 200), np.zeros(200), np.zeros(200)])
    assert np.all(np.diff(tone.apply_lab(grey)[:, 0]) >= 0.0)
    assert build_grid(model, SIZE).fold_fraction == 0.0


def test_dark_tones_are_not_sent_to_black(model: ZoneTransferModel) -> None:
    """The straight line of the Oklab transfer sends the low mid-tones under
    the reference's own shadows; the curve cannot."""
    tone = model.global_model
    assert isinstance(tone, ToneGlobalModel)
    dark = np.array([[0.20, 0.0, 0.0]])
    assert tone.apply_lab(dark)[0, 0] > 0.02
    assert tone.apply_lab(dark)[0, 0] >= tone.colour.apply_lab(dark)[0, 0]


# ── Skin window ──────────────────────────────────────────────────────────────


def test_skin_window_covers_dark_skin() -> None:
    dark_skins = np.array([[50, 32, 24], [72, 48, 36], [36, 24, 18]], dtype=np.float64) / 255.0
    weights = skin_mask(oklab_to_oklch(srgb_to_oklab(dark_skins)))
    assert weights[0] > 0.9 and weights[1] > 0.9
    assert weights[2] > 0.3  # a very dark skin is still partly covered
    not_skin = np.array([[20, 60, 160], [128, 128, 128]], dtype=np.float64) / 255.0
    assert skin_mask(oklab_to_oklch(srgb_to_oklab(not_skin))).max() < 0.05


# ── Recipes ──────────────────────────────────────────────────────────────────


def test_recipe_round_trip_keeps_the_retouch(model: ZoneTransferModel) -> None:
    zone = _skin_zone(model)
    edited = _with(
        model, zone, keep_tone=0.8, retouch=Retouch(exposure=0.35, warmth=-0.2)
    ).table
    data = recipe_to_dict(edited, GradingParams(), 1.0, model.analysis_id, 4)
    item = next(link for link in data["links"] if link["source_zone"] == zone)
    assert item["keep_tone"] == 0.8 and item["retouch"] == {"exposure": 0.35, "warmth": -0.2}
    untouched = next(link for link in data["links"] if link["source_zone"] != zone)
    assert "retouch" not in untouched and "keep_tone" not in untouched

    recipe = parse_recipe(dump_recipe(data))
    back = apply_recipe(recipe, model.table, model.reference.ids)
    assert back.link(zone).retouch == Retouch(exposure=0.35, warmth=-0.2)
    assert back.link(zone).keep_tone == 0.8


def test_recipe_accepts_levels_and_numbers_together() -> None:
    recipe = parse_recipe(
        {
            "links": [
                {
                    "source_zone": "S2",
                    "intent": {"warmth": "+small"},
                    "keep_tone": 1,
                    "retouch": {"exposure": 0.5, "tint": None},
                }
            ]
        }
    )
    link = recipe.links[0]
    assert link.intent == Intent(warmth="+small")
    assert link.keep_tone == 1.0 and link.retouch == Retouch(exposure=0.5)


@pytest.mark.parametrize(
    ("link", "word"),
    [
        ({"source_zone": 1, "retouch": {"exposure": 1.5}}, "retouch.exposure"),
        ({"source_zone": 1, "retouch": {"contrast": 0.2}}, "contrast"),
        ({"source_zone": 1, "retouch": 0.5}, "retouch"),
        ({"source_zone": 1, "keep_tone": -0.1}, "keep_tone"),
    ],
)
def test_recipe_rejects_a_wrong_retouch(link: dict[str, object], word: str) -> None:
    with pytest.raises(RecipeError, match=word):
        parse_recipe({"links": [link]})


# ── AI pack ──────────────────────────────────────────────────────────────────


def _render(model: ZoneTransferModel) -> np.ndarray:
    source = _scene()
    flat = source.reshape(-1, 3).astype(np.float64) / 255.0
    out = oklab_to_srgb(model.apply_lab(srgb_to_oklab(flat)))
    return (np.clip(out, 0.0, 1.0) * 255.0).round().astype(np.uint8).reshape(source.shape)


def test_measures_show_the_zone_that_darkened(model: ZoneTransferModel) -> None:
    zone = _skin_zone(model)
    measures = zone_measures(model, model.table, _render(model))
    assert [m["zone"] for m in measures] == list(model.source.ids)
    entry = next(m for m in measures if m["zone"] == zone)
    assert entry["source"]["L"] == pytest.approx(0.25, abs=0.03)
    assert entry["render"]["L"] < entry["source"]["L"] - 0.02
    assert entry["target"] is not None and entry["settings"] == {}

    lifted = _with(model, zone, keep_tone=1.0, retouch=Retouch(exposure=0.5))
    after = next(
        m for m in zone_measures(lifted, lifted.table, _render(lifted)) if m["zone"] == zone
    )
    assert after["render"]["L"] > entry["render"]["L"] + 0.1
    assert after["settings"] == {"keep_tone": 1.0, "retouch": {"exposure": 0.5}}


def test_prompt_shows_the_render_and_the_zone_retouch(model: ZoneTransferModel) -> None:
    plain = build_prompt(model)
    assert RENDER_FILE not in plain and "Rendu actuel" not in plain
    # The retouch settings are always explained and part of the example
    assert '"keep_tone"' in plain and '"retouch"' in plain and '"exposure"' in plain

    measures = zone_measures(model, model.table, _render(model))
    prompt = build_prompt(model, measures)
    assert RENDER_FILE in prompt and "## Rendu actuel" in prompt
    assert measures_table(measures) in prompt
    assert f"| S{_skin_zone(model)} |" in prompt


def test_pack_holds_the_render_and_its_measures(tmp_path: Path, model: ZoneTransferModel) -> None:
    import json

    render = _render(model)
    folder = write_pack(tmp_path, "pair", _scene(), _dark_reference(), model, 4, render)
    assert (folder / RENDER_FILE).exists()
    analysis = json.loads((folder / "analysis.json").read_text(encoding="utf-8"))
    assert len(analysis["current_render"]["zones"]) == len(model.source.zones)
    assert RENDER_FILE in (folder / "PROMPT.md").read_text(encoding="utf-8")

    # Written again without a render, the pack no longer shows a stale one
    write_pack(tmp_path, "pair", _scene(), _dark_reference(), model, 4)
    assert not (folder / RENDER_FILE).exists()
    analysis = json.loads((folder / "analysis.json").read_text(encoding="utf-8"))
    assert "current_render" not in analysis
    assert analysis == analysis_dict(model, (200, 100), (200, 100), 4)
