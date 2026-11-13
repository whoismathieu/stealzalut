"""Tests for the AI pack of the look method (the AI sees the current render)."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import numpy as np
import pytest

from stealzalut.core.look_pack import (
    LOOK_PACK_FILES,
    build_look_prompt,
    example_recipe,
    gaps_table,
    look_gaps,
    write_look_pack,
)
from stealzalut.core.palette import ZonePalette, analysis_id, analyze_zones
from stealzalut.core.recipe import look_focus, parse_recipe

U8 = np.ndarray


def _image(seed: int, top: tuple[int, int, int], bottom: tuple[int, int, int]) -> U8:
    rng = np.random.default_rng(seed)
    img = np.zeros((96, 128, 3))
    img[:48] = top
    img[48:] = bottom
    img[40:56, 40:90] = (230, 220, 200)
    return np.clip(img + rng.normal(0, 3, img.shape), 0, 255).astype(np.uint8)


@pytest.fixture(scope="module")
def reference() -> U8:
    return _image(1, (20, 70, 40), (120, 40, 110))


@pytest.fixture(scope="module")
def palette(reference: U8) -> ZonePalette:
    return analyze_zones(reference, 4)


@pytest.fixture(scope="module")
def render() -> U8:
    return _image(2, (40, 40, 44), (90, 90, 100))


# ── Gaps ─────────────────────────────────────────────────────────────────────


def test_gaps_measure_both_sides_the_same_way(reference: U8) -> None:
    gaps = look_gaps(reference, reference)
    assert gaps["render"] == gaps["reference"]
    side = gaps["reference"]
    assert set(side) == {
        "lightness_mean", "chroma_mean", "colourful_share", "shadows", "midtones", "highlights",
    }
    assert 0.0 < side["lightness_mean"] < 1.0 and side["colourful_share"] > 0.5


def test_gaps_describe_each_tone_range(reference: U8, render: U8) -> None:
    gaps = look_gaps(render, reference)
    shadows = gaps["reference"]["shadows"]
    assert shadows["hex"].startswith("#") and len(shadows["hex"]) == 7
    assert shadows["ab"][0] < -0.02  # the reference's dark half is green
    assert abs(gaps["render"]["shadows"]["ab"][0]) < 0.01  # the render's is grey
    assert gaps["render"]["chroma_mean"] < gaps["reference"]["chroma_mean"]


def test_gaps_say_when_a_tone_range_is_missing() -> None:
    dark = np.full((40, 40, 3), 20, dtype=np.uint8)
    gaps = look_gaps(dark, dark)
    assert gaps["render"]["highlights"] is None
    assert "presque absents" in gaps_table(gaps)


def test_gaps_table_is_markdown(reference: U8, render: U8) -> None:
    lines = gaps_table(look_gaps(render, reference)).splitlines()
    assert lines[0] == "| Mesure | Rendu actuel | Référence |"
    assert len(lines) == 8  # header, rule, three numbers, three tone ranges
    assert all(line.count("|") == 4 for line in lines)


# ── Prompt ───────────────────────────────────────────────────────────────────


def test_look_prompt_lists_reference_ids_and_current_settings(
    reference: U8, render: U8, palette: ZonePalette
) -> None:
    gaps = look_gaps(render, reference)
    prompt = build_look_prompt(palette, "abcd1234", 1.2, 0.8, ((2, 0.0),), gaps)
    assert f"R1 à R{palette.ids[-1]}" in prompt
    assert "actuellement 1.2" in prompt and "actuellement 0.8" in prompt
    assert "Actuellement : R2 = 0." in prompt
    assert "3_current_render.jpg" in prompt and "abcd1234" in prompt
    assert gaps_table(gaps) in prompt
    assert "{" + "gaps" + "}" not in prompt  # every placeholder is filled


def test_look_prompt_without_focus_says_so(
    reference: U8, render: U8, palette: ZonePalette
) -> None:
    prompt = build_look_prompt(palette, "abcd1234", 1.0, 0.0, (), look_gaps(render, reference))
    assert "Actuellement : aucun." in prompt


def test_look_prompt_example_parses(reference: U8, render: U8, palette: ZonePalette) -> None:
    """What the AI is shown as a model must itself be a valid look recipe."""
    prompt = build_look_prompt(palette, "abcd1234", 1.0, 0.0, (), look_gaps(render, reference))
    recipe = parse_recipe(prompt)  # the last ```json block of the text
    assert recipe.mode == "look" and recipe.analysis_id == "abcd1234"
    assert recipe.separation == 1.2 and recipe.protect_vivid == 0.5
    assert look_focus(recipe, palette.ids) == ((palette.ids[0], 0.0),)
    assert recipe.finishing is not None and recipe.finishing.shadows.amount == 0.3
    assert recipe.finishing_keys == {"vibrance", "shadows"}
    assert parse_recipe(example_recipe("abcd1234", palette.ids[0])) == recipe


# ── Pack ─────────────────────────────────────────────────────────────────────


def test_look_pack_contains_the_render(
    tmp_path: Path, reference: U8, render: U8, palette: ZonePalette
) -> None:
    source = _image(3, (60, 60, 60), (130, 130, 130))
    pair = analysis_id(source, reference, 4)
    folder, prompt = write_look_pack(
        tmp_path, "s_x_r", source, reference, render, palette, pair, 1.1, 0.0, ((1, 2.0),)
    )
    assert folder == tmp_path / "s_x_r_stealzalut_look_pack"
    assert sorted(p.name for p in folder.iterdir()) == sorted(LOOK_PACK_FILES)
    assert (folder / "3_current_render.jpg").stat().st_size > 500
    assert (folder / "PROMPT.md").read_text(encoding="utf-8") == prompt

    analysis = json.loads((folder / "analysis.json").read_text(encoding="utf-8"))
    assert analysis["mode"] == "look" and analysis["analysis_id"] == pair
    assert analysis["look"]["separation"] == 1.1
    assert analysis["look"]["focus"] == {"R1": 2.0}
    assert analysis["look"]["gaps"] == look_gaps(render, reference)
    assert [z["id"] for z in analysis["reference"]["zones"]] == list(palette.ids)

    with zipfile.ZipFile(tmp_path / "s_x_r_stealzalut_look_pack.zip") as archive:
        assert sorted(archive.namelist()) == sorted(LOOK_PACK_FILES)


def test_look_prompt_explains_the_locality(reference: U8, render: U8, palette: ZonePalette) -> None:
    gaps = look_gaps(render, reference)
    prompt = build_look_prompt(palette, "abcd1234", 1.0, 0.0, (), gaps, 0.35)
    assert "`look.locality` (0 à 1 ; actuellement 0.35)" in prompt
    assert "actuellement 0)" in build_look_prompt(palette, "abcd1234", 1.0, 0.5, (), gaps)
