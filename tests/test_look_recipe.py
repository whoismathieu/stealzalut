"""Tests for recipes of the look method (`.szl`, mode "look")."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from stealzalut.core.finishing import Finishing
from stealzalut.core.grading import GradingParams
from stealzalut.core.recipe import (
    RECIPE_MODES,
    RecipeError,
    dump_recipe,
    look_focus,
    look_recipe_to_dict,
    parse_recipe,
)

AI_ANSWER = (
    Path(__file__).resolve().parent / "data/ai_answer_tokyo_green_trimmed.txt"
)


def _look(**look: Any) -> dict[str, Any]:
    return {"format": "stealzalut.recipe", "version": 2, "mode": "look", "look": look}


# ── Format ───────────────────────────────────────────────────────────────────


def test_look_is_a_recipe_mode() -> None:
    assert RECIPE_MODES == ("zones", "ambiance", "look")


def test_look_recipe_roundtrip() -> None:
    params = GradingParams(chroma_strength=0.8, mix=0.9)
    finishing = Finishing(saturation=0.2)
    focus = ((1, 0.0), (5, 2.0))
    data = look_recipe_to_dict(params, 1.2, 0.5, focus, "abcd1234", "Nuit", finishing=finishing)
    recipe = parse_recipe(dump_recipe(data))
    assert recipe.mode == "look"
    assert recipe.analysis_id == "abcd1234" and recipe.name == "Nuit"
    assert recipe.separation == 1.2 and recipe.protect_vivid == 0.5
    assert recipe.focus == {1: 0.0, 5: 2.0}
    assert recipe.grading["chroma_strength"] == 0.8 and recipe.grading["mix"] == 0.9
    assert recipe.finishing is not None and recipe.finishing.saturation == 0.2
    assert look_focus(recipe, (1, 2, 3, 4, 5)) == focus
    assert recipe.links == () and recipe.tau is None


def test_look_recipe_omits_what_is_automatic() -> None:
    data = look_recipe_to_dict(GradingParams(), 1.0, None, (), None)
    assert data["look"] == {"separation": 1.0}
    assert "analysis_id" not in data and data["finishing"] is None
    recipe = parse_recipe(data)
    assert recipe.protect_vivid is None and recipe.focus == {} and recipe.analysis_id is None


def test_look_recipe_ignores_links() -> None:
    data = _look(separation=0.5)
    data["links"] = [{"nonsense": True}, "not even an object"]
    recipe = parse_recipe(data)
    assert recipe.links == () and recipe.separation == 0.5


def test_look_focus_accepts_ids_with_or_without_prefix() -> None:
    recipe = parse_recipe(_look(focus={"R1": 0, "r3": 2.0, "4": 1.0}))
    assert recipe.focus == {1: 0.0, 3: 2.0, 4: 1.0}
    assert look_focus(recipe, (1, 2, 3, 4)) == ((1, 0.0), (3, 2.0))  # weight 1 is the default


@pytest.mark.parametrize(
    ("look", "message"),
    [
        ({"separation": 2.0}, "« look.separation » doit être un nombre entre 0 et 1.5."),
        ({"separation": "fort"}, "« look.separation » doit être un nombre entre 0 et 1.5."),
        ({"protect_vivid": -0.1}, "« look.protect_vivid » doit être un nombre entre 0 et 1."),
        ({"focus": {"R1": 5}}, "« look.focus » : le poids de R1 doit être un nombre entre 0 et 4."),
        ({"focus": [1, 2]}, "« look.focus » doit être un objet"),
        ({"focus": {"ciel": 0}}, "n'est pas un identifiant de zone"),
    ],
)
def test_look_errors_are_explicit(look: dict[str, Any], message: str) -> None:
    with pytest.raises(RecipeError) as error:
        parse_recipe(_look(**look))
    assert message in str(error.value)


def test_look_block_must_be_an_object() -> None:
    data = _look()
    data["look"] = [1]
    with pytest.raises(RecipeError, match="« look » doit être un objet"):
        parse_recipe(data)


def test_look_focus_unknown_zone() -> None:
    recipe = parse_recipe(_look(focus={"R9": 0}))
    with pytest.raises(RecipeError) as error:
        look_focus(recipe, (1, 2, 3))
    assert str(error.value) == (
        "« look.focus » : « R9 » n'est pas une zone de la référence (R1 à R3)."
    )


def test_unknown_mode_lists_look() -> None:
    with pytest.raises(RecipeError, match=r"zones \| ambiance \| look"):
        parse_recipe({"format": "stealzalut.recipe", "version": 2, "mode": "film"})


def test_v2_recipes_still_parse() -> None:
    """A real zones recipe written before the look mode reads exactly as before."""
    recipe = parse_recipe(AI_ANSWER.read_text(encoding="utf-8"))
    assert recipe.mode is None and recipe.analysis_id == "262be4c2"
    assert len(recipe.links) == 8 and recipe.tau == 2.0
    assert recipe.grading["chroma_strength"] == 0.75
    assert recipe.separation is None and recipe.protect_vivid is None and recipe.focus == {}


# ── Locality ─────────────────────────────────────────────────────────────────


def test_look_recipe_carries_the_locality() -> None:
    data = look_recipe_to_dict(GradingParams(), 1.0, None, (), None, locality=0.6)
    assert data["look"] == {"separation": 1.0, "locality": 0.6}
    assert parse_recipe(dump_recipe(data)).locality == 0.6


def test_look_recipe_without_locality_leaves_it_alone() -> None:
    recipe = parse_recipe(look_recipe_to_dict(GradingParams(), 1.0, None, (), None))
    assert recipe.locality is None


@pytest.mark.parametrize("value", [-0.1, 1.5, "fort"])
def test_look_recipe_rejects_a_bad_locality(value: object) -> None:
    data = look_recipe_to_dict(GradingParams(), 1.0, None, (), None)
    data["look"]["locality"] = value
    with pytest.raises(RecipeError, match=r"look\.locality"):
        parse_recipe(data)
