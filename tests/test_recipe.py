"""Tests for stealzalut.core.recipe (.szl v2) and the analysis id."""

from __future__ import annotations

import json
from typing import ClassVar

import numpy as np
import pytest

from stealzalut.core.correspondence import CorrespondenceTable, Intent, Link
from stealzalut.core.grading import GradingParams
from stealzalut.core.palette import analysis_id
from stealzalut.core.recipe import (
    STRENGTH_WORDS,
    RecipeError,
    apply_global,
    apply_recipe,
    colour_dose,
    dump_recipe,
    extract_json,
    full_dose,
    parse_recipe,
    recipe_to_dict,
)

AUTO = CorrespondenceTable.from_matching({1: {1: 1.0}, 2: {2: 1.0}, 3: {2: 0.5, 3: 0.5}})
REFS = (1, 2, 3, 4)

AI_ANSWER = """\
Voici mon analyse :
- S1 : feuillage, S2 : ciel, S3 : peau.

Un premier essai (à ignorer) :
```json
{"format": "stealzalut.recipe", "version": 2, "links": []}
```

Version finale :
```json
{
  "format": "stealzalut.recipe",
  "version": 2,
  "analysis_id": "a3f9c2e1",
  "name": "Forêt vers dunes",
  "description": "Feuillage réchauffé.",
  "global": {"luminance_strength": 0.5, "zone_separation": "soft", "mix": 0.9},
  "links": [
    {"source_zone": 1, "targets": {"4": 0.7, "R2": 0.3}, "strength": "strong",
     "intent": {"warmth": "+small", "saturation": "-small"},
     "rationale": "Le feuillage prend la chaleur des dunes."},
    {"source_zone": "S3", "keep": true, "rationale": "Peau à préserver."}
  ],
  "finishing": null
}
```
J'espère que ça aide !
"""


class TestExtract:
    def test_last_json_block_wins(self) -> None:
        data = json.loads(extract_json(AI_ANSWER))
        assert data["name"] == "Forêt vers dunes"

    def test_bare_json_and_unfenced_text(self) -> None:
        assert extract_json('  {"a": 1}  ') == '{"a": 1}'
        assert extract_json('Réponse : {"a": 1} fin.') == '{"a": 1}'

    def test_no_json(self) -> None:
        with pytest.raises(RecipeError, match="Aucun bloc JSON"):
            extract_json("Je ne sais pas.")


class TestParse:
    def test_ai_answer(self) -> None:
        recipe = parse_recipe(AI_ANSWER)
        assert recipe.analysis_id == "a3f9c2e1"
        assert recipe.name == "Forêt vers dunes"
        assert recipe.grading == {"luminance_strength": 0.5, "mix": 0.9}
        assert recipe.tau == 2.0
        first, second = recipe.links
        assert first.targets == {4: 0.7, 2: 0.3}
        assert first.strength == STRENGTH_WORDS["strong"]
        assert first.intent == Intent(warmth="+small", saturation="-small")
        assert second.source_zone == 3 and second.keep and second.targets is None

    def test_weights_are_renormalised(self) -> None:
        recipe = parse_recipe({"links": [{"source_zone": 1, "targets": {"1": 3, "2": 1}}]})
        assert recipe.links[0].targets == {1: 0.75, 2: 0.25}

    def test_levels_accept_numbers(self) -> None:
        recipe = parse_recipe({"links": [{"source_zone": 1, "intent": {"tint": -2}}]})
        assert recipe.links[0].intent == Intent(tint="-medium")

    @pytest.mark.parametrize(
        ("data", "message"),
        [
            ({"format": "autre"}, "pas une recette"),
            ({"version": 1}, "Version"),
            ({"links": {}}, "liste"),
            ({"links": [{"targets": {"1": 1}}]}, "source_zone"),
            ({"links": [{"source_zone": 1, "targets": {"1": -0.5}}]}, "positif"),
            ({"links": [{"source_zone": 1, "targets": {"1": 0}}]}, "nuls"),
            ({"links": [{"source_zone": 1, "strength": "énorme"}]}, "énorme"),
            ({"links": [{"source_zone": 1, "strength": 1.5}]}, "entre 0 et 1"),
            ({"links": [{"source_zone": 1, "intent": {"contrast": "+small"}}]}, "contrast"),
            ({"links": [{"source_zone": 1, "intent": {"warmth": "+huge"}}]}, "huge"),
            ({"links": [{"source_zone": 1}, {"source_zone": "S1"}]}, "deux fois"),
            ({"global": {"mix": 2}}, "global.mix"),
            ({"global": {"zone_separation": "blurry"}}, "zone_separation"),
            ({"links": [{"source_zone": "X"}]}, "identifiant"),
        ],
    )
    def test_errors_are_explicit(self, data: dict, message: str) -> None:
        with pytest.raises(RecipeError, match=message):
            parse_recipe(data)

    def test_invalid_json_gives_position(self) -> None:
        with pytest.raises(RecipeError, match="ligne"):
            parse_recipe('```json\n{"links": [\n```')


class TestApply:
    def test_ai_answer_applies_over_auto(self) -> None:
        table = apply_recipe(parse_recipe(AI_ANSWER), AUTO, REFS)
        s1 = table.link(1)
        assert s1.targets == {4: 0.7, 2: 0.3} and s1.origin == "ai"
        assert s1.rationale.startswith("Le feuillage")
        assert table.link(3).locked and table.link(3).targets == AUTO.link(3).targets
        assert table.link(2) == AUTO.link(2)  # not mentioned → stays automatic

    def test_unknown_source_zone(self) -> None:
        recipe = parse_recipe({"links": [{"source_zone": 1}, {"source_zone": 9}]})
        with pytest.raises(RecipeError, match=r"Lien n°2 : la zone S9 n'existe pas .*S1 à S3"):
            apply_recipe(recipe, AUTO, REFS)

    def test_unknown_reference_zone(self) -> None:
        recipe = parse_recipe({"links": [{"source_zone": 2, "targets": {"7": 1}}]})
        with pytest.raises(RecipeError, match=r"Lien n°1 \(S2\) : la zone R7 n'existe pas"):
            apply_recipe(recipe, AUTO, REFS)

    def test_apply_global_keeps_unspecified(self) -> None:
        recipe = parse_recipe({"global": {"mix": 0.5}})
        params, tau = apply_global(recipe, GradingParams(skin_protection=0.7), 1.3)
        assert params == GradingParams(skin_protection=0.7, mix=0.5)
        assert tau == 1.3


class TestRoundTrip:
    def test_export_import_is_identical(self) -> None:
        table = AUTO.with_link(
            Link(
                source_zone=1,
                targets={2: 0.2, 4: 0.8},
                strength=0.65,
                luminance_strength=0.3,
                chroma_strength=0.9,
                intent=Intent(lightness="-small", tint="+large"),
                origin="user",
            )
        )
        table = table.with_link(
            Link(source_zone=3, targets={3: 1.0}, locked=True, origin="ai", rationale="Peau")
        )
        params = GradingParams(luminance_strength=0.4, preserve_neutrals=0.5, smoothing=1.25)
        text = dump_recipe(recipe_to_dict(table, params, 1.7, "abcd1234", 6, name="Mon look"))

        recipe = parse_recipe(text)
        assert recipe.analysis_id == "abcd1234" and recipe.zone_count == 6
        assert apply_recipe(recipe, AUTO, REFS, origin="ai") == table
        assert apply_global(recipe, GradingParams(), 1.0) == (params, 1.7)

    def test_file_is_readable_utf8(self) -> None:
        text = dump_recipe(recipe_to_dict(AUTO, GradingParams(), 1.0, "x", 3, name="Été doré"))
        assert "Été doré" in text  # not \\u-escaped


class TestDose:
    AREAS: ClassVar[dict[int, float]] = {1: 0.5, 2: 0.3, 3: 0.2}

    def test_doses_multiply(self) -> None:
        assert colour_dose(AUTO, GradingParams(), self.AREAS) == pytest.approx(1.0)
        # The Tokyo case: medium links, colour 75 %, mix 85 % → about 38 %
        table = AUTO
        for link in AUTO.links:
            table = table.with_link(Link(link.source_zone, link.targets, strength=0.6))
        params = GradingParams(chroma_strength=0.75, mix=0.85)
        assert colour_dose(table, params, self.AREAS) == pytest.approx(0.6 * 0.75 * 0.85)

    def test_locked_zones_do_not_count(self) -> None:
        table = AUTO.with_link(Link(1, {1: 1.0}, strength=0.1, locked=True))
        assert colour_dose(table, GradingParams(), self.AREAS) == pytest.approx(1.0)

    def test_full_dose_keeps_choices(self) -> None:
        table = AUTO.with_link(
            Link(1, {1: 0.7, 2: 0.3}, strength=0.6, chroma_strength=0.5,
                 intent=Intent(warmth="+small"), origin="ai", rationale="x")
        )
        table = table.with_link(Link(2, {2: 1.0}, strength=0.0))  # switched off on purpose
        table = table.with_link(Link(3, {3: 1.0}, strength=0.3, locked=True))
        params = GradingParams(luminance_strength=0.35, chroma_strength=0.75, mix=0.85,
                               preserve_neutrals=0.25, skin_protection=0.6)
        full_table, full_params = full_dose(table, params)

        s1 = full_table.link(1)
        assert s1.strength == 1.0 and s1.chroma_strength is None
        assert s1.targets == {1: 0.7, 2: 0.3} and s1.intent == Intent(warmth="+small")
        assert s1.rationale == "x" and s1.origin == "ai"
        assert full_table.link(2).strength == 0.0  # off stays off
        assert full_table.link(3) == table.link(3)  # locked untouched
        assert full_params == GradingParams(luminance_strength=0.35, skin_protection=0.6)


def test_analysis_id_is_deterministic() -> None:
    rng = np.random.default_rng(0)
    a = rng.integers(0, 255, (120, 160, 3), dtype=np.uint8)
    b = rng.integers(0, 255, (90, 90, 3), dtype=np.uint8)
    first = analysis_id(a, b, 8)
    assert first == analysis_id(a.copy(), b.copy(), 8)
    assert len(first) == 8
    assert first != analysis_id(a, b, 6)
    assert first != analysis_id(b, a, 8)
