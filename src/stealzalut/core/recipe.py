"""Recipe files (``.szl`` v2): correspondences + global settings as JSON.

One format, three uses: the answer of the user's own AI, a save of the
user's work, and sharing.  A recipe only holds zone ids, weights,
qualitative levels and explanations — never colour values.

Reading is tolerant where it costs nothing (free text around the JSON,
words or numbers for strengths, ``"S3"`` or ``3`` for ids) and strict
where a mistake would silently give a wrong result (unknown zone,
negative weight): those raise :class:`RecipeError` naming the faulty link.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field, fields
from typing import Any

from stealzalut.core.correspondence import LEVELS, CorrespondenceTable, Intent, Level, Link, Origin
from stealzalut.core.grading import GradingParams

RECIPE_FORMAT = "stealzalut.recipe"
RECIPE_VERSION = 2
RECIPE_SUFFIX = ".szl"

STRENGTH_WORDS: dict[str, float] = {
    "off": 0.0,
    "subtle": 0.3,
    "medium": 0.6,
    "strong": 0.85,
    "full": 1.0,
}
SEPARATION_WORDS: dict[str, float] = {"sharp": 0.5, "medium": 1.0, "soft": 2.0}
#: Range accepted for a numeric zone separation (τ).
SEPARATION_RANGE = (0.25, 3.0)

_INTENT_FIELDS = tuple(f.name for f in fields(Intent))
_GRADING_FIELDS = tuple(f.name for f in fields(GradingParams))
_GRADING_RANGES = {name: (0.0, 3.0 if name == "smoothing" else 1.0) for name in _GRADING_FIELDS}
_ORIGINS: tuple[Origin, ...] = ("auto", "user", "ai")

_FENCE = re.compile(r"```[ \t]*([A-Za-z0-9_-]*)[ \t]*\r?\n(.*?)```", re.DOTALL)


class RecipeError(ValueError):
    """A recipe that can't be read or doesn't fit the current zones."""


@dataclass(frozen=True, slots=True)
class RecipeLink:
    """One entry of ``links`` (``None`` = not given: keep the base value)."""

    source_zone: int
    targets: dict[int, float] | None = None
    strength: float | None = None
    luminance_strength: float | None = None
    chroma_strength: float | None = None
    keep: bool = False
    intent: Intent | None = None
    rationale: str = ""
    origin: Origin | None = None


@dataclass(frozen=True, slots=True)
class Recipe:
    """A parsed recipe.

    Attributes:
        analysis_id: Analysis the recipe was written for (``None`` if absent).
        grading: Global sliders given by the recipe (others keep their value).
        tau: Zone separation, if given.
        zone_count: Number of zones requested for the analysis, if given.
    """

    analysis_id: str | None
    name: str = ""
    description: str = ""
    grading: dict[str, float] = field(default_factory=dict)
    tau: float | None = None
    zone_count: int | None = None
    links: tuple[RecipeLink, ...] = ()


# ── Reading ──────────────────────────────────────────────────────────────────


def extract_json(text: str) -> str:
    """The JSON part of a free-text answer.

    The **last** fenced code block containing a JSON object wins (an AI
    reasons first, then concludes); a bare JSON text is accepted too.

    Raises:
        RecipeError: If no JSON object is found.
    """
    blocks = [body.strip() for _, body in _FENCE.findall(text)]
    candidates = [str(b) for b in blocks if b.startswith("{")]
    if candidates:
        return candidates[-1]
    stripped = text.strip()
    if stripped.startswith("{"):
        return stripped
    start, end = stripped.find("{"), stripped.rfind("}")
    if start != -1 and end > start:
        return stripped[start : end + 1]
    raise RecipeError(
        "Aucun bloc JSON trouvé. La réponse doit se terminer par un bloc ```json … ```."
    )


def parse_recipe(source: str | dict[str, Any]) -> Recipe:
    """Read a recipe from JSON text, free text containing it, or a dict.

    Raises:
        RecipeError: With a message naming what is wrong.
    """
    if isinstance(source, str):
        try:
            data = json.loads(extract_json(source))
        except json.JSONDecodeError as exc:
            raise RecipeError(
                f"Le JSON de la recette est invalide (ligne {exc.lineno}, colonne {exc.colno}) : "
                f"{exc.msg}."
            ) from exc
    else:
        data = source
    if not isinstance(data, dict):
        raise RecipeError("La recette doit être un objet JSON { … }.")

    fmt = data.get("format", RECIPE_FORMAT)
    if fmt != RECIPE_FORMAT:
        raise RecipeError(f"Ce n'est pas une recette StealZaLut (format « {fmt} »).")
    version = data.get("version", RECIPE_VERSION)
    if version != RECIPE_VERSION:
        raise RecipeError(
            f"Version de recette {version!r} non prise en charge (attendu : {RECIPE_VERSION})."
        )

    analysis = data.get("analysis_id")
    zone_count = data.get("zone_count")
    if zone_count is not None and (not _is_number(zone_count) or int(zone_count) != zone_count):
        raise RecipeError(f"zone_count : {zone_count!r} doit être un nombre entier.")

    grading, tau = _parse_global(data.get("global"))

    raw_links = data.get("links", [])
    if not isinstance(raw_links, list):
        raise RecipeError("« links » doit être une liste.")
    links = tuple(_parse_link(item, i) for i, item in enumerate(raw_links, start=1))
    seen: set[int] = set()
    for i, link in enumerate(links, start=1):
        if link.source_zone in seen:
            raise RecipeError(f"Lien n°{i} : la zone S{link.source_zone} apparaît deux fois.")
        seen.add(link.source_zone)

    return Recipe(
        analysis_id=None if analysis is None else str(analysis),
        name=str(data.get("name") or ""),
        description=str(data.get("description") or ""),
        grading=grading,
        tau=tau,
        zone_count=None if zone_count is None else int(zone_count),
        links=links,
    )


def _is_number(value: Any) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool) and math.isfinite(value)


def _number(value: Any, where: str, lo: float = 0.0, hi: float = 1.0) -> float:
    if not _is_number(value) or not lo <= value <= hi:
        raise RecipeError(f"{where} : {value!r} doit être un nombre entre {lo:g} et {hi:g}.")
    return float(value)


def _strength(value: Any, where: str) -> float:
    if isinstance(value, str):
        word = value.strip().lower()
        if word not in STRENGTH_WORDS:
            allowed = " | ".join(STRENGTH_WORDS)
            raise RecipeError(f"{where} : « {value} » inconnu (valeurs possibles : {allowed}).")
        return STRENGTH_WORDS[word]
    return _number(value, where)


def _separation(value: Any) -> float:
    if isinstance(value, str):
        word = value.strip().lower()
        if word not in SEPARATION_WORDS:
            allowed = " | ".join(SEPARATION_WORDS)
            raise RecipeError(
                f"global.zone_separation : « {value} » inconnu (valeurs possibles : {allowed})."
            )
        return SEPARATION_WORDS[word]
    return _number(value, "global.zone_separation", *SEPARATION_RANGE)


def _zone_id(value: Any, where: str, prefix: str) -> int:
    if isinstance(value, str):
        text = value.strip().upper()
        if text.startswith(prefix):
            text = text[1:]
        if text.isdigit():
            return int(text)
    elif isinstance(value, int) and not isinstance(value, bool) and value > 0:
        return value
    raise RecipeError(
        f"{where} : « {value} » n'est pas un identifiant de zone ({prefix}1, {prefix}2…)."
    )


def _level(value: Any, where: str) -> Level:
    if _is_number(value) and int(value) == value and -3 <= value <= 3:
        return LEVELS[int(value) + 3]
    text = str(value).strip().lower()
    if text in ("0", "+0", "-0", "none", "neutral"):
        return "0"
    for level in LEVELS:
        if text == level:
            return level
    allowed = " | ".join(LEVELS)
    raise RecipeError(f"{where} : « {value} » inconnu (valeurs possibles : {allowed}).")


def _parse_global(raw: Any) -> tuple[dict[str, float], float | None]:
    if raw is None:
        return {}, None
    if not isinstance(raw, dict):
        raise RecipeError("« global » doit être un objet { … }.")
    grading = {
        name: _number(raw[name], f"global.{name}", *_GRADING_RANGES[name])
        for name in _GRADING_FIELDS
        if raw.get(name) is not None
    }
    separation = raw.get("zone_separation")
    return grading, None if separation is None else _separation(separation)


def _parse_link(raw: Any, index: int) -> RecipeLink:
    if not isinstance(raw, dict):
        raise RecipeError(f"Lien n°{index} : doit être un objet {{ … }}.")
    if "source_zone" not in raw:
        raise RecipeError(f"Lien n°{index} : « source_zone » manquant.")
    zone = _zone_id(raw["source_zone"], f"Lien n°{index}", "S")
    where = f"Lien n°{index} (S{zone})"

    targets: dict[int, float] | None = None
    if raw.get("targets") is not None:
        raw_targets = raw["targets"]
        if not isinstance(raw_targets, dict) or not raw_targets:
            raise RecipeError(f"{where} : « targets » doit être un objet non vide {{\"R2\": 0.7}}.")
        parsed: dict[int, float] = {}
        for key, weight in raw_targets.items():
            ref = _zone_id(key, f"{where}, cible", "R")
            if not _is_number(weight) or weight < 0:
                raise RecipeError(f"{where} : le poids de R{ref} doit être un nombre positif.")
            parsed[ref] = parsed.get(ref, 0.0) + float(weight)
        total = sum(parsed.values())
        if total <= 0:
            raise RecipeError(f"{where} : les poids des cibles sont tous nuls.")
        targets = parsed if abs(total - 1.0) < 1e-9 else {k: w / total for k, w in parsed.items()}

    intent: Intent | None = None
    if raw.get("intent") is not None:
        raw_intent = raw["intent"]
        if not isinstance(raw_intent, dict):
            raise RecipeError(f"{where} : « intent » doit être un objet {{ … }}.")
        unknown = set(raw_intent) - set(_INTENT_FIELDS)
        if unknown:
            allowed = ", ".join(_INTENT_FIELDS)
            raise RecipeError(
                f"{where} : intention inconnue « {sorted(unknown)[0]} » (possibles : {allowed})."
            )
        levels = {k: _level(v, f"{where}, intent.{k}") for k, v in raw_intent.items()}
        intent = Intent(**levels)

    origin = raw.get("origin")
    return RecipeLink(
        source_zone=zone,
        targets=targets,
        strength=None if raw.get("strength") is None else _strength(raw["strength"], where),
        luminance_strength=_optional_number(raw, "luminance_strength", where),
        chroma_strength=_optional_number(raw, "chroma_strength", where),
        keep=bool(raw.get("keep", False)),
        intent=intent,
        rationale=str(raw.get("rationale") or ""),
        origin=origin if origin in _ORIGINS else None,
    )


def _optional_number(raw: dict[str, Any], key: str, where: str) -> float | None:
    value = raw.get(key)
    return None if value is None else _number(value, f"{where}, {key}")


# ── Applying ─────────────────────────────────────────────────────────────────


def apply_recipe(
    recipe: Recipe,
    base: CorrespondenceTable,
    reference_ids: tuple[int, ...],
    origin: Origin = "ai",
) -> CorrespondenceTable:
    """The table described by *recipe*, starting from *base*.

    Source zones the recipe doesn't mention keep their *base* link
    (typically the automatic one); a link only sets what it gives.

    Args:
        origin: Origin given to the recipe's links (unless they state one).

    Raises:
        RecipeError: If a link names a zone that doesn't exist.
    """
    source_ids = tuple(link.source_zone for link in base.links)
    refs = set(reference_ids)
    table = base
    for i, item in enumerate(recipe.links, start=1):
        if item.source_zone not in source_ids:
            raise RecipeError(
                f"Lien n°{i} : la zone S{item.source_zone} n'existe pas "
                f"(la source a les zones {_id_range('S', source_ids)})."
            )
        missing = sorted(set(item.targets or {}) - refs)
        if missing:
            raise RecipeError(
                f"Lien n°{i} (S{item.source_zone}) : la zone R{missing[0]} n'existe pas "
                f"(la référence a les zones {_id_range('R', reference_ids)})."
            )
        current = base.link(item.source_zone)
        table = table.with_link(
            Link(
                source_zone=item.source_zone,
                targets=dict(item.targets) if item.targets is not None else dict(current.targets),
                strength=current.strength if item.strength is None else item.strength,
                luminance_strength=item.luminance_strength,
                chroma_strength=item.chroma_strength,
                locked=item.keep,
                intent=item.intent or Intent(),
                origin=item.origin or origin,
                rationale=item.rationale,
            )
        )
    return table


def apply_global(recipe: Recipe, params: GradingParams, tau: float) -> tuple[GradingParams, float]:
    """*params* and *tau* with the values the recipe gives."""
    values = {name: recipe.grading.get(name, getattr(params, name)) for name in _GRADING_FIELDS}
    return GradingParams(**values), tau if recipe.tau is None else recipe.tau


def _id_range(prefix: str, ids: tuple[int, ...]) -> str:
    if not ids:
        return "aucune"
    return f"{prefix}{min(ids)} à {prefix}{max(ids)}"


# ── Writing ──────────────────────────────────────────────────────────────────


def recipe_to_dict(
    table: CorrespondenceTable,
    params: GradingParams,
    tau: float,
    analysis_id: str,
    zone_count: int,
    name: str = "",
    description: str = "",
) -> dict[str, Any]:
    """Serialise the current state; :func:`parse_recipe` + :func:`apply_recipe`
    read it back to the same table and settings."""
    links: list[dict[str, Any]] = []
    for link in table.links:
        item: dict[str, Any] = {
            "source_zone": link.source_zone,
            "targets": {str(k): w for k, w in sorted(link.targets.items())},
            "strength": link.strength,
        }
        if link.luminance_strength is not None:
            item["luminance_strength"] = link.luminance_strength
        if link.chroma_strength is not None:
            item["chroma_strength"] = link.chroma_strength
        if link.locked:
            item["keep"] = True
        intent = {
            k: getattr(link.intent, k) for k in _INTENT_FIELDS if getattr(link.intent, k) != "0"
        }
        if intent:
            item["intent"] = intent
        if link.rationale:
            item["rationale"] = link.rationale
        item["origin"] = link.origin
        links.append(item)

    global_settings: dict[str, Any] = {name: getattr(params, name) for name in _GRADING_FIELDS}
    global_settings["zone_separation"] = tau
    return {
        "format": RECIPE_FORMAT,
        "version": RECIPE_VERSION,
        "analysis_id": analysis_id,
        "zone_count": zone_count,
        "name": name,
        "description": description,
        "global": global_settings,
        "links": links,
        "finishing": None,
    }


def dump_recipe(data: dict[str, Any]) -> str:
    """JSON text of a recipe (readable, UTF-8 friendly)."""
    return json.dumps(data, ensure_ascii=False, indent=2) + "\n"
