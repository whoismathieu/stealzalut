"""Recipe files (``.szl`` v2): correspondences + global settings as JSON.

One format, three uses: the answer of the user's own AI, a save of the
user's work, and sharing.  A recipe only holds zone ids, weights,
qualitative levels, slider positions and explanations — never colour values.

Reading is tolerant where it costs nothing (free text around the JSON,
words or numbers for strengths, ``"S3"`` or ``3`` for ids) and strict
where a mistake would silently give a wrong result (unknown zone,
negative weight): those raise :class:`RecipeError` naming the faulty link.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field, fields, replace
from typing import Any

from stealzalut.core.correspondence import (
    LEVELS,
    RETOUCH_RANGE,
    CorrespondenceTable,
    Intent,
    Level,
    Link,
    Origin,
    Retouch,
)
from stealzalut.core.finishing import (
    BAND_WIDTH,
    Finishing,
    HueBand,
    Wheel,
    finishing_to_dict,
    normalize_curve,
    with_band,
)
from stealzalut.core.grading import GradingParams
from stealzalut.core.look import FOCUS_RANGE, LOCALITY_RANGE
from stealzalut.core.look import SEPARATION_RANGE as LOOK_SEPARATION_RANGE

RECIPE_FORMAT = "stealzalut.recipe"
RECIPE_VERSION = 2
RECIPE_SUFFIX = ".szl"
#: Transfer modes a recipe can ask for.
RECIPE_MODES = ("zones", "ambiance", "look")

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
_RETOUCH_FIELDS = tuple(f.name for f in fields(Retouch))
_GRADING_FIELDS = tuple(f.name for f in fields(GradingParams))
_FINISHING_FIELDS = tuple(f.name for f in fields(Finishing))
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
    keep_tone: float | None = None
    retouch: Retouch | None = None


@dataclass(frozen=True, slots=True)
class Recipe:
    """A parsed recipe.

    Attributes:
        analysis_id: Analysis the recipe was written for (``None`` if absent).
        grading: Global sliders given by the recipe (others keep their value).
        tau: Zone separation, if given.
        zone_count: Number of zones requested for the analysis, if given.
        finishing: Finishing tools, or ``None`` (``null``: keep the current ones).
    """

    analysis_id: str | None
    name: str = ""
    description: str = ""
    grading: dict[str, float] = field(default_factory=dict)
    tau: float | None = None
    zone_count: int | None = None
    links: tuple[RecipeLink, ...] = ()
    finishing: Finishing | None = None
    #: Finishing settings the recipe actually gives (an AI only gives
    #: saturation / vibrance; the user's wheels and curve must survive).
    finishing_keys: frozenset[str] = frozenset()
    #: "zones", "ambiance" or "look" (``None``: not given, stay in the current mode).
    mode: str | None = None
    #: Look method: separation, if given.
    separation: float | None = None
    #: Look method: protection of vivid source colours, if given.
    protect_vivid: float | None = None
    #: Look method: locality (0 = follow the reference, 1 = stay close to the picture), if given.
    locality: float | None = None
    #: Look method: weight per reference zone id (zones left out weigh 1).
    focus: dict[int, float] = field(default_factory=dict)


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
    mode = data.get("mode")
    if mode is not None and mode not in RECIPE_MODES:
        allowed = " | ".join(RECIPE_MODES)
        raise RecipeError(f"« mode » : « {mode} » inconnu (valeurs possibles : {allowed}).")
    zone_count = data.get("zone_count")
    if zone_count is not None and (not _is_number(zone_count) or int(zone_count) != zone_count):
        raise RecipeError(f"zone_count : {zone_count!r} doit être un nombre entier.")

    grading, tau = _parse_global(data.get("global"))

    separation, protect_vivid, locality, focus = _parse_look(data.get("look"))

    # The look method has no links: whatever an AI adds there is ignored
    raw_links = [] if mode == "look" else data.get("links", [])
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
        finishing=_parse_finishing(data.get("finishing")),
        finishing_keys=_finishing_keys(data.get("finishing")),
        mode=mode,
        separation=separation,
        protect_vivid=protect_vivid,
        locality=locality,
        focus=focus,
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


def _parse_look(
    raw: Any,
) -> tuple[float | None, float | None, float | None, dict[int, float]]:
    """Settings of the look method: separation, vivid protection, locality, zone weights."""
    if raw is None:
        return None, None, None, {}
    if not isinstance(raw, dict):
        raise RecipeError("« look » doit être un objet { … }.")

    def ranged(key: str, bounds: tuple[float, float], message: str) -> float | None:
        value = raw.get(key)
        if value is None:
            return None
        if not _is_number(value) or not bounds[0] <= value <= bounds[1]:
            raise RecipeError(message)
        return float(value)

    separation = ranged(
        "separation",
        LOOK_SEPARATION_RANGE,
        f"« look.separation » doit être un nombre entre 0 et {LOOK_SEPARATION_RANGE[1]:g}.",
    )
    protect_vivid = ranged(
        "protect_vivid", (0.0, 1.0), "« look.protect_vivid » doit être un nombre entre 0 et 1."
    )

    locality = ranged(
        "locality", LOCALITY_RANGE, "« look.locality » doit être un nombre entre 0 et 1."
    )

    raw_focus = raw.get("focus")
    if raw_focus is None:
        return separation, protect_vivid, locality, {}
    if not isinstance(raw_focus, dict):
        raise RecipeError('« look.focus » doit être un objet { "R1": 0, … }.')
    lo, hi = FOCUS_RANGE
    focus: dict[int, float] = {}
    for key, weight in raw_focus.items():
        zone = _zone_id(key, "look.focus", "R")
        if not _is_number(weight) or not lo <= weight <= hi:
            raise RecipeError(
                f"« look.focus » : le poids de {key} doit être un nombre entre 0 et {hi:g}."
            )
        focus[zone] = float(weight)
    return separation, protect_vivid, locality, focus


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

    retouch: Retouch | None = None
    if raw.get("retouch") is not None:
        raw_retouch = raw["retouch"]
        if not isinstance(raw_retouch, dict):
            raise RecipeError(f"{where} : « retouch » doit être un objet {{ … }}.")
        unknown = set(raw_retouch) - set(_RETOUCH_FIELDS)
        if unknown:
            allowed = ", ".join(_RETOUCH_FIELDS)
            raise RecipeError(
                f"{where} : retouche inconnue « {sorted(unknown)[0]} » (possibles : {allowed})."
            )
        retouch = Retouch(
            **{
                k: _number(v, f"{where}, retouch.{k}", *RETOUCH_RANGE)
                for k, v in raw_retouch.items()
                if v is not None
            }
        )

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
        keep_tone=_optional_number(raw, "keep_tone", where),
        retouch=retouch,
    )


def _finishing_keys(raw: Any) -> frozenset[str]:
    if not isinstance(raw, dict):
        return frozenset()
    return frozenset(name for name in _FINISHING_FIELDS if raw.get(name) is not None)


def _parse_finishing(raw: Any) -> Finishing | None:
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise RecipeError("« finishing » doit être un objet { … } ou null.")

    def wheel(key: str) -> Wheel:
        value = raw.get(key) or {}
        if not isinstance(value, dict):
            raise RecipeError(f"finishing.{key} doit être un objet {{ … }}.")
        where = f"finishing.{key}"
        return Wheel(
            hue=_number(value.get("hue", 0.0), f"{where}.hue", -360.0, 360.0) % 360.0,
            amount=_number(value.get("amount", 0.0), f"{where}.amount"),
            lightness=_number(value.get("lightness", 0.0), f"{where}.lightness", -1.0, 1.0),
        )

    curve_raw = raw.get("tone_curve") or [[0.0, 0.0], [1.0, 1.0]]
    if not isinstance(curve_raw, list) or len(curve_raw) < 2:
        raise RecipeError("finishing.tone_curve doit être une liste d'au moins 2 points [x, y].")
    points = []
    for i, point in enumerate(curve_raw, start=1):
        if not isinstance(point, list | tuple) or len(point) != 2:
            raise RecipeError(f"finishing.tone_curve, point n°{i} : attendu [x, y].")
        where = f"finishing.tone_curve, point n°{i}"
        points.append((_number(point[0], where), _number(point[1], where)))

    finishing = Finishing(
        shadows=wheel("shadows"),
        midtones=wheel("midtones"),
        highlights=wheel("highlights"),
        tone_curve=normalize_curve(tuple(points)),
        saturation=_number(raw.get("saturation", 0.0), "finishing.saturation", -1.0, 1.0),
        vibrance=_number(raw.get("vibrance", 0.0), "finishing.vibrance", -1.0, 1.0),
    )
    bands_raw = raw.get("bands") or []
    if not isinstance(bands_raw, list):
        raise RecipeError("finishing.bands doit être une liste.")
    for i, band in enumerate(bands_raw, start=1):
        where = f"finishing.bands n°{i}"
        if not isinstance(band, dict) or "center" not in band:
            raise RecipeError(f"{where} : objet avec « center » attendu.")
        finishing = with_band(
            finishing,
            HueBand(
                center=_number(band["center"], f"{where}.center", 0.0, 360.0),
                hue=_number(band.get("hue", 0.0), f"{where}.hue", -1.0, 1.0),
                saturation=_number(band.get("saturation", 0.0), f"{where}.saturation", -1.0, 1.0),
                lightness=_number(band.get("lightness", 0.0), f"{where}.lightness", -1.0, 1.0),
                width=_number(band.get("width", BAND_WIDTH), f"{where}.width", 5.0, 90.0),
            ),
        )
    return finishing


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
                keep_tone=item.keep_tone or 0.0,
                retouch=item.retouch or Retouch(),
            )
        )
    return table


def merge_finishing(recipe: Recipe, current: Finishing) -> Finishing:
    """*current* with only the finishing settings the recipe gives.

    An AI recipe giving ``{"saturation": 0.15, "vibrance": 0.5}`` keeps the
    user's wheels, curve and colour bands; a recipe exported by the app
    gives everything and replaces everything.
    """
    if recipe.finishing is None or not recipe.finishing_keys:
        return current
    merged = current
    for name in recipe.finishing_keys:
        merged = replace(merged, **{name: getattr(recipe.finishing, name)})
    return merged


def apply_global(recipe: Recipe, params: GradingParams, tau: float) -> tuple[GradingParams, float]:
    """*params* and *tau* with the values the recipe gives."""
    values = {name: recipe.grading.get(name, getattr(params, name)) for name in _GRADING_FIELDS}
    return GradingParams(**values), tau if recipe.tau is None else recipe.tau


#: Below this average colour dose, importing a recipe warns the user.
LOW_COLOUR_DOSE = 0.5


def colour_dose(
    table: CorrespondenceTable,
    params: GradingParams,
    areas: dict[int, float],
) -> float:
    """Average share of the colour transfer a table actually applies.

    Link strength, per-link and global colour strength and mix multiply;
    the result is averaged over unlocked zones, weighted by area (locked
    zones are deliberate and don't count).  The neutral protection is not
    included: it depends on each pixel's saturation.

    Returns:
        A value in ``[0, 1]`` (1 when every unlocked zone gets the full transfer).
    """
    total = weighted = 0.0
    for link in table.links:
        if link.locked:
            continue
        area = areas.get(link.source_zone, 0.0)
        chroma = 1.0 if link.chroma_strength is None else link.chroma_strength
        dose = link.strength * chroma if link.targets else 0.0
        total += area
        weighted += area * dose * params.chroma_strength * params.mix
    return weighted / total if total > 0 else 1.0


def full_dose(
    table: CorrespondenceTable, params: GradingParams
) -> tuple[CorrespondenceTable, GradingParams]:
    """The same correspondences at full dose.

    Every unlocked link that transfers something gets ``strength = 1`` and
    the default colour strength; global colour strength and mix go to 1 and
    the neutral protection to 0.  Links switched off (strength 0), locked
    zones, targets, intents, retouches and the other settings are kept.
    """
    for link in table.links:
        if not link.locked and link.strength > 0.0:
            table = table.with_link(replace(link, strength=1.0, chroma_strength=None))
    full = replace(params, chroma_strength=1.0, mix=1.0, preserve_neutrals=0.0)
    return table, full


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
    finishing: Finishing | None = None,
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
        if link.keep_tone > 0.0:
            item["keep_tone"] = link.keep_tone
        retouch = {
            k: getattr(link.retouch, k) for k in _RETOUCH_FIELDS if getattr(link.retouch, k) != 0.0
        }
        if retouch:
            item["retouch"] = retouch
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
        "mode": "zones",
        "zone_count": zone_count,
        "name": name,
        "description": description,
        "global": global_settings,
        "links": links,
        "finishing": (
            None if finishing is None or finishing.is_identity else finishing_to_dict(finishing)
        ),
    }


def look_focus(recipe: Recipe, reference_ids: tuple[int, ...]) -> tuple[tuple[int, float], ...]:
    """The recipe's zone weights as sorted ``(zone id, weight)`` pairs.

    Zones weighing 1 are left out (it is the default).

    Raises:
        RecipeError: If the recipe names a zone the reference does not have.
    """
    for zone in recipe.focus:
        if zone not in reference_ids:
            raise RecipeError(
                f"« look.focus » : « R{zone} » n'est pas une zone de la référence "
                f"(R1 à R{max(reference_ids)})."
            )
    return tuple(sorted((zone, w) for zone, w in recipe.focus.items() if w != 1.0))


def look_recipe_to_dict(
    params: GradingParams,
    separation: float,
    protect_vivid: float | None,
    focus: tuple[tuple[int, float], ...],
    analysis_id: str | None,
    name: str = "",
    description: str = "",
    finishing: Finishing | None = None,
    locality: float = 0.0,
) -> dict[str, Any]:
    """Serialise the settings of the look method; :func:`parse_recipe` reads them back."""
    look: dict[str, Any] = {"separation": separation}
    if locality > 0.0:
        look["locality"] = locality
    if protect_vivid is not None:
        look["protect_vivid"] = protect_vivid
    if focus:
        look["focus"] = {f"R{zone}": weight for zone, weight in focus}
    data: dict[str, Any] = {"format": RECIPE_FORMAT, "version": RECIPE_VERSION}
    if analysis_id:
        data["analysis_id"] = analysis_id
    data.update(
        {
            "mode": "look",
            "name": name,
            "description": description,
            "look": look,
            "global": {field_name: getattr(params, field_name) for field_name in _GRADING_FIELDS},
            "finishing": (
                None
                if finishing is None or finishing.is_identity
                else finishing_to_dict(finishing)
            ),
        }
    )
    return data


def dump_recipe(data: dict[str, Any]) -> str:
    """JSON text of a recipe (readable, UTF-8 friendly)."""
    return json.dumps(data, ensure_ascii=False, indent=2) + "\n"
