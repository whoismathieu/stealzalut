"""French wording for zones, links and intents (shared by the editor widgets)."""

from __future__ import annotations

from stealzalut.core.correspondence import LEVELS, Intent, Level, Link, Origin
from stealzalut.core.finishing import BANDS
from stealzalut.core.palette import Zone

#: (Intent field, label, wording for "+", wording for "-")
INTENT_KINDS: tuple[tuple[str, str, str, str], ...] = (
    ("lightness", "Luminosité", "Plus clair", "Plus sombre"),
    ("saturation", "Saturation", "Plus saturé", "Moins saturé"),
    ("warmth", "Température", "Plus chaud", "Plus froid"),
    ("tint", "Teinte", "Plus magenta", "Plus vert"),
)

_LEVEL_WORDS = {"small": "léger", "medium": "moyen", "large": "fort"}

ORIGIN_WORDS: dict[Origin, str] = {
    "auto": "automatiquement",
    "user": "par toi",
    "ai": "par l'IA",
}


def level_from_step(step: int) -> Level:
    """Slider step in ``[-3, 3]`` → intent level."""
    return LEVELS[max(-3, min(3, step)) + 3]


def step_from_level(level: Level) -> int:
    return LEVELS.index(level) - 3


def describe_level(kind: str, level: Level) -> str:
    if level == "0":
        return "Neutre"
    _, _, up, down = next(k for k in INTENT_KINDS if k[0] == kind)
    return f"{up if level[0] == '+' else down} ({_LEVEL_WORDS[level[1:]]})"


def describe_intent(intent: Intent) -> str:
    """``"plus chaud (léger), moins saturé (fort)"``, or ``""`` when neutral."""
    parts = [
        describe_level(kind, getattr(intent, kind)).lower()
        for kind, *_ in INTENT_KINDS
        if getattr(intent, kind) != "0"
    ]
    return ", ".join(parts)


def pct(fraction: float) -> str:
    return f"{round(fraction * 100)} %"


def targets_text(link: Link) -> str:
    if not link.targets:
        return "aucune (zone gardée telle quelle)"
    ordered = sorted(link.targets.items(), key=lambda kv: -kv[1])
    return ", ".join(f"R{ref} ({pct(w)})" for ref, w in ordered)


def source_tooltip(zone: Zone, link: Link) -> str:
    lines = [
        f"S{zone.id} — {pct(zone.area)} de l'image",
        f"Cible : {targets_text(link)}",
        f"Réglée {ORIGIN_WORDS[link.origin]}",
    ]
    if link.locked:
        lines.append("Verrouillée : cette zone n'est pas modifiée")
    elif link.strength < 1.0:
        lines.append(f"Force : {pct(link.strength)}")
    intent = describe_intent(link.intent)
    if intent:
        lines.append(f"Intentions : {intent}")
    if link.rationale:
        lines.append(f"Note : {link.rationale}")
    lines.append("")
    lines.append("Glisse vers une zone de droite pour changer sa cible (Maj : en ajouter une).")
    lines.append("Clic : régler · Clic droit : plus d'options")
    return "\n".join(lines)


def reference_tooltip(zone: Zone, used_by: list[int]) -> str:
    users = ", ".join(f"S{z}" for z in used_by) if used_by else "aucune zone source"
    return f"R{zone.id} — {pct(zone.area)} de la référence\nUtilisée par : {users}"


_COLOUR_ADJECTIVES = {
    "Rouges": "rouge",
    "Oranges": "orange",
    "Jaunes": "jaune",
    "Verts": "verte",
    "Cyans": "cyan",
    "Bleus": "bleue",
    "Violets": "violette",
    "Magentas": "magenta",
}


def dominant_colour_name(hue: float) -> str:
    """``"dominante violette"`` for an Oklch hue in degrees."""

    def distance(center: float) -> float:
        return abs((hue - center + 180.0) % 360.0 - 180.0)

    name = min(BANDS, key=lambda band: distance(band[1]))[0]
    return f"dominante {_COLOUR_ADJECTIVES[name]}"
