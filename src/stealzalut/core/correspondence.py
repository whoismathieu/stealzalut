"""The correspondence table: which reference zones colour each source zone.

The table is the single object that the automatic matching, the user (in
the correspondence editor) and the user's own AI (through a recipe file)
all edit.  The bake engine does not care who filled it.

Tables are immutable: every edit returns a new table, which makes
undo/redo a plain stack of tables.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Literal

Level = Literal["-large", "-medium", "-small", "0", "+small", "+medium", "+large"]
Origin = Literal["auto", "user", "ai"]

LEVELS: tuple[Level, ...] = ("-large", "-medium", "-small", "0", "+small", "+medium", "+large")


@dataclass(frozen=True, slots=True)
class Intent:
    """Qualitative nudges applied to a zone after its transfer.

    The app turns levels into numbers (see
    :data:`stealzalut.core.zone_transfer.INTENT_STEPS`); nobody ever has to
    type a colour value.  Levels are the older, coarser form of
    :class:`Retouch`: recipes may still use them, the editor writes a retouch.
    """

    lightness: Level = "0"
    saturation: Level = "0"
    warmth: Level = "0"  # + warmer (yellow/orange), - cooler
    tint: Level = "0"  # + magenta, - green

    @property
    def is_neutral(self) -> bool:
        return self == Intent()


@dataclass(frozen=True, slots=True)
class Retouch:
    """Manual retouch of a zone, applied on top of its transfer.

    Every value is a slider position in ``[-1, 1]`` (0 = nothing); the
    engine turns it into a move (see
    :func:`stealzalut.core.zone_transfer.zone_retouch`).  Unlike the
    transfer, a retouch is not dosed by the global adjustments: what the
    user sets by hand on a zone is what the zone gets.
    """

    exposure: float = 0.0  # + lighter, - darker
    saturation: float = 0.0
    warmth: float = 0.0  # + warmer (yellow/orange), - cooler
    tint: float = 0.0  # + magenta, - green

    @property
    def is_neutral(self) -> bool:
        return self == Retouch()


#: Range of every :class:`Retouch` value.
RETOUCH_RANGE: tuple[float, float] = (-1.0, 1.0)


@dataclass(frozen=True, slots=True)
class Link:
    """How one source zone is treated.

    Attributes:
        source_zone: Source zone id.
        targets: ``{reference_zone_id: weight}`` summing to 1; empty means
            "keep this zone's colours" (its tone still follows the image).
        strength: Dose of the zone's own choice (colour, opt-in target
            lightness): 0 = keeps its colours, 1 = full transfer.  It never
            changes the global tone mapping, which all unlocked zones share.
        luminance_strength: How much the zone takes its target's lightness
            (``None`` / 0 = follow the global tone mapping, 1 = target
            lightness; may fold the LUT if zones swap lightness order).
        chroma_strength: Per-zone chroma strength (``None`` = 1).
        locked: Zone is never changed (no transfer, no intent).
        intent: Qualitative nudges.
        origin: Who set this link.
        rationale: Free-text explanation (shown as a tooltip).
        keep_tone: How much the zone keeps its own lightness instead of
            following the image's tone mapping (0 = follows it, 1 = keeps
            the lightness it has in the source).
        retouch: Continuous manual retouch (exposure, saturation, warmth, tint).
    """

    source_zone: int
    targets: dict[int, float] = field(default_factory=dict)
    strength: float = 1.0
    luminance_strength: float | None = None
    chroma_strength: float | None = None
    locked: bool = False
    intent: Intent = Intent()
    origin: Origin = "auto"
    rationale: str = ""
    keep_tone: float = 0.0
    retouch: Retouch = Retouch()

    def __hash__(self) -> int:  # dict field: hash on identity-stable fields only
        return hash((self.source_zone, tuple(sorted(self.targets.items())), self.strength))

    @property
    def is_identity(self) -> bool:
        """True when this link leaves its zone completely untouched (locked);
        any unlocked zone at least follows the global tone mapping."""
        return self.locked

    @property
    def keeps_colours(self) -> bool:
        """True when the zone keeps its colours (only its tone may change)."""
        return self.locked or (
            (not self.targets or self.strength == 0.0)
            and self.intent.is_neutral
            and self.retouch.is_neutral
        )

    @property
    def is_retouched(self) -> bool:
        """True when the zone carries a manual retouch (a locked zone never does)."""
        return not self.locked and (
            self.keep_tone > 0.0 or not self.retouch.is_neutral or not self.intent.is_neutral
        )


@dataclass(frozen=True, slots=True)
class CorrespondenceTable:
    """Exactly one :class:`Link` per source zone, ordered by zone id."""

    links: tuple[Link, ...]

    @classmethod
    def from_matching(cls, matching: dict[int, dict[int, float]]) -> CorrespondenceTable:
        """Table with one automatic link per source zone."""
        return cls(
            links=tuple(
                Link(source_zone=zone_id, targets=dict(targets), origin="auto")
                for zone_id, targets in sorted(matching.items())
            )
        )

    def link(self, source_zone: int) -> Link:
        """The link of *source_zone*."""
        for link in self.links:
            if link.source_zone == source_zone:
                return link
        raise KeyError(f"No link for source zone {source_zone}")

    def with_link(self, new_link: Link) -> CorrespondenceTable:
        """A copy of the table where *new_link* replaces its zone's link."""
        self.link(new_link.source_zone)  # raises if unknown
        return replace(
            self,
            links=tuple(
                new_link if link.source_zone == new_link.source_zone else link
                for link in self.links
            ),
        )

    @property
    def has_manual_links(self) -> bool:
        """True when at least one link was set by the user or an AI."""
        return any(link.origin != "auto" for link in self.links)

    def is_compatible(self, source_ids: tuple[int, ...], reference_ids: tuple[int, ...]) -> bool:
        """True when every zone id of the table exists in these palettes."""
        refs = set(reference_ids)
        return tuple(link.source_zone for link in self.links) == tuple(source_ids) and all(
            set(link.targets) <= refs for link in self.links
        )


# ── Edits (pure: each returns a new table or link) ───────────────────────────


def retarget(
    table: CorrespondenceTable,
    source_zone: int,
    reference_zone: int,
    *,
    add: bool = False,
    origin: Origin = "user",
) -> CorrespondenceTable:
    """Send *source_zone* to *reference_zone*.

    Args:
        add: Add the target next to the existing ones (equal share for the
            newcomer, the others scaled down) instead of replacing them.

    A retargeted zone is unlocked; any AI rationale no longer applies.
    """
    link = table.link(source_zone)
    if add and link.targets:
        if reference_zone in link.targets:
            return table
        share = 1.0 / (len(link.targets) + 1)
        targets = {k: w * (1.0 - share) for k, w in link.targets.items()}
        targets[reference_zone] = share
    else:
        targets = {reference_zone: 1.0}
    return table.with_link(
        replace(link, targets=targets, locked=False, origin=origin, rationale="")
    )


def remove_target(
    table: CorrespondenceTable,
    source_zone: int,
    reference_zone: int,
    origin: Origin = "user",
) -> CorrespondenceTable:
    """Drop one target of *source_zone* (the last target can't be removed)."""
    link = table.link(source_zone)
    new_link = link_without_target(link, reference_zone, origin)
    return table if new_link is link else table.with_link(new_link)


def link_without_target(link: Link, reference_zone: int, origin: Origin = "user") -> Link:
    """*link* minus one target, the others renormalised (the last one stays)."""
    if reference_zone not in link.targets or len(link.targets) < 2:
        return link
    rest = {k: w for k, w in link.targets.items() if k != reference_zone}
    total = sum(rest.values())
    targets = {k: w / total for k, w in rest.items()} if total > 0 else _equal(rest)
    return replace(link, targets=targets, origin=origin)


def set_target_weight(link: Link, reference_zone: int, weight: float) -> Link:
    """Give *reference_zone* this share of the link; the others share the rest
    in their current proportions."""
    if reference_zone not in link.targets or len(link.targets) < 2:
        return link
    weight = min(max(weight, 0.0), 1.0)
    others = {k: w for k, w in link.targets.items() if k != reference_zone}
    total = sum(others.values())
    scaled = (
        {k: w / total * (1.0 - weight) for k, w in others.items()}
        if total > 0
        else {k: (1.0 - weight) / len(others) for k in others}
    )
    targets = {k: (weight if k == reference_zone else scaled[k]) for k in link.targets}
    return replace(link, targets=targets, origin="user")


def set_locked(table: CorrespondenceTable, source_zone: int, locked: bool) -> CorrespondenceTable:
    link = table.link(source_zone)
    if link.locked == locked:
        return table
    return table.with_link(replace(link, locked=locked, origin="user"))


def revert_to_auto(
    table: CorrespondenceTable,
    auto: CorrespondenceTable,
    source_zone: int,
) -> CorrespondenceTable:
    """Restore the automatic link of *source_zone*."""
    return table.with_link(auto.link(source_zone))


def _equal(targets: dict[int, float]) -> dict[int, float]:
    return {k: 1.0 / len(targets) for k in targets}
