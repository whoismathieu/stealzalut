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
    type a colour value.
    """

    lightness: Level = "0"
    saturation: Level = "0"
    warmth: Level = "0"  # + warmer (yellow/orange), - cooler
    tint: Level = "0"  # + magenta, - green

    @property
    def is_neutral(self) -> bool:
        return self == Intent()


@dataclass(frozen=True, slots=True)
class Link:
    """How one source zone is treated.

    Attributes:
        source_zone: Source zone id.
        targets: ``{reference_zone_id: weight}`` summing to 1; empty means
            "keep this zone as it is".
        strength: 0 = untouched, 1 = full transfer.
        luminance_strength: How much the zone takes its target's lightness
            (``None`` / 0 = follow the global tone mapping, 1 = target
            lightness; may fold the LUT if zones swap lightness order).
        chroma_strength: Per-zone chroma strength (``None`` = 1).
        locked: Zone is never changed (no transfer, no intent).
        intent: Qualitative nudges.
        origin: Who set this link.
        rationale: Free-text explanation (shown as a tooltip).
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

    def __hash__(self) -> int:  # dict field: hash on identity-stable fields only
        return hash((self.source_zone, tuple(sorted(self.targets.items())), self.strength))

    @property
    def is_identity(self) -> bool:
        """True when this link leaves its zone untouched."""
        return self.locked or (
            (not self.targets or self.strength == 0.0) and self.intent.is_neutral
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
