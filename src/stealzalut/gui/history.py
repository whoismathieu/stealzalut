"""Undo/redo history of the editing state.

The state is a small immutable value (correspondence table + adjustment
sliders + zone separation), so the history is just two stacks of states.
Continuous gestures (dragging a slider) are merged into a single undo step.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field

from stealzalut.core.correspondence import CorrespondenceTable
from stealzalut.core.finishing import Finishing
from stealzalut.core.grading import GradingParams

#: Edits with the same merge key closer than this (s) form one undo step.
MERGE_WINDOW_S = 1.0
HISTORY_LIMIT = 200


@dataclass(frozen=True, slots=True)
class EditState:
    """Everything the user can undo.

    Attributes:
        table: Zone correspondences (``None`` outside the zones method).
        params: Adjustment sliders.
        tau: Zone separation.
        finishing: Finishing tools (wheels, curve, colour bands…).
        separation: Separation of the look method.
        focus: Weights of the reference's zones for the look method, as
            sorted ``(zone id, weight)`` pairs; zones left out weigh 1.
    """

    table: CorrespondenceTable | None
    params: GradingParams
    tau: float
    finishing: Finishing = field(default_factory=Finishing)
    separation: float = 1.0
    focus: tuple[tuple[int, float], ...] = ()


class History:
    """Linear undo/redo history of :class:`EditState`."""

    def __init__(
        self,
        initial: EditState,
        limit: int = HISTORY_LIMIT,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._current = initial
        self._undo: list[EditState] = []
        self._redo: list[EditState] = []
        self._limit = limit
        self._clock = clock
        self._last_key: str | None = None
        self._last_time = 0.0

    @property
    def current(self) -> EditState:
        return self._current

    @property
    def can_undo(self) -> bool:
        return bool(self._undo)

    @property
    def can_redo(self) -> bool:
        return bool(self._redo)

    def push(self, state: EditState, merge_key: str | None = None) -> bool:
        """Make *state* current.

        Args:
            merge_key: Edits sharing a key within :data:`MERGE_WINDOW_S`
                replace each other instead of adding undo steps.

        Returns:
            ``False`` when *state* equals the current state (nothing done).
        """
        if state == self._current:
            return False
        now = self._clock()
        merge = (
            merge_key is not None
            and merge_key == self._last_key
            and now - self._last_time <= MERGE_WINDOW_S
            and bool(self._undo)
        )
        if not merge:
            self._undo.append(self._current)
            del self._undo[: -self._limit]
        self._current = state
        self._redo.clear()
        self._last_key = merge_key
        self._last_time = now
        return True

    def undo(self) -> EditState | None:
        if not self._undo:
            return None
        self._redo.append(self._current)
        self._current = self._undo.pop()
        self._last_key = None
        return self._current

    def redo(self) -> EditState | None:
        if not self._redo:
            return None
        self._undo.append(self._current)
        self._current = self._redo.pop()
        self._last_key = None
        return self._current

    def reset(self, state: EditState) -> None:
        """Start over from *state* (new images or new zones)."""
        self._current = state
        self._undo.clear()
        self._redo.clear()
        self._last_key = None
