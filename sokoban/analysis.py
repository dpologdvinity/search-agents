"""What the solver knows about one position: deadlocks, frozen boxes, and the heuristic values.

This is the shared read-out used by the terminal (hint and watch), the server's evaluate endpoint, and
the page's overlays. It describes a position without searching, so it is cheap enough to call on
every move in the browser.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .board import Level
from .deadlock import frozen_boxes
from .heuristics import h_matching, h_simple, matching_pairs
from .tables import tables


@dataclass
class Position:
    """A position read out for display. Cells are flat indices; `row`/`col` helpers are on the page side."""

    dead_squares: list[int]
    frozen: list[int]
    deadlock: str | None  # "dead square", "frozen box", or None
    h_simple: int
    h_matching: int
    pairs: list[tuple[int, int, int]] = field(default_factory=list)  # (box, goal, pushes)


def describe(level: Level, boxes: list[int] | tuple[int, ...]) -> Position:
    """Read the deadlock status and heuristic values of a box layout.

    The player's cell does not enter any of these values: deadlocks and push distances depend only on
    the boxes. The heuristic values are computed even for a deadlocked layout, using the Manhattan
    fallback in the tables, so the page can still show a number. The deadlock field says why it is hopeless.
    """
    t = tables(level)
    boxes = sorted(boxes)
    occ = set(boxes)
    dead_squares = [i for i, flag in enumerate(t.dead) if flag]
    frozen = sorted(frozen_boxes(level, occ))
    goals = set(level.goals)
    deadlock = None
    if any(t.dead[b] for b in boxes):
        deadlock = "dead square"
    elif any(b not in goals for b in frozen):
        deadlock = "frozen box"
    return Position(
        dead_squares=dead_squares,
        frozen=frozen,
        deadlock=deadlock,
        h_simple=h_simple(t, boxes),
        h_matching=h_matching(t, boxes),
        pairs=matching_pairs(t, boxes),
    )
