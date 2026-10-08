"""Two admissible lower bounds on the pushes still needed, from the push-distance tables.

simple    the sum over boxes of the push distance to the nearest goal. Cheap (one min per box), but two
          boxes may both claim the same goal, so it can undercount.
matching  the cheapest pairing of boxes with goals, one goal per box (Hungarian algorithm). It never
          lets two boxes claim one goal, so it is at least as large as the simple bound and usually
          much larger near the end of a level.

Both ignore the other boxes' positions, so they are fast and admissible: a real solution is one pairing
of boxes with goals, each pair costing at least its push distance. Pairs that no push path joins use
a Manhattan fallback (so the value stays finite when deadlock pruning is off). The fallback can make
the bound inconsistent, so search.py reopens a state that a cheaper path reaches later; A* then still
returns the fewest pushes.
"""

from __future__ import annotations

from collections.abc import Iterable

from .matching import min_cost_matching
from .tables import Tables

SIMPLE = "simple"
MATCHING = "matching"
NONE = "none"
HEURISTICS = (NONE, SIMPLE, MATCHING)


def h_simple(t: Tables, boxes: Iterable[int]) -> int:
    """Sum over boxes of the push distance to the nearest goal."""
    return sum(min(t.cost[b]) for b in boxes)


def h_matching(t: Tables, boxes: Iterable[int]) -> int:
    """Minimum total push distance over all one-to-one pairings of boxes and goals."""
    total, _ = min_cost_matching([t.cost[b] for b in boxes])
    return total


def matching_pairs(t: Tables, boxes: list[int]) -> list[tuple[int, int, int]]:
    """The pairing behind h_matching, as (box, goal, pushes) triples. Used by the page to draw it."""
    _, match = min_cost_matching([t.cost[b] for b in boxes])
    return [(b, t.goals[match[r]], t.cost[b][match[r]]) for r, b in enumerate(boxes)]
