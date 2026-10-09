"""Path bookkeeping and collision detection for multi-robot plans.

A plan is one path per robot. ``path[t]`` is the cell the robot occupies at time step ``t``.
A robot that has reached its goal stays there, so ``position(path, t)`` clamps ``t`` to the
last index. The arrival time of a path (``len(path) - 1``) is its cost; the sum of arrival times
is the sum-of-costs objective and the largest one is the makespan.

Two robots conflict when they break either rule:
- vertex conflict: both occupy the same cell at the same time step;
- edge (swap) conflict: they trade places, moving across the same pair of neighbouring cells in
  opposite directions in one step. Robots never pass through each other, so this has to be
  checked explicitly; checking only occupied cells would miss it.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

Path = tuple[int, ...]


@dataclass(frozen=True)
class Conflict:
    """A collision at time step ``t`` between two robots.

    For a vertex conflict ``cells`` is ``(cell,)``. For an edge conflict ``robots[0]`` moves
    ``cells[0] -> cells[1]`` while ``robots[1]`` moves ``cells[1] -> cells[0]``.
    """

    kind: str  # "vertex" or "edge"
    t: int
    robots: tuple[int, int]
    cells: tuple[int, ...]


def position(path: Path, t: int) -> int:
    """Where the robot is at time ``t``: it waits at the end of its path once it has arrived."""
    return path[min(t, len(path) - 1)]


def makespan(paths: Sequence[Path]) -> int:
    """Time step at which the last robot arrives."""
    return max(len(p) - 1 for p in paths)


def sum_of_costs(paths: Sequence[Path]) -> int:
    """Total of the robots' arrival times: the objective CBS minimises."""
    return sum(len(p) - 1 for p in paths)


def find_conflicts(paths: Sequence[Path], first_only: bool = False) -> list[Conflict]:
    """Every collision in the plan, ordered by time (or only the earliest one if ``first_only``).

    Checked at every step from 0 to the makespan. Finished robots stay put, so they can still be hit.
    Vertex rule: each robot on a cell that an earlier robot in the list holds at that step gives one conflict,
    paired with the first robot holding that cell. Edge rule: each swapping pair gives one conflict.
    """
    found: list[Conflict] = []
    if not paths:
        return found
    horizon = makespan(paths)
    prev = [position(p, 0) for p in paths]
    for t in range(horizon + 1):
        now = [position(p, t) for p in paths]
        # Vertex conflicts: map each occupied cell to the first robot that stands there.
        occupant: dict[int, int] = {}
        for i, c in enumerate(now):
            j = occupant.get(c)
            if j is None:
                occupant[c] = i
            else:
                found.append(Conflict("vertex", t, (j, i), (c,)))
                if first_only:
                    return found
        # Edge conflicts: robot i moved a -> b, and the robot that was on b (j) moved b -> a.
        # Requiring i < j reports each swapping pair once.
        if t > 0:
            before: dict[int, list[int]] = {}
            for i, c in enumerate(prev):
                before.setdefault(c, []).append(i)
            for i in range(len(paths)):
                a, b = prev[i], now[i]
                if a == b:
                    continue
                for j in before.get(b, ()):  # usually zero or one robot; more means a vertex conflict at t-1
                    if j > i and now[j] == a:
                        found.append(Conflict("edge", t, (i, j), (a, b)))
                        if first_only:
                            return found
        prev = now
    return found
