"""Precomputed per-level tables: dead squares and push distances from every cell to every goal.

Both come from one idea. A push moves a box one cell, and it needs the player on the far side of
the box, on a floor cell. Ignoring the other boxes and the player's position, a box at x can be
pushed to y in direction d only when x and y are floor and the cell x - d (where the player stands)
is floor. Run that relation backwards from each goal, and you get the set of cells a box can
still reach the goal from: its push distance. Cells outside every goal's reach are dead: a box
there can never reach a goal, so the state is a deadlock.

The push distance ignores the other boxes, so it is a lower bound on the real pushes. That makes it
admissible for the heuristics in heuristics.py. The tables depend only on the walls and goals, so
they are cached per geometry and shared by every starting position of the same level.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from functools import lru_cache

from .board import Level

UNREACHABLE = -1  # internal marker for cells no goal can reach by pushes


@dataclass(frozen=True)
class Tables:
    """Read-only lookup tables for one level geometry.

    dead[i]       1 when floor cell i is dead (no box there can reach any goal); 0 otherwise.
    cost[i]       tuple over the goals: the push distance from cell i to each goal, or a Manhattan
                  distance when the goal is unreachable from i. The fallback keeps the matching
                  heuristic defined when deadlock pruning is off; a solvable state never uses it,
                  because every box in a solvable state can reach its goal.
    nearest[i]    the smallest push distance from cell i to a goal it can actually reach by pushes
                  (no fallback), or -1 for walls and dead squares. Used for display.
    goals         the goal cells, in the order of the tuples in cost.
    """

    dead: bytes
    cost: tuple[tuple[int, ...], ...]
    nearest: tuple[int, ...]
    goals: tuple[int, ...]


def push_distances(level: Level, goal: int) -> list[int]:
    """Push distance from every cell to `goal`, by a backward breadth-first search from the goal.

    The search runs the push relation in reverse. From a cell y, the box could have come from
    x = y - d, provided x is floor and the cell x - d (the player's spot for that push) is floor.
    Each step back costs one push, so BFS levels are push counts. -1 marks unreachable cells.
    """
    floor = level.floor
    offs = level.offsets
    dist = [UNREACHABLE] * len(floor)
    dist[goal] = 0
    queue = deque([goal])
    while queue:
        y = queue.popleft()
        for off in offs:
            x = y - off  # the box came from x, pushed in direction off
            if not floor[x] or dist[x] != UNREACHABLE:
                continue
            if not floor[x - off]:  # the player must have stood at x - off
                continue
            dist[x] = dist[y] + 1
            queue.append(x)
    return dist


@lru_cache(maxsize=64)
def _build(width: int, height: int, floor: bytes, goals: tuple[int, ...]) -> Tables:
    # A stand-in level with no boxes is enough: the search only reads the walls and the goals.
    level = Level("", width, height, floor, goals, goals[0], goals, "")
    per_goal = [push_distances(level, g) for g in goals]
    dead = bytearray(len(floor))
    cost = []
    nearest = [-1] * len(floor)
    for i in range(len(floor)):
        if not floor[i]:
            cost.append(())
            continue
        row = []
        feasible = []
        r0, c0 = divmod(i, width)
        for g, dist in zip(goals, per_goal):
            if dist[i] != UNREACHABLE:
                row.append(dist[i])
                feasible.append(dist[i])
            else:
                gr, gc = divmod(g, width)
                row.append(abs(r0 - gr) + abs(c0 - gc))
        if feasible:
            nearest[i] = min(feasible)
        else:
            dead[i] = 1
        cost.append(tuple(row))
    return Tables(dead=bytes(dead), cost=tuple(cost), nearest=tuple(nearest), goals=goals)


def tables(level: Level) -> Tables:
    """The tables for a level. Cached on the wall layout and goals, so a mid-game position reuses them."""
    return _build(level.width, level.height, level.floor, level.goals)
