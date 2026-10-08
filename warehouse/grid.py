"""Warehouse map model: a 4-connected grid with shelves, the built-in layouts, and problem instances.

A map is a rectangle of cells. '#' is a shelf (blocked) and '.' is floor that a robot can enter.
Cells are numbered row-major (``cell = y * width + x``), so per-cell tables are plain lists and
neighbours are integer lookups in the search inner loops.

A ``Problem`` is one multi-robot pathfinding instance on a map: robot ``i`` starts on
``starts[i]`` and must reach ``goals[i]``. Each time step a robot moves to a free 4-neighbour
or waits in place.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass
from functools import cache

# Built-in warehouses, one string per row (y grows downward). Each has a title and a one-line blurb
# shown in the terminal and in the web page.
LAYOUTS: dict[str, tuple[str, ...]] = {
    "aisles": (
        "................",
        ".##.##.##.##.##.",
        ".##.##.##.##.##.",
        "................",
        ".##.##.##.##.##.",
        ".##.##.##.##.##.",
        "................",
        ".##.##.##.##.##.",
        ".##.##.##.##.##.",
        "................",
    ),
    "crossroads": (
        "...........",
        ".###...###.",
        ".###...###.",
        "...........",
        ".....#.....",
        "...........",
        ".###...###.",
        ".###...###.",
        "...........",
    ),
    "bottleneck": (
        ".............",
        ".###.###.###.",
        ".............",
        "#####.#######",
        ".............",
        ".###.###.###.",
        ".............",
    ),
}

LAYOUT_INFO: dict[str, dict[str, str]] = {
    "aisles": {"title": "Shelf aisles", "blurb": "Long parallel aisles with cross lanes at the ends."},
    "crossroads": {"title": "Crossroads", "blurb": "Two shelf blocks around a central plaza."},
    "bottleneck": {"title": "Bottleneck", "blurb": "Two halves joined by a single doorway cell."},
}


class Grid:
    """An immutable map: ``width`` x ``height`` cells, some of them shelves.

    Neighbour lists are computed once, so search code only reads ``neighbors[cell]``.
    Distance maps to a goal are computed on demand and cached: they are the admissible
    heuristic that the space-time search uses, and one map serves every robot with that goal.
    """

    def __init__(self, rows: Sequence[str]):
        if not rows:
            raise ValueError("a map needs at least one row")
        width = len(rows[0])
        if width == 0 or any(len(r) != width for r in rows):
            raise ValueError("all map rows must have the same non-zero length")
        if any(ch not in ".#" for r in rows for ch in r):
            raise ValueError("map cells must be '.' (floor) or '#' (shelf)")
        self.width = width
        self.height = len(rows)
        self.rows = tuple(rows)
        self.size = width * self.height
        self.free = [rows[y][x] == "." for y in range(self.height) for x in range(width)]
        self.neighbors: list[tuple[int, ...]] = []
        for c in range(self.size):
            x, y = c % width, c // width
            around = []
            for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
                if 0 <= nx < width and 0 <= ny < self.height and self.free[ny * width + nx]:
                    around.append(ny * width + nx)
            self.neighbors.append(tuple(around))
        self._distances: dict[int, list[int]] = {}

    def cell(self, x: int, y: int) -> int:
        return y * self.width + x

    def xy(self, cell: int) -> tuple[int, int]:
        return cell % self.width, cell // self.width

    def distances_to(self, goal: int) -> list[int]:
        """Shortest walking distance from every cell to ``goal`` (-1 if unreachable).

        Ignores the other robots, so it never overestimates the real cost. That makes it an
        admissible and consistent heuristic for space-time A*.
        """
        dist = self._distances.get(goal)
        if dist is None:
            dist = [-1] * self.size
            dist[goal] = 0
            queue = deque([goal])
            while queue:
                c = queue.popleft()
                for nb in self.neighbors[c]:
                    if dist[nb] < 0:
                        dist[nb] = dist[c] + 1
                        queue.append(nb)
            self._distances[goal] = dist
        return dist

    def floor_cells(self) -> list[int]:
        return [c for c in range(self.size) if self.free[c]]

    def largest_component(self) -> list[int]:
        """Floor cells of the biggest connected region. Robots placed here can all reach each other."""
        seen: set[int] = set()
        best: list[int] = []
        for start in self.floor_cells():
            if start in seen:
                continue
            comp = []
            queue = deque([start])
            seen.add(start)
            while queue:
                c = queue.popleft()
                comp.append(c)
                for nb in self.neighbors[c]:
                    if nb not in seen:
                        seen.add(nb)
                        queue.append(nb)
            if len(comp) > len(best):
                best = comp
        return sorted(best)


@cache
def layout_grid(name: str) -> Grid:
    """The built-in map called ``name``. Cached: the distance maps it builds are reused across requests."""
    if name not in LAYOUTS:
        raise KeyError(f"unknown layout {name!r}; choose from {', '.join(LAYOUTS)}")
    return Grid(LAYOUTS[name])


@dataclass(frozen=True)
class Problem:
    """Robot ``i`` goes from ``starts[i]`` to ``goals[i]``; cells are integers from ``Grid``.

    Starts must be distinct, goals must be distinct, and each goal must be reachable from its start.
    One robot's goal may be another robot's start; the planners handle that through timing.
    """

    grid: Grid
    starts: tuple[int, ...]
    goals: tuple[int, ...]

    def __post_init__(self):
        if len(self.starts) != len(self.goals) or not self.starts:
            raise ValueError("need one goal per robot and at least one robot")
        if len(set(self.starts)) != len(self.starts):
            raise ValueError("two robots start on the same cell")
        if len(set(self.goals)) != len(self.goals):
            raise ValueError("two robots share a goal cell")
        for s, g in zip(self.starts, self.goals):
            if not (self.grid.free[s] and self.grid.free[g]):
                raise ValueError("starts and goals must be floor cells")
            if self.grid.distances_to(g)[s] < 0:
                raise ValueError("a goal cannot be reached from its start")

    @property
    def robots(self) -> int:
        return len(self.starts)

    @classmethod
    def from_xy(cls, grid: Grid, starts: Sequence[tuple[int, int]], goals: Sequence[tuple[int, int]]) -> Problem:
        """Build a problem from (x, y) pairs, the form the web API and the CLI use."""
        return cls(
            grid,
            tuple(grid.cell(x, y) for x, y in starts),
            tuple(grid.cell(x, y) for x, y in goals),
        )
