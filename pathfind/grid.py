"""The grid model: a flat list of cell costs, 4- or 8-connected movement, and three heuristics.

A cell holds 0 for a wall, or an integer cost of at least 1 for entering it (1 is open ground, 5 is
swamp). Cell index is y * width + x. Moving into a cell costs its terrain value, times sqrt(2) for a
diagonal step. Because every terrain value is at least 1, the Manhattan, Euclidean and octile
heuristics never overestimate the cheapest route on a 4-connected grid, and octile never
overestimates on an 8-connected one. Manhattan does overestimate with diagonals, which the lab shows
as a suboptimal path.
"""

from __future__ import annotations

import math
from collections.abc import Iterator

WALL = 0
OPEN = 1
MUD = 3
SWAMP = 5
SQRT2 = 1.4142135623730951  # the same double as Math.SQRT2 in the browser

# Neighbour order is part of the contract: ties are broken by it, so Python and JS must agree.
DIRS4 = ((0, -1), (1, 0), (0, 1), (-1, 0))
DIAGS = ((1, -1), (1, 1), (-1, 1), (-1, -1))
HEURISTICS = ("manhattan", "euclidean", "octile")


class Grid:
    """A rectangular grid of cell costs (0 = wall)."""

    def __init__(self, width: int, height: int, cells: list[int] | None = None):
        if width < 1 or height < 1:
            raise ValueError("grid must be at least 1x1")
        self.width = width
        self.height = height
        self.cells = list(cells) if cells is not None else [OPEN] * (width * height)
        if len(self.cells) != width * height:
            raise ValueError("cells length must be width * height")

    def __len__(self) -> int:
        return self.width * self.height

    def xy(self, i: int) -> tuple[int, int]:
        return i % self.width, i // self.width

    def index(self, x: int, y: int) -> int:
        return y * self.width + x

    def inside(self, x: int, y: int) -> bool:
        return 0 <= x < self.width and 0 <= y < self.height

    def passable(self, x: int, y: int) -> bool:
        return self.inside(x, y) and self.cells[y * self.width + x] != WALL

    def neighbors(self, i: int, diagonal: bool = False) -> Iterator[tuple[int, float]]:
        """Yield (neighbour index, step length) in the fixed order.

        A diagonal is skipped when either orthogonal neighbour is a wall, so it never squeezes
        between two walls that touch at a corner.
        """
        x, y = self.xy(i)
        for dx, dy in DIRS4:
            if self.passable(x + dx, y + dy):
                yield self.index(x + dx, y + dy), 1
        if not diagonal:
            return
        for dx, dy in DIAGS:
            nx, ny = x + dx, y + dy
            if self.passable(nx, ny) and self.passable(x + dx, y) and self.passable(x, y + dy):
                yield self.index(nx, ny), SQRT2

    def step_cost(self, a: int, b: int) -> float:
        """Cost of moving from cell a to the adjacent cell b: terrain of b times the step length."""
        ax, ay = self.xy(a)
        bx, by = self.xy(b)
        step = SQRT2 if ax != bx and ay != by else 1
        return step * self.cells[b]

    def path_cost(self, path: list[int]) -> float | None:
        """Sum of step costs along a path, or None for an empty path."""
        if not path:
            return None
        total = 0.0
        for a, b in zip(path, path[1:], strict=False):
            total += self.step_cost(a, b)
        return total


def heuristic(name: str, x: int, y: int, gx: int, gy: int) -> float:
    """Estimate of the remaining cost from (x, y) to (gx, gy). Name is one of HEURISTICS."""
    dx = abs(x - gx)
    dy = abs(y - gy)
    if name == "manhattan":
        return dx + dy
    if name == "euclidean":
        return math.sqrt(dx * dx + dy * dy)
    if name == "octile":
        hi = dx if dx > dy else dy
        lo = dy if dx > dy else dx
        return hi + (SQRT2 - 1) * lo
    raise ValueError(f"unknown heuristic {name!r}")
