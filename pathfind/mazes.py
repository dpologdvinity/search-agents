"""Seeded maze generators: recursive backtracker, Prim's, random scatter, rooms, and open ground.

Every generator takes (kind, width, height, seed) and returns the same grid for the same seed, in
Python and in the browser (web/js/pathfind-core.js). The border is always wall, so a search never has
to worry about the edge of the map.

Perfect mazes (recursive, prim) carve passages on the odd-coordinate lattice: each lattice cell is
open, and the wall between two neighbouring lattice cells is removed when they join. The result is a
spanning tree of the lattice, so there is exactly one route between any two cells.
"""

from __future__ import annotations

from dataclasses import dataclass

from .grid import OPEN, SWAMP, WALL, Grid
from .rng import Rng

KINDS = ("recursive", "prim", "scatter", "rooms", "open")
# Steps of two cells on the lattice, in a fixed order (the order breaks ties, so it is part of the contract).
LATTICE_DIRS = ((0, -2), (2, 0), (0, 2), (-2, 0))


@dataclass(frozen=True)
class Maze:
    """A generated map with its default start and goal cell indices."""

    grid: Grid
    start: int
    goal: int


def make_maze(kind: str, width: int, height: int, seed: int, *, density: int = 28, swamp: int = 0) -> Maze:
    """Build a seeded map.

    density: percent of interior cells that become walls in the scatter map.
    swamp: percent of open cells (not start or goal) that become swamp, cost 5.
    """
    if kind not in KINDS:
        raise ValueError(f"unknown maze kind {kind!r}")
    if width < 5 or height < 5:
        raise ValueError("maze needs at least 5x5 cells")
    rng = Rng(seed)
    if kind == "recursive":
        cells = _recursive(width, height, rng)
        start, goal = _lattice_ends(width, height)
    elif kind == "prim":
        cells = _prim(width, height, rng)
        start, goal = _lattice_ends(width, height)
    elif kind == "scatter":
        cells = _scatter(width, height, rng, density)
        start, goal = _lattice_ends(width, height)
        cells[start] = OPEN
        cells[goal] = OPEN
    elif kind == "rooms":
        cells, rooms = _rooms(width, height, rng)
        start, goal = rooms[0], rooms[-1]
    else:
        cells = [OPEN] * (width * height)
        start, goal = 0, width * height - 1
    grid = Grid(width, height, cells)
    if swamp:
        _scatter_swamp(grid, rng, swamp, skip=(start, goal))
    return Maze(grid, start, goal)


def _lattice_ends(width: int, height: int) -> tuple[int, int]:
    """Start at (1, 1) and goal at the last odd interior coordinates, which are lattice cells."""
    gx = width - 2 if (width - 2) % 2 == 1 else width - 3
    gy = height - 2 if (height - 2) % 2 == 1 else height - 3
    return width + 1, gy * width + gx


def _recursive(width: int, height: int, rng: Rng) -> list[int]:
    """Recursive backtracker, written with an explicit stack. Walks to a random unvisited lattice
    neighbour and carves to it; when stuck, it backs up along the walk."""
    cells = [WALL] * (width * height)
    stack = [(1, 1)]
    cells[width + 1] = OPEN
    while stack:
        x, y = stack[-1]
        options = []
        for dx, dy in LATTICE_DIRS:
            nx, ny = x + dx, y + dy
            if 0 < nx < width - 1 and 0 < ny < height - 1 and cells[ny * width + nx] == WALL:
                options.append((dx, dy))
        if not options:
            stack.pop()
            continue
        dx, dy = options[rng.int(len(options))]
        nx, ny = x + dx, y + dy
        cells[(y + dy // 2) * width + (x + dx // 2)] = OPEN  # the wall between the two lattice cells
        cells[ny * width + nx] = OPEN
        stack.append((nx, ny))
    return cells


def _prim(width: int, height: int, rng: Rng) -> list[int]:
    """Prim's algorithm: grow a tree from (1, 1). The frontier holds (cell, tree neighbour) pairs; a
    random pair is taken out (swap with the last item, then pop), and its cell joins the tree if it
    is still unvisited."""
    cells = [WALL] * (width * height)
    cells[width + 1] = OPEN
    frontier: list[tuple[int, int, int, int]] = []

    def add_neighbours(x: int, y: int) -> None:
        for dx, dy in LATTICE_DIRS:
            nx, ny = x + dx, y + dy
            if 0 < nx < width - 1 and 0 < ny < height - 1 and cells[ny * width + nx] == WALL:
                frontier.append((nx, ny, x, y))

    add_neighbours(1, 1)
    while frontier:
        i = rng.int(len(frontier))
        frontier[i], frontier[-1] = frontier[-1], frontier[i]
        nx, ny, px, py = frontier.pop()
        if cells[ny * width + nx] != WALL:
            continue
        cells[(py + ny) // 2 * width + (px + nx) // 2] = OPEN
        cells[ny * width + nx] = OPEN
        add_neighbours(nx, ny)
    return cells


def _scatter(width: int, height: int, rng: Rng, density: int) -> list[int]:
    """Each interior cell is a wall with probability density%. No connectivity is promised."""
    cells = [WALL] * (width * height)
    for y in range(1, height - 1):
        for x in range(1, width - 1):
            if not rng.percent(density):
                cells[y * width + x] = OPEN
    return cells


def _rooms(width: int, height: int, rng: Rng) -> tuple[list[int], list[int]]:
    """Scatter rectangular rooms, then join them with L-shaped corridors: a chain through every room
    in creation order (so the map is connected), plus one random extra link per room for loops.
    Returns the cells and the room centres (first and last are the start and goal)."""
    cells = [WALL] * (width * height)
    target = max(2, (width * height) // 150)
    centres: list[tuple[int, int]] = []
    rects: list[tuple[int, int, int, int]] = []
    for _ in range(target * 6):
        if len(centres) >= target:
            break
        rw = 3 + rng.int(5)
        rh = 3 + rng.int(4)
        x0 = 1 + rng.int(max(1, width - rw - 2))
        y0 = 1 + rng.int(max(1, height - rh - 2))
        x1, y1 = x0 + rw - 1, y0 + rh - 1
        if x1 > width - 2 or y1 > height - 2:
            continue
        # Reject a room that touches another (one cell of margin), so rooms stay distinct.
        if any(x0 - 1 <= bx1 and bx0 <= x1 + 1 and y0 - 1 <= by1 and by0 <= y1 + 1 for bx0, by0, bx1, by1 in rects):
            continue
        for y in range(y0, y1 + 1):
            for x in range(x0, x1 + 1):
                cells[y * width + x] = OPEN
        rects.append((x0, y0, x1, y1))
        centres.append(((x0 + x1) // 2, (y0 + y1) // 2))
    if len(centres) < 2:
        raise ValueError("map too small for two rooms")

    def link(a: tuple[int, int], b: tuple[int, int]) -> None:
        (ax, ay), (bx, by) = a, b
        for x in range(min(ax, bx), max(ax, bx) + 1):
            cells[ay * width + x] = OPEN
        for y in range(min(ay, by), max(ay, by) + 1):
            cells[y * width + bx] = OPEN

    for i in range(1, len(centres)):
        link(centres[i - 1], centres[i])
    for i in range(len(centres)):
        j = rng.int(len(centres))
        if j != i:
            link(centres[i], centres[j])
    return cells, [centres[0][1] * width + centres[0][0], centres[-1][1] * width + centres[-1][0]]


def _scatter_swamp(grid: Grid, rng: Rng, percent: int, skip: tuple[int, int]) -> None:
    """Turn open cells into swamp (cost 5) with probability percent%, in row-major order."""
    for i, c in enumerate(grid.cells):
        if c == OPEN and i not in skip and rng.percent(percent):
            grid.cells[i] = SWAMP
