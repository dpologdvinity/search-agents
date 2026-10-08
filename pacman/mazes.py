"""Maze layouts for Pac-Man, parsed into integer-indexed grids.

A layout is a tuple of equal-length text rows:
    #   wall
    .   pellet
    o   power pellet: eating one scares every ghost for a few turns
    P   Pac-Man's start (an empty cell with no pellet)
    G   a ghost's start (also empty); ghosts are numbered in reading order
    ' ' an empty cell with no pellet

Cells are numbered row-major: id = row * width + col. Each open cell stores the ids of
its neighbours to the North, East, South and West (NO_CELL for a wall or the edge), so a
move is a single table lookup. Both layouts here are drawn by hand for this project.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from functools import cache

ACTIONS = ("N", "E", "S", "W")
# Row and column offsets for N, E, S, W, in the same order as ACTIONS.
_DELTAS = ((-1, 0), (0, 1), (1, 0), (0, -1))
NO_CELL = -1

LAYOUTS: dict[str, tuple[str, tuple[str, ...]]] = {
    "lanes": (
        "Neon Lanes",
        (
            "###############",
            "#o...#...#...o#",
            "#.##.#.#.#.##.#",
            "#......P......#",
            "#.##.##.##.##.#",
            "#.....#...#.G.#",
            "#.#.#.###.#.#.#",
            "#....#.....#..#",
            "#.#.....#.....#",
            "#o.G#.....#G.o#",
            "###############",
        ),
    ),
    "vault": (
        "Vault",
        (
            "###############",
            "#o....#.#....o#",
            "#.##.#...#.##.#",
            "#......P......#",
            "#.#.#.###.#.#.#",
            "#.G...#...#.G.#",
            "###.###.#.###.#",
            "#.#...#.G.#...#",
            "#.#.#.#.#.#.#.#",
            "#o.....G.....o#",
            "###############",
        ),
    ),
}


@dataclass(frozen=True)
class Maze:
    """A parsed and validated layout. Everything is a tuple so an instance can be cached and shared."""

    name: str
    title: str
    rows: tuple[str, ...]
    width: int
    height: int
    is_open: tuple[bool, ...]
    nbr: tuple[tuple[int, int, int, int], ...]  # per cell: neighbour ids for N, E, S, W
    row: tuple[int, ...]
    col: tuple[int, ...]
    pellets: frozenset[int]  # every pellet, including the power pellets
    powers: frozenset[int]
    pac_start: int
    ghost_starts: tuple[int, ...]
    corners: tuple[int, ...]  # the open cell nearest each corner: NW, NE, SW, SE

    @property
    def cells(self) -> int:
        return self.width * self.height


def parse(name: str) -> Maze:
    """Build a Maze from LAYOUTS[name] and check it.

    The checks matter because the rules assume a usable board: rows must be rectangular,
    there must be exactly one Pac-Man start and at least one ghost, and every open cell must
    be reachable from Pac-Man. Otherwise some pellets could never be eaten and a game could
    never be won.
    """
    title, rows = LAYOUTS[name]
    height, width = len(rows), len(rows[0])
    if any(len(line) != width for line in rows):
        raise ValueError(f"{name}: rows must all have the same length")
    if set("".join(rows)) - set("#.oPG "):
        raise ValueError(f"{name}: unknown character in layout")

    total = width * height
    is_open = [False] * total
    pellets: set[int] = set()
    powers: set[int] = set()
    pac_starts: list[int] = []
    ghost_starts: list[int] = []
    for r, line in enumerate(rows):
        for c, ch in enumerate(line):
            cell = r * width + c
            if ch == "#":
                continue
            is_open[cell] = True
            if ch in ".o":
                pellets.add(cell)
            if ch == "o":
                powers.add(cell)
            elif ch == "P":
                pac_starts.append(cell)
            elif ch == "G":
                ghost_starts.append(cell)
    if len(pac_starts) != 1 or not ghost_starts:
        raise ValueError(f"{name}: needs exactly one P and at least one G")

    # Neighbour table. Walls get four NO_CELL entries so lookups never need a wall check.
    nbr: list[tuple[int, int, int, int]] = []
    for cell in range(total):
        if not is_open[cell]:
            nbr.append((NO_CELL,) * 4)
            continue
        r, c = divmod(cell, width)
        options = []
        for dr, dc in _DELTAS:
            rr, cc = r + dr, c + dc
            inside = 0 <= rr < height and 0 <= cc < width
            options.append(rr * width + cc if inside and is_open[rr * width + cc] else NO_CELL)
        nbr.append((options[0], options[1], options[2], options[3]))

    # Connectivity: a flood fill from Pac-Man must reach every open cell.
    seen = {pac_starts[0]}
    queue = deque(seen)
    while queue:
        for nb in nbr[queue.popleft()]:
            if nb != NO_CELL and nb not in seen:
                seen.add(nb)
                queue.append(nb)
    if len(seen) != sum(is_open):
        raise ValueError(f"{name}: some open cells cannot be reached from P")

    row = tuple(cell // width for cell in range(total))
    col = tuple(cell % width for cell in range(total))
    open_cells = [cell for cell in range(total) if is_open[cell]]
    # Scatter ghosts retreat to these; each corner maps to the open cell closest to it.
    corners = tuple(
        min(open_cells, key=lambda cell, cr=cr, cc=cc: (abs(row[cell] - cr) + abs(col[cell] - cc), cell))
        for cr, cc in ((0, 0), (0, width - 1), (height - 1, 0), (height - 1, width - 1))
    )
    return Maze(
        name=name,
        title=title,
        rows=tuple(rows),
        width=width,
        height=height,
        is_open=tuple(is_open),
        nbr=tuple(nbr),
        row=row,
        col=col,
        pellets=frozenset(pellets),
        powers=frozenset(powers),
        pac_start=pac_starts[0],
        ghost_starts=tuple(ghost_starts),
        corners=corners,
    )


@cache
def get(name: str) -> Maze:
    """The parsed layout for `name`, built once per process."""
    if name not in LAYOUTS:
        raise KeyError(f"unknown maze {name!r}; choose from {', '.join(LAYOUTS)}")
    return parse(name)


def names() -> tuple[str, ...]:
    return tuple(LAYOUTS)
