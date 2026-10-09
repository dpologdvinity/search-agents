"""ASCII side-view rendering of a creature run, for `python -m walkers replay`.

The camera follows the centre of mass. A metre is CELLS_PER_M columns wide and ROWS_PER_M rows tall, so a 1 m
creature fills a few rows. Nodes are 'o'. Springs are drawn as '.', and each muscle as '#', '%' or '+' by its
actuation level (contracted, neutral, extended). The ground is '=' along the surface, with holes and steps taken
from the terrain function.
"""

from __future__ import annotations

from .core import DEFAULT_WORLD, DT, actuation, ground_at

CELLS_PER_M = 8
ROWS_PER_M = 5
COLS = 100
ROWS = 14


def _cell(x: float, y: float, cx: float) -> tuple[int, int]:
    """Map world metres to (row, col) with the camera centred on cx and the floor on the bottom row."""
    col = round((x - cx) * CELLS_PER_M + COLS / 2)
    row = (ROWS - 1) - round(y * ROWS_PER_M)
    return row, col


def _put(grid: list, row: int, col: int, ch: str) -> None:
    """Write one character if it lies inside the grid."""
    if 0 <= row < ROWS and 0 <= col < COLS:
        grid[row][col] = ch


def _line(grid: list, a: tuple, b: tuple, ch: str) -> None:
    """Draw a segment between two cells by stepping along the longer axis."""
    (r0, c0), (r1, c1) = a, b
    n = max(abs(r1 - r0), abs(c1 - c0), 1)
    for k in range(n + 1):
        r = r0 + round((r1 - r0) * k / n)
        c = c0 + round((c1 - c0) * k / n)
        if grid[r][c] not in "o":
            _put(grid, r, c, ch)


def render_frame(genome: dict, xy: list, t: float, terrain: str) -> str:
    """Return one frame as text. xy is a list of (x, y) per node at time t; t is used for the actuation levels."""
    cx = sum(p[0] for p in xy) / len(xy)
    grid = [[" "] * COLS for _ in range(ROWS)]
    # Ground: the surface cell under each column; holes stay blank.
    for col in range(COLS):
        x = (col - COLS / 2) / CELLS_PER_M + cx
        h, _ = ground_at(terrain, x)
        if h > -100:
            row = (ROWS - 1) - round(h * ROWS_PER_M)
            _put(grid, min(max(row, 0), ROWS - 1), col, "=")
    cells = [_cell(x, y, cx) for x, y in xy]
    for s in genome["springs"]:
        a, b = cells[s["a"]], cells[s["b"]]
        if s["muscle"]:
            u = actuation(s, t) / s["amp"] if s["amp"] else 0.0
            ch = "#" if u < -0.33 else "+" if u > 0.33 else "%"
        else:
            ch = "."
        _line(grid, a, b, ch)
    for r, c in cells:
        _put(grid, r, c, "o")
    return "\n".join("".join(row).rstrip() for row in grid)


def replay_frames(genome: dict, frames, sample_every: int, terrain: str) -> list:
    """Text frames for a simulated run. frames has shape (count, N, 2); frame i is at time i * sample_every * DT."""
    out = []
    for i in range(frames.shape[0]):
        xy = [(float(frames[i, n, 0]), float(frames[i, n, 1])) for n in range(frames.shape[1])]
        out.append(render_frame(genome, xy, i * sample_every * DT, terrain))
    return out


def world_for(terrain: str, duration: float | None = None) -> dict:
    """The default world with a terrain (and optionally a duration)."""
    w = dict(DEFAULT_WORLD, terrain=terrain)
    if duration is not None:
        w["duration"] = duration
    return w
