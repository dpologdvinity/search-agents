"""Tetromino shapes and their rotation states.

Each piece is four cells. The shapes are written top-down as text below and converted to (x, y)
with y counting up from the bottom, so the same numbers index the board (row 0 is the bottom row).
A rotation is a clockwise turn of the piece's bounding box, moved back so the box starts at (0, 0).
States that give the same cell set are dropped: O has one state, I, S and Z have two, the rest four.
The first state of each piece is the spawn orientation. Every spawn shape is at most two rows tall,
so a piece spawns inside the top two rows of the board.
"""

from __future__ import annotations

from dataclasses import dataclass

PIECES = "IOTSZJL"

_SHAPES = {
    "I": ["####"],
    "O": ["##", "##"],
    "T": [" # ", "###"],
    "S": [" ##", "## "],
    "Z": ["## ", " ##"],
    "J": ["#  ", "###"],
    "L": ["  #", "###"],
}


@dataclass(frozen=True)
class Rotation:
    """One orientation of a piece.

    masks[i] is row i of the piece's bounding box (bottom row first) as a bitmask, bit x = column x.
    Placing the piece at column px and bottom row py means row py + i gets masks[i] << px.
    """

    cells: frozenset
    width: int
    height: int
    masks: tuple[int, ...]


def _cells_from_text(rows: list[str]) -> frozenset:
    h = len(rows)
    return frozenset((x, h - 1 - i) for i, row in enumerate(rows) for x, ch in enumerate(row) if ch == "#")


def normalise(cells: frozenset) -> frozenset:
    """Shift a cell set so its bounding box starts at (0, 0)."""
    mx = min(x for x, _ in cells)
    my = min(y for _, y in cells)
    return frozenset((x - mx, y - my) for x, y in cells)


def rotate_cw(cells: frozenset) -> frozenset:
    """Clockwise quarter turn: (x, y) -> (y, -x), then renormalised."""
    return normalise(frozenset((y, -x) for x, y in cells))


def _make_rotation(cells: frozenset) -> Rotation:
    width = max(x for x, _ in cells) + 1
    height = max(y for _, y in cells) + 1
    masks = tuple(sum(1 << x for x, y in cells if y == row) for row in range(height))
    return Rotation(cells=cells, width=width, height=height, masks=masks)


def _build() -> dict[str, tuple[Rotation, ...]]:
    out: dict[str, tuple[Rotation, ...]] = {}
    for name, rows in _SHAPES.items():
        cur = normalise(_cells_from_text(rows))
        seen: list[frozenset] = []
        states: list[Rotation] = []
        for _ in range(4):
            if cur not in seen:
                seen.append(cur)
                states.append(_make_rotation(cur))
            cur = rotate_cw(cur)
        out[name] = tuple(states)
    return out


ROTATIONS: dict[str, tuple[Rotation, ...]] = _build()
