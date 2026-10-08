"""The puzzle model: clues, grids, and a plain-text renderer.

A nonogram is a picture of filled and empty cells. Each row and each column is described by its clue:
the lengths of the runs of filled cells, in order, so a row of "..##.###" has the clue (2, 3). An empty
line has the clue () and is shown as 0.

A grid is a list of rows of ints: UNKNOWN (-1) before the player or solver decides a cell, EMPTY (0),
or FILLED (1). In the player's grid, EMPTY also means "crossed out": the cell is known to be empty.
"""

from __future__ import annotations

from dataclasses import dataclass

UNKNOWN = -1
EMPTY = 0
FILLED = 1

MIN_SIZE = 1
MAX_SIZE = 25  # the largest grid the solver endpoints accept


def runs_of(line) -> tuple[int, ...]:
    """The clue of a line: the lengths of its runs of filled cells (1 = filled), in order."""
    runs, length = [], 0
    for v in line:
        if v == FILLED:
            length += 1
        elif length:
            runs.append(length)
            length = 0
    if length:
        runs.append(length)
    return tuple(runs)


def clues_of(picture) -> tuple[tuple[tuple[int, ...], ...], tuple[tuple[int, ...], ...]]:
    """Row and column clues of a fully filled picture (a list of rows of 0/1)."""
    rows = [runs_of(r) for r in picture]
    cols = [runs_of([picture[r][c] for r in range(len(picture))]) for c in range(len(picture[0]))]
    return tuple(rows), tuple(cols)


def picture_from_art(art) -> list[list[int]]:
    """Read a hand-drawn picture: '#' is filled, anything else is empty. All rows must be the same width."""
    width = len(art[0])
    if any(len(row) != width for row in art):
        raise ValueError("every row of the picture must have the same width")
    return [[1 if ch == "#" else 0 for ch in row] for row in art]


def validate_clues(row_clues, col_clues) -> tuple[int, int]:
    """Check that a clue set describes a grid we can solve. Returns (rows, cols) or raises ValueError.

    Checks the sizes, that every run is at least 1 long, and that each line's runs fit in its length
    (runs plus one separator cell between neighbours). A puzzle that passes may still have no solution;
    the solver reports that as a contradiction.
    """
    rows, cols = len(row_clues), len(col_clues)
    if not (MIN_SIZE <= rows <= MAX_SIZE and MIN_SIZE <= cols <= MAX_SIZE):
        raise ValueError(f"grids are {MIN_SIZE} to {MAX_SIZE} cells on each side, got {rows}x{cols}")
    for name, clues, length in (("row", row_clues, cols), ("column", col_clues, rows)):
        for i, clue in enumerate(clues):
            if any(int(v) != v or v < 1 for v in clue):
                raise ValueError(f"{name} {i + 1}: run lengths must be whole numbers of at least 1")
            needed = sum(clue) + max(0, len(clue) - 1)
            if needed > length:
                raise ValueError(f"{name} {i + 1}: clue {list(clue)} needs {needed} cells, the line has {length}")
    return rows, cols


@dataclass(frozen=True)
class Puzzle:
    """A puzzle: its clues, plus an id and a display name. The picture is not stored; the solver finds it."""

    name: str
    row_clues: tuple[tuple[int, ...], ...]
    col_clues: tuple[tuple[int, ...], ...]
    id: str | None = None

    @property
    def rows(self) -> int:
        return len(self.row_clues)

    @property
    def cols(self) -> int:
        return len(self.col_clues)

    @classmethod
    def from_picture(cls, name: str, picture, id: str | None = None) -> Puzzle:
        """Build a puzzle from a picture given as art strings ('#' filled) or a grid of 0/1."""
        if picture and isinstance(picture[0], str):
            picture = picture_from_art(picture)
        rows, cols = clues_of(picture)
        return cls(name, rows, cols, id)


def empty_grid(rows: int, cols: int) -> list[list[int]]:
    """A grid with every cell unknown."""
    return [[UNKNOWN] * cols for _ in range(rows)]


def clue_text(clue) -> str:
    """A clue as the player reads it: "3 1", or "0" for an empty line."""
    return " ".join(str(v) for v in clue) if clue else "0"


def render(puzzle: Puzzle, grid=None, solved: bool = False) -> str:
    """ASCII board: column clues stacked above the grid, row clues to its left.

    In the grid, '#' is filled, 'x' is crossed out, and '.' is unknown. With solved=True the grid is a
    finished picture, so its empty cells are shown as plain '.'. Pass no grid to see just the clues.
    """
    rows, cols = puzzle.rows, puzzle.cols
    grid = grid if grid is not None else empty_grid(rows, cols)
    depth = max(1, max((len(c) for c in puzzle.col_clues), default=1))
    row_labels = [clue_text(c) for c in puzzle.row_clues]
    label_w = max(len(s) for s in row_labels)
    lines = []
    for level in range(depth):
        cells = []
        for c in range(cols):
            clue = puzzle.col_clues[c]
            # Column clues are bottom-aligned: the last number sits just above the grid.
            k = len(clue) - depth + level
            cells.append(f"{clue[k]:>2}" if k >= 0 else "  ")
        lines.append(" " * (label_w + 4) + " ".join(cells))
    lines.append(" " * (label_w + 4) + " ".join(f"{chr(97 + c % 26):>2}" for c in range(cols)))
    for r in range(rows):
        # A solved picture has no crosses: its empty cells are plain dots.
        marks = {FILLED: "#", EMPTY: "." if solved else "x", UNKNOWN: "."}
        cells = " ".join(f" {marks[grid[r][c]]}" for c in range(cols))
        lines.append(f"{row_labels[r]:>{label_w}} {r + 1:>2} {cells}")
    return "\n".join(lines)


def picture_text(picture) -> str:
    """A solved picture as '#' and '.' rows."""
    return "\n".join("".join("#" if v else "." for v in row) for row in picture)
