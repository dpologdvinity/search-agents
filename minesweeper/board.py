"""Minesweeper rules on a rows x cols grid of hidden mines.

Cells are numbered row by row from the top-left, 0 .. rows*cols-1, so cell i sits at row
i // cols and column i % cols. A reveal shows how many of the cell's (up to eight) neighbours
hold a mine; a zero also reveals its neighbours, which is the flood fill that opens a game.

The mines are placed after the first click, around it: the clicked cell and its neighbours are
always safe, so the first reveal always opens something. Players never see the mine layout, and
neither does the agent: agents get a `View` of the revealed numbers and the mine count only.
"""

from __future__ import annotations

import random
import re
from collections import deque
from dataclasses import dataclass
from functools import lru_cache

# Classic sizes: (rows, cols, mines). Expert is 16 rows of 30 columns, as on the familiar board.
PRESETS = {
    "beginner": (9, 9, 10),
    "intermediate": (16, 16, 40),
    "expert": (16, 30, 99),
}
DEFAULT_PRESET = "beginner"

UNKNOWN = -1  # a View cell that is still covered
MINE_SHOWN = 9  # internal marker for the mine that ended a game; never a real number (0..8)


@dataclass(frozen=True)
class View:
    """What a player sees: the size, the mine count, and per cell either UNKNOWN or the revealed number 0..8."""

    rows: int
    cols: int
    mines: int
    cells: tuple[int, ...]

    @property
    def size(self) -> int:
        return self.rows * self.cols


@lru_cache(maxsize=16)
def neighbour_table(rows: int, cols: int) -> tuple[tuple[int, ...], ...]:
    """For each cell, the indices of its in-bounds neighbours (up to eight).

    Cached per board shape, since every reveal and every constraint looks up the same neighbourhoods.
    """
    table = []
    for i in range(rows * cols):
        r, c = divmod(i, cols)
        near = []
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if (dr or dc) and 0 <= r + dr < rows and 0 <= c + dc < cols:
                    near.append((r + dr) * cols + (c + dc))
        table.append(tuple(near))
    return tuple(table)


class Game:
    """One game: hidden mines, revealed cells, and the player's flags.

    `rng` places the mines on the first reveal. The benchmark passes a seeded generator, so the
    same seed gives the same board for every agent.
    """

    def __init__(self, rows: int, cols: int, mines: int, rng: random.Random | None = None):
        if not 0 < mines < rows * cols:
            raise ValueError("mines must be between 1 and rows*cols-1")
        self.rows, self.cols, self.mines = rows, cols, mines
        self.rng = rng or random.Random()
        self._mine: set[int] | None = None  # placed on the first reveal
        self.revealed: list[int] = [UNKNOWN] * (rows * cols)  # the number shown, or UNKNOWN
        self.flags: set[int] = set()
        self.lost_on: int | None = None  # the mine that ended the game, if any
        self.reveals = 0  # clicks that revealed something

    @property
    def size(self) -> int:
        return self.rows * self.cols

    @property
    def over(self) -> bool:
        return self.lost_on is not None or self.won

    @property
    def won(self) -> bool:
        return self.lost_on is None and self.revealed_count == self.size - self.mines

    @property
    def revealed_count(self) -> int:
        return sum(1 for v in self.revealed if v != UNKNOWN and v != MINE_SHOWN)

    def mine_set(self) -> frozenset[int]:
        """The true mine layout. Only the game and the end-of-game screen read this, never an agent."""
        return frozenset(self._mine or ())

    def _place_mines(self, first: int) -> None:
        # The first click and its neighbours stay clear, so the first reveal opens an area.
        clear = {first, *neighbour_table(self.rows, self.cols)[first]}
        pool = [i for i in range(self.size) if i not in clear]
        self._mine = set(self.rng.sample(pool, self.mines))

    def adjacent_mines(self, cell: int) -> int:
        """The number a reveal of `cell` shows (needs the mine layout)."""
        return sum(1 for n in neighbour_table(self.rows, self.cols)[cell] if n in self._mine)

    def reveal(self, cell: int) -> list[int]:
        """Reveal `cell` and return every cell that became visible. Revealing a flagged cell does nothing.

        A mine ends the game. A zero opens its neighbours, and any neighbour that is also zero opens
        its own, so the reveal spreads as a breadth-first flood fill.
        """
        if self.over or self.revealed[cell] != UNKNOWN or cell in self.flags:
            return []
        if self._mine is None:
            self._place_mines(cell)
        self.reveals += 1
        if cell in self._mine:
            self.lost_on = cell
            self.revealed[cell] = MINE_SHOWN
            return [cell]
        opened = [cell]
        self.revealed[cell] = self.adjacent_mines(cell)
        queue = deque([cell])
        table = neighbour_table(self.rows, self.cols)
        while queue:
            here = queue.popleft()
            if self.revealed[here] != 0:  # only zeros spread the reveal further
                continue
            for n in table[here]:
                if self.revealed[n] == UNKNOWN and n not in self.flags:
                    self.revealed[n] = self.adjacent_mines(n)
                    opened.append(n)
                    queue.append(n)
        return opened

    def toggle_flag(self, cell: int) -> bool:
        """Flag or unflag an unrevealed cell. Flags are the player's notes: they never change the rules."""
        if self.revealed[cell] != UNKNOWN or self.over:
            return False
        self.flags ^= {cell}
        return True

    def view(self) -> View:
        """The public state an agent may use: the numbers shown, and nothing about the hidden layout."""
        cells = tuple(UNKNOWN if v == MINE_SHOWN else v for v in self.revealed)
        return View(self.rows, self.cols, self.mines, cells)


def col_name(col: int) -> str:
    """Spreadsheet-style column name: a..z, then aa, ab, ... (30 columns need the second letter)."""
    name = ""
    n = col + 1
    while n:
        n, rem = divmod(n - 1, 26)
        name = chr(ord("a") + rem) + name
    return name


def label(cols: int, cell: int) -> str:
    """Name a cell as column letters then row number, both from 1, like `c5` or `ad12`."""
    r, c = divmod(cell, cols)
    return f"{col_name(c)}{r + 1}"


_CELL_RE = re.compile(r"^([a-z]+)(\d+)$")


def parse_cell(rows: int, cols: int, text: str) -> int | None:
    """Read a typed cell such as `c5` or `AD12`. Returns None if it does not name a cell on this board."""
    m = _CELL_RE.match(text.strip().lower())
    if not m:
        return None
    letters, digits = m.groups()
    col = 0
    for ch in letters:  # bijective base 26: a=1 .. z=26, aa=27
        col = col * 26 + (ord(ch) - ord("a") + 1)
    col -= 1
    row = int(digits) - 1
    if not (0 <= row < rows and 0 <= col < cols):
        return None
    return row * cols + col
