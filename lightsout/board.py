"""Lights Out rules on an n x n grid of lights.

Cells are numbered row by row from the top-left, 0 .. n*n-1, so cell i sits at
row i // n and column i % n. A board is a tuple of n*n ints: 1 = lit, 0 = dark.
Pressing a cell toggles it and its orthogonal neighbours; the goal is all dark.

Boards are also stored as integers (bit i = cell i). That makes a press a single
XOR of neighbourhood masks, which is the same arithmetic the solver uses over
GF(2): addition mod 2 is XOR, so presses combine by XOR and order does not matter.
"""

from __future__ import annotations

import random
from collections.abc import Iterable

MIN_N, MAX_N, DEFAULT_N = 3, 9, 5

_STEPS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def neighbourhood(n: int, cell: int) -> int:
    """Bit mask of what a press on `cell` toggles: the cell itself and its orthogonal neighbours."""
    r, c = divmod(cell, n)
    mask = 1 << cell
    for dr, dc in _STEPS:
        rr, cc = r + dr, c + dc
        if 0 <= rr < n and 0 <= cc < n:
            mask |= 1 << (rr * n + cc)
    return mask


def to_mask(board: Iterable[int]) -> int:
    """Pack a 0/1 board into an integer, bit i = cell i."""
    mask = 0
    for i, v in enumerate(board):
        if v:
            mask |= 1 << i
    return mask


def to_board(n: int, mask: int) -> tuple[int, ...]:
    """Unpack an integer into an n*n tuple of 0/1."""
    return tuple((mask >> i) & 1 for i in range(n * n))


def press_mask(n: int, presses: Iterable[int]) -> int:
    """XOR of the neighbourhoods of the pressed cells: the total toggle pattern.

    Pressing a cell twice cancels out, which is why a solution never needs repeats.
    """
    mask = 0
    for cell in presses:
        mask ^= neighbourhood(n, cell)
    return mask


def apply_presses(n: int, board: Iterable[int], presses: Iterable[int]) -> tuple[int, ...]:
    """The board after pressing each cell in `presses` (order does not matter)."""
    return to_board(n, to_mask(board) ^ press_mask(n, presses))


def is_dark(board: Iterable[int]) -> bool:
    """True once every light is off, the goal state."""
    return not any(board)


def random_board(n: int, rng: random.Random, solvable: bool = True) -> tuple[int, ...]:
    """A random board that is not already solved.

    solvable=True picks a random press pattern and applies it to the dark board. The
    result is reachable by construction, so it has a solution (the reverse of that
    pattern works, since every press is its own inverse). solvable=False draws every
    light independently, which is solvable only with probability 2**rank / 2**(n*n).
    """
    while True:
        if solvable:
            presses = [i for i in range(n * n) if rng.random() < 0.5]
            board = apply_presses(n, (0,) * (n * n), presses)
        else:
            board = tuple(rng.randint(0, 1) for _ in range(n * n))
        if not is_dark(board):
            return board


def label(n: int, cell: int) -> str:
    """Spreadsheet-style name: column letter then row number, both 1-based (cell 6 on 5x5 is 'b2')."""
    r, c = divmod(cell, n)
    return f"{chr(ord('a') + c)}{r + 1}"


def parse_cell(n: int, text: str) -> int | None:
    """Read a cell typed by the player, or None if it does not name one on this board.

    Two forms are accepted, both 1-based:
      "b3"   column b, row 3 (letter then number, like a spreadsheet)
      "3 2"  row 3, column 2 (row then column, separated by a space or comma)
    """
    text = text.strip().lower()
    row = col = None
    if len(text) >= 2 and text[0].isalpha() and text[1:].strip().isdigit():
        col, row = ord(text[0]) - ord("a"), int(text[1:])
        row -= 1
    else:
        parts = text.replace(",", " ").split()
        if len(parts) == 2 and all(p.isdigit() for p in parts):
            row, col = int(parts[0]) - 1, int(parts[1]) - 1
    if row is None or col is None or not (0 <= row < n and 0 <= col < n):
        return None
    return row * n + col
