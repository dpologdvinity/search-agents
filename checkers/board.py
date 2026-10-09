"""English checkers (American draughts) rules on 8x8, 10x10 and 12x12 boards.

Squares are the dark squares, numbered row by row from the top. An n x n board has
n*n/2 of them, n/2 per row. The 8x8 board:

     .  0  .  1  .  2  .  3        row 0
     4  .  5  .  6  .  7  .        row 1
     .  8  .  9  . 10  . 11        row 2
    ...
    28  . 29  . 30  . 31  .        row 7

A board is a tuple of n*n/2 ints: +1 red man, +2 red king, -1 white man,
-2 white king, 0 empty. The length of the tuple gives the size (32 cells is 8x8,
50 is 10x10, 72 is 12x12), so every function takes just the board. Each side starts
with 3, 4 or 5 rows of men on the 8x8, 10x10 and 12x12 boards. Red starts on the bottom
rows, moves up the board, and moves first.

The same American rules apply on every size: men move and capture forward only, kings
move and capture one step in any direction (no flying kings), and a man that reaches
the far row is crowned, which ends its move. A side with no legal move loses.

Captures come in two variants:

- forced (the default, standard rules): if any capture exists, a capture must be played.
- unforced: captures are optional, so the legal moves are every simple move plus every
  capture chain.

In both variants a capture chain is listed whole. Once a piece starts jumping it must
keep jumping while a jump is available, and the player cannot stop partway through.
Only a crowning ends a chain early, as in the standard rules.

A move is (path, captured, result): the squares visited, the squares of the pieces
taken, and the board afterwards.
"""

from __future__ import annotations

from typing import NamedTuple

RED, WHITE = 1, -1
UP_LEFT, UP_RIGHT, DOWN_LEFT, DOWN_RIGHT = 0, 1, 2, 3
_DIRS = ((-1, -1), (-1, 1), (1, -1), (1, 1))
FORWARD = {RED: (UP_LEFT, UP_RIGHT), WHITE: (DOWN_LEFT, DOWN_RIGHT)}
ALL_DIRS = (UP_LEFT, UP_RIGHT, DOWN_LEFT, DOWN_RIGHT)
SIZES = (8, 10, 12)
# Rows of men each side starts with. Two empty rows separate the armies on every size.
ROWS_PER_SIDE = {8: 3, 10: 4, 12: 5}


class Geometry:
    """Square tables for one board size, built once per size and shared.

    ADJ[sq][d] is the square one step from sq in direction d (or -1 off the board),
    and JUMP[sq][d] is the square two steps away, where the jump lands. Tables are
    indexed by direction, in the order UP_LEFT, UP_RIGHT, DOWN_LEFT, DOWN_RIGHT.
    """

    def __init__(self, n: int):
        self.n = n
        self.half = n // 2  # dark squares per row
        self.cells = n * self.half
        self.rc = tuple(self._rc(i) for i in range(self.cells))
        self.ADJ = tuple(tuple(self.index(r + dr, c + dc) for dr, dc in _DIRS) for r, c in self.rc)
        self.JUMP = tuple(tuple(self.index(r + 2 * dr, c + 2 * dc) for dr, dc in _DIRS) for r, c in self.rc)
        self.ROW = tuple(i // self.half for i in range(self.cells))
        # The far row for each side: a man that reaches it is crowned.
        self.CROWN = {RED: 0, WHITE: n - 1}
        # Centre control: the central 4x4 block (rows and columns 2-5 on 8x8).
        lo = (n - 4) // 2
        self.CENTER = frozenset(i for i, (r, c) in enumerate(self.rc) if lo <= r <= lo + 3 and lo <= c <= lo + 3)
        k = ROWS_PER_SIDE[n] * self.half  # men per side
        self.START = (-1,) * k + (0,) * (self.cells - 2 * k) + (1,) * k

    def _rc(self, i: int) -> tuple[int, int]:
        r = i // self.half
        return r, 2 * (i % self.half) + (1 if r % 2 == 0 else 0)

    def index(self, r: int, c: int) -> int:
        """Square number of a dark square, or -1 if off the board or on a light square."""
        if 0 <= r < self.n and 0 <= c < self.n and (r + c) % 2 == 1:
            return r * self.half + c // 2
        return -1


_GEOMETRY = {n: Geometry(n) for n in SIZES}
_BY_CELLS = {g.cells: g for g in _GEOMETRY.values()}
START: tuple[int, ...] = _GEOMETRY[8].START  # the 8x8 opening; other sizes use start_position()


def geometry(board) -> Geometry:
    """The tables for this board, found from its length."""
    try:
        return _BY_CELLS[len(board)]
    except KeyError:
        raise ValueError(f"board must have 32, 50 or 72 squares, not {len(board)}") from None


def for_size(n: int) -> Geometry:
    """The tables for an n x n board: n must be 8, 10 or 12."""
    if n not in _GEOMETRY:
        raise ValueError(f"board size must be 8, 10 or 12, not {n}")
    return _GEOMETRY[n]


def start_position(n: int = 8) -> tuple[int, ...]:
    """The opening position on an n x n board."""
    return for_size(n).START


def index(r: int, c: int, n: int = 8) -> int:
    """Square number of a dark square on an n x n board (8x8 by default)."""
    return for_size(n).index(r, c)


class Move(NamedTuple):
    path: tuple[int, ...]
    captured: tuple[int, ...]
    result: tuple[int, ...]


def _captures_from(board, sq, side, geo):
    """Every maximal capture chain that starts with the piece on sq."""
    piece = board[sq]
    king = abs(piece) == 2
    dirs = ALL_DIRS if king else FORWARD[side]
    ADJ, JUMP, ROW, CROWN = geo.ADJ, geo.JUMP, geo.ROW, geo.CROWN
    out = []

    def extend(b, cur, path, taken):
        found = False
        for d in dirs:
            mid, land = ADJ[cur][d], JUMP[cur][d]
            # A jump needs an enemy piece on mid that has not already been taken in this
            # chain (taken pieces stay on the board until the move ends), and an empty landing.
            if land < 0 or mid in taken or b[land] != 0 or b[mid] * side >= 0:
                continue
            found = True
            nb = list(b)
            nb[cur] = 0
            crowned = not king and ROW[land] == CROWN[side]
            nb[land] = 2 * side if crowned else piece
            if crowned:  # crowning ends the move
                out.append((path + (land,), taken + (mid,), nb))
            else:
                extend(nb, land, path + (land,), taken + (mid,))
        # A chain ends only when no jump is available, so a started capture is always finished.
        if not found and taken:
            out.append((path, taken, list(b)))

    extend(board, sq, (sq,), ())
    moves = []
    for path, taken, nb in out:
        for t in taken:  # captured pieces leave the board when the move ends
            nb[t] = 0
        moves.append(Move(path, taken, tuple(nb)))
    return moves


def _simple_moves(board, side, geo):
    """Every one-step move for side (men forward, kings any direction), crowning where it lands."""
    ADJ, ROW, CROWN = geo.ADJ, geo.ROW, geo.CROWN
    moves = []
    for sq in range(geo.cells):
        piece = board[sq]
        if piece * side <= 0:
            continue
        for d in (ALL_DIRS if abs(piece) == 2 else FORWARD[side]):
            to = ADJ[sq][d]
            if to < 0 or board[to] != 0:
                continue
            nb = list(board)
            nb[sq] = 0
            nb[to] = 2 * side if abs(piece) == 1 and ROW[to] == CROWN[side] else piece
            moves.append(Move((sq, to), (), tuple(nb)))
    return moves


def legal_moves(board, side, forced=True) -> list[Move]:
    """Every legal move for side, under forced captures (default) or optional captures.

    With forced=True the list holds only captures when any capture exists. With
    forced=False it holds the capture chains and then the simple moves.
    """
    geo = geometry(board)
    captures = []
    for sq in range(geo.cells):
        if board[sq] * side > 0:
            captures.extend(_captures_from(board, sq, side, geo))
    if captures and forced:
        return captures
    return captures + _simple_moves(board, side, geo)


def perft(board, side, depth, forced=True) -> int:
    """Number of move sequences of the given length (rules self-check)."""
    if depth == 0:
        return 1
    return sum(perft(m.result, -side, depth - 1, forced) for m in legal_moves(board, side, forced))


def notation(move: Move) -> str:
    """Standard notation: squares numbered from 1, '-' for a step, 'x' for a capture."""
    sep = "x" if move.captured else "-"
    return sep.join(str(s + 1) for s in move.path)


def parse_board(cells, size: int | None = None) -> tuple[int, ...]:
    """Validate a board given as a list of ints.

    The length must be 32, 50 or 72 and must match size when one is given. Each value
    must be from -2 to 2. Raises ValueError otherwise.
    """
    cells = tuple(int(v) for v in cells)
    geo = geometry(cells)
    if size is not None and geo.n != size:
        raise ValueError(f"a {size}x{size} board has {for_size(size).cells} squares, not {len(cells)}")
    if any(v not in (-2, -1, 0, 1, 2) for v in cells):
        raise ValueError("board values must be from -2 to 2")
    return cells
