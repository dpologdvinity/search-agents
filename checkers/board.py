"""English checkers (8x8 draughts) rules on a 32-square board.

Squares are the 32 dark squares, numbered row by row from the top:

     .  0  .  1  .  2  .  3        row 0
     4  .  5  .  6  .  7  .        row 1
     .  8  .  9  . 10  . 11        row 2
    ...
    28  . 29  . 30  . 31  .        row 7

A board is a tuple of 32 ints: +1 red man, +2 red king, -1 white man,
-2 white king, 0 empty. Red starts on squares 20-31, moves up the board,
and moves first. Captures are mandatory; a capture continues while more
jumps are available; a man reaching the far row is crowned, which ends the
move. A side with no legal move loses.

A move is (path, captured, result): the squares visited, the squares of the
pieces taken, and the board afterwards.
"""

from __future__ import annotations

from typing import NamedTuple

RED, WHITE = 1, -1
UP_LEFT, UP_RIGHT, DOWN_LEFT, DOWN_RIGHT = 0, 1, 2, 3
_DIRS = ((-1, -1), (-1, 1), (1, -1), (1, 1))
FORWARD = {RED: (UP_LEFT, UP_RIGHT), WHITE: (DOWN_LEFT, DOWN_RIGHT)}
ALL_DIRS = (UP_LEFT, UP_RIGHT, DOWN_LEFT, DOWN_RIGHT)
CROWN_ROW = {RED: 0, WHITE: 7}


def rc(i: int) -> tuple[int, int]:
    r = i // 4
    return r, 2 * (i % 4) + (1 if r % 2 == 0 else 0)


def index(r: int, c: int) -> int:
    """Square number of a dark square, or -1 if off the board."""
    if 0 <= r < 8 and 0 <= c < 8 and (r + c) % 2 == 1:
        return r * 4 + c // 2
    return -1


ADJ = tuple(tuple(index(rc(i)[0] + dr, rc(i)[1] + dc) for dr, dc in _DIRS) for i in range(32))
JUMP = tuple(tuple(index(rc(i)[0] + 2 * dr, rc(i)[1] + 2 * dc) for dr, dc in _DIRS) for i in range(32))
ROW = tuple(i // 4 for i in range(32))

START: tuple[int, ...] = (-1,) * 12 + (0,) * 8 + (1,) * 12


class Move(NamedTuple):
    path: tuple[int, ...]
    captured: tuple[int, ...]
    result: tuple[int, ...]


def _captures_from(board, sq, side):
    piece = board[sq]
    king = abs(piece) == 2
    dirs = ALL_DIRS if king else FORWARD[side]
    out = []

    def extend(b, cur, path, taken):
        found = False
        for d in dirs:
            mid, land = ADJ[cur][d], JUMP[cur][d]
            if land < 0 or mid in taken or b[land] != 0 or b[mid] * side >= 0:
                continue
            found = True
            nb = list(b)
            nb[cur] = 0
            crowned = not king and ROW[land] == CROWN_ROW[side]
            nb[land] = 2 * side if crowned else piece
            if crowned:  # crowning ends the move
                out.append((path + (land,), taken + (mid,), nb))
            else:
                extend(nb, land, path + (land,), taken + (mid,))
        if not found and taken:
            out.append((path, taken, list(b)))

    extend(board, sq, (sq,), ())
    moves = []
    for path, taken, nb in out:
        for t in taken:  # captured pieces leave the board when the move ends
            nb[t] = 0
        moves.append(Move(path, taken, tuple(nb)))
    return moves


def legal_moves(board, side) -> list[Move]:
    """Every legal move for side; captures only, if any capture exists."""
    captures = []
    for sq in range(32):
        if board[sq] * side > 0:
            captures.extend(_captures_from(board, sq, side))
    if captures:
        return captures
    moves = []
    for sq in range(32):
        piece = board[sq]
        if piece * side <= 0:
            continue
        for d in (ALL_DIRS if abs(piece) == 2 else FORWARD[side]):
            to = ADJ[sq][d]
            if to < 0 or board[to] != 0:
                continue
            nb = list(board)
            nb[sq] = 0
            nb[to] = 2 * side if abs(piece) == 1 and ROW[to] == CROWN_ROW[side] else piece
            moves.append(Move((sq, to), (), tuple(nb)))
    return moves


def perft(board, side, depth) -> int:
    """Number of move sequences of the given length (rules self-check)."""
    if depth == 0:
        return 1
    return sum(perft(m.result, -side, depth - 1) for m in legal_moves(board, side))


def notation(move: Move) -> str:
    """Standard notation: squares numbered 1-32, '-' for a step, 'x' for a capture."""
    sep = "x" if move.captured else "-"
    return sep.join(str(s + 1) for s in move.path)


def parse_board(cells) -> tuple[int, ...]:
    cells = tuple(int(v) for v in cells)
    if len(cells) != 32 or any(v not in (-2, -1, 0, 1, 2) for v in cells):
        raise ValueError("board must be 32 values from -2 to 2")
    return cells
