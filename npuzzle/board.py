"""N-Puzzle boards and search nodes.

A board is a tuple of n*n tiles read row by row, with 0 for the blank. The
goal puts the blank first, (0, 1, ..., n*n - 1), the convention of Korf's
15-puzzle benchmarks. Actions name the direction the blank moves.
"""

from __future__ import annotations

import math
from functools import cache

ACTIONS = ("Up", "Down", "Left", "Right")
INVERSE = {"Up": "Down", "Down": "Up", "Left": "Right", "Right": "Left"}


def size_of(board) -> int:
    n = math.isqrt(len(board))
    if n < 2 or n * n != len(board):
        raise ValueError(f"a board needs a square number of tiles, got {len(board)}")
    return n


def validate(board) -> tuple[int, ...]:
    """Return board as a tuple, or raise ValueError if it is not a permutation of 0..n*n-1."""
    board = tuple(board)
    n = size_of(board)
    if sorted(board) != list(range(n * n)):
        raise ValueError(f"board must contain each tile 0..{n * n - 1} exactly once")
    return board


@cache
def goal(n: int) -> tuple[int, ...]:
    return tuple(range(n * n))


@cache
def neighbors(n: int) -> tuple[tuple[tuple[int, str], ...], ...]:
    """neighbors(n)[blank] = ((index the blank moves to, action), ...), in ACTIONS order."""
    table = []
    for i in range(n * n):
        row, col = divmod(i, n)
        moves = []
        if row > 0:
            moves.append((i - n, "Up"))
        if row < n - 1:
            moves.append((i + n, "Down"))
        if col > 0:
            moves.append((i - 1, "Left"))
        if col < n - 1:
            moves.append((i + 1, "Right"))
        table.append(tuple(moves))
    return tuple(table)


def successors(board: tuple[int, ...], n: int):
    """Yield (action, next board) for every legal blank move."""
    blank = board.index(0)
    for target, action in neighbors(n)[blank]:
        nxt = list(board)
        nxt[blank], nxt[target] = board[target], 0
        yield action, tuple(nxt)


def apply(board: tuple[int, ...], action: str, n: int) -> tuple[int, ...]:
    """Board after one blank move; raises ValueError if the move leaves the board."""
    for a, nxt in successors(board, n):
        if a == action:
            return nxt
    raise ValueError(f"cannot move the blank {action}")


def replay(board, path, n: int) -> tuple[int, ...]:
    for action in path:
        board = apply(board, action, n)
    return board


def is_solvable(board, n: int) -> bool:
    """True if the goal is reachable.

    Counts inversions among the tiles (blank excluded). For odd n the count
    must be even. For even n each vertical move changes it by n - 1, an odd
    number, and the goal has the blank in row 0, so inversions plus the
    blank's row must be even.
    """
    tiles = [t for t in board if t]
    inversions = sum(1 for i in range(len(tiles)) for j in range(i + 1, len(tiles)) if tiles[i] > tiles[j])
    if n % 2:
        return inversions % 2 == 0
    return (inversions + board.index(0) // n) % 2 == 0


def format_board(board, n: int) -> str:
    width = len(str(n * n - 1))
    return "\n".join(" ".join(str(t).rjust(width) if t else "_".rjust(width)
                              for t in board[r * n:(r + 1) * n]) for r in range(n))


class Node:
    """A board reached during search, with the move that reached it.

    `depth` is the number of moves from the start (g). Following `parent`
    links back to the start rebuilds the path.
    """

    __slots__ = ("board", "parent", "action", "depth")

    def __init__(self, board, parent=None, action=None, depth=0):
        self.board = board
        self.parent = parent
        self.action = action
        self.depth = depth

    def child(self, action, board):
        return Node(board, self, action, self.depth + 1)

    def path(self) -> list[str]:
        actions, node = [], self
        while node.parent is not None:
            actions.append(node.action)
            node = node.parent
        actions.reverse()
        return actions
