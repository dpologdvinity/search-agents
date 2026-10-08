"""Hex rules on an n x n rhombus: cell indexing, adjacency, connection search, and the random playout.

Hex is played on a rhombus of hexagons. Cell (row r, column c) is stored at index r * n + c.
DOWN (player 1) wants a chain of its stones from the top row to the bottom row. ACROSS (player 2)
wants one from the left column to the right column. Because the board is a rhombus with the
acute corners at (0, 0) and (n-1, n-1), exactly one player can win a completed board, so
Hex has no draws. That fact drives the playout below.

Adjacency: (r, c) touches (r, c-1), (r, c+1), (r-1, c), (r+1, c), (r-1, c+1) and (r+1, c-1).
Drawn with row r shifted right by half a hexagon per row, these are the six hexagons around a cell.

Swap rule (optional, the "pie rule"): after the first stone, the second player may swap, which gives
that stone to the second player and passes the turn back. Encoded as the move SWAP (-1).
"""

from __future__ import annotations

import random
import re
from collections import deque
from functools import cache

MIN_N, MAX_N, DEFAULT_N = 5, 11, 7
EMPTY = 0
DOWN = 1    # connects the top row to the bottom row; moves first
ACROSS = 2  # connects the left column to the right column
SWAP = -1
SYMBOL = {EMPTY: ".", DOWN: "X", ACROSS: "O"}
NAME = {DOWN: "DOWN", ACROSS: "ACROSS"}


def other(player: int) -> int:
    """The opponent of `player`: 1 <-> 2."""
    return 3 - player


@cache
def neighbours(n: int) -> tuple[tuple[int, ...], ...]:
    """For every cell, the indexes of its on-board neighbours (up to six)."""
    out = []
    steps = ((0, -1), (0, 1), (-1, 0), (1, 0), (-1, 1), (1, -1))
    for r in range(n):
        for c in range(n):
            out.append(tuple((r + dr) * n + (c + dc) for dr, dc in steps
                             if 0 <= r + dr < n and 0 <= c + dc < n))
    return tuple(out)


@cache
def _edges(n: int, player: int) -> tuple[tuple[int, ...], bytes]:
    """The start edge (cells a chain begins on) and a 0/1 mask of the goal edge for `player`."""
    if player == DOWN:
        starts = tuple(range(n))                       # row 0
        goal = [(n - 1) * n + c for c in range(n)]     # row n-1
    else:
        starts = tuple(r * n for r in range(n))        # column 0
        goal = [r * n + n - 1 for r in range(n)]       # column n-1
    mask = bytearray(n * n)
    for g in goal:
        mask[g] = 1
    return starts, bytes(mask)


def connect(n: int, cells, player: int):
    """A shortest chain of `player`'s stones from its start edge to its goal edge, or None.

    Breadth-first search from every stone on the start edge. `parent` records how each stone was
    reached, so the chain can be read back from the goal once it is found. BFS gives a shortest chain,
    which is what the web page highlights.
    """
    starts, is_goal = _edges(n, player)
    nb = neighbours(n)
    parent: dict[int, int] = {}
    queue: deque[int] = deque()
    for s in starts:
        if cells[s] == player:
            parent[s] = -1
            queue.append(s)
    while queue:
        u = queue.popleft()
        if is_goal[u]:
            path = []
            while u != -1:
                path.append(u)
                u = parent[u]
            return path[::-1]
        for v in nb[u]:
            if v not in parent and cells[v] == player:
                parent[v] = u
                queue.append(v)
    return None


def winner_of(n: int, cells):
    """DOWN or ACROSS if one of them has a connecting chain, else None (the board is not finished)."""
    if connect(n, cells, DOWN) is not None:
        return DOWN
    if connect(n, cells, ACROSS) is not None:
        return ACROSS
    return None


def fill_playout(n: int, cells: list[int], to_move: int, rng: random.Random):
    """Finish the game at random by filling the board, then read off the winner.

    Shuffling the empty cells and alternating colours from `to_move` is the same as playing uniformly
    random legal moves to the end, because each random legal move is a uniformly random next cell.
    Filling is cheaper than stopping at the first connection: we need one connectivity check at the
    end instead of one per move. The winner is the same either way: a chain that exists at the moment
    of connection still exists on the full board, and two chains of opposite colours cannot both exist.

    `cells` is modified in place (pass a copy). Returns (winner, order), where `order` lists the cells
    in the order they were played. The search reuses `order` for its AMAF statistics.
    """
    order = [i for i, v in enumerate(cells) if v == EMPTY]
    rng.shuffle(order)
    player = to_move
    for i in order:
        cells[i] = player
        player = other(player)
    # On a full board exactly one player connects, so checking DOWN decides the game.
    winner = DOWN if connect(n, cells, DOWN) is not None else ACROSS
    return winner, order


def label(n: int, cell: int) -> str:
    """Spreadsheet-style name: column letter then row number, both from 1 at the top left, e.g. 'c4'."""
    if cell == SWAP:
        return "swap"
    r, c = divmod(cell, n)
    return f"{chr(97 + c)}{r + 1}"


_LABEL = re.compile(r"^([a-k])(\d{1,2})$")


def parse_cell(n: int, text: str):
    """The cell named by 'c4' (or 'swap' for the swap move), or None if the text names nothing on this board."""
    text = text.strip().lower()
    if text == "swap":
        return SWAP
    m = _LABEL.match(text)
    if not m:
        return None
    c = ord(m.group(1)) - 97
    r = int(m.group(2)) - 1
    if c >= n or not 0 <= r < n:
        return None
    return r * n + c


class Board:
    """A Hex position: the stones, whose turn it is, and the move history.

    `cells` holds 0 (empty), 1 (DOWN) or 2 (ACROSS). `moves` is the history, with SWAP as its own
    entry, so the turn is always len(moves) parity: DOWN moves at even indexes.
    """

    __slots__ = ("n", "cells", "to_move", "moves", "swap_rule")

    def __init__(self, n: int = DEFAULT_N, swap_rule: bool = False):
        if not MIN_N <= n <= MAX_N:
            raise ValueError(f"board size must be between {MIN_N} and {MAX_N}")
        self.n = n
        self.cells = [EMPTY] * (n * n)
        self.to_move = DOWN
        self.moves: list[int] = []
        self.swap_rule = swap_rule

    @classmethod
    def from_moves(cls, n: int, moves, swap_rule: bool = False) -> Board:
        """Replay a move history, raising ValueError at the first illegal move."""
        board = cls(n, swap_rule)
        for m in moves:
            board._apply(m)
        return board

    def copy(self) -> Board:
        b = Board.__new__(Board)
        b.n, b.cells, b.to_move = self.n, list(self.cells), self.to_move
        b.moves, b.swap_rule = list(self.moves), self.swap_rule
        return b

    def winner(self):
        return winner_of(self.n, self.cells)

    def chain(self):
        """The winning chain (cells, start edge first), or None while the game is open."""
        w = self.winner()
        return None if w is None else connect(self.n, self.cells, w)

    def legal(self) -> list[int]:
        """Every legal move: empty cells, plus SWAP if the swap rule allows it now."""
        if self.winner() is not None:
            return []
        moves = [i for i, v in enumerate(self.cells) if v == EMPTY]
        if self.swap_rule and len(self.moves) == 1:
            moves.append(SWAP)
        return moves

    def play(self, move: int) -> Board:
        """A new board after `move`. This board is not changed."""
        b = self.copy()
        b._apply(move)
        return b

    def _apply(self, move: int) -> None:
        if self.winner() is not None:
            raise ValueError("the game is already over")
        if move == SWAP:
            if not (self.swap_rule and len(self.moves) == 1):
                raise ValueError("swap is only allowed as the reply to the first move, with the swap rule on")
            first = self.moves[0]
            self.cells[first] = other(self.cells[first])  # the second player takes the stone
        else:
            if not 0 <= move < self.n * self.n:
                raise ValueError(f"cell {move} is off the board")
            if self.cells[move] != EMPTY:
                raise ValueError(f"{label(self.n, move)} is already taken")
            self.cells[move] = self.to_move
        self.moves.append(move)
        self.to_move = other(self.to_move)


def chain_marks(board: Board) -> dict[int, str]:
    """Cell -> '*' for every stone of the winning chain (empty while the game is open), for render()."""
    chain = board.chain()
    return {} if chain is None else {c: "*" for c in chain}


def render(board: Board, marks=None) -> str:
    """ASCII rhombus: X is DOWN (top to bottom), O is ACROSS (left to right), '.' is empty.

    Row r is indented by r spaces, which is what makes the board look like a rhombus. `marks` maps
    cell -> a one-character label, used to show hint or chain cells.
    """
    n = board.n
    lines = ["   " + " ".join(chr(97 + c) for c in range(n))]
    for r in range(n):
        syms = []
        for c in range(n):
            cell = r * n + c
            if marks and cell in marks:
                syms.append(marks[cell])
            else:
                syms.append(SYMBOL[board.cells[cell]])
        lines.append(f"{r + 1:>2} " + " " * r + " ".join(syms))
    return "\n".join(lines)
