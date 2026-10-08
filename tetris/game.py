"""Seeded games for training and benchmarks: a 7-bag piece sequence, policies, and a capped game loop.

The piece sequence uses the 7-bag rule that modern Tetris uses: each bag holds one of every piece in a
shuffled order, so no piece goes more than 12 draws without appearing. A seed fixes the whole sequence,
so every strategy plays the same games.

A policy is a function (board, piece, preview) -> Move or None. None means the piece has no legal
placement and the game is over (a top-out). A cap on pieces keeps games finite: a good bot can play for
millions of lines, and benchmarks need to stop. A game that reaches the cap is recorded as capped, not
as a loss.
"""

from __future__ import annotations

import random
from collections.abc import Callable
from dataclasses import dataclass

from .board import H, empty_board
from .pieces import PIECES
from .search import Move, best_move, legal_moves

Policy = Callable[[tuple, str, str], "Move | None"]


class PieceBag:
    """Seeded 7-bag: shuffle the seven pieces, deal them out, then shuffle again."""

    def __init__(self, seed: int):
        self._rng = random.Random(seed)
        self._queue: list[str] = []

    def _fill(self, n: int) -> None:
        while len(self._queue) < n:
            bag = list(PIECES)
            self._rng.shuffle(bag)
            self._queue.extend(bag)

    def take(self) -> str:
        self._fill(1)
        return self._queue.pop(0)

    def peek(self) -> str:
        """The piece after the current one, shown as the preview. Does not consume it."""
        self._fill(1)
        return self._queue[0]


def greedy_policy(weights, top_k: int | None = None) -> Policy:
    """One-piece greedy: the best-scoring placement of the current piece.

    top_k is unused here; it exists so that lookahead_policy and greedy_policy share a signature.
    """

    def policy(board, piece, preview):
        move, _ = best_move(board, piece, weights)
        return move

    return policy


def lookahead_policy(weights, top_k: int = 6) -> Policy:
    """One-piece lookahead with the preview piece, pruned to the top_k candidates (see search.best_move)."""

    def policy(board, piece, preview):
        move, _ = best_move(board, piece, weights, preview=preview, top_k=top_k)
        return move

    return policy


def random_policy(seed: int) -> Policy:
    """Baseline: a uniformly random legal placement. Seeded so its games repeat too."""
    rng = random.Random(seed)

    def policy(board, piece, preview):
        moves = legal_moves(board, piece, (0.0,) * 9)
        return rng.choice(moves) if moves else None

    return policy


@dataclass(frozen=True)
class GameResult:
    lines: int
    pieces: int  # pieces placed
    topped_out: bool  # the game ended because a piece had no legal placement
    capped: bool  # the game stopped at the piece cap while still alive


def play(policy: Policy, seed: int, max_pieces: int | None = None, height: int = H) -> GameResult:
    """Play one seeded game on a board `height` rows tall and return its lines, piece count and how it ended."""
    bag = PieceBag(seed)
    board = empty_board(height)
    lines = 0
    pieces = 0
    while True:
        if max_pieces is not None and pieces >= max_pieces:
            return GameResult(lines, pieces, topped_out=False, capped=True)
        piece = bag.take()
        preview = bag.peek()
        move = policy(board, piece, preview)
        if move is None:
            return GameResult(lines, pieces, topped_out=True, capped=False)
        board = move.board
        lines += move.lines
        pieces += 1
