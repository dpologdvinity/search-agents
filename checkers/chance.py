"""The chance agent: picks a legal move from a fixed table of weights, with no search.

It works like the 2048 tile spawner, which places a 2 with probability 90% and a 4 with 10%:
each legal move gets a weight from its type, the weights are renormalised over the legal
moves, and one move is sampled. It does not look at the position beyond the move's type, so
it is a baseline for "moves that look sensible by type" rather than a player.
"""

from __future__ import annotations

import random
import time

from .board import geometry, legal_moves
from .search import SearchInfo

# Weight of each move type. Captures are the most likely, then men moving into the centre,
# then other man moves, then king moves. The sampling renormalises these over the legal moves.
CHANCE_WEIGHTS = {"capture": 5, "centre_man": 3, "man": 2, "king": 1}


def move_kind(board, move) -> str:
    """The type of a move: capture, king, a man moving into the centre, or any other man move.

    A capture counts as a capture even when a king makes it. Centre squares come from the
    board's geometry, so the same rule holds on every size.
    """
    if move.captured:
        return "capture"
    if abs(board[move.path[0]]) == 2:
        return "king"
    return "centre_man" if move.path[-1] in geometry(board).CENTER else "man"


class Chance:
    """Samples a legal move with probability proportional to CHANCE_WEIGHTS[move_kind].

    The generator is seeded, so the same seed gives the same sequence of choices; choose()
    seeds it from the position, so the same position always gets the same move.
    """

    def __init__(self, seed=0, forced=True):
        self.rng = random.Random(seed)
        self.forced = forced

    def search(self, board, side) -> SearchInfo:
        """Choose one legal move by its weight. Nothing is scored, so the root list has no scores."""
        start = time.perf_counter()
        moves = legal_moves(board, side, self.forced)
        if not moves:
            raise ValueError("no legal moves")
        weights = [CHANCE_WEIGHTS[move_kind(board, m)] for m in moves]
        # random.choices divides by the total weight, which is the renormalisation over the legal moves.
        move = self.rng.choices(moves, weights=weights, k=1)[0]
        return SearchInfo(move, None, 0, 0, 0, time.perf_counter() - start, [(m, None) for m in moves])


def odds_text() -> str:
    """The weight table in words, for the description shown on the page and in the CLI."""
    w = CHANCE_WEIGHTS
    return (f"picks captures {w['capture']}×, men moving into the centre {w['centre_man']}×, "
            f"other man moves {w['man']}×, king moves {w['king']}×")
