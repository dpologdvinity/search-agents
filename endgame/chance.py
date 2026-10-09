"""The chance opponent: picks each move from fixed odds, with no search, evaluation or lookahead.

It plays like the tile spawner in 2048 (a 2 with 90%, a 4 with 10%): the odds depend only on what kind of
move a legal move is, never on what the move leads to. The table plays for either side, the white king and
piece or the lone black king, and it never looks at the result of a move.

How a move is picked:
1. Each legal move gets one category, checked in this order: capture (the lone king takes the piece),
   check (the move leaves the other side in check), king_centre (a king steps closer to the middle of the
   board), and other (everything else). The first category that fits is the one used.
2. The move's weight is the table weight of its category.
3. The move is drawn with probability weight / (sum of the weights of all legal moves). That is the
   renormalisation over the legal moves: the table is never used directly, only its shares.

The same draw is made from a uniform number u in [0, 1): see pick_at. The page (web/js/endgame-core.js) uses
the same table and the same rule, and tests/test_endgame_parity.py checks the two agree.
"""

from __future__ import annotations

import random

from .tablebase import MoveInfo

KEY = "chance"
LABEL = "Chance (fixed odds)"
TABLEBASE_KEY = "tablebase"
TABLEBASE_LABEL = "Tablebase (exact)"

# The fixed odds, as weights. Only the ratios matter: 4 : 2 : 2 : 1 is 4/9, 2/9, 2/9 and 1/9 when a move in
# every category is legal. Why these numbers: a capture of the piece is the most tempting move (it ends the
# game, a draw for the lone king), so it gets the most weight; a check forces a reply, the most common
# tactic in these endings, so it comes next; moving the king toward the centre is the technique that wins
# these endings, so it shares the second weight; every other move gets the base weight of 1. The odds are a
# fixed bias, not a model of the opponent: nothing here reads the position beyond the move's category.
CHANCE_WEIGHTS = {
    "capture": 4,
    "check": 2,
    "king_centre": 2,
    "other": 1,
}

CATEGORY_LABELS = {
    "capture": "takes the piece (the lone king's only capture)",
    "check": "gives check",
    "king_centre": "a king steps toward the centre",
    "other": "any other legal move",
}


def centre_distance(sq: int) -> int:
    """How far a square is from the middle of the board, doubled so it stays an integer.

    Squares are 8 * rank + file, both from 0. The four centre squares (d4, e4, d5, e5) give 1, the edges give
    7. Chebyshev distance is the king's own metric, so a king move that lowers this number brings the king
    closer to the centre in one step.
    """
    f, r = sq % 8, sq // 8
    return max(abs(2 * f - 7), abs(2 * r - 7))


def move_category(info: MoveInfo, pos) -> str:
    """The category of a legal move from pos: capture, check, king_centre or other (first match wins)."""
    mover, src, dst = info.move
    if mover == "k" and dst == pos[1]:
        return "capture"
    if info.check:
        return "check"
    if mover in ("K", "k") and centre_distance(dst) < centre_distance(src):
        return "king_centre"
    return "other"


def weights(infos: list[MoveInfo], pos) -> list[int]:
    """The table weight of each legal move, in the same order as infos."""
    return [CHANCE_WEIGHTS[move_category(info, pos)] for info in infos]


def probabilities(infos: list[MoveInfo], pos) -> list[float]:
    """The chance of each legal move: its weight over the total of the legal moves' weights. Sums to 1."""
    ws = weights(infos, pos)
    total = sum(ws)
    return [w / total for w in ws]


def pick_at(infos: list[MoveInfo], pos, u: float) -> MoveInfo | None:
    """The move chosen by the uniform number u in [0, 1), by walking the cumulative weights.

    The move whose slice of [0, total) contains u * total is chosen. Exposed separately from pick() so the
    page and the tests can feed in the same numbers and check that the two languages choose the same move.
    """
    if not infos:
        return None
    ws = weights(infos, pos)
    threshold = u * sum(ws)
    acc = 0
    for info, w in zip(infos, ws):
        acc += w
        if threshold < acc:
            return info
    return infos[-1]  # only reachable through rounding when u is extremely close to 1


def pick(infos: list[MoveInfo], pos, rng: random.Random) -> MoveInfo | None:
    """Choose a legal move by the fixed odds. The rng makes the choice repeatable for a given seed."""
    return pick_at(infos, pos, rng.random())


def table() -> dict:
    """The table as JSON for /meta and the page: the weight of each category and what it means."""
    return {
        "key": KEY,
        "label": LABEL,
        "weights": [
            {"category": cat, "weight": w, "meaning": CATEGORY_LABELS[cat]}
            for cat, w in CHANCE_WEIGHTS.items()
        ],
        "rule": "a legal move's chance is its category's weight over the total weight of all legal moves",
    }

