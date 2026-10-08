"""The checkers agents and their strength levels, shared by the terminal game and the web API.

choose() runs one agent on one position and returns its move plus a summary of
the search (depth reached, positions examined, branches pruned, and a score for
every root move), already in JSON-friendly form.
"""

from __future__ import annotations

import time

from .board import legal_moves, notation
from .search import WIN, AlphaBeta, Minimax

# Level -> budget. Alpha-beta gets a time limit (iterative deepening goes as deep
# as it can in that time); plain minimax gets a fixed depth because, without
# pruning, each extra ply multiplies its work by the branching factor (~8).
ALPHABETA_SECONDS = {1: 0.2, 2: 0.8, 3: 2.0}
MINIMAX_DEPTH = {1: 2, 2: 4, 3: 5}

DESCRIPTIONS = {
    "alphabeta": (
        "Alpha-beta search: assumes you will always answer with the reply that is worst for it, and skips "
        "lines it can prove are worse than one already found. Iterative deepening, a transposition table, "
        "and best-move-first ordering let it look deeper in the same time."
    ),
    "minimax": (
        "Plain minimax to a fixed depth: the same reasoning, but it examines every line. Compare its node "
        "count with alpha-beta's to see how much pruning saves."
    ),
}


def move_json(m) -> dict:
    """A Move as plain lists, plus its standard notation (squares numbered 1-32)."""
    return {"path": list(m.path), "captured": list(m.captured), "result": list(m.result), "notation": notation(m)}


def describe(score):
    """Turn a raw search score into something readable.

    Scores within 200 of WIN mean a forced win (or loss) was found; the distance
    from WIN is how many plies away it is. Anything else is a heuristic score.
    """
    if score is None:
        return None
    if score >= WIN - 200:
        return {"result": "win", "plies": WIN - score}
    if score <= -(WIN - 200):
        return {"result": "loss", "plies": WIN + score}
    return {"result": "eval", "score": score}


def choose(board, turn: int, agent: str = "alphabeta", level: int = 2) -> dict:
    """Pick a move for `turn` (+1 red, -1 white) with the named agent at the given level.

    Returns {"move", "analysis", "reply"}: the chosen move, what the search saw,
    and the opponent's legal replies (so a client doesn't need a second request).
    """
    if agent == "alphabeta":
        searcher = AlphaBeta(max_seconds=ALPHABETA_SECONDS[level])
    else:
        searcher = Minimax(MINIMAX_DEPTH[level])
    start = time.perf_counter()
    info = searcher.search(board, turn)
    reply_moves = legal_moves(info.move.result, -turn)
    return {
        "move": move_json(info.move),
        "analysis": {
            "agent": agent,
            "depth": info.depth,
            "nodes": info.nodes,
            "cutoffs": info.cutoffs,
            "best": describe(info.score),
            # Root moves best-first; unscored moves (search ran out of time) go last.
            "root": [{"notation": notation(m), "path": list(m.path), "score": describe(sc)}
                     for m, sc in sorted(info.root, key=lambda x: -(x[1] if x[1] is not None else -WIN))],
            "seconds": round(time.perf_counter() - start, 3),
        },
        "reply": [move_json(m) for m in reply_moves],
    }
