"""Connect Four agents behind one interface, shared by the web server and the command line.

Each agent function takes a position and a strength level (1-3) and returns
{"move": column, "analysis": {...}}, where the analysis explains how the
agent decided (network priors and visit counts, minimax scores, or MCTS
visit counts). The "chance" agent is the no-skill baseline: it draws a column
from fixed odds (see CHANCE_WEIGHTS) and returns the odds as its analysis.
"""

from __future__ import annotations

import random

from .board import COLS, Board
from .mcts import MCTS
from .minimax import WIN, Minimax

# Search budget per level. Seconds cap minimax so a turn stays responsive.
ALPHAZERO_SIMS = {1: 50, 2: 200, 3: 800}
MINIMAX_SECONDS = {1: 0.1, 2: 0.5, 3: 2.0}
MCTS_SIMS = {1: 300, 2: 1500, 3: 5000}

# Fixed column weights for the chance opponent, which picks a move with no search at all.
# The weights rise toward the centre column because the centre sits in the most four-in-a-row
# lines, so a player that favours it looks like a casual human rather than a uniform coin toss.
# They are relative: the weight of column c is CHANCE_WEIGHTS[c] / sum, renormalised over the
# columns that still have room. The total is 16, so the open-board odds are 6.25%, 12.5%,
# 18.75%, 25%, 18.75%, 12.5%, 6.25% (shown rounded as 6%/12%/19%/25%/19%/12%/6%).
CHANCE_WEIGHTS = (1, 2, 3, 4, 3, 2, 1)

DESCRIPTIONS = {
    "alphazero": "PUCT search guided by a policy-value network trained only by self-play.",
    "minimax": "Alpha-beta negamax with a transposition table and a hand-built window evaluation.",
    "mcts": "Monte Carlo tree search with random rollouts and no learned knowledge.",
    "chance": "Chance (fixed odds): picks a column from fixed odds 6%/12%/19%/25%/19%/12%/6%, "
              "renormalised over open columns. No search, evaluation or lookahead.",
}


def board_from(moves: str) -> Board:
    board = Board()
    for i, ch in enumerate(moves):
        col = int(ch) - 1
        if not board.can_play(col):
            raise ValueError(f"move {i + 1} plays full column {ch}")
        if board.last_player_won():
            raise ValueError("the game is already over")
        board = board.play(col)
    if board.last_player_won() or board.is_full():
        raise ValueError("the game is already over")
    return board


def alphazero_move(board: Board, level: int) -> dict:
    from .net import load
    from .puct import search

    net = load()
    tree = search(board, net.evaluate, simulations=ALPHAZERO_SIMS[level])
    root = tree.root
    visits = root.visits
    q = [float(root.value_sum[c] / visits[c]) if visits[c] else None for c in range(COLS)]
    _, value = net.evaluate([board])
    return {
        "move": int(visits.argmax()),
        "analysis": {
            "kind": "alphazero",
            "simulations": int(root.total_visits()),
            "prior": [round(float(p), 4) for p in root.prior],
            "visits": [int(v) for v in visits],
            "q": q,
            # Search value for the player to move, mapped from [-1, 1] to a win chance.
            "win_probability": round((tree.root_value() + 1) / 2, 4),
            "network_value": round(float(value[0]), 4),
        },
    }


def minimax_move(board: Board, level: int) -> dict:
    info = Minimax(max_depth=42, max_seconds=MINIMAX_SECONDS[level]).search(board)

    def describe(score):
        if score is None:
            return None
        if score >= WIN - 64:
            return {"result": "win", "plies": WIN - score}
        if score <= -(WIN - 64):
            return {"result": "loss", "plies": WIN + score}
        return {"result": "eval", "score": score}

    return {
        "move": info.move,
        "analysis": {
            "kind": "minimax",
            "depth": info.depth,
            "nodes": info.nodes,
            "scores": [describe(s) for s in info.root_scores] if info.root_scores else None,
            "best": describe(info.score),
        },
    }


def mcts_move(board: Board, level: int) -> dict:
    result = MCTS(simulations=MCTS_SIMS[level], max_seconds=3.0).search(board)
    return {
        "move": result["move"],
        "analysis": {
            "kind": "mcts",
            "simulations": result["simulations"],
            "visits": [int(result["visits"].get(c, 0)) for c in range(COLS)],
        },
    }


def chance_odds(board: Board) -> list[float]:
    """Probability of each column under the fixed weights, renormalised over the open columns.

    Full columns get 0. The list always has COLS entries and sums to 1 (the board is not full).
    """
    weights = [0 if not board.can_play(c) else CHANCE_WEIGHTS[c] for c in range(COLS)]
    total = sum(weights)
    return [w / total for w in weights]


def chance_move(board: Board, level: int = 1, rng: random.Random | None = None) -> dict:
    """Pick a column by sampling from chance_odds. No search, evaluation or lookahead.

    `level` is accepted so the agent has the same signature as the others, and is ignored.
    `rng` makes the draw reproducible: the same seed and position always give the same column.
    With no rng, a fresh unseeded generator is used.
    """
    if rng is None:
        rng = random.Random()
    odds = chance_odds(board)
    move = rng.choices(range(COLS), weights=odds)[0]
    return {"move": move, "analysis": {"kind": "chance", "odds": [round(p, 4) for p in odds]}}


AGENTS = {"alphazero": alphazero_move, "minimax": minimax_move, "mcts": mcts_move, "chance": chance_move}
