"""Connect Four agents behind one interface, shared by the web server and the command line.

Each agent function takes a position and a strength level (1-3) and returns
{"move": column, "analysis": {...}}, where the analysis explains how the
agent decided (network priors and visit counts, minimax scores, or MCTS
visit counts).
"""

from __future__ import annotations

from .board import COLS, Board
from .mcts import MCTS
from .minimax import WIN, Minimax

# Search budget per level. Seconds cap minimax so a turn stays responsive.
ALPHAZERO_SIMS = {1: 50, 2: 200, 3: 800}
MINIMAX_SECONDS = {1: 0.1, 2: 0.5, 3: 2.0}
MCTS_SIMS = {1: 300, 2: 1500, 3: 5000}

DESCRIPTIONS = {
    "alphazero": "PUCT search guided by a policy-value network trained only by self-play.",
    "minimax": "Alpha-beta negamax with a transposition table and a hand-built window evaluation.",
    "mcts": "Monte Carlo tree search with random rollouts and no learned knowledge.",
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


AGENTS = {"alphazero": alphazero_move, "minimax": minimax_move, "mcts": mcts_move}
