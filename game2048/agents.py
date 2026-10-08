"""2048 agents behind one interface, shared by the web server and the command line.

choose(board, agent) returns the chosen direction plus the expected value of
every legal direction, the search depth reached, and the nodes visited.
"""

from __future__ import annotations

from .board import legal_moves
from .expectimax import Expectimax

BUDGET = 0.1  # seconds of search per move

DESCRIPTIONS = {
    "expectimax": (
        "Iterative-deepening expectimax scoring boards with six hand-crafted features, weighted by "
        "a cross-entropy-method search over simulated games."
    ),
    "ntuple": "No search: picks the move whose afterstate a TD-learned n-tuple network values most.",
    "ntuple_search": (
        "Expectimax search that scores positions with the learned n-tuple network instead of the "
        "hand-tuned evaluation."
    ),
}


def choose(board: int, agent: str, budget: float = BUDGET) -> dict:
    """Pick a direction for `board` with the named agent."""
    if agent in ("ntuple", "ntuple_search"):
        from .ntuple import load

        net = load()
    if agent == "ntuple":
        # One-step lookahead: points gained now plus the learned value of the afterstate.
        values = {d: gained + net.value(after) for d, after, gained in legal_moves(board)}
        best = max(values, key=values.get)
        return {"move": best, "values": values, "depth": 1, "nodes": len(values)}
    if agent == "ntuple_search":
        searcher = Expectimax(budget=budget, evaluator=net.value, rewards=True)
    else:
        searcher = Expectimax(budget=budget)
    info = searcher.search(board)
    return {"move": info.move, "values": info.values, "depth": info.depth, "nodes": info.nodes}
