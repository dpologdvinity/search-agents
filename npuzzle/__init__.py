"""N-Puzzle solvers: uninformed search, A* variants, iterative deepening, and heuristics."""

from .board import Node, apply, format_board, goal, is_solvable, replay, successors, validate
from .heuristics import HEURISTICS, manhattan
from .iterative import ida_star, ids
from .search import (
    SearchLimits,
    SearchResult,
    a_star,
    best_first,
    bfs,
    bidirectional_bfs,
    dfs,
    greedy,
    ucs,
    weighted_a_star,
)

ALGORITHMS = {
    "bfs": bfs,
    "dfs": dfs,
    "ids": ids,
    "ucs": ucs,
    "bibfs": bidirectional_bfs,
    "greedy": greedy,
    "astar": a_star,
    "wastar": weighted_a_star,
    "idastar": ida_star,
}
INFORMED = {"greedy", "astar", "wastar", "idastar"}
