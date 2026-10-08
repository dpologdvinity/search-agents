"""Iterative-deepening searches: IDS and IDA*.

Both run repeated depth-first searches with a growing cost bound, so memory
is linear in the solution depth instead of exponential like BFS or A*. That
makes IDA* the standard optimal solver for the 15-puzzle.

The search works on one mutable board and undoes each move on the way back
up, instead of allocating a node per board. It skips the move that
reverses the previous one, but otherwise does not detect repeated states, so
the same state can be expanded more than once.
"""

from __future__ import annotations

import math

from . import pdb
from .board import INVERSE, Node, goal, is_solvable, neighbors
from .heuristics import _manhattan_table, manhattan, pattern_database
from .search import _start

_FOUND = -1


class _LimitReached(Exception):
    pass


def _iterative_deepening(start_board, heuristic, algorithm, limits, on_expand):
    start, n, run = _start(start_board, algorithm, limits, on_expand)
    if not is_solvable(start, n):
        return run.result("unsolvable")

    moves = neighbors(n)
    solved = list(goal(n))
    board = list(start)
    path = []

    # Manhattan distance and pattern database values change only for the one
    # tile that moves, so both update in O(1) instead of being recomputed.
    # For the PDB, only the moved tile's group key changes: the tile's
    # position is one base-16 digit of that key.
    incremental = heuristic is manhattan
    table = _manhattan_table(n) if incremental else None
    use_pdb = heuristic is pattern_database and n == pdb.N
    if use_pdb:
        tables = pdb._tables()
        digit = {}  # tile -> (group index, place value of its position in the key)
        for gi, group in enumerate(pdb.GROUPS):
            for j, t in enumerate(group):
                digit[t] = (gi, pdb.CELLS ** (len(group) - 1 - j))
        keys = [pdb._key(start.index(t) for t in group) for group in pdb.GROUPS]

    def h_of(blank, target, tile, h):
        if heuristic is None:
            return 0
        if incremental:
            return h + table[tile][blank] - table[tile][target]
        if use_pdb:
            gi, place = digit[tile]
            old = tables[gi][keys[gi]]
            keys[gi] += (blank - target) * place
            return h - old + tables[gi][keys[gi]]
        return heuristic(tuple(board), n)

    def undo_h(blank, target, tile):
        if use_pdb:
            gi, place = digit[tile]
            keys[gi] -= (blank - target) * place

    def visit(blank, g, h, prev, bound):
        f = g + h
        if f > bound:
            return f
        # Every heuristic here is 0 only at the goal, so test h first.
        if h == 0 and board == solved:
            return _FOUND
        if run.over_limit():
            raise _LimitReached
        # The "frontier" of a depth-first search is the current path.
        run.expand(Node(tuple(board) if on_expand else None, None, prev, g), len(path))

        minimum = math.inf
        for to, action in moves[blank]:
            if prev is not None and action == INVERSE[prev]:
                continue
            tile = board[to]
            board[blank], board[to] = tile, 0
            path.append(action)
            run.generated += 1
            t = visit(to, g + 1, h_of(blank, to, tile, h), action, bound)
            if t == _FOUND:
                return _FOUND
            undo_h(blank, to, tile)
            path.pop()
            board[blank], board[to] = 0, tile
            if t < minimum:
                minimum = t
        return minimum

    blank = board.index(0)
    h0 = heuristic(start, n) if heuristic is not None else 0
    bound = h0
    try:
        while True:
            t = visit(blank, 0, h0, None, bound)
            if t == _FOUND:
                return run.result("solved", list(path))
            if t == math.inf:
                return run.result("exhausted")
            bound = t
    except _LimitReached:
        return run.result("limit")


def ids(board, limits=None, on_expand=None):
    """Iterative-deepening DFS: depth bound 0, 1, 2, ... Optimal, no heuristic."""
    return _iterative_deepening(board, None, "ids", limits, on_expand)


def ida_star(board, heuristic=manhattan, limits=None, on_expand=None):
    """IDA*: the bound is on f = g + h and grows to the smallest f that exceeded it.

    Optimal with an admissible heuristic.
    """
    return _iterative_deepening(board, heuristic, "idastar", limits, on_expand)
