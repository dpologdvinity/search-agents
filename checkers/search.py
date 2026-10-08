"""Game-tree search for checkers: plain minimax and alpha-beta.

Both are negamax: a score is always from the point of view of the side to
move, and a child's score is negated for its parent. That is minimax: each
side assumes the other picks the reply that is worst for it.

AlphaBeta adds the window [alpha, beta]. Once one reply proves a move is
worse than an alternative already found, the remaining replies are skipped
(a cutoff). Iterative deepening, a transposition table, and ordering the
best move first make cutoffs come early, so far fewer positions are visited
for the same answer. Minimax visits every position to a fixed depth and is
here to show the difference.

Captures are forced, so a position with a capture pending is searched one
level deeper instead of being scored mid-exchange (quiescence).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from .board import ROW, legal_moves

WIN = 100_000
MAN, KING = 100, 165
CENTER = frozenset({9, 10, 13, 14, 17, 18, 21, 22})


def evaluate(board, side) -> int:
    """Material, advancement, back-row guard, and centre control, for side to move."""
    score = 0
    for sq, p in enumerate(board):
        if not p:
            continue
        if abs(p) == 2:
            v = KING
        else:
            advanced = 7 - ROW[sq] if p > 0 else ROW[sq]
            v = MAN + 3 * advanced
            if (p > 0 and ROW[sq] == 7) or (p < 0 and ROW[sq] == 0):
                v += 8  # men on the home row stop the opponent from crowning
        if sq in CENTER:
            v += 6
        score += v if p > 0 else -v
    return score * side


@dataclass
class SearchInfo:
    move: object
    score: int
    depth: int
    nodes: int
    cutoffs: int
    seconds: float
    root: list = field(default_factory=list)  # (move, score) at the deepest completed depth


class _Timeout(Exception):
    pass


EXACT, LOWER, UPPER = 0, 1, 2


class AlphaBeta:
    def __init__(self, max_depth=30, max_seconds=1.0, max_nodes=None, forced_depth=4):
        self.max_depth = max_depth
        self.forced_depth = min(forced_depth, max_depth)
        self.max_seconds = max_seconds
        self.max_nodes = max_nodes

    def _check(self):
        self.nodes += 1
        if self.nodes & 511 == 0 and time.perf_counter() > self.deadline:
            raise _Timeout
        if self.max_nodes is not None and self.nodes > self.max_nodes:
            raise _Timeout

    def _negamax(self, board, side, depth, alpha, beta, ply):
        self._check()
        moves = legal_moves(board, side)
        if not moves:
            return -WIN + ply  # no legal move: side to move loses
        if depth <= 0 and not moves[0].captured:
            return evaluate(board, side)

        alpha0, key = alpha, (board, side)
        entry = self.table.get(key)
        best_path = None
        if entry is not None:
            d, flag, value, best_path = entry
            usable = flag == EXACT or (flag == LOWER and value >= beta) or (flag == UPPER and value <= alpha)
            if d >= depth and usable:
                return value
        # Move ordering: the transposition table's best move, then captures by size.
        moves.sort(key=lambda m: (m.path != best_path, -len(m.captured)))

        best = -WIN * 2
        for m in moves:
            score = -self._negamax(m.result, -side, depth - 1, -beta, -alpha, ply + 1)
            if score > best:
                best, best_path = score, m.path
            if best > alpha:
                alpha = best
            if alpha >= beta:
                self.cutoffs += 1
                break
        flag = UPPER if best <= alpha0 else LOWER if best >= beta else EXACT
        self.table[key] = (depth, flag, best, best_path)
        return best

    def search(self, board, side) -> SearchInfo:
        start = time.perf_counter()
        self.deadline = start + self.max_seconds
        self.nodes = self.cutoffs = 0
        self.table = {}
        moves = legal_moves(board, side)
        if not moves:
            raise ValueError("no legal moves")
        info = SearchInfo(moves[0], 0, 0, 0, 0, 0.0, [])
        if len(moves) == 1:  # forced: play it, scored by a short fixed-depth search
            score = -self._negamax(moves[0].result, -side, self.forced_depth - 1, -WIN * 2, WIN * 2, 1)
            info = SearchInfo(moves[0], score, self.forced_depth, self.nodes, self.cutoffs,
                              time.perf_counter() - start, [(moves[0], score)])
            return info
        for depth in range(1, self.max_depth + 1):
            try:
                order = sorted(moves, key=lambda m: m is not info.move)
                scored, alpha = [], -WIN * 2
                for m in order:
                    s = -self._negamax(m.result, -side, depth - 1, -WIN * 2, -alpha, 1)
                    scored.append((m, s))
                    alpha = max(alpha, s)
            except _Timeout:
                break
            best = max(scored, key=lambda x: x[1])
            info = SearchInfo(best[0], best[1], depth, self.nodes, self.cutoffs, 0.0, scored)
            if abs(best[1]) > WIN - 100:
                break  # forced win or loss found
        info.nodes, info.cutoffs = self.nodes, self.cutoffs
        info.seconds = time.perf_counter() - start
        return info


class Minimax:
    """Full-width minimax to a fixed depth, with no pruning."""

    def __init__(self, depth=4):
        self.depth = depth

    def _negamax(self, board, side, depth, ply):
        self.nodes += 1
        moves = legal_moves(board, side)
        if not moves:
            return -WIN + ply
        if depth <= 0 and not moves[0].captured:
            return evaluate(board, side)
        return max(-self._negamax(m.result, -side, depth - 1, ply + 1) for m in moves)

    def search(self, board, side) -> SearchInfo:
        start = time.perf_counter()
        self.nodes = 0
        moves = legal_moves(board, side)
        if not moves:
            raise ValueError("no legal moves")
        scored = [(m, -self._negamax(m.result, -side, self.depth - 1, 1)) for m in moves]
        best = max(scored, key=lambda x: x[1])
        return SearchInfo(best[0], best[1], self.depth, self.nodes, 0, time.perf_counter() - start, scored)
