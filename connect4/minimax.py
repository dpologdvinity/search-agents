"""Alpha-beta negamax for Connect Four, the classic baseline.

Iterative deepening under a node or time budget, a transposition table,
and move ordering (transposition-table move first, then center columns).
At the depth limit, positions are scored by counting open four-cell
windows, the standard hand-built Connect Four evaluation.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from .board import CENTER_ORDER, COLS, H1, ROWS, Board

WIN = 1_000_000  # win scores are WIN - ply, so faster wins score higher


def _windows() -> list[int]:
    """Bitmasks of every four-cell line on the board (69 of them)."""
    out = []
    for c in range(COLS):
        for r in range(ROWS):
            for dc, dr in ((1, 0), (0, 1), (1, 1), (1, -1)):
                cells = [(c + k * dc, r + k * dr) for k in range(4)]
                if all(0 <= cc < COLS and 0 <= rr < ROWS for cc, rr in cells):
                    out.append(sum(1 << (cc * H1 + rr) for cc, rr in cells))
    return out


WINDOWS = _windows()
CENTER_MASK = ((1 << ROWS) - 1) << (3 * H1)
# Points for a window holding k stones of one player and none of the other.
WINDOW_POINTS = (0, 1, 8, 64, 0)


def evaluate(board: Board) -> int:
    """Heuristic score for the player to move: open windows, plus center control."""
    mine = board.current
    theirs = board.current ^ board.mask
    score = 0
    for w in WINDOWS:
        a = (mine & w).bit_count()
        b = (theirs & w).bit_count()
        if a and not b:
            score += WINDOW_POINTS[a]
        elif b and not a:
            score -= WINDOW_POINTS[b]
    score += 3 * ((mine & CENTER_MASK).bit_count() - (theirs & CENTER_MASK).bit_count())
    return score


@dataclass
class SearchInfo:
    move: int
    score: int
    depth: int
    nodes: int
    seconds: float
    # Score per column at the final depth (None if illegal). Only the chosen
    # move's score is exact; alpha-beta makes the others upper bounds.
    root_scores: list[int | None] = field(default_factory=list)


class _Timeout(Exception):
    pass


EXACT, LOWER, UPPER = 0, 1, 2


class Minimax:
    def __init__(self, max_depth: int = 8, max_seconds: float | None = None,
                 max_nodes: int | None = None):
        self.max_depth = max_depth
        self.max_seconds = max_seconds
        self.max_nodes = max_nodes
        # key -> (depth, flag, score, move). Win scores encode distance from the
        # node where they were stored, so a reused entry can misjudge which of
        # two wins is faster, never whether a position is won.
        self.table: dict[int, tuple[int, int, int, int]] = {}

    def _negamax(self, board: Board, depth: int, alpha: int, beta: int, ply: int) -> int:
        self.nodes += 1
        if self.nodes & 1023 == 0:
            if self.deadline is not None and time.perf_counter() > self.deadline:
                raise _Timeout
        if self.max_nodes is not None and self.nodes > self.max_nodes:
            raise _Timeout

        moves = board.legal_moves()
        for col in moves:
            if board.is_winning_move(col):
                return WIN - ply - 1
        if not moves:
            return 0  # draw
        if depth == 0:
            return evaluate(board)

        alpha0 = alpha
        key = board.key()
        tt_move = None
        entry = self.table.get(key)
        if entry is not None:
            d, flag, value, tt_move = entry
            if d >= depth:
                if flag == EXACT:
                    return value
                if flag == LOWER and value >= beta:
                    return value
                if flag == UPPER and value <= alpha:
                    return value

        order = [c for c in CENTER_ORDER if c in moves]
        if tt_move in order:
            order.remove(tt_move)
            order.insert(0, tt_move)

        best, best_move = -WIN * 2, order[0]
        for col in order:
            score = -self._negamax(board.play(col), depth - 1, -beta, -alpha, ply + 1)
            if score > best:
                best, best_move = score, col
            if best > alpha:
                alpha = best
            if alpha >= beta:
                break

        flag = UPPER if best <= alpha0 else LOWER if best >= beta else EXACT
        self.table[key] = (depth, flag, best, best_move)
        return best

    def search(self, board: Board) -> SearchInfo:
        """Best move for the player to move, deepening until a budget runs out."""
        start = time.perf_counter()
        self.deadline = start + self.max_seconds if self.max_seconds else None
        self.nodes = 0
        moves = board.legal_moves()
        if not moves:
            raise ValueError("no legal moves")
        for col in moves:  # take an immediate win without searching
            if board.is_winning_move(col):
                return SearchInfo(col, WIN - 1, 1, 1, time.perf_counter() - start)

        best = SearchInfo(next(c for c in CENTER_ORDER if c in moves), 0, 0, 0, 0.0)
        for depth in range(1, self.max_depth + 1):
            try:
                scores = {}
                alpha = -WIN * 2
                order = [c for c in CENTER_ORDER if c in moves]
                if best.depth:
                    order.remove(best.move)
                    order.insert(0, best.move)
                for col in order:
                    scores[col] = -self._negamax(board.play(col), depth - 1, -WIN * 2, -alpha, 1)
                    alpha = max(alpha, scores[col])
            except _Timeout:
                break
            move = max(scores, key=lambda c: (scores[c], -CENTER_ORDER.index(c)))
            best = SearchInfo(move, scores[move], depth, self.nodes, time.perf_counter() - start,
                              [scores.get(c) for c in range(COLS)])
            if abs(scores[move]) >= WIN - 64:
                break  # forced result found; deeper search cannot change it
        best.nodes = self.nodes
        best.seconds = time.perf_counter() - start
        return best
