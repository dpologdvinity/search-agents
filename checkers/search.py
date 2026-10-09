"""Game-tree search for checkers: alpha-beta, plain minimax, and one-ply greedy and random play.

The tree searches are negamax: a score is always from the point of view of the side
to move, and a child's score is negated for its parent. That is minimax: each side
assumes the other picks the reply that is worst for it.

AlphaBeta adds the window [alpha, beta]. Once one reply proves a move is worse than an
alternative already found, the remaining replies are skipped (a cutoff). Iterative
deepening, a transposition table, and ordering the best move first make cutoffs come
early, so far fewer positions are visited for the same answer. Minimax visits every
position to a fixed depth and is here to show the difference.

Quiescence: at the depth limit a position with a capture pending is not scored
mid-exchange. Only the captures are searched further, until none is left. With forced
captures the side to move must take one of them. With optional captures it may also
decline and keep the static evaluation (stand pat), so a quiet position is still scored
as it stands.

Greedy plays the single move that looks best after one ply, and Random plays a uniform
legal move. They are the baselines the tree searches are compared against.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field

from .board import geometry, legal_moves

WIN = 100_000
MAN, KING = 100, 165


def evaluate(board, side) -> int:
    """Material, advancement, back-row guard, and centre control, for side to move.

    Advancement and the home-row guard are measured from each side's own back row, so
    the terms scale with the board. Centre squares come from the board's geometry.
    """
    geo = geometry(board)
    far = geo.n - 1  # red's home row and the row a white man crowns on (red crowns on row 0)
    score = 0
    for sq, p in enumerate(board):
        if not p:
            continue
        if abs(p) == 2:
            v = KING
        else:
            advanced = far - geo.ROW[sq] if p > 0 else geo.ROW[sq]
            v = MAN + 3 * advanced
            if (p > 0 and geo.ROW[sq] == far) or (p < 0 and geo.ROW[sq] == 0):
                v += 8  # men on the home row stop the opponent from crowning
        if sq in geo.CENTER:
            v += 6
        score += v if p > 0 else -v
    return score * side


@dataclass
class SearchInfo:
    move: object
    score: object  # int for alpha-beta, minimax and greedy; a float win rate for MCTS; None if unscored
    depth: int
    nodes: int
    cutoffs: int
    seconds: float
    root: list = field(default_factory=list)  # (move, score) at the deepest completed depth
    rollouts: int = 0  # simulated games (MCTS only)


def quiet_leaf(board, side, moves, forced):
    """Decide what the depth limit does with a position's moves.

    Returns (moves, stand). If the position is quiet (no capture), moves is empty and
    stand is its static score. Otherwise moves are the captures to keep searching, and
    stand is the score for declining to capture, or None when declining is not allowed:
    with forced captures, or when every legal move is a capture.
    """
    captures = [m for m in moves if m.captured]
    if not captures:
        return [], evaluate(board, side)
    stand = None if forced or all(m.captured for m in moves) else evaluate(board, side)
    return captures, stand


class _Timeout(Exception):
    pass


EXACT, LOWER, UPPER = 0, 1, 2

# Scores beyond this are wins or losses found by the search (WIN minus the ply they happen at), not evaluations.
MATE_BOUND = WIN - 1000


def to_table(score: int, ply: int) -> int:
    """Store a win or loss as its distance from this node, so the entry is valid at any ply.

    A loss found at ply q scores -WIN + q from the root. Reached again at a different ply, the same position is
    the same number of moves from that loss, so the table keeps -WIN + (q - ply) and from_table adds the new ply back.
    """
    if score > MATE_BOUND:
        return score + ply
    if score < -MATE_BOUND:
        return score - ply
    return score


def from_table(score: int, ply: int) -> int:
    """Undo to_table: turn a stored distance-from-node win or loss back into a score from the root."""
    if score > MATE_BOUND:
        return score - ply
    if score < -MATE_BOUND:
        return score + ply
    return score


class AlphaBeta:
    """Iterative-deepening alpha-beta, stopped by a time or node budget.

    Each depth is searched completely before the next, and the move that is best at
    the deepest finished depth is returned. A search that runs out mid-depth keeps the
    previous depth's answer.
    """

    def __init__(self, max_depth=30, max_seconds=1.0, max_nodes=None, forced_depth=4, forced=True):
        self.max_depth = max_depth
        self.forced_depth = min(forced_depth, max_depth)
        self.max_seconds = max_seconds
        self.max_nodes = max_nodes
        self.forced = forced

    def _check(self):
        self.nodes += 1
        if self.nodes & 511 == 0 and time.perf_counter() > self.deadline:
            raise _Timeout
        if self.max_nodes is not None and self.nodes > self.max_nodes:
            raise _Timeout

    def _negamax(self, board, side, depth, alpha, beta, ply):
        self._check()
        moves = legal_moves(board, side, self.forced)
        if not moves:
            return -WIN + ply  # no legal move: side to move loses
        stand = None
        if depth <= 0:
            moves, stand = quiet_leaf(board, side, moves, self.forced)
            if not moves:
                return stand

        alpha0, key = alpha, (board, side)
        entry = self.table.get(key)
        best_path = None
        if entry is not None:
            d, flag, value, best_path = entry
            value = from_table(value, ply)
            usable = flag == EXACT or (flag == LOWER and value >= beta) or (flag == UPPER and value <= alpha)
            if d >= depth and usable:
                return value
        # Move ordering: the transposition table's best move, then captures by size.
        moves.sort(key=lambda m: (m.path != best_path, -len(m.captured)))

        # Declining to capture (stand pat) is the starting score when it is allowed.
        best = -WIN * 2 if stand is None else stand
        alpha = max(alpha, best)
        if alpha >= beta:
            return best
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
        self.table[key] = (depth, flag, to_table(best, ply), best_path)
        return best

    def search(self, board, side) -> SearchInfo:
        """Best move for side within the time and node budgets, from the deepest depth that finished."""
        start = time.perf_counter()
        self.deadline = start + self.max_seconds
        self.nodes = self.cutoffs = 0
        self.table = {}
        moves = legal_moves(board, side, self.forced)
        if not moves:
            raise ValueError("no legal moves")
        info = SearchInfo(moves[0], 0, 0, 0, 0, 0.0, [])
        if len(moves) == 1:  # forced: play it, scored by a short fixed-depth search
            # The fixed-depth score is not cut off by the time or node budget, so the
            # only exception that could escape _check() is removed here.
            self.deadline, self.max_nodes = float("inf"), None
            score = -self._negamax(moves[0].result, -side, self.forced_depth - 1, -WIN * 2, WIN * 2, 1)
            return SearchInfo(moves[0], score, self.forced_depth, self.nodes, self.cutoffs,
                              time.perf_counter() - start, [(moves[0], score)])
        for depth in range(1, self.max_depth + 1):
            try:
                # The previous best move goes first, so this depth's cutoffs come early.
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

    def __init__(self, depth=4, forced=True):
        self.depth = depth
        self.forced = forced

    def _negamax(self, board, side, depth, ply):
        self.nodes += 1
        moves = legal_moves(board, side, self.forced)
        if not moves:
            return -WIN + ply
        stand = None
        if depth <= 0:
            moves, stand = quiet_leaf(board, side, moves, self.forced)
            if not moves:
                return stand
        best = -WIN * 2 if stand is None else stand
        for m in moves:
            best = max(best, -self._negamax(m.result, -side, depth - 1, ply + 1))
        return best

    def search(self, board, side) -> SearchInfo:
        """Best move for side at the fixed depth, scoring every root move."""
        start = time.perf_counter()
        self.nodes = 0
        moves = legal_moves(board, side, self.forced)
        if not moves:
            raise ValueError("no legal moves")
        scored = [(m, -self._negamax(m.result, -side, self.depth - 1, 1)) for m in moves]
        best = max(scored, key=lambda x: x[1])
        return SearchInfo(best[0], best[1], self.depth, self.nodes, 0, time.perf_counter() - start, scored)


class Greedy:
    """One ply: play the move whose resulting position the static evaluation likes most.

    A move that leaves the opponent with no legal move is scored as a win. Ties go to the
    earlier move in legal_moves order, so the choice is deterministic.
    """

    def __init__(self, forced=True):
        self.forced = forced

    def search(self, board, side) -> SearchInfo:
        """Score every legal move one ply deep and return the best."""
        start = time.perf_counter()
        moves = legal_moves(board, side, self.forced)
        if not moves:
            raise ValueError("no legal moves")
        scored = []
        for m in moves:
            if not legal_moves(m.result, -side, self.forced):
                score = WIN - 1  # one ply from the win
            else:
                score = -evaluate(m.result, -side)
            scored.append((m, score))
        best = max(scored, key=lambda x: x[1])
        return SearchInfo(best[0], best[1], 1, len(moves), 0, time.perf_counter() - start, scored)


class RandomMove:
    """A uniform random legal move from a seeded generator, so a run can be repeated."""

    def __init__(self, seed=0, forced=True):
        self.rng = random.Random(seed)
        self.forced = forced

    def search(self, board, side) -> SearchInfo:
        """Pick a uniform random legal move from the seeded generator."""
        start = time.perf_counter()
        moves = legal_moves(board, side, self.forced)
        if not moves:
            raise ValueError("no legal moves")
        move = moves[self.rng.randrange(len(moves))]
        return SearchInfo(move, None, 0, 0, 0, time.perf_counter() - start, [(m, None) for m in moves])
