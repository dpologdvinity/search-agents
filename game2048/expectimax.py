"""Expectimax search for 2048 with a hand-crafted, automatically tuned evaluation.

Search alternates MAX nodes (the player picks a direction) and CHANCE nodes
(a 2 with probability 0.9 or a 4 with probability 0.1 appears in a uniformly
random empty cell). Iterative deepening runs until the per-move time budget
is spent and keeps the deepest completed answer. Two optional speedups:
a transposition table keyed by (board, depth), and skipping spawn outcomes
whose cumulative probability falls below a cutoff.

The evaluation is a weighted sum of six board features. The weights are not
set by hand: game2048.tune searches for them with the cross-entropy method
and stores the result in data/heuristic_weights.json.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from .board import ROW_MASK, legal_moves, transpose

WEIGHTS = Path(__file__).parent / "data" / "heuristic_weights.json"
FEATURES = ("empty", "monotonic", "smooth", "corner", "merges", "max_tile")


def _row_features(row: int):
    """(empty count, increasing-order penalty, decreasing-order penalty, roughness, equal neighbours).

    Penalties are in tile exponents (log2 of tile values) over the row's
    non-empty tiles, so empty cells between two tiles are skipped. Equal
    neighbours are adjacent cells holding the same tile.
    """
    cells = [(row >> (4 * i)) & 0xF for i in range(4)]
    tiles = [v for v in cells if v]
    up = down = rough = 0
    for a, b in zip(tiles, tiles[1:]):
        if a > b:
            up -= a - b
        elif a < b:
            down -= b - a
        rough -= abs(a - b)
    equal = sum(1 for a, b in zip(cells, cells[1:]) if a and a == b)
    return cells.count(0), up, down, rough, equal


def _tables():
    t = np.array([_row_features(r) for r in range(65536)], dtype=np.int64)
    return tuple(t[:, k] for k in range(5))


EMPTY, UP, DOWN, ROUGH, EQUAL = _tables()
_LISTS = tuple(col.tolist() for col in (EMPTY, UP, DOWN, ROUGH, EQUAL))
CORNER_SHIFTS = (0, 12, 48, 60)


def features(board: int) -> tuple[float, ...]:
    """The six features of one board, in FEATURES order.

    Sign convention: every feature is oriented so that a larger value is
    better. Penalties (monotonic, smooth) are zero or negative, rewards
    (empty, corner, merges, max tile) are zero or positive, so a non-negative
    weight makes the term reward order. The tuner bounds weights at zero. The
    shipped smoothness weight is negative anyway: its run was unbounded, and
    it rewards rougher rows and columns (see the README for the comparison).
    """
    empty_t, up_t, down_t, rough_t, equal_t = _LISTS
    rows = [(board >> s) & ROW_MASK for s in (0, 16, 32, 48)]
    t = transpose(board)
    cols = [(t >> s) & ROW_MASK for s in (0, 16, 32, 48)]
    empty = sum(empty_t[r] for r in rows)
    monotonic = (max(sum(up_t[r] for r in rows), sum(down_t[r] for r in rows))
                 + max(sum(up_t[c] for c in cols), sum(down_t[c] for c in cols)))
    smooth = sum(rough_t[r] for r in rows) + sum(rough_t[c] for c in cols)
    merges = sum(equal_t[r] for r in rows) + sum(equal_t[c] for c in cols)
    peak = max((board >> (4 * i)) & 0xF for i in range(16))
    corner = peak if peak and any((board >> s) & 0xF == peak for s in CORNER_SHIFTS) else 0
    return (math.log2(empty + 1), monotonic, smooth, corner, merges, peak)


def features_batch(boards: np.ndarray) -> np.ndarray:
    """(B,) uint64 boards -> (B, 6) features, matching features() row by row."""
    from .batch import transpose_batch

    m = np.uint64(ROW_MASK)
    rows = [((boards >> np.uint64(s)) & m).astype(np.int64) for s in (0, 16, 32, 48)]
    t = transpose_batch(boards)
    cols = [((t >> np.uint64(s)) & m).astype(np.int64) for s in (0, 16, 32, 48)]
    empty = sum(EMPTY[r] for r in rows)
    monotonic = (np.maximum(sum(UP[r] for r in rows), sum(DOWN[r] for r in rows))
                 + np.maximum(sum(UP[c] for c in cols), sum(DOWN[c] for c in cols)))
    smooth = sum(ROUGH[r] for r in rows) + sum(ROUGH[c] for c in cols)
    merges = sum(EQUAL[r] for r in rows) + sum(EQUAL[c] for c in cols)
    cells = np.stack([(boards >> np.uint64(4 * i)) & np.uint64(0xF) for i in range(16)], axis=1).astype(np.int64)
    peak = cells.max(axis=1)
    in_corner = (cells[:, [0, 3, 12, 15]] == peak[:, None]).any(axis=1) & (peak > 0)
    corner = np.where(in_corner, peak, 0)
    return np.stack([np.log2(empty + 1), monotonic, smooth, corner, merges, peak], axis=1).astype(np.float64)


def load_weights(path=WEIGHTS) -> tuple[float, ...]:
    """Feature weights in FEATURES order, from a file written by game2048.tune."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} missing; tune it with: python -m game2048.tune")
    data = json.loads(path.read_text())
    return tuple(data["weights"][f] for f in FEATURES)


@lru_cache(maxsize=1)
def tuned_weights() -> tuple[float, ...]:
    return load_weights(WEIGHTS)


def evaluate(board: int, weights=None) -> float:
    w = weights or tuned_weights()
    return sum(a * b for a, b in zip(w, features(board)))


class _TimesUp(Exception):
    pass


@dataclass
class MoveInfo:
    move: int
    depth: int
    nodes: int
    seconds: float
    values: dict  # direction -> expected value at the deepest completed depth


class Expectimax:
    def __init__(self, budget: float = 0.1, cache: bool = True, prob_cutoff: float = 1e-4,
                 max_depth: int = 20, evaluator=None, rewards: bool = False):
        """rewards=True adds the points scored by merges along each line of
        play, for evaluators that estimate future points (the n-tuple network).
        The hand-crafted evaluation scores board shape and ignores points."""
        self.budget = budget
        self.cache = cache
        self.prob_cutoff = prob_cutoff
        self.max_depth = max_depth
        self.evaluate = evaluator or evaluate
        self.rewards = rewards

    def _check(self):
        self.nodes += 1
        if self.nodes & 255 == 0 and time.perf_counter() >= self.deadline:
            raise _TimesUp

    def _max(self, board, depth, prob):
        """Best expected value over legal moves; depth counts remaining MAX levels."""
        self._check()
        moves = legal_moves(board)
        if not moves or depth == 0:
            return self.evaluate(board)
        if self.rewards:
            return max(gained + self._chance(new, depth - 1, prob) for _, new, gained in moves)
        return max(self._chance(new, depth - 1, prob) for _, new, _ in moves)

    def _chance(self, board, depth, prob):
        """Expected value over tile spawns."""
        self._check()
        if depth == 0 or prob < self.prob_cutoff:
            return self.evaluate(board)
        key = (board, depth)
        if self.cache and key in self.table:
            return self.table[key]
        empties = [i for i in range(16) if not (board >> (4 * i)) & 0xF]
        if not empties:
            return self.evaluate(board)
        share = 1.0 / len(empties)
        total = 0.0
        for i in empties:
            shift = 4 * i
            total += 0.9 * share * self._max(board | (1 << shift), depth, prob * 0.9 * share)
            total += 0.1 * share * self._max(board | (2 << shift), depth, prob * 0.1 * share)
        if self.cache:
            self.table[key] = total
        return total

    def search(self, board: int) -> MoveInfo:
        start = time.perf_counter()
        self.deadline = start + self.budget
        self.nodes = 0
        moves = legal_moves(board)
        if not moves:
            raise ValueError("no legal moves")
        best = MoveInfo(moves[0][0], 0, 0, 0.0, {})
        for depth in range(1, self.max_depth + 1):
            self.table = {}
            try:
                values = {d: (gained if self.rewards else 0) + self._chance(new, depth - 1, 1.0)
                          for d, new, gained in moves}
            except _TimesUp:
                break
            best = MoveInfo(max(values, key=values.get), depth, self.nodes,
                            time.perf_counter() - start, values)
            if time.perf_counter() >= self.deadline:
                break
        best.nodes = self.nodes
        best.seconds = time.perf_counter() - start
        return best
