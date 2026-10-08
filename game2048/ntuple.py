"""N-tuple network: a learned value function for 2048 afterstates.

V(board) is a sum of table lookups. Each pattern is a fixed set of five
cells; the tile exponents in those cells form a 20-bit index into that
pattern's weight table. Every pattern is applied in all 8 rotations and
reflections of the board, sharing one table, so V sums 6 x 8 = 48 weights.

This is a linear model over sparse features. Weights are learned by
temporal-difference learning in game2048.train_td; this module evaluates
them, one board at a time (for search) or on NumPy batches (for training).
"""

from __future__ import annotations

from array import array
from functools import lru_cache
from pathlib import Path

import numpy as np

WEIGHTS = Path(__file__).parent / "data" / "ntuple_2048.npz"

# Cells are numbered row-major: index = 4 * row + col.
PATTERNS = (
    (0, 1, 2, 3, 4),      # top row plus the cell below its first tile
    (4, 5, 6, 7, 8),      # second row plus one below
    (0, 1, 2, 4, 5),      # 2x3 block minus one corner (top-left area)
    (4, 5, 6, 8, 9),      # the same block one row down
    (0, 1, 4, 5, 8),      # tall L in the left columns
    (1, 2, 5, 6, 9),      # block shifted one column right
)
TUPLE_LEN = 5
TABLE_SIZE = 16 ** TUPLE_LEN


def _symmetries():
    """The 8 maps from a cell index to its image under rotation/reflection."""
    def rot(i):  # 90 degrees clockwise
        r, c = divmod(i, 4)
        return 4 * c + (3 - r)

    def flip(i):  # left-right mirror
        r, c = divmod(i, 4)
        return 4 * r + (3 - c)

    maps, m = [], list(range(16))
    for _ in range(4):
        maps.append(m[:])
        maps.append([flip(x) for x in m])
        m = [rot(x) for x in m]
    return maps


SYMMETRIES = _symmetries()
# FEATURES[p] = list over symmetries of the 5 cell indices that pattern p reads.
FEATURES = [[[sym[c] for c in pattern] for sym in SYMMETRIES] for pattern in PATTERNS]
# Bit shifts for each cell, for fast extraction from the 64-bit board.
SHIFTS = [[[4 * c for c in cells] for cells in per_sym] for per_sym in FEATURES]


def indices(board: int) -> list[list[int]]:
    """Table index for every (pattern, symmetry) pair of one board."""
    out = []
    for per_sym in SHIFTS:
        row = []
        for shifts in per_sym:
            idx = 0
            for s in shifts:
                idx = (idx << 4) | ((board >> s) & 0xF)
            row.append(idx)
        out.append(row)
    return out


def indices_batch(boards: np.ndarray) -> np.ndarray:
    """(B,) uint64 boards -> (patterns, symmetries, B) int64 table indices."""
    boards = boards.astype(np.uint64)
    out = np.empty((len(PATTERNS), len(SYMMETRIES), len(boards)), dtype=np.int64)
    for p, per_sym in enumerate(SHIFTS):
        for s, shifts in enumerate(per_sym):
            idx = np.zeros(len(boards), dtype=np.uint64)
            for sh in shifts:
                idx = (idx << np.uint64(4)) | ((boards >> np.uint64(sh)) & np.uint64(0xF))
            out[p, s] = idx.astype(np.int64)
    return out


class NTupleNetwork:
    def __init__(self, weights: np.ndarray | None = None):
        self.weights = (np.zeros((len(PATTERNS), TABLE_SIZE), dtype=np.float32)
                        if weights is None else weights)
        self._lists = None

    def value_batch(self, boards: np.ndarray) -> np.ndarray:
        idx = indices_batch(boards)
        total = np.zeros(idx.shape[2], dtype=np.float64)
        for p in range(len(PATTERNS)):
            total += self.weights[p][idx[p]].sum(axis=0)
        return total

    def value(self, board: int) -> float:
        # array('f') indexes about as fast as a list one element at a time, at
        # 4 bytes per weight instead of about 32 for a list of Python floats.
        if self._lists is None:
            self._lists = [array("f", w.astype(np.float32).tobytes()) for w in self.weights]
        return sum(table[i] for table, row in zip(self._lists, indices(board)) for i in row)

    def save(self, path=WEIGHTS):
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, weights=self.weights, patterns=np.array(PATTERNS))

    @classmethod
    def load(cls, path=WEIGHTS) -> NTupleNetwork:
        with np.load(path) as data:
            if tuple(map(tuple, data["patterns"])) != PATTERNS:
                raise ValueError("weights were trained with different patterns")
            return cls(data["weights"].astype(np.float32))


@lru_cache(maxsize=1)
def load() -> NTupleNetwork:
    if not WEIGHTS.exists():
        raise FileNotFoundError(f"{WEIGHTS} missing; train it with: python -m game2048.train_td")
    return NTupleNetwork.load()
