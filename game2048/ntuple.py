"""N-tuple network: a learned value function for 2048 afterstates.

V(board) is a sum of table lookups. Each pattern is a fixed set of cells;
the tile exponents in those cells form an index (4 bits per cell) into
that pattern's table, which has 16 ** len(pattern) entries. Every pattern is
applied in all 8 rotations and reflections of the board, sharing one table,
so V sums 6 patterns x 8 symmetries = 48 lookups.

Four patterns are 5 cells and two are 6 cells. The 6-cell tables are the
large part of the file (16.7 million entries each), so tables are stored as
float16 and read back as float16 for serving; training keeps float32.

This is a linear model over sparse features. Weights are learned by
temporal-difference learning in game2048.train_td; this module evaluates
them, one board at a time (for search) or on NumPy batches (for training).
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np

WEIGHTS = Path(__file__).parent / "data" / "ntuple_2048.npz"

# Cells are numbered row-major: index = 4 * row + col. Shapes are drawn on the 4x4 board.
PATTERNS = (
    (0, 1, 2, 3, 4),        # 5 cells: top row plus the cell below its first tile
    (4, 5, 6, 7, 8),        # 5 cells: second row plus the cell below its first tile
    (0, 1, 2, 4, 5),        # 5 cells: top-left 2x3 block without its bottom-right corner
    (4, 5, 6, 8, 9),        # 5 cells: the same block one row down
    (0, 1, 2, 3, 4, 5),     # 6 cells: top row plus the first two cells of row 1
    (0, 1, 2, 4, 5, 6),     # 6 cells: top-left 2x3 rectangle
)
# The two 6-cell patterns are taken from the 4x6-tuple network of Yeh et al. (as in the
# moporgic/TDL2048-Demo). Its other two, {4,5,6,7,8,9} and {4,5,6,8,9,10}, are left out to keep
# the table memory at two 6-cell tables (32 MB each in float16).


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


def pattern_shifts(pattern) -> list[list[int]]:
    """Bit shift of each cell of `pattern`, for each of the 8 symmetries.

    Cell i of the pattern contributes its exponent at shift 4 * (image of i),
    so the board's nibbles can be read straight off the 64-bit integer.
    """
    return [[4 * sym[c] for c in pattern] for sym in SYMMETRIES]


def canonical(pattern) -> tuple[int, ...]:
    """The smallest cell set among a pattern's 8 images; equal for shapes that are symmetric copies."""
    return min(tuple(sorted(sym[c] for c in pattern)) for sym in SYMMETRIES)


class NTupleNetwork:
    def __init__(self, tables=None, patterns=PATTERNS):
        """tables: one array per pattern, of length 16 ** len(pattern). Default: zeros, float32."""
        self.patterns = tuple(tuple(p) for p in patterns)
        self.shifts = [pattern_shifts(p) for p in self.patterns]
        if tables is None:
            tables = [np.zeros(16 ** len(p), dtype=np.float32) for p in self.patterns]
        self.tables = [np.asarray(t) for t in tables]

    def indices_batch(self, boards: np.ndarray) -> list[np.ndarray]:
        """For each pattern, a (symmetries, B) int64 array of table indices for B boards."""
        boards = boards.astype(np.uint64)
        out = []
        for per_sym in self.shifts:
            idx_p = np.empty((len(SYMMETRIES), len(boards)), dtype=np.int64)
            for s, shifts in enumerate(per_sym):
                idx = np.zeros(len(boards), dtype=np.uint64)
                for sh in shifts:
                    idx = (idx << np.uint64(4)) | ((boards >> np.uint64(sh)) & np.uint64(0xF))
                idx_p[s] = idx.astype(np.int64)
            out.append(idx_p)
        return out

    def value_batch(self, boards: np.ndarray) -> np.ndarray:
        """V for each board in a (B,) uint64 array; sums are accumulated in float64."""
        total = np.zeros(len(boards), dtype=np.float64)
        for table, idx in zip(self.tables, self.indices_batch(boards)):
            total += table[idx].sum(axis=0, dtype=np.float64)
        return total

    def value(self, board: int) -> float:
        """V for one board. Plain Python index arithmetic and item() reads beat NumPy fancy indexing here."""
        total = 0.0
        for table, per_sym in zip(self.tables, self.shifts):
            for shifts in per_sym:
                i = 0
                for s in shifts:
                    i = (i << 4) | ((board >> s) & 0xF)
                total += table.item(i)
        return total

    def save(self, path=WEIGHTS):
        """Write the export format: per pattern, its cells and its float16 table."""
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays = {}
        for i, (cells, table) in enumerate(zip(self.patterns, self.tables)):
            arrays[f"cells{i}"] = np.array(cells, dtype=np.int8)
            arrays[f"table{i}"] = table.astype(np.float16)
        np.savez_compressed(path, **arrays)

    @classmethod
    def load(cls, path=WEIGHTS, dtype=np.float16) -> NTupleNetwork:
        with np.load(path) as data:
            if "weights" in data.files:
                # Legacy layout (the first committed network): one (patterns, 16 ** 5) array.
                patterns = [tuple(int(c) for c in row) for row in data["patterns"]]
                tables = [t.astype(dtype) for t in data["weights"]]
            else:
                n = sum(1 for name in data.files if name.startswith("table"))
                patterns = [tuple(data[f"cells{i}"].tolist()) for i in range(n)]
                tables = [data[f"table{i}"].astype(dtype) for i in range(n)]
        return cls(tables, patterns)


@lru_cache(maxsize=1)
def load() -> NTupleNetwork:
    if not WEIGHTS.exists():
        raise FileNotFoundError(f"{WEIGHTS} missing; train it with: python -m game2048.train_td")
    return NTupleNetwork.load()
