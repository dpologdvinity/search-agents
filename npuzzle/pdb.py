"""Additive disjoint pattern databases for the 15-puzzle.

The 15 tiles are split into three groups of five. For each group, a table
stores the exact number of moves *of that group's tiles* needed to bring them
home, with every other tile treated as indistinguishable. Moves of other
tiles cost 0, so the groups never count the same move twice and their sum is
an admissible heuristic. It is much stronger than Manhattan distance, which
ignores every interaction between tiles.

Build the tables once (about a minute per group):

    python -m npuzzle.pdb
"""

from __future__ import annotations

from collections import deque
from functools import lru_cache
from pathlib import Path

import numpy as np

from .board import neighbors

N = 4
CELLS = N * N
# Compact groups in the goal layout (blank at index 0):
#    .  1  2  3       A = top row and two below it
#    4  5  6  7       B = left block
#    8  9 10 11       C = bottom right
#   12 13 14 15
GROUPS = ((1, 2, 3, 6, 7), (4, 5, 8, 9, 12), (10, 11, 13, 14, 15))
DATA = Path(__file__).parent / "data" / "pdb_555.npz"
_UNSEEN = 255


def _key(positions) -> int:
    """Base-16 encoding of the group's tile positions."""
    key = 0
    for p in positions:
        key = key * CELLS + p
    return key


def build_group(group, progress=False) -> np.ndarray:
    """Return dist[_key(positions)] for one group, by 0-1 BFS from the goal.

    The search state is (group tile positions, blank position). A blank move
    costs 1 if it swaps with a group tile and 0 otherwise. The deque pops
    states in nondecreasing cost, so the first time a key is reached gives
    its minimum over all blank positions.
    """
    moves = neighbors(N)
    dist = np.full(CELLS ** len(group), _UNSEEN, dtype=np.uint8)
    seen = bytearray(CELLS ** len(group) * CELLS)

    start = tuple(group)  # tile t sits at index t in the goal
    queue = deque([(start, 0, 0)])
    popped = 0
    while queue:
        positions, blank, cost = queue.popleft()
        key = _key(positions)
        slot = key * CELLS + blank
        if seen[slot]:
            continue
        seen[slot] = 1
        if dist[key] == _UNSEEN:
            dist[key] = cost
        popped += 1
        if progress and popped % 1_000_000 == 0:
            print(f"  {popped:,} states, cost {cost}", flush=True)

        for target, _ in moves[blank]:
            if target in positions:
                i = positions.index(target)
                moved = positions[:i] + (blank,) + positions[i + 1 :]
                if not seen[_key(moved) * CELLS + target]:
                    queue.append((moved, target, cost + 1))
            elif not seen[key * CELLS + target]:
                queue.appendleft((positions, target, cost))
    return dist


def build(path=DATA):
    tables = {}
    for i, group in enumerate(GROUPS):
        print(f"group {i}: tiles {group}", flush=True)
        tables[f"group{i}"] = build_group(group, progress=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **tables)
    print(f"wrote {path}")


@lru_cache(maxsize=1)
def _tables_np():
    if not DATA.exists():
        raise FileNotFoundError(f"{DATA} missing; build it with: python -m npuzzle.pdb")
    with np.load(DATA) as data:
        return tuple(data[f"group{i}"].astype(np.int64) for i in range(len(GROUPS)))


@lru_cache(maxsize=1)
def _tables():
    # bytes index as fast as lists for single lookups (values fit in a byte)
    # and take 1 MB per table instead of about 8 MB.
    return tuple(t.astype(np.uint8).tobytes() for t in _tables_np())


def lookup(config, n) -> int:
    if n != N:
        raise ValueError("the pattern database covers the 4x4 puzzle only")
    position = [0] * CELLS
    for i, tile in enumerate(config):
        position[tile] = i
    return sum(
        table[_key(position[t] for t in group)]
        for table, group in zip(_tables(), GROUPS)
    )


if __name__ == "__main__":
    build()
