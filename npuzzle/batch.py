"""Vectorized N-Puzzle operations on batches of boards, for training and batched search.

A batch is an int8 array of shape (B, n*n), one board per row, in the same
layout as the tuple boards in npuzzle.board.
"""

from __future__ import annotations

from functools import cache

import numpy as np

from .board import goal, neighbors


@cache
def move_table(n: int) -> np.ndarray:
    """table[blank, a] = index the blank moves to for action a (UDLR), or -1."""
    order = {"Up": 0, "Down": 1, "Left": 2, "Right": 3}
    table = np.full((n * n, 4), -1, dtype=np.int64)
    for blank, moves in enumerate(neighbors(n)):
        for target, action in moves:
            table[blank, order[action]] = target
    return table


# Index of the action that undoes each action: Up<->Down, Left<->Right.
_INVERSE = np.array([1, 0, 3, 2])


def scramble(batch: int, n: int, depths: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Boards made by random walks from the goal, walk i taking depths[i] moves.

    Walks never undo their previous move, so the walk depth is a closer
    upper bound on the true distance.
    """
    table = move_table(n)
    boards = np.tile(np.array(goal(n), dtype=np.int8), (batch, 1))
    blank = np.zeros(batch, dtype=np.int64)
    prev = np.full(batch, -1)
    rows = np.arange(batch)
    for step in range(int(depths.max())):
        active = depths > step
        targets = table[blank]  # (B, 4)
        valid = targets >= 0
        has_prev = prev >= 0
        valid[rows[has_prev], _INVERSE[prev[has_prev]]] = False
        # Pick a uniformly random valid action per row.
        scores = np.where(valid, rng.random((batch, 4)), -1.0)
        action = scores.argmax(axis=1)
        target = targets[rows, action]
        idx = rows[active]
        t = target[active]
        boards[idx, blank[active]] = boards[idx, t]
        boards[idx, t] = 0
        blank[active] = t
        prev[active] = action[active]
    return boards


def children(boards: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    """All children of each board.

    Returns (kids, valid): kids has shape (B, 4, n*n) in UDLR order, and
    valid[b, a] is False where action a would move the blank off the board
    (that slot then holds a copy of the parent).
    """
    table = move_table(n)
    blank = (boards == 0).argmax(axis=1)
    targets = table[blank]  # (B, 4)
    valid = targets >= 0
    kids = np.repeat(boards[:, None, :], 4, axis=1)
    b_idx, a_idx = np.nonzero(valid)
    t = targets[b_idx, a_idx]
    kids[b_idx, a_idx, blank[b_idx]] = boards[b_idx, t]
    kids[b_idx, a_idx, t] = 0
    return kids, valid


def is_goal(boards: np.ndarray, n: int) -> np.ndarray:
    return (boards == np.arange(n * n, dtype=boards.dtype)).all(axis=1)


def manhattan_batch(boards: np.ndarray, n: int) -> np.ndarray:
    tiles = boards.astype(np.int64)
    pos = np.arange(n * n)
    d = np.abs(pos // n - tiles // n) + np.abs(pos % n - tiles % n)
    return np.where(tiles == 0, 0, d).sum(axis=1)


def pdb_batch(boards: np.ndarray) -> np.ndarray:
    """Vectorized 5-5-5 pattern database lookup for a batch of 4x4 boards."""
    from .pdb import CELLS, GROUPS, _tables_np

    # position[b, tile] = index of tile on board b
    position = np.argsort(boards, axis=1)
    total = np.zeros(len(boards), dtype=np.int64)
    for table, group in zip(_tables_np(), GROUPS):
        key = np.zeros(len(boards), dtype=np.int64)
        for t in group:
            key = key * CELLS + position[:, t]
        total += table[key]
    return total
