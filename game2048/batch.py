"""Vectorized 2048 moves and spawns over NumPy arrays of uint64 boards."""

from __future__ import annotations

import numpy as np

from .board import LEFT, LEFT_TABLE, RIGHT, RIGHT_TABLE, SCORE_TABLE, UP

_LEFT = np.array(LEFT_TABLE, dtype=np.uint64)
_RIGHT = np.array(RIGHT_TABLE, dtype=np.uint64)
_SCORE = np.array(SCORE_TABLE, dtype=np.int64)
_M16 = np.uint64(0xFFFF)
_SHIFTS = [np.uint64(s) for s in (0, 16, 32, 48)]


def transpose_batch(x: np.ndarray) -> np.ndarray:
    u = np.uint64
    a1 = x & u(0xF0F00F0FF0F00F0F)
    a2 = x & u(0x0000F0F00000F0F0)
    a3 = x & u(0x0F0F00000F0F0000)
    a = a1 | (a2 << u(12)) | (a3 >> u(12))
    b1 = a & u(0xFF00FF0000FF00FF)
    b2 = a & u(0x00FF00FF00000000)
    b3 = a & u(0x00000000FF00FF00)
    return b1 | (b2 >> u(24)) | (b3 << u(24))


def _rows(x, table):
    out = np.zeros_like(x)
    score = np.zeros(len(x), dtype=np.int64)
    for s in _SHIFTS:
        row = ((x >> s) & _M16).astype(np.int64)
        out |= table[row] << s
        score += _SCORE[row]
    return out, score


def move_batch(boards: np.ndarray, direction: int):
    """Slide every board in one direction: (new boards, score gained)."""
    if direction in (LEFT, RIGHT):
        return _rows(boards, _LEFT if direction == LEFT else _RIGHT)
    t = transpose_batch(boards)
    moved, score = _rows(t, _LEFT if direction == UP else _RIGHT)
    return transpose_batch(moved), score


def spawn_batch(boards: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Add a 2 (p=0.9) or 4 (p=0.1) to a random empty cell of each board that has one."""
    cells = np.stack([(boards >> np.uint64(4 * i)) & np.uint64(0xF) for i in range(16)], axis=1)
    empty = cells == 0
    scores = np.where(empty, rng.random(empty.shape), -1.0)
    cell = scores.argmax(axis=1)
    has_empty = empty.any(axis=1)
    value = np.where(rng.random(len(boards)) < 0.9, 1, 2).astype(np.uint64)
    tile = value << (np.uint64(4) * cell.astype(np.uint64))
    return np.where(has_empty, boards | tile, boards)


def new_games(n: int, rng: np.random.Generator) -> np.ndarray:
    return spawn_batch(spawn_batch(np.zeros(n, dtype=np.uint64), rng), rng)


def max_exponent_batch(boards: np.ndarray) -> np.ndarray:
    return np.stack([(boards >> np.uint64(4 * i)) & np.uint64(0xF) for i in range(16)], axis=1).max(axis=1)
