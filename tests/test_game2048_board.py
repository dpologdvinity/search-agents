import random

import pytest

from game2048.board import DOWN, LEFT, RIGHT, UP, empty_cells, from_grid, legal_moves, move, spawn, to_grid, transpose


def slide_row_left(row):
    """Reference 2048 slide on tile values."""
    tiles = [v for v in row if v]
    out, score, i = [], 0, 0
    while i < len(tiles):
        if i + 1 < len(tiles) and tiles[i] == tiles[i + 1]:
            out.append(2 * tiles[i])
            score += 2 * tiles[i]
            i += 2
        else:
            out.append(tiles[i])
            i += 1
    return out + [0] * (4 - len(out)), score


def reference_move(grid, d):
    g = [row[:] for row in grid]
    rotate = {LEFT: lambda g: g, RIGHT: lambda g: [r[::-1] for r in g],
              UP: lambda g: [list(c) for c in zip(*g)], DOWN: lambda g: [list(c)[::-1] for c in zip(*g)]}
    undo = {LEFT: lambda g: g, RIGHT: lambda g: [r[::-1] for r in g],
            UP: lambda g: [list(c) for c in zip(*g)], DOWN: lambda g: [list(c) for c in zip(*[r[::-1] for r in g])]}
    total, rows = 0, []
    for row in rotate[d](g):
        moved, s = slide_row_left(row)
        rows.append(moved)
        total += s
    return undo[d](rows), total


def test_merge_rules():
    board = from_grid([[2, 2, 2, 2], [4, 4, 8, 8], [2, 0, 0, 2], [0, 0, 0, 0]])
    new, score = move(board, LEFT)
    assert to_grid(new) == [[4, 4, 0, 0], [8, 16, 0, 0], [4, 0, 0, 0], [0, 0, 0, 0]]
    assert score == 4 + 4 + 8 + 16 + 4


@pytest.mark.parametrize("seed", range(200))
def test_moves_match_reference(seed):
    rng = random.Random(seed)
    grid = [[rng.choice([0, 0, 2, 4, 8, 16, 32]) for _ in range(4)] for _ in range(4)]
    board = from_grid(grid)
    for d in (UP, DOWN, LEFT, RIGHT):
        expected, score = reference_move(grid, d)
        new, gained = move(board, d)
        assert to_grid(new) == expected, (d, grid)
        assert gained == score


def test_transpose_roundtrip_and_meaning():
    rng = random.Random(0)
    for _ in range(100):
        x = rng.getrandbits(64)
        assert transpose(transpose(x)) == x
    grid = [[2, 4, 8, 16], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 32]]
    assert to_grid(transpose(from_grid(grid))) == [list(c) for c in zip(*grid)]


def test_spawn_and_legal_moves():
    rng = random.Random(0)
    board = spawn(0, rng)
    assert len(empty_cells(board)) == 15
    full = from_grid([[2, 4, 2, 4], [4, 2, 4, 2], [2, 4, 2, 4], [4, 2, 4, 2]])
    assert legal_moves(full) == []
