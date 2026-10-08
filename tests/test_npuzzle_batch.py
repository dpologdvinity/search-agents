import numpy as np
import pytest

from npuzzle import is_solvable, manhattan, successors
from npuzzle.batch import children, is_goal, manhattan_batch, pdb_batch, scramble
from npuzzle.heuristics import pattern_database


@pytest.mark.parametrize("n", [3, 4])
def test_scramble_produces_valid_solvable_boards(n):
    rng = np.random.default_rng(0)
    depths = rng.integers(0, 60, size=200)
    boards = scramble(200, n, depths, rng)
    for board, depth in zip(boards, depths):
        config = tuple(int(t) for t in board)
        assert sorted(config) == list(range(n * n))
        assert is_solvable(config, n)
        # A walk of d moves changes Manhattan distance by at most d, with matching parity.
        assert manhattan(config, n) <= depth
        assert manhattan(config, n) % 2 == depth % 2
    assert is_goal(boards[depths == 0], n).all()


@pytest.mark.parametrize("n", [3, 4])
def test_children_match_scalar_expand(n):
    rng = np.random.default_rng(1)
    boards = scramble(50, n, rng.integers(0, 40, size=50), rng)
    kids, valid = children(boards, n)
    order = {"Up": 0, "Down": 1, "Left": 2, "Right": 3}
    for b, board in enumerate(boards):
        expected = {order[a]: b for a, b in successors(tuple(int(t) for t in board), n)}
        assert set(np.nonzero(valid[b])[0]) == set(expected)
        for a, config in expected.items():
            assert tuple(int(t) for t in kids[b, a]) == config


@pytest.mark.parametrize("n", [3, 4])
def test_manhattan_batch_matches_scalar(n):
    rng = np.random.default_rng(2)
    boards = scramble(100, n, rng.integers(0, 80, size=100), rng)
    expected = [manhattan(tuple(int(t) for t in b), n) for b in boards]
    assert manhattan_batch(boards, n).tolist() == expected


def test_pdb_batch_matches_scalar():
    rng = np.random.default_rng(3)
    boards = scramble(100, 4, rng.integers(0, 120, size=100), rng)
    expected = [pattern_database(tuple(int(t) for t in b), 4) for b in boards]
    assert pdb_batch(boards).tolist() == expected
