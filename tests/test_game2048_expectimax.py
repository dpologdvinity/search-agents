import random

import numpy as np
import pytest

from game2048.board import from_grid, legal_moves, new_game, spawn
from game2048.expectimax import Expectimax, evaluate, features, features_batch

WEIGHTS = (1.0, 1.0, 0.5, 1.0, 0.5, 0.2)


def random_boards(n, seed):
    rng = random.Random(seed)
    return [from_grid([[rng.choice([0, 0, 2, 4, 8, 16, 64, 512]) for _ in range(4)] for _ in range(4)])
            for _ in range(n)]


def test_features_by_hand():
    board = from_grid([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 2, 2]])
    empty, monotonic, smooth, corner, merges, peak = features(board)
    assert empty == pytest.approx(np.log2(15))
    assert merges == 1 and peak == 1 and corner == 1
    assert smooth == 0 and monotonic == 0


def test_batch_features_match_scalar():
    boards = random_boards(200, 0)
    expected = np.array([features(b) for b in boards])
    np.testing.assert_allclose(features_batch(np.array(boards, dtype=np.uint64)), expected)


@pytest.mark.parametrize("seed", range(5))
def test_cache_does_not_change_values(seed):
    rng = random.Random(seed)
    board = new_game(rng)
    for _ in range(6):
        board = spawn(rng.choice(legal_moves(board))[1], rng)
    ev = lambda b: evaluate(b, WEIGHTS)  # noqa: E731
    full = Expectimax(budget=60, cache=False, prob_cutoff=0.0, max_depth=2, evaluator=ev).search(board)
    cached = Expectimax(budget=60, cache=True, prob_cutoff=0.0, max_depth=2, evaluator=ev).search(board)
    assert full.depth == cached.depth == 2
    for d in full.values:
        assert cached.values[d] == pytest.approx(full.values[d])
    assert cached.nodes <= full.nodes


def test_cache_and_cutoff_expand_fewer_nodes():
    board = from_grid([[2, 4, 8, 16], [0, 2, 4, 8], [0, 0, 2, 4], [0, 0, 0, 2]])
    ev = lambda b: evaluate(b, WEIGHTS)  # noqa: E731
    full = Expectimax(budget=60, cache=False, prob_cutoff=0.0, max_depth=3, evaluator=ev).search(board)
    fast = Expectimax(budget=60, max_depth=3, evaluator=ev).search(board)
    assert full.depth == fast.depth == 3
    assert fast.nodes < full.nodes
    assert fast.move in {d for d, _, _ in legal_moves(board)}
