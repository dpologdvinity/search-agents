import random

import pytest

from npuzzle import SearchLimits, bfs, ida_star
from npuzzle.batched import batch_heuristic, batch_weighted_a_star
from npuzzle.heuristics import pattern_database
from tests.puzzles import solves, walk


@pytest.mark.parametrize("seed", range(10))
def test_batch_size_one_is_a_star(seed):
    # With one state per batch, g_weight 1, and a consistent heuristic, BWAS is A*.
    start = walk(3, 30, random.Random(seed))
    result = batch_weighted_a_star(start, batch_heuristic("manhattan", 3), batch_size=1)
    assert result.cost == bfs(start).cost
    assert solves(start, result.path)


@pytest.mark.parametrize("seed", range(5))
def test_large_batches_reach_goal_on_4x4(seed):
    start = walk(4, 40, random.Random(seed))
    optimum = ida_star(start, heuristic=pattern_database).cost
    for size in (10, 100):
        result = batch_weighted_a_star(start, batch_heuristic("pdb", 4), batch_size=size)
        assert result.solved
        assert solves(start, result.path)
        assert result.cost >= optimum


def test_unsolvable_and_limit():
    unsolvable = (1, 0, 2, 3, 4, 5, 6, 8, 7)
    h = batch_heuristic("manhattan", 3)
    assert batch_weighted_a_star(unsolvable, h).status == "unsolvable"
    hard = walk(4, 80, random.Random(0))
    result = batch_weighted_a_star(hard, batch_heuristic("manhattan", 4),
                                   batch_size=10, limits=SearchLimits(max_nodes=30))
    assert result.status == "limit"


def test_on_expand_sees_every_expansion():
    seen = []
    start = walk(3, 25, random.Random(3))
    result = batch_weighted_a_star(start, batch_heuristic("manhattan", 3), batch_size=8,
                                   on_expand=lambda node, f: seen.append(node.board))
    assert len(seen) == result.expanded
