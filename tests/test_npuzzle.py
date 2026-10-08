import random

import pytest

from npuzzle import (
    ALGORITHMS,
    HEURISTICS,
    SearchLimits,
    a_star,
    apply,
    bfs,
    goal,
    ida_star,
    is_solvable,
    manhattan,
    replay,
    successors,
    validate,
    weighted_a_star,
)
from npuzzle.__main__ import main
from npuzzle.board import Node
from npuzzle.frontier import PriorityFrontier
from npuzzle.heuristics import linear_conflict, pattern_database
from tests.puzzles import random_board, walk

OPTIMAL = ["bfs", "ids", "ucs", "bibfs", "astar", "idastar"]
UNSOLVABLE = (1, 0, 2, 3, 4, 5, 6, 8, 7)  # one swapped pair: odd inversions


def test_board_basics():
    assert goal(3) == (0, 1, 2, 3, 4, 5, 6, 7, 8)
    assert [a for a, _ in successors(goal(3), 3)] == ["Down", "Right"]
    assert apply(goal(3), "Right", 3) == (1, 0, 2, 3, 4, 5, 6, 7, 8)
    assert replay(goal(3), ["Right", "Left"], 3) == goal(3)
    with pytest.raises(ValueError):
        apply(goal(3), "Up", 3)
    with pytest.raises(ValueError):
        validate((1, 2, 3))
    with pytest.raises(ValueError):
        validate((0, 1, 1, 3, 4, 5, 6, 7, 8))


def test_solvability():
    assert not is_solvable(UNSOLVABLE, 3)
    assert is_solvable(goal(4), 4)
    swapped = list(goal(4))
    swapped[1], swapped[2] = swapped[2], swapped[1]
    assert not is_solvable(tuple(swapped), 4)
    for seed in range(20):
        assert is_solvable(walk(4, 30, random.Random(seed)), 4)


@pytest.mark.parametrize("algorithm", OPTIMAL)
@pytest.mark.parametrize("seed", range(10))
def test_optimal_algorithms_agree(algorithm, seed):
    start = random_board(3, random.Random(seed))
    result = ALGORITHMS[algorithm](start)
    assert result.solved
    assert result.cost == bfs(start).cost == len(result.path)
    assert replay(start, result.path, 3) == goal(3)


@pytest.mark.parametrize("seed", range(10))
def test_suboptimal_algorithms_reach_goal(seed):
    start = random_board(3, random.Random(seed))
    optimum = bfs(start).cost
    for algorithm in ("dfs", "greedy", "wastar"):
        result = ALGORITHMS[algorithm](start)
        assert result.solved
        assert replay(start, result.path, 3) == goal(3)
    assert weighted_a_star(start, weight=2.0).cost <= 2 * optimum


def test_ida_star_matches_a_star_on_4x4():
    for seed in range(5):
        start = walk(4, 40, random.Random(seed))
        assert ida_star(start).cost == a_star(start).cost


def test_already_solved():
    for name, search in ALGORITHMS.items():
        result = search(goal(3))
        assert result.solved and result.cost == 0, name


def test_unsolvable_is_detected():
    for search in ALGORITHMS.values():
        assert search(UNSOLVABLE).status == "unsolvable"


@pytest.mark.parametrize("algorithm", sorted(ALGORITHMS))
def test_every_algorithm_respects_node_limit(algorithm):
    hard = walk(3, 60, random.Random(7))
    result = ALGORITHMS[algorithm](hard, limits=SearchLimits(max_nodes=50))
    assert result.status in ("limit", "solved")
    assert result.expanded <= 50


@pytest.mark.parametrize("algorithm", sorted(ALGORITHMS))
def test_on_expand_sees_every_expansion(algorithm):
    seen = []
    start = walk(3, 25, random.Random(3))
    result = ALGORITHMS[algorithm](start, on_expand=lambda node, size: seen.append(node.depth))
    assert len(seen) == result.expanded
    assert max(seen) == result.max_depth


def test_statistics_are_consistent():
    start = walk(3, 30, random.Random(11))
    result = a_star(start)
    assert result.generated >= result.expanded - 1
    assert result.max_depth <= result.cost
    assert result.max_frontier > 0 and result.seconds >= 0


def test_priority_frontier_empties_with_stale_entries():
    # A heap holding only stale entries must count as empty.
    frontier = PriorityFrontier()
    node = Node(walk(3, 10, random.Random(0)))
    frontier.add(node, 10)
    assert frontier.decrease_key(node, 5)
    assert not frontier.decrease_key(node, 7)
    assert frontier.pop() is node
    assert len(frontier) == 0
    with pytest.raises(IndexError):
        frontier.pop()


def test_manhattan():
    assert manhattan(goal(3), 3) == 0
    assert manhattan(apply(goal(3), "Down", 3), 3) == 1
    assert manhattan((8, 7, 6, 5, 4, 3, 2, 1, 0), 3) == 20


def test_cli_prints_summary(capsys):
    assert main(["astar", "7,2,4,5,0,6,8,3,1"]) == 0
    out = capsys.readouterr().out
    assert "astar: solved" in out and "moves:" in out
    assert main(["bfs", ",".join(map(str, UNSOLVABLE))]) == 1


@pytest.mark.parametrize("seed", range(30))
def test_linear_conflict_is_admissible_and_dominates_manhattan(seed):
    start = walk(3, 30, random.Random(seed))
    h = linear_conflict(start, 3)
    assert manhattan(start, 3) <= h <= bfs(start).cost


def test_linear_conflict_counts_reversed_pair():
    # Tiles 2 and 1 are both in their goal row but swapped: Manhattan is 2,
    # and one tile must step out of the row and back, adding 2.
    board = (0, 2, 1, 3, 4, 5, 6, 7, 8)
    assert manhattan(board, 3) == 2
    assert linear_conflict(board, 3) == 4


def test_pattern_database_is_zero_at_goal():
    assert pattern_database(goal(4), 4) == 0


@pytest.mark.parametrize("seed", range(5))
def test_heuristics_agree_on_optimal_cost_4x4(seed):
    start = walk(4, 35, random.Random(seed))
    costs = {name: ida_star(start, heuristic=h).cost for name, h in HEURISTICS.items()}
    assert len(set(costs.values())) == 1
    for name, h in HEURISTICS.items():
        assert h(start, 4) <= costs[name]
    assert pattern_database(start, 4) >= manhattan(start, 4)
