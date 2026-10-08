import random

import pytest

from sudoku.generate import count_solutions, load_puzzles, make_puzzle, random_solution
from sudoku.solver import SOLVERS, is_valid_solution, parse, to_string

PUZZLES = load_puzzles()

# Arto Inkala's published "world's hardest sudoku" (2012).
HARD = "800000000003600000070090000050007000000045700000100030001000068068000400000000000"


def test_generated_set_is_well_formed():
    assert len(PUZZLES) == 200
    for row in PUZZLES[:20]:
        puzzle, solution = parse(row["puzzle"]), parse(row["solution"])
        assert is_valid_solution(solution, puzzle)
        assert count_solutions(puzzle) == 1


@pytest.mark.parametrize("name", sorted(SOLVERS))
@pytest.mark.parametrize("k", range(0, 200, 25))
def test_solvers_find_the_unique_solution(name, k):
    row = PUZZLES[k]
    result = SOLVERS[name](row["puzzle"], max_nodes=2_000_000)
    assert result.status == "solved"
    assert result.solution == row["solution"]


@pytest.mark.parametrize("name", ["mrv_fc", "propagate"])
def test_hard_puzzle(name):
    result = SOLVERS[name](HARD)
    assert result.status == "solved"
    assert is_valid_solution(parse(result.solution), parse(HARD))


def test_propagation_needs_fewer_guesses_than_mrv_fc():
    a = SOLVERS["mrv_fc"](HARD)
    b = SOLVERS["propagate"](HARD)
    assert b.nodes < a.nodes


@pytest.mark.parametrize("name", sorted(SOLVERS))
def test_unsolvable_and_limit(name):
    bad = "11" + "0" * 79  # two 1s in the first row
    assert SOLVERS[name](bad).status == "unsolvable"
    assert SOLVERS[name](HARD, max_nodes=3).status in ("limit", "solved")


def test_trace_records_guesses_and_backtracks():
    result = SOLVERS["mrv_fc"](HARD, trace_limit=100_000)
    assigns = [t for t in result.trace if t[0] == "assign"]
    undos = [t for t in result.trace if t[0] == "undo"]
    assert len(assigns) == result.nodes and len(undos) == result.backtracks


@pytest.mark.parametrize("seed", range(3))
def test_generator_makes_minimal_unique_puzzles(seed):
    rng = random.Random(seed)
    assert is_valid_solution(random_solution(rng))
    puzzle, solution = make_puzzle(rng)
    cells = parse(puzzle)
    assert count_solutions(cells) == 1
    assert is_valid_solution(parse(solution), cells)
    # Minimal: removing any remaining clue makes the solution non-unique.
    for i in [i for i, v in enumerate(cells) if v][:5]:
        fewer = cells[:]
        fewer[i] = 0
        assert count_solutions(fewer) == 2


def test_count_solutions_on_empty_and_contradictory_grids():
    assert count_solutions([0] * 81) == 2
    assert count_solutions(parse("11" + "0" * 79)) == 0
    assert to_string(parse(HARD)) == HARD
