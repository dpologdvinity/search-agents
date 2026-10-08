import itertools
import random

import pytest

from nonogram import __main__ as cli
from nonogram.automaton import build
from nonogram.cnf import encode
from nonogram.dpll import DPLL, SAT, UNKNOWN, UNSAT, check_model, pure_literal_eliminate
from nonogram.generate import random_picture, random_puzzle
from nonogram.library import PICTURES, get, library
from nonogram.lines import analyse, solve_line
from nonogram.puzzle import Puzzle, clues_of, render, runs_of, validate_clues
from nonogram.solvers import count_solutions, hint, solve, verify


def words(n):
    return itertools.product((0, 1), repeat=n)


# -- automaton and line solver ------------------------------------------------------------------


def test_automaton_accepts_exactly_the_lines_that_fit_the_clue():
    rng = random.Random(7)
    for _ in range(300):
        n = rng.randint(1, 8)
        runs = tuple(rng.randint(1, 3) for _ in range(rng.randint(0, 3)))
        aut = build(runs)
        for w in words(n):
            assert aut.accepts(w) == (runs_of(w) == runs), (runs, w)


def test_line_analysis_matches_brute_force_over_all_arrangements():
    rng = random.Random(11)
    for _ in range(1500):
        n = rng.randint(1, 8)
        runs = tuple(rng.randint(1, 3) for _ in range(rng.randint(0, 3)))
        known = [rng.choice([-1, -1, 0, 1]) for _ in range(n)]
        fit = [w for w in words(n) if runs_of(w) == runs and all(known[i] in (-1, w[i]) for i in range(n))]
        info = analyse(runs, known)
        assert info.consistent == bool(fit)
        for i in range(n):
            expected = 0
            for w in fit:
                expected |= 1 << w[i]
            assert info.possible[i] == expected


def test_valid_starts_match_the_run_starts_of_the_fitting_lines():
    rng = random.Random(5)
    for _ in range(400):
        n = rng.randint(2, 8)
        runs = tuple(rng.randint(1, 2) for _ in range(rng.randint(1, 3)))
        info = analyse(runs, [-1] * n)
        fit = [w for w in words(n) if runs_of(w) == runs]
        for k in range(len(runs)):
            seen = set()
            for w in fit:
                starts = [i for i in range(n) if w[i] == 1 and (i == 0 or w[i - 1] == 0)]
                seen.add(starts[k])
            assert set(info.starts[k]) == seen


def test_solve_line_fixes_only_forced_cells():
    # "3" in 5 cells: the middle cell is always filled; the ends are not forced.
    assert solve_line((3,), [-1] * 5) == [-1, -1, 1, -1, -1]
    # "1 1" in 3 cells is only X.X, so the whole line is forced.
    assert solve_line((1, 1), [-1] * 3) == [1, 0, 1]
    # A filled first cell opens the run of 2 at cell 0, so the cell after it must be empty.
    assert solve_line((2,), [1, -1, -1, -1]) == [1, 1, 0, 0]
    assert solve_line((3,), [1, 0, -1, -1, -1]) is None


# -- SAT: DPLL and the encoding -----------------------------------------------------------------


def _brute_sat(nvars, clauses):
    for bits in itertools.product((False, True), repeat=nvars):
        model = [False] + list(bits)
        if check_model(clauses, model):
            return True
    return False


def test_dpll_agrees_with_brute_force_on_random_cnf():
    rng = random.Random(3)
    for _ in range(300):
        nvars = rng.randint(1, 8)
        clauses = []
        for _ in range(rng.randint(1, 20)):
            k = rng.randint(1, 3)
            clauses.append([rng.choice([-1, 1]) * rng.randint(1, nvars) for _ in range(k)])
        result = DPLL(nvars, clauses).solve()
        assert result == (SAT if _brute_sat(nvars, clauses) else UNSAT), clauses
        if result == SAT:
            solver = DPLL(nvars, clauses)
            solver.solve()
            assert check_model(clauses, solver.model())


def test_pure_literal_elimination_keeps_satisfiability():
    clauses = [[1, 2], [1, -3], [-2, 3], [4, 4]]  # 1 and 4 are pure (positive only)
    remaining, fixed = pure_literal_eliminate(clauses, 4)
    assert fixed.get(1) is True and fixed.get(4) is True
    assert all(abs(d) not in (1, 4) for cl in remaining for d in cl)
    assert DPLL(4, clauses).solve() == SAT


def test_budget_stops_the_search_with_unknown():
    # Units force propagation before a decision is needed, so a zero budget is exceeded.
    clauses = [[1], [2, 3], [-2, -3]]
    assert DPLL(3, clauses).solve(max_propagations=0) == UNKNOWN


def test_encoding_models_are_exactly_the_solutions():
    # A 2x2 diagonal has two solutions; any model of the encoding must be one of them.
    p = Puzzle.from_picture("tiny", ["#.", ".#"])
    cnf = encode(p)
    solver = DPLL(cnf.nvars, cnf.clauses, priority=cnf.cell_vars())
    assert solver.solve() == SAT
    model = solver.model()
    grid = [[1 if model[cnf.cell_var[r][c]] else 0 for c in range(2)] for r in range(2)]
    assert grid in ([[1, 0], [0, 1]], [[0, 1], [1, 0]])
    # Every grid that is not a solution must violate the encoding.
    for bits in itertools.product((0, 1), repeat=4):
        g = [list(bits[:2]), list(bits[2:])]
        if g in ([[1, 0], [0, 1]], [[0, 1], [1, 0]]):
            continue
        assumptions = [v if g[r][c] else -v for r in range(2) for c in range(2) for v in [cnf.cell_var[r][c]]]
        assert DPLL(cnf.nvars, cnf.clauses + [[a] for a in assumptions]).solve() == UNSAT


# -- library and methods --------------------------------------------------------------------------


def test_library_has_twelve_uniquely_solvable_pictures_in_range():
    puzzles = library()
    assert 10 <= len(puzzles) <= 20
    assert len({p.id for p in puzzles}) == len(puzzles)
    for p in puzzles:
        assert 5 <= p.rows <= 20 and 5 <= p.cols <= 20
        status, sols, _ = count_solutions(p, limit=2)
        assert status == "unique", p.id


@pytest.mark.parametrize("method", ["line", "hybrid", "sat"])
def test_each_method_recovers_the_library_picture(method):
    for pid, _, art in PICTURES:
        p = get(pid)
        res = solve(p, method, trace=True)
        if method == "line" and res.status == "undecided":
            assert res.partial is not None  # line solving alone may stop; its partial grid is consistent
            continue
        picture = [[1 if ch == "#" else 0 for ch in row] for row in art]
        assert res.status == "unique", (pid, method)
        assert res.solution == picture
        assert verify(p, res.solution)


def test_line_solving_alone_finishes_the_easy_pictures_and_stops_on_the_rest():
    done = {p.id for p in library() if solve(p, "line").status == "unique"}
    assert {"heart", "arrow", "invader", "key", "house"} <= done
    assert "rocket" not in done  # the rocket needs guessing


def test_line_trace_records_each_pass_with_its_fixed_cells():
    p = get("heart")
    res = solve(p, "line", trace=True)
    assert res.rounds and res.rounds[0]["axis"] == "rows"
    fixed = [cell for rnd in res.rounds for line in rnd["lines"] for cell in line["fixed"]]
    assert len(fixed) == p.rows * p.cols  # every cell of the heart is fixed by line passes
    assert all(line["line"][0] in "rc" for rnd in res.rounds for line in rnd["lines"])


def test_two_solutions_are_reported_as_multiple_by_every_method():
    p = Puzzle("diagonals", ((1,), (1,)), ((1,), (1,)))
    for method in ("hybrid", "sat"):
        res = solve(p, method)
        assert res.status == "multiple", method
        assert res.solution != res.second
        assert verify(p, res.solution) and verify(p, res.second)
    assert solve(p, "line").status == "undecided"  # line solving cannot choose between the two


def test_contradictory_clues_are_reported_as_contradictions():
    p = Puzzle("clash", ((2,), ()), ((1,), (1,)))
    p2 = Puzzle("clash2", ((2,), (2,)), ((1,), (1,)))
    for method in ("line", "hybrid", "sat"):
        assert solve(p2, method).status == "contradiction", method
    assert solve(p, "hybrid").status == "unique"  # sanity: a consistent variant


def test_validate_clues_rejects_impossible_lines():
    with pytest.raises(ValueError):
        validate_clues(((3, 3),), ((1,), (1,), (1,), (1,), (1,), (1,)))  # 3 + 1 + 3 = 7 > 6 cells
    with pytest.raises(ValueError):
        validate_clues(((0,),), ((1,),))  # runs must be at least 1
    assert validate_clues(((1, 1),), ((1,), (1,), (1,))) == (1, 3)


def test_hint_names_a_forced_cell_and_its_reason():
    p = get("heart")
    grid = [[-1] * p.cols for _ in range(p.rows)]
    h = hint(p, grid)
    assert h["found"] and h["value"] in (0, 1)
    assert h["reason"]
    # The solution's first forced cell must agree with the hint's value.
    solution = solve(p, "hybrid").solution
    assert solution[h["row"]][h["col"]] == h["value"]


def test_hint_reports_marks_that_break_a_line():
    p = get("heart")
    grid = [[-1] * p.cols for _ in range(p.rows)]
    grid[1][0] = 0  # row 2 of the heart is "#####" (clue 5): crossing out its first cell cannot fit
    h = hint(p, grid)
    assert h["contradiction"] and not h["found"]


def test_hint_on_a_complete_correct_grid_says_so():
    p = get("arrow")
    solution = solve(p, "hybrid").solution
    h = hint(p, solution)
    assert not h["found"] and not h["contradiction"]


# -- random generator -----------------------------------------------------------------------------


def test_random_puzzles_are_unique_and_repeat_for_a_seed():
    a, seed, _ = random_puzzle(10, 10, seed=42)
    b, seed2, _ = random_puzzle(10, 10, seed=42)
    assert a == b and seed == seed2 == 42
    assert count_solutions(a, limit=2)[0] == "unique"
    assert (a.rows, a.cols) == (10, 10)


def test_random_picture_is_blobby_not_noise():
    pic = random_picture(15, 15, random.Random(9))
    changes = sum(pic[r][c] != pic[r][c + 1] for r in range(15) for c in range(14))
    assert changes < 15 * 14 * 0.4  # neighbours mostly agree


def test_clues_of_a_picture():
    pic = [[1, 0, 1, 1], [0, 0, 0, 1]]
    rows, cols = clues_of(pic)
    assert rows == ((1, 2), (1,))
    assert cols == ((1,), (), (1,), (2,))


# -- CLI -------------------------------------------------------------------------------------------


def test_play_solves_the_heart_when_every_cell_is_filled_correctly(capsys):
    p = get("heart")
    picture = [[1 if ch == "#" else 0 for ch in row] for row in PICTURES[0][2]]
    moves = [f"f {r + 1} {c + 1}" for r in range(p.rows) for c in range(p.cols) if picture[r][c]]
    feed = iter(moves)
    assert cli.play(p, read=lambda prompt: next(feed), out=lambda *a: None) is True


def test_play_quits_on_q_and_shows_hints(capsys):
    p = get("heart")
    feed = iter(["h", "q"])
    lines = []
    assert cli.play(p, read=lambda prompt: next(feed), out=lambda *a: lines.append(" ".join(map(str, a)))) is False
    assert any("hint:" in line for line in lines)


def test_cli_list_and_solve_print_the_expected_text(capsys):
    assert cli.main(["list"]) == 0
    out = capsys.readouterr().out
    assert "heart" in out and "sailboat" in out
    assert cli.main(["solve", "--puzzle", "arrow", "--method", "sat", "--stats"]) == 0
    out = capsys.readouterr().out
    assert "unique" in out and "variables" in out


def test_cli_random_is_repeatable(capsys):
    cli.main(["random", "8x8", "--seed", "5"])
    first = capsys.readouterr().out
    cli.main(["random", "8x8", "--seed", "5"])
    assert first == capsys.readouterr().out


def test_render_shows_column_clues_above_the_grid():
    p = get("heart")
    lines = render(p).splitlines()
    # One line of column clues, the column letters, then one line per row.
    assert len(lines) == 1 + 1 + 5
    assert lines[0].split() == ["2", "4", "4", "4", "2"]
    assert lines[1].split() == ["a", "b", "c", "d", "e"]
