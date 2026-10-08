import random

import pytest

from queens.__main__ import draw, hint, main, parse_square, play
from queens.agents import AGENT_NAMES, anneal, backtrack, hill_climb, min_conflicts, run
from queens.benchmark import run_benchmark, to_markdown, write_results
from queens.board import Board, brute_conflicts, count_conflicts, is_solution, render

EIGHT_QUEENS = [0, 4, 7, 5, 2, 6, 1, 3]  # one valid 8-queens placement, by column for each row


def brute_attacks_if(cols, r, c):
    """Queens (other than the one in row r) that attack square (r, c), by checking each queen."""
    count = 0
    for r2, c2 in enumerate(cols):
        if r2 == r or c2 < 0:
            continue
        if c2 == c or abs(r2 - r) == abs(c2 - c):
            count += 1
    return count


def test_incremental_conflicts_match_brute_force_under_random_moves():
    rng = random.Random(3)
    for n in (4, 7, 12):
        board = Board(n, [rng.randrange(n) for _ in range(n)])
        for _ in range(300):
            r = rng.randrange(n)
            if rng.random() < 0.2 and board.cols[r] >= 0:
                board.remove(r)
            else:
                board.move(r, rng.randrange(n))
            assert board.conflicts == brute_conflicts(board.cols)


def test_attacks_if_matches_brute_force_including_own_row():
    rng = random.Random(5)
    n = 9
    cols = [rng.randrange(n) if rng.random() < 0.8 else -1 for _ in range(n)]
    board = Board(n, cols)
    for r in range(n):
        for c in range(n):
            assert board.attacks_if(r, c) == brute_attacks_if(board.cols, r, c)
        if board.cols[r] >= 0:
            assert board.queen_attacks(r) == brute_attacks_if(board.cols, r, board.cols[r])


def test_count_conflicts_is_exact_and_is_solution_agrees():
    rng = random.Random(9)
    for _ in range(50):
        n = rng.randrange(1, 12)
        cols = [rng.randrange(n) for _ in range(n)]
        assert count_conflicts(cols) == brute_conflicts(cols)
    assert is_solution(EIGHT_QUEENS)
    assert not is_solution(EIGHT_QUEENS[:-1] + [EIGHT_QUEENS[-1] ^ 1])  # one queen shifted: conflicts
    assert not is_solution([0, 1, -1])  # an empty row is never a solution


def test_board_rejects_double_place_and_empty_remove():
    board = Board(4)
    board.place(0, 1)
    with pytest.raises(ValueError):
        board.place(0, 2)
    with pytest.raises(ValueError):
        board.remove(1)


def test_backtrack_finds_valid_solutions_and_proves_small_cases_impossible():
    for n in range(4, 11):
        res = backtrack(n)
        assert res.solved and is_solution(res.cols) and res.conflicts == 0
    for n in (2, 3):
        res = backtrack(n)
        assert not res.solved and res.reason == "exhausted"


def test_backtrack_node_counts_show_the_blow_up():
    # The row-order search is deterministic, so these counts are fixed. The eight-queens search places 113
    # queens before it finds its first solution; a 16-queens search needs over 10,000.
    assert backtrack(8).evaluations == 113
    assert backtrack(16).evaluations > 10_000


def test_backtrack_respects_its_step_cap():
    res = backtrack(32, max_steps=2_000)
    assert not res.solved and res.reason == "step cap" and res.steps == 2_000


@pytest.mark.parametrize("agent", ["hill", "anneal", "minconf"])
def test_local_search_solves_eight_queens_and_reports_it_honestly(agent):
    for seed in range(1, 6):
        res = run(agent, 8, seed, time_limit=10)
        assert res.solved, (agent, seed, res.reason)
        assert is_solution(res.cols) and res.conflicts == 0
        assert brute_conflicts(res.cols) == 0


def test_min_conflicts_solves_a_thousand_queens_in_few_steps():
    res = min_conflicts(1000, random.Random(1), time_limit=30)
    assert res.solved and is_solution(res.cols)
    assert res.steps < 1000  # the greedy start leaves a few hundred conflicts at most


def test_min_conflicts_never_increases_the_conflict_count():
    # Each repair removes a queen (its attacks go) and puts it on a least-attacked column. The current
    # column is among the candidates, so the new column is never worse: the trace must not rise. A budget
    # of 400 steps makes the recorder keep every step, so the check covers the whole run.
    res = min_conflicts(64, random.Random(4), max_steps=400)
    assert res.solved and res.restarts == 0  # a restart is a deliberate jump up, so the check applies without one
    assert len(res.trace) == res.steps + 1
    assert all(a >= b for a, b in zip(res.trace, res.trace[1:]))
    assert res.trace[-1] == 0


def test_hill_climb_counts_restarts_and_respects_the_restart_cap():
    res = hill_climb(12, random.Random(2), max_restarts=0, time_limit=10)
    if not res.solved:
        assert res.reason == "restart cap" and res.restarts == 1
    else:
        assert res.restarts == 0


def test_anneal_frames_start_from_the_initial_placement():
    res = anneal(8, random.Random(1), max_steps=5000, record_frames=True)
    assert res.solved
    assert res.frames is not None and len(res.frames[0]) == 8
    assert res.frames[-1] == res.cols


def test_run_rejects_unknown_agent_names():
    with pytest.raises(ValueError):
        run("genetic", 8)
    assert set(AGENT_NAMES) == {"backtrack", "hill", "anneal", "minconf"}


def test_render_and_draw_mark_queens():
    text = render(EIGHT_QUEENS)
    assert text.count("Q") == 8
    board = Board(4, [0, 1, 2, 3])  # a diagonal line of queens: all attack each other
    assert draw(board).count("!") == 4 and "Q" not in draw(board).split("\n", 1)[1]


def test_parse_square_reads_column_letter_then_row():
    assert parse_square("c3", 8) == (2, 2)
    assert parse_square("H8", 8) == (7, 7)
    assert parse_square("i1", 8) is None  # off the board
    assert parse_square("a9", 8) is None
    assert parse_square("3c", 8) is None


def test_hint_names_a_move_for_a_conflicted_board():
    board = Board(8, [0, 0, -1, -1, -1, -1, -1, -1])  # two queens in column a, in rows 1 and 2
    text = hint(board)
    assert text.startswith("min-conflicts: move the row 1 queen from a1")


def test_scripted_play_solves_the_board():
    moves = iter([f"{chr(97 + c)}{r + 1}" for r, c in enumerate(EIGHT_QUEENS)])
    lines = []
    n = play(8, read=lambda prompt: next(moves), out=lines.append)
    assert n == 8
    assert any("Solved in 8 moves" in line for line in lines)


def test_play_square_toggles_and_moves():
    answers = iter(["a1", "b1", "b1", "zz", "q"])  # place, move within the row, take it back, bad input, quit
    lines = []
    assert play(4, read=lambda prompt: next(answers), out=lines.append) == 3
    assert any("'zz' is not a square" in line for line in lines)


def test_solve_command_prints_a_checked_summary(capsys):
    assert main(["solve", "-n", "8", "--agent", "backtrack", "--board"]) == 0
    out = capsys.readouterr().out
    assert "solved (solved)" in out and "checked: no two queens attack each other" in out
    assert out.count("Q") == 8


def test_solve_command_exit_code_reports_failure(capsys):
    assert main(["solve", "-n", "3", "--agent", "backtrack"]) == 1
    assert "exhausted" in capsys.readouterr().out


def test_benchmark_writes_json_and_markdown(tmp_path):
    plan = [
        ("backtrack", 8, 1, 5.0, 100_000),
        ("hill", 8, 3, 5.0, None),
        ("anneal", 8, 3, 5.0, None),
        ("minconf", 16, 3, 5.0, None),
        ("hill", 10000, 0, 0.0, None),
    ]
    data = run_benchmark(plan)
    assert data["agents"]["backtrack"]["8"]["success_rate"] == 1.0
    assert data["agents"]["minconf"]["16"]["runs"] == 3
    assert "hill N=10000" in data["skipped"]
    json_path, md_path = write_results(data, tmp_path)
    assert json_path.exists() and md_path.exists()
    md = md_path.read_text()
    assert "| minconf | 16 |" in md and "not run" in to_markdown(data)
