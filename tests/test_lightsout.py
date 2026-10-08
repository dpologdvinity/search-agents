import random

import pytest

from lightsout.__main__ import main, parse_board_text, play, print_solution
from lightsout.board import (
    apply_presses,
    is_dark,
    label,
    neighbourhood,
    parse_cell,
    random_board,
    to_board,
    to_mask,
)
from lightsout.solver import null_space, solve


def dark(n):
    return (0,) * (n * n)


def unsolvable_board(n):
    """A board with one lit cell, chosen so the board cannot be cleared (see test_unsolvable_boards_are_detected)."""
    v = null_space(n)[0]
    return to_board(n, 1 << ((v & -v).bit_length() - 1))


def brute_force_minimum(n):
    """Map every reachable board to its minimum press count, by trying all 2**(n*n) press sets.

    The board for press set x is the XOR of the neighbourhoods of its cells, so each
    entry is built from a smaller press set (x without its lowest cell) in one XOR.
    """
    cells = n * n
    cols = [neighbourhood(n, i) for i in range(cells)]
    board = [0] * (1 << cells)
    count = [0] * (1 << cells)
    best = {}
    for x in range(1, 1 << cells):
        low = x & -x
        i = low.bit_length() - 1
        board[x] = board[x ^ low] ^ cols[i]
        count[x] = count[x ^ low] + 1
    for x in range(1 << cells):
        b = board[x]
        if b not in best or count[x] < best[b]:
            best[b] = count[x]
    return best


def test_press_toggles_cell_and_neighbours():
    n = 5
    assert {i for i in range(25) if neighbourhood(n, 0) >> i & 1} == {0, 1, 5}  # corner: 3 lights
    assert {i for i in range(25) if neighbourhood(n, 2) >> i & 1} == {1, 2, 3, 7}  # edge: 4 lights
    assert {i for i in range(25) if neighbourhood(n, 12) >> i & 1} == {7, 11, 12, 13, 17}  # centre: 5 lights


def test_presses_commute_and_cancel():
    b = random_board(5, random.Random(7))
    assert apply_presses(5, b, [3, 9, 14]) == apply_presses(5, b, [14, 3, 9])
    assert apply_presses(5, b, [3, 3]) == b


def test_five_by_five_rank_and_null_space():
    sol = solve(5, dark(5))
    assert sol.rank == 23 and sol.nullity == 2
    basis = null_space(5)
    assert len(basis) == 2
    for v in basis:  # a null-space press pattern changes no light
        assert is_dark(apply_presses(5, dark(5), [i for i in range(25) if v >> i & 1]))
    # A solvable 5x5 board has exactly 2**nullity = 4 solutions.
    assert solve(5, apply_presses(5, dark(5), [0, 6, 18])).count == 4


@pytest.mark.parametrize("n", range(3, 10))
def test_applying_the_solution_clears_the_board(n):
    rng = random.Random(n)
    for _ in range(5):
        b = random_board(n, rng, solvable=True)
        sol = solve(n, b)
        assert sol.solvable
        assert is_dark(apply_presses(n, b, sol.presses))


@pytest.mark.parametrize("n", range(3, 10))
def test_unsolvable_boards_are_detected(n):
    # If v is a null vector of A (A is symmetric), then v . b = 0 (mod 2) for every solvable b.
    # A board with one lit cell c where v[c] = 1 has v . b = 1, so it cannot be cleared.
    for v in null_space(n):
        assert is_dark(apply_presses(n, dark(n), [i for i in range(n * n) if v >> i & 1]))
        b = unsolvable_board(n)
        sol = solve(n, b)
        assert not sol.solvable and sol.presses is None and sol.count == 0
        assert sol.trace is None


def test_solvability_matches_the_orthogonality_test():
    # Solvable iff b is orthogonal to every null vector (mod 2). An independent check of the elimination.
    n = 5
    basis = null_space(n)
    rng = random.Random(3)
    seen = {True: 0, False: 0}
    for _ in range(300):
        b = random_board(n, rng, solvable=rng.random() < 0.5)
        orthogonal = all(bin(v & to_mask(b)).count("1") % 2 == 0 for v in basis)
        assert solve(n, b).solvable == orthogonal
        seen[orthogonal] += 1
    assert seen[True] and seen[False]  # the sample covers both outcomes


@pytest.mark.parametrize("n", [3, 4])
def test_minimum_press_matches_brute_force(n):
    best = brute_force_minimum(n)
    cells = n * n
    if n == 3:
        boards = range(1 << cells)  # every board, exhaustively
    else:
        rng = random.Random(11)
        boards = [rng.getrandbits(cells) for _ in range(1500)]
    for m in boards:
        b = to_board(n, m)
        sol = solve(n, b)
        if m in best:
            assert sol.solvable and len(sol.presses) == best[m]
            assert is_dark(apply_presses(n, b, sol.presses))
        else:
            assert not sol.solvable


def test_minimum_is_the_lightest_of_all_solutions():
    # The 4 solutions of a 5x5 board are the minimum solution XOR each combination of the two null vectors.
    n = 5
    rng = random.Random(5)
    basis = null_space(n)
    checked = 0
    for _ in range(20):
        b = random_board(n, rng)
        sol = solve(n, b)
        if not sol.solvable:
            continue
        checked += 1
        base = sum(1 << c for c in sol.presses)
        sizes = set()
        for combo in range(4):
            press = base ^ (basis[0] if combo & 1 else 0) ^ (basis[1] if combo & 2 else 0)
            cells = [i for i in range(25) if press >> i & 1]
            assert is_dark(apply_presses(n, b, cells))
            sizes.add(len(cells))
        assert len(sizes) >= 1 and min(sizes) == len(sol.presses) == sol.solution_sizes[0]
        assert len(sol.solution_sizes) == 4
    assert checked > 0


def test_elimination_trace_replays_to_a_solution():
    n = 5
    rng = random.Random(9)
    for _ in range(20):
        b = random_board(n, rng, solvable=rng.random() < 0.7)
        sol = solve(n, b, trace=True)
        width = n * n + 1  # the right-hand side is the last column
        rows = [[int(ch) for ch in text] for text in sol.trace.start]
        for step in sol.trace.steps:
            if step.swap:
                a, c = step.swap
                rows[a], rows[c] = rows[c], rows[a]
            if step.pivot is not None:
                pivot = rows[step.pivot]
                for r in step.cleared:
                    rows[r] = [x ^ y for x, y in zip(rows[r], pivot)]
        assert len(sol.trace.steps) == n * n
        assert sum(1 for s in sol.trace.steps if s.pivot is not None) == sol.rank
        # After the replay, each pivot row reads "pivot variable = right-hand side" with free variables at 0.
        presses = [s.col for s in sol.trace.steps if s.pivot is not None and rows[s.pivot][width - 1]]
        if sol.solvable:
            assert all(rows[r][width - 1] == 0 for r in range(sol.rank, n * n))
            assert is_dark(apply_presses(n, b, presses))
        else:
            assert any(rows[r][width - 1] for r in range(sol.rank, n * n))


def test_parse_cell_accepts_both_forms():
    # 5x5: 'b3' is column b, row 3; '3 2' is row 3, column 2. Both are cell 11 (0-based row 2, column 1).
    assert parse_cell(5, "b3") == 11 == parse_cell(5, "3 2") == parse_cell(5, "3,2") == parse_cell(5, "B3")
    assert label(5, 11) == "b3"
    assert parse_cell(5, "f1") is None  # column off the board
    assert parse_cell(5, "6 1") is None  # row off the board
    assert parse_cell(5, "h") is None and parse_cell(5, "0 1") is None


def test_parse_board_text():
    assert parse_board_text("100/010/001") == (3, (1, 0, 0, 0, 1, 0, 0, 0, 1))
    assert parse_board_text("1#./0.0/..1") == (3, (1, 1, 0, 0, 0, 0, 0, 0, 1))
    with pytest.raises(ValueError):
        parse_board_text("1010")  # 2x2 is below the 3x3 minimum
    with pytest.raises(ValueError):
        parse_board_text("10101")  # not a square
    with pytest.raises(ValueError):
        parse_board_text("10x1")  # bad character


def test_scripted_game_reaches_dark_in_minimum_presses():
    n = 5
    b = random_board(n, random.Random(21))
    sol = solve(n, b)
    script = [label(n, c) for c in sol.presses]
    out = []
    moves = play(n, b, read=lambda _prompt: script.pop(0), out=out.append)
    assert moves == len(sol.presses)
    assert any("Solved in" in line for line in out)


def test_hint_and_quit_in_play():
    n = 5
    b = random_board(n, random.Random(4))
    sol = solve(n, b)
    script = ["h", "bogus", "q"]
    out = []
    moves = play(n, b, read=lambda _prompt: script.pop(0), out=out.append)
    assert moves == 0
    assert any(f"hint: press {label(n, sol.presses[0])}" in line for line in out)
    assert any("not a cell" in line for line in out)


def test_unsolvable_board_cannot_be_played():
    out = []
    assert play(5, unsolvable_board(5), read=lambda _p: "q", out=out.append) == -1
    assert any("cannot be cleared" in line for line in out)


def test_print_solution_reports_unsolvable():
    out = []
    print_solution(5, unsolvable_board(5), out=out.append)
    assert any("cannot be cleared" in line for line in out)


def test_cli_solve_mode(capsys):
    b = random_board(3, random.Random(2))
    assert main(["--solve", "".join(str(v) for v in b)]) == 0
    assert "minimum solution" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        main(["--solve", "10101"])  # not a square board
