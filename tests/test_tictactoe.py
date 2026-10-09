import os
import subprocess
import sys
from functools import cache
from pathlib import Path

from tictactoe import legal_moves, minimax, reachable_positions, search, winner
from tictactoe.core import EMPTY_BOARD

ROOT = Path(__file__).resolve().parent.parent


def other(turn):
    return "O" if turn == "X" else "X"


@cache
def memo_minimax(board, turn):
    """Plain minimax with a cache: the reference value for every reachable position, computed once."""
    w = winner(board)
    if w:
        return 1 if w == "X" else -1
    moves = legal_moves(board)
    if not moves:
        return 0
    vals = [memo_minimax(board[:m] + turn + board[m + 1 :], other(turn)) for m in moves]
    return max(vals) if turn == "X" else min(vals)


def turn_of(board):
    return "X" if board.count("X") == board.count("O") else "O"


def cli(*args, stdin=""):
    env = {**os.environ, "PYTHONPATH": str(ROOT)}
    return subprocess.run(
        [sys.executable, "-m", "tictactoe", *args],
        input=stdin,
        capture_output=True,
        text=True,
        cwd=ROOT,
        env=env,
        timeout=120,
    )


def test_winner_lines():
    assert winner("XXX......") == "X"
    assert winner("O..O..O..") == "O"
    assert winner("X...X...X") == "X"
    assert winner("..O.O.O..") == "O"
    assert winner("XO.XO.XO.") == "X"  # the first column, 0-3-6
    assert winner("XOXXOOOXX") == ""
    assert winner(EMPTY_BOARD) == ""


def test_reachable_positions_count():
    # 5,478 distinct boards are reachable; 4,520 of them are still in play (no win, not full).
    assert len(reachable_positions()) == 4520


def test_empty_board_is_a_draw_under_perfect_play():
    assert minimax(EMPTY_BOARD, "X") == 0
    assert search(EMPTY_BOARD, "X").value == 0


def test_alpha_beta_agrees_with_minimax_on_every_position():
    for board in reachable_positions():
        turn = turn_of(board)
        r = search(board, turn)
        assert r.value == memo_minimax(board, turn), board
        assert r.move in legal_moves(board)
        # The chosen move achieves the minimax value.
        after = board[: r.move] + turn + board[r.move + 1 :]
        assert memo_minimax(after, other(turn)) == r.value, board


def test_agent_finds_the_block_when_one_exists():
    # X threatens the top row (0, 1, 2); O must block at square 2 (index 2), and it does.
    r = search("XX.O.....", "O")
    assert r.move == 2


def test_search_prunes_and_records_the_first_plies():
    r = search(EMPTY_BOARD, "X")
    assert r.pruned > 0
    assert r.searched < 549946  # far below the full game tree
    # Every legal root move is recorded, in square order, with its own replies below it.
    assert [rec["m"] for rec in r.tree] == list(range(9))
    assert all(rec["kids"] for rec in r.tree)
    assert all(rec["pruned"] or rec["v"] is not None for rec in r.tree[0]["kids"])
    # Counters are running totals, so they never decrease as the tree is drawn.
    seq = [rec["s"] for rec in r.tree]
    assert seq == sorted(seq)


def test_recorded_values_are_exact_or_bounds():
    r = search("X...O....", "X")
    exact = [rec for rec in r.tree if rec["b"] == ""]
    assert exact, "the chosen move must be exact"
    best = max(rec["v"] for rec in exact)
    assert best == r.value


def test_game_over_positions_have_no_move():
    r = search("XXXOO....", "O")
    assert r.move is None and r.value == 1 and r.tree == []


def test_stats_command_reports_positions_and_draw():
    out = cli("stats")
    assert out.returncode == 0, out.stderr
    assert "positions:       5,478 distinct boards reachable (4,520 still in play)" in out.stdout
    assert "game-tree nodes: 549,946" in out.stdout
    assert "perfect play: draw" in out.stdout


def test_play_command_answers_a_move():
    out = cli("play", stdin="5\n")
    assert out.returncode == 0, out.stderr
    assert "agent plays" in out.stdout


def test_play_command_rejects_taken_squares():
    out = cli("play", stdin="5\n5\n")
    assert "taken or off the board" in out.stdout


def test_unknown_command_fails():
    assert cli("nope").returncode != 0


def test_help_prints_usage(capsys):
    from tictactoe.__main__ import main

    main(["tictactoe", "--help"])
    assert "python -m tictactoe stats" in capsys.readouterr().out
