import os
import subprocess
import sys
from functools import cache
from pathlib import Path

import pytest

from tictactoe import legal_moves, minimax, reachable_positions, search, winner
from tictactoe.core import EMPTY_BOARD, LEVELS

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


def human_can_force_win(depth, human):
    """True if a human playing `human` can force a win against the agent searching `depth` plies."""

    @cache
    def win(board):
        w = winner(board)
        if w:
            return w == human
        if "." not in board:
            return False
        turn = turn_of(board)
        if turn != human:
            m = search(board, turn, depth).move
            return win(board[:m] + turn + board[m + 1 :])
        return any(win(board[:m] + human + board[m + 1 :]) for m in legal_moves(board))

    return win(EMPTY_BOARD)


def test_levels_are_beatable_except_perfect():
    # EASY and MEDIUM can be beaten whichever side the human takes; PERFECT never.
    for human in "XO":
        assert human_can_force_win(LEVELS["easy"], human)
        assert human_can_force_win(LEVELS["medium"], human)
        assert not human_can_force_win(LEVELS["perfect"], human)
    # The boundary the LEVELS comment states: depth 5 loses only to X, depth 6 never loses.
    assert human_can_force_win(5, "X") and not human_can_force_win(5, "O")
    assert not human_can_force_win(6, "X") and not human_can_force_win(6, "O")


def test_depth_below_one_is_rejected():
    # Depth 0 would score the root itself and return no move for a game still in play.
    with pytest.raises(ValueError):
        search(EMPTY_BOARD, "X", 0)
    assert cli("positions", "0").returncode != 0


def test_depth_limit_scores_the_horizon_as_a_draw():
    # X threatens 6-7-8. One ply sees its own wins but not the reply, so it takes the first empty square;
    # two plies see the threat and block.
    assert search("O.....XX.", "O", 1).move == 1
    assert search("O.....XX.", "O", 2).move == 8
    # A cap deeper than the game changes nothing.
    assert search(EMPTY_BOARD, "X", 9) == search(EMPTY_BOARD, "X")
    # A shallow search visits far fewer positions.
    assert search(EMPTY_BOARD, "X", 2).searched < search(EMPTY_BOARD, "X").searched


def test_positions_cli_takes_a_depth():
    import json

    out = json.loads(cli("positions", "2").stdout)
    p = next(p for p in out if p["board"] == "O.....XX.")
    assert p["move"] == 8
