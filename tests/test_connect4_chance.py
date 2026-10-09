"""The chance opponent: a fixed-odds column picker with no search, used as a baseline.

Covers the weight table, legality around full columns, seeded determinism, the empirical
frequencies against the table, and the command-line match.
"""

import random

import numpy as np
import pytest

from connect4.__main__ import main
from connect4.agents import AGENTS, CHANCE_WEIGHTS, DESCRIPTIONS, chance_move, chance_odds
from connect4.board import COLS, Board


def test_table_sums_to_sixteen_and_open_board_odds_match_it():
    assert sum(CHANCE_WEIGHTS) == 16
    odds = chance_odds(Board())
    assert sum(odds) == pytest.approx(1.0)
    assert odds == pytest.approx([w / 16 for w in CHANCE_WEIGHTS])
    # The centre column is the most likely, and the table is symmetric about it.
    assert odds[3] == max(odds) and odds == pytest.approx(odds[::-1])


def test_full_column_gets_no_odds_and_the_rest_renormalise():
    board = Board()
    for _ in range(6):  # alternate-colour stones in column 4 fill it without a four-in-a-row
        board = board.play(3)
    assert not board.can_play(3)
    odds = chance_odds(board)
    assert odds[3] == 0
    assert sum(odds) == pytest.approx(1.0)
    # With column 4 gone, column 3 keeps its 3 of the remaining 12 weight.
    assert odds[2] == pytest.approx(3 / 12)


def test_chance_only_plays_legal_moves():
    rng = random.Random(0)
    board = Board()
    # Fill columns 1 and 7 so the legal set is smaller; chance must never pick a full column.
    for _ in range(6):
        board = board.play(0)
    for _ in range(6):
        board = board.play(6)
    legal = set(board.legal_moves())
    assert 0 not in legal and 6 not in legal
    for _ in range(2000):
        assert chance_move(board, 1, rng)["move"] in legal


def test_same_seed_gives_the_same_moves():
    board = Board().play(3).play(2)
    assert chance_move(board, 1, random.Random(7)) == chance_move(board, 1, random.Random(7))
    a, b = random.Random(7), random.Random(7)
    first = [chance_move(board, 1, a)["move"] for _ in range(50)]
    assert first == [chance_move(board, 1, b)["move"] for _ in range(50)]


def test_empirical_frequencies_match_the_table_at_the_start():
    rng = random.Random(1)
    counts = np.zeros(COLS)
    samples = 20000
    for _ in range(samples):
        counts[chance_move(Board(), 1, rng)["move"]] += 1
    freq = counts / samples
    assert np.abs(freq - np.array(CHANCE_WEIGHTS) / 16).max() < 0.01


def test_analysis_reports_the_odds_and_no_search_is_run():
    result = chance_move(Board(), 1, random.Random(0))
    assert result["analysis"]["kind"] == "chance"
    assert result["analysis"]["odds"] == pytest.approx([round(w / 16, 4) for w in CHANCE_WEIGHTS])


def test_registry_and_description_name_the_table():
    assert AGENTS["chance"] is chance_move
    assert "6%/12%/19%/25%/19%/12%/6%" in DESCRIPTIONS["chance"]


def test_cli_match_prints_the_record_and_both_algorithms(capsys):
    assert main(["--match", "2", "--agent", "alphazero", "--level", "1"]) == 0
    out = capsys.readouterr().out
    assert "AlphaZero (alphazero)" in out and "Chance (chance, fixed odds)" in out
    # Two games: the two rows' wins and losses mirror each other, and the draws column is counted.
    rows = [line for line in out.splitlines() if line.startswith("| ")][1:]
    counts = [[int(x) for x in row.split("|")[2:5]] for row in rows]
    assert sum(counts[0]) == 2 and counts[0][0] == counts[1][2] and counts[0][2] == counts[1][0]


def test_cli_match_refuses_chance_against_itself():
    with pytest.raises(SystemExit) as e:
        main(["--match", "2", "--agent", "chance"])
    assert e.value.code == 2
