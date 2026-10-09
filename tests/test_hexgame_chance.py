"""The Hex chance opponent: fixed weights by hex ring from the centre, with no search.

Covers the ring geometry, the odds table, legality (never an occupied cell, never the swap), seeded
determinism, the empirical frequencies against the table, and the command-line match.
"""

import random

import numpy as np
import pytest

from hexgame.__main__ import main
from hexgame.agents import (
    AGENTS,
    CHANCE_RING_WEIGHTS,
    chance_odds,
    choose,
    describe_move,
    hex_ring,
    player,
)
from hexgame.board import EMPTY, SWAP, Board, neighbours


def test_centre_ring_is_the_centre_and_its_neighbours_are_ring_one():
    n = 7
    centre = 3 * n + 3
    assert hex_ring(n, centre) == 0
    assert {hex_ring(n, c) for c in neighbours(n)[centre]} == {1}
    assert len(neighbours(n)[centre]) == 6


def test_ring_distance_follows_the_six_neighbour_grid():
    n = 7
    # Corner (0,0) has dr = dc = -3, so the same-sign offsets need |dr| + |dc| = 6 steps (no diagonal step).
    assert hex_ring(n, 0) == 6
    # Corner (0,6) has dr = -3 and dc = +3, opposite signs, so three (-1,+1) steps reach it: distance 3.
    assert hex_ring(n, 6) == 3


def test_odds_sum_to_one_and_occupied_cells_get_nothing():
    cells = [EMPTY] * 49
    cells[24] = 1  # the centre is taken
    odds = chance_odds(7, cells)
    assert odds[24] == 0
    assert sum(odds) == pytest.approx(1.0)
    # The centre ring's weight is gone, so the next-biggest cells keep their share of the total.
    assert odds[23] == pytest.approx(CHANCE_RING_WEIGHTS[1] / (76 - 4))


def test_chance_only_plays_empty_cells_and_never_swaps():
    rng = random.Random(3)
    board = Board(7, swap_rule=True).play(24)  # the swap rule is live on the second player's first turn
    cells = list(board.cells)
    for _ in range(500):
        move = player("chance")(7, cells, board.to_move, rng)
        assert cells[move] == EMPTY
        assert move != SWAP
    move, analysis = choose(board, "chance", seed=5)
    assert move != SWAP and cells[move] == EMPTY
    assert analysis["swap"] is None


def test_same_seed_gives_the_same_moves():
    board = Board(7).play(0)
    assert choose(board, "chance", seed=9)[0] == choose(board, "chance", seed=9)[0]
    a, b = random.Random(4), random.Random(4)
    cells = list(board.cells)
    assert [player("chance")(7, cells, board.to_move, a) for _ in range(30)] == \
        [player("chance")(7, cells, board.to_move, b) for _ in range(30)]


def test_empirical_frequencies_match_the_table_on_an_empty_board():
    rng = random.Random(1)
    cells = [EMPTY] * 49
    counts = np.zeros(49)
    samples = 20000
    play = player("chance")
    for _ in range(samples):
        counts[play(7, cells, 1, rng)] += 1
    freq = counts / samples
    assert np.abs(freq - np.array(chance_odds(7, cells))).max() < 0.01


def test_describe_move_names_the_odds():
    board = Board(7)
    move, analysis = choose(board, "chance", seed=2)
    text = describe_move(7, move, analysis)
    assert "fixed odds" in text and "visits" not in text


def test_registry_describes_the_ring_weights():
    assert "4/3/2/1" in AGENTS["chance"]


def test_cli_match_prints_a_table_with_both_names(capsys):
    assert main(["match", "--agent", "uct", "--level", "1", "--games", "2"]) == 0
    out = capsys.readouterr().out
    assert "Plain UCT" in out and "Chance (fixed odds)" in out
    assert "| 0 |" in out  # Hex has no draws
