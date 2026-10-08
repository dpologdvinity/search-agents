import numpy as np

from connect4.board import Board
from connect4.evaluate import match, play_game
from connect4.minimax import Minimax


def first_legal(board: Board) -> int:
    return board.legal_moves()[0]


def test_play_game_reports_the_winner_from_the_first_players_view():
    mm = Minimax(max_depth=4)
    strong = lambda b: mm.search(b).move  # noqa: E731
    rng = np.random.default_rng(0)
    assert play_game(strong, first_legal, rng) == 1
    assert play_game(first_legal, strong, rng) == -1


def test_match_alternates_colours_and_counts_every_game():
    mm = Minimax(max_depth=4)
    counts = match(lambda b: mm.search(b).move, first_legal, 6, np.random.default_rng(1))
    assert sum(counts.values()) == 6
    assert counts["win"] == 6
