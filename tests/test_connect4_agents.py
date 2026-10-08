import random

import pytest

from connect4.board import Board
from connect4.mcts import MCTS
from connect4.minimax import Minimax, evaluate

# X to move, with three stacked in column 1: playing column 1 wins.
WIN_NOW = "121212"
# X to move; O has three stacked in column 2 and X has no threat: X must block.
BLOCK = "323272"


def move_of(result):
    return result["move"] if isinstance(result, dict) else result.move


def play_out(first, second, rng, opening=2):
    """Play one game after `opening` random moves; returns 1 if `first` wins, -1 if `second`, 0 draw."""
    board = Board()
    for _ in range(opening):
        board = board.play(rng.choice(board.legal_moves()))
    agents = (first, second) if board.moves % 2 == 0 else (second, first)
    turn = 0
    while board.legal_moves():
        col = agents[turn % 2](board)
        if board.is_winning_move(col):
            winner = agents[turn % 2]
            return 1 if winner is first else -1
        board = board.play(col)
        turn += 1
    return 0


def random_agent(rng):
    return lambda board: rng.choice(board.legal_moves())


@pytest.mark.parametrize("make", [lambda: Minimax(max_depth=4), lambda: MCTS(simulations=300, seed=0)])
def test_takes_immediate_win(make):
    board = Board.from_moves(WIN_NOW)
    assert board.is_winning_move(0)
    assert move_of(make().search(board)) == 0


@pytest.mark.parametrize("make", [lambda: Minimax(max_depth=4), lambda: MCTS(simulations=2000, seed=0)])
def test_blocks_immediate_threat(make):
    board = Board.from_moves(BLOCK)
    assert not any(board.is_winning_move(c) for c in board.legal_moves())
    assert board.play(0).is_winning_move(1)  # if X ignores it, O wins in column 2
    assert move_of(make().search(board)) == 1


def test_minimax_finds_forced_win():
    # X to move can play column 4 to make two open threats on the bottom row.
    board = Board.from_moves("3355")
    info = Minimax(max_depth=6).search(board)
    assert info.score >= 1_000_000 - 64
    assert info.move in (1, 3, 5)


def test_evaluate_is_antisymmetric():
    rng = random.Random(0)
    board = Board()
    for _ in range(10):
        board = board.play(rng.choice(board.legal_moves()))
    flipped = Board(board.current ^ board.mask, board.mask, board.moves)
    assert evaluate(board) == -evaluate(flipped)


def test_search_agents_beat_random():
    rng = random.Random(1)
    mm = Minimax(max_depth=3)
    mc = MCTS(simulations=400, seed=1)
    minimax = lambda b: mm.search(b).move
    mcts = lambda b: mc.search(b)["move"]
    for agent in (minimax, mcts):
        results = [play_out(agent, random_agent(rng), rng) for _ in range(10)]
        assert results.count(1) >= 9, results


def test_time_budget_is_respected():
    info = Minimax(max_depth=42, max_seconds=0.3).search(Board())
    assert info.seconds < 0.6
    assert info.depth >= 3
