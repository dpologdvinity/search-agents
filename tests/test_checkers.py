import random

import pytest

from checkers.board import RED, START, WHITE, index, legal_moves, notation, perft
from checkers.search import WIN, AlphaBeta, Minimax, evaluate, from_table, to_table


def board_with(pieces):
    """Board from {(row, col): piece}."""
    b = [0] * 32
    for (r, c), p in pieces.items():
        b[index(r, c)] = p
    return tuple(b)


def test_perft_matches_published_counts():
    assert [perft(START, RED, d) for d in range(1, 6)] == [7, 49, 302, 1469, 7361]


def test_captures_are_mandatory_and_chain():
    # Red man at (6,1) can double-jump white men at (5,2) and (3,4).
    b = board_with({(6, 1): 1, (5, 2): -1, (3, 4): -1, (0, 7): -1})
    moves = legal_moves(b, RED)
    assert len(moves) == 1
    m = moves[0]
    assert m.path == (index(6, 1), index(4, 3), index(2, 5))
    assert sorted(m.captured) == sorted([index(5, 2), index(3, 4)])
    assert m.result[index(5, 2)] == 0 and m.result[index(3, 4)] == 0
    assert notation(m).count("x") == 2


def test_crowning_ends_the_move():
    # Red man jumps onto the top row and is crowned; it may not keep jumping as a king.
    b = board_with({(2, 3): 1, (1, 2): -1, (1, 0): -1, (7, 0): -1})
    moves = legal_moves(b, RED)
    assert any(m.result[index(0, 1)] == 2 for m in moves)
    assert all(len(m.captured) == 1 for m in moves)


def test_kings_move_backward():
    b = board_with({(3, 2): 2, (0, 7): -1})
    targets = {m.path[-1] for m in legal_moves(b, RED)}
    assert targets == {index(2, 1), index(2, 3), index(4, 1), index(4, 3)}


def test_no_moves_is_a_loss():
    b = board_with({(7, 0): 1, (0, 7): -1, (6, 1): -1, (5, 2): -1})  # red man blocked in the corner
    assert legal_moves(b, RED) == []
    with pytest.raises(ValueError):
        AlphaBeta().search(b, RED)


def test_evaluation_is_symmetric():
    assert evaluate(START, RED) == 0 == evaluate(START, WHITE)


@pytest.mark.parametrize("seed", range(4))
def test_alpha_beta_matches_minimax(seed):
    rng = random.Random(seed)
    b, side = START, RED
    for _ in range(6):
        b = rng.choice(legal_moves(b, side)).result
        side = -side
    mm = Minimax(depth=4).search(b, side)
    ab = AlphaBeta(max_depth=4, max_seconds=60).search(b, side)
    assert ab.score == mm.score
    assert ab.nodes < mm.nodes


def test_finds_a_winning_capture():
    # White's last piece is capturable: red should take it and win.
    b = board_with({(5, 2): 1, (4, 3): -1})
    info = AlphaBeta(max_seconds=5).search(b, RED)
    assert info.move.captured == (index(4, 3),)
    assert info.score > WIN - 100


def test_table_stores_wins_and_losses_by_distance_from_the_node():
    """A loss 3 plies below a node stored at ply 2 reads back as the same loss distance at ply 5."""
    loss_at_ply_5 = -WIN + 5
    stored = to_table(loss_at_ply_5, 2)
    assert stored == -WIN + 3
    assert from_table(stored, 5) == -WIN + 8
    assert from_table(to_table(123, 4), 9) == 123  # ordinary evaluations pass through unchanged


def _random_endgame(seed):
    """Play seeded random moves from the start until a sparse position, as in the endgame agreement test."""
    rng = random.Random(seed)
    b, side = START, RED
    for _ in range(rng.randint(30, 70)):
        b = rng.choice(legal_moves(b, side)).result
        side = -side
    return b, side


# Endgames where depth-5 minimax finds a forced win or loss and the side to move has a choice of moves.
MATE_SEEDS = (71, 111, 156, 223, 307, 343, 380)


@pytest.mark.parametrize("seed", MATE_SEEDS)
def test_alpha_beta_matches_minimax_on_forced_wins_and_losses(seed):
    """Win and loss scores (-WIN + ply) must survive the transposition table: alpha-beta agrees with minimax."""
    b, side = _random_endgame(seed)
    mm = Minimax(depth=5).search(b, side)
    ab = AlphaBeta(max_depth=5, max_seconds=60).search(b, side)
    assert abs(mm.score) > WIN - 1000
    assert ab.score == mm.score
