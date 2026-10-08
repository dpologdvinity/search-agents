import random

import pytest

from connect4.board import COLS, ROWS, Board


def naive_winner(grid):
    """Reference win check on a top-to-bottom grid; returns 1, -1, or 0."""
    for r in range(ROWS):
        for c in range(COLS):
            v = grid[r][c]
            if not v:
                continue
            for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
                if all(0 <= r + k * dr < ROWS and 0 <= c + k * dc < COLS
                       and grid[r + k * dr][c + k * dc] == v for k in range(4)):
                    return v
    return 0


@pytest.mark.parametrize("moves", [
    "1212121",       # vertical
    "1122334",       # horizontal
    "12233434744",   # diagonal rising to the right: X at a1 b2 c3 d4
    "76655454144",   # the same, mirrored: rising to the left
])
def test_wins_in_each_direction(moves):
    before = Board.from_moves(moves[:-1])
    assert not before.last_player_won()
    assert before.is_winning_move(int(moves[-1]) - 1)
    assert Board.from_moves(moves).last_player_won()


@pytest.mark.parametrize("seed", range(300))
def test_matches_naive_rules_on_random_games(seed):
    rng = random.Random(seed)
    board = Board()
    while True:
        moves = board.legal_moves()
        if not moves:
            assert board.is_full()
            break
        col = rng.choice(moves)
        wins = board.is_winning_move(col)
        board = board.play(col)
        # After the move the mover is the opponent (-1) in grid terms.
        assert board.last_player_won() == wins == (naive_winner(board.grid()) == -1)
        if wins:
            break


def test_full_columns_and_mirror():
    board = Board.from_moves("111111")
    assert not board.can_play(0)
    assert 0 not in board.legal_moves()
    with pytest.raises(ValueError):
        Board.from_moves("1111111")
    b = Board.from_moves("1234567112")
    m = b.mirror()
    assert m.grid() == [list(reversed(row)) for row in b.grid()]
    assert m.mirror() == b


def test_key_is_unique_per_position():
    rng = random.Random(0)
    seen = {}
    for _ in range(2000):
        board = Board()
        for _ in range(rng.randint(0, 12)):
            moves = [c for c in board.legal_moves() if not board.is_winning_move(c)]
            if not moves:
                break
            board = board.play(rng.choice(moves))
        g = tuple(map(tuple, board.grid()))
        assert seen.setdefault(board.key(), (g, board.moves)) == (g, board.moves)
