"""The terminal checkers game: move parsing, scripted games, and watch mode."""

from checkers.__main__ import main, parse_move, play, render
from checkers.board import RED, START, WHITE, index, legal_moves, notation


def scripted(*lines):
    """A stand-in for input() that returns the given lines in order."""
    it = iter(lines)
    return lambda prompt="": next(it)


def board_with(pieces):
    """Board from {(row, col): piece}."""
    b = [0] * 32
    for (r, c), p in pieces.items():
        b[index(r, c)] = p
    return tuple(b)


def test_parse_move_by_number_and_notation():
    moves = legal_moves(START, RED)
    assert parse_move("1", moves) is moves[0]
    assert parse_move("7", moves) is moves[6]
    assert notation(parse_move("22-18", moves)) == "22-18"


def test_parse_move_round_trips_every_legal_move():
    b = board_with({(6, 1): 1, (5, 2): -1, (3, 4): -1, (0, 7): -1})
    for moves in (legal_moves(START, RED), legal_moves(b, RED)):
        for m in moves:
            assert parse_move(notation(m), moves) == m
    # Captures are written with "x", but the separator does not change which squares are named.
    capture = legal_moves(b, RED)[0]
    assert parse_move(notation(capture).replace("x", "-"), [capture]) == capture


def test_parse_move_rejects_illegal_and_malformed_text():
    moves = legal_moves(START, RED)
    for text in ["0", "8", "99", "22-16", "23x14x5", "abc", "", "22-"]:
        assert parse_move(text, moves) is None


def test_render_shows_numbers_on_empty_dark_squares_and_pieces():
    text = render(START)
    assert " 13 " in text  # square 13 is empty at the start
    assert " r  " in text and " w  " in text
    assert "R" not in text and "W" not in text  # no kings yet


def test_human_quits_immediately():
    assert play({RED: "human", WHITE: "minimax"}, 1, read=scripted("q")) is None


def test_human_move_by_number_then_quit(capsys):
    # 3 is 22-18 in the printed list; minimax answers, then the human quits.
    assert play({RED: "human", WHITE: "minimax"}, 1, read=scripted("3", "q")) is None
    assert "White (minimax) plays" in capsys.readouterr().out


def test_hint_and_bad_input_are_reported(capsys):
    assert play({RED: "human", WHITE: "minimax"}, 1, read=scripted("h", "99", "22-16", "q")) is None
    out = capsys.readouterr().out
    assert "hint:" in out
    assert out.count("not a legal move") == 2


def test_blockade_ends_the_game(capsys):
    # Red's only move captures white's last piece, so white has no legal move and loses.
    b = board_with({(6, 1): 1, (5, 2): -1})
    assert play({RED: "human", WHITE: "minimax"}, 1, read=scripted("1"), board=b) == RED
    assert "White has no legal moves." in capsys.readouterr().out


def test_quiet_plies_end_in_a_draw(monkeypatch):
    # Two kings far apart cannot capture within the limit, so the draw rule ends the game.
    import checkers.__main__ as cli

    monkeypatch.setattr(cli, "QUIET_PLIES", 4)
    kings = board_with({(0, 1): 2, (7, 6): -2})
    assert play({RED: "minimax", WHITE: "minimax"}, 1, board=kings) == 0


def test_watch_endgame_at_level_1_terminates():
    # A full game from the opening takes about 2 s on this laptop, so the watch test uses an
    # endgame: both agents play until one side has no legal move.
    b = board_with({(6, 1): 1, (5, 2): -1})
    assert play({RED: "minimax", WHITE: "alphabeta"}, 1, board=b) == RED


def test_main_quits_from_the_human_prompt(capsys):
    assert main(["--level", "1"], read=scripted("q")) == 0
    assert "You quit." in capsys.readouterr().out
