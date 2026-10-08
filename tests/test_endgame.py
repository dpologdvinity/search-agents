import random

import numpy as np
import pytest

from endgame.__main__ import judge, parse_move, play, stats_lines, strong_mate_in
from endgame.retro import ILLEGAL, check_positions, index, position, solve
from endgame.rules import (
    BLACK,
    KING_NEIGHBOURS,
    WHITE,
    adjacent,
    apply_move,
    is_legal,
    legal_moves,
    parse_fen,
    parse_square,
    predecessors,
    square_name,
    to_fen,
)
from endgame.tablebase import load


def sample_legal(piece, n, seed):
    """n random legal positions from a fixed seed, so the tests are repeatable."""
    rng = random.Random(seed)
    found = []
    while len(found) < n:
        pos = (rng.randrange(64), rng.randrange(64), rng.randrange(64), rng.randrange(2))
        if len(set(pos[:3])) == 3 and is_legal(piece, pos):
            found.append(pos)
    return found


def test_squares_round_trip():
    assert square_name(0) == "a1" and square_name(63) == "h8"
    assert square_name(parse_square("e4")) == "e4"
    with pytest.raises(ValueError):
        parse_square("i9")


def test_king_neighbours_count_by_region():
    assert len(KING_NEIGHBOURS[0]) == 3  # corner
    assert len(KING_NEIGHBOURS[1]) == 5  # edge
    assert len(KING_NEIGHBOURS[27]) == 8  # centre
    assert adjacent(parse_square("e4"), parse_square("f5"))
    assert not adjacent(parse_square("e4"), parse_square("e6"))


def test_illegal_positions_are_rejected():
    e4, e5, a1, e8, h1 = (parse_square(s) for s in ("e4", "e5", "a1", "e8", "h1"))
    assert not is_legal("Q", (e4, h1, e5, WHITE))  # kings touch
    assert not is_legal("Q", (a1, e8, e4, WHITE))  # White to move while Black is in check
    assert is_legal("Q", (a1, e8, e4, BLACK))  # Black to move in that check: Black must answer it


@pytest.mark.parametrize("piece", ["Q", "R"])
def test_unmoves_are_the_exact_inverse_of_moves(piece):
    for pos in sample_legal(piece, 300, seed=11):
        for move in legal_moves(piece, pos):
            after = apply_move(pos, move)
            if after is not None:
                assert pos in predecessors(piece, after), (pos, move)
        for prev in predecessors(piece, pos):
            assert pos in [apply_move(prev, m) for m in legal_moves(piece, prev)]


def test_fen_round_trip():
    fen = "8/8/8/5k2/8/8/1Q6/K7 w - - 0 1"
    piece, pos = parse_fen(fen)
    assert piece == "Q" and to_fen(piece, pos) == fen


@pytest.mark.parametrize(
    "fen",
    [
        "8/8/8/8/8/8/8/K6k w - - 0 1",  # no piece
        "8/8/8/8/8/8/8/K1Q1k1n1 w - - 0 1",  # a knight
        "8/8/8/8/8/8/8/K1Q1k1R1 w - - 0 1",  # two white pieces
        "8/8/8/8/4kK2/8/8/7Q w - - 0 1",  # kings touch
        "4k3/8/8/8/8/8/8/K3Q3 w - - 0 1",  # White to move, Black in check
        "8/8/8/8/8/8/8/K3Q3 w - - 0 1",  # no black king
    ],
)
def test_bad_fens_are_rejected(fen):
    with pytest.raises(ValueError):
        parse_fen(fen)


def test_black_to_move_in_check_is_a_legal_fen():
    piece, pos = parse_fen("4k3/8/8/8/8/8/8/K3Q3 b - - 0 1")
    assert piece == "Q" and pos[3] == BLACK


def test_known_maxima_and_counts():
    """KQK: longest forced mate is 10 moves; KRK: 16 moves. Both are the known values for these endgames."""
    sq, sr = load("Q").summary(), load("R").summary()
    assert sq["max_win_moves"] == 10 and sq["max_win_plies"] == 19
    assert sr["max_win_moves"] == 16 and sr["max_win_plies"] == 31
    assert sq["max_loss_moves"] == 10 and sr["max_loss_moves"] == 16
    for s in (sq, sr):
        assert s["win"] + s["loss"] + s["draw"] == s["legal"]
    # With White to move, the side with the piece wins every legal position in both endgames.
    assert sq["legal_white_to_move"] == sq["win_white_to_move"] == 144508
    assert sr["draw_white_to_move"] == 0 and sr["loss_white_to_move"] == 0
    # The totals are pinned: any change to the rules or the solver shows up here.
    assert sq["legal"] == 368452 and sr["legal"] == 399112


@pytest.mark.parametrize("piece", ["Q", "R"])
def test_table_matches_the_definition_on_a_sample(piece):
    tb = load(piece)
    assert check_positions(piece, tb.codes, sample_legal(piece, 1500, seed=3)) == []


def test_checker_catches_a_wrong_entry():
    codes = load("Q").codes.copy()
    victim = next(p for p in sample_legal("Q", 500, 9) if codes[index(p)] < 65000)
    codes[index(victim)] = int(codes[index(victim)]) + 2  # same parity, wrong distance
    assert check_positions("Q", codes, [victim]) != []


def test_solver_reproduces_the_stored_table():
    codes, _, legal = solve("R")
    assert legal == 399112
    assert (codes == load("R").codes).all()
    assert int((codes == ILLEGAL).sum()) == 2 * 64**3 - 399112


def test_position_index_round_trip():
    for pos in sample_legal("Q", 200, seed=4):
        assert position(index(pos)) == pos


@pytest.mark.parametrize("piece", ["Q", "R"])
def test_optimal_play_mates_in_exactly_the_stored_distance(piece):
    """Both sides play the table's best move from a winning position. The mate must come exactly when promised."""
    tb = load(piece)
    pos = tb.random_position(random.Random(2), want="strong_wins")
    promised = tb.result(pos)[1]
    plies = 0
    while tb.move_infos(pos):
        pos = apply_move(pos, tb.best_move(pos).move)
        assert pos is not None, "optimal play never takes the piece from a won position"
        plies += 1
    assert plies == promised


def test_longest_mates_are_played_out():
    for piece, expect in (("Q", 19), ("R", 31)):
        tb = load(piece)
        pos = parse_fen(tb.summary()["longest_win_example"])[1]
        assert tb.result(pos) == ("win", expect)
        plies = 0
        while tb.move_infos(pos):
            pos = apply_move(pos, tb.best_move(pos).move)
            plies += 1
        assert plies == expect


def test_mate_in_one_is_recognised():
    tb = load("Q")
    i = int(np.flatnonzero(tb.codes == 1)[0])  # a White-to-move position with a mate in one
    pos = position(i)
    best = tb.best_move(pos)
    assert best.mate and best.plies == 1 and best.outcome == "win" and best.san.endswith("#")


def test_capture_is_a_draw_and_stalemate_is_drawn():
    tb = load("Q")
    # Black king on c3 takes the undefended queen on b2: king against king.
    pos = parse_fen("7K/8/8/8/8/2k5/1Q6/8 b - - 0 1")[1]
    takes = [m for m in tb.move_infos(pos) if m.uci == "c3b2"]
    assert takes and takes[0].outcome == "draw" and takes[0].plies is None
    # Black king on a8 with no legal move and no check: stalemate.
    assert tb.result(parse_fen("k7/2Q5/1K6/8/8/8/8/8 b - - 0 1")[1]) == ("draw", None)


def test_judge_grades_moves():
    tb = load("Q")
    pos = parse_fen("8/8/8/5k2/8/8/1Q6/K7 w - - 0 1")[1]
    infos = tb.move_infos(pos)
    best = tb.best_move(pos)
    drawing = next(m for m in infos if m.outcome == "draw")
    assert judge(best, best, strong=True) == "best move."
    assert judge(drawing, best, strong=True).startswith("blunder")
    assert judge(drawing, best, strong=False) is None
    assert strong_mate_in(tb, pos) == 10


def test_parse_move_accepts_several_notations():
    tb = load("Q")
    infos = tb.move_infos(parse_fen("8/8/8/5k2/8/8/1Q6/K7 w - - 0 1")[1])
    assert parse_move("Kb1", infos).uci == "a1b1"
    assert parse_move("a1b1", infos).uci == "a1b1"
    assert parse_move("Qc3", infos).uci == "b2c3"
    assert parse_move("b2c3", infos).uci == "b2c3"
    with pytest.raises(ValueError, match="not a legal move"):
        parse_move("Qh1", infos)
    with pytest.raises(ValueError, match="not a move"):
        parse_move("zz", infos)


def test_scripted_strong_game_grades_the_blunder():
    tb = load("Q")
    pos = parse_fen("8/8/8/5k2/8/8/1Q6/K7 w - - 0 1")[1]
    script = ["Qe5"]
    lines = []

    def read(prompt):
        return script.pop(0) if script else "q"

    result = play(tb, pos, WHITE, read=read, out=lines.append)
    text = "\n".join(lines)
    assert result == "draw"  # the queen is offered to the king, and the agent takes it
    assert "blunder" in text and "Kxe5" in text


def test_hint_lists_every_move_and_marks_the_best():
    tb = load("R")
    pos = parse_fen("8/8/8/8/8/2k5/1R6/K7 w - - 0 1")[1]  # the longest KRK mate: White to move, mate in 16
    script = ["h", "q"]
    lines = []
    result = play(tb, pos, WHITE, read=lambda prompt: script.pop(0), out=lines.append)
    text = "\n".join(lines)
    assert result == "quit" and "<- best" in text and "mate in 16" in text


def test_stats_text_mentions_both_extremes():
    text = "\n".join(stats_lines(load("R")))
    assert "KRK" in text and "16 moves (31 plies)" in text


def test_random_position_is_legal_and_wanted():
    tb = load("R")
    rng = random.Random(7)
    for _ in range(20):
        pos = tb.random_position(rng, want="strong_wins")
        assert is_legal("R", pos) and strong_mate_in(tb, pos) is not None
    with pytest.raises(ValueError):
        tb.random_position(rng, stm=WHITE, want="draw", max_tries=2000)


def test_heatmap_row_matches_positions():
    tb = load("Q")
    wk, wp = parse_square("a1"), parse_square("b2")
    row = tb.king_row(wk, wp, BLACK)
    for bk in (parse_square("f5"), parse_square("h8")):
        assert row[bk] == tb.value((wk, wp, bk, BLACK))
    assert row[wk] == ILLEGAL and row[wp] == ILLEGAL
