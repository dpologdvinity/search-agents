"""The chance opponent for the endgames: the fixed odds, the legal-move guarantee, seeded repeats, and matches.

The odds are checked three ways: the table sums as written, the probabilities over any legal move list sum to
one, and the draws match those probabilities, both on an exact grid of uniform numbers and by sampling.
"""

import random

import pytest

from endgame import chance
from endgame.__main__ import main, play
from endgame.match import MAX_PLIES, play_out, run_match
from endgame.rules import BLACK, WHITE, apply_move, in_check, is_legal
from endgame.tablebase import load


def sample_positions(piece, n, seed):
    """n legal positions with the king pair distinct, drawn from a fixed seed so the tests repeat."""
    tb = load(piece)
    rng = random.Random(seed)
    found = []
    while len(found) < n:
        pos = (rng.randrange(64), rng.randrange(64), rng.randrange(64), rng.randrange(2))
        if len({pos[0], pos[1], pos[2]}) == 3 and is_legal(piece, pos):
            found.append(pos)
    return tb, found


def test_table_is_the_documented_odds():
    assert chance.CHANCE_WEIGHTS == {"capture": 4, "check": 2, "king_centre": 2, "other": 1}
    assert sum(chance.CHANCE_WEIGHTS.values()) == 9
    assert set(chance.CATEGORY_LABELS) == set(chance.CHANCE_WEIGHTS)


@pytest.mark.parametrize("piece", ["Q", "R"])
def test_probabilities_over_the_legal_moves_sum_to_one(piece):
    tb, positions = sample_positions(piece, 200, seed=3)
    for pos in positions:
        infos = tb.move_infos(pos)
        if not infos:
            continue
        probs = chance.probabilities(infos, pos)
        assert sum(probs) == pytest.approx(1.0, abs=1e-12)
        assert all(p > 0 for p in probs)


def test_every_category_occurs_and_capture_outranks_check():
    tb, positions = sample_positions("Q", 400, seed=5)
    seen = set()
    for pos in positions:
        for info in tb.move_infos(pos):
            cat = chance.move_category(info, pos)
            seen.add(cat)
            if info.move[0] == "k" and info.move[2] == pos[1]:
                assert cat == "capture"  # a capture is never reported as a check or a king step
    assert seen == set(chance.CHANCE_WEIGHTS)


def test_check_flag_matches_the_rules():
    tb, positions = sample_positions("R", 200, seed=9)
    for pos in positions:
        for info in tb.move_infos(pos):
            after = apply_move(pos, info.move)
            expected = after is not None and in_check("R", after)
            assert info.check == expected


def test_king_centre_steps_reduce_the_centre_distance():
    assert chance.centre_distance(27) == 1  # d4
    assert chance.centre_distance(36) == 1  # e5
    assert chance.centre_distance(0) == 7  # a1
    tb, positions = sample_positions("Q", 100, seed=12)
    for pos in positions:
        for info in tb.move_infos(pos):
            if chance.move_category(info, pos) == "king_centre":
                mover, src, dst = info.move
                assert mover in ("K", "k")
                assert chance.centre_distance(dst) < chance.centre_distance(src)


@pytest.mark.parametrize("piece", ["Q", "R"])
def test_chance_only_picks_legal_moves(piece):
    tb, positions = sample_positions(piece, 150, seed=21)
    rng = random.Random(1)
    for pos in positions:
        infos = tb.move_infos(pos)
        if not infos:
            assert chance.pick(infos, pos, rng) is None
            continue
        chosen = chance.pick(infos, pos, rng)
        assert chosen in infos


def test_pick_is_deterministic_for_a_seed():
    tb, positions = sample_positions("Q", 20, seed=30)
    for pos in positions:
        infos = tb.move_infos(pos)
        if not infos:
            continue
        first = [chance.pick(infos, pos, random.Random(7)) for _ in range(5)]
        again = [chance.pick(infos, pos, random.Random(7)) for _ in range(5)]
        assert first == again


def test_grid_of_uniform_numbers_reproduces_the_probabilities_exactly():
    # Every u in a fine grid lands in one slice; the slice sizes are the probabilities up to 1/N.
    tb, positions = sample_positions("Q", 12, seed=41)
    grid = 4_000
    checked = 0
    for pos in positions:
        infos = tb.move_infos(pos)
        if len(infos) < 2:
            continue
        counts = {}
        for i in range(grid):
            info = chance.pick_at(infos, pos, (i + 0.5) / grid)
            counts[info.move] = counts.get(info.move, 0) + 1
        for info, p in zip(infos, chance.probabilities(infos, pos)):
            assert counts.get(info.move, 0) / grid == pytest.approx(p, abs=1 / grid + 1e-12)
        checked += 1
    assert checked >= 3


def test_sampled_draws_match_the_odds_within_tolerance():
    # The position with the most categories among its moves: mixed weights make the test meaningful.
    tb, positions = sample_positions("Q", 60, seed=52)
    best = max(((pos, tb.move_infos(pos)) for pos in positions),
               key=lambda item: len({chance.move_category(i, item[0]) for i in item[1]}))
    pos, infos = best
    assert len({chance.move_category(i, pos) for i in infos}) >= 3
    rng = random.Random(2024)
    n = 40_000
    counts = {}
    for _ in range(n):
        info = chance.pick(infos, pos, rng)
        counts[info.move] = counts.get(info.move, 0) + 1
    for info, p in zip(infos, chance.probabilities(infos, pos)):
        assert counts.get(info.move, 0) / n == pytest.approx(p, abs=0.012)


def test_lone_king_takes_the_unguarded_queen_and_the_game_is_drawn():
    tb = load("Q")
    # Black king on g7 next to the queen on h8, which the white king on a1 cannot guard: taking it is a draw,
    # so the tablebase (Black) takes it, and the loop ends at once.
    pos = (0, 63, 54, BLACK)
    assert is_legal("Q", pos)
    assert any(chance.move_category(i, pos) == "capture" for i in tb.move_infos(pos))
    assert play_out(tb, pos, BLACK, random.Random(0)) == ("draw", 1)


@pytest.mark.parametrize("piece", ["Q", "R"])
def test_strong_tablebase_always_wins_against_chance(piece):
    # The table forces mate from every position it calls a win, whatever the defence, so the tablebase side
    # wins every game from a winning start; the chance side cannot change that.
    tb = load(piece)
    for i in range(20):
        pos = tb.random_position(random.Random(i), want="strong_wins")
        result, plies = play_out(tb, pos, WHITE, random.Random(i))
        assert result == "win", (pos, result, plies)
        assert plies <= MAX_PLIES


def test_play_out_is_repeatable_for_a_seed():
    tb = load("R")
    for i in range(10):
        pos = tb.random_position(random.Random(i), want="strong_wins")
        a = play_out(tb, pos, BLACK, random.Random(i))
        b = play_out(tb, pos, BLACK, random.Random(i))
        assert a == b


def test_match_tallies_every_game_and_names_the_algorithms():
    report = run_match("Q", games=6, seed=4)
    assert report["tablebase"] == "Tablebase (exact)"
    assert report["chance"] == "Chance (fixed odds)"
    assert [s["seat"] for s in report["seats"]] == ["strong", "weak"]
    for s in report["seats"]:
        assert s["wins"] + s["draws"] + s["losses"] + s["unfinished"] == 6


def test_match_command_prints_the_result_quickly(capsys):
    assert main(["match", "--games", "4", "--piece", "Q", "--seed", "1"]) == 0
    out = capsys.readouterr().out
    assert "Tablebase (exact) vs Chance (fixed odds)" in out
    assert "tablebase as White (strong seat)" in out
    assert "tablebase as Black (weak seat)" in out


def test_match_command_json_is_parseable(capsys):
    import json

    assert main(["match", "--games", "2", "--piece", "R", "--as", "strong", "--json"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["piece"] == "R" and len(report["seats"]) == 1


def test_interactive_game_names_the_chance_opponent_and_its_odds():
    tb = load("Q")
    lines = []
    pos = tb.random_position(random.Random(3), want="strong_wins")
    # The human is the lone king; with the piece's side to move first the chance agent answers, then we quit.
    replies = iter(["q"])
    play(tb, pos, BLACK, read=lambda _: next(replies), out=lines.append, opponent="chance", seed=3)
    text = "\n".join(lines)
    assert "Agent: Chance (fixed odds)." in text
    assert "captures 4, checks 2, king steps toward the centre 2" in text


def test_interactive_tablebase_game_still_says_tablebase():
    tb = load("Q")
    lines = []
    pos = tb.random_position(random.Random(3), want="strong_wins")
    play(tb, pos, BLACK, read=lambda _: "q", out=lines.append)
    assert "Agent: Tablebase (exact)." in "\n".join(lines)
