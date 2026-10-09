"""Tests for the fixed-odds chance opponent: its table, its legal moves, its seeding, its odds, and the CLI.

The API side of chance (deal option, hidden card, meta) is in test_server_poker_chance.py.
"""

import random
import time

import pytest

from poker import strategy
from poker.betting import BET, CALL, CHECK, FOLD, RAISE
from poker.bots import OPPONENTS, CFRBot, deal, head_to_head, make_opponent, play_hand
from poker.chance import ACTION_ODDS, KIND, ChanceBot, legal_weights
from poker.leduc import LeducState


def test_the_fixed_table_sums_to_one_and_names_three_kinds():
    assert set(ACTION_ODDS) == {"fold", "call_or_check", "bet_or_raise"}
    assert sum(ACTION_ODDS.values()) == pytest.approx(1.0)
    assert ACTION_ODDS == {"fold": 0.15, "call_or_check": 0.55, "bet_or_raise": 0.30}


@pytest.mark.parametrize("legal", [(CHECK, BET), (FOLD, CALL, RAISE), (FOLD, CALL)])
def test_renormalised_weights_sum_to_one_over_every_legal_set(legal):
    weights = legal_weights(legal)
    assert list(weights) == list(legal)
    assert sum(weights.values()) == pytest.approx(1.0)
    # Each action keeps its kind's odds relative to the others in the set.
    raw = {a: ACTION_ODDS[KIND[a]] for a in legal}
    total = sum(raw.values())
    for a in legal:
        assert weights[a] == pytest.approx(raw[a] / total)


def test_facing_a_bet_the_weights_are_the_table_itself():
    weights = legal_weights((FOLD, CALL, RAISE))
    assert weights == pytest.approx({"f": 0.15, "c": 0.55, "r": 0.30})


def test_chance_only_plays_legal_actions():
    rng = random.Random(12)
    bot = ChanceBot()
    for _ in range(500):
        s = deal(rng)
        while not s.is_terminal():
            action = bot.act(s, rng)
            assert action in s.actions()
            s = s.apply(action)


def test_chance_is_deterministic_given_the_seed():
    def run(seed):
        rng = random.Random(seed)
        bot = ChanceBot()
        out = []
        for _ in range(50):
            s = deal(rng)
            while not s.is_terminal():
                a = bot.act(s, rng)
                out.append(a)
                s = s.apply(a)
        return out

    assert run(7) == run(7)
    assert run(7) != run(8)


def test_chance_ignores_its_card():
    """Same seed and same betting spot, but chance holds a queen in one deal and a king in the other."""
    # dealt(seat0, seat1, public): chance is seat 1 after seat 0 bets, so only seat 1's card changes.
    holds_queen = LeducState.dealt(0, 2, 4).apply("b")
    holds_king = LeducState.dealt(0, 4, 2).apply("b")
    assert holds_queen.player() == holds_king.player() == 1
    for seed in range(40):
        a = ChanceBot().act(holds_queen, random.Random(seed))
        b = ChanceBot().act(holds_king, random.Random(seed))
        assert a == b


def test_chance_draws_match_the_table_over_many_samples():
    rng = random.Random(3)
    bot = ChanceBot()
    facing = LeducState.dealt(0, 2, 4).apply("b")  # seat 1 faces a bet: fold, call, or raise
    n = 20_000
    counts = {FOLD: 0, CALL: 0, RAISE: 0}
    for _ in range(n):
        counts[bot.act(facing, rng)] += 1
    assert abs(counts[FOLD] / n - 0.15) < 0.01
    assert abs(counts[CALL] / n - 0.55) < 0.01
    assert abs(counts[RAISE] / n - 0.30) < 0.01

    opening = LeducState.dealt(0, 2, 4)  # nothing owed: check or bet, renormalised to 0.55 and 0.30
    counts = {CHECK: 0, BET: 0}
    for _ in range(n):
        counts[bot.act(opening, rng)] += 1
    assert abs(counts[CHECK] / n - 0.55 / 0.85) < 0.01
    assert abs(counts[BET] / n - 0.30 / 0.85) < 0.01


def test_make_opponent_maps_names_and_refuses_unknown_ones():
    table = strategy.load()["strategy"]
    assert OPPONENTS == ("cfr", "chance")
    assert isinstance(make_opponent("cfr", table), CFRBot)
    assert isinstance(make_opponent("chance", table), ChanceBot)
    with pytest.raises(ValueError):
        make_opponent("minimax", table)


def test_head_to_head_chance_is_seeded_and_the_cfr_bot_beats_it():
    table = strategy.load()["strategy"]
    a = head_to_head(CFRBot(table), ChanceBot(), 2000, seed=5)
    b = head_to_head(CFRBot(table), ChanceBot(), 2000, seed=5)
    assert a.mbb_per_hand == b.mbb_per_hand
    assert a.mbb_per_hand > 4 * a.mbb_se  # the equilibrium mix beats fixed odds by more than noise


def test_play_hand_accepts_chance_as_the_bot():
    rng = random.Random(4)
    state = LeducState.dealt(0, 2, 4)
    payoff = play_hand(state, (CFRBot(strategy.load()["strategy"]), ChanceBot()), rng)
    assert isinstance(payoff, float) and abs(payoff) <= 13  # the largest pot in Leduc is 13 chips for one seat


def test_cli_session_plays_against_chance(capsys):
    from poker.__main__ import session

    table = strategy.load()["strategy"]
    replies = iter(["k", "k", "k", "k", "q"])
    session(seed=3, table=table, read=lambda _p: next(replies, "q"), out=print, hands=None, opponent="chance")
    out = capsys.readouterr().out
    assert "Chance (fixed odds" in out and "Hand 1" in out and "bye" in out


def test_cli_match_runs_fast_and_names_both_algorithms(capsys):
    from poker.__main__ import main

    start = time.time()
    assert main(["match", "--opponent", "chance", "--hands", "400", "--seed", "2"]) == 0
    assert time.time() - start < 5.0
    out = capsys.readouterr().out
    assert "CFR+ average strategy" in out and "Chance (fixed odds" in out
    assert "400 hands (200 deals" in out and "95% CI" in out


def test_cli_match_is_seeded(capsys):
    from poker.__main__ import main

    main(["match", "--opponent", "chance", "--hands", "300", "--seed", "9"])
    first = capsys.readouterr().out.splitlines()[2:]  # skip the header and the timing line, which varies
    main(["match", "--opponent", "chance", "--hands", "300", "--seed", "9"])
    second = capsys.readouterr().out.splitlines()[2:]
    assert first == second
