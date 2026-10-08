import itertools
import random

import pytest

from poker import kuhn, leduc, strategy
from poker.betting import commitments, legal_actions, round_closed
from poker.bots import AlwaysCall, CFRBot, HandStrength, RandomBot, deal, head_to_head, play_hand
from poker.cfr import Solver, regret_matching, train
from poker.cfr import table as average_table
from poker.leduc import LeducState
from poker.train import KUHN_VALUE, tree_for
from poker.tree import best_response, build_tree, expected_value, exploitability


class Script:
    """A bot that plays a fixed list of actions, one per decision."""

    name = "script"

    def __init__(self, actions):
        self.actions = list(actions)

    def act(self, state, rng):
        return self.actions.pop(0)


# ── Betting rules ────────────────────────────────────────────────────────

def test_round_closes_on_call_or_double_check():
    assert round_closed("kbc")
    assert round_closed("kk")
    assert round_closed("bc")
    assert not round_closed("k")
    assert not round_closed("b")
    assert not round_closed("kb")


def test_legal_actions_respect_the_raise_cap():
    assert legal_actions("", 2) == ("k", "b")
    assert legal_actions("k", 2) == ("k", "b")
    assert legal_actions("b", 2) == ("f", "c", "r")
    assert legal_actions("br", 2) == ("f", "c")   # two raises made: fold or call only
    assert legal_actions("b", 1) == ("f", "c")    # Kuhn: one bet per round


def test_commitments_match_bet_sizes():
    assert commitments("", 2) == (0, 0)
    assert commitments("b", 2) == (2, 0)
    assert commitments("bc", 2) == (2, 2)
    assert commitments("br", 2) == (2, 4)  # the raise puts seat 1 at the bet plus one bet size; seat 0 keeps its 2
    assert commitments("brc", 4) == (8, 8)
    assert commitments("bf", 2) == (2, 0)


# ── Kuhn ─────────────────────────────────────────────────────────────────

def test_kuhn_has_six_deals_and_twelve_information_sets():
    tree = build_tree(kuhn.root())
    assert len(tree.info_actions) == 12
    assert len(tree.children[0]) == 6
    assert set(tree.infosets) == {"J|", "Q|", "K|", "J|k", "Q|k", "K|k", "J|b", "Q|b", "K|b",
                                  "J|kb", "Q|kb", "K|kb"}


def test_kuhn_payoffs():
    assert kuhn.KuhnState((2, 1), "kk").payoff() == 1.0    # check-check: K beats Q for the ante
    assert kuhn.KuhnState((0, 2), "kk").payoff() == -1.0
    assert kuhn.KuhnState((0, 2), "bc").payoff() == -2.0   # bet and call: showdown for 2 each
    assert kuhn.KuhnState((0, 2), "bf").payoff() == 1.0    # bet and fold: the bettor takes the ante
    assert kuhn.KuhnState((2, 0), "kbf").payoff() == -1.0  # seat 0 checks, seat 1 bets, seat 0 folds


def test_kuhn_cfr_plus_reaches_the_known_value_and_equilibrium():
    tree = build_tree(kuhn.root())
    avg = train(tree, "cfr+", 5000).average()
    assert abs(expected_value(tree, avg) - KUHN_VALUE) < 1e-4
    assert exploitability(tree, avg)[0] < 1e-4


# ── Best response and regret matching ────────────────────────────────────

def test_best_response_matches_brute_force_on_a_random_profile():
    """Against any fixed profile, the two-pass best response equals the best pure strategy found by brute force."""
    rng = random.Random(3)
    tree = build_tree(kuhn.root())
    sigma = []
    for actions in tree.info_actions:
        w = [rng.random() + 0.05 for _ in actions]
        sigma.append([x / sum(w) for x in w])
    for seat in (0, 1):
        mine = [i for i in range(len(tree.infosets)) if tree.info_player[i] == seat]
        best = -1e9
        for combo in itertools.product(range(2), repeat=len(mine)):
            sig = [list(x) for x in sigma]
            for iid, j in zip(mine, combo):
                sig[iid] = [1.0 if k == j else 0.0 for k in range(len(sig[iid]))]
            v = expected_value(tree, sig)
            best = max(best, v if seat == 0 else -v)
        assert best_response(tree, sigma, seat) == pytest.approx(best, abs=1e-12)


def test_regret_matching_plays_positive_regret_only():
    assert regret_matching([3.0, -1.0]) == [1.0, 0.0]
    assert regret_matching([-1.0, -2.0]) == [0.5, 0.5]
    assert regret_matching([1.0, 3.0]) == pytest.approx([0.25, 0.75])


def test_cfr_plus_converges_faster_than_vanilla_on_kuhn():
    tree = build_tree(kuhn.root())
    vanilla = train(tree, "cfr", 2000)
    plus = train(tree, "cfr+", 2000)
    assert plus.exploitability() < vanilla.exploitability()
    with pytest.raises(ValueError):
        Solver(tree, "bogus")


# ── Leduc ────────────────────────────────────────────────────────────────

def test_leduc_tree_size_and_information_sets():
    tree = tree_for("leduc")
    assert len(tree.children[0]) == 30  # ordered pairs of distinct cards out of six
    assert len(tree.infosets) == 288    # the standard Leduc count
    assert len(tree.levels) == 11


def test_leduc_showdown_winner():
    # Card ids: 0 and 1 are J, 2 and 3 are Q, 4 and 5 are K.
    assert LeducState((2, 0), 3, "kk/kk").showdown_winner() == 0  # queen pairs the public queen
    assert LeducState((0, 2), 3, "kk/kk").showdown_winner() == 1  # the same, seat 1 holds the pair
    assert LeducState((4, 2), 0, "kk/kk").showdown_winner() == 0  # king beats queen, no pairs
    assert LeducState((2, 3), 0, "kk/kk").showdown_winner() is None  # two queens: a tie
    assert LeducState((2, 4), 3, "kk/kk").showdown_winner() == 0  # a pair beats a higher unpaired king


def test_leduc_payoff_amounts():
    assert LeducState((4, 2), 0, "kk/kk").payoff() == 1.0    # checks: the loser's ante
    assert LeducState((4, 0), 2, "bc/kk").payoff() == 3.0    # bet 2 and call: each put in 3
    assert LeducState((0, 2), 4, "bf").payoff() == 1.0       # bet 2, seat 1 folds: its ante
    assert LeducState((0, 2), 4, "bc/bc").payoff() == -7.0   # 1+2 then 4 more each: seat 0 loses 7
    assert LeducState((2, 3), 0, "kk/kk").payoff() == 0.0    # tie


def test_leduc_public_card_is_hidden_until_round_two():
    s = LeducState.dealt(4, 0, 2)
    assert s.infoset(0) == "K|-|"
    s = s.apply("k").apply("b")
    assert s.infoset(0) == "K|-|kb"  # still round 1: the public card is not part of the key
    s = s.apply("c")
    assert s.hist == "kbc/"          # round 1 closed: the "/" marks the public card's deal
    assert s.infoset(0) == "K|Q|kbc/"


def test_leduc_terminal_and_chance_states():
    assert leduc.root().is_chance() and len(leduc.root().chance_outcomes()) == 30
    s = LeducState.dealt(0, 2, 4)
    assert not s.is_chance() and not s.is_terminal()
    closed = s.apply("k").apply("k")
    assert closed.hist == "kk/" and not closed.is_terminal()  # round 2 comes next
    pub_chance = LeducState((0, 2), None, "kk/")
    assert pub_chance.is_chance() and len(pub_chance.chance_outcomes()) == 4
    assert LeducState((0, 2), 4, "bf").is_terminal()


def test_leduc_raise_cap_in_round_one():
    s = LeducState.dealt(0, 2, 4).apply("b").apply("r")  # a bet and a raise: no more raises allowed
    assert s.actions() == ("f", "c")
    assert s.player() == 0


# ── Committed strategy and file format ───────────────────────────────────

def test_committed_leduc_strategy_is_a_valid_equilibrium():
    data = strategy.load()
    tree = tree_for("leduc")
    assert data["game"] == "leduc" and data["algorithm"] == "CFR+"
    table = data["strategy"]
    assert set(table) == set(tree.infosets)
    for row in table.values():
        assert sum(row.values()) == pytest.approx(1.0)
    sigma = strategy.profile(tree, table)
    assert exploitability(tree, sigma)[0] < 1e-3
    assert expected_value(tree, sigma) == pytest.approx(-0.0856, abs=2e-4)


def test_saved_strategy_round_trips(tmp_path):
    tree = tree_for("kuhn")
    table = strategy.round_table(average_table(tree, train(tree, "cfr+", 500).average()))
    path = tmp_path / "kuhn.json"
    strategy.save(path, {"game": "kuhn"}, table)
    loaded = strategy.load(path)
    assert loaded["game"] == "kuhn"
    for key, row in table.items():
        assert loaded["strategy"][key] == pytest.approx(row, abs=1e-4)


def test_sample_follows_the_row_and_is_seeded():
    row = {"b": 0.25, "k": 0.75}
    assert strategy.sample(row, random.Random(5)) == strategy.sample(row, random.Random(5))
    rng = random.Random(9)
    draws = [strategy.sample(row, rng) for _ in range(4000)]
    assert 0.20 < draws.count("b") / 4000 < 0.30


# ── Bots and matches ─────────────────────────────────────────────────────

def test_bots_only_choose_legal_actions():
    table = strategy.load()["strategy"]
    rng = random.Random(4)
    bots = [CFRBot(table), RandomBot(), AlwaysCall(), HandStrength()]
    for _ in range(300):
        for bot in bots:
            s = deal(rng)
            while not s.is_terminal():
                action = bot.act(s, rng)
                assert action in s.actions()
                s = s.apply(action)


def test_always_call_never_folds_and_hand_strength_reads_its_card():
    rng = random.Random(0)
    facing_j = LeducState.dealt(4, 0, 2).apply("b")  # seat 0 holds a K and bets; seat 1 holds a J
    assert AlwaysCall().act(facing_j, rng) == "c"
    assert HandStrength().act(facing_j, rng) == "f"
    assert HandStrength().act(LeducState.dealt(4, 0, 2), rng) == "b"  # a king with nothing owed bets
    # The public card only counts once round 2 has started: seat 1 holds a queen that pairs the public queen.
    paired_facing = LeducState((4, 3), 2, "kk/").apply("b")
    assert HandStrength().act(paired_facing, rng) == "r"
    assert HandStrength().act(LeducState.dealt(4, 3, 2).apply("b"), rng) == "c"  # round 1: no pair is visible yet
    assert HandStrength().act(LeducState.dealt(0, 3, 4), rng) == "k"  # a jack with nothing owed checks


def test_play_hand_returns_seat_zero_payoff():
    rng = random.Random(8)
    # Seat 0 bets and seat 1 folds: seat 0 wins seat 1's ante.
    assert play_hand(LeducState.dealt(0, 2, 4), (Script(["b"]), Script(["f"])), rng) == 1.0
    # Seat 0 bets and seat 1 calls, both check round 2 and the queen (seat 1) beats the jack.
    assert play_hand(LeducState.dealt(0, 2, 4), (Script(["b", "k", "k"]), Script(["c", "k", "k"])), rng) == -3.0


def test_head_to_head_is_seeded_and_the_bot_beats_random():
    table = strategy.load()["strategy"]
    a = head_to_head(CFRBot(table), RandomBot(), 2000, seed=11)
    b = head_to_head(CFRBot(table), RandomBot(), 2000, seed=11)
    assert a.mbb_per_hand == b.mbb_per_hand
    assert a.mbb_per_hand > 200 and a.mbb_per_hand > 4 * a.mbb_se


# ── CLI ──────────────────────────────────────────────────────────────────

def test_cli_session_plays_and_quits(capsys):
    from poker.__main__ import session

    table = strategy.load()["strategy"]
    replies = iter(["h", "k", "c", "b", "f", "k", "k", "q"])
    session(seed=3, table=table, read=lambda _p: next(replies, "q"), out=print, hands=None)
    out = capsys.readouterr().out
    assert "hint:" in out and "Hand 1" in out and "bye" in out


def test_cli_rejects_illegal_input(capsys):
    from poker.__main__ import session

    table = strategy.load()["strategy"]
    replies = iter(["c", "q"])  # calling with nothing owed is not legal
    session(seed=3, table=table, read=lambda _p: next(replies), out=print, hands=1)
    assert "not a legal choice" in capsys.readouterr().out


def test_cli_train_kuhn_and_exploit(capsys):
    from poker.__main__ import main

    assert main(["train", "--game", "kuhn", "--iterations", "3000"]) == 0
    assert main(["exploit", "--game", "leduc"]) == 0
    out = capsys.readouterr().out
    assert "exploitability" in out and "game value for seat 0" in out
