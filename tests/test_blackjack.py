import random

import pytest

from blackjack.game import Round, Shoe, simulate
from blackjack.learner import agreement, learn
from blackjack.rules import CARD_PROB
from blackjack.solver import UPCARDS, best, dealer_outcomes, get_solver, table_rows


@pytest.fixture(scope="module")
def solver():
    return get_solver()


def test_card_probabilities_sum_to_one():
    assert sum(CARD_PROB.values()) == pytest.approx(1.0)


@pytest.mark.parametrize("upcard", UPCARDS)
def test_dealer_distribution_sums_to_one(upcard):
    p_bj, dist = dealer_outcomes(upcard)
    assert sum(dist) == pytest.approx(1.0)
    assert 0.0 <= p_bj < 1.0
    # Only ace and ten upcards can have a blackjack, and only with the matching hole card.
    assert (p_bj > 0) == (upcard in (1, 10))


def test_dealer_busts_about_42_percent_from_six(solver):
    # A dealer 6 is the classic weak upcard: it busts a little over 42% of the time.
    assert 0.40 < dealer_outcomes(6)[1][5] < 0.44


@pytest.mark.parametrize(("kind", "t", "upcard", "action"), [
    ("hard", 16, 10, "hit"),
    ("hard", 11, 6, "double"),
    ("hard", 12, 4, "stand"),
    ("hard", 9, 3, "double"),
    ("soft", 18, 9, "hit"),
    ("soft", 18, 2, "stand"),
    ("pair", 8, 10, "split"),
    ("pair", 10, 6, "stand"),
    ("pair", 1, 10, "split"),
])
def test_known_basic_strategy_cells(solver, kind, t, upcard, action):
    assert best(solver.cell_values(kind, t, upcard)) == action


def test_house_edge_is_in_the_expected_range(solver):
    edge = solver.house_edge()
    assert 0.003 < edge < 0.007


def test_strategy_table_has_every_row_and_upcard(solver):
    rows = solver.strategy()["rows"]
    assert len(rows) == len(table_rows())
    assert all([c["upcard"] for c in row["cells"]] == list(UPCARDS) for row in rows)


def test_naturals_pay_three_to_two_unless_dealer_peek_pushes(solver):
    assert solver.hand_values((1, 10), 6) == {"stand": pytest.approx(1.5)}
    # Against an ace the dealer's peek pushes 4/13 of the time.
    assert solver.hand_values((1, 10), 1) == {"stand": pytest.approx((1 - 4 / 13) * 1.5)}


def test_split_aces_only_stand(solver):
    assert set(solver.hand_values((1, 5), 6, from_split=True, split_aces=True)) == {"stand"}


def test_pairs_split_only_once(solver):
    assert "split" in solver.hand_values((8, 8), 10)
    assert "split" not in solver.hand_values((8, 8), 10, from_split=True)


# Engine: the deal order is player, dealer upcard, player, dealer hole card (see Round).

def test_player_natural_pays_three_to_two():
    rnd = Round(Shoe(stacked=[1, 6, 13, 9]), 10)  # A-K against 6 and 9
    assert rnd.current is None
    assert rnd.settle() == [pytest.approx(15.0)]


def test_dealer_natural_ends_the_round_at_once():
    rnd = Round(Shoe(stacked=[10, 1, 9, 10]), 10)  # 19 against an ace and a ten
    assert rnd.current is None and rnd.dealer_natural
    assert rnd.settle() == [-10]


def test_dealer_stands_on_soft_17():
    rnd = Round(Shoe(stacked=[10, 1, 8, 6]), 4)  # 18 against A-6 (soft 17)
    assert rnd.current is not None
    rnd.act("stand")
    assert rnd.settle() == [4]
    assert len(rnd.dealer) == 2  # soft 17 stands, so no extra card


def test_double_gets_one_card_and_pays_twice():
    # Player 5-6 doubles, draws a ten to make 21; the dealer 6-10 draws a ten and busts.
    rnd = Round(Shoe(stacked=[5, 6, 6, 10, 10, 10]), 3)
    assert rnd.legal_actions() == ["stand", "hit", "double"]
    rnd.act("double")
    assert rnd.settle() == [6]


def test_split_aces_get_one_card_each_and_cannot_act():
    # Aces split: hands A-9 and A-5. The dealer's 16 draws a ten and busts, so both hands win.
    rnd = Round(Shoe(stacked=[1, 6, 1, 10, 9, 5, 10]), 1)
    assert rnd.legal_actions() == ["stand", "hit", "double", "split"]
    rnd.act("split")
    assert len(rnd.hands) == 2 and rnd.current is None
    assert rnd.settle() == [1, 1]


def test_only_one_split_and_no_split_after_split():
    rnd = Round(Shoe(stacked=[8, 6, 8, 10, 8, 8, 2, 3]), 1)  # 8-8 against 6
    rnd.act("split")
    assert "split" not in rnd.legal_actions()  # hands from a split cannot split again
    with pytest.raises(ValueError):
        rnd.act("split")


def test_illegal_actions_are_refused():
    rnd = Round(Shoe(stacked=[10, 6, 9, 10]), 1)  # 19 against a 6 cannot split
    with pytest.raises(ValueError):
        rnd.act("split")


def test_shoe_reshuffles_after_penetration():
    shoe = Shoe(decks=1, rng=random.Random(4))
    for _ in range(40):
        shoe.begin_round()
        for _ in range(3):
            shoe.draw()
    assert shoe.shuffles > 1


def test_simulated_basic_strategy_is_near_the_exact_edge(solver):
    result = simulate(20_000, seed=3)
    assert result.rounds == 20_000
    assert abs(result.mean + solver.house_edge()) < 5 * result.stderr + 0.01


# Terminal game and subcommands.

def test_terminal_session_hints_then_settles():
    from blackjack.__main__ import play

    lines = []
    state = {"bets": 0, "hinted": False}

    def read(prompt):
        if prompt.startswith("Bet"):
            state["bets"] += 1
            return "10" if state["bets"] <= 3 else "q"
        if not state["hinted"]:
            state["hinted"] = True
            return "h"
        return "s"

    final = play(100, seed=5, read=read, out=lines.append)
    text = "\n".join(lines)
    assert "Basic strategy:" in text and "Expected chips:" in text
    assert text.count("Bankroll:") == 3
    assert final == pytest.approx(100 + sum(float(x.split()[-1]) * (1 if "win" in x else -1)
                                            for x in lines if x.strip().startswith(("you win", "you lose"))))


def test_strategy_and_simulate_subcommands(capsys):
    from blackjack.__main__ import main

    assert main(["strategy"]) == 0
    out = capsys.readouterr().out
    assert "House edge with perfect basic strategy" in out and " 16  " in out
    assert main(["simulate", "--hands", "2000", "--seed", "4"]) == 0
    assert "Measured return per unit bet" in capsys.readouterr().out


# Monte Carlo control.

def test_learning_is_reproducible_with_a_seed():
    a = learn(1500, seed=7, snapshots=2)
    b = learn(1500, seed=7, snapshots=2)
    assert [s.actions for s in a] == [s.actions for s in b]


def test_learning_improves_with_more_hands():
    snaps = learn(20_000, seed=1, snapshots=4)
    first, last = snaps[0], snaps[-1]
    assert last.episodes == 20_000
    assert last.agreement > first.agreement
    assert last.loss < first.loss
    assert last.agreement > 0.6


def test_learner_cells_are_all_tried_after_a_short_run():
    final = learn(5_000, seed=2, snapshots=1)[-1]
    assert "?" not in final.actions
    assert len(final.actions) == len(table_rows()) * len(UPCARDS)


def test_agreement_with_untried_table_is_zero():
    from blackjack.learner import MonteCarloControl

    share, decisive, loss, letters = agreement(MonteCarloControl(seed=0))
    assert letters == "?" * len(letters)
    assert share == 0.0 and decisive > 0 and loss > 0
