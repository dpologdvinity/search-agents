"""Chance: a seat that picks its actions from fixed odds. It never reads its card, searches, or looks ahead.

Think of the 2048 tile spawner, which places a 2 with 90% and a 4 with 10%: the rule is a fixed
distribution, not a judgement about the board. Chance works the same way. At every decision it
looks only at which actions are legal, weights each one by the odds for its kind, renormalises
over the legal set, and draws one with the seeded random generator.

Why these weights: fold 15%, call-or-check 55%, bet-or-raise 30%. Chance folds less often than it
continues, because a fold gives up the pot at once and a random player that folds often loses to
almost any strategy. It bets or raises a minority of the time, so it is passive, but not so passive
that it never pressures. These are round numbers chosen for the game, not fitted to any result, and
the same table is used at every decision in both rounds and for both seats.

The table is the whole algorithm. Nothing in this module changes with the cards, the pot, or the
history beyond which actions are legal, so a seeded chance player is fully reproducible.
"""

from __future__ import annotations

import random

from .betting import BET, CALL, CHECK, FOLD, RAISE
from .strategy import sample

# Odds for each kind of action. They sum to one; `legal_weights` renormalises over the legal subset.
ACTION_ODDS = {"fold": 0.15, "call_or_check": 0.55, "bet_or_raise": 0.30}

# Which kind of action each letter is. Check and call are the same kind: both continue without adding a bet.
KIND = {FOLD: "fold", CHECK: "call_or_check", CALL: "call_or_check", BET: "bet_or_raise", RAISE: "bet_or_raise"}


def legal_weights(legal) -> dict[str, float]:
    """The probability of each legal action: its kind's odds, divided by the total over the legal actions.

    Example: facing a bet, the legal set is (f, c, r) with weights 0.15, 0.55, 0.30, which already sum to 1.
    At the raise cap the set is (f, c), so the weights become 0.15/0.70 and 0.55/0.70. Each kind has at most
    one legal action in any state, so the dict is built in the order the actions were listed.
    """
    raw = {a: ACTION_ODDS[KIND[a]] for a in legal}
    total = sum(raw.values())
    return {a: p / total for a, p in raw.items()}


class ChanceBot:
    """Draws each action from `legal_weights`. The same bot object works for either seat."""

    name = "chance"

    def act(self, state, rng: random.Random) -> str:
        return sample(legal_weights(state.actions()), rng)
