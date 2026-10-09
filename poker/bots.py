"""Players for Leduc: the CFR bot, the fixed-odds chance player, three baselines, and a seeded match runner.

Every bot has `act(state, rng) -> action`, where `state` is a decision state and the bot acts for
`state.player()`. Bots see only what their seat sees (`state.infoset(seat)` for the CFR bot, the seat's
own card for the heuristic), so none of them can read the opponent's card. Chance sees no card at all.

The two opponents the web game and the CLI offer are `cfr` (the trained average strategy) and `chance`
(fixed action odds, see chance.py). `make_opponent` maps those names to bots.

Baselines, from weakest to strongest:
  random       uniform over the legal actions
  always-call  call any bet, check when nothing is owed (never folds, never bets)
  hand-strength a fixed rule from the seat's own card: pairs with the public card bet and raise,
               kings bet and call, queens check and call, jacks check and fold to a bet

The benchmark reports winnings in milli-big-blinds per hand, where the big blind is the round 1
bet size (2 chips), and it plays every deal twice, once in each seat, to cancel the luck of the cards.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

from .betting import BET, CALL, CHECK, FOLD, RAISE
from .chance import ChanceBot
from .leduc import BET_SIZES, DECK, LeducState, rank_of
from .strategy import sample

BIG_BLIND = BET_SIZES[0]  # chips: the round 1 bet, which is the unit poker results are quoted in


class RandomBot:
    name = "random"

    def act(self, state, rng: random.Random) -> str:
        return rng.choice(state.actions())


class AlwaysCall:
    name = "always-call"

    def act(self, state, rng: random.Random) -> str:
        acts = state.actions()
        return CALL if CALL in acts else CHECK


class HandStrength:
    """Rule-based player that reads only its own card and the public card.

    A pair with the public card is the strongest hand, then a king, then a queen, then a jack.
    The rules are simple value betting: bet or raise with the strongest hands, check or call with
    the middle ones, and check and fold the jacks when a bet comes.
    """

    name = "hand-strength"

    def act(self, state, rng: random.Random) -> str:
        seat = state.player()
        acts = state.actions()
        rank = rank_of(state.cards[seat])
        shown = state.public is not None and "/" in state.hist
        paired = shown and rank_of(state.public) == rank
        facing = FOLD in acts  # fold is only legal when a bet is owed
        if paired:
            if facing:
                return RAISE if RAISE in acts else CALL
            return BET
        if rank == 2:  # king, no pair
            return CALL if facing else BET
        if rank == 1:  # queen
            return CALL if facing else CHECK
        return FOLD if facing else CHECK  # jack


class CFRBot:
    """Plays the average strategy from a trained table: the action is sampled from the row for its information set."""

    name = "cfr"

    def __init__(self, table: dict[str, dict[str, float]]):
        self.table = table

    def act(self, state, rng: random.Random) -> str:
        return sample(self.table[state.infoset(state.player())], rng)


OPPONENTS = ("cfr", "chance")  # the opponents the web game and the CLI offer, by name


def make_opponent(name: str, table: dict[str, dict[str, float]]):
    """The bot for an opponent name: "cfr" plays the trained table, "chance" draws from the fixed odds."""
    if name == "cfr":
        return CFRBot(table)
    if name == "chance":
        return ChanceBot()
    raise ValueError(f"unknown opponent {name!r}: choose from {', '.join(OPPONENTS)}")


def deal(rng: random.Random) -> LeducState:
    """A random hand: two private cards and the public card, all distinct, from the six-card deck."""
    a, b, pub = rng.sample(DECK, 3)
    return LeducState.dealt(a, b, pub)


def play_hand(state: LeducState, bots, rng: random.Random) -> float:
    """Play one hand with bots[0] in seat 0 and bots[1] in seat 1. Returns seat 0's net chips."""
    while not state.is_terminal():
        state = state.apply(bots[state.player()].act(state, rng))
    return state.payoff()


@dataclass
class MatchResult:
    """Head-to-head result for bot `a` against bot `b`, from a's side."""

    a: str
    b: str
    deals: int
    hands: int           # each deal is played twice, once in each seat
    mean_chips: float    # chips per hand, a's side
    se_chips: float      # standard error of the mean, from the spread over deals
    mbb_per_hand: float  # milli-big-blinds per hand
    mbb_se: float        # standard error in milli-big-blinds
    seconds: float = 0.0

    def ci95_mbb(self) -> tuple[float, float]:
        return self.mbb_per_hand - 1.96 * self.mbb_se, self.mbb_per_hand + 1.96 * self.mbb_se


def head_to_head(a, b, deals: int, seed: int) -> MatchResult:
    """Play `deals` random deals, each twice (a in seat 0, then a in seat 1), and summarize a's winnings.

    The per-deal figure is (a's chips in seat 0 - a's chips in seat 1) / 2, which is a's average per hand
    for that deal. Each deal is an independent sample, so the standard error comes from the spread of
    those per-deal figures. The seed fixes the cards, the bots' mixed choices, and the random bot's choices.
    """
    rng = random.Random(seed)
    total = 0.0
    total_sq = 0.0
    for _ in range(deals):
        state = deal(rng)
        seat0_first = play_hand(state, (a, b), rng)     # a in seat 0: payoff is seat 0's
        seat0_second = play_hand(state, (b, a), rng)    # a in seat 1: a's payoff is minus seat 0's
        d = (seat0_first - seat0_second) / 2
        total += d
        total_sq += d * d
    mean = total / deals
    var = max(total_sq / deals - mean * mean, 0.0) * deals / (deals - 1) if deals > 1 else 0.0
    se = math.sqrt(var / deals)
    scale = 1000 / BIG_BLIND
    return MatchResult(
        a=a.name, b=b.name, deals=deals, hands=2 * deals,
        mean_chips=mean, se_chips=se,
        mbb_per_hand=mean * scale, mbb_se=se * scale,
    )

