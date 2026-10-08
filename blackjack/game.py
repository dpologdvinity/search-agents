"""Playable blackjack engine: a multi-deck shoe, one round at a time, the dealer's play, and payouts.

The engine knows nothing about strategy. The caller asks for the legal actions and chooses one: the terminal
game asks the player, and the simulation asks the exact solver (`basic_action`). Rules are those in `rules.py`:
S17, dealer peek, 3:2 blackjack, double on any two cards, one split, doubling after a split allowed.

Cards are ranks 1..13 (see `rules.py`). Chips are floats so that a 3:2 payout of an odd bet stays exact.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

from .rules import BLACKJACK_PAYOUT, DEALER_STAND, hard_and_ace, rank_value, total
from .solver import basic_action


class Shoe:
    """A multi-deck shoe that reshuffles once `penetration` of it has been dealt.

    Dealing from a shoe, rather than drawing each card independently, is the real-casino model. Its edge differs
    slightly from the infinite-deck solver's (see `simulate`). `stacked` ranks are dealt first, which tests use
    to set up exact hands.
    """

    def __init__(self, decks: int = 6, penetration: float = 0.75, rng: random.Random | None = None,
                 stacked: list[int] | None = None) -> None:
        self.decks = decks
        self.penetration = penetration
        self.rng = rng if rng is not None else random.Random()
        self._stacked = list(stacked or [])
        self._cards: list[int] = []
        self._pos = 0
        self.shuffles = 0
        self._shuffle()

    def _shuffle(self) -> None:
        cards = [rank for rank in range(1, 14) for _ in range(4 * self.decks)]
        self.rng.shuffle(cards)
        self._cards = cards
        self._pos = 0
        self.shuffles += 1

    def begin_round(self) -> None:
        """Reshuffle between rounds, once the shoe is past its penetration (never mid-round)."""
        if self._pos >= self.penetration * len(self._cards):
            self._shuffle()

    def draw(self) -> int:
        """The next rank. Stacked ranks come first; past the end of the shoe, the shoe reshuffles."""
        if self._stacked:
            return self._stacked.pop(0)
        if self._pos >= len(self._cards):
            self._shuffle()
        rank = self._cards[self._pos]
        self._pos += 1
        return rank


@dataclass(eq=False)
class Hand:
    """One of the player's hands: its ranks, bet, and state.

    `from_split` hands cannot split again. A split ace is done after one card. A two-card 21 is a natural
    (blackjack) unless the hand came from a split.
    """

    ranks: list[int]
    bet: float
    from_split: bool = False
    done: bool = False
    doubled: bool = field(default=False)

    @property
    def values(self) -> list[int]:
        return [rank_value(r) for r in self.ranks]

    @property
    def hard_ace(self) -> tuple[int, bool]:
        return hard_and_ace(self.values)

    @property
    def best(self) -> int:
        return total(*self.hard_ace)

    @property
    def busted(self) -> bool:
        return self.hard_ace[0] > 21

    @property
    def natural(self) -> bool:
        return len(self.ranks) == 2 and self.best == 21 and not self.from_split

    @property
    def split_aces(self) -> bool:
        return self.from_split and self.ranks[0] == 1

    @property
    def can_split(self) -> bool:
        return (len(self.ranks) == 2 and not self.from_split
                and rank_value(self.ranks[0]) == rank_value(self.ranks[1]))


class Round:
    """One hand of blackjack from the deal to settlement.

    Deal order follows a real table: player, dealer upcard, player, dealer hole card. The dealer peeks at once:
    a dealer natural ends the round before the player acts. Then `act` applies the player's choices to each
    hand in turn, and `settle` plays the dealer and pays out.
    """

    def __init__(self, shoe: Shoe, bet: float) -> None:
        if bet <= 0:
            raise ValueError("the bet must be positive")
        self.shoe = shoe
        first = shoe.draw()
        up = shoe.draw()
        second = shoe.draw()
        self.hands = [Hand([first, second], bet)]
        self.dealer = [up, shoe.draw()]
        self.dealer_natural = len(self.dealer) == 2 and total(*hard_and_ace([rank_value(r) for r in self.dealer])) == 21
        self._settled = False
        for hand in self.hands:
            hand.done = hand.natural or self.dealer_natural

    @property
    def upcard_value(self) -> int:
        return rank_value(self.dealer[0])

    @property
    def current(self) -> Hand | None:
        """The hand the player is acting on, or None once every hand is finished."""
        for hand in self.hands:
            if not hand.done:
                return hand
        return None

    @property
    def wagered(self) -> float:
        return sum(h.bet for h in self.hands)

    def legal_actions(self, funds: float | None = None) -> list[str]:
        """Actions the rules allow on the current hand. `funds` is the chips left to bet beyond the wagers so far.

        Doubling and splitting both need one more bet of the hand's size. `None` means no cash limit.
        """
        hand = self.current
        if hand is None:
            return []
        actions = ["stand", "hit"]
        if len(hand.ranks) == 2:
            can_afford = funds is None or funds >= hand.bet
            if can_afford:
                actions.append("double")
            if hand.can_split and len(self.hands) == 1 and can_afford:
                actions.append("split")
        return actions

    def act(self, action: str, funds: float | None = None) -> None:
        """Apply one player action to the current hand."""
        if action not in self.legal_actions(funds):
            raise ValueError(f"{action!r} is not allowed here")
        hand = self.current
        if action == "stand":
            hand.done = True
        elif action == "hit":
            hand.ranks.append(self.shoe.draw())
            hand.done = hand.busted or hand.best == 21  # a hand at 21 stands automatically
        elif action == "double":
            hand.bet *= 2
            hand.doubled = True
            hand.ranks.append(self.shoe.draw())
            hand.done = True
        else:  # split: each hand keeps one card of the pair and receives a second card at once
            rank = hand.ranks[0]
            hands = []
            for _ in range(2):
                new = Hand([rank, self.shoe.draw()], hand.bet, from_split=True)
                new.done = new.split_aces or new.best == 21
                hands.append(new)
            self.hands = hands

    def _dealer_play(self) -> None:
        """The dealer draws to 17 or more, and soft 17 stands (S17). Only called while a hand is still live."""
        while total(*hard_and_ace([rank_value(r) for r in self.dealer])) < DEALER_STAND:
            self.dealer.append(self.shoe.draw())

    def _net(self, hand: Hand, dealer_hard: int, dealer_ace: bool) -> float:
        """Chips won (+) or lost (-) on one hand. A dealer natural beats everything except a player natural."""
        if self.dealer_natural:
            return 0.0 if hand.natural else -hand.bet
        if hand.natural:
            return BLACKJACK_PAYOUT * hand.bet
        if hand.busted:
            return -hand.bet
        if dealer_hard > 21:
            return hand.bet
        dealer_total = total(dealer_hard, dealer_ace)
        if hand.best > dealer_total:
            return hand.bet
        if hand.best < dealer_total:
            return -hand.bet
        return 0.0

    def settle(self) -> list[float]:
        """Finish the round: the dealer plays if any hand still needs it, then each hand is paid.

        Returns the chips won or lost per hand (in the same order as `self.hands`). Idempotent.
        """
        if self.current is not None:
            raise RuntimeError("the player has not finished the round")
        if not self._settled:
            if any(not h.busted and not h.natural for h in self.hands) and not self.dealer_natural:
                self._dealer_play()
            self._settled = True
        dealer_hard, dealer_ace = hard_and_ace([rank_value(r) for r in self.dealer])
        return [self._net(h, dealer_hard, dealer_ace) for h in self.hands]


@dataclass(frozen=True)
class SimResult:
    """Outcome of a basic-strategy simulation, per unit of the opening bet."""

    rounds: int
    mean: float  # average chips won per round (negative = the player loses)
    stderr: float  # standard error of the mean


def simulate(rounds: int, seed: int = 0, decks: int = 6) -> SimResult:
    """Play `rounds` rounds of exact basic strategy through a shoe and measure the average result per unit bet.

    The policy is the exact table's best action for each hand, given the dealer's upcard. Comparing the mean with
    the solver's house edge shows the cost of a finite shoe versus the infinite-deck model.
    """
    rng = random.Random(seed)
    shoe = Shoe(decks=decks, rng=rng)
    total_net = total_sq = 0.0
    for _ in range(rounds):
        shoe.begin_round()
        rnd = Round(shoe, 1.0)
        while (hand := rnd.current) is not None:
            rnd.act(basic_action(tuple(hand.values), rnd.upcard_value, hand.from_split, hand.split_aces))
        net = sum(rnd.settle())
        total_net += net
        total_sq += net * net
    mean = total_net / rounds
    var = max(0.0, total_sq / rounds - mean * mean)
    return SimResult(rounds=rounds, mean=mean, stderr=math.sqrt(var / rounds))
