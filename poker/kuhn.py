"""Kuhn poker: three cards (J, Q, K), one card each, one betting round, one bet of 1 chip.

Kuhn is the smallest poker game with bluffing and calling decisions, and it is small enough to
check the solver exactly: its game value for the first player is -1/18 at equilibrium (the
second player wins 1/18 of a chip per hand on average under perfect play).

The state is immutable. The root is a chance node that deals two distinct cards out of three,
so there are six equally likely deals. A decision state exposes `infoset(seat)`: the cards the
seat can see plus the betting history, which is the key the solver uses to share strategy
across histories that look the same to the player.
"""

from __future__ import annotations

import itertools

from .betting import commitments, folder, legal_actions, round_closed

RANKS = ("J", "Q", "K")  # card ids 0, 1, 2 are these ranks: a higher id beats a lower one at showdown
ANTE = 1
BET_SIZE = 1
MAX_RAISES = 1  # one bet per round in Kuhn: no raise after a bet


class KuhnState:
    """A point in a Kuhn hand. `cards` is None only at the root, where the deal is still a chance node."""

    __slots__ = ("cards", "hist")

    def __init__(self, cards: tuple[int, int] | None = None, hist: str = ""):
        self.cards = cards  # (seat 0's card, seat 1's card)
        self.hist = hist    # betting actions so far, e.g. "kb" (check, then bet)

    def is_chance(self) -> bool:
        return self.cards is None

    def chance_outcomes(self) -> list[tuple[float, KuhnState]]:
        """All ordered pairs of distinct cards, equally likely. Seat 0 and seat 1 each get one card."""
        deals = list(itertools.permutations(range(len(RANKS)), 2))
        p = 1 / len(deals)
        return [(p, KuhnState(d)) for d in deals]

    def is_terminal(self) -> bool:
        if self.cards is None:
            return False
        return folder([self.hist]) is not None or round_closed(self.hist)

    def player(self) -> int:
        # Seat 0 acts first and turns alternate, so the length of the history says who is due.
        return len(self.hist) % 2

    def actions(self) -> tuple[str, ...]:
        return legal_actions(self.hist, MAX_RAISES)

    def apply(self, action: str) -> KuhnState:
        assert action in self.actions(), (action, self.hist)
        return KuhnState(self.cards, self.hist + action)

    def payoff(self) -> float:
        """Net chips for seat 0 at a terminal state. Zero-sum: seat 1's payoff is the negative."""
        put0, put1 = commitments(self.hist, BET_SIZE)
        put = (ANTE + put0, ANTE + put1)  # chips each seat has in the pot
        who = folder([self.hist])
        if who == 1:
            return float(put[1])  # seat 1 folded: seat 0 takes seat 1's chips
        if who == 0:
            return float(-put[0])
        # Showdown: the higher card takes the pot; equal cards cannot happen with one card per rank.
        if self.cards[0] == self.cards[1]:
            return 0.0
        return float(put[1]) if self.cards[0] > self.cards[1] else float(-put[0])

    def infoset(self, seat: int) -> str:
        """What the seat knows: its own card and the betting so far, e.g. "K|kb"."""
        return f"{RANKS[self.cards[seat]]}|{self.hist}"

    def __repr__(self) -> str:  # for debugging and test failure messages
        return f"KuhnState(cards={self.cards}, hist={self.hist!r})"


def root() -> KuhnState:
    """The state before the deal."""
    return KuhnState()
