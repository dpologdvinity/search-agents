"""Leduc hold'em: a six-card deck (two each of J, Q, K), two betting rounds, and one public card.

The hand, in order:
  1. Each seat antes 1 chip and is dealt one private card, face down.
  2. Round 1 betting, with bets of 2 chips and at most two bets or raises in the round.
  3. A public card is turned up (chance).
  4. Round 2 betting, with bets of 4 chips and the same cap.
  5. Showdown. A card that matches the public card makes a pair, which beats any unpaired hand.
     Otherwise the higher rank wins, and equal ranks split the pot.

Leduc is the standard small benchmark for imperfect-information solvers: 288 information sets
(each one a set of situations the player cannot tell apart), large enough that CFR's behaviour
shows, and small enough that exploitability can be computed exactly by a best-response pass.

Card ids are 0..5 and the rank of card i is i // 2, so ids 0 and 1 are the two jacks, 2 and 3 the
two queens, 4 and 5 the two kings. The state is immutable. The root is a chance node that deals
two distinct private cards (30 equally likely deals). The public card is a chance node after
round 1 (four equally likely cards). Both can also be fixed in advance (`LeducState.dealt`), which
is how the benchmark and the web game sample hands without walking the chance nodes.
"""

from __future__ import annotations

import itertools

from .betting import commitments, folder, legal_actions, round_closed

RANKS = ("J", "Q", "K")
DECK = tuple(range(6))
ANTE = 1
BET_SIZES = (2, 4)  # round 1 and round 2 bet sizes
MAX_RAISES = 2      # bets plus raises allowed in one round


def rank_of(card: int) -> int:
    """Rank index 0..2 (J, Q, K) of a card id."""
    return card // 2


def card_name(card: int) -> str:
    """Rank letter of a card, e.g. "K"."""
    return RANKS[rank_of(card)]


class LeducState:
    """A point in a Leduc hand.

    `cards` is None only at the root (deal not yet made). `public` is None until the public card is
    dealt. `hist` is the betting so far: round 1 actions, then "/" once round 1 closes, then round 2.
    """

    __slots__ = ("cards", "public", "hist")

    def __init__(self, cards: tuple[int, int] | None = None, public: int | None = None, hist: str = ""):
        self.cards = cards
        self.public = public
        self.hist = hist

    @classmethod
    def dealt(cls, card0: int, card1: int, public: int) -> LeducState:
        """A hand whose cards are already fixed. The public card is known but stays hidden until round 2."""
        assert len({card0, card1, public}) == 3, "the three cards must be distinct"
        return cls((card0, card1), public, "")

    # ── Chance ───────────────────────────────────────────────────────────

    def is_chance(self) -> bool:
        if self.cards is None:
            return True  # the private deal
        # Round 1 closed but the public card not yet turned up: that is a chance node.
        return self.public is None and self.hist.endswith("/") and folder(self.hist.split("/")) is None

    def chance_outcomes(self) -> list[tuple[float, LeducState]]:
        if self.cards is None:
            deals = list(itertools.permutations(DECK, 2))
            p = 1 / len(deals)
            return [(p, LeducState(d, None, "")) for d in deals]
        # Public card: any of the four cards not held by a seat, equally likely.
        left = [c for c in DECK if c not in self.cards]
        p = 1 / len(left)
        return [(p, LeducState(self.cards, c, self.hist)) for c in left]

    # ── Decisions ────────────────────────────────────────────────────────

    def is_terminal(self) -> bool:
        if self.cards is None or self.is_chance():
            return False
        rounds = self.hist.split("/")
        if folder(rounds) is not None:
            return True
        return len(rounds) == 2 and round_closed(rounds[1])  # round 2 closed: showdown

    def player(self) -> int:
        # Seat 0 opens each round, so the length of the current round says who acts.
        return len(self.hist.split("/")[-1]) % 2

    def actions(self) -> tuple[str, ...]:
        return legal_actions(self.hist.split("/")[-1], MAX_RAISES)

    def apply(self, action: str) -> LeducState:
        assert action in self.actions(), (action, self.hist)
        hist = self.hist + action
        rounds = hist.split("/")
        # Round 1 just closed, and nobody folded: put the "/" that marks the chance node for the public card.
        if len(rounds) == 1 and round_closed(hist) and folder(rounds) is None:
            hist += "/"
        return LeducState(self.cards, self.public, hist)

    # ── Payoffs and information ─────────────────────────────────────────

    def contributions(self) -> tuple[int, int]:
        """Chips each seat has put into the pot so far: the antes plus every round's commitments."""
        put = [ANTE, ANTE]
        for r, text in enumerate(self.hist.split("/")):
            a, b = commitments(text, BET_SIZES[r])
            put[0] += a
            put[1] += b
        return put[0], put[1]

    def payoff(self) -> float:
        """Net chips for seat 0 at a terminal state. Zero-sum: seat 1's payoff is the negative."""
        put0, put1 = self.contributions()
        who = folder(self.hist.split("/"))
        if who == 1:
            return float(put1)  # seat 1 folded: seat 0 takes seat 1's chips
        if who == 0:
            return float(-put0)
        winner = self.showdown_winner()
        if winner is None:
            return 0.0
        return float(put1) if winner == 0 else float(-put0)

    def showdown_winner(self) -> int | None:
        """0 or 1 for the seat with the better hand, None for a tie. Pairs with the public card win first."""
        pub = rank_of(self.public)
        r0, r1 = rank_of(self.cards[0]), rank_of(self.cards[1])
        if r0 == pub and r1 != pub:
            return 0
        if r1 == pub and r0 != pub:
            return 1
        if r0 == r1:
            return None
        return 0 if r0 > r1 else 1

    def infoset(self, seat: int) -> str:
        """What the seat knows, as a key: its card, the public card once it is shown, and the betting.

        Examples: "K|-|kb" (king, round 1 check-bet) and "Q|K|kb/c" (queen, public king, round 2 check).
        The public card is left out during round 1 because nobody has seen it yet, so a player's
        round 1 information sets do not depend on which public card will come.
        """
        pub = card_name(self.public) if (self.public is not None and "/" in self.hist) else "-"
        return f"{card_name(self.cards[seat])}|{pub}|{self.hist}"

    def __repr__(self) -> str:  # for debugging and test failure messages
        return f"LeducState(cards={self.cards}, public={self.public}, hist={self.hist!r})"


def root() -> LeducState:
    """The state before the deal."""
    return LeducState()
