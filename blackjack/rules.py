"""Card model and table rules shared by the exact solver, the Monte Carlo learner, and the playable game.

Two card encodings appear in this package:
- rank: 1..13 as dealt from a shoe (1 = ace, 11 = jack, 12 = queen, 13 = king).
- value: 1..10 for the probability model (ace = 1, every ten-card = 10). Hands are scored by value.

Rules, identical everywhere in the package:
- The dealer stands on every total of 17 or more, soft 17 included (S17).
- The dealer checks ("peeks") for blackjack when showing an ace or a ten-card; a dealer blackjack ends
  the round before the player acts, so the player loses the opening bet (a player blackjack pushes).
- Blackjack pays 3:2. Other wins pay 1:1, pushes return the bet, losses lose it.
- Double down on any two cards. One split only (two hands). Split aces get one card each and no further
  action. Doubling after a split is allowed. A split hand that makes 21 is not a blackjack and pays 1:1.
"""

from __future__ import annotations

# Probability that one draw from an infinite deck has each value. There are 13 ranks per suit: the ace and
# 2..9 are one rank each (1/13 apiece), while 10, J, Q, K all score 10, so that value has 4/13.
CARD_PROB: dict[int, float] = {v: (4 / 13 if v == 10 else 1 / 13) for v in range(1, 11)}

BLACKJACK_PAYOUT = 1.5  # 3:2 on the bet
DEALER_STAND = 17  # the dealer draws while its best total is below this, so soft 17 stands (S17)
MAX_HANDS = 2  # one split means at most two hands per round

RANK_LABELS = {1: "A", 11: "J", 12: "Q", 13: "K"}


def rank_value(rank: int) -> int:
    """Blackjack value of a dealt rank. The ace is 1 here; the caller decides whether it counts as 11."""
    return min(rank, 10)


def rank_label(rank: int) -> str:
    """Short printable name of a rank, e.g. "A", "10", "K"."""
    return RANK_LABELS.get(rank, str(rank))


def total(hard: int, ace: bool) -> int:
    """Best total of a hand from its hard total (aces counted as 1) and whether it holds an ace.

    An ace counts as 11 whenever that keeps the hand at 21 or less; otherwise the hand is hard.
    """
    return hard + 10 if ace and hard + 10 <= 21 else hard


def hard_and_ace(values: tuple[int, ...] | list[int]) -> tuple[int, bool]:
    """Hard total and ace flag of a hand given its card values."""
    return sum(values), 1 in values


def is_soft(values: tuple[int, ...] | list[int]) -> bool:
    """True when an ace is counted as 11 in the hand's best total."""
    hard, ace = hard_and_ace(values)
    return ace and total(hard, ace) != hard
