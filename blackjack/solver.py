"""Exact blackjack values by dynamic programming in the infinite-deck model.

Why this is exact: with an infinite deck every card is an independent draw, so the player's decisions form a
Markov decision process. A state is small: the hard total, whether an ace is in the hand, whether the hand is
still its first two cards, and the dealer's upcard. A hit always adds a card, so every state's value depends
only on states with a larger hard total. That acyclic structure makes a memoised recursion exact backward
induction: nothing is approximated except the infinite-deck assumption itself.

Steps, all in units of the bet (+1 win, -1 loss, 0 push):
1. Dealer distribution. For each hard total and ace flag, the probability that the dealer finishes on
   17..21 or busts (S17 rule). Recursing on the dealer's next card gives this exactly.
2. Peek. With an ace or ten showing, the dealer checks for blackjack first. The hole card is therefore
   conditioned on "not blackjack" for the decisions, and the peek's own outcome is added at the end.
3. Stand, hit, and double for each player state. Stand compares the player's total with the dealer's
   distribution; hit and double recurse on the next card.
4. Split. Under the infinite deck the two hands of a split are independent draws, so the split is twice the
   average, over the second card, of one hand's best play.
"""

from __future__ import annotations

from functools import cache, lru_cache

from .rules import BLACKJACK_PAYOUT, CARD_PROB, DEALER_STAND, hard_and_ace, total

BUST = 5  # index of "dealer busts" in a dealer distribution; indexes 0..4 are the totals 17..21
DEALER_TOTALS = tuple(range(DEALER_STAND, 22))
UPCARDS = (2, 3, 4, 5, 6, 7, 8, 9, 10, 1)  # column order of the strategy table; 1 is the ace
ACTION_LETTERS = {"stand": "S", "hit": "H", "double": "D", "split": "P"}
MIN_DECISIVE_MARGIN = 0.01  # cells whose best and second-best action differ by less than this are near ties


def _one_hot(index: int) -> tuple[float, ...]:
    out = [0.0] * (BUST + 1)
    out[index] = 1.0
    return tuple(out)


@cache
def dealer_from(hard: int, ace: bool) -> tuple[float, ...]:
    """Probability of each final dealer outcome from a dealer hand, [17, 18, 19, 20, 21, bust].

    Recursion on the next card. The hard total strictly grows on every draw, so the recursion ends.
    """
    if hard > 21:
        return _one_hot(BUST)
    best = total(hard, ace)
    if best >= DEALER_STAND:  # S17: soft 17 stands too
        return _one_hot(best - DEALER_STAND)
    out = [0.0] * (BUST + 1)
    for value, p in CARD_PROB.items():
        for i, q in enumerate(dealer_from(hard + value, ace or value == 1)):
            out[i] += p * q
    return tuple(out)


def dealer_outcomes(upcard: int) -> tuple[float, tuple[float, ...]]:
    """(probability of a dealer blackjack, final-outcome distribution given no blackjack) for an upcard value.

    The hole card that would make blackjack is 10 under an ace and an ace under a ten. Those hole cards are
    removed and the rest renormalised: the player's decisions are made only in the world where the peek
    found nothing. Upcards 2..9 cannot make blackjack, so nothing is removed.
    """
    bj_hole = {1: 10, 10: 1}.get(upcard)
    p_bj = CARD_PROB[bj_hole] if bj_hole is not None else 0.0
    dist = [0.0] * (BUST + 1)
    for hole, p in CARD_PROB.items():
        if hole == bj_hole:
            continue
        for i, q in enumerate(dealer_from(upcard + hole, upcard == 1 or hole == 1)):
            dist[i] += p * q / (1 - p_bj)
    return p_bj, tuple(dist)


def stand_value(player_total: int, dist: tuple[float, ...]) -> float:
    """Expected return per unit bet for standing on `player_total` against a dealer distribution.

    A dealer bust pays 1. Otherwise compare totals: +1 for each dealer total the player beats, -1 for each
    total that beats the player, nothing on a tie.
    """
    ev = dist[BUST]
    for i, dealer_total in enumerate(DEALER_TOTALS):
        if player_total > dealer_total:
            ev += dist[i]
        elif player_total < dealer_total:
            ev -= dist[i]
    return ev


def table_rows() -> list[tuple[str, int]]:
    """Rows of the strategy table as (kind, total): hard totals, then soft totals, then pairs.

    Hard 20 has no two-card form without a pair (only 10+10), so it is covered by the 10,10 pair row. Hard 4 is
    likewise only 2,2. For pairs, the total is the card value of the pair (1 = aces, 10 = any ten-card pair).
    """
    return ([("hard", t) for t in range(5, 20)] + [("soft", t) for t in range(13, 21)]
            + [("pair", r) for r in range(1, 11)])


def row_label(kind: str, t: int) -> str:
    """Human label for a table row: "16", "A,7", "8,8", "A,A", "10,10"."""
    if kind == "soft":
        return f"A,{t - 11}"
    if kind == "pair":
        return "A,A" if t == 1 else ("10,10" if t == 10 else f"{t},{t}")
    return str(t)


def row_state(kind: str, t: int) -> tuple[int, bool, bool]:
    """The (hard total, ace flag, two-card flag) of a two-card table row. Pairs use their own keys."""
    if kind == "soft":
        return t - 10, True, True
    return t, False, True


def best(values: dict[str, float]) -> str:
    """Action with the highest value. Ties go to the earlier action, so stand wins a tie with hit."""
    return max(values, key=values.get)


class Solver:
    """Exact values for every player state against every dealer upcard, in the infinite-deck model.

    Values are conditional on "no dealer blackjack" unless a method says otherwise; `hand_values` and
    `cell_values(peek=True)` add the peek's effect so the numbers are what the player actually expects.
    """

    def __init__(self) -> None:
        self._dealer = {u: dealer_outcomes(u) for u in UPCARDS}
        self._states: dict[tuple[int, bool, bool, int], dict[str, float]] = {}
        self._pairs: dict[tuple[int, int], dict[str, float]] = {}

    # Conditional values, no peek.

    def _state(self, hard: int, ace: bool, two: bool, u: int) -> dict[str, float]:
        """Values of stand, hit, and (for two cards) double, for a hand that cannot split.

        Only legal actions appear. Cached by state: the recursion touches each state once.
        """
        key = (hard, ace, two, u)
        cached = self._states.get(key)
        if cached is not None:
            return cached
        dist = self._dealer[u][1]
        values = {"stand": stand_value(total(hard, ace), dist)}
        # Hit: a card that takes the hard total past 21 busts the hand (-1). Any other card leaves the
        # player in a smaller state with a larger hard total, whose best action is already known.
        values["hit"] = sum(p * (-1.0 if hard + v > 21 else self._best(hard + v, ace or v == 1, False, u))
                            for v, p in CARD_PROB.items())
        if two:
            # Double: the bet doubles, exactly one card arrives, and the hand must stand afterwards.
            values["double"] = 2 * sum(
                p * (-1.0 if hard + v > 21 else stand_value(total(hard + v, ace or v == 1), dist))
                for v, p in CARD_PROB.items())
        self._states[key] = values
        return values

    def _best(self, hard: int, ace: bool, two: bool, u: int) -> float:
        return max(self._state(hard, ace, two, u).values())

    def _pair(self, r: int, u: int) -> dict[str, float]:
        """Values of a two-card pair of value r: the non-split actions plus split."""
        key = (r, u)
        cached = self._pairs.get(key)
        if cached is not None:
            return cached
        values = dict(self._state(2 * r, r == 1, True, u))
        dist = self._dealer[u][1]
        second = 0.0
        for x, p in CARD_PROB.items():
            if r == 1:
                # Split aces take one card each and cannot act again: stand on the soft total.
                sub = stand_value(total(1 + x, True), dist)
            else:
                # A non-ace split hand is a fresh two-card hand, but it cannot split again (one split).
                sub = self._best(r + x, x == 1, True, u)
            second += p * sub
        values["split"] = 2 * second  # two hands, each worth one unit of the opening bet
        self._pairs[key] = values
        return values

    def _adjust(self, cond: dict[str, float], u: int) -> dict[str, float]:
        """Add the dealer's blackjack peek: the player loses 1 to a dealer blackjack, and no action helps.

        Each conditional value v becomes -p_bj + (1 - p_bj) * v. The map is increasing, so the best action
        is the same with or without the peek.
        """
        p_bj = self._dealer[u][0]
        return {a: -p_bj + (1 - p_bj) * v for a, v in cond.items()}

    # Public API.

    def cell_values(self, kind: str, t: int, u: int, *, peek: bool = True) -> dict[str, float]:
        """Action values for a strategy-table row against upcard u (see `table_rows`)."""
        if kind == "pair":
            cond = self._pair(t, u)
        else:
            cond = self._state(*row_state(kind, t), u)
        return self._adjust(cond, u) if peek else dict(cond)

    def hand_values(self, values: tuple[int, ...] | list[int], u: int, *, from_split: bool = False,
                    split_aces: bool = False) -> dict[str, float]:
        """Expected return per unit bet for each legal action of a specific hand, peek included.

        `values` are card values (1..10). A two-card hand that is a natural pays 3:2 and has no decisions.
        A hand from a split cannot split again; split aces stand on their one card.
        """
        values = tuple(values)
        hard, ace = hard_and_ace(values)
        if total(hard, ace) > 21:
            raise ValueError("the hand is over 21")
        p_bj, dist = self._dealer[u]
        if len(values) == 2 and sorted(values) == [1, 10] and not from_split:
            # A natural wins 3:2 unless the dealer's peek shows a blackjack, which pushes it.
            return {"stand": (1 - p_bj) * BLACKJACK_PAYOUT}
        if split_aces:
            cond = {"stand": stand_value(total(hard, ace), dist)}
        elif len(values) == 2 and values[0] == values[1] and not from_split:
            cond = self._pair(values[0], u)
        else:
            cond = dict(self._state(hard, ace, len(values) == 2, u))
        return self._adjust(cond, u)

    def house_edge(self) -> float:
        """Expected loss per unit bet for optimal play from a fresh deal (positive means the house wins).

        Averages the best action's value over every first two cards and every upcard, with the upcard and
        both cards drawn independently from the infinite deck.
        """
        ev = 0.0
        for u, pu in CARD_PROB.items():
            for c1, p1 in CARD_PROB.items():
                for c2, p2 in CARD_PROB.items():
                    ev += pu * p1 * p2 * max(self.hand_values((c1, c2), u).values())
        return -ev

    def strategy(self) -> dict:
        """The full basic-strategy table: every row and upcard, with the best action and every action's EV."""
        rows = []
        for kind, t in table_rows():
            cells = []
            for u in UPCARDS:
                evs = self.cell_values(kind, t, u)
                action = best(evs)
                cells.append({"upcard": u, "action": action, "ev": evs[action], "evs": evs})
            rows.append({"kind": kind, "total": t, "label": row_label(kind, t), "cells": cells})
        return {"upcards": list(UPCARDS), "rows": rows, "house_edge": self.house_edge()}


@lru_cache(maxsize=1)
def get_solver() -> Solver:
    """The shared solver. Building it is cheap, but the memo tables are worth keeping."""
    return Solver()


@lru_cache(maxsize=200_000)
def basic_action(values: tuple[int, ...], u: int, from_split: bool, split_aces: bool) -> str:
    """Best action for a hand, from the exact table. Used by the simulation and the terminal hint."""
    return best(get_solver().hand_values(values, u, from_split=from_split, split_aces=split_aces))
