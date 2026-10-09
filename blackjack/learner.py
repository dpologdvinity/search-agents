"""Monte Carlo control: learn blackjack action values from simulated hands, without the model.

The learner never reads the probability tables or the dynamic program. It plays simulated hands against an
infinite deck (the same model the exact solver assumes, so the two are comparable), records every decision,
and when a round settles it credits each decision with what the hands it affected actually paid:

- Exploring starts. Each hand begins from a uniformly chosen cell of the strategy table: a two-card hand of
  that row and an upcard. Without this, rare starts such as a two-card 5 against a 6 would be seen only a few
  times per thousand hands. Starting a hand in a state does not bias that state's action values, because a
  value is the payoff from that state onward.
- Shared statistics. Stand, hit, and double are valued by (upcard, hard total, ace flag) alone: the future
  after a stand or hit does not depend on how many cards the hand already holds. So every hand that passes
  through a state teaches the same estimates; double is legal only on two cards. Split has its own estimate
  for each pair.
- Every-visit Monte Carlo with sample averages: Q(s, a) is the mean payoff of all hands in which action a
  was taken in state s. There is no discounting; the only reward arrives when the hand settles.
- Epsilon-greedy exploration: a hand's first decision picks a legal action uniformly (an exploring start);
  later ones take a random legal action with probability epsilon, else the best Q (untried actions count 0).
- A split credits its decision with the sum of both hands' payoffs, because the split is worth both hands.

Statistics live in flat lists indexed by state, not nested dicts: a hand is a few dozen array operations, so
the learner can afford tens of thousands of hands in a request.

`agreement` is the share of "decisive" table cells where the learned greedy action is the exact best action.
A cell is decisive when the exact best beats the runner-up by a clear margin; near ties can be learned either
way and would only add noise. `loss` is the exact expected value given up by following the learned greedy
policy, averaged over every table cell (an untried cell counts as its worst action).
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from functools import lru_cache

from .rules import DEALER_STAND, total
from .solver import ACTION_LETTERS, MIN_DECISIVE_MARGIN, UPCARDS, get_solver, row_state, table_rows

# One entry per rank of a 13-rank deck: the ace and 2..9 once each, the four ten-valued ranks as 10.
_RANKS = tuple(v for v in range(1, 11) for _ in range(4 if v == 10 else 1))
EPSILON_START = 0.2
EPSILON_MIN = 0.01
EPSILON_HALF_LIFE = 2000  # hands: exploration halves every this many hands

# Action codes and the names used in the API: stand, hit, and double are state actions; split is per pair.
STAND, HIT, DOUBLE, SPLIT = 0, 1, 2, 3
CODE_NAME = ("stand", "hit", "double", "split")

# Flat storage. A state is (upcard 1..10, hard total 0..21, ace flag) and owns three slots (stand, hit, double).
# Pairs own one slot each for split, placed after all state slots.
_HARD_SLOTS = 22
_STATE_SLOTS = 10 * _HARD_SLOTS * 2
_PAIR_BASE = _STATE_SLOTS * 3
_SIZE = _PAIR_BASE + 100

Option = tuple[int, int]  # (action code, flat index of its statistics slot)


def _options(u: int, hard: int, ace: bool, n: int, pair_value: int) -> list[Option]:
    """Legal actions of a decision with their statistics slots, following the rules.

    A pair may split; double needs exactly two cards; stand and hit are always legal.
    """
    base = (((u - 1) * _HARD_SLOTS + hard) * 2 + int(ace)) * 3
    if pair_value:
        return [(SPLIT, _PAIR_BASE + (u - 1) * 10 + pair_value - 1), (STAND, base), (HIT, base + 1),
                (DOUBLE, base + 2)]
    if n == 2:
        return [(STAND, base), (HIT, base + 1), (DOUBLE, base + 2)]
    return [(STAND, base), (HIT, base + 1)]


@lru_cache(maxsize=1)
def cell_specs() -> tuple[tuple[int, int, bool, int, int], ...]:
    """(upcard, hard total, ace flag, card count, pair value) for every strategy-table cell, in table order."""
    out = []
    for kind, t in table_rows():
        for u in UPCARDS:
            if kind == "pair":
                out.append((u, 2 * t, t == 1, 2, t))
            else:
                hard, ace, _ = row_state(kind, t)
                out.append((u, hard, ace, 2, 0))
    return tuple(out)


@lru_cache(maxsize=1)
def start_hands() -> tuple[tuple[int, tuple[tuple[int, ...], ...]], ...]:
    """Exploring-start options per table cell, in table order: (upcard, two-card hands that realise the row).

    Non-pair hands are listed with the smaller card first. Soft rows are (ace, x). Pair rows are (r, r).
    """
    out = []
    for kind, t in table_rows():
        if kind == "pair":
            combos = ((t, t),)
        elif kind == "soft":
            combos = ((1, t - 11),)
        else:
            combos = tuple((a, t - a) for a in range(2, 11) if a < t - a <= 10)
        for u in UPCARDS:
            out.append((u, combos))
    return tuple(out)


@lru_cache(maxsize=1)
def exact_cells() -> tuple[dict[str, float], ...]:
    """Exact action values per table cell, in the same order as `cell_specs`.

    Taken without the dealer-blackjack peek: the learner's rounds already exclude dealer blackjacks, so the
    comparable quantity is the value conditional on no dealer blackjack.
    """
    solver = get_solver()
    return tuple(solver.cell_values(kind, t, u, peek=False) for kind, t in table_rows() for u in UPCARDS)


def exact_actions() -> str:
    """Exact best action letter for every table cell (the target that `agreement` compares against)."""
    return "".join(ACTION_LETTERS[max(values, key=values.get)] for values in exact_cells())


class _Hand:
    """One player hand in a simulated round, with the hard total and ace flag kept incrementally."""

    __slots__ = ("hard", "ace", "n", "pair_value", "from_split", "split_aces", "bet", "done", "payoff")

    def __init__(self, cards: list[int], from_split: bool = False) -> None:
        self.hard = sum(cards)
        self.ace = 1 in cards
        self.n = len(cards)
        # Only the opening two-card hand can be a pair that splits (one split per round).
        self.pair_value = cards[0] if not from_split and self.n == 2 and cards[0] == cards[1] else 0
        self.from_split = from_split
        self.split_aces = from_split and cards[0] == 1  # split aces take one card and stop
        self.bet = 1.0
        self.payoff = 0.0
        self.done = self.split_aces or self.best == 21

    @property
    def best(self) -> int:
        return total(self.hard, self.ace)

    def add(self, value: int) -> None:
        """Draw one more card. A hand with three or more cards is no longer a pair or a two-card hand."""
        self.hard += value
        self.ace = self.ace or value == 1
        self.n += 1
        self.pair_value = 0


class MonteCarloControl:
    """Learns Q(state, action) from simulated hands. `episodes` counts the hands played so far."""

    def __init__(self, seed: int = 0) -> None:
        self._rng = random.Random(seed)
        self._n = [0] * _SIZE  # visits per statistics slot
        self._q = [0.0] * _SIZE  # mean payoff per statistics slot
        self.episodes = 0

    def epsilon(self) -> float:
        """Exploration START / (1 + n / HALF_LIFE) after n hands, floored: half at HALF_LIFE hands, then about 1/n."""
        return max(EPSILON_MIN, EPSILON_START / (1 + self.episodes / EPSILON_HALF_LIFE))

    def run(self, hands: int) -> None:
        """Play and learn from `hands` simulated rounds."""
        for _ in range(hands):
            self._episode()
            self.episodes += 1

    def greedy(self, opts: list[Option]) -> int | None:
        """Code of the best-valued option among those already tried (None if none has been tried)."""
        best_code, best_q = None, 0.0
        for code, idx in opts:
            if self._n[idx] and (best_code is None or self._q[idx] > best_q):
                best_code, best_q = code, self._q[idx]
        return best_code

    def _choose(self, opts: list[Option], eps: float, explore: bool) -> Option:
        """Pick an action. `explore` (the first decision of a hand) picks uniformly, which is the exploring
        start over actions: each legal action at each table cell is tried equally often. Other decisions
        are epsilon-greedy on the current estimates.
        """
        rng = self._rng
        if explore or rng.random() < eps:
            return rng.choice(opts)
        best = opts[0]
        for o in opts[1:]:
            if self._q[o[1]] > self._q[best[1]]:
                best = o
        return best

    def _credit(self, idx: int, payoff: float) -> None:
        """Incremental sample average, Q <- Q + (payoff - Q) / n, which equals the mean of all visits."""
        n = self._n[idx] + 1
        self._n[idx] = n
        self._q[idx] += (payoff - self._q[idx]) / n

    def _episode(self) -> None:
        """Play one round from an exploring start, then credit each decision with the payoff it affected."""
        rng = self._rng
        eps = self.epsilon()
        starts = start_hands()
        upcard, combos = starts[rng.randrange(len(starts))]
        hole = _RANKS[rng.randrange(13)]
        # A dealer blackjack ends the round before the player acts, so there is no decision to learn from.
        if (upcard, hole) in ((1, 10), (10, 1)):
            return

        hands = [_Hand(list(combos[rng.randrange(len(combos))]))]
        queue = list(hands)  # hands still to play, in order
        log: list[tuple[int, tuple[_Hand, ...]]] = []  # (statistics slot, hands the decision affected)
        first = True  # only the first decision of a round is an exploring start
        while queue:
            hand = queue.pop(0)
            while not hand.done:
                opts = _options(upcard, hand.hard, hand.ace, hand.n, hand.pair_value)
                code, idx = self._choose(opts, eps, first)
                first = False
                log.append((idx, (hand,)))
                if code == STAND:
                    hand.done = True
                elif code == HIT:
                    hand.add(_RANKS[rng.randrange(13)])
                    hand.done = hand.best >= 21
                elif code == DOUBLE:
                    hand.bet = 2.0
                    hand.add(_RANKS[rng.randrange(13)])
                    hand.done = True
                else:  # split: the pair becomes two hands, each keeping one card of the pair
                    r = hand.pair_value
                    left = _Hand([r, _RANKS[rng.randrange(13)]], from_split=True)
                    right = _Hand([r, _RANKS[rng.randrange(13)]], from_split=True)
                    log[-1] = (idx, (left, right))  # the split's credit is both hands' payoffs
                    hands = [h for h in hands if h is not hand] + [left, right]
                    queue[0:0] = [left, right]
                    hand.done = True

        # The dealer plays once, and only if some hand is still live: draw to 17 or more (S17, soft 17 stands).
        dealer_hard, dealer_ace = upcard + hole, upcard == 1 or hole == 1
        if any(h.hard <= 21 for h in hands):
            while total(dealer_hard, dealer_ace) < DEALER_STAND:
                card = _RANKS[rng.randrange(13)]
                dealer_hard += card
                dealer_ace = dealer_ace or card == 1
        dealer_total = total(dealer_hard, dealer_ace)
        dealer_bust = dealer_hard > 21
        for h in hands:
            if h.hard > 21:
                h.payoff = -h.bet
            elif dealer_bust or h.best > dealer_total:
                h.payoff = h.bet
            elif h.best < dealer_total:
                h.payoff = -h.bet
            else:
                h.payoff = 0.0
        for idx, affected in log:
            self._credit(idx, sum(h.payoff for h in affected))


def agreement(mc: MonteCarloControl) -> tuple[float, int, float, str]:
    """Learned policy against the exact table.

    Returns (agreement on decisive cells, number of decisive cells, mean exact value lost by the greedy
    action over all cells, greedy action letters for every cell in table order with "?" for untried cells).
    """
    letters = []
    matches = decisive = 0
    loss = 0.0
    for spec, exact in zip(cell_specs(), exact_cells(), strict=True):
        opts = _options(*spec)
        greedy = mc.greedy(opts)
        letters.append(ACTION_LETTERS[CODE_NAME[greedy]] if greedy is not None else "?")
        best_action = max(exact, key=exact.get)
        # An untried cell gives up the most it could (its worst action); it is a miss when decisive.
        loss += exact[best_action] - (exact[CODE_NAME[greedy]] if greedy is not None else min(exact.values()))
        ranked = sorted(exact.values(), reverse=True)
        if ranked[0] - ranked[1] < MIN_DECISIVE_MARGIN:
            continue
        decisive += 1
        matches += greedy is not None and CODE_NAME[greedy] == best_action
    cells = len(letters)
    return (matches / decisive if decisive else 0.0), decisive, loss / cells, "".join(letters)


@dataclass
class Snapshot:
    """What the learner had learned after `episodes` hands."""

    episodes: int
    agreement: float
    decisive: int
    loss: float
    actions: str  # one letter per table cell (S/H/D/P), "?" where untried


def learn(hands: int, seed: int = 0, snapshots: int = 10) -> list[Snapshot]:
    """Run Monte Carlo control for `hands` hands, snapshotting the learned policy at evenly spaced points.

    The last snapshot is always at `hands`. The API caps `hands` and `snapshots`, since this is CPU-bound.
    """
    mc = MonteCarloControl(seed)
    marks = sorted({max(1, round(hands * (i + 1) / snapshots)) for i in range(snapshots)})
    out = []
    for mark in marks:
        mc.run(mark - mc.episodes)
        share, decisive, loss, letters = agreement(mc)
        out.append(Snapshot(episodes=mc.episodes, agreement=share, decisive=decisive, loss=loss, actions=letters))
    return out
