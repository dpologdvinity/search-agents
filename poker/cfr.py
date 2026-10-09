"""Counterfactual regret minimization (CFR) and CFR+ on an explicit game tree.

The idea in one paragraph. At each information set I, keep a regret for every action: how much
better the seat would have done, summed over the iterations, by always taking that action instead of
the mix it played. Regret matching turns the positive regrets into the next strategy (play each
action in proportion to its positive regret). The average of those strategies over time converges
to a Nash equilibrium in two-player zero-sum games. The "counterfactual" part weights each history by
the opponent's and chance's reach, not the seat's own, so the regret is the change in the seat's
payoff from that information set alone.

Vanilla CFR (Zinkevich et al., 2007) keeps every regret, positive or negative, and averages all
iterations equally. CFR+ (Tammelin, 2014) floors cumulative regrets at zero after each update, which
makes the current strategy settle faster, and averages with weight t on iteration t (linear
averaging), so later, better strategies count more.

Each iteration walks the tree once for each seat. A walk for seat i updates the regrets and the
average strategy of seat i's information sets only. The updates alternate: seat 1's walk runs after
seat 0's regrets are updated, so it best-responds to seat 0's newest strategy (not a simultaneous
update from the start of the iteration).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .tree import CHANCE, TERMINAL, GameTree, exploitability


def regret_matching(regret: list[float]) -> list[float]:
    """Play each action in proportion to its positive regret; uniform when no action has positive regret."""
    pos = [r if r > 0 else 0.0 for r in regret]
    total = sum(pos)
    if total > 0:
        return [p / total for p in pos]
    return [1.0 / len(regret)] * len(regret)


@dataclass
class Solver:
    """Regrets and average-strategy sums for every information set of a tree."""

    tree: GameTree
    algo: str = "cfr+"  # "cfr" (vanilla) or "cfr+"
    t: int = 0          # iterations completed
    regret: list[list[float]] = field(default_factory=list)
    strat_sum: list[list[float]] = field(default_factory=list)

    def __post_init__(self):
        if self.algo not in ("cfr", "cfr+"):
            raise ValueError(f"unknown algorithm {self.algo!r}")
        self.regret = [[0.0] * len(a) for a in self.tree.info_actions]
        self.strat_sum = [[0.0] * len(a) for a in self.tree.info_actions]

    @property
    def plus(self) -> bool:
        return self.algo == "cfr+"

    def current(self) -> list[list[float]]:
        """The strategy each information set plays this iteration, from its regrets."""
        return [regret_matching(r) for r in self.regret]

    def average(self) -> list[list[float]]:
        """The average strategy: the normalized sum of the strategies played. This is the one that converges."""
        out = []
        for s in self.strat_sum:
            total = sum(s)
            out.append([x / total for x in s] if total > 0 else [1.0 / len(s)] * len(s))
        return out

    def iterate(self) -> None:
        """One iteration: for each seat in turn, walk the tree and update that seat's regrets.

        Alternating updates: the second seat's walk already sees the first seat's new regrets, so each
        seat best-responds to the other's most recent strategy. This is the form CFR+ is usually run in.
        """
        self.t += 1
        # CFR+ weights iteration t by t in the average. Vanilla CFR weights every iteration equally.
        weight = float(self.t) if self.plus else 1.0
        for seat in (0, 1):
            sigma = self.current()
            delta = [[0.0] * len(a) for a in self.tree.info_actions]
            self._walk(0, seat, 1.0, 1.0, sigma, delta, weight)
            for iid in range(len(delta)):
                if self.tree.info_player[iid] != seat:
                    continue
                reg = self.regret[iid]
                for j, d in enumerate(delta[iid]):
                    r = reg[j] + d
                    reg[j] = max(r, 0.0) if self.plus else r

    def _walk(self, n: int, seat: int, pi_seat: float, pi_other: float,
              sigma: list[list[float]], delta: list[list[float]], weight: float) -> float:
        """Expected payoff to `seat` at node n. pi_seat and pi_other are the reach probabilities of the
        seat and of everyone else (the opponent plus chance). Regret updates use pi_other only, and the
        average strategy uses pi_seat only, which is what makes the regret counterfactual.
        """
        tree = self.tree
        k = tree.kind[n]
        if k == TERMINAL:
            return tree.util[n] if seat == 0 else -tree.util[n]
        if k == CHANCE:
            total = 0.0
            for p, c in zip(tree.probs[n], tree.children[n]):
                total += p * self._walk(c, seat, pi_seat, pi_other * p, sigma, delta, weight)
            return total
        iid = tree.info[n]
        s = sigma[iid]
        kids = tree.children[n]
        if tree.player[n] != seat:
            # The opponent's move is part of the environment for this walk: it weights the regrets.
            total = 0.0
            for j, c in enumerate(kids):
                total += s[j] * self._walk(c, seat, pi_seat, pi_other * s[j], sigma, delta, weight)
            return total
        # The seat's own move: value of each action, then regrets relative to the mixed value.
        vals = [self._walk(c, seat, pi_seat * s[j], pi_other, sigma, delta, weight) for j, c in enumerate(kids)]
        node_value = sum(s[j] * vals[j] for j in range(len(kids)))
        d = delta[iid]
        avg = self.strat_sum[iid]
        for j in range(len(kids)):
            d[j] += pi_other * (vals[j] - node_value)
            avg[j] += weight * pi_seat * s[j]
        return node_value

    def exploitability(self) -> float:
        return exploitability(self.tree, self.average())[0]


def checkpoints(total: int, points: int = 40) -> list[int]:
    """Iteration numbers at which to measure: roughly log-spaced, always including 1 and `total`."""
    if total <= points:
        return list(range(1, total + 1))
    marks = {1, total}
    for i in range(points):
        marks.add(round(math.exp(math.log(total) * i / (points - 1))))
    return sorted(m for m in marks if 1 <= m <= total)


def train(tree: GameTree, algo: str, iterations: int, on_point=None) -> Solver:
    """Run `iterations` of CFR or CFR+. `on_point(t, exploitability)` is called at each checkpoint."""
    solver = Solver(tree, algo)
    marks = set(checkpoints(iterations))
    for t in range(1, iterations + 1):
        solver.iterate()
        if t in marks and on_point is not None:
            on_point(t, solver.exploitability())
    return solver


def table(tree: GameTree, avg: list[list[float]]) -> dict[str, dict[str, float]]:
    """The average strategy keyed by information-set key: {key: {action: probability}}."""
    return {
        tree.infosets[iid]: dict(zip(tree.info_actions[iid], avg[iid]))
        for iid in range(len(tree.infosets))
    }

