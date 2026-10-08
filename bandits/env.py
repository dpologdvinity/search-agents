"""The casino: seeded slot machines, with every outcome rolled in advance.

Three kinds of machine row:

  bernoulli  each machine pays 1 with a hidden probability p_i, else 0.
  gaussian   each machine pays mu_i + SIGMA * noise, a real-valued reward.
  drifting   Bernoulli machines whose probabilities are redrawn every DRIFT_PERIOD pulls,
             so yesterday's best machine can stop being the best.

The machines come from one generator (`Rng(seed)`) and the outcomes from another
(`Rng(seed + REWARD_OFFSET)`). The outcome table is rolled once, for every pull and every machine,
so all agents face the same luck: when two agents pull the same machine at the same time they get
the same reward. That makes the comparisons fair, and it makes a seed reproducible on the command
line and on the page.

Regret is measured on the means, not the luck: at each pull the regret is best_mean - mean of the
machine that was pulled. That is the expected regret, which has less noise than the realised one.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .rng import Rng

KINDS = ("bernoulli", "gaussian", "drifting")
SIGMA = 0.2  # noise scale of the Gaussian machines
DRIFT_PERIOD = 500  # pulls between redraws of the drifting machines
GAP = 0.05  # the best machine must lead the second best by at least this, so it is identifiable
LOW, HIGH = 0.1, 0.9  # machine means are drawn uniformly from [LOW, HIGH]
REWARD_OFFSET = 1_000_003  # keeps the outcome stream apart from the machine stream


def draw_means(rng: Rng, k: int) -> list[float]:
    """k machine means, redrawn until the best leads the second best by GAP."""
    while True:
        means = [LOW + (HIGH - LOW) * rng.uniform() for _ in range(k)]
        ordered = sorted(means)
        if ordered[-1] - ordered[-2] >= GAP:
            return means


@dataclass(frozen=True)
class Machines:
    """The hidden part of the casino: the mean schedule of each segment, and how long a segment lasts."""

    kind: str
    k: int
    seed: int
    horizon: int
    period: int  # pulls per segment; the whole horizon for stationary kinds
    schedule: tuple[tuple[float, ...], ...]  # one tuple of k means per segment

    def mean_at(self, t: int) -> tuple[float, ...]:
        """The k means in force at pull t (0-based)."""
        return self.schedule[min(t // self.period, len(self.schedule) - 1)]


def make_machines(kind: str, k: int, horizon: int, seed: int) -> Machines:
    """Draw the machine schedule for a seed. The first segment's means do not depend on horizon."""
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {', '.join(KINDS)}")
    if not 2 <= k <= 20:
        raise ValueError("a casino has between 2 and 20 machines")
    if horizon < 1:
        raise ValueError("horizon must be at least 1 pull")
    period = DRIFT_PERIOD if kind == "drifting" else horizon
    segments = -(-horizon // period)  # ceiling division
    rng = Rng(seed)
    schedule = tuple(tuple(draw_means(rng, k)) for _ in range(segments))
    return Machines(kind, k, seed, horizon, period, schedule)


class Environment:
    """A casino with its outcome table rolled: `reward(t, arm)` is the same every time it is asked."""

    def __init__(self, kind: str, k: int, horizon: int, seed: int):
        self.machines = make_machines(kind, k, horizon, seed)
        self.kind = kind
        self.k = k
        self.horizon = horizon
        self.seed = seed
        self.sigma = SIGMA if kind == "gaussian" else 1.0
        # One uniform (Bernoulli) or normal (Gaussian) per pull and machine, rolled in pull-major order.
        # Using a uniform u with "pays when u < p" gives a Bernoulli(p) reward and keeps the table
        # the same for any mean schedule.
        rng = Rng(seed + REWARD_OFFSET)
        if kind == "gaussian":
            self._luck = [rng.normal() for _ in range(horizon * k)]
        else:
            self._luck = [rng.uniform() for _ in range(horizon * k)]

    def mean(self, t: int) -> tuple[float, ...]:
        return self.machines.mean_at(t)

    def best(self, t: int) -> float:
        return max(self.mean(t))

    def best_arm(self, t: int) -> int:
        """The machine with the highest mean at pull t (the first one, if two tie)."""
        m = self.mean(t)
        return m.index(max(m))

    def reward(self, t: int, arm: int) -> float:
        """What machine `arm` pays on pull t."""
        luck = self._luck[t * self.k + arm]
        p = self.mean(t)[arm]
        if self.kind == "gaussian":
            return p + self.sigma * luck
        return 1.0 if luck < p else 0.0


def bernoulli_kl(p: float, q: float) -> float:
    """KL divergence between Bernoulli(p) and Bernoulli(q), in nats."""
    return p * math.log(p / q) + (1.0 - p) * math.log((1.0 - p) / (1.0 - q))


def lai_robbins_rate(means: tuple[float, ...], kind: str, sigma: float = SIGMA) -> float:
    """The constant c in the Lai-Robbins lower bound, regret >= c * ln(T), for fixed machines.

    Lai and Robbins (1985): every consistent policy has regret at least sum over the suboptimal
    arms of gap_i / KL(p_i || p*) times ln T, asymptotically. For Bernoulli arms KL is the
    Bernoulli divergence. For Gaussian arms with known variance sigma^2, KL is gap^2 / (2 sigma^2),
    so each arm contributes 2 sigma^2 / gap. The bound is asymptotic: it is a reference shape,
    not a guarantee at a particular T.
    """
    if kind == "drifting":
        raise ValueError("the Lai-Robbins bound is for fixed machines; drifting machines have no single bound")
    best = max(means)
    rate = 0.0
    for p in means:
        gap = best - p
        if gap <= 0:
            continue
        kl = bernoulli_kl(p, best) if kind == "bernoulli" else gap * gap / (2.0 * sigma * sigma)
        rate += gap / kl
    return rate
