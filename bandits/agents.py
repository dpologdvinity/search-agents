"""Bandit agents: each one decides which slot machine to pull, from the rewards it has seen so far.

Every agent has the same loop, so the simulator can drive any of them:

    arm = agent.choose()        # uses only past rewards, and its own randomness
    agent.update(arm, reward)   # the reward is the only feedback

Along the way `choose` fills in two things for display: `scores`, the numbers the decision was
based on (means, UCB indices, posterior samples, or EXP3 probabilities), and `reason`, a sentence
for the watch mode and the page. Ties go to the lowest machine number, so runs are deterministic
for a seed.

The exploration problem: a machine's mean is known only from its own pulls. Pull a machine you
already know to pay well and you collect reward (exploit), but a machine you never tried might be
better (explore). Each agent resolves that trade-off differently:

  greedy        always the best average so far. Locks onto whichever machine looked good early.
  epsilon       the best average, except with probability epsilon a uniformly random machine.
  epsilon-decay as epsilon, but epsilon_t = min(1, K / t), so exploration fades as t grows.
  UCB1          the best optimistic bound: mean + sqrt(2 ln t / n). Machines pulled rarely get a
                large bonus, so the bonus shrinks as evidence accumulates (Auer et al. 2002).
  Thompson      sample each machine's mean from its Beta posterior and pull the best sample.
                Probability matching: the chance of a pull is the chance it is the best.
  EXP3          adversarial: exponential weights over machines, mixed with uniform exploration.
                Makes no stochastic assumption, so it copes with drifting machines.
  SW-UCB        UCB over only the last W pulls (sliding window). Old evidence is forgotten, so it
                follows machines that drift (Garivier and Moulines 2011).

The initial rounds are forced for the mean-based agents: each machine is tried once before any mean
is used, which avoids dividing by a zero count.
"""

from __future__ import annotations

import math
from collections import deque

from .rng import Rng


def argmax(xs: list[float]) -> int:
    """Index of the largest value; the lowest index wins a tie."""
    best = 0
    for i in range(1, len(xs)):
        if xs[i] > xs[best]:
            best = i
    return best


class Agent:
    """Base class: counts, sums, the pull counter, and the display fields."""

    key = ""
    label = ""
    blurb = ""

    def __init__(self, k: int, horizon: int, rng: Rng, sigma: float = 1.0, bernoulli: bool = True):
        self.k = k
        self.horizon = horizon
        self.rng = rng
        self.sigma = sigma  # reward noise scale, used by the UCB bonus for Gaussian machines
        self.bernoulli = bernoulli  # 0/1 rewards (Beta posteriors) or real rewards (Gaussian posteriors)
        self.counts = [0] * k
        self.sums = [0.0] * k
        self.t = 0  # pulls so far
        self.scores = [0.0] * k
        self.reason = ""

    def means(self) -> list[float]:
        return [s / n if n else 0.0 for s, n in zip(self.sums, self.counts)]

    def choose(self) -> int:
        raise NotImplementedError

    def update(self, arm: int, reward: float) -> None:
        self.counts[arm] += 1
        self.sums[arm] += reward
        self.t += 1

    def _forced(self) -> int:
        """During the first K pulls, try each machine once. Returns the machine to try."""
        self.scores = self.means()
        self.reason = f"first round: try machine {self.t + 1} once, since nothing is known yet"
        return self.t


class Greedy(Agent):
    key = "greedy"
    label = "greedy"
    blurb = "Always the best average so far. Locks on early, and never looks back."

    def choose(self) -> int:
        if self.t < self.k:
            return self._forced()
        m = self.means()
        self.scores = m
        arm = argmax(m)
        self.reason = f"exploit: machine {arm + 1} has the best average so far ({m[arm]:.2f})"
        return arm


class EpsilonGreedy(Agent):
    key = "eps"
    label = "epsilon-greedy 0.1"
    blurb = "The best average, except one pull in ten goes to a random machine."
    decay = False
    epsilon = 0.1

    def epsilon_now(self) -> float:
        # epsilon_t = min(1, K / t): explore a lot early, then less and less.
        return min(1.0, self.k / self.t) if self.decay else self.epsilon

    def choose(self) -> int:
        if self.t < self.k:
            return self._forced()
        m = self.means()
        self.scores = m
        eps = self.epsilon_now()
        # Two draws: one decides explore or exploit, one picks the random machine if exploring.
        if self.rng.uniform() < eps:
            arm = self.rng.index(self.k)
            self.reason = f"explore: a random machine ({arm + 1}), with epsilon {eps:.3f}"
        else:
            arm = argmax(m)
            self.reason = f"exploit: machine {arm + 1} has the best average ({m[arm]:.2f}); epsilon {eps:.3f}"
        return arm


class EpsilonDecay(EpsilonGreedy):
    key = "eps_decay"
    label = "epsilon-greedy decaying"
    blurb = "Epsilon-greedy with epsilon_t = min(1, K/t): exploration fades over time."
    decay = True


class UCB1(Agent):
    key = "ucb1"
    label = "UCB1"
    blurb = "Optimism in the face of uncertainty: mean plus a bonus that shrinks with each pull."

    def choose(self) -> int:
        if self.t < self.k:
            return self._forced()
        # The bonus sqrt(2 ln t / n) is large for machines pulled rarely. For Gaussian machines it is
        # scaled by the noise sigma, the usual sub-Gaussian form of the bound.
        t = self.t
        m = self.means()
        idx = [m[i] + self.sigma * math.sqrt(2.0 * math.log(t) / self.counts[i]) for i in range(self.k)]
        self.scores = idx
        arm = argmax(idx)
        bonus = idx[arm] - m[arm]
        self.reason = (f"highest optimistic bound: machine {arm + 1} has mean {m[arm]:.2f} "
                       f"plus a bonus of {bonus:.2f} for its {self.counts[arm]} pulls")
        return arm


class Thompson(Agent):
    key = "thompson"
    label = "Thompson sampling"
    blurb = "Sample each machine's payout from its posterior, then pull the best sample."

    def posterior(self) -> list[tuple[float, float]]:
        """Per machine: (alpha, beta) of its Beta posterior for Bernoulli rewards, or (mean, sd) for Gaussian.

        With a uniform Beta(1, 1) prior and 0/1 rewards, the posterior after s wins and f losses is
        Beta(1 + s, 1 + f). For Gaussian rewards with known sigma and a N(0, 1) prior, the posterior of
        the mean is normal with precision 1 + n / sigma^2 and mean (sum / sigma^2) / precision.
        """
        out = []
        for n, s in zip(self.counts, self.sums):
            if self.bernoulli:
                out.append((1.0 + s, 1.0 + n - s))
            else:
                prec = 1.0 + n / (self.sigma * self.sigma)
                out.append(((s / (self.sigma * self.sigma)) / prec, 1.0 / math.sqrt(prec)))
        return out

    def choose(self) -> int:
        post = self.posterior()
        samples = []
        for a, b in post:
            if self.bernoulli:
                samples.append(self.rng.beta(a, b))
            else:
                samples.append(a + b * self.rng.normal())  # a = posterior mean, b = posterior sd
        self.scores = samples
        arm = argmax(samples)
        a, b = post[arm]
        if self.bernoulli:
            self.reason = (f"sampled {samples[arm]:.2f} from machine {arm + 1}'s Beta({a:.0f}, {b:.0f}) "
                           "posterior, the best sample this round")
        else:
            self.reason = (f"sampled {samples[arm]:.2f} from machine {arm + 1}'s posterior "
                           f"(mean {a:.2f}, sd {b:.2f}), the best sample this round")
        return arm


class EXP3(Agent):
    key = "exp3"
    label = "EXP3"
    blurb = "Exponential weights with forced exploration. Assumes nothing about the payouts."

    def __init__(self, k: int, horizon: int, rng: Rng, sigma: float = 1.0, bernoulli: bool = True,
                 gamma: float | None = None):
        super().__init__(k, horizon, rng, sigma, bernoulli)
        # Auer et al.'s tuning for a known horizon: gamma = sqrt(K ln K / ((e - 1) T)), capped at 1.
        self.gamma = gamma if gamma is not None else min(1.0, math.sqrt(k * math.log(k) / ((math.e - 1) * horizon)))
        self.weights = [1.0] * k
        self._probs = [1.0 / k] * k

    def probabilities(self) -> list[float]:
        # Mix the weight-proportional distribution with uniform, so every machine keeps probability >= gamma / K.
        total = sum(self.weights)
        return [(1.0 - self.gamma) * w / total + self.gamma / self.k for w in self.weights]

    def choose(self) -> int:
        probs = self.probabilities()
        self._probs = probs
        self.scores = probs
        u = self.rng.uniform()
        cum = 0.0
        arm = self.k - 1
        for i, p in enumerate(probs):
            cum += p
            if u < cum:
                arm = i
                break
        self.reason = f"sample machine {arm + 1} with probability {probs[arm]:.2f}"
        return arm

    def update(self, arm: int, reward: float) -> None:
        super().update(arm, reward)
        # Importance weighting: divide the reward by the probability of pulling this machine, so the
        # estimate is unbiased. Weights are rescaled by their maximum so they never overflow.
        estimate = reward / self._probs[arm]
        self.weights[arm] *= math.exp(self.gamma * estimate / self.k)
        top = max(self.weights)
        self.weights = [w / top for w in self.weights]


class SlidingWindowUCB(Agent):
    key = "sw_ucb"
    label = "sliding-window UCB"
    blurb = "UCB1 over only the last 200 pulls, so old evidence about a drifted machine expires."
    window = 200
    xi = 0.6

    def __init__(self, k: int, horizon: int, rng: Rng, sigma: float = 1.0, bernoulli: bool = True,
                 window: int | None = None):
        super().__init__(k, horizon, rng, sigma, bernoulli)
        self.window = window if window is not None else self.window
        self._recent: deque[tuple[int, float]] = deque()
        self.wcounts = [0] * k
        self.wsums = [0.0] * k

    def update(self, arm: int, reward: float) -> None:
        super().update(arm, reward)
        self._recent.append((arm, reward))
        self.wcounts[arm] += 1
        self.wsums[arm] += reward
        if len(self._recent) > self.window:
            old_arm, old_reward = self._recent.popleft()
            self.wcounts[old_arm] -= 1
            self.wsums[old_arm] -= old_reward

    def choose(self) -> int:
        for i in range(self.k):
            if self.wcounts[i] == 0:
                self.scores = [0.0] * self.k
                self.reason = f"machine {i + 1} has not paid out in the last {self.window} pulls: look again"
                return i
        # Inside the window: mean + sigma * sqrt(xi * ln(min(t, W)) / n). xi is smaller than UCB1's 2,
        # because the window already caps how long evidence lasts; xi = 0.6 is a fixed choice, not tuned.
        log_t = math.log(min(self.t, self.window))
        idx = [self.wsums[i] / self.wcounts[i] + self.sigma * math.sqrt(self.xi * log_t / self.wcounts[i])
               for i in range(self.k)]
        self.scores = idx
        arm = argmax(idx)
        self.reason = (f"highest window index: machine {arm + 1} averages {self.wsums[arm] / self.wcounts[arm]:.2f} "
                       f"over its last {self.wcounts[arm]} pulls in the window")
        return arm


AGENTS: dict[str, type[Agent]] = {
    cls.key: cls for cls in (Greedy, EpsilonGreedy, EpsilonDecay, UCB1, Thompson, EXP3, SlidingWindowUCB)
}

# The agents that run on each kind of casino, in the order the tables list them.
LINEUP: dict[str, tuple[str, ...]] = {
    "bernoulli": ("greedy", "eps", "eps_decay", "ucb1", "thompson"),
    "gaussian": ("greedy", "eps", "eps_decay", "ucb1", "thompson"),
    "drifting": ("greedy", "eps", "eps_decay", "ucb1", "thompson", "exp3", "sw_ucb"),
}


def make_agent(key: str, k: int, horizon: int, rng: Rng, sigma: float = 1.0, bernoulli: bool = True) -> Agent:
    if key not in AGENTS:
        raise ValueError(f"unknown agent {key!r}; choose from {', '.join(AGENTS)}")
    return AGENTS[key](k, horizon, rng, sigma=sigma, bernoulli=bernoulli)  # every class takes these by keyword
