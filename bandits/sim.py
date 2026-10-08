"""Run agents on a casino and measure them: cumulative regret, the share of best-machine pulls, and bands.

The loop is the same for every agent and for a human player. `score_pulls` takes the arms a human
chose and returns the same record an agent run produces, so the CLI and the page can rank a player
against the agents on exactly the same machines.

Statistics across seeds: a run is one seed, so the mean over seeds is the estimate and its 95%
band is mean +/- 1.96 * sd / sqrt(n). The band is a normal approximation, which is fine for the
100-seed benchmark; it is not a formal simultaneous band over the whole curve.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .agents import Agent, make_agent
from .env import Environment
from .rng import Rng

AGENT_OFFSET = 2_000_029  # keeps the agent's random stream apart from the machines and outcomes


@dataclass(frozen=True)
class Run:
    """One play of a casino, pull by pull. regret[t] is cumulative after pull t (0-based)."""

    arms: list[int]
    rewards: list[float]
    regret: list[float]
    optimal: list[bool]  # True when the pull went to a machine tied for the best mean

    @property
    def total_regret(self) -> float:
        return self.regret[-1] if self.regret else 0.0

    def share_optimal(self, start: int = 0) -> float:
        """Fraction of pulls from `start` on that went to the best machine."""
        tail = self.optimal[start:]
        return sum(tail) / len(tail) if tail else 0.0


def run_agent(agent: Agent, env: Environment, on_pull=None) -> Run:
    """Let `agent` play the whole horizon of `env`. The agent sees only the reward of each pull.

    `on_pull(t, arm, reward, agent)`, if given, is called after each pull; the watch mode uses it to
    print the agent's reason for the pull.
    """
    arms, rewards, regret, optimal = [], [], [], []
    cum = 0.0
    for t in range(env.horizon):
        arm = agent.choose()
        reward = env.reward(t, arm)
        m = env.mean(t)
        # Expected regret of this pull: how far its machine's mean is below the best mean right now.
        cum += max(m) - m[arm]
        arms.append(arm)
        rewards.append(reward)
        regret.append(cum)
        optimal.append(m[arm] == max(m))
        agent.update(arm, reward)
        if on_pull is not None:
            on_pull(t, arm, reward, agent)
    return Run(arms, rewards, regret, optimal)


def make_for(key: str, env: Environment) -> Agent:
    """The agent named `key`, set up for `env`, with its own random stream derived from the seed."""
    rng = Rng(env.seed + AGENT_OFFSET)
    return make_agent(key, env.k, env.horizon, rng, sigma=env.sigma, bernoulli=env.kind != "gaussian")


def play(key: str, env: Environment, on_pull=None) -> Run:
    """Run the agent named `key` on `env`."""
    return run_agent(make_for(key, env), env, on_pull)


def score_pulls(env: Environment, arms: list[int]) -> Run:
    """Score a human's arm choices on `env`. len(arms) must not exceed the horizon of the environment.

    Rewards come from the same outcome table the agents see, so the player's luck is the agents' luck.
    """
    if len(arms) > env.horizon:
        raise ValueError("more pulls than the casino's horizon")
    rewards, regret, optimal = [], [], []
    cum = 0.0
    for t, arm in enumerate(arms):
        if not 0 <= arm < env.k:
            raise ValueError(f"machine {arm} does not exist; the casino has {env.k} machines")
        m = env.mean(t)
        cum += max(m) - m[arm]
        rewards.append(env.reward(t, arm))
        regret.append(cum)
        optimal.append(m[arm] == max(m))
    return Run(list(arms), rewards, regret, optimal)


def band(series: list[list[float]]) -> tuple[list[float], list[float], list[float]]:
    """Pointwise mean and 95% band (mean +/- 1.96 standard errors) over runs of equal length."""
    n = len(series)
    if n == 0:
        raise ValueError("no runs")
    length = len(series[0])
    mean, lo, hi = [], [], []
    for i in range(length):
        col = [s[i] for s in series]
        mu = sum(col) / n
        if n > 1:
            var = sum((x - mu) ** 2 for x in col) / (n - 1)
            half = 1.96 * math.sqrt(var / n)
        else:
            half = 0.0
        mean.append(mu)
        lo.append(mu - half)
        hi.append(mu + half)
    return mean, lo, hi


def interval(values: list[float]) -> tuple[float, float, float]:
    """Mean and 95% half-width band endpoints for a list of per-seed numbers: (mean, lo, hi)."""
    mean, lo, hi = band([[v] for v in values])
    return mean[0], lo[0], hi[0]
