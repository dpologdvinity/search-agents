"""Particle filter: approximate the same belief with N weighted samples of the hidden state.

The exact filter touches every state; a particle filter keeps N guesses ("particles") and only needs to be able
to sample the motion model and evaluate the sonar. It is the approximation you use when the state space is too
large to enumerate, so the benchmark measures how fast it approaches the exact belief as N grows.

Each turn:
  predict  every particle moves by sampling one successor from the motion model (one uniform per particle);
  update   each particle's weight is multiplied by the sonar likelihood of its distance to the player;
  resample when the effective sample size (1 / sum of squared normalised weights) falls below N/2, so
           particles with negligible weight are replaced by copies of likely ones. Systematic resampling uses
           one uniform and walks the cumulative weights, so it adds the least sampling noise of the usual schemes.

The belief is the weighted histogram of particles over open cells. Randomness comes from a mulberry32 stream,
so a particle filter run is reproducible and the JavaScript port can match it.
"""

from __future__ import annotations

from bandits.rng import Rng

from . import motion


class ParticleFilter:
    """N particles over hidden states, initialised from `prior` (a list of 4K probabilities)."""

    def __init__(self, maze, model: str, prior: list[float], emis: list[list[float]], n: int, rng: Rng):
        self.maze = maze
        self.model = model
        self.emis = emis
        self.n = n
        self.rng = rng
        states = [s for s, p in enumerate(prior) if p > 0.0]
        # Draw each particle from the prior by inverting its CDF at a fresh uniform.
        cum, acc = [], 0.0
        for s in states:
            acc += prior[s]
            cum.append(acc)
        self.parts = [states[_search(cum, rng.uniform() * acc)] for _ in range(n)]
        self.w = [1.0 / n] * n
        self.resamples = 0

    def predict(self, player: int) -> None:
        """Move every particle by one sample of the motion model."""
        trans = motion.table(self.maze, self.model, player)
        for i, s in enumerate(self.parts):
            options = trans[s]
            if len(options) == 1:
                self.parts[i] = options[0][0]  # deterministic move: no draw needed, the stream stays aligned
                continue
            u = self.rng.uniform()
            acc = 0.0
            chosen = options[-1][0]
            for s2, q in options:
                acc += q
                if u < acc:
                    chosen = s2
                    break
            self.parts[i] = chosen

    def update(self, reading: int, player: int) -> None:
        """Weight each particle by its sonar likelihood, then resample if the weights have degenerated."""
        dist = self.maze.dist
        emis = self.emis
        w = [wi * emis[dist[s >> 2][player]][reading] for wi, s in zip(self.w, self.parts)]
        total = sum(w)
        if total > 0.0:
            w = [x / total for x in w]
        else:  # numerical guard: every particle underflowed, so treat them as equally (un)likely
            w = [1.0 / self.n] * self.n
        ess = 1.0 / sum(x * x for x in w)
        if ess < self.n / 2.0:
            self._resample(w)
        else:
            self.w = w

    def _resample(self, w: list[float]) -> None:
        """Systematic resampling: N evenly spaced pointers over the cumulative weights, from one random offset."""
        n = self.n
        cum, acc = [], 0.0
        for x in w:
            acc += x
            cum.append(acc)
        u0 = self.rng.uniform() / n
        new = []
        for i in range(n):
            new.append(self.parts[_search(cum, u0 + i / n)])
        self.parts = new
        self.w = [1.0 / n] * n
        self.resamples += 1

    def marginal(self) -> list[float]:
        """Weighted histogram of particles over open cells (sums to 1)."""
        out = [0.0] * self.maze.K
        for wi, s in zip(self.w, self.parts):
            out[s >> 2] += wi
        return out


def _search(cum: list[float], u: float) -> int:
    """Index of the first cumulative value greater than u (clamped to the last entry for rounding)."""
    lo, hi = 0, len(cum) - 1
    while lo < hi:
        mid = (lo + hi) // 2
        if u < cum[mid]:
            hi = mid
        else:
            lo = mid + 1
    return lo
