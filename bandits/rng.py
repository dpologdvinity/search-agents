"""Seeded random numbers that the page's JavaScript port can reproduce.

Python's `random` (Mersenne Twister) cannot be matched in JavaScript, so the bandit code uses a
small generator instead: mulberry32, a 32-bit state advanced by an add and a few multiply-xor
steps. Uniforms are 32-bit integers divided by 2**32, so both languages produce the same doubles.

Normals use the Marsaglia polar method. Gamma variates use Marsaglia and Tsang (2000), and a Beta
variate is X / (X + Y) for two gammas. Beta draws drive Thompson sampling, so Beta(a, b) is only
needed with a, b >= 1 (a posterior after a uniform prior), which is the only case implemented here.
The only transcendental functions are log and sqrt, and sqrt is exact in both languages, so the two
streams agree except in rare cases where a log differs in the last bit.
"""

from __future__ import annotations

import math

MASK = 0xFFFFFFFF
_TWO32 = 4294967296.0


class Rng:
    """mulberry32. `seed` is taken modulo 2**32, so every seed in the JSON API is valid."""

    def __init__(self, seed: int):
        self.state = seed & MASK

    def uniform(self) -> float:
        """A double in [0, 1). Each call advances the state by one step and mixes it."""
        self.state = (self.state + 0x6D2B79F5) & MASK
        t = self.state
        # The multiplies keep only the low 32 bits, like Math.imul in JavaScript.
        t = ((t ^ (t >> 15)) * (1 | t)) & MASK
        t = (((t + (((t ^ (t >> 7)) * (61 | t)) & MASK)) & MASK) ^ t)
        return ((t ^ (t >> 14)) & MASK) / _TWO32

    def index(self, k: int) -> int:
        """A uniform integer in [0, k)."""
        return min(k - 1, int(self.uniform() * k))

    def normal(self) -> float:
        """A standard normal, by the polar method: draw points in the unit disc and keep one coordinate."""
        while True:
            u = 2.0 * self.uniform() - 1.0
            v = 2.0 * self.uniform() - 1.0
            s = u * u + v * v
            if 0.0 < s < 1.0:
                break
        return u * math.sqrt(-2.0 * math.log(s) / s)

    def gamma(self, shape: float) -> float:
        """Gamma(shape, 1) for shape >= 1, by Marsaglia and Tsang's rejection method.

        d = shape - 1/3 and c = 1/sqrt(9d) give a fast squeeze: most candidates are accepted
        by the cheap first test, and the log test handles the rest.
        """
        if shape < 1.0:
            raise ValueError("gamma is implemented for shape >= 1 only")
        d = shape - 1.0 / 3.0
        c = 1.0 / math.sqrt(9.0 * d)
        while True:
            x = self.normal()
            v = 1.0 + c * x
            if v <= 0.0:
                continue
            v = v * v * v
            u = self.uniform()
            if u <= 0.0:
                continue
            if u < 1.0 - 0.0331 * x * x * x * x:
                return d * v
            if math.log(u) < 0.5 * x * x + d * (1.0 - v + math.log(v)):
                return d * v

    def beta(self, a: float, b: float) -> float:
        """Beta(a, b) for a, b >= 1: X / (X + Y) with X ~ Gamma(a) and Y ~ Gamma(b)."""
        x = self.gamma(a)
        y = self.gamma(b)
        return x / (x + y)
