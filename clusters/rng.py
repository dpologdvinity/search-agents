"""Seeded PRNG shared with the browser: mulberry32, plus Box-Muller normals.

NumPy's generators are not used on purpose. The page has to reproduce every draw bit for bit, and a
32-bit integer PRNG with a few multiplies does that in both languages. Each `random()` call is one
step of the stream; `normal()` always consumes exactly two, so call order alone decides the output.
"""

import math

M32 = 0xFFFFFFFF


class Mulberry32:
    """A 32-bit generator; the same seed gives the same stream in Python and JavaScript."""

    def __init__(self, seed: int) -> None:
        self.a = seed & M32

    def random(self) -> float:
        """Uniform float in [0, 1), built from the next 32-bit state."""
        self.a = (self.a + 0x6D2B79F5) & M32
        t = self.a
        t = ((t ^ (t >> 15)) * (t | 1)) & M32
        t = (t ^ ((t + (((t ^ (t >> 7)) * (t | 61)) & M32)) & M32)) & M32
        return ((t ^ (t >> 14)) & M32) / 4294967296.0

    def uniform(self, lo: float, hi: float) -> float:
        """Uniform float in [lo, hi)."""
        return lo + (hi - lo) * self.random()

    def normal(self) -> float:
        """Standard normal by Box-Muller; always uses two uniforms, so the stream stays aligned."""
        u1 = self.random()
        u2 = self.random()
        if u1 < 1e-12:
            u1 = 1e-12
        return math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)

    def index(self, n: int) -> int:
        """Uniform integer in [0, n)."""
        j = int(self.random() * n)
        return j if j < n else n - 1
