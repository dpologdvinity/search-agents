"""A seeded uniform generator and an approximately normal sampler, identical in Python and JavaScript.

mulberry32 is a 32-bit generator that is easy to port exactly: every step is an integer operation that both
languages express the same way. Normal noise is the sum of 12 uniforms minus 6. It has mean 0 and variance 1,
and it needs no log, sqrt or cos, so both ports produce bit-identical streams.
"""

from __future__ import annotations

M32 = 0xFFFFFFFF


def _imul(a: int, b: int) -> int:
    """Low 32 bits of a * b, like JavaScript's Math.imul."""
    return (a * b) & M32


class Rng:
    """mulberry32: uniform() returns a float in [0, 1) with 32 bits of resolution."""

    def __init__(self, seed: int = 0):
        self.a = seed & M32

    def uniform(self) -> float:
        self.a = (self.a + 0x6D2B79F5) & M32
        a = self.a
        t = _imul(a ^ (a >> 15), 1 | a)
        t = ((t + _imul(t ^ (t >> 7), 61 | t)) & M32) ^ t
        return ((t ^ (t >> 14)) & M32) / 4294967296.0

    def normal(self) -> float:
        """Sum of 12 uniforms minus 6: mean 0, variance 1, bounded in [-6, 6]."""
        s = 0.0
        for _ in range(12):
            s += self.uniform()
        return s - 6.0
