"""Seeded 32-bit random numbers (mulberry32) shared with the browser.

The JavaScript port uses Math.imul and unsigned shifts to produce the same bits, so a seed means
the same maze in Python and in the page. Integer draws use a plain modulo: the bias is far below
anything a maze can show, and it keeps both sides free of floating point.
"""

from __future__ import annotations

M32 = 0xFFFFFFFF


class Rng:
    """Deterministic generator. Create one per maze so a seed fully determines the output."""

    def __init__(self, seed: int):
        self.a = seed & M32

    def u32(self) -> int:
        """Next unsigned 32-bit value."""
        self.a = (self.a + 0x6D2B79F5) & M32
        a = self.a
        t = ((a ^ (a >> 15)) * (1 | a)) & M32  # Math.imul(a ^ (a >>> 15), 1 | a)
        t = ((t + ((t ^ (t >> 7)) * (61 | t))) & M32) ^ t  # Math.imul(t ^ (t >>> 7), 61 | t)
        return (t ^ (t >> 14)) & M32

    def int(self, n: int) -> int:
        """Integer in [0, n)."""
        return self.u32() % n

    def percent(self, p: int) -> bool:
        """True with probability p/100 (integer percent, so no floats are involved)."""
        return self.u32() % 100 < p
