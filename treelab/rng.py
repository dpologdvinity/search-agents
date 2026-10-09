"""Mulberry32: a small seeded 32-bit generator, copied exactly into web/js/treelab-core.js.

Every random draw in the lab (presets, bootstrap samples, feature subsets, train/test shuffles) comes from
this one stream, so the Python reference and the browser build the same trees for the same seed.
"""

from __future__ import annotations

MASK = 0xFFFFFFFF


class Mulberry32:
    """Stateful generator. next() returns a float in [0, 1) with 32 bits of precision."""

    def __init__(self, seed: int):
        self.a = int(seed) & MASK

    def next(self) -> float:
        # The JavaScript port uses Math.imul (a 32-bit product); masking the product to 32 bits is the same.
        self.a = (self.a + 0x6D2B79F5) & MASK
        t = self.a
        t = ((t ^ (t >> 15)) * (t | 1)) & MASK
        t ^= (t + (((t ^ (t >> 7)) * (t | 61)) & MASK)) & MASK
        return ((t ^ (t >> 14)) & MASK) / 4294967296.0

    def rand_int(self, n: int) -> int:
        """Uniform integer in [0, n)."""
        return int(self.next() * n)
