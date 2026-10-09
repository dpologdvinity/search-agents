"""Seeded 32-bit random numbers that match web/js/nnlab-core.js bit for bit.

The browser and the terminal must draw the same datasets, the same initial weights and the same
mini-batch orders for a given seed, so both use mulberry32 (a 32-bit generator with a 2^32 period)
instead of NumPy's generator. Every operation is masked to 32 bits, mirroring the JavaScript code
where Math.imul and the >>> operator do the same job.
"""

from __future__ import annotations

import math

_M = 0xFFFFFFFF


class Rng:
    """mulberry32. random() is uniform on [0, 1); normal() is standard normal (Box-Muller).

    normal() uses both uniforms of a pair and throws the second Box-Muller value away, so the number
    of draws per call is fixed and the JavaScript port can follow the same sequence.
    """

    def __init__(self, seed: int):
        self.a = seed & _M

    def random(self) -> float:
        self.a = (self.a + 0x6D2B79F5) & _M
        t = self.a
        # Scramble the counter: multiply by an odd number so every input bit reaches the high bits.
        t = ((t ^ (t >> 15)) * (t | 1)) & _M
        t = ((t + ((t ^ (t >> 7)) * (t | 61))) & _M) ^ t
        return ((t ^ (t >> 14)) & _M) / 4294967296.0

    def normal(self) -> float:
        # Box-Muller: if u1 and u2 are uniform, sqrt(-2 ln u1) cos(2 pi u2) is standard normal.
        u1 = max(self.random(), 1e-12)  # ln(0) is undefined
        u2 = self.random()
        return math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)
