"""A seeded random stream that the numpy filters and the JavaScript port can both reproduce.

It is mulberry32 (the same generator as bandits/rng.py), but written so that a whole batch of draws can be
computed at once. mulberry32's state is a plain counter: after i draws the state is seed + C * i (mod 2**32),
and each output is a fixed mixing function of that state. So the i-th uniform can be computed directly, and
a numpy array of 10**4 draws costs one vector expression instead of 10**4 Python calls.

Normals use the Marsaglia polar method, consuming uniforms in pairs exactly as a sequential loop would:
a pair is rejected unless 0 < s < 1, and each accepted pair yields one normal. `Stream.normals` therefore
returns the same numbers as calling `normal()` in a loop, and the JavaScript `Mulberry32.normal()` matches.
The test suite checks the stream against bandits.rng.Rng directly.
"""

from __future__ import annotations

import numpy as np

M = 0xFFFFFFFF
C = 0x6D2B79F5
TWO32 = 4294967296.0
_M64 = np.uint64(M)
_C = np.uint64(C)


def _mix(state: np.ndarray) -> np.ndarray:
    """The output mixing of mulberry32, applied to an array of 32-bit states (as uint64)."""
    one, sixtyone = np.uint64(1), np.uint64(61)
    t = state
    # Products of two 32-bit values fit in 64 bits, so uint64 arithmetic keeps every bit we need.
    t = ((t ^ (t >> np.uint64(15))) * (one | t)) & _M64
    t = (((t + (((t ^ (t >> np.uint64(7))) * (sixtyone | t)) & _M64)) & _M64) ^ t)
    return (t ^ (t >> np.uint64(14))) & _M64


class Stream:
    """Draws numbers from mulberry32 for one seed. `pos` counts the uniforms consumed so far."""

    def __init__(self, seed: int):
        self.seed = np.uint64(seed & M)
        self.pos = 0

    def _at(self, start: int, n: int) -> np.ndarray:
        """Uniforms number start .. start+n-1 of the stream, as float64 in [0, 1)."""
        idx = np.arange(start + 1, start + n + 1, dtype=np.uint64)
        state = (self.seed + _C * idx) & _M64
        return _mix(state).astype(np.float64) / TWO32

    def uniforms(self, n: int) -> np.ndarray:
        """The next n uniforms in [0, 1)."""
        out = self._at(self.pos, n)
        self.pos += n
        return out

    def index(self, k: int, n: int = 1) -> np.ndarray:
        """n integers uniform in [0, k), one uniform each: min(k-1, int(u*k)), as in bandits.rng."""
        return np.minimum(k - 1, (self.uniforms(n) * k).astype(np.int64))

    def normals(self, n: int) -> np.ndarray:
        """The next n standard normals by the polar method, consuming uniform pairs in order."""
        out = np.empty(n)
        got = 0
        while got < n:
            # Draw a batch of pairs a little larger than needed; about pi/4 of the pairs are accepted.
            pairs = int((n - got) * 1.3) + 8
            u = 2.0 * self._at(self.pos, 2 * pairs).reshape(pairs, 2) - 1.0
            s = u[:, 0] * u[:, 0] + u[:, 1] * u[:, 1]
            take = np.nonzero((s > 0.0) & (s < 1.0))[0][: n - got]
            out[got:got + len(take)] = u[take, 0] * np.sqrt(-2.0 * np.log(s[take]) / s[take])
            got += len(take)
            if got == n:
                # Only the pairs up to the last accepted one were consumed; rewind past the rest.
                self.pos += 2 * (int(take[-1]) + 1)
            else:
                self.pos += 2 * pairs
        return out
