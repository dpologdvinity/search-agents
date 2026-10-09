"""The noisy sonar: how a true distance becomes the reading the player sees, and how likely each reading is.

A ping reports the maze distance from the player to a ghost, plus noise, as a whole number in 0..R (R is the
maze diameter, the largest distance any two cells can have). The noise is a discrete Gaussian with standard
deviation sigma:

    P(reading = r | true distance = d)  =  exp(-(r - d)^2 / (2 sigma^2)) / Z(d),   Z(d) = sum over r of the same

Z(d) normalises over the readings the sonar can actually show, so each row is a proper distribution. The
emission is the only place the sensor enters the filters: they multiply their belief by one table entry per
hidden state. Only exp() is needed, so the JavaScript port reproduces it.

Noise levels offered on the page and in the CLI: LOW 0.5, MED 1.0, HIGH 2.0 (in cells).
"""

from __future__ import annotations

import math

NOISE_LEVELS = {"low": 0.5, "med": 1.0, "high": 2.0}
FLOOR = 1e-300


def emission_table(sigma: float, R: int) -> list[list[float]]:
    """rows[d][r] = P(reading r | true distance d) for d and r in 0..R."""
    rows = []
    for d in range(R + 1):
        w = [math.exp(-((r - d) ** 2) / (2.0 * sigma * sigma)) for r in range(R + 1)]
        z = sum(w)
        # Floor at FLOOR: far readings underflow to exactly 0 when sigma is small and R is large, and a zero
        # would make log() undefined in Viterbi and wipe out a belief. The floor is far below any real chance.
        rows.append([max(x / z, FLOOR) for x in w])
    return rows


def sample_reading(row: list[float], u: float) -> int:
    """Draw a reading from one row of the table by inverting its CDF at the uniform u."""
    acc = 0.0
    for r, p in enumerate(row):
        acc += p
        if u < acc:
            return r
    return len(row) - 1  # guards against rounding leaving the last bucket just short of 1
