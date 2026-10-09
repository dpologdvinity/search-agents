"""Seeded toy datasets for the lab: three regression curves and two classification shapes.

Every point is drawn from bandits.rng.Rng (mulberry32), in a fixed order, so the JavaScript page
reproduces the same points for the same seed. The order of draws is part of the contract: one
uniform for x (or for the angle t), then the normal noise terms, point by point.
"""

from __future__ import annotations

import math

import numpy as np

from bandits.rng import Rng

REGRESSION_PRESETS = ("line", "noisy-sine", "quadratic")
CLASSIFICATION_PRESETS = ("moons", "blobs")

REGRESSION_NOISE = 0.15
MOONS_NOISE = 0.1
BLOBS_NOISE = 0.22

_TRUE_FUNCTIONS = {
    "line": lambda x: 0.6 * x - 0.2,
    "noisy-sine": lambda x: 0.8 * math.sin(2.0 * math.pi * x),
    "quadratic": lambda x: 1.2 * x * x - 0.6,
}


def make_regression(name: str, n: int, seed: int, noise: float = REGRESSION_NOISE):
    """Noisy samples of a known curve on x in [-1, 1].

    For each point: x = -1 + 2u with u uniform, then eps = noise * (standard normal), y = f(x) + eps.
    Returns (x, y), both length n. The noise is always drawn, even when noise is 0, so the x values
    do not change when only the noise level changes.
    """
    if name not in _TRUE_FUNCTIONS:
        raise ValueError(f"unknown regression preset {name!r}: choose from {', '.join(REGRESSION_PRESETS)}")
    f = _TRUE_FUNCTIONS[name]
    rng = Rng(seed)
    xs = np.empty(n)
    ys = np.empty(n)
    for i in range(n):
        x = -1.0 + 2.0 * rng.uniform()
        eps = noise * rng.normal()
        xs[i] = x
        ys[i] = f(x) + eps
    return xs, ys


def make_classification(name: str, n: int, seed: int, noise: float | None = None):
    """Two interleaved classes in the plane. Labels alternate 0, 1, 0, 1, ... so both classes are balanced.

    moons: two half-circles. Class 0 is the upper arc, class 1 the lower arc shifted right. Each point
    draws an angle t = pi * u first, then two normal noise terms (x then y). The result is rescaled
    so both arcs fit in the page's [-1, 1] frame with a small margin.

    blobs: two Gaussian clouds at (-0.45, -0.35) and (0.45, 0.35) with standard deviation noise.
    Blobs draw no angle, only the two normals.

    Returns (xy, labels): xy is (n, 2) float, labels is (n,) int with values 0 and 1.
    """
    if name not in CLASSIFICATION_PRESETS:
        raise ValueError(f"unknown classification preset {name!r}: choose from {', '.join(CLASSIFICATION_PRESETS)}")
    if noise is None:
        noise = MOONS_NOISE if name == "moons" else BLOBS_NOISE
    rng = Rng(seed)
    xy = np.empty((n, 2))
    labels = np.empty(n, dtype=int)
    for i in range(n):
        c = i % 2
        labels[i] = c
        if name == "moons":
            t = math.pi * rng.uniform()
            if c == 0:
                px, py = math.cos(t), math.sin(t)
            else:
                px, py = 1.0 - math.cos(t), 0.5 - math.sin(t)
            # Rescale into the frame: the arcs are 0.6 wide and 1 tall, centred near the origin.
            px = (px - 0.5) * 0.6
            py = (py - 0.25) * 1.0
            px += noise * rng.normal()
            py += noise * rng.normal()
        else:
            cx, cy = (-0.45, -0.35) if c == 0 else (0.45, 0.35)
            px = cx + noise * rng.normal()
            py = cy + noise * rng.normal()
        xy[i] = (px, py)
    return xy, labels
