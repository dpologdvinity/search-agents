"""Seeded 2-D presets for the lab, and the feature expansion the trees split on.

All points live in the unit square. Presets draw only from the shared generator and use no transcendental
functions except the spirals (sin and cos), so the browser matches them to rounding error.
The label of each point is decided before any noise, except the diagonal's explicit flips.
"""

from __future__ import annotations

import math

import numpy as np

from .rng import Mulberry32

PRESETS = {
    "blobs": 3,         # three round clusters
    "xor": 2,           # the four quadrants, alternating
    "checkerboard": 2,  # a 4 x 4 board, alternating squares
    "spiral": 2,        # two interleaved arms
    "nested": 3,        # three concentric rings
    "diagonal": 2,      # x > y, with 15% of labels flipped
}
FLIP = 0.15
BLOB_CENTRES = ((0.25, 0.3), (0.7, 0.35), (0.5, 0.75))
BLOB_RADIUS = 0.16
RING_R2 = (0.04, 0.13)  # squared radii of the two inner ring edges around (0.5, 0.5)

# Expanded features: the plane's x and y, then the cross term and squared radius. Names are used in rules.
FEATURE_NAMES = ("x", "y", "x*y", "x^2+y^2")
FEATURE_NAMES_WEB = ("x", "y", "x·y", "x²+y²")


def expand(X: np.ndarray, extras: bool = True) -> np.ndarray:
    """Return the feature matrix: [x, y] plus, when extras is set, [x*y, x*x + y*y]."""
    X = np.asarray(X, dtype=np.float64)
    cols = [X[:, 0], X[:, 1]]
    if extras:
        cols.append(X[:, 0] * X[:, 1])
        cols.append(X[:, 0] * X[:, 0] + X[:, 1] * X[:, 1])
    return np.column_stack(cols)


def _disc(r: Mulberry32) -> tuple[float, float]:
    """Uniform point in the unit disc by rejection (no trigonometry)."""
    while True:
        u = 2.0 * r.next() - 1.0
        v = 2.0 * r.next() - 1.0
        if u * u + v * v <= 1.0:
            return u, v


def make_preset(name: str, n: int = 240, seed: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Return (X, y): n points in [0, 1]^2 with integer labels 0..PRESETS[name]-1, for a seed."""
    if name not in PRESETS:
        raise ValueError(f"unknown preset {name!r}; choose from {', '.join(PRESETS)}")
    r = Mulberry32(seed)
    pts: list[tuple[float, float]] = []
    labels: list[int] = []
    for i in range(n):
        if name == "blobs":
            c = i % 3
            u, v = _disc(r)
            pts.append((BLOB_CENTRES[c][0] + BLOB_RADIUS * u, BLOB_CENTRES[c][1] + BLOB_RADIUS * v))
            labels.append(c)
        elif name == "xor":
            x, y = r.next(), r.next()
            pts.append((x, y))
            labels.append(int((x > 0.5) != (y > 0.5)))
        elif name == "checkerboard":
            x, y = r.next(), r.next()
            pts.append((x, y))
            labels.append((math.floor(x * 4) + math.floor(y * 4)) % 2)
        elif name == "spiral":
            arm = i % 2
            t = r.next()
            theta = 3.5 * math.pi * t + arm * math.pi
            rad = 0.04 + 0.4 * t
            jx = 0.02 * (2.0 * r.next() - 1.0)
            jy = 0.02 * (2.0 * r.next() - 1.0)
            pts.append((0.5 + rad * math.cos(theta) + jx, 0.5 + rad * math.sin(theta) + jy))
            labels.append(arm)
        elif name == "nested":
            x, y = r.next(), r.next()
            d2 = (x - 0.5) * (x - 0.5) + (y - 0.5) * (y - 0.5)
            pts.append((x, y))
            labels.append(0 if d2 < RING_R2[0] else (1 if d2 < RING_R2[1] else 2))
        else:  # diagonal
            x, y = r.next(), r.next()
            lab = int(x > y)
            if r.next() < FLIP:
                lab = 1 - lab
            pts.append((x, y))
            labels.append(lab)
    return np.array(pts, dtype=np.float64), np.array(labels, dtype=np.int64)


def train_test_split(n: int, test_fraction: float, seed: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Seeded shuffle of range(n). Returns (train_idx, test_idx); the test set is round(test_fraction * n) rows."""
    r = Mulberry32(seed)
    perm = list(range(n))
    for i in range(n - 1, 0, -1):  # Fisher-Yates, from the end
        j = r.rand_int(i + 1)
        perm[i], perm[j] = perm[j], perm[i]
    n_test = int(math.floor(test_fraction * n + 0.5))
    perm_arr = np.array(perm, dtype=np.int64)
    return perm_arr[n_test:], perm_arr[:n_test]
