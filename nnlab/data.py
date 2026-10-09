"""Toy 2-D classification datasets, feature maps and the train/test split.

Each point is a dict with its base position (x, y), its class label (0 cyan, 1 pink), a jitter
vector (jx, jy) and a split draw u. The effective position is base + noise * jitter, so the noise
slider can change live without re-drawing the dataset. A point is a test point when u < test_frac,
which keeps a point in the same set as the test fraction changes.

web/js/nnlab-core.js implements the same generators with the same random-number order, so a
preset is the same set of points in the browser and in the terminal.
"""

from __future__ import annotations

import math

import numpy as np

from .rng import Rng

PRESETS = ("blobs", "xor", "circles", "spiral", "moons")
FEATURES = ("x", "y", "x2", "y2", "xy", "sinx", "siny")


def _base(name: str, i: int, rng: Rng) -> tuple[float, float, int]:
    """Position and label of the i-th point of a preset. Labels alternate, except for XOR."""
    if name == "blobs":
        label = i % 2
        c = 0.45 if label else -0.45
        return c + 0.25 * rng.normal(), c + 0.25 * rng.normal(), label
    if name == "xor":
        # Four clusters at the corners; a point is pink when its two signs agree.
        cluster = i % 4
        sx = 0.5 if cluster & 1 else -0.5
        sy = 0.5 if cluster & 2 else -0.5
        label = 1 if (sx > 0) == (sy > 0) else 0
        return sx + 0.2 * rng.normal(), sy + 0.2 * rng.normal(), label
    if name == "circles":
        # Cyan is an inner disc and pink a surrounding ring: no straight line separates them.
        label = i % 2
        r = (0.85 if label else 0.4) + 0.05 * rng.normal()
        a = 2.0 * math.pi * rng.random()
        return r * math.cos(a), r * math.sin(a), label
    if name == "spiral":
        # Two interleaved arms, 1.6 turns each, offset by half a turn.
        label = i % 2
        t = rng.random()
        a = 3.2 * math.pi * t + label * math.pi
        r = 0.1 + 0.8 * t
        return r * math.cos(a) + 0.03 * rng.normal(), r * math.sin(a) + 0.03 * rng.normal(), label
    if name == "moons":
        # Two arcs that interlock like a yin-yang.
        label = i % 2
        t = math.pi * rng.random()
        if label == 0:
            x, y = 0.6 * math.cos(t) - 0.3, 0.6 * math.sin(t) - 0.15
        else:
            x, y = -0.6 * math.cos(t) + 0.3, -0.6 * math.sin(t) + 0.15
        return x + 0.06 * rng.normal(), y + 0.06 * rng.normal(), label
    raise ValueError(f"unknown dataset {name!r}; choose from {', '.join(PRESETS)}")


def generate(name: str, n: int = 200, seed: int = 0) -> list[dict]:
    """n points of a preset. Each point draws its base position, then jitter (2 normals), then u."""
    rng = Rng(seed)
    points = []
    for i in range(n):
        x, y, label = _base(name, i, rng)
        jx = rng.normal()
        jy = rng.normal()
        u = rng.random()
        points.append({"x": x, "y": y, "label": label, "jx": jx, "jy": jy, "u": u})
    return points


def featurize(x: float, y: float, names: tuple[str, ...] | list[str]) -> list[float]:
    """The chosen input features of one point. sin uses pi * coordinate so it wiggles over [-1, 1]."""
    table = {
        "x": x,
        "y": y,
        "x2": x * x,
        "y2": y * y,
        "xy": x * y,
        "sinx": math.sin(math.pi * x),
        "siny": math.sin(math.pi * y),
    }
    return [table[k] for k in names]


def build_split(points: list[dict], features, noise: float = 0.0, test_frac: float = 0.2):
    """Feature matrices for the training and test sets.

    Returns (X_train, y_train, X_test, y_test): X arrays are (n, F) float64, y arrays are (n, 1) in {0, 1}.
    A point's effective position is base + noise * jitter. With test_frac 0 the test set is empty.
    """
    Xtr, ytr, Xte, yte = [], [], [], []
    for p in points:
        row = featurize(p["x"] + noise * p["jx"], p["y"] + noise * p["jy"], features)
        if p["u"] < test_frac:
            Xte.append(row)
            yte.append(p["label"])
        else:
            Xtr.append(row)
            ytr.append(p["label"])
    F = len(features)
    return (
        np.array(Xtr, dtype=np.float64).reshape(-1, F),
        np.array(ytr, dtype=np.float64).reshape(-1, 1),
        np.array(Xte, dtype=np.float64).reshape(-1, F),
        np.array(yte, dtype=np.float64).reshape(-1, 1),
    )
