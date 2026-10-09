"""Seeded point presets: blobs, anisotropic blobs, rings, moons, uniform noise, and a smiley.

Points live in the unit square. Each preset returns the points and the generating component of every
point (`truth`), which the page and CLI can show next to what the algorithms found. The JavaScript
port in web/js/clusters-core.js runs the same draws in the same order, so a preset matches exactly.
"""

import math

import numpy as np

from clusters.rng import Mulberry32

PRESETS = ("blobs", "aniso", "rings", "moons", "uniform", "smiley")
BLOB_CENTRES = ((0.25, 0.3), (0.75, 0.25), (0.3, 0.75), (0.72, 0.72))
# Shear applied to every anisotropic blob: stretches the cloud along a diagonal.
SHEAR = ((0.6, -0.6), (-0.4, 0.8))


def _split(n: int, parts: int) -> list[int]:
    """Points per component: n divided evenly, the remainder going to the first components."""
    base, extra = divmod(n, parts)
    return [base + (1 if i < extra else 0) for i in range(parts)]


def make_dataset(name: str, seed: int = 0, n: int = 300) -> tuple[np.ndarray, np.ndarray]:
    """Return (points (n, 2) in [0, 1]^2, truth (n,)) for a named preset and seed."""
    if name not in PRESETS:
        raise ValueError(f"unknown preset {name!r}; choose from {', '.join(PRESETS)}")
    rng = Mulberry32(seed)
    pts: list[tuple[float, float]] = []
    truth: list[int] = []

    def emit(comp: int, x: float, y: float) -> None:
        pts.append((min(1.0, max(0.0, x)), min(1.0, max(0.0, y))))
        truth.append(comp)

    if name in ("blobs", "aniso"):
        if name == "blobs":
            # Each centre is nudged a little so no two seeds give the same picture.
            centres = [(cx + rng.uniform(-0.04, 0.04), cy + rng.uniform(-0.04, 0.04)) for cx, cy in BLOB_CENTRES]
        else:
            centres = [(0.25, 0.3), (0.7, 0.3), (0.5, 0.75)]
        counts = _split(n, len(centres))
        for comp, (cnt, (cx, cy)) in enumerate(zip(counts, centres, strict=True)):
            for _ in range(cnt):
                gx, gy = rng.normal() * 0.05, rng.normal() * 0.05
                if name == "aniso":
                    (a, b), (c, d) = SHEAR
                    gx, gy = a * gx + b * gy, c * gx + d * gy
                emit(comp, cx + gx, cy + gy)
    elif name == "rings":
        for comp, (cnt, radius) in enumerate(zip(_split(n, 2), (0.2, 0.42), strict=True)):
            for _ in range(cnt):
                t = rng.uniform(0.0, 2.0 * math.pi)
                r = radius + rng.normal() * 0.02
                emit(comp, 0.5 + r * math.cos(t), 0.5 + r * math.sin(t))
    elif name == "moons":
        for comp, (cnt, sign) in enumerate(zip(_split(n, 2), (1.0, -1.0), strict=True)):
            for _ in range(cnt):
                t = rng.uniform(0.0, math.pi)
                x = 0.35 + 0.28 * math.cos(t) if sign > 0 else 0.65 - 0.28 * math.cos(t)
                y = 0.45 + 0.28 * math.sin(t) if sign > 0 else 0.55 - 0.28 * math.sin(t)
                emit(comp, x + rng.normal() * 0.025, y + rng.normal() * 0.025)
    elif name == "uniform":
        for _ in range(n):
            emit(0, rng.random(), rng.random())
    else:  # smiley: two eyes, a smile arc, and the face outline
        counts = _split(n, 4)
        for _ in range(counts[0]):
            emit(0, 0.36 + rng.normal() * 0.025, 0.66 + rng.normal() * 0.025)
        for _ in range(counts[1]):
            emit(1, 0.64 + rng.normal() * 0.025, 0.66 + rng.normal() * 0.025)
        for _ in range(counts[2]):
            t = rng.uniform(0.15 * math.pi, 0.85 * math.pi)
            emit(2, 0.5 + 0.22 * math.cos(t) + rng.normal() * 0.012, 0.5 - 0.22 * math.sin(t) + rng.normal() * 0.012)
        for _ in range(counts[3]):
            t = rng.uniform(0.0, 2.0 * math.pi)
            emit(3, 0.5 + 0.42 * math.cos(t) + rng.normal() * 0.012, 0.5 + 0.42 * math.sin(t) + rng.normal() * 0.012)

    return np.array(pts, dtype=np.float64), np.array(truth, dtype=np.int64)
