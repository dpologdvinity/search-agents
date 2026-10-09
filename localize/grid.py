"""Histogram (grid) Bayes filter over (x, y, heading).

The state space is a finite set of bins: every quarter-cell position (SUB = 2 per cell, so 0.25 m) times 16
headings (22.5 degrees). The belief is one probability per bin, so there is no sampling error: each step is
the exact Bayes update on this discretisation.

Each step does two things, in this order:

  predict  belief'(s') = sum_s  p(s' | s, odometry) * belief(s)
           The motion kernel p(s' | s, u) is the odometry noise pushed through the motion model. Its integral over
           a bin is approximated with 3-point Gauss-Hermite quadrature in each of the two noise dimensions (turn and
           drive): 9 weighted sample poses per source bin. A drive that lands in a wall keeps the robot's position but
           still applies the turn, the same rule as world.motion.

  update   belief(s) proportional to p(z | s) * belief(s)
           p(z | s) multiplies one range likelihood per beam, with the expected ranges read from a table computed once
           per map: the ray cast from each bin's centre.

Cost is proportional to bins x quadrature points x beams, so it grows quickly with resolution. The page runs
this exact filter on 24 576 bins (48 by 32 positions, 16 headings); the particle filter's cost depends on the
particle count instead.
"""

from __future__ import annotations

import math

import numpy as np

from .world import TAU, Floor, blocked, cast, noise_sd, ray_loglik, wrap

SUB = 2        # position bins per cell along each axis (0.25 m bins)
HEADS = 16     # heading bins over the full circle (22.5 degrees)
# Three-point Gauss-Hermite rule for a standard normal: nodes -sqrt(3), 0, sqrt(3) with weights 1/6, 2/3, 1/6.
GH_NODES = (-math.sqrt(3.0), 0.0, math.sqrt(3.0))
GH_WEIGHTS = (1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0)


class GridFilter:
    """Discrete Bayes filter. `belief` is a flat array over S bins, indexed (j * W2 + i) * HEADS + k."""

    def __init__(self, floor: Floor, rays: int, sigma: float, pdrop: float, motion_scale: float = 1.0):
        self.floor = floor
        self.rays = rays
        self.sigma = sigma
        self.pdrop = pdrop
        self.scale = motion_scale
        self.w2 = floor.w * SUB
        self.h2 = floor.h * SUB
        self.dh = TAU / HEADS
        s = self.h2 * self.w2 * HEADS
        self.size = s
        # Decode every bin's indices once. The arrays are in index order, so flat index = (j*W2+i)*HEADS+k.
        j = np.repeat(np.arange(self.h2), self.w2 * HEADS)
        i = np.tile(np.repeat(np.arange(self.w2), HEADS), self.h2)
        k = np.tile(np.arange(HEADS), self.h2 * self.w2)
        self.x = (i + 0.5) / SUB           # bin centre, in cells
        self.y = (j + 0.5) / SUB
        self.th = k * self.dh
        self.cos_th = np.cos(self.th)
        self.sin_th = np.sin(self.th)
        cell_free = ~floor.walls[j // SUB, i // SUB]
        # Start uniform over every free bin, which is what "the robot could be anywhere" means.
        self.belief = cell_free.astype(float)
        self.belief /= self.belief.sum()
        # Expected range of every beam from every bin centre: computed once, reused at each update.
        ang = self.th[:, None] + (TAU * np.arange(rays)) / rays
        xs = np.repeat(self.x, rays)
        ys = np.repeat(self.y, rays)
        self.expected = cast(floor.walls, xs, ys, ang.ravel()).reshape(s, rays)
        self.steps = 0

    def predict(self, rot: float, fwd: float) -> None:
        """Push the belief through the odometry kernel for one commanded (turn, drive). See the module docstring."""
        sr, st = noise_sd(rot, fwd, self.scale)
        s = self.size
        dest = np.empty((9, s), dtype=np.int64)
        weight = np.empty((9, s))
        self_index = np.arange(s)
        # Flat index of each bin's position at heading 0. A bump keeps the position but still turns (world.motion).
        here = self_index - self_index % HEADS
        q = 0
        for a, wa in zip(GH_NODES, GH_WEIGHTS):
            for b, wb in zip(GH_NODES, GH_WEIGHTS):
                # The sample pose: drive first (noise b), then turn (noise a), as in world.motion.
                r = rot + a * sr
                f = fwd + b * st
                nx = self.x + f * self.cos_th
                ny = self.y + f * self.sin_th
                nth = wrap(self.th + r)
                bump = blocked(self.floor.walls, nx, ny)
                i2 = np.clip(np.floor(nx * SUB).astype(np.int64), 0, self.w2 - 1)
                j2 = np.clip(np.floor(ny * SUB).astype(np.int64), 0, self.h2 - 1)
                k2 = np.mod(np.floor(nth / self.dh + 0.5).astype(np.int64), HEADS)
                dest[q] = np.where(bump, here + k2, (j2 * self.w2 + i2) * HEADS + k2)
                weight[q] = wa * wb
                q += 1
        mass = (weight * self.belief[None, :]).ravel()
        self.belief = np.bincount(dest.ravel(), weights=mass, minlength=s)
        self.steps += 1

    def update(self, z) -> None:
        """Multiply the belief by the likelihood of the beams z, then renormalise."""
        logl = ray_loglik(np.asarray(z)[None, :], self.expected, self.sigma, self.pdrop).sum(axis=1)
        live = self.belief > 0
        top = logl[live].max()
        # Subtracting the largest log-likelihood before exp keeps the numbers in range; the normalising sum is
        # the Bayes evidence p(z), which cancels here.
        self.belief = self.belief * np.exp(np.where(live, logl - top, -np.inf))
        self.belief /= self.belief.sum()

    def estimate(self):
        """The most probable bin: returns (x, y, th) of its centre, and its probability mass."""
        i = int(np.argmax(self.belief))
        return float(self.x[i]), float(self.y[i]), float(self.th[i]), float(self.belief[i])

    def position_map(self) -> np.ndarray:
        """Belief summed over heading, shape (h2, w2): the heat map the page draws."""
        return self.belief.reshape(self.h2, self.w2, HEADS).sum(axis=2)
