"""Monte Carlo localisation (a particle filter) with kidnapped-robot recovery (augmented MCL).

A particle is one guess at the robot's pose (x, y, th). Each step:

  1. motion    every particle is moved by the commanded odometry plus its own noise sample. That draws from the same
               model the true robot obeys, so the cloud spreads out the way the real uncertainty does.
  2. sensing   each particle is scored by how well its own ray cast matches the measured beams. The score is a product
               of per-beam likelihoods, computed in log space because 8 or 16 beams underflow quickly.
  3. resample  when the effective sample size ESS = 1 / sum(w^2) falls below N/2, particles are redrawn in proportion
               to their weights with systematic (low-variance) resampling: one random offset, then N evenly spaced
               pointers. Good particles are copied, bad ones die, and the weights reset to 1/N.
  4. recover   augmented MCL (Thrun, Burgard and Fox, ch. 8.3). Two running averages of the measurement likelihood
               are kept, a fast one and a slow one. When the fast average falls well below the slow one, the robot
               has probably been kidnapped (or the map and the sensor disagree), so a fraction of the particles is
               replaced by poses drawn uniformly over the floor. The fraction is 1 - fast/slow, from the ratio of
               the two.

The estimate reports the densest cluster of particles, not the mean of the whole cloud. With two possible
locations the mean would fall between them, in a wall; the cluster around the heaviest particle is a place the robot
could actually be, and its covariance is the uncertainty ellipse the page draws.
"""

from __future__ import annotations

import math

import numpy as np

from .prng import Stream
from .world import TAU, Floor, beam_angles, cast, motion, noise_sd, ray_loglik, wrap

ALPHA_FAST = 0.1      # smoothing of the fast likelihood average (per step)
ALPHA_SLOW = 0.01     # smoothing of the slow average
KIDNAP_NATS = 2.5     # the fast average must fall this far (in log units) below the slow one before injecting
CLUSTER_R = 1.5       # cells: particles this close to the heaviest one, and within CLUSTER_ANGLE, form its cluster
CLUSTER_ANGLE = 0.6   # radians


class ParticleFilter:
    """A set of N weighted pose guesses over the free cells of `floor`."""

    def __init__(self, floor: Floor, n: int, rays: int, sigma: float, pdrop: float, motion_scale: float,
                 seed: int, augmented: bool = True):
        self.floor = floor
        self.n = n
        self.rays = rays
        self.sigma = sigma
        self.pdrop = pdrop
        self.scale = motion_scale
        self.augmented = augmented
        self.stream = Stream(seed)
        cells = floor.free_cells()
        self.cx = np.array([c[0] for c in cells], dtype=float)
        self.cy = np.array([c[1] for c in cells], dtype=float)
        # Global start: every particle is a uniform draw over the free cells and headings. Four uniforms each:
        # which cell, the offset inside it (x, y), and the heading.
        u = self.stream.uniforms(4 * n).reshape(n, 4)
        k = np.minimum(len(cells) - 1, (u[:, 0] * len(cells)).astype(np.int64))
        self.x = self.cx[k] + u[:, 1]
        self.y = self.cy[k] + u[:, 2]
        self.th = u[:, 3] * TAU - math.pi
        self.w = np.full(n, 1.0 / n)
        self.ess = float(n)
        self.p_inject = 0.0
        self.injected = 0
        self.resampled = False
        self.ls = None   # slow log-likelihood average
        self.lf = None   # fast log-likelihood average
        self.steps = 0

    def _sample_cells(self, m: int) -> None:
        """Helper for injection: returns m fresh uniform poses over the free cells, as (x, y, th)."""
        u = self.stream.uniforms(4 * m).reshape(m, 4)
        k = np.minimum(len(self.cx) - 1, (u[:, 0] * len(self.cx)).astype(np.int64))
        return self.cx[k] + u[:, 1], self.cy[k] + u[:, 2], u[:, 3] * TAU - math.pi

    def step(self, rot: float, fwd: float, z) -> None:
        """One predict + update cycle for the commanded (rot, fwd) and the measured beams z."""
        n = self.n
        walls = self.floor.walls
        # 1. Motion: two normals per particle, the turn noise first, then the drive noise.
        nn = self.stream.normals(2 * n)
        sr, st = noise_sd(rot, fwd, self.scale)
        self.x, self.y, self.th = motion(walls, self.x, self.y, self.th,
                                         rot + sr * nn[0::2], fwd + st * nn[1::2])
        # 2. Sensing: each particle's expected beams, then its log-likelihood summed over beams.
        ang = beam_angles(self.th, self.rays)
        e = cast(walls, np.repeat(self.x, self.rays), np.repeat(self.y, self.rays), ang.ravel()).reshape(n, self.rays)
        logl = ray_loglik(np.asarray(z)[None, :], e, self.sigma, self.pdrop).sum(axis=1)
        top = logl.max()
        lik = np.exp(logl - top)
        # Average likelihood this step, in log form: log(mean(exp(logl))) = top + log(mean(lik)).
        lavg = top + math.log(lik.mean())
        if self.ls is None:
            self.ls = self.lf = lavg
        else:
            self.ls += ALPHA_SLOW * (lavg - self.ls)
            self.lf += ALPHA_FAST * (lavg - self.lf)
        # Injection fraction: zero unless the fast average has dropped KIDNAP_NATS below the slow one.
        drop = self.lf - self.ls
        self.p_inject = (1.0 - math.exp(drop)) if (self.augmented and drop < -KIDNAP_NATS) else 0.0
        self.p_inject = min(1.0, max(0.0, self.p_inject))
        self.w = self.w * lik
        self.w /= self.w.sum()
        # 3. Resample when the weights have concentrated on too few particles.
        self.ess = 1.0 / float((self.w * self.w).sum())
        self.resampled = self.ess < n / 2.0
        if self.resampled:
            self._systematic_resample()
        # 4. Recovery: replace the lowest-weight particles with uniform draws.
        self.injected = int(round(self.p_inject * n))
        if self.injected > 0:
            idx = np.argsort(self.w, kind="stable")[: self.injected]
            x, y, th = self._sample_cells(self.injected)
            self.x[idx], self.y[idx], self.th[idx] = x, y, th
            self.w[idx] = 1.0 / n
            self.w /= self.w.sum()
        self.steps += 1

    def _systematic_resample(self) -> None:
        """Low-variance resampling: one uniform u0 in [0, 1/N), then pointers u0 + i/N through the cumulative
        weights. Each particle is copied about N * w times, and the copies are spread evenly, so the variance is
        lower than N independent draws."""
        n = self.n
        u0 = self.stream.uniforms(1)[0] / n
        pointers = u0 + np.arange(n) / n
        cum = np.cumsum(self.w)
        cum[-1] = 1.0
        idx = np.minimum(np.searchsorted(cum, pointers, side="left"), n - 1)
        self.x, self.y, self.th = self.x[idx], self.y[idx], self.th[idx]
        self.w = np.full(n, 1.0 / n)

    def estimate(self) -> dict:
        """The densest cluster: the heaviest particle plus the particles near it, averaged with their weights.

        Returns the pose (x, y, th), the cluster's share of the total weight, its covariance in cells (for the
        uncertainty ellipse), and the particle index of the heaviest one."""
        i = int(np.argmax(self.w))
        dist = np.hypot(self.x - self.x[i], self.y - self.y[i])
        dang = np.abs(wrap(self.th - self.th[i]))
        mask = (dist < CLUSTER_R) & (dang < CLUSTER_ANGLE)
        ww = self.w * mask
        mass = float(ww.sum())
        ww = ww / mass
        ex = float((ww * self.x).sum())
        ey = float((ww * self.y).sum())
        eth = math.atan2(float((ww * np.sin(self.th)).sum()), float((ww * np.cos(self.th)).sum()))
        dx = self.x - ex
        dy = self.y - ey
        cov = [[float((ww * dx * dx).sum()), float((ww * dx * dy).sum())],
               [float((ww * dx * dy).sum()), float((ww * dy * dy).sum())]]
        return {"x": ex, "y": ey, "th": eth, "mass": mass, "cov": cov, "heaviest": i}

    def position_histogram(self) -> dict:
        """Particle weight per 1-cell bin: used for the mode count. Keys are (bx, by), values are weights."""
        out: dict = {}
        bx = np.floor(self.x).astype(np.int64)
        by = np.floor(self.y).astype(np.int64)
        for a, b, w in zip(bx.tolist(), by.tolist(), self.w.tolist()):
            out[(a, b)] = out.get((a, b), 0.0) + w
        return out
