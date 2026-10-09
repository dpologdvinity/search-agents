"""The true robot, a route-following autopilot, and the mode count shown next to the particle cloud.

The true robot is the hidden state the filters try to recover. Its motion uses the same odometry noise the filters
assume, and its sensor is the same ray caster, plus the Gaussian noise and dropouts. The filters are told only the
commanded (turn, drive) and the beams; they never see the true pose, except through the beams.

The autopilot steers the robot along shortest routes between random destinations. It steers by the true pose,
which is what a real robot's controller would know about its own motion; the localisation problem is what it
does not know. The destinations come from the same PRNG stream as everything else, so runs repeat.
"""

from __future__ import annotations

import math
from collections import deque

import numpy as np

from .prng import Stream
from .world import TAU, Floor, motion, noise_sd, sense, wrap

NEIGHBOURS = ((0, -1), (1, 0), (0, 1), (-1, 0))   # N, E, S, W: the fixed order every search uses


class Sim:
    """The hidden robot: a pose, the odometry noise, and the most recent beams."""

    def __init__(self, floor: Floor, seed: int, rays: int = 8, sigma: float = 0.3, pdrop: float = 0.05,
                 motion_scale: float = 1.0):
        self.floor = floor
        self.rays = rays
        self.sigma = sigma
        self.pdrop = pdrop
        self.scale = motion_scale
        self.stream = Stream(seed)
        self.x = self.y = self.th = 0.0
        self.z = None
        self.place_random()
        self.z = sense(floor.walls, self.x, self.y, self.th, rays, sigma, pdrop, self.stream)

    def place_random(self):
        """Put the robot at a uniform random pose in the floor's main region. Also used for kidnapping."""
        cells = self.floor.region_cells()
        u = self.stream.uniforms(4)
        k = min(len(cells) - 1, int(u[0] * len(cells)))
        cx, cy = cells[k]
        self.x = cx + float(u[1])
        self.y = cy + float(u[2])
        self.th = float(u[3]) * TAU - math.pi
        return self.x, self.y, self.th

    def kidnap(self):
        """Teleport the robot. Nothing tells the filters; they only notice through the next beams."""
        self.place_random()
        self.z = sense(self.floor.walls, self.x, self.y, self.th, self.rays, self.sigma, self.pdrop, self.stream)
        return self.z

    def drive(self, rot: float, fwd: float):
        """Apply one commanded (turn, drive) with odometry noise, then take a new set of beams."""
        n = self.stream.normals(2)
        sr, st = noise_sd(rot, fwd, self.scale)
        x, y, th = motion(self.floor.walls, self.x, self.y, self.th, rot + sr * n[0], fwd + st * n[1])
        self.x, self.y, self.th = float(x), float(y), float(th)
        self.z = sense(self.floor.walls, self.x, self.y, self.th, self.rays, self.sigma, self.pdrop, self.stream)
        return self.z


def bfs_path(walls: np.ndarray, start: tuple, goal: tuple) -> list:
    """Shortest 4-connected path between two free cells, excluding the start. Empty when there is none.
    Neighbours are tried in NEIGHBOURS order, so ties break the same way in every run and in JavaScript."""
    if start == goal:
        return []
    h, w = walls.shape
    prev = {start: None}
    q = deque([start])
    while q:
        c = q.popleft()
        for dx, dy in NEIGHBOURS:
            n = (c[0] + dx, c[1] + dy)
            if not (0 <= n[0] < w and 0 <= n[1] < h) or walls[n[1], n[0]] or n in prev:
                continue
            prev[n] = c
            if n == goal:
                path = [n]
                while prev[path[-1]] != start:
                    path.append(prev[path[-1]])
                return path[::-1]
            q.append(n)
    return []


class Autopilot:
    """Drives the true robot along BFS routes to random destinations. Returns (turn, drive) commands."""

    MAX_TURN = 0.25   # radians per step
    SPEED = 0.3       # cells per step (less than a cell, so the robot cannot tunnel through a wall)
    REACH = 0.35      # a waypoint counts as reached within this distance (cells)
    FACE = 0.5        # the robot only drives forward once it faces the waypoint within this angle

    def __init__(self, floor: Floor, seed: int):
        self.floor = floor
        self.stream = Stream(seed)
        self.route: list = []

    def _new_route(self, x: float, y: float) -> None:
        """Pick a random destination in the main region that has a route; try a few before giving up."""
        cells = self.floor.region_cells()
        start = (int(math.floor(x)), int(math.floor(y)))
        for _ in range(8):
            goal = cells[int(self.stream.index(len(cells))[0])]
            path = bfs_path(self.floor.walls, start, goal)
            if path:
                self.route = path
                return
        self.route = []

    def command(self, x: float, y: float, th: float):
        if not self.route:
            self._new_route(x, y)
            if not self.route:
                return 0.0, 0.0
        tx, ty = self.route[0][0] + 0.5, self.route[0][1] + 0.5
        if math.hypot(tx - x, ty - y) < self.REACH:
            self.route.pop(0)
            return 0.0, 0.0   # stop for one step; the next waypoint is aimed at on the following call
        err = float(wrap(math.atan2(ty - y, tx - x) - th))
        rot = max(-self.MAX_TURN, min(self.MAX_TURN, err))
        fwd = self.SPEED if abs(err) < self.FACE else 0.0
        return rot, fwd


def count_modes(hist: dict, threshold: float = 0.05) -> int:
    """Number of separate places the particles sit in: connected groups (4-neighbour) of 1-cell bins that each hold
    at least `threshold` of the total weight. This is the "belief has split" indicator the page shows."""
    live = {k for k, v in hist.items() if v >= threshold}
    seen: set = set()
    groups = 0
    for c in live:
        if c in seen:
            continue
        groups += 1
        stack = [c]
        seen.add(c)
        while stack:
            cx, cy = stack.pop()
            for dx, dy in NEIGHBOURS:
                n = (cx + dx, cy + dy)
                if n in live and n not in seen:
                    seen.add(n)
                    stack.append(n)
    return groups
