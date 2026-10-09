"""The floor plans, the robot's motion and sensor models, and the ray caster that all three estimators share.

Units: one grid cell is CELL_M = 0.5 m. A pose is continuous: x and y in cells (x grows to the right, y down),
th in radians with th = 0 along +x. A cell is solid when its wall flag is set, and any point outside the floor
counts as a wall, so a ray can never escape the map. The robot is a point; walls are whole cells.

Floor plans are hand-drawn rectangles plus a list of pillar slots. The seed decides which slots hold a pillar,
so a seed gives a different floor from the same layout while the corridors and rooms stay fixed. The layouts
are duplicated in web/js/localize-core.js; tests/test_localize_parity.py checks the two copies agree.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .prng import Stream

CELL_M = 0.5
MAX_RANGE = 8.0       # cells (4 m); a ray that hits nothing reads this value
RAY_STEP = 0.25       # cells per step of the ray march (the hit distance is quantised to this)
PILLAR_PROB = 0.5     # chance that each pillar slot holds a pillar in a seeded floor
OUTLIER = 0.02        # mixture weight of a uniform "unexpected reading" term in the range likelihood
TAU = 2.0 * math.pi

# Default odometry noise: standard deviations grow with how far the robot turns or drives. Scaled by the
# motion-noise slider. These are the values the filters assume, so the page's "motion noise" moves both.
ROT_NOISE = (0.05, 0.15)     # radians: base, plus this fraction of |turn|
FWD_NOISE = (0.03, 0.10)     # cells: base, plus this fraction of |drive|

# Two floors. Walls are inclusive rectangles (x0, y0, x1, y1); pillar slots are single cells.
# "halls" has six identical rooms (two rows of three) that look the same from inside, so its belief
# splits between them until a feature shows up; the fourth room on the right is the only one that differs.
# "vault" is an irregular plan with a few long walls and no symmetry.
LAYOUTS = {
    "halls": {
        "title": "TWIN HALLS",
        "w": 24,
        "h": 16,
        "walls": [
            (0, 0, 23, 0), (0, 15, 23, 15), (0, 0, 0, 15), (23, 0, 23, 15),          # outer wall
            # top wall: gaps (doors) at x = 3, 9, 15 and 20
            (1, 6, 2, 6), (4, 6, 8, 6), (10, 6, 14, 6), (16, 6, 19, 6), (21, 6, 22, 6),
            # bottom wall: the same doors
            (1, 9, 2, 9), (4, 9, 8, 9), (10, 9, 14, 9), (16, 9, 19, 9), (21, 9, 22, 9),
            (6, 1, 6, 5), (12, 1, 12, 5), (18, 1, 18, 5),                             # top room dividers
            (6, 10, 6, 14), (12, 10, 12, 14), (18, 10, 18, 14),                       # bottom room dividers
        ],
        "slots": [(21, 3), (6, 7), (16, 8), (15, 12)],
    },
    "vault": {
        "title": "THE VAULT",
        "w": 24,
        "h": 16,
        "walls": [
            (0, 0, 23, 0), (0, 15, 23, 15), (0, 0, 0, 15), (23, 0, 23, 15),
            (5, 1, 5, 6), (10, 4, 10, 11), (15, 1, 15, 8), (19, 6, 19, 14),
            (5, 10, 8, 10), (12, 13, 14, 13),
        ],
        "slots": [(2, 3), (7, 8), (13, 2), (17, 11), (21, 4)],
    },
}


@dataclass
class Floor:
    """One floor plan. `walls` is a (h, w) boolean array; `region` marks the largest connected free area,
    where the robot is placed and where the autopilot picks its destinations."""

    key: str
    title: str
    seed: int
    w: int
    h: int
    walls: np.ndarray
    pillars: list
    region: np.ndarray

    def free_cells(self) -> list:
        """Every free cell as (x, y), in row-major order. The order matters: both languages index it the same way."""
        return [(x, y) for y in range(self.h) for x in range(self.w) if not self.walls[y, x]]

    def region_cells(self) -> list:
        """Free cells of the largest connected area, row-major."""
        return [(x, y) for y in range(self.h) for x in range(self.w) if self.region[y, x]]


def _largest_component(walls: np.ndarray) -> np.ndarray:
    """Mask of the largest 4-connected group of free cells. A pillar can cut a corridor, so the robot is only
    ever placed in the part of the floor it can drive around in."""
    h, w = walls.shape
    seen = np.zeros_like(walls)
    best: list = []
    for y in range(h):
        for x in range(w):
            if walls[y, x] or seen[y, x]:
                continue
            comp, stack = [], [(x, y)]
            seen[y, x] = True
            while stack:
                cx, cy = stack.pop()
                comp.append((cx, cy))
                for nx, ny in ((cx, cy - 1), (cx + 1, cy), (cx, cy + 1), (cx - 1, cy)):
                    if 0 <= nx < w and 0 <= ny < h and not walls[ny, nx] and not seen[ny, nx]:
                        seen[ny, nx] = True
                        stack.append((nx, ny))
            if len(comp) > len(best):
                best = comp
    mask = np.zeros_like(walls)
    for cx, cy in best:
        mask[cy, cx] = True
    return mask


def make_floor(key: str, seed: int = 1) -> Floor:
    """Build a floor from a layout and a seed. Each pillar slot draws one uniform, in slot order, and holds a
    pillar when the draw is below PILLAR_PROB. The draws are made whether or not a pillar results, so the JS
    port can follow the same stream."""
    lay = LAYOUTS[key]
    w, h = lay["w"], lay["h"]
    walls = np.zeros((h, w), dtype=bool)
    for x0, y0, x1, y1 in lay["walls"]:
        walls[y0:y1 + 1, x0:x1 + 1] = True
    stream = Stream(seed)
    pillars = []
    for x, y in lay["slots"]:
        if stream.uniforms(1)[0] < PILLAR_PROB:
            walls[y, x] = True
            pillars.append((x, y))
    return Floor(key=key, title=lay["title"], seed=seed, w=w, h=h, walls=walls, pillars=pillars,
                 region=_largest_component(walls))


def blocked(walls: np.ndarray, x, y):
    """True where the point (x, y) is inside a wall or outside the floor. Works on arrays or scalars."""
    h, w = walls.shape
    i = np.floor(x).astype(np.int64)
    j = np.floor(y).astype(np.int64)
    out = (i < 0) | (i >= w) | (j < 0) | (j >= h)
    return out | walls[np.clip(j, 0, h - 1), np.clip(i, 0, w - 1)]


def wrap(a):
    """Map an angle into [-pi, pi). Written with floor so that JavaScript's Math.floor gives the same value."""
    return a - TAU * np.floor((a + math.pi) / TAU)


def noise_sd(rot, fwd, scale: float):
    """Standard deviations of the odometry noise for a commanded turn (rad) and drive (cells)."""
    sr = scale * (ROT_NOISE[0] + ROT_NOISE[1] * abs(rot))
    st = scale * (FWD_NOISE[0] + FWD_NOISE[1] * abs(fwd))
    return sr, st


def motion(walls: np.ndarray, x, y, th, rot, fwd):
    """One odometry step: drive fwd along the current heading, then turn by rot. A drive whose end point is in a
    wall is cancelled (the robot bumps and stays put), but the turn still happens. Steps are shorter than a cell,
    so the robot cannot jump over a wall between two checks."""
    nx = x + fwd * np.cos(th)
    ny = y + fwd * np.sin(th)
    bump = blocked(walls, nx, ny)
    x2 = np.where(bump, x, nx)
    y2 = np.where(bump, y, ny)
    return x2, y2, wrap(th + rot)


def erfc_pos(x):
    """erfc for x >= 0 (Numerical Recipes erfcc, error below 1.2e-7). Used for the chance that a noisy
    reading would have exceeded the maximum range. It is written out, not taken from math.erfc, so the
    JavaScript port computes the same numbers."""
    t = 1.0 / (1.0 + 0.5 * x)
    poly = (-1.26551223 + t * (1.00002368 + t * (0.37409196 + t * (0.09678418 + t * (-0.18628806 + t * (
        0.27886807 + t * (-1.13520398 + t * (1.48851587 + t * (-0.82215223 + t * 0.17087277)))))))))
    return t * np.exp(-x * x + poly)


def cast(walls: np.ndarray, x, y, ang, rmax: float = MAX_RANGE, step: float = RAY_STEP):
    """Distance from each origin (x, y) along each angle to the first wall, capped at rmax.

    Marches every ray in steps of `step` cells and records the first step that lands in a wall. The distance is
    the step count times `step`, so it is quantised to that step. All rays advance together, which is what keeps
    this fast enough for thousands of particles."""
    h, w = walls.shape
    x = np.asarray(x, dtype=float).copy()
    y = np.asarray(y, dtype=float).copy()
    dx = np.cos(ang) * step
    dy = np.sin(ang) * step
    d = np.full(x.shape, rmax)
    hit = np.zeros(x.shape, dtype=bool)
    for k in range(1, int(round(rmax / step)) + 1):
        x = x + dx
        y = y + dy
        i = np.floor(x).astype(np.int64)
        j = np.floor(y).astype(np.int64)
        bl = (i < 0) | (i >= w) | (j < 0) | (j >= h) | walls[np.clip(j, 0, h - 1), np.clip(i, 0, w - 1)]
        new = bl & ~hit
        d[new] = k * step
        hit |= bl
        if hit.all():
            break
    return d


def beam_angles(th, rays: int):
    """Headings of the beams: `rays` evenly spaced around the full circle, starting at the robot's heading."""
    return th[..., None] + (TAU * np.arange(rays)) / rays


def ray_loglik(z, e, sigma: float, pdrop: float, rmax: float = MAX_RANGE):
    """Log-likelihood of each measured range z given the expected range e (both in cells).

    A reading equal to rmax is a "no return": either a dropout (probability pdrop) or a true range so long that
    noise pushed the reading past the limit. Anything shorter is a Gaussian around the expected range, mixed with
    a small uniform floor (OUTLIER) so one surprising beam cannot erase a particle entirely."""
    gauss = (1.0 - pdrop) * np.exp(-0.5 * ((z - e) / sigma) ** 2) / (sigma * math.sqrt(2.0 * math.pi)) \
        + OUTLIER / rmax
    tail = 0.5 * erfc_pos((rmax - e) / (sigma * math.sqrt(2.0)))
    maxp = pdrop + (1.0 - pdrop) * tail + OUTLIER / rmax
    return np.log(np.where(z >= rmax, maxp, gauss))


def sense(walls: np.ndarray, x, y, th, rays: int, sigma: float, pdrop: float, stream: Stream):
    """Simulate the range sensor on the true robot: the true ray distance, plus Gaussian noise, with dropouts
    reading the maximum. Draws all dropout uniforms first, then all the noise normals, in beam order."""
    d = cast(walls, np.full(rays, float(x)), np.full(rays, float(y)), float(th) + (TAU * np.arange(rays)) / rays)
    u = stream.uniforms(rays)
    n = stream.normals(rays)
    z = np.clip(d + sigma * n, 0.0, MAX_RANGE)
    return np.where(u < pdrop, MAX_RANGE, z)
