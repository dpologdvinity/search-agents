"""The race: several optimizers start from one point on one surface and take their steps in lockstep.

All optimizers see the same gradient noise on every step (common random numbers), so differences between
them come from the update rules and not from luck. Each track records its path, its loss at every step,
the first step at which it reached a minimum (gradient below GTOL with a positive definite Hessian), and
the step at which it diverged (a non-finite value, or a position beyond BLOW). After divergence a track
stops moving; the browser lab draws the explosion at its last finite position.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from . import optimizers
from .optimizers import Hyper
from .rng import Rng
from .surfaces import Surface, is_minimum

GTOL = 1e-3  # gradient-norm threshold for "reached minimum"
BLOW = 1e6  # a position this large counts as divergence

# Default step sizes, one per optimizer. Chosen so each one behaves sensibly on the bowl; the behaviour on
# the ravine and Rosenbrock is measured by the tests and reported in the lab's "How it works" section.
DEFAULT_LR = {
    "sgd": 0.05,
    "momentum": 0.02,
    "nesterov": 0.02,
    "rmsprop": 0.05,
    "adam": 0.1,
    "adagrad": 0.5,
}
DEFAULT_HYPER = Hyper()


@dataclass
class Track:
    """One optimizer's run: its path, losses, and milestones."""

    name: str
    lr: float
    opt: optimizers.Optimizer
    x: float
    y: float
    traj: list[tuple[float, float]] = field(default_factory=list)
    losses: list[float] = field(default_factory=list)
    steps_to_tol: int | None = None
    diverged_at: int | None = None

    @property
    def diverged(self) -> bool:
        return self.diverged_at is not None


class Race:
    """Several optimizers on one surface, advanced together by step()."""

    def __init__(
        self,
        surface: Surface,
        start,
        names=optimizers.NAMES,
        lrs=None,
        hyper: Hyper = DEFAULT_HYPER,
        noise: float = 0.0,
        seed: int = 0,
        gtol: float = GTOL,
        blow: float = BLOW,
    ):
        lrs = dict(DEFAULT_LR if lrs is None else lrs)
        self.surface = surface
        self.hyper = hyper
        self.noise = float(noise)
        self.gtol = gtol
        self.blow = blow
        self.rng = Rng(seed)
        self.t = 0
        self.tracks: list[Track] = []
        x0, y0 = float(start[0]), float(start[1])
        for name in names:
            lr = float(lrs.get(name, DEFAULT_LR[name]))
            tr = Track(name, lr, optimizers.make(name, lr, hyper), x0, y0)
            tr.traj.append((x0, y0))
            tr.losses.append(surface.f(x0, y0))
            if is_minimum(surface, x0, y0, gtol):
                tr.steps_to_tol = 0
            self.tracks.append(tr)

    def step(self) -> None:
        """Advance every live track by one update, all sharing this step's noise draw."""
        self.t += 1
        nx = self.rng.normal()
        ny = self.rng.normal()
        surface = self.surface
        for tr in self.tracks:
            if tr.diverged:
                continue
            x = [tr.x, tr.y]
            q = tr.opt.query(x)
            gx, gy = surface.grad(q[0], q[1])
            g = [gx + self.noise * nx, gy + self.noise * ny]
            nxt = tr.opt.update(x, g)
            # Check the position before evaluating the loss: math.cos(inf) raises, and the loss is
            # only meaningful at a finite, in-range point anyway.
            if not _in_range(nxt[0], nxt[1], self.blow):
                tr.diverged_at = self.t
                continue
            loss = surface.f(nxt[0], nxt[1])
            if not math.isfinite(loss):
                tr.diverged_at = self.t
                continue
            tr.x, tr.y = nxt[0], nxt[1]
            tr.traj.append((tr.x, tr.y))
            tr.losses.append(loss)
            if tr.steps_to_tol is None and is_minimum(surface, tr.x, tr.y, self.gtol):
                tr.steps_to_tol = self.t

    def run(self, steps: int) -> Race:
        for _ in range(steps):
            self.step()
        return self

    def summary(self) -> list[dict]:
        """One row per track for the CLI table."""
        rows = []
        for tr in self.tracks:
            rows.append(
                {
                    "name": tr.name,
                    "lr": tr.lr,
                    "x": tr.x,
                    "y": tr.y,
                    "loss": tr.losses[-1],
                    "steps_to_tol": tr.steps_to_tol,
                    "diverged_at": tr.diverged_at,
                }
            )
        return rows


def _in_range(x: float, y: float, blow: float) -> bool:
    """A position is usable when both coordinates are finite and within the blow-up radius."""
    return math.isfinite(x) and math.isfinite(y) and abs(x) <= blow and abs(y) <= blow
