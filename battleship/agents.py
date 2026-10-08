"""Shooting strategies: each agent picks the next cell to fire at from what it knows.

Every agent has the same interface: choose(knowledge) returns an unknown cell. They differ only in how they
use the knowledge:
  - RandomAgent: any unknown cell. The floor every other agent should beat.
  - HuntAgent: the textbook baseline. Fire on a checkerboard (every ship spans at least two cells, so one of
    the two colours is always hit) until a hit appears, then fire next to the hit, extending along the line
    once two hits are aligned.
  - ProbabilityAgent: the Bayesian agent. Fire at the unknown cell most likely to hold a ship.
"""

from __future__ import annotations

import random
from itertools import combinations

from .board import SIZE, Knowledge, bits
from .probability import SAMPLES, choose_cell, ship_probabilities


class Agent:
    """Base class: a strategy with its own random generator, so games are reproducible from a seed."""

    name = "agent"

    def __init__(self, rng: random.Random):
        self.rng = rng

    def choose(self, k: Knowledge) -> int:
        raise NotImplementedError


class RandomAgent(Agent):
    """Fires at a uniformly random unknown cell."""

    name = "random"

    def choose(self, k: Knowledge) -> int:
        return self.rng.choice(k.unknown_cells())


def _neighbours(c: int) -> list[int]:
    """Orthogonal neighbours of cell c that are on the board."""
    r, col = divmod(c, SIZE)
    out = []
    if r > 0:
        out.append(c - SIZE)
    if r < SIZE - 1:
        out.append(c + SIZE)
    if col > 0:
        out.append(c - 1)
    if col < SIZE - 1:
        out.append(c + 1)
    return out


class HuntAgent(Agent):
    """Hunt/target: checkerboard search, then finish off wounded ships."""

    name = "hunt"

    def choose(self, k: Knowledge) -> int:
        fired = k.fired
        targets = _line_targets(k, fired)
        if targets:
            return self.rng.choice(targets)
        unknown = k.unknown_cells()
        # Checkerboard: (row + col) even. Any ship of length 2 or more covers an even cell.
        parity = [c for c in unknown if (c // SIZE + c % SIZE) % 2 == 0]
        return self.rng.choice(parity or unknown)


def _line_targets(k: Knowledge, fired: int) -> list[int]:
    """Cells to fire at next to the open hits, or [] if there are no open hits.

    If two open hits share a row or column, the ship lies along that line, so only the cells on the line
    (between the hits and just beyond them) are candidates. Otherwise every unknown neighbour of an open hit
    is a candidate.
    """
    open_hits = list(bits(k.open_hits))  # ascending
    if not open_hits:
        return []
    for a, b in combinations(open_hits, 2):
        if a // SIZE == b // SIZE:
            step = 1
        elif a % SIZE == b % SIZE:
            step = SIZE
        else:
            continue
        line = list(range(a, b + 1, step))
        beyond = [line[0] - step, line[-1] + step]
        cands = [c for c in line + beyond if 0 <= c < SIZE * SIZE and not (fired >> c) & 1
                 and _same_line(a, c, step)]
        if cands:
            return cands
    cands = {n for h in open_hits for n in _neighbours(h) if not (fired >> n) & 1}
    return sorted(cands)


def _same_line(a: int, c: int, step: int) -> bool:
    """True if cell c is on the same row (step 1) or column (step SIZE) as cell a."""
    return (c // SIZE == a // SIZE) if step == 1 else (c % SIZE == a % SIZE)


class ProbabilityAgent(Agent):
    """Bayesian targeting: fire at the unknown cell with the highest probability of holding a ship.

    `last` keeps the belief from the most recent choice, so a display can show the odds that drove the shot.
    """

    name = "probability"

    def __init__(self, rng: random.Random, samples: int = SAMPLES):
        super().__init__(rng)
        self.samples = samples
        self.last = None

    def choose(self, k: Knowledge) -> int:
        self.last = ship_probabilities(k, self.rng, samples=self.samples)
        return choose_cell(self.last, k.unknown_cells(), self.rng)


AGENTS = {"random": RandomAgent, "hunt": HuntAgent, "probability": ProbabilityAgent}


def make_agent(name: str, rng: random.Random) -> Agent:
    """Build a named agent. Raises KeyError for an unknown name."""
    return AGENTS[name](rng)
