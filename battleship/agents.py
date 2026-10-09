"""Shooting strategies: each agent picks the next cell to fire at from what it knows.

Every agent has the same interface: choose(knowledge) returns an unknown cell. They differ only in how they
use the knowledge:
  - RandomAgent: any unknown cell. The floor every other agent should beat.
  - HuntAgent: the textbook baseline. Fire on a checkerboard (every ship spans at least two cells, so one of
    the two colours is always hit) until a hit appears, then fire next to the hit, extending along the line
    once two hits are aligned.
  - ProbabilityAgent: the Bayesian agent. Fire at the unknown cell most likely to hold a ship.
  - ChanceAgent: fixed odds. Fire at a random unknown cell, weighted by a fixed table of cell weights. It
    never looks at its hits or misses, so it is a dice roller rather than a reasoner.
"""

from __future__ import annotations

import random
from itertools import combinations

from .board import SIZE, Knowledge, bits
from .probability import SAMPLES, choose_cell, ship_probabilities


class Agent:
    """Base class: a strategy with its own random generator, so games are reproducible from a seed."""

    name = "agent"
    label = "agent"

    def __init__(self, rng: random.Random):
        self.rng = rng

    def choose(self, k: Knowledge) -> int:
        raise NotImplementedError


class RandomAgent(Agent):
    """Fires at a uniformly random unknown cell."""

    name = "random"
    label = "Random"

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
    label = "Hunt/target"

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
    label = "Probability (Bayesian)"

    def __init__(self, rng: random.Random, samples: int = SAMPLES):
        super().__init__(rng)
        self.samples = samples
        self.last = None

    def choose(self, k: Knowledge) -> int:
        self.last = ship_probabilities(k, self.rng, samples=self.samples)
        return choose_cell(self.last, k.unknown_cells(), self.rng)


# Chance's fixed odds. Each cell's weight is profile[row] * profile[col], where the profile rises from the edge
# to the middle, so a centre cell weighs 16 times a corner cell. The shape is fixed in advance and never changes
# with the game: it is the Battleship analogue of the 2048 spawner's fixed 90/10 split. The bias is not arbitrary:
# a ship can cover more placements through the middle of the board than along the edges, so a centre cell is
# slightly likelier to hold a ship. Chance does not know that; it just fires where the table says.
CHANCE_PROFILE = (1, 2, 3, 4, 4, 4, 4, 3, 2, 1)
CHANCE_WEIGHTS = tuple(CHANCE_PROFILE[c // SIZE] * CHANCE_PROFILE[c % SIZE] for c in range(SIZE * SIZE))


class ChanceAgent(Agent):
    """Fixed odds: draws an unknown cell with probability proportional to CHANCE_WEIGHTS.

    Only the cells still unknown are candidates, so every shot is legal, and the sampler renormalises the weights
    over them. There is no search and no lookahead: the only thing taken from the knowledge is which cells are
    still unknown, so hits and misses change nothing. With a seeded rng the sequence of shots is fixed.
    """

    name = "chance"
    label = "Chance (fixed odds)"

    def choose(self, k: Knowledge) -> int:
        unknown = k.unknown_cells()
        return self.rng.choices(unknown, weights=[CHANCE_WEIGHTS[c] for c in unknown])[0]


def chance_odds(k: Knowledge) -> list[float]:
    """Chance's probability of each cell on its next shot: the weights renormalised over the unknown cells.

    Fired cells are 0. The list is row-major, like ProbabilityAgent's belief, so the page can draw both the same way.
    """
    unknown = k.unknown_cells()
    total = sum(CHANCE_WEIGHTS[c] for c in unknown)
    odds = [0.0] * (SIZE * SIZE)
    for c in unknown:
        odds[c] = CHANCE_WEIGHTS[c] / total
    return odds


AGENTS = {"random": RandomAgent, "hunt": HuntAgent, "probability": ProbabilityAgent, "chance": ChanceAgent}


def make_agent(name: str, rng: random.Random) -> Agent:
    """Build a named agent. Raises KeyError for an unknown name."""
    return AGENTS[name](rng)
