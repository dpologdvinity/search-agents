"""Four agents of rising strength, and the loop that plays one seeded game.

Each agent sees only a View (the numbers on the board and the mine count). Every move is either
a certain reveal, which the agent has proven safe, or a guess, which it made because nothing was
proven:

random       a uniformly random covered cell. The baseline: no reasoning at all.
rules        level 1. Single-cell rules, applied until nothing new appears. Guesses at random.
csp          level 2. Adds the constraint components and the global mine count, so it proves
             every cell that is safe or a mine in all layouts that fit. Guesses at random.
probability  level 3. The same proofs, and when nothing is proven it guesses the covered cell with
             the lowest exact mine probability. Ties go to the cell with the most covered
             neighbours, since a zero there would open the most board.

A game starts with a click chosen from the seed, which places the mines around it (see board.py),
so every agent plays the same board for the same seed. That first click is free and is not a guess.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass

from .board import UNKNOWN, Game, View, label
from .inference import LEVEL_CSP, LEVEL_PROBABILITY, LEVEL_RULES, Analysis, analyse

AGENT_NAMES = ("random", "rules", "csp", "probability")
AGENT_LEVEL = {"rules": LEVEL_RULES, "csp": LEVEL_CSP, "probability": LEVEL_PROBABILITY}
DESCRIPTIONS = {
    "random": "a uniformly random covered cell, with no reasoning",
    "rules": "single-cell rules to a fixpoint; random guess when nothing is proven",
    "csp": "constraint components with exact layout counts; random guess when nothing is proven",
    "probability": "constraint components, then the lowest exact mine probability when nothing is proven",
}


@dataclass(frozen=True)
class Choice:
    """One move: the cell, whether it is proven safe, its mine probability if the agent computed it, and why."""

    cell: int
    certain: bool
    p_mine: float | None
    why: str


def best_guess(analysis: Analysis) -> int:
    """The covered cell with the lowest exact mine probability.

    Ties (common on the interior, where every cell has the same probability) go to the cell with the
    most covered neighbours: a zero there would reveal the most, which is the opening the agent wants.
    The final tie-break on the index keeps the choice deterministic.
    """
    candidates = [c for c in analysis.covered if c in analysis.numerator]
    if not candidates:
        raise ValueError("no covered cell has a computed probability")
    return min(candidates, key=lambda c: (analysis.numerator[c], -analysis.unknown_neighbours(c), c))


class Agent:
    """Chooses moves for one game. `rng` only drives the random guesses and the random baseline."""

    def __init__(self, kind: str, rng: random.Random | None = None):
        if kind not in AGENT_NAMES:
            raise ValueError(f"unknown agent {kind!r}; choose from {', '.join(AGENT_NAMES)}")
        self.kind = kind
        self.rng = rng or random.Random()

    def choose(self, view: View) -> Choice:
        if self.kind == "random":
            covered = [i for i, v in enumerate(view.cells) if v == UNKNOWN]
            return Choice(self.rng.choice(covered), False, None, "a random covered cell")
        level = AGENT_LEVEL[self.kind]
        analysis = analyse(view, level)
        if analysis.safe:
            cell = analysis.safe[0]
            p = analysis.probability(cell)
            return Choice(cell, True, 0.0 if p is None else p, analysis.why.get(cell, "proven safe"))
        if level == LEVEL_PROBABILITY:
            cell = best_guess(analysis)
            p = analysis.probability(cell)
            return Choice(cell, False, p, f"lowest mine probability: {p:.1%}")
        # Rules and CSP have no ranking of covered cells, so they guess uniformly among the candidates
        # (analysis.covered already leaves out the proven mines).
        cell = self.rng.choice(analysis.covered)
        return Choice(cell, False, analysis.probability(cell), "nothing proven: a random guess")


@dataclass(frozen=True)
class GameResult:
    """How one game ended. `guesses` counts moves that were not proven safe; the first click is not counted."""

    won: bool
    guesses: int
    reveals: int
    seconds: float


def play(rows: int, cols: int, mines: int, seed: int, kind: str, max_moves: int | None = None) -> GameResult:
    """Play one game to the end with agent `kind`, on the board that `seed` determines.

    The seed fixes the first click and the mine layout around it. The agent gets its own generator,
    derived from the seed, so an agent's random choices do not depend on the board.
    """
    rng = random.Random(seed)
    game = Game(rows, cols, mines, rng)
    game.reveal(rng.randrange(rows * cols))  # the first click: always safe
    agent = Agent(kind, random.Random(seed * 7919 + 1))
    cap = max_moves if max_moves is not None else 2 * rows * cols
    guesses = 0
    t0 = time.perf_counter()
    steps = 0
    while not game.over and steps < cap:
        choice = agent.choose(game.view())
        if not choice.certain:
            guesses += 1
        game.reveal(choice.cell)
        steps += 1
    seconds = time.perf_counter() - t0
    return GameResult(game.won, guesses, game.reveals, seconds)


def describe_move(view: View, choice: Choice) -> str:
    """One line for the log: the cell, whether it was proven, and the reason."""
    kind = "certain" if choice.certain else "guess"
    return f"{kind}: {label(view.cols, choice.cell)} ({choice.why})"
