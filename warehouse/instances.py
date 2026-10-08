"""Random multi-robot instances that are guaranteed solvable.

An instance is drawn from the largest connected floor region of a map, so every robot can reach
every cell. Starts and goals are all distinct cells, so no robot begins on another's goal. Not every
draw is solvable (robots can box each other into a dead end), so each candidate is checked with CBS
under a node budget. Only instances CBS solves are returned. That makes the instance reproducible
from its seed, and the check doubles as the proof that a collision-free plan exists.
"""

from __future__ import annotations

import random
from collections.abc import Sequence

from .cbs import cbs
from .grid import Grid, Problem, layout_grid


def random_instance(layout: str | Grid | Sequence[str], robots: int, seed: int, *, attempts: int = 30,
                    max_nodes: int = 3000, max_seconds: float = 5.0) -> Problem:
    """A solvable instance with ``robots`` robots on a built-in layout name, a ``Grid``, or raw map rows."""
    if isinstance(layout, str):
        grid = layout_grid(layout)
    elif isinstance(layout, Grid):
        grid = layout
    else:
        grid = Grid(list(layout))
    floor = grid.largest_component()
    if len(floor) < 2 * robots:
        raise ValueError(f"the map has too few floor cells for {robots} robots")
    rng = random.Random(seed)
    for _ in range(attempts):
        cells = rng.sample(floor, 2 * robots)
        problem = Problem(grid, tuple(cells[:robots]), tuple(cells[robots:]))
        if cbs(problem, max_nodes=max_nodes, max_seconds=max_seconds, trace_limit=0).status == "solved":
            return problem
    raise ValueError(f"no solvable instance with {robots} robots found in {attempts} tries")
