"""Warehouse robots: multi-agent pathfinding with Conflict-Based Search and two baselines.

Several robots each have a start and a goal on a grid of shelves. They move in discrete time
steps (up, down, left, right, or wait). No two robots may share a cell at the same time, and no
two may swap cells. Modules:

- ``grid``: the map model, the built-in layouts, and ``Problem`` (starts and goals).
- ``conflicts``: path positions, sum of costs, and collision detection.
- ``space_time``: single-robot space-time A* with reservations (the CBS low level).
- ``cbs``: Conflict-Based Search, the optimal high level.
- ``planners``: prioritized planning, independent A*, and the ``PlanResult`` they return.
- ``instances``: random solvable instances for the CLI, benchmark, and web page.

``solve`` is the one entry point the CLI, benchmark, and server use.
"""

from __future__ import annotations

from .cbs import cbs
from .conflicts import Conflict, find_conflicts, makespan, sum_of_costs
from .grid import LAYOUT_INFO, LAYOUTS, Grid, Problem, layout_grid
from .instances import random_instance
from .planners import PlanResult, independent, prioritized

PLANNERS = ("cbs", "prioritized", "independent")

# Human-readable names for the CLI, the benchmark table, and the web page.
PLANNER_INFO = {
    "cbs": {
        "title": "Conflict-Based Search",
        "description": (
            "Optimal sum of costs. Searches a tree of collision constraints; "
            "each node replans only the robots it constrains."
        ),
    },
    "prioritized": {
        "title": "Prioritized planning",
        "description": (
            "Plans robots one at a time, treating earlier paths as moving obstacles. "
            "Fast, but can fail or cost more."
        ),
    },
    "independent": {
        "title": "Independent A*",
        "description": (
            "Each robot ignores the others. Fastest, and the collisions it leaves are what CBS solves."
        ),
    },
}


def solve(problem: Problem, planner: str = "cbs", **limits) -> PlanResult:
    """Run one planner. ``limits`` (``max_nodes``, ``max_seconds``) only apply to CBS."""
    if planner == "cbs":
        return cbs(problem, **limits)
    if planner == "prioritized":
        return prioritized(problem)
    if planner == "independent":
        return independent(problem)
    raise ValueError(f"unknown planner {planner!r}; choose from {', '.join(PLANNERS)}")


__all__ = [
    "LAYOUTS", "LAYOUT_INFO", "PLANNERS", "PLANNER_INFO", "Conflict", "Grid", "PlanResult", "Problem",
    "cbs", "find_conflicts", "independent", "layout_grid", "makespan", "prioritized",
    "random_instance", "solve", "sum_of_costs",
]
