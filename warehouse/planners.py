"""The baselines and the shared result type: prioritized planning and independent A*.

- Independent A*: every robot takes its own shortest path and ignores the others. It is the
  fastest possible answer, and usually wrong: the result is a set of paths that collide.
  Counting its collisions shows why coordination is needed.
- Prioritized planning: robots are planned one at a time in a fixed order. Each later robot
  treats the earlier robots' paths (including their parked goal cells) as moving obstacles.
  Every plan is collision-free by construction, but a robot can be boxed in by the ones before it
  (then the status is ``failed``), and the early robots' greedy routes make the total cost higher
  than the optimum.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass, field

from .conflicts import Conflict, Path, find_conflicts, makespan, sum_of_costs
from .grid import Problem
from .space_time import Reservations, space_time_astar


@dataclass
class PlanResult:
    """Outcome of one planner run. ``paths`` is None when no plan was produced.

    ``status``: ``solved`` (collision-free, and optimal for CBS), ``collisions`` (independent
    paths that clash), ``failed`` (prioritized planning could not route a robot), ``node_limit``,
    ``time_limit``, or ``no_solution`` (CBS limits and exhaustion).
    """

    planner: str
    status: str
    paths: tuple[Path, ...] | None
    sum_of_costs: int | None
    makespan: int | None
    conflicts: list[Conflict] = field(default_factory=list)
    high_nodes: int = 0  # CBS: constraint-tree nodes created
    expanded_nodes: int = 0  # CBS: constraint-tree nodes expanded (each resolves one collision)
    low_expansions: int = 0  # space-time A* states expanded, over all robots and calls
    conflicts_resolved: int = 0  # CBS: collisions that were branched on
    seconds: float = 0.0
    trace: list[dict] = field(default_factory=list)  # CBS: bounded record of the tree
    solution: int | None = None  # CBS: id of the collision-free tree node returned (None if no plan)


def independent(problem: Problem) -> PlanResult:
    """Each robot takes its own shortest space-time path, ignoring every other robot."""
    started = time.perf_counter()
    paths: list[Path] = []
    low = 0
    for r in range(problem.robots):
        path, expanded = space_time_astar(problem.grid, problem.starts[r], problem.goals[r], Reservations())
        low += expanded
        paths.append(path)  # never None: Problem validation guarantees each goal is reachable
    conflicts = find_conflicts(paths)
    return PlanResult(
        planner="independent",
        status="collisions" if conflicts else "solved",
        paths=tuple(paths),
        sum_of_costs=sum_of_costs(paths),
        makespan=makespan(paths),
        conflicts=conflicts,
        low_expansions=low,
        seconds=time.perf_counter() - started,
    )


def prioritized(problem: Problem, order: Sequence[int] | None = None) -> PlanResult:
    """Plan robots one at a time in ``order`` (default: robot index), each avoiding the earlier ones.

    The returned plan, when it exists, is collision-free by construction. The reservation
    table grows with every planned robot, so each new search sees all earlier paths.
    """
    started = time.perf_counter()
    order = list(range(problem.robots)) if order is None else list(order)
    res = Reservations()
    paths: dict[int, Path] = {}
    low = 0
    for r in order:
        path, expanded = space_time_astar(problem.grid, problem.starts[r], problem.goals[r], res)
        low += expanded
        if path is None:
            return PlanResult(
                planner="prioritized",
                status="failed",
                paths=None,
                sum_of_costs=None,
                makespan=None,
                low_expansions=low,
                seconds=time.perf_counter() - started,
            )
        paths[r] = path
        res.add_path(path)
    ordered = tuple(paths[r] for r in range(problem.robots))
    return PlanResult(
        planner="prioritized",
        status="solved",
        paths=ordered,
        sum_of_costs=sum_of_costs(ordered),
        makespan=makespan(ordered),
        low_expansions=low,
        seconds=time.perf_counter() - started,
    )
