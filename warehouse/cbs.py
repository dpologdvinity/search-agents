"""Conflict-Based Search (Sharon et al., 2012): optimal multi-robot planning for sum of costs.

CBS works on two levels.

- Low level: each robot is planned alone by space-time A*, under the constraints that apply to it.
- High level: a constraint tree. The root plans every robot with no constraints, so each robot gets
  its own cheapest path. The plan may collide. Take the earliest collision between robots i and j
  and split: one child forbids i from that cell (or that move) at that time, the other forbids j.
  Each child replans only the robot it constrains. Nodes are expanded cheapest first, by the sum
  of their path costs.

Why the result is optimal: the root cost is a lower bound on every child, since constraints can
only raise a robot's cost. Each collision must be resolved by at least one of its two children, so
the first collision-free node expanded is the cheapest possible plan.

The tree is capped by ``max_nodes`` and ``max_seconds``. Running out is reported as a status,
not an exception, so a caller can show partial progress.
"""

from __future__ import annotations

import heapq
import time
from dataclasses import dataclass, field

from .conflicts import Conflict, Path, find_conflicts, makespan, sum_of_costs
from .grid import Problem
from .planners import PlanResult
from .space_time import EdgeConstraint, Reservations, VertexConstraint, space_time_astar

Constraint = VertexConstraint | EdgeConstraint


@dataclass
class _Node:
    """One high-level search node: a set of constraints and the plan they produce."""

    id: int
    parent: int | None
    constraints: tuple[Constraint, ...]
    paths: tuple[Path, ...]
    cost: int
    added: Constraint | None = None  # the constraint this node adds to its parent's set


@dataclass
class CBSStats:
    """Counters that the caller can show: how big the tree got and how hard the low level worked."""

    generated: int = 0
    expanded: int = 0
    low_expansions: int = 0
    infeasible: int = 0
    trace: list[dict] = field(default_factory=list)


def branch_constraints(conflict: Conflict) -> list[tuple[int, Constraint]]:
    """The two children of a collision, as ``(robot, constraint)`` pairs.

    Vertex: ``robots[0]`` may not be on the cell at time t, and neither may ``robots[1]``.
    Edge: ``robots[0]`` may not move a -> b at time t, and ``robots[1]`` may not move b -> a.
    Each child forbids one of the two robots from the collision, so both children together cover
    every way of resolving it.
    """
    i, j = conflict.robots
    if conflict.kind == "vertex":
        (cell,) = conflict.cells
        return [(i, VertexConstraint(i, cell, conflict.t)), (j, VertexConstraint(j, cell, conflict.t))]
    a, b = conflict.cells
    return [(i, EdgeConstraint(i, a, b, conflict.t)), (j, EdgeConstraint(j, b, a, conflict.t))]


def plan_one(problem: Problem, robot: int, constraints: tuple[Constraint, ...], stats: CBSStats) -> Path | None:
    """Plan one robot against only the constraints addressed to it."""
    res = Reservations(c for c in constraints if c.robot == robot)
    path, expanded = space_time_astar(problem.grid, problem.starts[robot], problem.goals[robot], res)
    stats.low_expansions += expanded
    return path


def cbs(problem: Problem, *, max_nodes: int = 2000, max_seconds: float = 5.0, trace_limit: int = 200) -> PlanResult:
    """Find a collision-free plan with minimum sum of costs, or report why the search stopped.

    Statuses: ``solved`` (optimal plan in ``paths``), ``node_limit`` or ``time_limit`` (search cut off;
    no plan), ``no_solution`` (some robot has no path even alone, or the open list emptied). Infeasible
    instances usually end on a budget instead: a constraint only bounds a robot's timing, so the tree can
    keep branching without ever running out. Budgets are what make the search always terminate.
    ``trace`` records up to ``trace_limit`` expanded nodes with the collision they resolved and their
    two children, which is what the web page draws as the growing constraint tree.
    """
    started = time.perf_counter()
    stats = CBSStats()

    # Root: every robot alone. Its cost is a lower bound on all later nodes.
    root_paths = []
    for r in range(problem.robots):
        path, expanded = space_time_astar(problem.grid, problem.starts[r], problem.goals[r], Reservations())
        stats.low_expansions += expanded
        if path is None:
            return _result("no_solution", None, stats, started)
        root_paths.append(path)
    root = _Node(0, None, (), tuple(root_paths), sum_of_costs(root_paths))
    stats.generated = 1

    # Open list ordered by cost, then creation order, so ties are deterministic.
    open_heap: list[tuple[int, int, _Node]] = [(root.cost, root.id, root)]
    while open_heap:
        if stats.expanded >= max_nodes:
            return _result("node_limit", None, stats, started)
        if time.perf_counter() - started > max_seconds:
            return _result("time_limit", None, stats, started)

        _, _, node = heapq.heappop(open_heap)
        conflicts = find_conflicts(node.paths, first_only=True)
        if not conflicts:
            return _result("solved", node, stats, started)

        stats.expanded += 1
        conflict = conflicts[0]
        children = []
        for robot, constraint in branch_constraints(conflict):
            cons = node.constraints + (constraint,)
            path = plan_one(problem, robot, cons, stats)
            child_info = {"robot": robot, "constraint": constraint.to_dict()}
            if path is None:
                stats.infeasible += 1
                child_info["cost"] = None  # this child has no plan within the horizon
            else:
                paths = list(node.paths)
                paths[robot] = path
                child = _Node(stats.generated, node.id, cons, tuple(paths), sum_of_costs(paths), constraint)
                stats.generated += 1
                heapq.heappush(open_heap, (child.cost, child.id, child))
                child_info["id"] = child.id
                child_info["cost"] = child.cost
            children.append(child_info)
        if len(stats.trace) < trace_limit:
            stats.trace.append({
                "id": node.id,
                "parent": node.parent,
                "cost": node.cost,
                "added": node.added.to_dict() if node.added else None,
                "conflict": _conflict_dict(conflict),
                "children": children,
            })
    return _result("no_solution", None, stats, started)


def _conflict_dict(c: Conflict) -> dict:
    return {"kind": c.kind, "t": c.t, "robots": list(c.robots), "cells": list(c.cells)}


def _result(status: str, node: _Node | None, stats: CBSStats, started: float) -> PlanResult:
    """Package the outcome. ``node`` is the collision-free node when the search solved, else None."""
    paths = node.paths if node is not None else None
    return PlanResult(
        planner="cbs",
        status=status,
        paths=paths,
        sum_of_costs=node.cost if node is not None else None,
        makespan=makespan(paths) if paths is not None else None,
        conflicts=[],
        high_nodes=stats.generated,
        expanded_nodes=stats.expanded,
        low_expansions=stats.low_expansions,
        conflicts_resolved=stats.expanded if node is not None else 0,
        seconds=time.perf_counter() - started,
        trace=stats.trace,
        solution=node.id if node is not None else None,
    )
