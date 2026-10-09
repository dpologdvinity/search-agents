"""Warehouse robots API: built-in maps, random solvable instances, and multi-robot planning.

GET  /api/warehouse/meta    layouts, planner descriptions, and the limits the server enforces
POST /api/warehouse/random  {"rows": [...], "robots": 1-8, "seed": int} -> a solvable start/goal set
POST /api/warehouse/solve   {"rows", "starts", "goals", "planner", "max_nodes", "max_seconds"} ->
                            paths, stats, collisions, and (for cbs) the constraint-tree trace

A map is a list of strings: '.' is floor and '#' is a shelf. A point is [x, y], with x the column and
y the row. Every cell index inside the core package is converted to [x, y] before it leaves this module.

Searches are CPU-bound, so they share the SearchSlots semaphore, and each client is rate limited with
app.state.rate. The core planners bound time but not size, so map and robot counts are capped here.
"""

from __future__ import annotations

import asyncio
from typing import Annotated, Literal

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field, StringConstraints, model_validator

from warehouse import (
    LAYOUT_INFO,
    LAYOUTS,
    PLANNER_INFO,
    PLANNERS,
    Grid,
    PlanResult,
    Problem,
    layout_grid,
    random_instance,
    solve,
)

from .limits import Busy, client_key

router = APIRouter()

MAX_ROBOTS = 8
MAX_WIDTH = 24
MAX_HEIGHT = 16
MAX_NODES = 3000
MAX_SECONDS = 3.0
DEFAULT_NODES = 1500
DEFAULT_SECONDS = 2.0
# Random instances run CBS once per candidate, so the worst case is RANDOM_ATTEMPTS x RANDOM_SECONDS
# (about 4 s). A candidate that CBS cannot settle in that budget is skipped, and a map that yields no
# solvable draw returns 400 so the page can ask for another seed or fewer robots.
RANDOM_ATTEMPTS = 10
RANDOM_NODES = 800
RANDOM_SECONDS = 0.4
TRACE_LIMIT = 120  # expanded constraint-tree nodes sent to the page; the full tree can be much larger
CELL_KEYS = ("cell", "src", "dst")  # keys whose int value is a cell index in the core's to_dict() output

MapRow = Annotated[str, StringConstraints(pattern=rf"^[.#]{{3,{MAX_WIDTH}}}$")]
Point = Annotated[list[int], Field(min_length=2, max_length=2)]


class MapRequest(BaseModel):
    # Two rows is the smallest map that still has a corridor with a side pocket (the swap test instance).
    rows: list[MapRow] = Field(min_length=2, max_length=MAX_HEIGHT)


class RandomRequest(MapRequest):
    robots: int = Field(4, ge=1, le=MAX_ROBOTS)
    seed: int = 1


class SolveRequest(MapRequest):
    starts: list[Point] = Field(min_length=1, max_length=MAX_ROBOTS)
    goals: list[Point] = Field(min_length=1, max_length=MAX_ROBOTS)
    planner: Literal["cbs", "prioritized", "independent"] = "cbs"
    max_nodes: int = Field(DEFAULT_NODES, ge=1, le=MAX_NODES)
    max_seconds: float = Field(DEFAULT_SECONDS, gt=0, le=MAX_SECONDS)

    @model_validator(mode="after")
    def one_goal_per_start(self):
        if len(self.starts) != len(self.goals):
            raise ValueError("starts and goals must have one entry per robot")
        return self


def _rate_limit(request: Request) -> None:
    if not request.app.state.rate.allow(client_key(request)):
        raise HTTPException(429, "rate limit: try again in a few seconds")


def _grid(rows: list[str]) -> Grid:
    """Build the map, turning the core's shape errors (ragged rows, unknown characters) into 400s."""
    try:
        return Grid(rows)
    except ValueError as e:
        raise HTTPException(400, str(e)) from None


def _points(grid: Grid, points: list[list[int]], what: str) -> list[tuple[int, int]]:
    """Check each [x, y] is on the map before converting it: an out-of-range cell would index the wrong row."""
    out = []
    for x, y in points:
        if not (0 <= x < grid.width and 0 <= y < grid.height):
            raise HTTPException(400, f"{what} ({x}, {y}) is outside the map")
        out.append((x, y))
    return out


def _plain(node, grid: Grid):
    """Walk a core to_dict() payload and replace cell indices with [x, y] pairs.

    The core reports cells as integers, which the page cannot place without the map width. Only keys
    that name a cell are converted, so counters such as ``t``, ``cost``, or ``robot`` stay as they are.
    """
    if isinstance(node, dict):
        out = {}
        for key, value in node.items():
            if key in CELL_KEYS and isinstance(value, int):
                out[key] = list(grid.xy(value))
            elif key == "cells" and isinstance(value, list):
                out[key] = [list(grid.xy(c)) for c in value]
            else:
                out[key] = _plain(value, grid)
        return out
    if isinstance(node, list):
        return [_plain(v, grid) for v in node]
    return node


def _solve_json(result: PlanResult, grid: Grid, planner: str) -> dict:
    paths = None
    if result.paths is not None:
        paths = [[list(grid.xy(c)) for c in path] for path in result.paths]
    conflicts = [
        _plain({"kind": c.kind, "t": c.t, "robots": list(c.robots), "cells": list(c.cells)}, grid)
        for c in result.conflicts
    ]
    body = {
        "planner": result.planner,
        "status": result.status,
        "paths": paths,
        "stats": {
            "sum_of_costs": result.sum_of_costs,
            "makespan": result.makespan,
            "conflicts": len(result.conflicts),
            "high_nodes": result.high_nodes,
            "expanded_nodes": result.expanded_nodes,
            "conflicts_resolved": result.conflicts_resolved,
            "low_expansions": result.low_expansions,
            "seconds": round(result.seconds, 4),
        },
        "conflicts": conflicts,
    }
    if planner == "cbs":
        body["trace"] = [_plain(entry, grid) for entry in result.trace]
    return body


@router.get("/api/warehouse/meta")
async def meta():
    layouts = []
    for name, rows in LAYOUTS.items():
        grid = layout_grid(name)
        info = LAYOUT_INFO[name]
        layouts.append({"name": name, "title": info["title"], "blurb": info["blurb"],
                        "rows": list(rows), "width": grid.width, "height": grid.height})
    planners = [{"name": n, "title": PLANNER_INFO[n]["title"], "description": PLANNER_INFO[n]["description"]}
                for n in PLANNERS]
    return {
        "layouts": layouts,
        "planners": planners,
        "limits": {"max_robots": MAX_ROBOTS, "max_width": MAX_WIDTH, "max_height": MAX_HEIGHT,
                   "max_nodes": MAX_NODES, "max_seconds": MAX_SECONDS},
    }


@router.post("/api/warehouse/random")
async def random_route(req: RandomRequest, request: Request):
    _rate_limit(request)
    grid = _grid(req.rows)
    try:
        async with request.app.state.slots.acquire():
            problem = await asyncio.to_thread(
                random_instance, grid, req.robots, req.seed,
                attempts=RANDOM_ATTEMPTS, max_nodes=RANDOM_NODES, max_seconds=RANDOM_SECONDS,
            )
    except Busy:
        raise HTTPException(503, "server busy: try again shortly") from None
    except ValueError as e:
        raise HTTPException(400, str(e)) from None
    return {
        "rows": req.rows,
        "starts": [list(grid.xy(c)) for c in problem.starts],
        "goals": [list(grid.xy(c)) for c in problem.goals],
        "seed": req.seed,
    }


@router.post("/api/warehouse/solve")
async def solve_route(req: SolveRequest, request: Request):
    _rate_limit(request)
    grid = _grid(req.rows)
    starts = _points(grid, req.starts, "start")
    goals = _points(grid, req.goals, "goal")
    try:
        problem = Problem.from_xy(grid, starts, goals)
    except ValueError as e:  # e.g. a start or goal on a shelf, or an unreachable goal
        raise HTTPException(400, str(e)) from None
    try:
        async with request.app.state.slots.acquire():
            # The limits only reach CBS; the other planners take no budget and ignore these keyword arguments.
            result = await asyncio.to_thread(
                solve, problem, req.planner,
                max_nodes=req.max_nodes, max_seconds=req.max_seconds, trace_limit=TRACE_LIMIT,
            )
    except Busy:
        raise HTTPException(503, "server busy: try again shortly") from None
    return _solve_json(result, grid, req.planner)
