"""Rover API: seeded maps for the page, and a one-shot planning check against the Python reference.

GET  /api/rover/meta
GET  /api/rover/map?size=21&density=0.2&seed=1      the same map the CLI draws for these arguments
POST /api/rover/plan  {"n": 21, "walls": [441 ints, 1 = wall], "start": 0, "goal": 440}

The page runs the planners in JavaScript, so the server is not on the hot path. /plan lets the page (and
the tests) check the JavaScript port against the Python reference on any map: it runs D* Lite and A* once
on the given map and returns both costs, paths, and expansion counts. The walls are treated as known,
so this is a single planning problem, not an exploration.

Boards are flat lists in row-major order, the same as the rest of the project.
"""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel, Field

from rover.astar import AStar
from rover.dstar import INF, DStarLite
from rover.world import generate

from .limits import Busy

router = APIRouter()

MIN_SIZE, MAX_SIZE = 5, 81
DENSITIES = [0.1, 0.2, 0.3]
MAX_RADIUS = 4

DESCRIPTION = (
    "A rover on an unknown grid senses a square around itself and learns walls as it moves. D* Lite repairs "
    "its shortest paths when the sensor changes the map; A* searches from scratch each time."
)


class PlanRequest(BaseModel):
    n: int = Field(ge=MIN_SIZE, le=MAX_SIZE)
    walls: list[int] = Field(max_length=MAX_SIZE * MAX_SIZE)
    start: int = Field(0, ge=0)
    goal: int = Field(-1)


def _client_key(request: Request) -> str:
    # Behind Fly.io's proxy the client address arrives in a header.
    return request.headers.get("fly-client-ip") or (request.client.host if request.client else "unknown")


def _check(req: PlanRequest) -> int:
    """Validate a plan request (HTTP 400 on bad input). Returns the goal cell, defaulting to the bottom-right."""
    n = req.n
    if len(req.walls) != n * n:
        raise HTTPException(400, f"a {n}x{n} map needs {n * n} cells, got {len(req.walls)}")
    if any(v not in (0, 1) for v in req.walls):
        raise HTTPException(400, "cells must be 0 (free) or 1 (wall)")
    goal = n * n - 1 if req.goal < 0 else req.goal
    for name, cell in (("start", req.start), ("goal", goal)):
        if not 0 <= cell < n * n:
            raise HTTPException(400, f"{name} must be a cell between 0 and {n * n - 1}")
        if req.walls[cell]:
            raise HTTPException(400, f"{name} cell is a wall")
    return goal


def plan_json(n: int, walls: list[int], start: int, goal: int) -> dict:
    """Run both planners once on a known map and describe the results (CPU work, runs in a worker thread)."""
    # D* Lite starts from an all-free map and receives every wall as a change, the same way the rover
    # would learn them, so the numbers include its repair work.
    changes = [(c, 1) for c, w in enumerate(walls) if w]
    d = DStarLite(n, start, goal)
    d.on_change(changes, start)
    a = AStar(n, start, goal)  # same protocol for A*: plan on the free map, then replan once the walls are known
    a.on_change(changes, start)
    d_cost = d.cost_to_goal(start)
    a_cost = len(a.path) - 1 if a.path else INF
    d_path = d.path_from(start)
    return {
        "n": n,
        "dstar": {"cost": None if d_cost == INF else int(d_cost), "expansions": d.expansions, "path": d_path},
        "astar": {"cost": None if a_cost == INF else int(a_cost), "expansions": a.expansions, "path": a.path},
        "agree": d_cost == a_cost,
    }


@router.get("/api/rover/meta")
async def meta():
    return {
        "sizes": [21, 41, 61],
        "densities": DENSITIES,
        "min_size": MIN_SIZE,
        "max_size": MAX_SIZE,
        "max_radius": MAX_RADIUS,
        "description": DESCRIPTION,
    }


@router.get("/api/rover/map")
async def seeded_map(size: int = Query(21, ge=MIN_SIZE, le=MAX_SIZE),
                     density: float = Query(0.2, ge=0.0, le=0.4),
                     seed: int = Query(1, ge=0, le=2**32 - 1)):
    walls = generate(size, density, seed)
    return {"n": size, "start": 0, "goal": size * size - 1, "walls": list(walls)}


@router.post("/api/rover/plan")
async def plan(req: PlanRequest, request: Request):
    app = request.app
    if not app.state.move_rate.allow(_client_key(request)):
        raise HTTPException(429, "rate limit: try again in a few seconds")
    goal = _check(req)
    try:
        async with app.state.slots.acquire():
            return await asyncio.to_thread(plan_json, req.n, req.walls, req.start, goal)
    except Busy:
        raise HTTPException(503, "server busy: try again shortly") from None
