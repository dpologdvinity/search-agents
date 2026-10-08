"""Route planner API: solve a traveling salesman instance and return its progress.

POST /api/routes/solve
  {"cities": [[x, y], ...], "algorithm": "simulated_annealing", "params": {...}}
  -> {"tour": [...], "length": ..., "frames": [...], "optimal": ... or null, ...}

Coordinates are expected in [0, 1]. Parameters are optional and clamped to
limits that keep one request around a second of server time.
"""

from __future__ import annotations

import asyncio
import time
from typing import Literal

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from routes.tsp import HELD_KARP_LIMIT, SOLVERS, held_karp

from .limits import Busy

router = APIRouter()

MAX_CITIES = 60

DESCRIPTIONS = {
    "nearest_neighbor_2opt": (
        "Greedy: always visit the closest unvisited city, then keep reversing segments of the route "
        "while that shortens it. Fast, but it stops at the first route it cannot improve."
    ),
    "simulated_annealing": (
        "Reverses random segments of a random route. Shorter routes are always kept; longer ones are "
        "sometimes kept too, less often as the temperature cools, which lets it escape dead ends."
    ),
    "genetic_algorithm": (
        "Breeds a population of routes: picks short parents, combines them so the child keeps "
        "sub-routes from both, mutates some children, and always keeps the best few."
    ),
}


class Params(BaseModel):
    """Tuning knobs, each bounded so a request stays cheap."""

    iterations: int = Field(60_000, ge=1_000, le=150_000)       # simulated annealing
    cooling: float | None = Field(None, gt=0.9, lt=1.0)          # simulated annealing
    population_size: int = Field(120, ge=10, le=200)             # genetic algorithm
    generations: int = Field(300, ge=10, le=500)                 # genetic algorithm
    mutation_rate: float = Field(0.3, ge=0.0, le=1.0)            # genetic algorithm
    seed: int | None = None


class SolveRequest(BaseModel):
    cities: list[tuple[float, float]] = Field(min_length=3, max_length=MAX_CITIES)
    algorithm: Literal["nearest_neighbor_2opt", "simulated_annealing", "genetic_algorithm"] = "simulated_annealing"
    params: Params = Params()


def run(req: SolveRequest) -> dict:
    p = req.params
    if req.algorithm == "simulated_annealing":
        kwargs = {"iterations": p.iterations, "cooling": p.cooling, "seed": p.seed}
    elif req.algorithm == "genetic_algorithm":
        kwargs = {"population_size": p.population_size, "generations": p.generations,
                  "mutation_rate": p.mutation_rate, "seed": p.seed}
    else:
        kwargs = {}
    start = time.perf_counter()
    result = SOLVERS[req.algorithm](req.cities, **kwargs)
    seconds = time.perf_counter() - start
    # For small maps, also compute the true optimum so the page can show the gap.
    optimal = held_karp(req.cities) if len(req.cities) <= HELD_KARP_LIMIT else None
    return {
        "algorithm": result.algorithm,
        "tour": result.tour,
        "length": result.length,
        "steps": result.steps,
        "frames": result.frames,
        "seconds": round(seconds, 3),
        "optimal": {"length": optimal.length, "tour": optimal.tour} if optimal else None,
    }


@router.get("/api/routes/meta")
async def meta():
    return {"algorithms": [{"name": k, "description": v} for k, v in DESCRIPTIONS.items()],
            "max_cities": MAX_CITIES, "exact_limit": HELD_KARP_LIMIT}


@router.post("/api/routes/solve")
async def solve(req: SolveRequest, request: Request):
    app = request.app
    key = request.headers.get("fly-client-ip") or (request.client.host if request.client else "unknown")
    if not app.state.rate.allow(key):
        raise HTTPException(429, "rate limit: try again in a few seconds")
    if any(not (0 <= x <= 1 and 0 <= y <= 1) for x, y in req.cities):
        raise HTTPException(400, "city coordinates must be between 0 and 1")
    try:
        async with app.state.slots.acquire():
            return await asyncio.to_thread(run, req)
    except Busy:
        raise HTTPException(503, "server busy: try again shortly") from None
