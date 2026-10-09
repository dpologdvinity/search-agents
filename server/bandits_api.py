"""Neon casino API: seeded machine sets, the committed benchmark, and a server-side score for a player's pulls.

GET  /api/bandits/meta                                   kinds, agents, lineups and the shared constants
GET  /api/bandits/machines?kind=&k=&pulls=&seed=         the hidden means, segment by segment
GET  /api/bandits/benchmark                              bandits/data/bandits_benchmark.json (404 if missing)
POST /api/bandits/score  {"kind", "k", "seed", "arms"}   regret of a player's pulls next to every agent's

The page simulates the agents in JavaScript for the live race. The score endpoint runs the Python
agents on the same casino (same seed, same outcomes), so the end-of-game table is computed by the
reference implementation. Nothing is stored: a score is returned and forgotten.
"""

from __future__ import annotations

import asyncio
from functools import lru_cache
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import Response
from pydantic import BaseModel, Field

from bandits.agents import AGENTS, LINEUP, SlidingWindowUCB
from bandits.benchmark import LAST, TABLE_AT, K
from bandits.env import (
    DRIFT_PERIOD,
    GAP,
    HIGH,
    KINDS,
    LOW,
    REWARD_OFFSET,
    SIGMA,
    Environment,
    make_machines,
)
from bandits.sim import AGENT_OFFSET, play, score_pulls

from .limits import Busy, client_key

router = APIRouter()

MAX_PULLS = 1000  # a game on the page is at most this long; the benchmark runs elsewhere
MAX_MACHINES = 20
MIN_MACHINES = 2
SEED_MAX = 2**31 - 1
# Shipped inside the package so the Docker image has it. It is a copy of results/bandits_benchmark.json: after
# `python -m bandits benchmark`, copy the file here again.
BENCHMARK = Path(__file__).resolve().parent.parent / "bandits" / "data" / "bandits_benchmark.json"


class ScoreRequest(BaseModel):
    kind: Literal["bernoulli", "gaussian", "drifting"] = "bernoulli"
    k: int = Field(K, ge=MIN_MACHINES, le=MAX_MACHINES)
    seed: int = Field(0, ge=0, le=SEED_MAX)
    arms: list[int] = Field(min_length=1, max_length=MAX_PULLS)


def _meta() -> dict:
    return {
        "kinds": list(KINDS),
        "agents": [
            {"key": key, "label": cls.label, "blurb": cls.blurb,
             "lineup": [kind for kind, keys in LINEUP.items() if key in keys]}
            for key, cls in AGENTS.items()
        ],
        "lineup": {kind: list(keys) for kind, keys in LINEUP.items()},
        "defaults": {"kind": "bernoulli", "k": K, "pulls": 300},
        "limits": {"pulls": [1, MAX_PULLS], "machines": [MIN_MACHINES, MAX_MACHINES], "seed_max": SEED_MAX},
        # The JavaScript port reads these so it draws the same machines, outcomes and agent streams.
        "constants": {
            "sigma": SIGMA, "drift_period": DRIFT_PERIOD, "gap": GAP, "low": LOW, "high": HIGH,
            "reward_offset": REWARD_OFFSET, "agent_offset": AGENT_OFFSET,
            "window": SlidingWindowUCB.window, "window_xi": SlidingWindowUCB.xi,
            "last_pulls": LAST, "table_at": list(TABLE_AT),
        },
        "benchmark": BENCHMARK.is_file(),
    }


@lru_cache(maxsize=1)
def _benchmark_bytes() -> bytes | None:
    return BENCHMARK.read_bytes() if BENCHMARK.is_file() else None


@router.get("/api/bandits/meta")
async def meta():
    return _meta()


@router.get("/api/bandits/machines")
async def machines(
    kind: Literal["bernoulli", "gaussian", "drifting"] = "bernoulli",
    k: int = Query(K, ge=MIN_MACHINES, le=MAX_MACHINES),
    pulls: int = Query(300, ge=1, le=MAX_PULLS),
    seed: int = Query(0, ge=0, le=SEED_MAX),
):
    m = make_machines(kind, k, pulls, seed)
    return {
        "kind": kind, "k": k, "seed": seed, "pulls": pulls, "period": m.period,
        "schedule": [list(seg) for seg in m.schedule],
        "best": [seg.index(max(seg)) for seg in m.schedule],
    }


@router.get("/api/bandits/benchmark")
async def benchmark():
    body = _benchmark_bytes()
    if body is None:
        raise HTTPException(404, "the benchmark has not been run: python -m bandits benchmark")
    return Response(content=body, media_type="application/json")  # already JSON text: no parse per request


def score_json(req: ScoreRequest) -> dict:
    """Score the player's pulls and run every agent in the kind's lineup on the same casino."""
    if any(not 0 <= a < req.k for a in req.arms):
        raise ValueError(f"arms must name machines 0..{req.k - 1}")
    env = Environment(req.kind, req.k, len(req.arms), req.seed)
    mine = score_pulls(env, req.arms)

    def summary(run) -> dict:
        return {
            "regret": round(run.total_regret, 4),
            "share_best": round(run.share_optimal(), 4),
            "reward": round(sum(run.rewards), 4),
        }

    agents = {}
    for key in LINEUP[req.kind]:
        agents[key] = {"label": AGENTS[key].label, **summary(play(key, env))}
    return {
        "kind": req.kind, "k": req.k, "seed": req.seed, "pulls": len(req.arms),
        "player": summary(mine),
        "agents": agents,
    }


@router.post("/api/bandits/score")
async def score(req: ScoreRequest, request: Request):
    app = request.app
    if not app.state.rate.allow(client_key(request)):
        raise HTTPException(429, "rate limit: try again in a few seconds")
    try:
        async with app.state.slots.acquire():
            return await asyncio.to_thread(score_json, req)
    except Busy:
        raise HTTPException(503, "server busy: try again shortly") from None
    except ValueError as e:
        raise HTTPException(400, str(e)) from None
