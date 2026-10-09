"""N-Queens API: run one agent on one board and return the run, with board frames for small boards.

GET  /api/queens/meta
POST /api/queens/solve  {"agent": "minconf", "n": 8, "seed": 1, "frames": false}

Each request runs one agent under a time cap, and the caps by size keep the demo light on a shared
1 GB machine: backtracking is limited to 64 squares a side (it cannot finish beyond that in the
time cap anyway), hill climbing to 100 (one step scores n^2 squares), annealing to 2,000, and
min-conflicts to 10,000. Frames (the board after each step, for the watch view) are returned only
up to FRAMES_MAX_N. The big-N live view runs min-conflicts in the browser, so the server never
has to stream a million-queen run.

The response keeps the agent's own step count and its candidate-square count, and the conflict
trace (or, for backtracking, the rows still open) is thinned to a few hundred points.
"""

from __future__ import annotations

import asyncio
from typing import Literal

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from queens.agents import run

from .limits import Busy, client_key

router = APIRouter()

TIME_LIMIT = 2.0  # seconds of CPU per request, whatever the agent
FRAMES_MAX_N = 32  # board frames are returned only for boards up to this size
MAX_N = 10_000

# Largest board each agent is served at. Past these sizes a request would only run into the time cap
# (backtracking, hill climbing) or spend most of a shared server's CPU budget (annealing, min-conflicts).
LIMITS = {
    "backtrack": 64,
    "hill": 100,
    "anneal": 2_000,
    "minconf": MAX_N,
}

LABELS = {
    "backtrack": "Backtracking with bitmasks",
    "hill": "Steepest-ascent hill climbing",
    "anneal": "Simulated annealing",
    "minconf": "Min-conflicts",
}

DESCRIPTION = (
    "One queen per row. A conflict is a pair of queens on the same column or diagonal. Each agent is run "
    "with a time cap, and the page shows its steps, its candidate squares scored, and how it ended."
)


class SolveRequest(BaseModel):
    agent: Literal["backtrack", "hill", "anneal", "minconf"]
    n: int = Field(8, ge=1, le=MAX_N)
    seed: int = Field(1, ge=0, le=2**31 - 1)
    frames: bool = False


def solve_json(req: SolveRequest) -> dict:
    """Run the agent and describe the result in the JSON shape the page reads."""
    want_frames = req.frames and req.n <= FRAMES_MAX_N
    res = run(req.agent, req.n, req.seed, time_limit=TIME_LIMIT, record_frames=want_frames)
    return {
        "agent": req.agent,
        "label": LABELS[req.agent],
        "n": req.n,
        "seed": req.seed,
        "solved": res.solved,
        "reason": res.reason,
        "steps": res.steps,
        "evaluations": res.evaluations,
        "restarts": res.restarts,
        "seconds": round(res.seconds, 4),
        "conflicts": res.conflicts,
        "cols": res.cols,
        # Backtracking never has a conflict, so its trace counts the rows still open instead.
        "trace_kind": "open_rows" if req.agent == "backtrack" else "conflicts",
        "trace": res.trace,
        "trace_steps": res.trace_steps,
        "frames": res.frames if want_frames else None,
        "frame_steps": res.frame_steps if want_frames else None,
    }


@router.get("/api/queens/meta")
async def meta():
    return {
        "agents": {a: {"label": LABELS[a], "max_n": LIMITS[a]} for a in LABELS},
        "max_n": MAX_N,
        "frames_max_n": FRAMES_MAX_N,
        "time_limit": TIME_LIMIT,
        "description": DESCRIPTION,
    }


@router.post("/api/queens/solve")
async def solve(req: SolveRequest, request: Request):
    app = request.app
    if not app.state.rate.allow(client_key(request)):
        raise HTTPException(429, "rate limit: try again in a few seconds")
    if req.n > LIMITS[req.agent]:
        raise HTTPException(400, f"{LABELS[req.agent]} is served up to {LIMITS[req.agent]:,} queens")
    try:
        async with app.state.slots.acquire():
            return await asyncio.to_thread(solve_json, req)
    except Busy:
        raise HTTPException(503, "server busy: try again shortly") from None
