"""CartPole API: trained weights for the in-browser forward pass, learning curves, benchmark, and rollouts.

GET  /api/cartpole/meta                 physics constants, agent names, the observation scale
GET  /api/cartpole/policy/{name}       weights of reinforce, actor_critic or cem (plain nested lists)
GET  /api/cartpole/curves              per-seed training returns for each algorithm
GET  /api/cartpole/benchmark           the committed benchmark (every agent, every episode's step count)
POST /api/cartpole/rollout             {"agent": "...", "seed": 1, "steps": 500}: one server-side episode

The page runs the physics and the policy in JavaScript, so almost everything above is static
data. The rollout endpoint lets the page check that its JavaScript port matches the Python
physics: from the server's start state (states[0]), each step's action and next state must match.
The browser's own start state for a seed differs from Python's (different generators), so the check
never starts from it. Rollouts cost a few milliseconds of CPU, so they go through the same per-client
rate limit as the other move endpoints.
"""

from __future__ import annotations

import asyncio
import functools
import json

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from cartpole.agents import AGENT_NAMES, DATA_DIR, LABELS, PD_GAINS, load_agent, load_weights
from cartpole.env import ACTIONS, MAX_STEPS, THETA_LIMIT, X_LIMIT, run_episode
from cartpole.nets import OBS_SCALE
from cartpole.train import ALGOS, GAMMA, HIDDEN, VALUE_SCALE

from .limits import Busy, client_key

router = APIRouter()

POLICY_NAMES = ("reinforce", "actor_critic", "cem")


class RolloutRequest(BaseModel):
    agent: str = "reinforce"
    seed: int = Field(0, ge=0, le=2**31 - 1)
    steps: int = Field(MAX_STEPS, ge=1, le=MAX_STEPS)


@functools.cache
def _weights(name: str) -> dict:
    """Weights as plain nested lists. The .npz holds float32, and tolist() writes each one exactly
    (as a float64 with the same value), so the browser's forward pass uses the same numbers as Python."""
    return {k: arr.tolist() for k, arr in load_weights(name, DATA_DIR).items()}


@functools.lru_cache(maxsize=1)
def _curves() -> dict:
    return json.loads((DATA_DIR / "curves.json").read_text())


@functools.lru_cache(maxsize=1)
def _benchmark() -> dict:
    return json.loads((DATA_DIR / "benchmark.json").read_text())


def rollout_json(agent_name: str, seed: int, steps: int) -> dict:
    """One episode as JSON: the states (steps + 1 rows), the actions taken, and whether the pole fell."""
    agent = load_agent(agent_name, seed=seed)
    n, traj = run_episode(agent, seed, max_steps=steps, record=True)
    states = [traj[0]["obs"]] + [t["next"] for t in traj] if traj else []
    return {
        "agent": agent_name,
        "seed": seed,
        "steps": n,
        "fell": n < steps,
        "states": [list(s) for s in states],
        "actions": [t["action"] for t in traj],
    }


@router.get("/api/cartpole/meta")
async def meta():
    return {
        "agents": [{"name": n, "label": LABELS[n]} for n in AGENT_NAMES],
        "physics": {
            "gravity": 9.8, "mass_cart": 1.0, "mass_pole": 0.1, "half_length": 0.5, "force": 10.0,
            "tau": 0.02, "theta_limit": THETA_LIMIT, "x_limit": X_LIMIT, "max_steps": MAX_STEPS,
        },
        "actions": list(ACTIONS),
        "obs_scale": [float(v) for v in OBS_SCALE],
        "hidden": HIDDEN,
        "value_scale": VALUE_SCALE,
        "gamma": GAMMA,
        "pd_gains": list(PD_GAINS),
        "algorithms": list(ALGOS),
    }


@router.get("/api/cartpole/policy/{name}")
async def policy(name: str):
    if name not in POLICY_NAMES:
        raise HTTPException(404, f"no trained weights for {name!r}; choose from {', '.join(POLICY_NAMES)}")
    try:
        return {"name": name, "hidden": HIDDEN, "weights": _weights(name)}
    except FileNotFoundError:
        raise HTTPException(404, f"weights for {name!r} are not trained yet") from None


@router.get("/api/cartpole/curves")
async def curves():
    try:
        return _curves()
    except FileNotFoundError:
        raise HTTPException(404, "no learning curves yet") from None


@router.get("/api/cartpole/benchmark")
async def benchmark():
    try:
        return _benchmark()
    except FileNotFoundError:
        raise HTTPException(404, "no benchmark yet") from None


@router.post("/api/cartpole/rollout")
async def rollout(req: RolloutRequest, request: Request):
    app = request.app
    if req.agent not in AGENT_NAMES:
        raise HTTPException(400, f"unknown agent {req.agent!r}")
    if not app.state.move_rate.allow(client_key(request)):
        raise HTTPException(429, "rate limit: try again in a few seconds")
    try:
        async with app.state.slots.acquire():
            return await asyncio.to_thread(rollout_json, req.agent, req.seed, req.steps)
    except Busy:
        raise HTTPException(503, "server busy: try again shortly") from None
    except FileNotFoundError as e:
        raise HTTPException(404, str(e)) from None
