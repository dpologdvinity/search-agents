"""2048 API: ask an agent for its move on a board.

POST /api/2048/move
  {"grid": [[0, 2, 4, 0], ...], "agent": "expectimax" | "ntuple" | "ntuple_search"}
  -> {"move": 2, "direction": "Left", "values": {...}, "depth": 3, ...}
"""

from __future__ import annotations

import asyncio
import time
from typing import Literal

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from game2048.board import DIRECTIONS, from_grid, legal_moves
from game2048.expectimax import Expectimax

from .limits import Busy

router = APIRouter()
BUDGET = 0.1  # seconds of search per move

DESCRIPTIONS = {
    "expectimax": (
        "Iterative-deepening expectimax scoring boards with six hand-crafted features, weighted by "
        "a cross-entropy-method search over simulated games."
    ),
    "ntuple": "No search: picks the move whose afterstate a TD-learned n-tuple network values most.",
    "ntuple_search": (
        "Expectimax search that scores positions with the learned n-tuple network instead of the "
        "hand-tuned evaluation."
    ),
}


class MoveRequest(BaseModel):
    grid: list[list[int]] = Field(min_length=4, max_length=4)
    agent: Literal["expectimax", "ntuple", "ntuple_search"] = "expectimax"


def choose(board: int, agent: str) -> dict:
    if agent in ("ntuple", "ntuple_search"):
        from game2048.ntuple import load

        net = load()
    if agent == "ntuple":
        values = {d: gained + net.value(after) for d, after, gained in legal_moves(board)}
        best = max(values, key=values.get)
        return {"move": best, "values": values, "depth": 1, "nodes": len(values)}
    if agent == "ntuple_search":
        searcher = Expectimax(budget=BUDGET, evaluator=net.value, rewards=True)
    else:
        searcher = Expectimax(budget=BUDGET)
    info = searcher.search(board)
    return {"move": info.move, "values": info.values, "depth": info.depth, "nodes": info.nodes}


@router.get("/api/2048/meta")
async def meta():
    from game2048.ntuple import WEIGHTS

    return {"agents": [{"name": k, "description": v,
                        "available": not k.startswith("ntuple") or WEIGHTS.exists()}
                       for k, v in DESCRIPTIONS.items()],
            "budget_seconds": BUDGET}


@router.post("/api/2048/move")
async def move(req: MoveRequest, request: Request):
    app = request.app
    key = request.headers.get("fly-client-ip") or (request.client.host if request.client else "unknown")
    if not app.state.move_rate.allow(key):
        raise HTTPException(429, "rate limit: try again in a few seconds")
    if any(len(row) != 4 for row in req.grid):
        raise HTTPException(400, "grid must be 4x4")
    try:
        board = from_grid(req.grid)
    except ValueError as e:
        raise HTTPException(400, str(e)) from None
    if not legal_moves(board):
        raise HTTPException(400, "no legal moves: the game is over")
    start = time.perf_counter()
    try:
        async with app.state.slots.acquire():
            result = await asyncio.to_thread(choose, board, req.agent)
    except FileNotFoundError:
        raise HTTPException(503, "the n-tuple network is not available on this server") from None
    except Busy:
        raise HTTPException(503, "server busy: try again shortly") from None
    return {
        "agent": req.agent,
        "move": result["move"],
        "direction": DIRECTIONS[result["move"]],
        "values": {DIRECTIONS[d]: round(v, 3) for d, v in result["values"].items()},
        "depth": result["depth"],
        "nodes": result["nodes"],
        "seconds": round(time.perf_counter() - start, 3),
    }
