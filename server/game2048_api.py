"""2048 API: ask an agent for its move on a board.

POST /api/2048/move
  {"grid": [[0, 2, 4, 0], ...], "agent": "expectimax" | "ntuple" | "ntuple_search"}
  -> {"move": 2, "direction": "Left", "values": {...}, "depth": 3, ...}
"""

from __future__ import annotations

import asyncio
import time
from typing import Annotated, Literal

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from game2048.agents import BUDGET, DESCRIPTIONS, choose
from game2048.board import DIRECTIONS, from_grid, legal_moves

from .limits import Busy, client_key

router = APIRouter()
class MoveRequest(BaseModel):
    # The board is 4x4: exactly 4 rows, each with exactly 4 tiles.
    grid: list[Annotated[list[int], Field(min_length=4, max_length=4)]] = Field(min_length=4, max_length=4)
    agent: Literal["expectimax", "ntuple", "ntuple_search"] = "expectimax"


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
    key = client_key(request)
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
