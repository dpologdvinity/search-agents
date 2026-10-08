"""Connect Four API: ask an agent for its move and see how it decided.

POST /api/connect4/move
  {"moves": "4453", "agent": "alphazero" | "minimax" | "mcts", "level": 1-3}
  -> {"move": 3, "analysis": {...}, "seconds": ...}

`moves` lists the columns played so far, 1-based, as in Connect Four solver
test sets. `level` scales the search budget.
"""

from __future__ import annotations

import asyncio
import time
from typing import Literal

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field, field_validator

from connect4.agents import AGENTS, ALPHAZERO_SIMS, DESCRIPTIONS, MCTS_SIMS, MINIMAX_SECONDS, board_from

from .limits import Busy

router = APIRouter()


class MoveRequest(BaseModel):
    moves: str = Field("", max_length=42)
    agent: Literal["alphazero", "minimax", "mcts"] = "alphazero"
    level: int = Field(2, ge=1, le=3)

    @field_validator("moves")
    @classmethod
    def digits(cls, v):
        if any(c not in "1234567" for c in v):
            raise ValueError("moves must be digits 1-7")
        return v



@router.get("/api/connect4/meta")
async def meta():
    from connect4.net import WEIGHTS

    return {
        "agents": [{"name": k, "description": v, "available": k != "alphazero" or WEIGHTS.exists()}
                   for k, v in DESCRIPTIONS.items()],
        "levels": {"alphazero_simulations": ALPHAZERO_SIMS, "minimax_seconds": MINIMAX_SECONDS,
                   "mcts_simulations": MCTS_SIMS},
    }


@router.post("/api/connect4/move")
async def move(req: MoveRequest, request: Request):
    app = request.app
    key = request.headers.get("fly-client-ip") or (request.client.host if request.client else "unknown")
    if not app.state.move_rate.allow(key):
        raise HTTPException(429, "rate limit: try again in a few seconds")
    try:
        board = board_from(req.moves)
    except ValueError as e:
        raise HTTPException(400, str(e)) from None
    try:
        async with app.state.slots.acquire():
            result = await asyncio.to_thread(_timed, AGENTS[req.agent], board, req.level)
    except FileNotFoundError:
        raise HTTPException(503, "the AlphaZero network is not available on this server") from None
    except Busy:
        raise HTTPException(503, "server busy: try again shortly") from None
    return {"agent": req.agent, "level": req.level, **result}


def _timed(fn, board, level):
    start = time.perf_counter()
    result = fn(board, level)
    result["seconds"] = round(time.perf_counter() - start, 3)
    return result
