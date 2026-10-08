"""Checkers API: legal moves for the player, and moves with analysis for the agents.

POST /api/checkers/legal {"board": [32 ints], "turn": 1 | -1}
POST /api/checkers/move  {"board": [...], "turn": 1 | -1, "agent": "alphabeta" | "minimax", "level": 1-3}

Boards use checkers.board's encoding: +1 red man, +2 red king, -1 white man,
-2 white king, 0 empty; red moves up the board and moves first.
"""

from __future__ import annotations

import asyncio
from typing import Literal

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from checkers.agents import ALPHABETA_SECONDS, DESCRIPTIONS, MINIMAX_DEPTH, choose, move_json
from checkers.board import START, legal_moves, parse_board

from .limits import Busy

router = APIRouter()


class BoardRequest(BaseModel):
    board: list[int] = Field(min_length=32, max_length=32)
    turn: Literal[1, -1] = 1


class MoveRequest(BoardRequest):
    agent: Literal["alphabeta", "minimax"] = "alphabeta"
    level: int = Field(2, ge=1, le=3)


def _board(req: BoardRequest):
    try:
        return parse_board(req.board)
    except ValueError as e:
        raise HTTPException(400, str(e)) from None


@router.get("/api/checkers/meta")
async def meta():
    return {"agents": [{"name": k, "description": v} for k, v in DESCRIPTIONS.items()],
            "start": list(START), "levels": {"alphabeta_seconds": ALPHABETA_SECONDS, "minimax_depth": MINIMAX_DEPTH}}


@router.post("/api/checkers/legal")
async def legal(req: BoardRequest):
    board = _board(req)
    return {"moves": [move_json(m) for m in legal_moves(board, req.turn)]}


@router.post("/api/checkers/move")
async def move(req: MoveRequest, request: Request):
    app = request.app
    key = request.headers.get("fly-client-ip") or (request.client.host if request.client else "unknown")
    if not app.state.move_rate.allow(key):
        raise HTTPException(429, "rate limit: try again in a few seconds")
    board = _board(req)
    if not legal_moves(board, req.turn):
        raise HTTPException(400, "no legal moves: the game is over")
    try:
        async with app.state.slots.acquire():
            return await asyncio.to_thread(choose, board, req.turn, req.agent, req.level)
    except Busy:
        raise HTTPException(503, "server busy: try again shortly") from None
