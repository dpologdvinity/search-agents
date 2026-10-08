"""Checkers API: legal moves for the player, and moves with analysis for the agents.

POST /api/checkers/legal {"board": [32 ints], "turn": 1 | -1}
POST /api/checkers/move  {"board": [...], "turn": 1 | -1, "agent": "alphabeta" | "minimax", "level": 1-3}

Boards use checkers.board's encoding: +1 red man, +2 red king, -1 white man,
-2 white king, 0 empty; red moves up the board and moves first.
"""

from __future__ import annotations

import asyncio
import time
from typing import Literal

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from checkers.board import START, legal_moves, notation, parse_board
from checkers.search import WIN, AlphaBeta, Minimax

from .limits import Busy

router = APIRouter()

ALPHABETA_SECONDS = {1: 0.2, 2: 0.8, 3: 2.0}
MINIMAX_DEPTH = {1: 2, 2: 4, 3: 5}
DESCRIPTIONS = {
    "alphabeta": (
        "Alpha-beta search: assumes you will always answer with the reply that is worst for it, and skips "
        "lines it can prove are worse than one already found. Iterative deepening, a transposition table, "
        "and best-move-first ordering let it look deeper in the same time."
    ),
    "minimax": (
        "Plain minimax to a fixed depth: the same reasoning, but it examines every line. Compare its node "
        "count with alpha-beta's to see how much pruning saves."
    ),
}


class BoardRequest(BaseModel):
    board: list[int] = Field(min_length=32, max_length=32)
    turn: Literal[1, -1] = 1


class MoveRequest(BoardRequest):
    agent: Literal["alphabeta", "minimax"] = "alphabeta"
    level: int = Field(2, ge=1, le=3)


def move_json(m):
    return {"path": list(m.path), "captured": list(m.captured), "result": list(m.result), "notation": notation(m)}


def describe(score):
    if score is None:
        return None
    if score >= WIN - 200:
        return {"result": "win", "plies": WIN - score}
    if score <= -(WIN - 200):
        return {"result": "loss", "plies": WIN + score}
    return {"result": "eval", "score": score}


def _board(req: BoardRequest):
    try:
        return parse_board(req.board)
    except ValueError as e:
        raise HTTPException(400, str(e)) from None


def choose(board, turn, agent, level):
    if agent == "alphabeta":
        searcher = AlphaBeta(max_seconds=ALPHABETA_SECONDS[level])
    else:
        searcher = Minimax(MINIMAX_DEPTH[level])
    start = time.perf_counter()
    info = searcher.search(board, turn)
    reply_moves = legal_moves(info.move.result, -turn)
    return {
        "move": move_json(info.move),
        "analysis": {
            "agent": agent,
            "depth": info.depth,
            "nodes": info.nodes,
            "cutoffs": info.cutoffs,
            "best": describe(info.score),
            "root": [{"notation": notation(m), "path": list(m.path), "score": describe(sc)}
                     for m, sc in sorted(info.root, key=lambda x: -(x[1] if x[1] is not None else -WIN))],
            "seconds": round(time.perf_counter() - start, 3),
        },
        "reply": [move_json(m) for m in reply_moves],
    }


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
