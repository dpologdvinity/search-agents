"""Hex API: replay a position, report the result, and ask an agent for its move with the search it ran.

GET  /api/hexgame/meta
POST /api/hexgame/move  {"size": 7, "moves": [24, 0], "swap": false, "agent": "rave", "level": 2}
  -> {"size", "to_move", "winner", "chain", "move", "label", "analysis", "seconds"}

`moves` is the full history in play order: cell indexes (row * size + column), with -1 for the swap
move. The server replays it, so the client never has to be trusted with the board. If the game is
open and `agent` is given, the response includes that agent's move and its analysis. If the game is
over, `winner` and `chain` (the winning cells, start edge first) are set and no move is made.

Search budgets come from hexgame.agents.LEVELS, and each request holds a slot, so the server does
not run more than SEARCH_SLOTS searches at once.
"""

from __future__ import annotations

import asyncio
from typing import Literal

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from hexgame.agents import AGENTS, CHANCE_RING_WEIGHTS, LEVELS, choose
from hexgame.board import DEFAULT_N, MAX_N, MIN_N, SWAP, Board, label

from .limits import Busy, client_key

router = APIRouter()


class MoveRequest(BaseModel):
    size: int = Field(DEFAULT_N, ge=MIN_N, le=MAX_N)
    moves: list[int] = Field(default_factory=list, max_length=MAX_N * MAX_N + 1)
    swap: bool = False
    agent: Literal["rave", "uct", "shortest", "random", "chance"] | None = None
    level: int = Field(2, ge=1, le=3)


def _analysis_json(analysis: dict, n: int) -> dict:
    """The search summary the page reads. Labels are added so the page need not rebuild them."""
    out = dict(analysis)
    if out.get("pv"):
        out["pv_labels"] = [label(n, c) for c in out["pv"]]
    return out


@router.get("/api/hexgame/meta")
async def meta():
    return {
        "sizes": list(range(MIN_N, MAX_N + 1)),
        "default_size": DEFAULT_N,
        "agents": [{"name": k, "description": v} for k, v in AGENTS.items()],
        "levels": {str(k): v for k, v in LEVELS.items()},
        # The chance table: the weight per hex ring from the centre (rings 3 and beyond share the last).
        "chance": {"ring_weights": list(CHANCE_RING_WEIGHTS)},
    }


@router.post("/api/hexgame/move")
async def move(req: MoveRequest, request: Request):
    app = request.app
    key = client_key(request)
    if not app.state.move_rate.allow(key):
        raise HTTPException(429, "rate limit: try again in a few seconds")
    try:
        board = Board.from_moves(req.size, req.moves, swap_rule=req.swap)
    except ValueError as e:
        raise HTTPException(400, str(e)) from None

    winner = board.winner()
    chain = board.chain() if winner is not None else None
    result = {
        "size": req.size,
        "to_move": board.to_move,
        "winner": winner,
        "chain": chain,
        "move": None,
        "label": None,
        "analysis": None,
        "seconds": 0.0,
    }
    if winner is not None or req.agent is None:
        return result

    try:
        async with app.state.slots.acquire():
            cell, analysis = await asyncio.to_thread(choose, board, req.agent, level=req.level)
    except Busy:
        raise HTTPException(503, "server busy: try again shortly") from None
    result["move"] = cell
    result["label"] = "swap" if cell == SWAP else label(req.size, cell)
    result["analysis"] = _analysis_json(analysis, req.size)
    result["seconds"] = round(analysis.get("seconds", 0.0), 3)
    return result
