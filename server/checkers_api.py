"""Checkers API: legal moves for the player, and moves with analysis for the agents.

GET  /api/checkers/meta
GET  /api/checkers/describe?red=...&red_level=...&white=...&white_level=...&size=...&forced=...
     the line that says who plays ("Red: MCTS (1,000 playouts) vs White: ..."), the same one the CLI prints
POST /api/checkers/legal {"board": [...], "size": 8 | 10 | 12, "turn": 1 | -1, "forced": true | false}
POST /api/checkers/move  {"board": [...], "size": ..., "turn": ..., "forced": ...,
                          "agent": "alphabeta" | "minimax" | "mcts" | "greedy" | "random" | "chance", "level": 1-3}

Boards use checkers.board's encoding: +1 red man, +2 red king, -1 white man,
-2 white king, 0 empty; red moves up the board and moves first. A size x size board
has size*size/2 squares, so the board's length must match "size" (422 otherwise).
"forced" picks the capture rule: true (default) makes captures mandatory, false
lets a player make a simple move even when a capture exists.
"""

from __future__ import annotations

import asyncio
from typing import Literal

from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel, Field, model_validator

from checkers.agents import (
    ALPHABETA_SECONDS,
    DESCRIPTIONS,
    MCTS_ITERATIONS,
    MINIMAX_DEPTH,
    choose,
    matchup_text,
    move_json,
)
from checkers.board import RED, ROWS_PER_SIDE, SIZES, WHITE, for_size, legal_moves, parse_board
from checkers.chance import CHANCE_WEIGHTS

from .limits import Busy, client_key

router = APIRouter()


class BoardRequest(BaseModel):
    board: list[int] = Field(min_length=32, max_length=72)
    size: Literal[8, 10, 12] = 8
    turn: Literal[1, -1] = 1
    forced: bool = True

    @model_validator(mode="after")
    def _board_fits_size(self):
        # Checked here rather than by the length bounds above, so a 50-square board sent
        # with the default size of 8 is rejected with a message about the size.
        cells = for_size(self.size).cells
        if len(self.board) != cells:
            raise ValueError(f"a {self.size}x{self.size} board has {cells} squares, not {len(self.board)}")
        return self


class MoveRequest(BoardRequest):
    agent: Literal["alphabeta", "minimax", "mcts", "greedy", "random", "chance"] = "alphabeta"
    level: int = Field(2, ge=1, le=3)


def _board(req: BoardRequest):
    try:
        return parse_board(req.board, req.size)
    except ValueError as e:
        raise HTTPException(400, str(e)) from None


@router.get("/api/checkers/meta")
async def meta():
    return {
        "agents": [{"name": k, "description": v} for k, v in DESCRIPTIONS.items()],
        "forced": True,  # the default capture rule; requests may send "forced": false
        "start": list(for_size(8).START),
        "sizes": [{"size": n, "squares": for_size(n).cells, "rows_per_side": ROWS_PER_SIDE[n],
                   "start": list(for_size(n).START)} for n in SIZES],
        "levels": {"alphabeta_seconds": ALPHABETA_SECONDS,
                   "minimax_depth": {f"{n}x{n}": {"forced": MINIMAX_DEPTH[(n, True)],
                                                  "optional": MINIMAX_DEPTH[(n, False)]} for n in SIZES},
                   "mcts_iterations": {f"{n}x{n}": MCTS_ITERATIONS[n] for n in SIZES}},
        "chance_weights": CHANCE_WEIGHTS,
    }


PLAYER = Literal["alphabeta", "minimax", "mcts", "greedy", "random", "chance", "human"]


@router.get("/api/checkers/describe")
async def describe(red: PLAYER = "human", red_level: int = Query(2, ge=1, le=3), white: PLAYER = "alphabeta",
                   white_level: int = Query(2, ge=1, le=3), size: int = 8, forced: bool = True):
    """Who plays, named with its algorithm and budget. The page and the CLI both show this text."""
    if size not in SIZES:  # a query string is not coerced to a Literal, so check the size here
        raise HTTPException(422, "size must be 8, 10 or 12")
    players = {RED: red, WHITE: white}
    levels = {RED: red_level, WHITE: white_level}
    return {"text": matchup_text(players, levels, size, forced=forced)}


@router.post("/api/checkers/legal")
async def legal(req: BoardRequest):
    board = _board(req)
    return {"moves": [move_json(m) for m in legal_moves(board, req.turn, req.forced)]}


@router.post("/api/checkers/move")
async def move(req: MoveRequest, request: Request):
    app = request.app
    key = client_key(request)
    if not app.state.move_rate.allow(key):
        raise HTTPException(429, "rate limit: try again in a few seconds")
    board = _board(req)
    if not legal_moves(board, req.turn, req.forced):
        raise HTTPException(400, "no legal moves: the game is over")
    try:
        async with app.state.slots.acquire():
            return await asyncio.to_thread(choose, board, req.turn, req.agent, req.level, forced=req.forced)
    except Busy:
        raise HTTPException(503, "server busy: try again shortly") from None
