"""Snake API: the committed champion's weights and training history, and the planner's hint for a position.

GET  /api/snake/meta      board size, input and output names, layer sizes, agent descriptions, champion summary
GET  /api/snake/champion  the 291 weights as JSON, plus the champion's metadata
GET  /api/snake/history   one record per generation: best and mean fitness and apples
POST /api/snake/hint      {"body": [[x, y], ...], "heading": 1, "food": [x, y]}  ->  the planner's move and route

The page runs the evolved network itself (the weights are small), so the server only does the
planner: its BFS path and tail-chasing check. A request costs about a millisecond on a 12x12 board.
The champion file is read once and cached.
"""

from __future__ import annotations

import asyncio
import json
from functools import lru_cache

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from snake.agents import DESCRIPTIONS, planner, planner_path
from snake.board import ACTION_NAMES, BOARD, Game
from snake.net import CHAMPION_PATH, INPUT_NAMES, N_HIDDEN, OUTPUT_NAMES

from .limits import Busy

router = APIRouter()


@lru_cache(maxsize=1)
def _champion_data() -> dict:
    return json.loads(CHAMPION_PATH.read_text())


class HintRequest(BaseModel):
    body: list[list[int]] = Field(min_length=3, max_length=BOARD * BOARD)
    heading: int = Field(ge=0, le=3)
    food: list[int] | None = Field(None, min_length=2, max_length=2)


def _client_key(request: Request) -> str:
    # Behind Fly.io's proxy the client address arrives in a header.
    return request.headers.get("fly-client-ip") or (request.client.host if request.client else "unknown")


def _check_position(req: HintRequest) -> Game:
    """Validate a position from the page: in-bounds cells, a connected body, food off the body (HTTP 400)."""
    body = [tuple(c) for c in req.body]
    if any(len(c) != 2 for c in req.body):
        raise HTTPException(400, "each body cell is [x, y]")
    for x, y in body:
        if not (0 <= x < BOARD and 0 <= y < BOARD):
            raise HTTPException(400, f"cell {[x, y]} is off the {BOARD}x{BOARD} board")
    if len(set(body)) != len(body):
        raise HTTPException(400, "the body overlaps itself")
    for a, b in zip(body, body[1:]):
        if abs(a[0] - b[0]) + abs(a[1] - b[1]) != 1:
            raise HTTPException(400, "body segments must be neighbours in order, head first")
    food = tuple(req.food) if req.food is not None else None
    if food is not None and (not (0 <= food[0] < BOARD and 0 <= food[1] < BOARD) or food in body):
        raise HTTPException(400, "food must be on an empty cell of the board")
    return Game.from_state(BOARD, body, req.heading, food)


def hint_json(game: Game) -> dict:
    """The planner's move for `game`, and the route behind it (the cells after the head).

    "food" means the route ends on the food and passed the tail-chasing check; "tail" means the planner
    is following its tail until the food route opens; "none" means no route exists at all.
    """
    path = planner_path(game)
    if not path:
        route = "none"
    elif game.food is not None and path[-1] == game.food:
        route = "food"
    else:
        route = "tail"
    action = planner(game)
    return {
        "action": action,
        "action_name": ACTION_NAMES[action],
        "route": route,
        "path": [list(c) for c in path] if path else [],
    }


@router.get("/api/snake/meta")
async def meta():
    data = _champion_data()
    summary = {k: v for k, v in data.items() if k not in ("w1", "b1", "w2", "b2", "history")}
    return {
        "board": BOARD,
        "input_names": list(INPUT_NAMES),
        "output_names": list(OUTPUT_NAMES),
        "hidden": N_HIDDEN,
        "agents": DESCRIPTIONS,
        "champion": summary,
    }


@router.get("/api/snake/champion")
async def champion():
    return _champion_data()


@router.get("/api/snake/history")
async def history():
    return {"history": _champion_data().get("history", [])}


@router.post("/api/snake/hint")
async def hint(req: HintRequest, request: Request):
    app = request.app
    if not app.state.move_rate.allow(_client_key(request)):
        raise HTTPException(429, "rate limit: try again in a few seconds")
    game = _check_position(req)
    try:
        async with app.state.slots.acquire():
            return await asyncio.to_thread(hint_json, game)
    except Busy:
        raise HTTPException(503, "server busy: try again shortly") from None
