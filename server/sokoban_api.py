"""Sokoban API: levels, one search run with its progress samples, and a position read-out.

GET  /api/sokoban/levels                     the level list with optimal push counts
GET  /api/sokoban/level/{number}             one level: its rows, dead squares, push distances
POST /api/sokoban/solve     {level, boxes?, player?, algorithm, heuristic, prune, budget}
POST /api/sokoban/evaluate  {level, boxes}   deadlock status, frozen boxes, heuristic values, pairing

Cells are interior indices i = row * cols + col (0-based, walls included), which is how the page
addresses the grid. The padded grid used inside sokoban/ is converted at the edges of this file.
The searches live in sokoban.search; this file validates input, applies the abuse guards, and shapes JSON.
"""

from __future__ import annotations

import asyncio
from typing import Literal

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from sokoban.analysis import describe
from sokoban.board import Level, render
from sokoban.levels import OPTIMAL_PUSHES, all_levels, get_level
from sokoban.search import DEFAULT_NODES, solve
from sokoban.tables import tables

from .limits import Busy, client_key

router = APIRouter()

# A search expands at most this many states and stops after SECONDS_CAP. The serving machine is
# shared and small, so these caps are the limit on CPU per request, not a suggestion.
NODE_CAP = 20_000
SECONDS_CAP = 5.0
LEVEL_COUNT = len(all_levels())


class SolveRequest(BaseModel):
    level: int = Field(1, ge=1, le=LEVEL_COUNT)
    boxes: list[int] | None = Field(None, max_length=64)  # interior indices; None = the level's start
    player: int | None = None
    algorithm: Literal["bfs", "greedy", "astar"] = "astar"
    heuristic: Literal["none", "simple", "matching"] = "matching"
    prune: bool = True
    budget: int = Field(DEFAULT_NODES, ge=1, le=NODE_CAP)


class EvaluateRequest(BaseModel):
    level: int = Field(1, ge=1, le=LEVEL_COUNT)
    boxes: list[int] = Field(max_length=64)


def _interior(level: Level, padded: int) -> int:
    """Padded flat index to the page's interior index."""
    r, c = level.rc(padded)
    return (r - 1) * (level.width - 2) + (c - 1)


def _padded(level: Level, interior: int) -> int:
    """Interior index to the padded flat index used by sokoban/."""
    cols = level.width - 2
    r, c = divmod(interior, cols)
    return level.at(r + 1, c + 1)


def _check_layout(level: Level, boxes: list[int], player: int | None) -> tuple[list[int], int]:
    """Validate interior indices and convert them. Raises HTTP 400 for a layout that cannot exist."""
    cols = level.width - 2
    rows = level.height - 2
    cells = rows * cols
    if any(not 0 <= b < cells for b in boxes):
        raise HTTPException(400, f"box indices must be between 0 and {cells - 1}")
    padded = [_padded(level, b) for b in boxes]
    if len(set(padded)) != len(padded):
        raise HTTPException(400, "two boxes on one cell")
    if any(not level.floor[p] for p in padded):
        raise HTTPException(400, "a box is on a wall")
    if len(padded) != len(level.goals):
        raise HTTPException(400, f"this level has {len(level.goals)} boxes")
    if player is None:
        return padded, level.start_player
    if not 0 <= player < cells:
        raise HTTPException(400, f"player index must be between 0 and {cells - 1}")
    p = _padded(level, player)
    if not level.floor[p] or p in padded:
        raise HTTPException(400, "the player must stand on floor, not on a box")
    return padded, p


def level_json(level: Level, number: int) -> dict:
    """One level for the page: the rows as text, the goals, dead squares, and the push distance per cell."""
    t = tables(level)
    cols = level.width - 2
    rows = level.height - 2
    interior = [_interior(level, p) for p in range(len(level.floor)) if level.floor[p]]
    # Push distance to the nearest goal the box can reach, per interior cell; -1 for walls and dead squares.
    nearest = [-1] * (rows * cols)
    dead = []
    for p in range(len(level.floor)):
        if not level.floor[p]:
            continue
        i = _interior(level, p)
        nearest[i] = t.nearest[p]
        if t.dead[p]:
            dead.append(i)
    return {
        "number": number,
        "name": level.name,
        "cols": cols,
        "rows": rows,
        "optimal_pushes": OPTIMAL_PUSHES[level.name],
        "text": render(level, level.start_player, level.start_boxes),
        "player": _interior(level, level.start_player),
        "boxes": [_interior(level, b) for b in level.start_boxes],
        "goals": [_interior(level, g) for g in level.goals],
        "dead": sorted(dead),
        "nearest": nearest,
        "floor": sorted(interior),
    }


def solve_json(level_number: int, boxes: list[int] | None, player: int | None, algorithm: str, heuristic: str,
               prune: bool, budget: int) -> dict:
    """Run one search and describe it in the JSON shape the page reads."""
    level = get_level(level_number)
    if boxes is not None:
        padded, start = _check_layout(level, boxes, player)
        if player is None and start in padded:
            # Without a player the level's start cell stands in for one, so it cannot hold a box.
            raise HTTPException(400, "a box is on the level's start cell: send the player's cell too")
        level = level.with_state(start, padded)
    res = solve(level, algorithm, heuristic, prune, node_budget=budget, time_limit=SECONDS_CAP)
    return {
        "level": level_number,
        "algorithm": algorithm,
        "heuristic": "none" if algorithm == "bfs" else heuristic,
        "prune": prune,
        "status": res.status,
        "solved": res.solved,
        "pushes": res.pushes,
        "moves": res.moves,
        "optimal_pushes": OPTIMAL_PUSHES[level.name] if boxes is None else None,
        "expanded": res.expanded,
        "generated": res.generated,
        "pruned": res.pruned,
        "seconds": round(res.seconds, 4),
        "h_start": res.h_start,
        # (nodes expanded, seconds) every so often, so the page can replay the run's progress.
        "samples": [[n, round(s, 4)] for n, s in res.samples],
    }


def evaluate_json(level_number: int, boxes: list[int]) -> dict:
    """Deadlock status, frozen boxes, dead squares and heuristic values for a box layout."""
    level = get_level(level_number)
    padded, _ = _check_layout(level, boxes, None)
    pos = describe(level, padded)
    return {
        "deadlock": pos.deadlock,
        "frozen": sorted(_interior(level, b) for b in pos.frozen),
        "dead": sorted(_interior(level, i) for i in pos.dead_squares),
        "h_simple": pos.h_simple,
        "h_matching": pos.h_matching,
        "pairs": [[_interior(level, b), _interior(level, g), n] for b, g, n in pos.pairs],
    }


@router.get("/api/sokoban/levels")
async def levels():
    return {
        "count": LEVEL_COUNT,
        "levels": [
            {"number": i, "name": lv.name, "optimal_pushes": OPTIMAL_PUSHES[lv.name], "boxes": len(lv.goals)}
            for i, lv in enumerate(all_levels(), 1)
        ],
        "node_cap": NODE_CAP,
        "seconds_cap": SECONDS_CAP,
    }


@router.get("/api/sokoban/level/{number}")
async def one_level(number: int):
    if not 1 <= number <= LEVEL_COUNT:
        raise HTTPException(404, f"there are {LEVEL_COUNT} levels")
    return level_json(get_level(number), number)


@router.post("/api/sokoban/solve")
async def solve_level(req: SolveRequest, request: Request):
    app = request.app
    if not app.state.rate.allow(client_key(request)):
        raise HTTPException(429, "rate limit: try again in a few seconds")
    try:
        async with app.state.slots.acquire():
            return await asyncio.to_thread(solve_json, req.level, req.boxes, req.player, req.algorithm,
                                           req.heuristic, req.prune, req.budget)
    except Busy:
        raise HTTPException(503, "server busy: try again shortly") from None


@router.post("/api/sokoban/evaluate")
async def evaluate(req: EvaluateRequest, request: Request):
    app = request.app
    if not app.state.move_rate.allow(client_key(request)):
        raise HTTPException(429, "rate limit: try again in a few seconds")
    return await asyncio.to_thread(evaluate_json, req.level, req.boxes)
