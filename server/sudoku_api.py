"""Sudoku API: puzzles, and solves that return the search trace for animation.

GET  /api/sudoku/puzzle?set=generated|hard&index=N
POST /api/sudoku/solve  {"puzzle": "<81 chars>", "solver": "plain" | "mrv_fc" | "propagate"}
"""

from __future__ import annotations

import asyncio
import random
from dataclasses import asdict
from functools import lru_cache
from typing import Literal

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from sudoku.__main__ import HARD
from sudoku.generate import load_puzzles
from sudoku.solver import SOLVERS, parse

from .limits import Busy, client_key

router = APIRouter()
MAX_NODES = 200_000
TRACE_LIMIT = 20_000

DESCRIPTIONS = {
    "plain": "Backtracking over cells in row order. No heuristics.",
    "mrv_fc": "Fill the cell with the fewest legal values first, and back up as soon as any cell has none left.",
    "propagate": (
        "After every guess, repeatedly fill forced cells (only one candidate, or the only place for "
        "a digit in a row, column, or box) before guessing again."
    ),
}


@lru_cache(maxsize=1)
def generated_puzzles() -> list[dict]:
    return load_puzzles()


class SolveRequest(BaseModel):
    puzzle: str
    solver: Literal["plain", "mrv_fc", "propagate"] = "mrv_fc"


@router.get("/api/sudoku/meta")
async def meta():
    return {"solvers": [{"name": k, "description": v} for k, v in DESCRIPTIONS.items()],
            "hard": list(HARD), "generated_count": len(generated_puzzles()),
            "max_nodes": MAX_NODES, "trace_limit": TRACE_LIMIT}


@router.get("/api/sudoku/puzzle")
async def puzzle(set: Literal["generated", "hard"] = "generated", index: int | None = None,
                 name: str | None = None):
    if set == "hard":
        if name is None:
            name = random.choice(list(HARD))
        if name not in HARD:
            raise HTTPException(404, f"no hard puzzle named {name!r}")
        return {"set": "hard", "name": name, "puzzle": HARD[name]}
    puzzles = generated_puzzles()
    if index is None:
        index = random.randrange(len(puzzles))
    if not 0 <= index < len(puzzles):
        raise HTTPException(404, "index out of range")
    row = puzzles[index]
    return {"set": "generated", "index": index, "puzzle": row["puzzle"],
            "clues": row["clues"], "guesses": row["guesses"]}


@router.post("/api/sudoku/solve")
async def solve(req: SolveRequest, request: Request):
    app = request.app
    key = client_key(request)
    if not app.state.rate.allow(key):
        raise HTTPException(429, "rate limit: try again in a few seconds")
    try:
        parse(req.puzzle)
    except ValueError as e:
        raise HTTPException(400, str(e)) from None
    try:
        async with app.state.slots.acquire():
            result = await asyncio.to_thread(
                SOLVERS[req.solver], req.puzzle, max_nodes=MAX_NODES, trace_limit=TRACE_LIMIT)
    except Busy:
        raise HTTPException(503, "server busy: try again shortly") from None
    return asdict(result)
