"""Nonogram API: the picture library, random puzzles with a unique solution, solving, and hints.

GET  /api/nonogram/meta
GET  /api/nonogram/puzzles
GET  /api/nonogram/puzzle/{id}
GET  /api/nonogram/random?rows=10&cols=10&seed=3
POST /api/nonogram/solve  {"row_clues": [[...]], "col_clues": [[...]], "method": "line|hybrid|sat", "trace": true}
POST /api/nonogram/hint   {"row_clues": ..., "col_clues": ..., "grid": [[-1|0|1]]}

Clues are lists of run lengths, [] for an empty line. Grids are row-major lists of rows: -1 unknown,
0 empty (or crossed out), 1 filled. Row and column indices in responses are 0-based; line names are 1-based.
The solver lives in the nonogram package; this file validates input, applies the abuse guards, and
shapes the JSON.
"""

from __future__ import annotations

import asyncio
import random
from typing import Annotated

from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel, Field

from nonogram.generate import RANDOM_MAX, RANDOM_MIN, NoUniquePicture, random_puzzle
from nonogram.library import get, library
from nonogram.puzzle import MAX_SIZE, Puzzle, validate_clues
from nonogram.solvers import GUESSES_SERVER, SAT_PROPAGATIONS_SERVER, SECONDS_SERVER, hint, solve

from .limits import Busy, client_key

router = APIRouter()

DESCRIPTION = (
    "Each row and column has a clue: the lengths of its runs of filled cells. Line solving reads each clue "
    "as an automaton and fixes the cells every arrangement agrees on. Where that stops, a DPLL SAT solver "
    "or a guess-and-propagate search finishes, and it counts solutions to prove the picture is unique."
)

# A line of n cells holds at most (n + 1) // 2 runs, so a longer clue list cannot describe any line.
Clues = list[Annotated[list[int], Field(max_length=(MAX_SIZE + 1) // 2)]]


class SolveRequest(BaseModel):
    row_clues: Clues = Field(max_length=MAX_SIZE)
    col_clues: Clues = Field(max_length=MAX_SIZE)
    method: str = "hybrid"
    trace: bool = True


class HintRequest(BaseModel):
    row_clues: Clues = Field(max_length=MAX_SIZE)
    col_clues: Clues = Field(max_length=MAX_SIZE)
    grid: list[Annotated[list[int], Field(max_length=MAX_SIZE)]] = Field(max_length=MAX_SIZE)


def _puzzle_from_clues(row_clues: Clues, col_clues: Clues) -> Puzzle:
    """A Puzzle from request clues, after the same checks the library gets. Raises HTTP 422 on bad clues."""
    rows = tuple(tuple(int(v) for v in clue) for clue in row_clues)
    cols = tuple(tuple(int(v) for v in clue) for clue in col_clues)
    try:
        validate_clues(rows, cols)
    except ValueError as e:
        raise HTTPException(422, str(e)) from None
    return Puzzle("request", rows, cols)


def _check_grid(grid: list[list[int]], rows: int, cols: int) -> None:
    """A grid of the right shape with values -1, 0 or 1 only (HTTP 400)."""
    if len(grid) != rows or any(len(row) != cols for row in grid):
        raise HTTPException(400, f"the grid must be {rows} rows of {cols} cells")
    if any(v not in (-1, 0, 1) for row in grid for v in row):
        raise HTTPException(400, "cells must be -1 (unknown), 0 (empty) or 1 (filled)")


def _puzzle_json(p: Puzzle) -> dict:
    return {
        "id": p.id,
        "name": p.name,
        "rows": p.rows,
        "cols": p.cols,
        "row_clues": [list(c) for c in p.row_clues],
        "col_clues": [list(c) for c in p.col_clues],
    }


def solve_json(puzzle: Puzzle, method: str, trace: bool) -> dict:
    """Run one solve with the server's budgets and time limit, and return the JSON the page reads.

    The time limit is the one that binds on large pictures; the counted budgets are backstops on small ones.
    A run that stops at either reports status "undecided", with timed_out telling the page which one.
    """
    kwargs = {"time_limit": SECONDS_SERVER}
    if method == "sat":
        kwargs["max_propagations"] = SAT_PROPAGATIONS_SERVER
    elif method == "hybrid":
        kwargs["max_guesses"] = GUESSES_SERVER
    return solve(puzzle, method, trace=trace, **kwargs).as_dict()


@router.get("/api/nonogram/meta")
async def meta():
    return {
        "sizes": list(range(RANDOM_MIN, RANDOM_MAX + 1)),
        "max_size": MAX_SIZE,
        "methods": ["line", "hybrid", "sat"],
        "description": DESCRIPTION,
    }


@router.get("/api/nonogram/puzzles")
async def puzzles():
    return {"puzzles": [{"id": p.id, "name": p.name, "rows": p.rows, "cols": p.cols} for p in library()]}


@router.get("/api/nonogram/puzzle/{pid}")
async def puzzle(pid: str):
    found = get(pid)
    if found is None:
        raise HTTPException(404, f"no puzzle called {pid!r}")
    return _puzzle_json(found)


@router.get("/api/nonogram/random")
async def random_endpoint(request: Request,
                          rows: int = Query(10, ge=RANDOM_MIN, le=RANDOM_MAX),
                          cols: int = Query(10, ge=RANDOM_MIN, le=RANDOM_MAX),
                          seed: int | None = Query(None, ge=0, lt=2**31)):
    app = request.app
    if not app.state.rate.allow(client_key(request)):
        raise HTTPException(429, "rate limit: try again in a few seconds")
    if seed is None:
        seed = random.randrange(2**31)
    try:
        async with app.state.slots.acquire():
            puzzle_, used_seed, attempts = await asyncio.to_thread(random_puzzle, rows, cols, seed)
    except NoUniquePicture:
        raise HTTPException(503, f"no unique {rows}x{cols} picture with seed {seed}: try another seed") from None
    except Busy:
        raise HTTPException(503, "server busy: try again shortly") from None
    body = _puzzle_json(puzzle_)
    body.update({"id": None, "name": f"Random {rows}x{cols}", "seed": used_seed, "attempts": attempts})
    return body


@router.post("/api/nonogram/solve")
async def solve_endpoint(req: SolveRequest, request: Request):
    app = request.app
    if req.method not in ("line", "hybrid", "sat"):
        raise HTTPException(422, "method must be line, hybrid or sat")
    if not app.state.rate.allow(client_key(request)):
        raise HTTPException(429, "rate limit: try again in a few seconds")
    puzzle_ = _puzzle_from_clues(req.row_clues, req.col_clues)
    try:
        async with app.state.slots.acquire():
            return await asyncio.to_thread(solve_json, puzzle_, req.method, req.trace)
    except Busy:
        raise HTTPException(503, "server busy: try again shortly") from None


@router.post("/api/nonogram/hint")
async def hint_endpoint(req: HintRequest, request: Request):
    app = request.app
    if not app.state.move_rate.allow(client_key(request)):
        raise HTTPException(429, "rate limit: try again in a few seconds")
    puzzle_ = _puzzle_from_clues(req.row_clues, req.col_clues)
    _check_grid(req.grid, puzzle_.rows, puzzle_.cols)
    # One pass of line analysis is cheap, so the hint runs on the event loop without a slot.
    return hint(puzzle_, req.grid)
