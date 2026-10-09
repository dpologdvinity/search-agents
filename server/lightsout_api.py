"""Lights Out API: random boards, and exact solutions with the elimination trace for the visualizer.

GET  /api/lightsout/meta
GET  /api/lightsout/random?n=5&solvable=true
POST /api/lightsout/solve  {"n": 5, "board": [25 ints, 0 = dark, 1 = lit]}

Boards are flat lists in row-major order: cell i is row i // n, column i % n.
Presses are the same kind of list. The solver lives in lightsout.solver; this file only
validates input, applies the abuse guards, and shapes the JSON.
"""

from __future__ import annotations

import asyncio
import random

from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel, Field

from lightsout.board import DEFAULT_N, MAX_N, MIN_N, label, random_board
from lightsout.solver import Solution, solve

from .limits import Busy, client_key

router = APIRouter()

# The elimination trace has one step per column and lists every row operation, so it grows
# roughly with n**4. Send it only up to 7x7 to keep responses small; bigger boards get null.
TRACE_MAX_N = 7

DESCRIPTION = (
    "Press a light to toggle it and its orthogonal neighbours. Goal: every light dark. Each press is a "
    "vector over GF(2) and the board is a linear system A x = b, solved by Gaussian elimination mod 2."
)


class SolveRequest(BaseModel):
    n: int = Field(DEFAULT_N, ge=MIN_N, le=MAX_N)
    board: list[int] = Field(max_length=MAX_N * MAX_N)


def _check_board(req: SolveRequest) -> None:
    """Reject boards of the wrong shape or with values other than 0 and 1 (HTTP 400)."""
    if len(req.board) != req.n * req.n:
        raise HTTPException(400, f"a {req.n}x{req.n} board needs {req.n * req.n} cells, got {len(req.board)}")
    if any(v not in (0, 1) for v in req.board):
        raise HTTPException(400, "cells must be 0 (dark) or 1 (lit)")


def _trace_json(sol: Solution) -> dict | None:
    """The elimination as plain JSON. Steps are replayed client-side: swap, then XOR the pivot into each cleared row."""
    if sol.trace is None:
        return None
    return {
        "start": list(sol.trace.start),
        "steps": [
            {"col": s.col, "pivot": s.pivot, "swap": list(s.swap) if s.swap else None, "cleared": list(s.cleared)}
            for s in sol.trace.steps
        ],
    }


def solve_json(n: int, board: list[int]) -> dict:
    """Solve one board and describe the result in the JSON shape the page reads."""
    sol = solve(n, board, trace=n <= TRACE_MAX_N)
    presses = sol.presses
    if sol.solvable:
        pressed = set(presses)
        press_grid = [[1 if r * n + c in pressed else 0 for c in range(n)] for r in range(n)]
        reason = None
    else:
        press_grid = None
        reason = (f"no press sequence clears this board: rank {sol.rank} of {n * n}, and the board is outside "
                  "the reachable set")
    return {
        "n": n,
        "solvable": sol.solvable,
        "rank": sol.rank,
        "nullity": sol.nullity,
        "solutions": sol.count,
        "solution_sizes": list(sol.solution_sizes),
        "presses": list(presses) if presses is not None else None,
        "moves": [label(n, c) for c in presses] if presses is not None else None,
        "min_presses": len(presses) if presses is not None else None,
        "press_grid": press_grid,
        "reason": reason,
        "elimination": _trace_json(sol),
    }


@router.get("/api/lightsout/meta")
async def meta():
    return {
        "sizes": list(range(MIN_N, MAX_N + 1)),
        "default_size": DEFAULT_N,
        "trace_max_size": TRACE_MAX_N,
        "description": DESCRIPTION,
    }


@router.get("/api/lightsout/random")
async def random_puzzle(n: int = Query(DEFAULT_N, ge=MIN_N, le=MAX_N), solvable: bool = True):
    board = random_board(n, random.Random(), solvable=solvable)
    return {"n": n, "board": list(board), "solvable": solve(n, board).solvable}


@router.post("/api/lightsout/solve")
async def solve_board(req: SolveRequest, request: Request):
    app = request.app
    if not app.state.rate.allow(client_key(request)):
        raise HTTPException(429, "rate limit: try again in a few seconds")
    _check_board(req)
    try:
        async with app.state.slots.acquire():
            return await asyncio.to_thread(solve_json, req.n, req.board)
    except Busy:
        raise HTTPException(503, "server busy: try again shortly") from None
