"""Minesweeper API: what the agent can prove about one board, and how likely each covered cell is to be a mine.

GET  /api/minesweeper/meta
POST /api/minesweeper/analyze  {"rows": 9, "cols": 9, "mines": 10, "cells": [81 ints], "level": 3}

`cells` is row-major: -1 for a covered cell, 0..8 for a revealed number. The page keeps the hidden
mines and applies the moves itself. This endpoint sees only what a player sees, so it cannot leak
the layout. The counting lives in minesweeper.inference; this file validates input, applies the
abuse guards, and shapes the JSON.

A request costs a few milliseconds on a typical expert board. The worst single position seen in
testing took about half a second, which is why the search runs in a thread behind the slot limit.
"""

from __future__ import annotations

import asyncio
import math

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from minesweeper.agents import AGENT_NAMES, DESCRIPTIONS, best_guess
from minesweeper.board import PRESETS, UNKNOWN, View, label
from minesweeper.inference import LEVEL_PROBABILITY, LEVEL_RULES, analyse

from .limits import Busy

router = APIRouter()

# The largest board the page offers is the expert size; the API accepts nothing bigger.
MAX_ROWS = max(rows for rows, _, _ in PRESETS.values())
MAX_COLS = max(cols for _, cols, _ in PRESETS.values())


class AnalyseRequest(BaseModel):
    rows: int = Field(ge=2, le=MAX_ROWS)
    cols: int = Field(ge=2, le=MAX_COLS)
    mines: int = Field(ge=1)
    cells: list[int] = Field(max_length=MAX_ROWS * MAX_COLS)
    level: int = Field(LEVEL_PROBABILITY, ge=LEVEL_RULES, le=LEVEL_PROBABILITY)


def _client_key(request: Request) -> str:
    # Behind Fly.io's proxy the client address arrives in a header.
    return request.headers.get("fly-client-ip") or (request.client.host if request.client else "unknown")


def _check_view(req: AnalyseRequest) -> View:
    """Reject boards of the wrong shape, cell values outside -1..8, or a mine count that cannot fit (HTTP 400)."""
    size = req.rows * req.cols
    if len(req.cells) != size:
        raise HTTPException(400, f"a {req.rows}x{req.cols} board needs {size} cells, got {len(req.cells)}")
    if any(v != UNKNOWN and not 0 <= v <= 8 for v in req.cells):
        raise HTTPException(400, "cells must be -1 (covered) or a number from 0 to 8")
    if req.mines >= size:
        raise HTTPException(400, "the mine count must be less than the number of cells")
    return View(req.rows, req.cols, req.mines, tuple(req.cells))


def analysis_json(view: View, level: int) -> dict:
    """Run the agent's analysis and describe it in the shape the page reads.

    `probability` has one entry per cell: null for a revealed cell, otherwise the exact mine probability
    rounded to six places (1.0 for a proven mine, 0.0 for a proven safe cell).
    """
    a = analyse(view, level)
    probability: list[float | None] = []
    for c, v in enumerate(view.cells):
        if v != UNKNOWN:
            probability.append(None)
        elif c in a.mines:
            probability.append(1.0)
        elif c in a.safe:
            probability.append(0.0)
        else:
            p = a.probability(c)
            probability.append(None if p is None else round(p, 6))
    guess = None
    if level >= LEVEL_PROBABILITY and not a.safe and any(c in a.numerator for c in a.covered):
        cell = best_guess(a)
        guess = {"cell": cell, "label": label(view.cols, cell), "probability": probability[cell]}
    why = {str(c): a.why[c] for c in (*a.safe, *a.mines) if c in a.why}
    components = [
        {"cells": list(comp.cells), "constraints": comp.constraints, "layouts_log2": round(math.log2(comp.layouts), 3)}
        for comp in a.components
    ]
    return {
        "rows": view.rows,
        "cols": view.cols,
        "mines": view.mines,
        "level": level,
        "safe": list(a.safe),
        "certain_mines": list(a.mines),
        "why": why,
        "probability": probability,
        "guess": guess,
        "components": components,
        "interior": len(a.interior),
        "remaining": a.remaining,
    }


@router.get("/api/minesweeper/meta")
async def meta():
    return {
        "presets": [
            {"name": name, "rows": r, "cols": c, "mines": m} for name, (r, c, m) in PRESETS.items()
        ],
        "max_rows": MAX_ROWS,
        "max_cols": MAX_COLS,
        "agents": [{"name": name, "description": DESCRIPTIONS[name]} for name in AGENT_NAMES],
        "levels": {"rules": LEVEL_RULES, "csp": 2, "probability": LEVEL_PROBABILITY},
    }


@router.post("/api/minesweeper/analyze")
async def analyze_board(req: AnalyseRequest, request: Request):
    app = request.app
    if not app.state.move_rate.allow(_client_key(request)):
        raise HTTPException(429, "rate limit: try again in a few seconds")
    view = _check_view(req)
    try:
        async with app.state.slots.acquire():
            return await asyncio.to_thread(analysis_json, view, req.level)
    except Busy:
        raise HTTPException(503, "server busy: try again shortly") from None
    except ValueError as e:  # the revealed numbers contradict the mine count
        raise HTTPException(400, str(e)) from None
