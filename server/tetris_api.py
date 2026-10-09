"""Tetris API: the tuned weights for the page, and a placement chooser for hints and tests.

GET  /api/tetris/meta
POST /api/tetris/choose  {"board": [20 row bitmasks, row 0 = bottom], "piece": "T", "preview": "I" or null,
                          "weights": [9 floats] or null, "lookahead": false}

The page plays the AI itself in JavaScript (web/js/tetris_engine.js) with the weights from /meta, so the
server is not on the game's path. /choose exists for the terminal tools, for scripts, and as the reference
the JavaScript engine is tested against. One call is a few milliseconds of CPU.

Boards are bitmasks in the same layout as tetris/board.py: bit x of row r = column x filled, row 0 at the
bottom. A row is an integer from 0 to 1023 (ten columns).
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from tetris.board import FULL, H, W
from tetris.features import FEATURES, HAND_WEIGHTS, WEIGHT_LIMIT
from tetris.pieces import PIECES
from tetris.search import best_move
from tetris.tuned import load_tuned

from .limits import Busy, client_key

router = APIRouter()

LOOKAHEAD_TOP_K = 6
# Weights from the request can be a little outside the GA's range; anything wildly large is rejected
# so the scores stay finite.
WEIGHT_CAP = 10 * WEIGHT_LIMIT


class ChooseRequest(BaseModel):
    board: list[int] = Field(min_length=H, max_length=H)
    piece: str = Field(pattern=f"^[{PIECES}]$")
    preview: str | None = Field(None, pattern=f"^[{PIECES}]$")
    weights: list[float] | None = Field(None, min_length=len(FEATURES), max_length=len(FEATURES))
    lookahead: bool = False


@router.get("/api/tetris/meta")
async def meta():
    doc = load_tuned()
    return {
        "width": W,
        "height": H,
        "pieces": list(PIECES),
        "features": list(FEATURES),
        "hand_weights": list(HAND_WEIGHTS),
        "weight_limit": WEIGHT_LIMIT,
        "tuned_weights": doc["weights"],
        "source": doc.get("source", "tetris/tuned.json"),
        "train_fitness": doc.get("train_fitness"),
        "config": doc.get("config"),
        "history": doc.get("history", []),
        "summary": doc.get("summary"),
        "lookahead_top_k": LOOKAHEAD_TOP_K,
    }


def choose_json(board: list[int], piece: str, preview: str | None, weights, lookahead: bool) -> dict:
    """Run the placement search and shape the answer: the choice and every one-piece candidate."""
    move, moves = best_move(tuple(board), piece, weights,
                            preview=preview if lookahead else None, top_k=LOOKAHEAD_TOP_K)
    chosen = None
    if move is not None:
        chosen = {"rot": move.rot, "x": move.x, "y": move.y, "lines": move.lines,
                  "score": round(move.score, 4), "total": round(move.total, 4)}
    return {
        "piece": piece,
        "game_over": move is None,
        "chosen": chosen,
        "candidates": [{"rot": m.rot, "x": m.x, "y": m.y, "lines": m.lines, "score": round(m.score, 4)}
                       for m in moves],
    }


@router.post("/api/tetris/choose")
async def choose(req: ChooseRequest, request: Request):
    app = request.app
    if not app.state.move_rate.allow(client_key(request)):
        raise HTTPException(429, "rate limit: slow down a little")
    if any(v < 0 or v > FULL for v in req.board):
        raise HTTPException(400, f"each row is a bitmask from 0 to {FULL}")
    weights = list(HAND_WEIGHTS) if req.weights is None else req.weights
    if any(abs(w) > WEIGHT_CAP for w in weights):
        raise HTTPException(400, f"weights must be within +-{WEIGHT_CAP:g}")
    if req.lookahead and req.preview is None:
        raise HTTPException(400, "lookahead needs a preview piece")
    try:
        # The search takes a few milliseconds, so it runs inline under the shared search-slot limit.
        async with app.state.slots.acquire():
            return choose_json(req.board, req.piece, req.preview, weights, req.lookahead)
    except Busy:
        raise HTTPException(503, "server busy: try again shortly") from None
