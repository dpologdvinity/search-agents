"""Endgame API: solved king-and-piece endgames, served from the committed tablebases.

GET  /api/endgame/meta                      sizes, outcome counts, and DTM histograms for KQK and KRK
GET  /api/endgame/random?piece=Q&want=strong_wins&stm=0
POST /api/endgame/analyze   {"piece": "Q", "wk": 1, "wp": 9, "bk": 36, "stm": 0}   or   {"fen": "..."}
GET  /api/endgame/heatmap?piece=Q&wk=0&wp=9&stm=1   the lone king's DTM on each of the 64 squares

Squares are numbers 0..63 with a1 = 0 and h8 = 63 (square = 8 * rank + file), the same as endgame.rules.
stm is the side to move: 0 = White (the side with the piece), 1 = Black (the lone king).

Every request is a table lookup or a few dozen legality checks, so nothing here runs a search. The tables are
loaded on first use and cached, and the per-request cost is microseconds. Requests share the move rate
limiter, the same one the game pages use for moves.
"""

from __future__ import annotations

import random

from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel, Field

from endgame.retro import ILLEGAL
from endgame.rules import BLACK, PIECES, WHITE, is_legal, parse_fen, square_name, to_fen
from endgame.tablebase import describe, load

router = APIRouter()

DESCRIPTION = (
    "A white king and one white piece against a lone black king, solved by retrograde analysis: every "
    "position's distance to mate (DTM) is known exactly, counted from checkmate backwards."
)


class AnalyzeRequest(BaseModel):
    fen: str | None = Field(None, max_length=120)
    piece: str | None = Field(None, pattern="^[QR]$")
    wk: int | None = Field(None, ge=0, le=63)
    wp: int | None = Field(None, ge=0, le=63)
    bk: int | None = Field(None, ge=0, le=63)
    stm: int | None = Field(None, ge=0, le=1)


def _client_key(request: Request) -> str:
    # Behind Fly.io's proxy the client address arrives in a header.
    return request.headers.get("fly-client-ip") or (request.client.host if request.client else "unknown")


def _limit(request: Request) -> None:
    if not request.app.state.move_rate.allow(_client_key(request)):
        raise HTTPException(429, "rate limit: try again in a few seconds")


def _table(piece: str):
    try:
        return load(piece)
    except FileNotFoundError:
        raise HTTPException(503, f"the K{piece}K tablebase is not built on this server") from None


def _meta_piece(piece: str) -> dict:
    tb = _table(piece)
    s = tb.summary()
    return {
        "name": tb.name,
        "legal": s["legal"],
        "legal_white_to_move": s["legal_white_to_move"],
        "legal_black_to_move": s["legal_black_to_move"],
        "won": s["win"],
        "lost": s["loss"],
        "drawn": s["draw"],
        "max_win_moves": s["max_win_moves"],
        "max_loss_moves": s["max_loss_moves"],
        "longest_win_example": s["longest_win_example"],
        "win_histogram": s["win_histogram"],
        "loss_histogram": s["loss_histogram"],
        "build_seconds": round(tb.build_seconds, 3),
    }


def _outcome_json(outcome: str, plies: int | None) -> dict:
    return {"outcome": outcome, "plies": plies, "moves": None if plies is None else (plies + 1) // 2}


def analyze_json(piece: str, pos) -> dict:
    """The position's value and every legal move's result, from the mover's point of view."""
    tb = _table(piece)
    outcome, plies = tb.result(pos)
    infos = tb.move_infos(pos)
    best = tb.best_move(pos)
    return {
        "piece": piece,
        "name": tb.name,
        "wk": pos[0], "wp": pos[1], "bk": pos[2], "stm": pos[3],
        "fen": to_fen(piece, pos),
        "value": _outcome_json(outcome, plies),
        "moves": [
            {
                "uci": info.uci,
                "san": info.san,
                "from": info.move[1],
                "to": info.move[2],
                "mover": info.move[0],
                "outcome": info.outcome,
                "plies": info.plies,
                "mate_in": info.moves,
                "mate": info.mate,
                "capture": info.move[0] == "k" and info.move[2] == pos[1],
            }
            for info in infos
        ],
        "best": None if best is None else {"uci": best.uci, "san": best.san},
    }


@router.get("/api/endgame/meta")
async def meta():
    return {
        "pieces": {p: _meta_piece(p) for p in PIECES},
        "description": DESCRIPTION,
    }


@router.get("/api/endgame/random")
async def random_position(request: Request,
                          piece: str = Query("Q", pattern="^[QR]$"),
                          want: str = Query("strong_wins", pattern="^(strong_wins|draw|any)$"),
                          stm: int | None = Query(None, ge=0, le=1)):
    _limit(request)
    tb = _table(piece)
    try:
        pos = tb.random_position(random.Random(), stm=stm, want=want)
    except ValueError as e:
        raise HTTPException(400, str(e)) from None
    return analyze_json(piece, pos)


@router.post("/api/endgame/analyze")
async def analyze(req: AnalyzeRequest, request: Request):
    _limit(request)
    if req.fen is not None:
        try:
            piece, pos = parse_fen(req.fen)
        except ValueError as e:
            raise HTTPException(400, str(e)) from None
    else:
        if None in (req.piece, req.wk, req.wp, req.bk, req.stm):
            raise HTTPException(400, "send a fen, or piece, wk, wp, bk and stm")
        piece = req.piece
        pos = (req.wk, req.wp, req.bk, req.stm)
        if not is_legal(piece, pos):
            raise HTTPException(400, "not a legal position: squares repeat, the kings touch, "
                                     "or the side not to move is in check")
    return analyze_json(piece, pos)


@router.get("/api/endgame/heatmap")
async def heatmap(request: Request,
                  piece: str = Query("Q", pattern="^[QR]$"),
                  wk: int = Query(..., ge=0, le=63),
                  wp: int = Query(..., ge=0, le=63),
                  stm: int = Query(BLACK, ge=0, le=1)):
    """For each square of the lone black king, with the white king and piece fixed: the value for the side to move.

    Squares the black king cannot stand on (it would overlap a piece, or touch the white king) are null.
    """
    _limit(request)
    if wk == wp:
        raise HTTPException(400, "the white king and piece cannot share a square")
    tb = _table(piece)
    row = tb.king_row(wk, wp, stm)
    outcomes, plies = [], []
    for code in row.tolist():
        if code == ILLEGAL:
            outcomes.append(None)
            plies.append(None)
            continue
        outcome, n = describe(code)
        outcomes.append(outcome)
        plies.append(n)
    return {
        "piece": piece,
        "wk": wk, "wp": wp, "stm": stm,
        "squares": [square_name(sq) for sq in range(64)],
        "outcome": outcomes,
        "plies": plies,
        "moves": [None if p is None else (p + 1) // 2 for p in plies],
        "side_to_move": "white" if stm == WHITE else "black",
    }
