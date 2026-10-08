"""Connect Four API: ask an agent for its move and see how it decided.

POST /api/connect4/move
  {"moves": "4453", "agent": "alphazero" | "minimax" | "mcts", "level": 1-3}
  -> {"move": 3, "analysis": {...}, "seconds": ...}

`moves` lists the columns played so far, 1-based, as in Connect Four solver
test sets. `level` scales the search budget.
"""

from __future__ import annotations

import asyncio
import time
from typing import Literal

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field, field_validator

from connect4.board import COLS, Board
from connect4.mcts import MCTS
from connect4.minimax import WIN, Minimax

from .limits import Busy

router = APIRouter()

# Search budget per level. Seconds cap minimax so a turn stays responsive.
ALPHAZERO_SIMS = {1: 50, 2: 200, 3: 800}
MINIMAX_SECONDS = {1: 0.1, 2: 0.5, 3: 2.0}
MCTS_SIMS = {1: 300, 2: 1500, 3: 5000}

DESCRIPTIONS = {
    "alphazero": "PUCT search guided by a policy-value network trained only by self-play.",
    "minimax": "Alpha-beta negamax with a transposition table and a hand-built window evaluation.",
    "mcts": "Monte Carlo tree search with random rollouts and no learned knowledge.",
}


class MoveRequest(BaseModel):
    moves: str = Field("", max_length=42)
    agent: Literal["alphazero", "minimax", "mcts"] = "alphazero"
    level: int = Field(2, ge=1, le=3)

    @field_validator("moves")
    @classmethod
    def digits(cls, v):
        if any(c not in "1234567" for c in v):
            raise ValueError("moves must be digits 1-7")
        return v


def board_from(moves: str) -> Board:
    board = Board()
    for i, ch in enumerate(moves):
        col = int(ch) - 1
        if not board.can_play(col):
            raise ValueError(f"move {i + 1} plays full column {ch}")
        if board.last_player_won():
            raise ValueError("the game is already over")
        board = board.play(col)
    if board.last_player_won() or board.is_full():
        raise ValueError("the game is already over")
    return board


def alphazero_move(board: Board, level: int) -> dict:
    from connect4.net import load
    from connect4.puct import search

    net = load()
    tree = search(board, net.evaluate, simulations=ALPHAZERO_SIMS[level])
    root = tree.root
    visits = root.visits
    q = [float(root.value_sum[c] / visits[c]) if visits[c] else None for c in range(COLS)]
    _, value = net.evaluate([board])
    return {
        "move": int(visits.argmax()),
        "analysis": {
            "kind": "alphazero",
            "simulations": int(root.total_visits()),
            "prior": [round(float(p), 4) for p in root.prior],
            "visits": [int(v) for v in visits],
            "q": q,
            # Search value for the player to move, mapped from [-1, 1] to a win chance.
            "win_probability": round((tree.root_value() + 1) / 2, 4),
            "network_value": round(float(value[0]), 4),
        },
    }


def minimax_move(board: Board, level: int) -> dict:
    info = Minimax(max_depth=42, max_seconds=MINIMAX_SECONDS[level]).search(board)

    def describe(score):
        if score is None:
            return None
        if score >= WIN - 64:
            return {"result": "win", "plies": WIN - score}
        if score <= -(WIN - 64):
            return {"result": "loss", "plies": WIN + score}
        return {"result": "eval", "score": score}

    return {
        "move": info.move,
        "analysis": {
            "kind": "minimax",
            "depth": info.depth,
            "nodes": info.nodes,
            "scores": [describe(s) for s in info.root_scores] if info.root_scores else None,
            "best": describe(info.score),
        },
    }


def mcts_move(board: Board, level: int) -> dict:
    result = MCTS(simulations=MCTS_SIMS[level], max_seconds=3.0).search(board)
    return {
        "move": result["move"],
        "analysis": {
            "kind": "mcts",
            "simulations": result["simulations"],
            "visits": [int(result["visits"].get(c, 0)) for c in range(COLS)],
        },
    }


AGENTS = {"alphazero": alphazero_move, "minimax": minimax_move, "mcts": mcts_move}


@router.get("/api/connect4/meta")
async def meta():
    from connect4.net import WEIGHTS

    return {
        "agents": [{"name": k, "description": v, "available": k != "alphazero" or WEIGHTS.exists()}
                   for k, v in DESCRIPTIONS.items()],
        "levels": {"alphazero_simulations": ALPHAZERO_SIMS, "minimax_seconds": MINIMAX_SECONDS,
                   "mcts_simulations": MCTS_SIMS},
    }


@router.post("/api/connect4/move")
async def move(req: MoveRequest, request: Request):
    app = request.app
    key = request.headers.get("fly-client-ip") or (request.client.host if request.client else "unknown")
    if not app.state.move_rate.allow(key):
        raise HTTPException(429, "rate limit: try again in a few seconds")
    try:
        board = board_from(req.moves)
    except ValueError as e:
        raise HTTPException(400, str(e)) from None
    try:
        async with app.state.slots.acquire():
            result = await asyncio.to_thread(_timed, AGENTS[req.agent], board, req.level)
    except FileNotFoundError:
        raise HTTPException(503, "the AlphaZero network is not available on this server") from None
    except Busy:
        raise HTTPException(503, "server busy: try again shortly") from None
    return {"agent": req.agent, "level": req.level, **result}


def _timed(fn, board, level):
    start = time.perf_counter()
    result = fn(board, level)
    result["seconds"] = round(time.perf_counter() - start, 3)
    return result
