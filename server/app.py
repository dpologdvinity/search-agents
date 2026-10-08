"""FastAPI backend for the search-agents demo.

    uvicorn server.app:app --reload

Environment:
  ALLOWED_ORIGINS  comma-separated origins allowed to call the API (default: any)
  SEARCH_SLOTS     searches allowed to run at once (default 2)
  SOLVES_PER_MIN   N-Puzzle solves per client per minute (default 30)
  MOVES_PER_MIN    game moves per client per minute (default 120)
  PACMAN_EPISODES_PER_MIN  Pac-Man whole-game requests per client per minute (default 30)
  PACMAN_PLAYS_PER_MIN     Pac-Man human-play requests per client per minute (default 300)
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from . import (
    battleship_api,
    blackjack_api,
    checkers_api,
    connect4_api,
    game2048_api,
    lightsout_api,
    npuzzle_api,
    pacman_api,
    routes_api,
    sudoku_api,
    warehouse_api,
)
from .limits import Busy, RateLimiter, SearchSlots

ORIGINS = [o.strip() for o in os.environ.get("ALLOWED_ORIGINS", "").split(",") if o.strip()]
SLOTS = int(os.environ.get("SEARCH_SLOTS", "2"))
SOLVES_PER_MIN = int(os.environ.get("SOLVES_PER_MIN", "30"))
MOVES_PER_MIN = int(os.environ.get("MOVES_PER_MIN", "120"))


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.slots = SearchSlots(SLOTS, wait_seconds=5.0)
    app.state.rate = RateLimiter(SOLVES_PER_MIN, per=60.0)
    app.state.move_rate = RateLimiter(MOVES_PER_MIN, per=60.0)
    # Load data files at startup so the first request is not slow.
    from npuzzle.pdb import _tables

    _tables()
    from connect4.net import load as load_alphazero
    from game2048.ntuple import load as load_ntuple
    from npuzzle.neural import load as load_neural

    for load in (load_neural, load_alphazero, load_ntuple):
        try:
            load()
        except FileNotFoundError:
            pass
    yield


app = FastAPI(title="search-agents", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=ORIGINS or ["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)
app.include_router(npuzzle_api.router)
app.include_router(connect4_api.router)
app.include_router(game2048_api.router)
app.include_router(sudoku_api.router)
app.include_router(checkers_api.router)
app.include_router(pacman_api.router)
app.include_router(battleship_api.router)
app.include_router(lightsout_api.router)
app.include_router(blackjack_api.router)
app.include_router(routes_api.router)
app.include_router(warehouse_api.router)

WEB = Path(__file__).resolve().parent.parent / "web"


@app.get("/api/health")
async def health():
    return {"ok": True}


def client_key(ws: WebSocket) -> str:
    # Behind Fly.io's proxy the client address arrives in a header.
    return ws.headers.get("fly-client-ip") or (ws.client.host if ws.client else "unknown")


@app.websocket("/ws/npuzzle")
async def npuzzle_ws(ws: WebSocket):
    if ORIGINS and ws.headers.get("origin") not in ORIGINS:
        await ws.close(code=1008)
        return
    await ws.accept()
    try:
        while True:
            data = await ws.receive_json()
            if isinstance(data, dict) and data.get("type") == "cancel":
                continue  # nothing running
            if not app.state.rate.allow(client_key(ws)):
                await ws.send_json({"type": "error", "message": "rate limit: try again in a few seconds"})
                continue
            try:
                req = npuzzle_api.parse_request(data)
            except ValueError as e:
                await ws.send_json({"type": "error", "message": str(e)})
                continue
            try:
                async with app.state.slots.acquire():
                    await npuzzle_api.solve_stream(ws, req)
            except Busy:
                await ws.send_json({"type": "error", "message": "server busy: try again shortly"})
    except (WebSocketDisconnect, RuntimeError):
        # RuntimeError: the client closed the socket while a search was streaming.
        pass


# Serve the frontend from the same origin. Mounted last so API routes win.
if WEB.is_dir():
    app.mount("/", StaticFiles(directory=WEB, html=True), name="web")
