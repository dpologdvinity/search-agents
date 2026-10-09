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

import math
import os
import threading
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException
from starlette.responses import FileResponse
from starlette.types import Scope

from . import (
    bandits_api,
    battleship_api,
    blackjack_api,
    cartpole_api,
    checkers_api,
    connect4_api,
    endgame_api,
    game2048_api,
    ghosthunt_api,
    hexgame_api,
    lightsout_api,
    localize_api,
    minesweeper_api,
    nonogram_api,
    npuzzle_api,
    pacman_api,
    pathfind_api,
    poker_api,
    queens_api,
    routes_api,
    rover_api,
    snake_api,
    sokoban_api,
    sudoku_api,
    tetris_api,
    warehouse_api,
    wordle_api,
)
from .limits import MAX_BODY_BYTES, BodyLimit, Busy, RateLimiter, SearchSlots, client_key, loads_finite

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
    # Wordle's feedback table takes about 5 s to build, so build it in a daemon thread and let startup go on.
    # A request that arrives before the build ends waits on the same lock inside get_lexicon, so nothing is
    # built twice.
    from wordle.lexicon import get_lexicon as load_wordle

    threading.Thread(target=load_wordle, daemon=True).start()
    yield


app = FastAPI(title="search-agents", lifespan=lifespan)
# Added first so it sits inside CORS: a 413 or 422 from it still carries the CORS headers the page needs to read it.
app.add_middleware(BodyLimit)
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
app.include_router(minesweeper_api.router)
app.include_router(hexgame_api.router)
app.include_router(bandits_api.router)
app.include_router(cartpole_api.router)
app.include_router(queens_api.router)
app.include_router(snake_api.router)
app.include_router(rover_api.router)
app.include_router(tetris_api.router)
app.include_router(nonogram_api.router)
app.include_router(ghosthunt_api.router)
app.include_router(localize_api.router)
app.include_router(pathfind_api.router)
app.include_router(endgame_api.router)
app.include_router(sokoban_api.router)
app.include_router(blackjack_api.router)
app.include_router(routes_api.router)
app.include_router(warehouse_api.router)
app.include_router(wordle_api.router)
app.include_router(poker_api.router)

WEB = Path(__file__).resolve().parent.parent / "web"


@app.get("/api/health")
async def health():
    return {"ok": True}


def _json_safe(value):
    """Replace non-finite floats with their repr ("nan", "inf") so the value can be written as JSON."""
    if isinstance(value, float) and not math.isfinite(value):
        return repr(value)
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


@app.exception_handler(RequestValidationError)
async def validation_error(request: Request, exc: RequestValidationError):
    # FastAPI's default handler echoes each bad input in its error. A NaN there cannot be written as JSON, so the
    # default answer was a 500. Non-finite inputs are sent as strings instead.
    return JSONResponse(status_code=422, content={"detail": jsonable_encoder(_json_safe(exc.errors()))})


@app.websocket("/ws/npuzzle")
async def npuzzle_ws(ws: WebSocket):
    if ORIGINS and ws.headers.get("origin") not in ORIGINS:
        await ws.close(code=1008)
        return
    await ws.accept()
    try:
        while True:
            # Frames are read raw: receive_json() would raise on a bad frame and drop the socket with a traceback.
            message = await ws.receive()
            if message["type"] == "websocket.disconnect":
                break
            text = message.get("text")
            if text is None:
                await ws.send_json({"type": "error", "message": "send a JSON text message"})
                continue
            if len(text) > MAX_BODY_BYTES:
                await ws.send_json({"type": "error", "message": "message is too large"})
                continue
            try:
                data = loads_finite(text)
            except ValueError as e:
                await ws.send_json({"type": "error", "message": f"bad JSON: {e}"})
                continue
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


def not_found_page() -> FileResponse:
    return FileResponse(WEB / "404.html", status_code=404)


class WebFiles(StaticFiles):
    """The frontend. An unknown GET page gets web/404.html with status 404 (StaticFiles does that itself in
    HTML mode). An unknown path under /api/ must stay a JSON 404 instead, so it is raised here, where the
    framework turns it into {"detail": "Not Found"}."""

    async def get_response(self, path: str, scope: Scope):
        if path == "api" or path.startswith("api/"):
            raise HTTPException(status_code=404)
        try:
            response = await super().get_response(path, scope)
        except HTTPException as exc:
            if exc.status_code != 404:
                raise
            # StaticFiles refuses some paths outright (one with a NUL byte, for example). They get the styled page too.
            return not_found_page()
        if path == "404.html":
            # StaticFiles serves this file by name with status 200. Asking for the 404 page directly is still a 404.
            return not_found_page()
        return response


# Serve the frontend from the same origin. Mounted last so API routes win.
if WEB.is_dir():
    app.mount("/", WebFiles(directory=WEB, html=True), name="web")
