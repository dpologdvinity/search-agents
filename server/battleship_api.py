"""Battleship API: the agent's fleet, the player's shots at it, and the agent's odds of the player's fleet.

GET  /api/battleship/meta
POST /api/battleship/new          -> {"id": ..., "size": 10, "fleet": [{"name", "length"}, ...]}
POST /api/battleship/shot         {"id": ..., "cell": "C7"}  -> miss / hit / sunk, and the fleet once won
POST /api/battleship/agent-shot   {"shots": [...], "agent": "probability" | "chance", "seed": 0}
                                                             -> the agent's odds and its next cell

Who knows what. The agent's fleet is random and lives only on the server, in a bounded in-memory store keyed by
an unguessable id, so the player cannot read it from the page; it is revealed when the game is won. The player's
own fleet never reaches the server: the page sends the agent's shots at it and their results (with the cells of
each sunk ship) to /agent-shot, and the page works out the answers itself. The server then only computes the
agent's odds from that history, so a game is stateless apart from the agent's fleet.
"""

from __future__ import annotations

import asyncio
import random
import secrets
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Literal

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from battleship.agents import CHANCE_PROFILE, ChanceAgent, ProbabilityAgent, chance_odds
from battleship.board import COLUMNS, FLEET, FLEET_NAMES, SIZE, Fleet, Knowledge, cell_name, parse_cell, random_fleet
from battleship.probability import SAMPLES, choose_cell, ship_probabilities

from .limits import Busy, client_key

router = APIRouter()

GAME_TTL = 30 * 60  # seconds an unfinished game is kept
MAX_GAMES = 500  # oldest games are dropped first, so memory stays bounded
DESCRIPTION = (
    "Bayesian targeting: it counts every fleet placement that fits its misses and hits, and fires at the cell "
    "most likely to hold a ship. Exact counts when they are cheap, importance sampling when they are not."
)
CHANCE_DESCRIPTION = (
    "Fixed odds: no search and no reading of its hits. It fires at a random unknown cell, weighted by a fixed "
    "table: a cell's weight is profile[row] * profile[col], so the middle of the board is favoured."
)


@dataclass
class _Game:
    fleet: Fleet
    created: float
    over: bool = False


_GAMES: OrderedDict[str, _Game] = OrderedDict()


class CellIn(BaseModel):
    id: str = Field(min_length=1, max_length=64)
    cell: str = Field(min_length=2, max_length=3)


class ShipReport(BaseModel):
    cells: list[str] = Field(min_length=2, max_length=5)


class ShotReport(BaseModel):
    cell: str = Field(min_length=2, max_length=3)
    result: Literal["miss", "hit", "sunk"]
    ship: ShipReport | None = None  # required for "sunk": every cell of the ship just sunk


class ObserveIn(BaseModel):
    shots: list[ShotReport] = Field(default_factory=list, max_length=SIZE * SIZE)
    agent: Literal["probability", "chance"] = "probability"  # whose odds to compute: the Bayesian AI or chance
    seed: int = Field(default=0, ge=0, le=2**31 - 1)  # chance only: the same seed and shots give the same cell


def _store(fleet: Fleet) -> str:
    """Keep a new game and return its id. Expired games are dropped, then the oldest if still full."""
    now = time.monotonic()
    for gid in [g for g, rec in _GAMES.items() if now - rec.created > GAME_TTL]:
        del _GAMES[gid]
    while len(_GAMES) >= MAX_GAMES:
        _GAMES.popitem(last=False)
    gid = secrets.token_urlsafe(16)
    _GAMES[gid] = _Game(fleet, now)
    return gid


def _lookup(gid: str) -> _Game:
    rec = _GAMES.get(gid)
    if rec is None or time.monotonic() - rec.created > GAME_TTL:
        raise HTTPException(404, "unknown or expired game: start a new one")
    return rec


def _limit(request: Request) -> None:
    """Per-client rate limit shared with the other move endpoints (MOVES_PER_MIN)."""
    key = client_key(request)
    if not request.app.state.move_rate.allow(key):
        raise HTTPException(429, "rate limit: try again in a few seconds")


def _cell(text: str) -> int:
    try:
        return parse_cell(text)
    except ValueError as e:
        raise HTTPException(400, str(e)) from None


def _knowledge(reports: list[ShotReport]) -> Knowledge:
    """Build the agent's knowledge from its reported shots. Knowledge.observe checks each report's rules."""
    k = Knowledge()
    try:
        for r in reports:
            if (r.result == "sunk") != (r.ship is not None):
                raise ValueError("a sunk report must carry the ship's cells, and only a sunk report may")
            ship = tuple(_cell(c) for c in r.ship.cells) if r.ship else ()
            k.observe(_cell(r.cell), r.result, ship)
    except ValueError as e:
        raise HTTPException(400, str(e)) from None
    return k


def _choose(k: Knowledge) -> dict:
    """The agent's odds over the player's fleet, and the cell it fires at next. Runs in a worker thread."""
    rng = random.Random()
    belief = ship_probabilities(k, rng, samples=SAMPLES)
    unknown = k.unknown_cells()
    cell = choose_cell(belief, unknown, rng)
    return _reply(k, cell, belief.probs, belief.method, belief.count)


def _choose_chance(k: Knowledge, seed: int, shots_fired: int) -> dict:
    """Chance's next cell and its fixed odds. Its rng is seeded from the game seed and how many shots it has
    fired, so the same history always produces the same shot and a new shot gets a fresh draw."""
    cell = ChanceAgent(random.Random(seed * 1_000_003 + shots_fired)).choose(k)
    return _reply(k, cell, chance_odds(k), "fixed odds", None)


def _reply(k: Knowledge, cell: int, probs, method: str, count) -> dict:
    """The response shape shared by both agents: the choice plus the odds behind it."""
    top = sorted(k.unknown_cells(), key=lambda c: -probs[c])[:5]
    return {
        "choice": cell_name(cell),
        "choice_index": cell,
        "probability": round(probs[cell], 4),
        "method": method,
        "count": count,
        "remaining": k.remaining_lengths(),
        "probs": [round(p, 4) for p in probs],  # row-major, cell = row * 10 + col
        "top": [{"cell": cell_name(c), "probability": round(probs[c], 4)} for c in top],
    }


@router.get("/api/battleship/meta")
async def meta():
    return {
        "size": SIZE,
        "columns": COLUMNS,
        "fleet": [{"name": n, "length": length} for n, length in zip(FLEET_NAMES, FLEET)],
        "agent": {"name": ProbabilityAgent.name, "label": ProbabilityAgent.label, "description": DESCRIPTION},
        "chance": {"name": ChanceAgent.name, "label": ChanceAgent.label, "description": CHANCE_DESCRIPTION,
                   "profile": list(CHANCE_PROFILE), "rule": "weight(row, col) = profile[row] * profile[col]"},
        "samples": SAMPLES,
        "game_ttl_seconds": GAME_TTL,
    }


@router.post("/api/battleship/new")
async def new_game(request: Request):
    _limit(request)
    fleet = Fleet(random_fleet(random.Random()))
    gid = _store(fleet)
    return {"id": gid, "size": SIZE, "fleet": [{"name": n, "length": length} for n, length in zip(FLEET_NAMES, FLEET)]}


@router.post("/api/battleship/shot")
async def shot(req: CellIn, request: Request):
    _limit(request)
    rec = _lookup(req.id)
    if rec.over:
        raise HTTPException(400, "the game is over: start a new one")
    cell = _cell(req.cell)
    try:
        s = rec.fleet.fire(cell)
    except ValueError as e:
        raise HTTPException(400, str(e)) from None
    won = rec.fleet.all_sunk
    body = {"cell": cell_name(cell), "result": s.result, "afloat": rec.fleet.afloat(), "won": won, "fleet": None}
    if s.result == "sunk":
        body["ship"] = {"name": FLEET_NAMES[s.ship_index], "length": len(s.ship),
                        "cells": [cell_name(c) for c in s.ship]}
    if won:
        rec.over = True
        body["fleet"] = [{"name": n, "cells": [cell_name(c) for c in ship]}
                         for n, ship in zip(FLEET_NAMES, rec.fleet.ships)]
    return body


@router.post("/api/battleship/agent-shot")
async def agent_shot(req: ObserveIn, request: Request):
    _limit(request)
    k = _knowledge(req.shots)
    if not k.remaining_lengths():
        raise HTTPException(400, "no ships are afloat: the game is over")
    if req.agent == "chance":
        # Chance is a table lookup, so it skips the search slots that cap the Bayesian work.
        return _choose_chance(k, req.seed, len(req.shots))
    try:
        async with request.app.state.slots.acquire():
            return await asyncio.to_thread(_choose, k)
    except Busy:
        raise HTTPException(503, "server busy: try again shortly") from None
