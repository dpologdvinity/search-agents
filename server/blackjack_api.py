"""Blackjack API: the exact basic-strategy table, advice for any hand, and a Monte Carlo learning run.

GET  /api/blackjack/strategy   the exact table: every row and upcard with the best action and each action's EV
POST /api/blackjack/advice     {"cards": [1, 13], "upcard": 6, "from_split": false, "split_aces": false}
                               -> expected return per unit bet of each legal action, and the best one
POST /api/blackjack/learn      {"episodes": 40000, "seed": 1, "snapshots": 10}
                               -> Monte Carlo control's greedy table at evenly spaced points, with agreement

Cards are ranks 1..13 (1 = ace, 11..13 = face cards). The server keeps no per-player state: the client shuffles
and deals, and sends only the cards it can see. Advice depends on the player's cards and the dealer's upcard, so
the dealer's hole card is never sent. Rate limits follow checkers_api.py: `move_rate` for advice, `rate` for
learning runs, and `slots` caps how many learning runs execute at once (they are CPU-bound).
"""

from __future__ import annotations

import asyncio
from functools import lru_cache

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field, field_validator

from blackjack.learner import exact_actions, learn
from blackjack.rules import hard_and_ace, rank_value, total
from blackjack.solver import ACTION_LETTERS, best, get_solver

from .limits import Busy

router = APIRouter()

MAX_LEARN_HANDS = 100_000  # about 7 seconds of CPU on a laptop; the server runs at most SEARCH_SLOTS at once
RULES = [
    "Infinite-deck model: every card is an independent draw, so the table is exact for that model.",
    "The dealer stands on all 17s (S17). The dealer checks for blackjack when showing an ace or a ten.",
    "Blackjack pays 3:2. Double on any two cards. One split (two hands); split aces take one card each.",
    "Doubling after a split is allowed. A split hand that makes 21 pays 1:1.",
]


class AdviceRequest(BaseModel):
    cards: list[int] = Field(min_length=2, max_length=11)
    upcard: int = Field(ge=1, le=13)
    from_split: bool = False
    split_aces: bool = False

    @field_validator("cards")
    @classmethod
    def ranks(cls, v):
        if any(not 1 <= r <= 13 for r in v):
            raise ValueError("cards are ranks 1 (ace) to 13 (king)")
        return v


class LearnRequest(BaseModel):
    episodes: int = Field(40_000, ge=1_000, le=MAX_LEARN_HANDS)
    seed: int = Field(1, ge=0, le=2**31 - 1)
    snapshots: int = Field(10, ge=2, le=40)


@lru_cache(maxsize=1)
def strategy_payload() -> dict:
    """The exact table as JSON. Solved once per process; the solve takes milliseconds."""
    table = get_solver().strategy()
    rows = []
    for row in table["rows"]:
        cells = [{"upcard": c["upcard"], "action": c["action"], "ev": round(c["ev"], 4),
                  "evs": {k: round(v, 4) for k, v in c["evs"].items()}} for c in row["cells"]]
        rows.append({"kind": row["kind"], "total": row["total"], "label": row["label"], "cells": cells})
    return {
        "upcards": table["upcards"],
        "rows": rows,
        "house_edge": round(table["house_edge"], 5),
        "letters": ACTION_LETTERS,
        "rules": RULES,
        "exact_actions": exact_actions(),  # the learner's target, one letter per cell in row order
    }


def _key(request: Request) -> str:
    return request.headers.get("fly-client-ip") or (request.client.host if request.client else "unknown")


@router.get("/api/blackjack/strategy")
async def strategy():
    return strategy_payload()


@router.post("/api/blackjack/advice")
async def advice(req: AdviceRequest, request: Request):
    if not request.app.state.move_rate.allow(_key(request)):
        raise HTTPException(429, "rate limit: try again in a few seconds")
    if req.split_aces and not req.from_split:
        raise HTTPException(400, "split aces are only dealt to a hand from a split")
    values = [rank_value(r) for r in req.cards]
    hard, ace = hard_and_ace(values)
    if hard > 21:
        raise HTTPException(400, "the hand is over 21: no decision is left")
    evs = get_solver().hand_values(tuple(values), rank_value(req.upcard), from_split=req.from_split,
                                   split_aces=req.split_aces)
    natural = len(values) == 2 and sorted(values) == [1, 10] and not req.from_split
    return {
        "total": total(hard, ace),
        "soft": ace and total(hard, ace) != hard,
        "blackjack": natural,
        "best": best(evs),
        "actions": {k: round(v, 4) for k, v in evs.items()},
    }


@router.post("/api/blackjack/learn")
async def learn_route(req: LearnRequest, request: Request):
    app = request.app
    if not app.state.rate.allow(_key(request)):
        raise HTTPException(429, "rate limit: try again in a few seconds")
    try:
        async with app.state.slots.acquire():
            snaps = await asyncio.to_thread(learn, req.episodes, req.seed, req.snapshots)
    except Busy:
        raise HTTPException(503, "server busy: try again shortly") from None
    return {
        "episodes": req.episodes,
        "seed": req.seed,
        "exact": exact_actions(),
        "snapshots": [{"episodes": s.episodes, "agreement": round(s.agreement, 4), "decisive": s.decisive,
                       "loss": round(s.loss, 5), "actions": s.actions} for s in snaps],
    }
