"""Poker API: play Leduc hold'em against the CFR+ bot, and serve the training curves.

GET  /api/poker/meta              the game's rules in numbers, and the committed strategy's stats
GET  /api/poker/training          exploitability against iterations for CFR and CFR+, and the Kuhn check
GET  /api/poker/strategy          the average strategy table: {information set: {action: probability}}
POST /api/poker/deal   {"seed": 7}      start a hand; you are seat 0 and act first. Returns a view.
POST /api/poker/act    {"hand": "...", "action": "c"}
                                        your action (k, b, c, r, f). The bot then answers from its
                                        average strategy until it is your turn again or the hand ends.

Hands live on the server in a bounded table keyed by an id, so the bot's card never reaches the
browser until the showdown. The bot's move is a lookup in the committed table (poker/data), so a
request costs microseconds of CPU: no search runs here. Training is never run by the server.
"""

from __future__ import annotations

import json
import random
import secrets
from collections import OrderedDict
from dataclasses import dataclass
from functools import lru_cache

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from poker import strategy
from poker.leduc import ANTE, BET_SIZES, DECK, LeducState, card_name
from poker.strategy import sample

router = APIRouter()

MAX_HANDS = 2_000  # open hands kept in memory; the oldest is dropped first
DESCRIPTION = (
    "Six cards (two each of J, Q, K), a private card each, betting in two rounds with a public card between "
    "them, and a showdown. Bets are 2 chips in round 1 and 4 in round 2, with at most two bets or raises a round."
)
RESULT_WORDS = {1: "win", -1: "lose", 0: "split"}


@dataclass
class _Hand:
    state: LeducState
    rng: random.Random


_hands: OrderedDict[str, _Hand] = OrderedDict()


class DealRequest(BaseModel):
    seed: int | None = Field(None, ge=0, le=2**31 - 1)


class ActRequest(BaseModel):
    hand: str = Field(min_length=8, max_length=32)
    action: str = Field(pattern="^[kbcrf]$")


@lru_cache(maxsize=1)
def _table() -> dict:
    return strategy.load()


@lru_cache(maxsize=1)
def _training() -> dict:
    return json.loads(strategy.LEDUC_TRAINING.read_text())


def _client_key(request: Request) -> str:
    # Behind Fly.io's proxy the client address arrives in a header.
    return request.headers.get("fly-client-ip") or (request.client.host if request.client else "unknown")


def _round_of(state: LeducState) -> int:
    return 2 if "/" in state.hist else 1


def _view(hand_id: str, hand: _Hand, bot_moves: list[dict]) -> dict:
    """What the browser may see of a hand. The bot's card is withheld until the hand is over."""
    state = hand.state
    put0, put1 = state.contributions()
    shown = state.public is not None and "/" in state.hist
    terminal = state.is_terminal()
    out = {
        "hand": hand_id,
        "card": card_name(state.cards[0]),
        "public": card_name(state.public) if shown else None,
        "history": state.hist.replace("/", ""),
        "round": _round_of(state),
        "pot": put0 + put1,
        "you_put": put0,
        "bot_put": put1,
        "to_act": None if terminal else state.player(),
        "legal": [] if terminal or state.player() != 0 else list(state.actions()),
        "hint": None,
        "bot_moves": bot_moves,
        "terminal": terminal,
        "payoff": None,
        "bot_card": None,
        "result": None,
    }
    if not terminal and state.player() == 0:
        key = state.infoset(0)
        out["hint"] = {"infoset": key, "probs": _table()["strategy"][key]}
    if terminal:
        payoff = state.payoff()  # seat 0 is you
        out["payoff"] = payoff
        out["bot_card"] = card_name(state.cards[1])
        out["result"] = RESULT_WORDS[(payoff > 0) - (payoff < 0)]
    return out


def _bot_turns(hand: _Hand) -> list[dict]:
    """Let the bot act while it is its turn. Each move records the mix it was drawn from, for the explainer."""
    moves = []
    while not hand.state.is_terminal() and hand.state.player() == 1:
        state = hand.state
        key = state.infoset(1)
        row = _table()["strategy"][key]
        action = sample(row, hand.rng)
        moves.append({"round": _round_of(state), "infoset": key, "probs": row, "action": action})
        hand.state = state.apply(action)
    return moves


@router.get("/api/poker/meta")
async def meta():
    data = _table()
    return {
        "description": DESCRIPTION,
        "deck": [card_name(c) for c in DECK],
        "ante": ANTE,
        "bet_sizes": list(BET_SIZES),
        "big_blind": BET_SIZES[0],
        "algorithm": data["algorithm"],
        "iterations": data["iterations"],
        "infosets": data["infosets"],
        "value_seat0": data["value_seat0"],
        "exploitability": data["exploitability"],
        "kuhn_value": -1 / 18,
    }


@router.get("/api/poker/training")
async def training():
    return _training()


@router.get("/api/poker/strategy")
async def strategy_table():
    """The whole average strategy, so the page can compare the bot's mix across cards (about 10 KB)."""
    return _table()["strategy"]


@router.post("/api/poker/deal")
async def deal(req: DealRequest, request: Request):
    if not request.app.state.rate.allow(_client_key(request)):
        raise HTTPException(429, "rate limit: try again in a few seconds")
    rng = random.Random(req.seed) if req.seed is not None else random.Random()
    a, b, pub = rng.sample(DECK, 3)
    hand = _Hand(LeducState.dealt(a, b, pub), rng)
    hand_id = secrets.token_urlsafe(12)
    _hands[hand_id] = hand
    while len(_hands) > MAX_HANDS:
        _hands.popitem(last=False)
    return _view(hand_id, hand, [])


@router.post("/api/poker/act")
async def act(req: ActRequest, request: Request):
    if not request.app.state.move_rate.allow(_client_key(request)):
        raise HTTPException(429, "rate limit: try again in a few seconds")
    hand = _hands.get(req.hand)
    if hand is None:
        raise HTTPException(404, "unknown or expired hand: deal a new one")
    state = hand.state
    if state.is_terminal():
        raise HTTPException(409, "this hand is over: deal a new one")
    if state.player() != 0:
        raise HTTPException(409, "it is the bot's turn")
    if req.action not in state.actions():
        raise HTTPException(400, f"{req.action!r} is not legal here: {', '.join(state.actions())}")
    hand.state = state.apply(req.action)
    moves = _bot_turns(hand)
    if hand.state.is_terminal():
        _hands.pop(req.hand, None)  # settled: the id cannot be replayed
    return _view(req.hand, hand, moves)
