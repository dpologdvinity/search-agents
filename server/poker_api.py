"""Poker API: play Leduc hold'em against the CFR+ bot or the chance player, and serve the training curves.

GET  /api/poker/meta              the game's rules in numbers, the committed strategy's stats, and both opponents
GET  /api/poker/training          exploitability against iterations for CFR and CFR+, and the Kuhn check
GET  /api/poker/strategy          the average strategy table: {information set: {action: probability}}
POST /api/poker/deal   {"seed": 7, "opponent": "chance"}
                                        start a hand; you are seat 0 and act first. Returns a view.
                                        opponent is "cfr" (the default) or "chance".
POST /api/poker/act    {"hand": "...", "action": "c"}
                                        your action (k, b, c, r, f). The bot then answers until it is your
                                        turn again or the hand ends: from its average strategy (cfr), or
                                        from the fixed action odds (chance, which ignores the cards).

Hands live on the server in a bounded table keyed by an id. Until the showdown a response carries only
the bot's actions, never its card or the mix it drew them from: the mix depends on the card, so it is
kept back with the hand and sent in bot_reveal once the hand is over. The bot's move is a lookup in the
committed table (poker/data), so a request costs microseconds of CPU: no search runs here. Training is
never run by the server.
"""

from __future__ import annotations

import json
import random
import secrets
from collections import OrderedDict
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Literal

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from poker import chance, strategy
from poker.leduc import ANTE, BET_SIZES, DECK, LeducState, card_name
from poker.strategy import sample

from .limits import client_key

router = APIRouter()

MAX_HANDS = 2_000  # open hands kept in memory; the oldest is dropped first
DESCRIPTION = (
    "Six cards (two each of J, Q, K), a private card each, betting in two rounds with a public card between "
    "them, and a showdown. Bets are 2 chips in round 1 and 4 in round 2, with at most two bets or raises a round."
)
RESULT_WORDS = {1: "win", -1: "lose", 0: "split"}
OPPONENT_INFO = {
    "cfr": {
        "label": "CFR+ (average strategy)",
        "algorithm": "CFR+ average strategy: a table of mixed actions per information set, sampled at each move",
        "description": "The bot plays the committed CFR+ average strategy. Its card changes its mix.",
    },
    "chance": {
        "label": "Chance (fixed odds)",
        "algorithm": "fixed action odds: no card, search, or lookahead; legal actions are weighted and sampled",
        "description": (
            f"The bot draws each action from fixed odds (fold {chance.ACTION_ODDS['fold']:.0%}, "
            f"call or check {chance.ACTION_ODDS['call_or_check']:.0%}, "
            f"bet or raise {chance.ACTION_ODDS['bet_or_raise']:.0%}, renormalised over the legal actions). "
            "It ignores its card, like a dice roll."
        ),
    },
}


@dataclass
class _Hand:
    state: LeducState
    rng: random.Random
    opponent: str = "cfr"
    decisions: list[dict] = field(default_factory=list)  # every bot decision so far, with its full mix


_hands: OrderedDict[str, _Hand] = OrderedDict()


class DealRequest(BaseModel):
    seed: int | None = Field(None, ge=0, le=2**31 - 1)
    opponent: Literal["cfr", "chance"] = "cfr"


class ActRequest(BaseModel):
    hand: str = Field(min_length=8, max_length=32)
    action: str = Field(pattern="^[kbcrf]$")


@lru_cache(maxsize=1)
def _table() -> dict:
    return strategy.load()


@lru_cache(maxsize=1)
def _training() -> dict:
    return json.loads(strategy.LEDUC_TRAINING.read_text())


def _round_of(state: LeducState) -> int:
    return 2 if "/" in state.hist else 1


def _view(hand_id: str, hand: _Hand, bot_moves: list[dict]) -> dict:
    """What the browser may see of a hand. The bot's card and mixes are withheld until the hand is over."""
    state = hand.state
    put0, put1 = state.contributions()
    shown = state.public is not None and "/" in state.hist
    terminal = state.is_terminal()
    out = {
        "hand": hand_id,
        "opponent": hand.opponent,
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
        "bot_reveal": None,
        "result": None,
    }
    if not terminal and state.player() == 0:
        key = state.infoset(0)
        out["hint"] = {"infoset": key, "probs": _table()["strategy"][key]}
    if terminal:
        payoff = state.payoff()  # seat 0 is you
        out["payoff"] = payoff
        out["bot_card"] = card_name(state.cards[1])
        out["bot_reveal"] = hand.decisions  # the whole hand: each decision with its card and mix
        out["result"] = RESULT_WORDS[(payoff > 0) - (payoff < 0)]
    return out


def _bot_turns(hand: _Hand) -> list[dict]:
    """Let the bot act while it is its turn. Returns the public part of each move: round, spot and action.

    The spot is the bot's information set with its card replaced by "?", so the browser can look up the
    mix it would play with each card in that same betting spot. The full decision (card and mix) goes to
    hand.decisions and is only returned after the showdown.
    """
    moves = []
    while not hand.state.is_terminal() and hand.state.player() == 1:
        state = hand.state
        key = state.infoset(1)
        if hand.opponent == "chance":
            # Chance reads no card: its row depends only on which actions are legal, so the same row
            # would be shown for any card in this spot. It is still recorded with the card, so the
            # reveal after showdown shows exactly what was drawn from.
            row = chance.legal_weights(state.actions())
        else:
            row = _table()["strategy"][key]
        action = sample(row, hand.rng)
        round_no = _round_of(state)
        spot = "?" + key[1:]  # the card is the first character of the key
        hand.decisions.append({"round": round_no, "spot": spot, "infoset": key, "probs": row, "action": action})
        moves.append({"round": round_no, "spot": spot, "action": action})
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
        "opponents": [
            {"name": name, **info, **({"action_odds": chance.ACTION_ODDS} if name == "chance" else {})}
            for name, info in OPPONENT_INFO.items()
        ],
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
    if not request.app.state.rate.allow(client_key(request)):
        raise HTTPException(429, "rate limit: try again in a few seconds")
    rng = random.Random(req.seed) if req.seed is not None else random.Random()
    a, b, pub = rng.sample(DECK, 3)
    hand = _Hand(LeducState.dealt(a, b, pub), rng, req.opponent)
    hand_id = secrets.token_urlsafe(12)
    _hands[hand_id] = hand
    while len(_hands) > MAX_HANDS:
        _hands.popitem(last=False)
    return _view(hand_id, hand, [])


@router.post("/api/poker/act")
async def act(req: ActRequest, request: Request):
    if not request.app.state.move_rate.allow(client_key(request)):
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
