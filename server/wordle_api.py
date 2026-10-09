"""Wordle API: a game against a hidden answer, and the solver's hints for it.

GET  /api/wordle/meta                          word list sizes, strategies, the rules
POST /api/wordle/new                           start a game: {game_id, ...}
POST /api/wordle/guess  {game_id, word}        score a guess: feedback, bits gained vs expected, candidates left
GET  /api/wordle/hint?game_id=..&strategy=..   the solver's best guesses from the candidates still possible
                                               (omit game_id for the opening, before any guess)

The answer stays on the server until the game is over, so the page cannot read it from the network. Games live
in a bounded in-memory table (the oldest is dropped), so a long-running server does not grow without limit.
Each game keeps only its candidate indices (at most 3,568 small integers), so the memory per game is small.

The word list, feedback table and strategies live in wordle; this file only validates input, applies the abuse
guards, and shapes the JSON. Scoring the whole guess list is the expensive part, so /hint takes a search slot
and runs in a worker thread, like the other solvers. /new and /guess are cheap and only use the move limiter.
"""

from __future__ import annotations

import asyncio
import math
import random
import secrets
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel, Field

from wordle import feedback, solver
from wordle.feedback import ALL_GREEN, WORD_LENGTH
from wordle.lexicon import Lexicon, get_lexicon

from .limits import Busy, client_key

router = APIRouter()

MAX_GUESSES = 6
MAX_GAMES = 500  # bounded table of live games; the oldest is dropped when it is full
LIST_UP_TO = 200  # the hint returns the remaining words by name only when there are at most this many

DESCRIPTION = (
    "A hidden five-letter answer. Each guess returns a feedback pattern: green (right letter, right place), "
    "yellow (right letter, wrong place) or gray. The solver scores every allowed guess by the entropy of its "
    "feedback over the candidates still possible, and plays the one that gives the most bits."
)

Strategy = Literal["entropy", "minimax", "random"]


@dataclass
class Game:
    secret: int  # answer index; never sent until the game is over
    cand: np.ndarray  # answer indices still consistent with the feedback so far
    guesses: list = field(default_factory=list)  # (guess index, pattern) pairs, in order
    over: bool = False


_GAMES: OrderedDict[str, Game] = OrderedDict()


class GuessRequest(BaseModel):
    game_id: str = Field(max_length=64)
    word: str = Field(min_length=WORD_LENGTH, max_length=WORD_LENGTH)


async def _lexicon() -> Lexicon:
    # The first call builds the feedback table (a few seconds), so it runs off the event loop.
    return await asyncio.to_thread(get_lexicon)


def _hint_json(lex: Lexicon, cand: np.ndarray, strategy: str, top: int) -> dict:
    """The solver's ranking for a candidate set, in the shape the page draws as a bar chart."""
    options = solver.rank(lex, cand, strategy, top=top)
    words = [lex.answers[i] for i in cand] if cand.size <= LIST_UP_TO else None
    return {"remaining": int(cand.size), "strategy": strategy, "options": options, "words": words}


@router.get("/api/wordle/meta")
async def meta():
    lex = await _lexicon()
    return {
        "word_length": WORD_LENGTH,
        "max_guesses": MAX_GUESSES,
        "answers": len(lex.answers),
        "guesses": len(lex.guesses),
        "strategies": list(solver.STRATEGIES),
        "description": DESCRIPTION,
    }


@router.post("/api/wordle/new")
async def new_game(request: Request):
    if not request.app.state.move_rate.allow(client_key(request)):
        raise HTTPException(429, "rate limit: try again in a few seconds")
    lex = await _lexicon()
    game_id = secrets.token_urlsafe(9)
    _GAMES[game_id] = Game(secret=random.randrange(len(lex.answers)), cand=np.arange(len(lex.answers), dtype=np.intp))
    while len(_GAMES) > MAX_GAMES:
        _GAMES.popitem(last=False)
    return {"game_id": game_id, "word_length": WORD_LENGTH, "max_guesses": MAX_GUESSES,
            "answers": len(lex.answers), "guesses": len(lex.guesses)}


@router.post("/api/wordle/guess")
async def guess(req: GuessRequest, request: Request):
    if not request.app.state.move_rate.allow(client_key(request)):
        raise HTTPException(429, "rate limit: try again in a few seconds")
    # The await comes first: it yields to other requests, so the game is read and changed only after it.
    # Checking "over" before an await let concurrent guesses pass the check and push past the six allowed.
    lex = await _lexicon()
    game = _GAMES.get(req.game_id)
    if game is None:
        raise HTTPException(404, "unknown game: start a new one")
    if game.over:
        raise HTTPException(409, "this game is over: start a new one")
    word = req.word.lower()
    if not word.isascii() or word not in lex.guess_index:
        raise HTTPException(400, "not in the allowed word list")

    g = lex.guess_index[word]
    pattern = int(lex.table[g, game.secret])
    before = game.cand
    expected = solver.guess_bits(lex, before, g)  # what this guess was worth, before the feedback
    after = solver.narrow(lex, before, g, pattern)
    game.cand = after
    game.guesses.append((g, pattern))
    solved = pattern == ALL_GREEN
    game.over = solved or len(game.guesses) >= MAX_GUESSES
    # Bits gained = log2(candidates before / after). The secret is always a candidate, so `after` is never
    # empty here; the guard only protects the division.
    gained = math.log2(before.size / after.size) if after.size else None
    return {
        "word": word,
        "feedback": feedback.to_text(pattern),
        "tiles": list(feedback.tiles(pattern)),
        "guess_no": len(game.guesses),
        "solved": solved,
        "over": game.over,
        "remaining": int(after.size),
        "bits_gained": round(gained, 3) if gained is not None else None,
        "expected_bits": round(expected, 3),
        "answer": lex.answers[game.secret] if game.over else None,
    }


@router.get("/api/wordle/hint")
async def hint(request: Request, game_id: str | None = Query(None, max_length=64),
               strategy: Strategy = "entropy", top: int = Query(8, ge=1, le=20)):
    if not request.app.state.rate.allow(client_key(request)):
        raise HTTPException(429, "rate limit: try again in a few seconds")
    lex = await _lexicon()
    if game_id is None:
        cand = np.arange(len(lex.answers), dtype=np.intp)
    else:
        game = _GAMES.get(game_id)
        if game is None:
            raise HTTPException(404, "unknown game: start a new one")
        cand = game.cand
    try:
        async with request.app.state.slots.acquire():
            return await asyncio.to_thread(_hint_json, lex, cand, strategy, top)
    except Busy:
        raise HTTPException(503, "server busy: try again shortly") from None
