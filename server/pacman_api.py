"""Pac-Man API: whole games for the animated watch mode, and replayed moves for human play.

GET  /api/pacman/meta
    Mazes, agents, feature names with their learned weights, ghost personalities, the two ghost
    policies with the chance odds, and the rule constants.

POST /api/pacman/episode  {"agent": "q" | "reflex" | "random", "maze": "lanes" | "vault", "seed": int,
                           "ghosts": "ai" | "chance"}
    Plays a whole game on the server and returns every turn: positions before and after, each
    ghost's target and planned route, the events, and for the Q-agent the Q-value and feature
    vector of every legal move. The browser animates the list; Python is the only game logic.
    "ghosts" picks the opponent: "ai" (A* routes, the default) or "chance" (fixed odds). The
    chance moves are drawn on the server from the game's seed, so the page never needs the odds.

POST /api/pacman/play  {"maze": ..., "seed": int, "actions": ["N" | "E" | "S" | "W", ...],
                        "ghosts": "ai" | "chance"}
    Human play. The game is deterministic given the seed and the moves, so the browser sends
    the whole move history and gets back the state after it, plus the legal moves next. The
    server does no per-session bookkeeping.

Limits: episodes are capped at MAX_TURNS turns, and both endpoints are rate limited per client
and share the search-slot semaphore with the other solvers.
"""

from __future__ import annotations

import asyncio
import os
from typing import Literal

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from pacman.agents import AGENTS, DESCRIPTIONS, load_weights, make_agent
from pacman.engine import (
    GHOST_POINTS,
    MAX_TURNS,
    PELLET_POINTS,
    POWER_POINTS,
    SCARED_TURNS,
    WIN_POINTS,
    Game,
    State,
    Turn,
    initial_state,
    legal_actions,
)
from pacman.episode import Episode, run_episode
from pacman.features import FEATURE_DOCS, FEATURE_NAMES
from pacman.ghosts import CHANCE_ODDS, GHOST_DOCS, GHOST_LABELS, GHOST_POLICIES, PERSONALITY_DOCS
from pacman.mazes import ACTIONS, Maze, get, names

from .limits import Busy, RateLimiter, client_key

router = APIRouter()

EPISODES_PER_MIN = int(os.environ.get("PACMAN_EPISODES_PER_MIN", "30"))
PLAYS_PER_MIN = int(os.environ.get("PACMAN_PLAYS_PER_MIN", "300"))
_episode_rate = RateLimiter(EPISODES_PER_MIN, per=60.0)
_play_rate = RateLimiter(PLAYS_PER_MIN, per=60.0)


class EpisodeRequest(BaseModel):
    agent: str = Field("q", pattern="^(q|reflex|random)$")
    maze: str = Field("lanes", max_length=20)
    seed: int = Field(0, ge=0, le=2**31 - 1)
    max_turns: int = Field(MAX_TURNS, ge=10, le=MAX_TURNS)
    ghosts: str = Field("ai", pattern="^(ai|chance)$")


class PlayRequest(BaseModel):
    maze: str = Field("lanes", max_length=20)
    seed: int = Field(0, ge=0, le=2**31 - 1)
    actions: list[Literal["N", "E", "S", "W"]] = Field(default_factory=list, max_length=MAX_TURNS)
    ghosts: str = Field("ai", pattern="^(ai|chance)$")


def _xy(maze: Maze, cell: int) -> list[int]:
    return [maze.row[cell], maze.col[cell]]


def _maze_json(maze: Maze, rows: bool = True) -> dict:
    out = {"name": maze.name, "title": maze.title, "width": maze.width, "height": maze.height}
    if rows:
        out["rows"] = list(maze.rows)
    return out


def _ghosts_json(maze: Maze, state: State) -> list[dict]:
    return [
        {"pos": _xy(maze, g.pos), "personality": g.personality, "scared": g.scared}
        for g in state.ghosts
    ]


def _state_json(state: State) -> dict:
    maze = state.maze
    return {
        "pac": _xy(maze, state.pac),
        "ghosts": _ghosts_json(maze, state),
        "points": state.points,
        "turn": state.turn,
        "status": state.status,
        "pellets": [_xy(maze, p) for p in sorted(state.pellets)],
    }


def _turn_json(turn: Turn, decision=None) -> dict:
    maze = turn.after.maze
    out = {
        "action": ACTIONS[turn.action],
        "pac": _xy(maze, turn.after.pac),
        "ghosts": _ghosts_json(maze, turn.after),
        "eaten": _xy(maze, turn.eaten) if turn.eaten is not None else None,
        "points": turn.points,
        "events": list(turn.events),
        "status": turn.after.status,
        "score": turn.after.points,
        "plans": [
            {
                "ghost": p.ghost,
                "target": _xy(maze, p.target) if p.target is not None else None,
                "path": [_xy(maze, c) for c in p.path],
            }
            for p in turn.plans
        ],
        "q": None,
        "features": None,
    }
    if decision is not None and decision.values:
        out["q"] = {ACTIONS[a]: round(v, 3) for a, v in decision.values.items()}
        out["features"] = {ACTIONS[a]: [round(x, 4) for x in f] for a, f in decision.features.items()}
    return out


def _episode_json(ep: Episode) -> dict:
    maze = ep.maze
    start = initial_state(maze)
    return {
        "agent": ep.agent,
        "description": DESCRIPTIONS.get(ep.agent, ""),
        "ghosts": ep.ghosts,
        "ghosts_label": GHOST_LABELS[ep.ghosts],
        "seed": ep.seed,
        "maze": _maze_json(maze),
        "start": {
            "pac": _xy(maze, start.pac),
            "ghosts": _ghosts_json(maze, start),
            "pellets": [_xy(maze, p) for p in sorted(start.pellets)],
        },
        "turns": [_turn_json(t, d) for t, d in zip(ep.turns, ep.decisions, strict=True)],
        "result": {"status": ep.final.status, "won": ep.won, "score": ep.score, "turns": ep.final.turn},
    }


def _check_maze(name: str) -> Maze:
    if name not in names():
        raise HTTPException(400, f"unknown maze {name!r}")
    return get(name)


def _episode(agent_name: str, maze: Maze, seed: int, max_turns: int, ghosts: str) -> dict:
    """Runs in a worker thread: the whole game is pure Python and can take a few hundred milliseconds."""
    ep = run_episode(make_agent(agent_name), maze, seed, agent_name, max_turns=max_turns, ghosts=ghosts)
    payload = _episode_json(ep)
    if agent_name == "q":
        # The learned weights, so the page can show each feature's share of every Q-value.
        payload["weights"] = list(load_weights().weights)
    return payload


def _replay(maze: Maze, seed: int, actions: list[str], ghosts: str) -> dict:
    """Runs in a worker thread: applies the move history to a fresh game and reports the new state."""
    game = Game(maze, seed=seed, ghosts=ghosts)
    last = None
    for i, letter in enumerate(actions):
        if game.state.over:
            raise HTTPException(400, "the game is already over")
        action = ACTIONS.index(letter)
        if action not in legal_actions(game.state):
            raise HTTPException(400, f"move {i + 1} ({letter}) runs into a wall")
        last = game.step(action)
    return {
        "state": _state_json(game.state),
        "legal": [ACTIONS[a] for a in legal_actions(game.state)] if not game.state.over else [],
        "last": _turn_json(last) if last is not None else None,
        "maze": _maze_json(maze),
        "ghosts": ghosts,
    }


@router.get("/api/pacman/meta")
async def meta():
    try:
        w = load_weights()
        weights = {"features": list(FEATURE_NAMES), "weights": list(w.weights), "trained": w.trained}
    except FileNotFoundError:
        weights = None
    return {
        "mazes": [_maze_json(get(n)) for n in names()],
        "agents": [{"name": a, "description": DESCRIPTIONS[a]} for a in AGENTS],
        "ghost_policies": [
            {"name": p, "label": GHOST_LABELS[p], "description": GHOST_DOCS[p]} for p in GHOST_POLICIES
        ],
        "chance_odds": CHANCE_ODDS,
        "features": [{"name": n, "description": FEATURE_DOCS[n]} for n in FEATURE_NAMES],
        "weights": weights,
        "personalities": PERSONALITY_DOCS,
        "constants": {
            "pellet_points": PELLET_POINTS,
            "power_points": POWER_POINTS,
            "ghost_points": GHOST_POINTS,
            "win_points": WIN_POINTS,
            "scared_turns": SCARED_TURNS,
            "max_turns": MAX_TURNS,
        },
    }


@router.post("/api/pacman/episode")
async def episode(req: EpisodeRequest, request: Request):
    if not _episode_rate.allow(client_key(request)):
        raise HTTPException(429, "rate limit: try again in a few seconds")
    maze = _check_maze(req.maze)
    try:
        async with request.app.state.slots.acquire():
            return await asyncio.to_thread(_episode, req.agent, maze, req.seed, req.max_turns, req.ghosts)
    except Busy:
        raise HTTPException(503, "server busy: try again shortly") from None
    except FileNotFoundError as e:  # Q-agent weights not trained yet
        raise HTTPException(503, str(e)) from None


@router.post("/api/pacman/play")
async def play(req: PlayRequest, request: Request):
    if not _play_rate.allow(client_key(request)):
        raise HTTPException(429, "rate limit: try again in a few seconds")
    maze = _check_maze(req.maze)
    try:
        async with request.app.state.slots.acquire():
            return await asyncio.to_thread(_replay, maze, req.seed, req.actions, req.ghosts)
    except Busy:
        raise HTTPException(503, "server busy: try again shortly") from None
