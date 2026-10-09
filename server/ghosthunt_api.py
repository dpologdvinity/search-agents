"""Ghost Hunt API: game metadata and seeded mazes for the page.

Mounted in server/app.py. The page runs the game client-side, so these endpoints are not on the game's hot path: they
serve the metadata and the seeded mazes that scripts and the parity tests compare against.

GET /api/ghosthunt/meta                       sizes, motion models, noise levels, filters and the defaults
GET /api/ghosthunt/maze?size=15&seed=7        the same maze the CLI and the page draw for these arguments

The page runs the inference in JavaScript (web/js/ghosthunt-core.js); the maze endpoint lets the tests check that
the JavaScript generator matches the Python one. Mazes are flat row-major lists: 1 is a wall, 0 is open, as in the
rest of the project.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from ghosthunt import agent, sonar
from ghosthunt.game import FILTERS, MIN_START_DISTANCE, TURN_CAP
from ghosthunt.maze import SIZES, generate
from ghosthunt.motion import MODELS

router = APIRouter()

DESCRIPTION = (
    "Invisible ghosts drift through a neon maze. Each turn the player hears noisy sonar readings of the distance "
    "to every ghost. A hidden Markov model tracks the ghosts: the exact forward algorithm, a particle filter, "
    "and Viterbi for the most likely path after a bust."
)


@router.get("/api/ghosthunt/meta")
async def meta():
    return {
        "sizes": list(SIZES),
        "models": list(MODELS),
        "noise": sonar.NOISE_LEVELS,
        "filters": list(FILTERS),
        "particles": [10, 30, 100, 300, 1000],
        "defaults": {"size": 15, "ghosts": 2, "model": "random", "noise": "med", "filter": "exact",
                     "particles": 200, "seed": 1, "bust_threshold": agent.BUST_THRESHOLD, "max_turns": TURN_CAP,
                     "min_start_distance": MIN_START_DISTANCE},
        "description": DESCRIPTION,
    }


@router.get("/api/ghosthunt/maze")
async def seeded_maze(size: int = Query(15), seed: int = Query(1, ge=0, le=2**32 - 1)):
    if size not in SIZES:
        raise HTTPException(400, f"size must be one of {list(SIZES)}")
    walls = generate(size, seed)
    return {"n": size, "seed": seed, "start": (size + 1), "walls": list(walls)}
