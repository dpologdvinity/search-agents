"""Lost Robot API: the floor plans and their seeded pillars, for the page and for the parity tests.

Mounted in server/app.py. The page runs the game client-side, so these endpoints are not on the game's hot path: they
serve the metadata and the floors that scripts and the parity tests compare against.

GET /api/localize/meta                              layouts, slider ranges and defaults
GET /api/localize/map?layout=halls&seed=1           the floor as the Python package builds it

The page runs both filters in JavaScript (web/js/localize-core.js). The map endpoint lets the tests check the
JavaScript floor against the Python one.
Walls come back as a flat list in row-major order, 1 for a wall and 0 for free, like the rest of the project.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from localize.world import CELL_M, LAYOUTS, make_floor

router = APIRouter()

PARTICLE_CHOICES = [100, 500, 2000]
RAY_RANGE = [4, 16]
SIGMA_RANGE = [0.05, 1.5]
SCALE_RANGE = [0.0, 3.0]
DESCRIPTION = (
    "A robot somewhere on a known floor plan does not know where it is. A histogram filter keeps a probability "
    "for every position and heading; a particle filter keeps a few thousand guesses. Both start from total "
    "ignorance and narrow down with odometry and noisy range sensors."
)


@router.get("/api/localize/meta")
async def meta():
    return {
        "layouts": [{"key": k, "title": v["title"], "w": v["w"], "h": v["h"]} for k, v in LAYOUTS.items()],
        "particles": PARTICLE_CHOICES,
        "rays": RAY_RANGE,
        "sigma": SIGMA_RANGE,
        "scale": SCALE_RANGE,
        "cell_m": CELL_M,
        "defaults": {"layout": "halls", "particles": 2000, "rays": 8, "sigma": 0.3, "scale": 1.0, "pdrop": 0.05},
        "description": DESCRIPTION,
    }


@router.get("/api/localize/map")
async def floor_map(layout: str = Query("halls", pattern="^[a-z]+$"), seed: int = Query(1, ge=0, le=2**32 - 1)):
    if layout not in LAYOUTS:
        raise HTTPException(400, f"unknown layout {layout!r}")
    floor = make_floor(layout, seed)
    walls = [int(bool(v)) for v in floor.walls.ravel()]
    return {
        "layout": layout,
        "title": floor.title,
        "seed": seed,
        "w": floor.w,
        "h": floor.h,
        "walls": walls,
        "pillars": [list(p) for p in floor.pillars],
    }
