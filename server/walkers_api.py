"""EVOLVING WALKERS API: constants for the page, and the champion presets.

Not mounted in server/app.py: the page runs client-side. Kept for scripts and tests.

GET /api/walkers/meta      physics and GA constants, terrains, and the default world
GET /api/walkers/presets   the preset champions (text codes, distances, behaviour)

The page runs the physics and the genetic algorithm in a Web Worker, so this API only serves static data. It is
optional: the page reads web/data/walkers-champions.json directly when the API is not mounted. Like the other
routers it is not registered in server/app.py here; the integration notes say where it goes.
"""

from __future__ import annotations

import functools
import json
from pathlib import Path

from fastapi import APIRouter

import walkers.core as core

router = APIRouter()
PRESETS_FILE = Path(core.__file__).resolve().parent / "data" / "champions.json"


@functools.lru_cache(maxsize=1)
def _presets() -> dict:
    """The presets document, read once."""
    return json.loads(PRESETS_FILE.read_text())


@router.get("/api/walkers/meta")
def meta() -> dict:
    """Constants the page and the Python port share: time step, limits, terrains, GA defaults and default world."""
    return {
        "dt": core.DT,
        "terrains": list(core.TERRAINS),
        "world": dict(core.DEFAULT_WORLD),
        "ga": dict(core.GA_DEFAULTS),
        "limits": {
            "nodes": [core.NODE_MIN, core.NODE_MAX],
            "rest": [core.REST_MIN, core.REST_MAX],
            "k": [core.K_MIN, core.K_MAX],
            "c": [core.C_MIN, core.C_MAX],
            "amp": [0, core.AMP_MAX],
            "freq": [core.FREQ_MIN, core.FREQ_MAX],
        },
    }


@router.get("/api/walkers/presets")
def presets() -> dict:
    """The preset champions: each has a text code (see walkers.core.encode), its distance and its behaviour."""
    return _presets()
