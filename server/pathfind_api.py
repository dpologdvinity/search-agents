"""Pathfinding lab API: the algorithm list and the preset maps. The race itself runs in the browser.

GET /api/pathfind/meta      algorithms, labels, heuristics, map kinds, default weight
GET /api/pathfind/presets   named map recipes (maze kind, size, seed, swamp share, diagonal flag)

Nothing here runs a search. The maps are rebuilt in the page from the same seeds, and the page works
without this router: it falls back to the same preset list when the request fails.
"""

from __future__ import annotations

from fastapi import APIRouter

from pathfind.grid import HEURISTICS
from pathfind.mazes import KINDS
from pathfind.presets import PRESETS
from pathfind.search import ALGOS, DEFAULT_WEIGHT, HEURISTIC_ALGOS, LABELS

router = APIRouter()


@router.get("/api/pathfind/meta")
def meta() -> dict:
    """What the lab can run: the seven algorithms, which of them take a heuristic, and the map kinds."""
    return {
        "algorithms": [{"id": a, "label": LABELS[a], "heuristic": a in HEURISTIC_ALGOS} for a in ALGOS],
        "heuristics": list(HEURISTICS),
        "maps": list(KINDS),
        "default_weight": DEFAULT_WEIGHT,
    }


@router.get("/api/pathfind/presets")
def presets() -> dict:
    """Named map recipes. Each one rebuilds the same map in Python and in the browser."""
    return {"presets": PRESETS}
