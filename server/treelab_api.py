"""Tree lab API: the preset list and the Python reference's points for a preset.

Not mounted in server/app.py: the page runs client-side. Kept for scripts and tests.

GET /api/treelab/meta                       preset names with class counts, feature names, defaults and limits
GET /api/treelab/presets?name=&n=&seed=     the points and labels that treelab.data.make_preset draws

The page runs entirely in the browser (web/js/treelab-core.js reproduces these presets and every tree and
forest). These routes let the Python side be inspected or diffed against the page. Nothing is stored.
This router is optional: it is not registered in server/app.py yet (see tmp/INTEGRATION_treelab.md).
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from treelab.data import FEATURE_NAMES_WEB, PRESETS, make_preset
from treelab.tree import CRITERIA

router = APIRouter()

N_MIN, N_MAX = 10, 600
SEED_MAX = 2**31 - 1


@router.get("/api/treelab/meta")
def meta() -> dict:
    return {
        "presets": [{"name": name, "classes": k} for name, k in PRESETS.items()],
        "features": list(FEATURE_NAMES_WEB),
        "criteria": list(CRITERIA),
        "defaults": {"criterion": "gini", "depth": 5, "min_leaf": 3, "test": 0.3, "trees": 25, "n": 240,
                     "extras": True},
        "limits": {"n": [N_MIN, N_MAX], "seed": [0, SEED_MAX], "depth": [1, 12], "trees": [1, 100]},
    }


@router.get("/api/treelab/presets")
def preset(name: str = Query(..., pattern="^[a-z]+$"), n: int = Query(240, ge=N_MIN, le=N_MAX),
           seed: int = Query(1, ge=0, le=SEED_MAX)) -> dict:
    if name not in PRESETS:
        raise HTTPException(status_code=404, detail=f"unknown preset {name!r}")
    X, y = make_preset(name, n, seed)
    return {"name": name, "classes": PRESETS[name], "X": X.round(12).tolist(), "y": y.tolist()}
