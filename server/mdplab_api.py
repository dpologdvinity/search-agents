"""Grid MDP lab API: the presets and the shared constants the page and the command line both use.

Not mounted in server/app.py: the page runs client-side. Kept for scripts and tests.

GET /api/mdplab/meta              action names, cell kinds, default parameters, and every preset
GET /api/mdplab/presets/{key}     one preset: its rows and parameters (404 for an unknown key)

The page works without this router: its presets are in web/js/mdplab-core.js, and a parity test keeps that
list equal to mdplab.grid.PRESETS. The router only exposes the Python copy for other tools. Nothing is stored.
"""

from __future__ import annotations

from dataclasses import asdict

from fastapi import APIRouter, HTTPException

from mdplab.grid import ACTIONS, ARROWS, KINDS, PRESETS, TIE, Params

router = APIRouter()


def _preset_json(p) -> dict:
    return {"key": p.key, "label": p.label, "blurb": p.blurb, "rows": list(p.rows), "params": asdict(p.params)}


@router.get("/api/mdplab/meta")
def meta() -> dict:
    return {
        "actions": list(ACTIONS),
        "arrows": list(ARROWS),
        "kinds": list(KINDS),
        "tie": TIE,
        "defaults": asdict(Params()),
        "presets": [_preset_json(p) for p in PRESETS.values()],
    }


@router.get("/api/mdplab/presets/{key}")
def preset_detail(key: str) -> dict:
    if key not in PRESETS:
        raise HTTPException(status_code=404, detail=f"unknown preset {key!r}")
    return _preset_json(PRESETS[key])
