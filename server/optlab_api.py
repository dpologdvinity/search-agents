"""Optimizer race API: the landscapes, optimizers and default hyperparameters, as JSON.

Not mounted in server/app.py: the page runs client-side. Kept for scripts and tests.

GET /api/optlab/meta       every built-in surface (domain, start, known minima) and optimizer (label, default lr)
GET /api/optlab/defaults   the shared hyperparameters and the convergence and divergence thresholds

The lab runs entirely in the browser (web/js/optlab-core.js), so these routes only describe what the page
already ships. They give the Python reference a machine-readable form for other tools. The router is not
registered in server/app.py; include it there if the lab should serve these routes.
"""

from __future__ import annotations

from dataclasses import asdict

from fastapi import APIRouter

from optlab import optimizers, surfaces
from optlab.race import BLOW, DEFAULT_HYPER, DEFAULT_LR, GTOL

router = APIRouter()


@router.get("/api/optlab/meta")
def meta() -> dict:
    return {
        "surfaces": [
            {
                "name": s.name,
                "label": s.label,
                "domain": list(s.domain),
                "start": list(s.start),
                "fmin": s.fmin,
                "minima": [list(m) for m in s.minima],
                "note": s.note,
            }
            for s in surfaces.SURFACES.values()
        ],
        "optimizers": [{"name": n, "label": optimizers.LABELS[n], "lr": DEFAULT_LR[n]} for n in optimizers.NAMES],
    }


@router.get("/api/optlab/defaults")
def defaults() -> dict:
    return {"lr": dict(DEFAULT_LR), "hyper": asdict(DEFAULT_HYPER), "gtol": GTOL, "blow": BLOW}
