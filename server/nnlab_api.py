"""NN lab API: the lab's defaults and option lists, for tools that want them without reading the page.

Not mounted in server/app.py: the page runs client-side. Kept for scripts and tests.

GET /api/nnlab/meta    presets, features, activations, optimisers, the defaults and the layer limits

The page itself does not call this: it carries the same defaults in web/js/nnlab-core.js, and training runs
in the browser. The route is optional, and it is not registered in server/app.py (see the integrator notes).
"""

from __future__ import annotations

from fastapi import APIRouter

from nnlab.config import DEFAULTS, MAX_HIDDEN_LAYERS, MAX_NEURONS, OPTIONS

router = APIRouter()


@router.get("/api/nnlab/meta")
def meta() -> dict:
    return {
        "options": OPTIONS,
        "defaults": DEFAULTS,
        "limits": {"hidden_layers": [0, MAX_HIDDEN_LAYERS], "neurons": [1, MAX_NEURONS]},
        "rng": "mulberry32",
    }
