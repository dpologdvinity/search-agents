"""Cluster lab API: the preset list and seeded preset points, for tools and tests.

Not mounted in server/app.py: the page runs client-side. Kept for scripts and tests.

GET /api/clusters/meta                          algorithms, presets and the page's default parameters
GET /api/clusters/presets?name=&seed=&n=        one preset: points in the unit square and their true component

The page computes everything in the browser (web/js/clusters-core.js), so nothing here is needed to run it. The
points come from the same mulberry32 stream as the page, so this endpoint and the browser agree exactly. Nothing
is stored. The router is not registered in server/app.py.
"""

from typing import Literal

from fastapi import APIRouter, Query

from clusters.data import PRESETS, make_dataset

router = APIRouter(prefix="/api/clusters")

SEED_MAX = 2**31 - 1
N_MIN = 2
N_MAX = 1200  # the page caps painting at the same count


@router.get("/meta")
def meta() -> dict:
    """What the page offers: the algorithms, the presets, and the defaults the controls start from."""
    return {
        "algorithms": ["kmeans", "dbscan", "gmm"],
        "presets": list(PRESETS),
        "defaults": {
            "k": 4,
            "init": "kmeans++",
            "eps": 0.08,
            "min_pts": 5,
            "n": 300,
            "seed": 2,
            "kmeans_max_iter": 100,
            "em_tol": 1e-6,
            "em_max_iter": 200,
            "em_reg": 1e-6,
        },
        "max_points": N_MAX,
    }


@router.get("/presets")
def preset(
    name: Literal["blobs", "aniso", "rings", "moons", "uniform", "smiley"] = "blobs",
    seed: int = Query(0, ge=0, le=SEED_MAX),
    n: int = Query(300, ge=N_MIN, le=N_MAX),
) -> dict:
    """Seeded points for one preset, with the component each point was drawn from."""
    X, truth = make_dataset(name, seed, n)
    return {
        "name": name,
        "seed": seed,
        "n": n,
        "points": X.round(12).tolist(),
        "truth": truth.tolist(),
    }
