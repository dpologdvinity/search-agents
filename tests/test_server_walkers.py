"""The optional EVOLVING WALKERS router serves its constants and the presets as JSON.

The router is mounted on a fresh FastAPI app here, because server/app.py does not include it.
"""

import json
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

from server.walkers_api import router

PRESETS = Path(__file__).resolve().parent.parent / "walkers" / "data" / "champions.json"


def client():
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


def test_meta_serves_the_constants():
    body = client().get("/api/walkers/meta").json()
    assert body["dt"] == 1 / 240
    assert body["terrains"] == ["flat", "hills", "steps", "gaps"]
    assert body["ga"]["popSize"] == 80
    assert body["limits"]["nodes"] == [2, 12]


def test_presets_serves_the_presets_file():
    res = client().get("/api/walkers/presets")
    assert res.status_code == 200
    assert res.json() == json.loads(PRESETS.read_text())
