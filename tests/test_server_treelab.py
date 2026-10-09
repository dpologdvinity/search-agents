"""Tree lab router: meta and preset routes, mounted on a bare FastAPI app (server/app.py does not include it yet)."""

import numpy as np
from fastapi import FastAPI
from fastapi.testclient import TestClient

from server.treelab_api import router
from treelab.data import make_preset

app = FastAPI()
app.include_router(router)
client = TestClient(app)


def test_meta_lists_presets_and_limits():
    body = client.get("/api/treelab/meta").json()
    names = {p["name"]: p["classes"] for p in body["presets"]}
    assert names["blobs"] == 3 and names["xor"] == 2
    assert body["features"] == ["x", "y", "x·y", "x²+y²"]
    assert body["defaults"]["depth"] == 5 and body["limits"]["trees"] == [1, 100]


def test_preset_route_matches_python_reference():
    body = client.get("/api/treelab/presets", params={"name": "spiral", "n": 50, "seed": 4}).json()
    X, y = make_preset("spiral", 50, 4)
    assert body["classes"] == 2
    assert np.allclose(np.array(body["X"]), X, atol=1e-11)
    assert body["y"] == y.tolist()


def test_unknown_preset_is_404():
    assert client.get("/api/treelab/presets", params={"name": "moons"}).status_code == 404


def test_bad_size_is_422():
    assert client.get("/api/treelab/presets", params={"name": "xor", "n": 5}).status_code == 422
