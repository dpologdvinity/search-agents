"""The optional cluster lab router, mounted on its own app (it is not registered in server/app.py)."""

import numpy as np
from fastapi import FastAPI
from fastapi.testclient import TestClient

from clusters.data import make_dataset
from server.clusters_api import router

client = TestClient(FastAPI())
client.app.include_router(router)


def test_meta_lists_algorithms_presets_and_defaults():
    body = client.get("/api/clusters/meta").json()
    assert body["algorithms"] == ["kmeans", "dbscan", "gmm"]
    assert "blobs" in body["presets"] and "smiley" in body["presets"]
    assert body["defaults"]["k"] == 4 and body["defaults"]["min_pts"] == 5


def test_preset_points_match_the_python_reference():
    body = client.get("/api/clusters/presets", params={"name": "moons", "seed": 4, "n": 90}).json()
    X, truth = make_dataset("moons", 4, 90)
    assert np.allclose(np.array(body["points"]), X, atol=1e-11)
    assert body["truth"] == truth.tolist()


def test_preset_rejects_unknown_names_and_bad_sizes():
    assert client.get("/api/clusters/presets", params={"name": "spiral"}).status_code == 422
    assert client.get("/api/clusters/presets", params={"n": 5000}).status_code == 422
    assert client.get("/api/clusters/presets", params={"seed": -1}).status_code == 422
