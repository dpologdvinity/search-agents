"""The optimizer race router: its JSON describes the same landscapes and defaults as the Python package."""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from optlab import optimizers, surfaces
from optlab.race import BLOW, DEFAULT_LR, GTOL
from server import optlab_api


@pytest.fixture
def client():
    # The router is not registered in server.app, so the test mounts it on its own app.
    app = FastAPI()
    app.include_router(optlab_api.router)
    with TestClient(app) as c:
        yield c


def test_meta_lists_every_surface_and_optimizer(client):
    body = client.get("/api/optlab/meta").json()
    assert [s["name"] for s in body["surfaces"]] == list(surfaces.SURFACES)
    assert [o["name"] for o in body["optimizers"]] == list(optimizers.NAMES)
    rosen = next(s for s in body["surfaces"] if s["name"] == "rosenbrock")
    assert rosen["start"] == [-1.5, 2.0] and rosen["minima"] == [[1.0, 1.0]]


def test_defaults_match_the_python_reference(client):
    body = client.get("/api/optlab/defaults").json()
    assert body["lr"] == DEFAULT_LR
    assert body["gtol"] == GTOL and body["blow"] == BLOW
    assert body["hyper"]["momentum"] == 0.9 and body["hyper"]["beta2"] == 0.999
