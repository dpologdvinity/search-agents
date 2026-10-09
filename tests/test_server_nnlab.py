"""The NN lab meta route, mounted on a bare FastAPI app (server/app.py does not register it yet)."""

from fastapi import FastAPI
from fastapi.testclient import TestClient

from nnlab.config import DEFAULTS
from server.nnlab_api import router


def test_meta_lists_options_and_defaults():
    app = FastAPI()
    app.include_router(router)
    res = TestClient(app).get("/api/nnlab/meta")
    assert res.status_code == 200
    body = res.json()
    assert body["defaults"]["hidden"] == DEFAULTS["hidden"]
    assert "circles" in body["options"]["presets"]
    assert body["limits"] == {"hidden_layers": [0, 4], "neurons": [1, 8]}
