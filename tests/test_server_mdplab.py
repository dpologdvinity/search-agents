"""The grid MDP lab router: the presets it serves match the Python presets, and unknown keys are 404s."""

from fastapi import FastAPI
from fastapi.testclient import TestClient

from mdplab.grid import PRESETS
from server.mdplab_api import router


def client() -> TestClient:
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


def test_meta_lists_every_preset_with_its_rows():
    body = client().get("/api/mdplab/meta").json()
    assert [p["key"] for p in body["presets"]] == list(PRESETS)
    assert body["actions"] == ["up", "right", "down", "left"]
    assert body["defaults"]["gamma"] == 0.95


def test_preset_detail_and_unknown_key():
    c = client()
    got = c.get("/api/mdplab/presets/cliff").json()
    assert got["rows"] == list(PRESETS["cliff"].rows)
    assert c.get("/api/mdplab/presets/nope").status_code == 404
