"""Lost Robot API: the meta and map endpoints serve the same floors the Python package builds."""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from localize.world import make_floor
from server import localize_api
from server.app import app as real_app


@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(localize_api.router)
    with TestClient(app) as c:
        yield c


def test_meta_lists_layouts_and_defaults(client):
    body = client.get("/api/localize/meta").json()
    assert {lay["key"] for lay in body["layouts"]} == {"halls", "vault"}
    assert body["particles"] == [100, 500, 2000]
    assert body["defaults"]["particles"] == 2000 and body["defaults"]["rays"] == 8


def test_map_is_the_packages_floor(client):
    body = client.get("/api/localize/map", params={"layout": "halls", "seed": 3}).json()
    f = make_floor("halls", 3)
    assert body["walls"] == [int(v) for v in f.walls.ravel()]
    assert body["pillars"] == [list(p) for p in f.pillars]
    assert body["w"] == f.w and body["h"] == f.h


def test_unknown_layout_is_rejected(client):
    assert client.get("/api/localize/map", params={"layout": "nope"}).status_code in (400, 422)


def test_meta_is_served_by_the_real_app():
    """The router is mounted in server/app.py, so the endpoints answer on the server the page is served from."""
    with TestClient(real_app) as c:
        assert c.get("/api/localize/meta").status_code == 200
        assert c.get("/api/localize/map", params={"layout": "halls", "seed": 1}).status_code == 200
