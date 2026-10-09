"""Ghost Hunt API: metadata and seeded mazes match the Python generator; bad sizes are rejected."""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from ghosthunt.maze import generate
from server import ghosthunt_api
from server.app import app as real_app


@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(ghosthunt_api.router)
    with TestClient(app) as c:
        yield c


def test_meta_lists_the_models_filters_and_defaults(client):
    body = client.get("/api/ghosthunt/meta").json()
    assert body["sizes"] == [11, 15, 21]
    assert body["models"] == ["random", "lurker", "patrol"]
    assert body["filters"] == ["exact", "particles"]
    assert set(body["noise"]) == {"low", "med", "high"}
    assert body["defaults"]["size"] == 15 and body["defaults"]["ghosts"] == 2


def test_maze_is_the_generators_maze(client):
    body = client.get("/api/ghosthunt/maze", params={"size": 21, "seed": 4}).json()
    assert body["walls"] == list(generate(21, 4))
    assert body["n"] == 21 and body["start"] == 22


def test_maze_rejects_sizes_the_game_does_not_offer(client):
    assert client.get("/api/ghosthunt/maze", params={"size": 13}).status_code == 400


def test_meta_is_served_by_the_real_app():
    """The router is mounted in server/app.py, so the endpoints answer on the server the page is served from."""
    with TestClient(real_app) as c:
        assert c.get("/api/ghosthunt/meta").status_code == 200
        assert c.get("/api/ghosthunt/maze", params={"size": 11, "seed": 7}).status_code == 200
