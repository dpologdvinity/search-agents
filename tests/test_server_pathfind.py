"""Tests for the pathfinding lab router. It is mounted on a bare app here, as in the other router tests,
because server.app does not register it."""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from pathfind.mazes import make_maze
from pathfind.presets import PRESETS
from pathfind.search import ALGOS
from server import pathfind_api


@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(pathfind_api.router)
    with TestClient(app) as c:
        yield c


def test_meta_lists_all_algorithms(client):
    body = client.get("/api/pathfind/meta").json()
    assert [a["id"] for a in body["algorithms"]] == list(ALGOS)
    heur = {a["id"]: a["heuristic"] for a in body["algorithms"]}
    assert heur["astar"] and heur["wastar"] and heur["greedy"]
    assert not heur["bfs"] and not heur["ucs"]
    assert body["heuristics"] == ["manhattan", "euclidean", "octile"]
    assert "prim" in body["maps"]


def test_presets_are_buildable_recipes(client):
    body = client.get("/api/pathfind/presets").json()
    assert body["presets"] == PRESETS
    for p in body["presets"]:
        maze = make_maze(p["maze"], p["size"], p["size"], p["seed"], swamp=p["swamp"], density=p.get("density", 28))
        assert len(maze.grid) == p["size"] * p["size"]
