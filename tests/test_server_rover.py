"""Rover API: seeded maps match the CLI's generator, and /plan agrees with a breadth-first reference."""

from collections import deque

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from rover.world import generate, neighbours
from server import rover_api
from server.limits import RateLimiter, SearchSlots


def make_app(rate: int = 60) -> FastAPI:
    """A minimal app with just this router, set up the way server.app's lifespan sets it up."""
    app = FastAPI()
    app.include_router(rover_api.router)
    app.state.move_rate = RateLimiter(rate, per=60.0)
    app.state.slots = SearchSlots(2, wait_seconds=5.0)
    return app


@pytest.fixture
def client():
    with TestClient(make_app()) as c:
        yield c


def bfs(n, walls, start, goal):
    dist = {start: 0}
    queue = deque([start])
    while queue:
        u = queue.popleft()
        for v in neighbours(n, u):
            if not walls[v] and v not in dist:
                dist[v] = dist[u] + 1
                queue.append(v)
    return dist.get(goal)


def test_meta(client):
    body = client.get("/api/rover/meta").json()
    assert body["sizes"] == [21, 41, 61] and body["densities"] == [0.1, 0.2, 0.3]


def test_map_is_the_generators_map(client):
    body = client.get("/api/rover/map", params={"size": 21, "density": 0.2, "seed": 4}).json()
    assert body["walls"] == list(generate(21, 0.2, 4))
    assert body["start"] == 0 and body["goal"] == 21 * 21 - 1


def test_plan_agrees_with_breadth_first_search(client):
    n = 15
    walls = list(generate(n, 0.3, 9))
    body = client.post("/api/rover/plan", json={"n": n, "walls": walls}).json()
    want = bfs(n, walls, 0, n * n - 1)
    assert body["agree"] is True
    assert body["dstar"]["cost"] == body["astar"]["cost"] == want
    assert body["dstar"]["path"][0] == 0 and body["dstar"]["path"][-1] == n * n - 1
    assert len(body["dstar"]["path"]) - 1 == want
    assert all(walls[c] == 0 for c in body["dstar"]["path"])


def test_plan_reports_no_route_as_null(client):
    n = 6
    walls = [1 if i % n == 3 else 0 for i in range(n * n)]   # a wall down the middle column
    body = client.post("/api/rover/plan", json={"n": n, "walls": walls}).json()
    assert body["dstar"]["cost"] is None and body["astar"]["cost"] is None and body["dstar"]["path"] == []


def test_plan_rejects_bad_boards(client):
    assert client.post("/api/rover/plan", json={"n": 5, "walls": [0] * 24}).status_code == 400
    assert client.post("/api/rover/plan", json={"n": 5, "walls": [2] * 25}).status_code == 400
    walls = [0] * 25
    walls[0] = 1
    assert client.post("/api/rover/plan", json={"n": 5, "walls": walls, "start": 0}).status_code == 400


def test_plan_is_rate_limited():
    with TestClient(make_app(rate=2)) as c:
        body = {"n": 5, "walls": [0] * 25}
        codes = [c.post("/api/rover/plan", json=body).status_code for _ in range(4)]
    assert codes[:2] == [200, 200] and 429 in codes[2:]
