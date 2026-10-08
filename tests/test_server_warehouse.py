"""HTTP contract for the warehouse router, tested on a standalone app.

The router is not registered in server/app.py yet, so each test builds a small FastAPI app that
includes it and sets the same rate limiter and search slots that the real lifespan creates.
"""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from server.limits import RateLimiter, SearchSlots
from server.warehouse_api import router

# Two robots must swap through a one-cell pocket: the same instance the core tests use.
SWAP_ROWS = ["##.##", "....."]
SWAP_STARTS = [[0, 1], [4, 1]]
SWAP_GOALS = [[4, 1], [0, 1]]


@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(router)
    app.state.rate = RateLimiter(1000, per=60.0)
    app.state.slots = SearchSlots(2, wait_seconds=5.0)
    with TestClient(app) as c:
        yield c


def swap_body(**extra):
    return {"rows": SWAP_ROWS, "starts": SWAP_STARTS, "goals": SWAP_GOALS, **extra}


def test_meta_lists_layouts_planners_and_limits(client):
    body = client.get("/api/warehouse/meta").json()
    assert len(body["layouts"]) == 3
    assert [p["name"] for p in body["planners"]] == ["cbs", "prioritized", "independent"]
    layout = body["layouts"][0]
    assert layout["height"] == len(layout["rows"]) and layout["width"] == len(layout["rows"][0])
    assert body["limits"]["max_robots"] == 8


def test_cbs_solves_swap_corridor_optimally(client):
    body = client.post("/api/warehouse/solve", json=swap_body(planner="cbs")).json()
    assert body["status"] == "solved"
    assert body["stats"]["sum_of_costs"] == 11
    assert body["conflicts"] == []
    # Paths are [x, y] pairs and start and end where the request said.
    assert body["paths"][0][0] == [0, 1] and body["paths"][0][-1] == [4, 1]
    assert body["paths"][1][0] == [4, 1] and body["paths"][1][-1] == [0, 1]
    # The trace lists expanded nodes; every cell in it has been converted to [x, y].
    trace = body["trace"]
    assert len(trace) > 0
    for entry in trace:
        for conflict_cell in entry["conflict"]["cells"]:
            assert isinstance(conflict_cell, list) and len(conflict_cell) == 2
        for child in entry["children"]:
            constraint = child["constraint"]
            cells = [constraint[k] for k in ("cell", "src", "dst") if k in constraint]
            assert cells and all(isinstance(c, list) and len(c) == 2 for c in cells)


def test_independent_reports_collisions(client):
    body = client.post("/api/warehouse/solve", json=swap_body(planner="independent")).json()
    assert body["status"] == "collisions"
    assert body["stats"]["conflicts"] >= 1
    assert len(body["conflicts"]) >= 1
    assert body["paths"] is not None
    assert "trace" not in body  # only CBS has a constraint tree


def test_prioritized_fails_on_swap_corridor(client):
    body = client.post("/api/warehouse/solve", json=swap_body(planner="prioritized")).json()
    assert body["status"] == "failed"
    assert body["paths"] is None
    assert body["stats"]["sum_of_costs"] is None


@pytest.mark.parametrize("rows, status", [
    (["..", "..", ".."], 422),  # rows shorter than the 3-cell minimum fail the pattern
    (["...", "...", "...", "...", "..."] * 4, 422),  # 20 rows exceed the 16-row maximum
    (["##.##", "...", "....."], 400),  # ragged rows pass the pattern but the core rejects them
])
def test_map_shapes_are_checked(client, rows, status):
    body = {"rows": rows, "starts": [[0, 1]], "goals": [[4, 1]]}
    assert client.post("/api/warehouse/solve", json=body).status_code == status


def test_unknown_map_character_is_rejected(client):
    body = swap_body(rows=["##x##", "....."])
    assert client.post("/api/warehouse/solve", json=body).status_code == 422


def test_too_many_robots_is_rejected(client):
    starts = [[x, 1] for x in range(5)] + [[0, 1]] * 4
    goals = [[x, 1] for x in range(5)] + [[1, 1]] * 4
    body = {"rows": ["......", "......", "......"], "starts": starts, "goals": goals}
    assert client.post("/api/warehouse/solve", json=body).status_code == 422


def test_mismatched_starts_and_goals_are_rejected(client):
    body = swap_body(goals=[[4, 1]])
    assert client.post("/api/warehouse/solve", json=body).status_code == 422


def test_goal_on_a_shelf_is_a_bad_request(client):
    body = swap_body(goals=[[0, 0], [0, 1]])  # (0, 0) is a shelf
    assert client.post("/api/warehouse/solve", json=body).status_code == 400


def test_point_outside_the_map_is_a_bad_request(client):
    body = swap_body(starts=[[9, 9], [4, 1]])
    assert client.post("/api/warehouse/solve", json=body).status_code == 400


def test_random_instance_is_solvable(client):
    rows = ["......", "..##..", "......"]
    body = client.post("/api/warehouse/random", json={"rows": rows, "robots": 3, "seed": 2}).json()
    assert len(body["starts"]) == 3 and len(body["goals"]) == 3
    assert len({tuple(p) for p in body["starts"] + body["goals"]}) == 6  # all distinct
    solved = client.post("/api/warehouse/solve", json={"rows": rows, "starts": body["starts"],
                                                       "goals": body["goals"]}).json()
    assert solved["status"] == "solved"


def test_random_with_too_many_robots_is_a_bad_request(client):
    # Five robots need ten distinct floor cells; a 3 x 3 map has nine.
    body = {"rows": ["...", "...", "..."], "robots": 5, "seed": 1}
    assert client.post("/api/warehouse/random", json=body).status_code == 400


def test_rate_limit_returns_429(client):
    client.app.state.rate = RateLimiter(1, per=60.0)
    assert client.post("/api/warehouse/solve", json=swap_body(planner="independent")).status_code == 200
    assert client.post("/api/warehouse/solve", json=swap_body(planner="independent")).status_code == 429
