from fastapi import FastAPI
from fastapi.testclient import TestClient

from server import sokoban_api
from server.limits import RateLimiter, SearchSlots
from sokoban.levels import NAMES, all_levels


def make_app(rate: int = 30) -> FastAPI:
    """A minimal app with just this router, set up the way server.app's lifespan sets it up.

    The router is not registered in server.app here, so the tests mount it directly.
    Each test gets its own rate limiters, so one test's requests cannot throttle another.
    """
    app = FastAPI()
    app.include_router(sokoban_api.router)
    app.state.rate = RateLimiter(rate, per=60.0)
    app.state.move_rate = RateLimiter(rate, per=60.0)
    app.state.slots = SearchSlots(2, wait_seconds=5.0)
    return app


def client_for(rate: int = 30):
    return TestClient(make_app(rate))


def test_levels_lists_every_level_with_its_optimal_pushes():
    with client_for() as c:
        body = c.get("/api/sokoban/levels").json()
    assert body["count"] == len(all_levels()) == 12
    assert [lv["name"] for lv in body["levels"]] == list(NAMES)
    assert body["levels"][0]["optimal_pushes"] == 1


def test_one_level_has_rows_goals_dead_squares_and_distances():
    with client_for() as c:
        body = c.get("/api/sokoban/level/5").json()
    assert body["cols"] * body["rows"] == len(body["nearest"])
    assert len(body["goals"]) == len(body["boxes"]) > 0
    assert body["player"] not in body["boxes"]
    assert all(body["nearest"][i] == -1 for i in body["dead"])
    assert all(body["nearest"][g] == 0 for g in body["goals"])
    assert len(body["text"]) == body["rows"]  # one string per row, walls included


def test_level_number_out_of_range_is_404():
    with client_for() as c:
        assert c.get("/api/sokoban/level/0").status_code == 404
        assert c.get("/api/sokoban/level/13").status_code == 404


def test_solve_the_warm_up_level():
    with client_for() as c:
        body = c.post("/api/sokoban/solve", json={"level": 1}).json()
    assert body["solved"] is True and body["status"] == "solved"
    assert body["moves"] == "D" and body["pushes"] == 1
    assert body["optimal_pushes"] == 1
    assert body["samples"][-1][0] <= body["expanded"]


def test_solve_reports_budget_exhaustion_without_a_plan():
    with client_for() as c:
        body = c.post("/api/sokoban/solve", json={"level": 12, "algorithm": "bfs", "prune": False,
                                                    "heuristic": "none", "budget": 40}).json()
    assert body["status"] == "budget" and body["solved"] is False and body["moves"] is None


def test_solve_rejects_budgets_over_the_cap():
    with client_for() as c:
        r = c.post("/api/sokoban/solve", json={"level": 1, "budget": sokoban_api.NODE_CAP + 1})
    assert r.status_code == 422


def test_solve_from_a_custom_position_uses_interior_indices():
    # Indices count the text grid with its walls: level 1 is "#####" / "#@$.#" / "#####", five columns wide.
    # So the player is at 6, the box at 7, the goal at 8, and index 1 is a wall.
    with client_for() as c:
        ok = c.post("/api/sokoban/solve", json={"level": 1, "boxes": [7], "player": 6}).json()
        moved = c.post("/api/sokoban/solve", json={"level": 1, "boxes": [8], "player": 6}).json()
        bad_count = c.post("/api/sokoban/solve", json={"level": 1, "boxes": [7, 8], "player": 6})
        on_wall = c.post("/api/sokoban/solve", json={"level": 1, "boxes": [1], "player": 6})
        off_board = c.post("/api/sokoban/solve", json={"level": 1, "boxes": [99], "player": 6})
        on_player = c.post("/api/sokoban/solve", json={"level": 1, "boxes": [6], "player": 6})
    assert ok["solved"] is True and ok["moves"] == "D" and ok["optimal_pushes"] is None
    assert moved["moves"] == ""  # the box already sits on the goal
    for bad in (bad_count, on_wall, off_board, on_player):
        assert bad.status_code == 400


def test_evaluate_flags_a_dead_square():
    # Level 5's dead squares are its room corners; a box moved into one is a deadlock.
    with client_for() as c:
        start = c.get("/api/sokoban/level/5").json()
        ok = c.post("/api/sokoban/evaluate", json={"level": 5, "boxes": start["boxes"]}).json()
        assert ok["deadlock"] is None
        dead_cell = start["dead"][0]
        dead = c.post("/api/sokoban/evaluate", json={"level": 5, "boxes": [dead_cell] + start["boxes"][1:]}).json()
    assert dead["deadlock"] == "dead square"
    assert dead_cell in dead["dead"]


def test_evaluate_matching_pairs_cover_every_box():
    with client_for() as c:
        start = c.get("/api/sokoban/level/7").json()
        body = c.post("/api/sokoban/evaluate", json={"level": 7, "boxes": start["boxes"]}).json()
    assert sorted(p[0] for p in body["pairs"]) == sorted(start["boxes"])
    assert sorted(p[1] for p in body["pairs"]) == sorted(start["goals"])
    assert body["h_matching"] == sum(p[2] for p in body["pairs"])
    assert body["h_matching"] >= body["h_simple"]


def test_rate_limit_returns_429():
    with client_for(rate=1) as c:
        assert c.post("/api/sokoban/solve", json={"level": 1}).status_code == 200
        assert c.post("/api/sokoban/solve", json={"level": 1}).status_code == 429
