from fastapi import FastAPI
from fastapi.testclient import TestClient

from nonogram.library import PICTURES, get
from nonogram.solvers import verify
from server import nonogram_api
from server.limits import RateLimiter, SearchSlots


def make_app(rate: int = 30, move_rate: int = 120) -> FastAPI:
    """A minimal app with just this router, set up as server.app's lifespan sets it up.

    Each test gets its own limiters, so one test's requests cannot throttle another.
    """
    app = FastAPI()
    app.include_router(nonogram_api.router)
    app.state.rate = RateLimiter(rate, per=60.0)
    app.state.move_rate = RateLimiter(move_rate, per=60.0)
    app.state.slots = SearchSlots(2, wait_seconds=5.0)
    return app


def heart_clues(client):
    body = client.get("/api/nonogram/puzzle/heart").json()
    return body["row_clues"], body["col_clues"]


def test_meta_and_puzzle_list():
    with TestClient(make_app()) as client:
        meta = client.get("/api/nonogram/meta").json()
        assert meta["sizes"] == list(range(5, 21)) and meta["max_size"] == 25
        assert meta["methods"] == ["line", "hybrid", "sat"]
        listing = client.get("/api/nonogram/puzzles").json()["puzzles"]
        assert len(listing) == len(PICTURES)
        assert {p["id"] for p in listing} == {pid for pid, _, _ in PICTURES}


def test_puzzle_returns_clues_without_the_picture():
    with TestClient(make_app()) as client:
        body = client.get("/api/nonogram/puzzle/heart").json()
        assert body["row_clues"] == [list(c) for c in get("heart").row_clues]
        assert "solution" not in body
        assert client.get("/api/nonogram/puzzle/nope").status_code == 404


def test_random_is_unique_and_repeats_for_a_seed():
    with TestClient(make_app()) as client:
        a = client.get("/api/nonogram/random", params={"rows": 8, "cols": 9, "seed": 4}).json()
        b = client.get("/api/nonogram/random", params={"rows": 8, "cols": 9, "seed": 4}).json()
        assert a == b
        assert a["rows"] == 8 and a["cols"] == 9 and a["seed"] == 4 and a["id"] is None
        assert client.get("/api/nonogram/random", params={"rows": 4, "cols": 9}).status_code == 422


def test_solve_by_each_method_returns_the_picture():
    with TestClient(make_app()) as client:
        row_clues, col_clues = heart_clues(client)
        picture = [[1 if ch == "#" else 0 for ch in row] for row in PICTURES[0][2]]
        for method in ("line", "hybrid", "sat"):
            body = client.post("/api/nonogram/solve", json={
                "row_clues": row_clues, "col_clues": col_clues, "method": method, "trace": True}).json()
            assert body["status"] == "unique", method
            assert body["solution"] == picture
            assert body["stats"]["method"] == method
            assert verify(get("heart"), body["solution"])
        body = client.post("/api/nonogram/solve", json={
            "row_clues": row_clues, "col_clues": col_clues, "method": "sat", "trace": True}).json()
        assert body["stats"]["variables"] > 25 and body["events"] is not None
        assert body["events_truncated"] is False


def test_line_method_returns_rounds_only_when_traced():
    with TestClient(make_app()) as client:
        row_clues, col_clues = heart_clues(client)
        body = client.post("/api/nonogram/solve", json={
            "row_clues": row_clues, "col_clues": col_clues, "method": "line", "trace": True}).json()
        assert body["rounds"] and body["rounds"][0]["axis"] == "rows"
        body = client.post("/api/nonogram/solve", json={
            "row_clues": row_clues, "col_clues": col_clues, "method": "line"}).json()
        assert body["rounds"]  # trace defaults to true
        plain = client.post("/api/nonogram/solve", json={
            "row_clues": row_clues, "col_clues": col_clues, "method": "line", "trace": False}).json()
        assert plain["rounds"] is None


def test_invalid_clues_and_methods_are_rejected():
    with TestClient(make_app()) as client:
        bad = client.post("/api/nonogram/solve", json={"row_clues": [[3, 3]], "col_clues": [[1]] * 6,
                                                       "method": "hybrid"})
        assert bad.status_code == 422
        unknown = client.post("/api/nonogram/solve", json={"row_clues": [[1]], "col_clues": [[1]],
                                                           "method": "magic"})
        assert unknown.status_code == 422


def test_solve_reports_multiple_solutions():
    with TestClient(make_app()) as client:
        body = client.post("/api/nonogram/solve", json={
            "row_clues": [[1], [1]], "col_clues": [[1], [1]], "method": "hybrid"}).json()
        assert body["status"] == "multiple"
        assert body["second"] is not None and body["second"] != body["solution"]


def test_hint_endpoint_and_grid_checks():
    with TestClient(make_app()) as client:
        row_clues, col_clues = heart_clues(client)
        grid = [[-1] * 5 for _ in range(5)]
        body = client.post("/api/nonogram/hint", json={"row_clues": row_clues, "col_clues": col_clues,
                                                       "grid": grid}).json()
        assert body["found"] is True and body["reason"]
        wrong_shape = client.post("/api/nonogram/hint", json={"row_clues": row_clues, "col_clues": col_clues,
                                                              "grid": [[-1] * 4] * 5})
        assert wrong_shape.status_code == 400
        wrong_value = client.post("/api/nonogram/hint", json={"row_clues": row_clues, "col_clues": col_clues,
                                                              "grid": [[2] * 5] * 5})
        assert wrong_value.status_code == 400


def test_solve_is_rate_limited():
    with TestClient(make_app(rate=1)) as client:
        row_clues, col_clues = heart_clues(client)
        payload = {"row_clues": row_clues, "col_clues": col_clues, "method": "line"}
        assert client.post("/api/nonogram/solve", json=payload).status_code == 200
        assert client.post("/api/nonogram/solve", json=payload).status_code == 429
