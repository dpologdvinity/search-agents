import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from lightsout.board import apply_presses, is_dark, to_board
from lightsout.solver import null_space
from server import lightsout_api
from server.limits import RateLimiter, SearchSlots


def make_app(rate: int = 30) -> FastAPI:
    """A minimal app with just this router, set up the way server.app's lifespan sets it up.

    The router is not registered in server.app here, so the tests mount it directly.
    Each test gets its own rate limiter, so one test's requests cannot throttle another.
    """
    app = FastAPI()
    app.include_router(lightsout_api.router)
    app.state.rate = RateLimiter(rate, per=60.0)
    app.state.slots = SearchSlots(2, wait_seconds=5.0)
    return app


@pytest.fixture
def client():
    with TestClient(make_app()) as c:
        yield c


def test_meta_lists_sizes(client):
    body = client.get("/api/lightsout/meta").json()
    assert body["sizes"] == list(range(3, 10))
    assert body["default_size"] == 5 and body["trace_max_size"] == 7


def test_random_board_has_the_right_shape(client):
    body = client.get("/api/lightsout/random", params={"n": 7}).json()
    assert body["n"] == 7 and len(body["board"]) == 49
    assert set(body["board"]) <= {0, 1} and body["solvable"] is True


def test_solve_clears_a_random_solvable_board(client):
    board = client.get("/api/lightsout/random", params={"n": 5}).json()["board"]
    body = client.post("/api/lightsout/solve", json={"n": 5, "board": board}).json()
    assert body["solvable"] is True and body["rank"] == 23 and body["nullity"] == 2
    assert body["solutions"] == 4 and len(body["solution_sizes"]) == 4
    assert body["min_presses"] == len(body["presses"]) == body["solution_sizes"][0]
    assert body["moves"] == [f"{chr(97 + p % 5)}{p // 5 + 1}" for p in body["presses"]]
    assert is_dark(apply_presses(5, board, body["presses"]))
    # The press grid is the same set of cells as the press list.
    assert [i for i, v in enumerate(sum(body["press_grid"], [])) if v] == body["presses"]


def test_unsolvable_board_is_reported(client):
    v = null_space(5)[0]
    board = list(to_board(5, 1 << ((v & -v).bit_length() - 1)))
    body = client.post("/api/lightsout/solve", json={"n": 5, "board": board}).json()
    assert body["solvable"] is False and body["presses"] is None and body["solutions"] == 0
    assert "no press sequence" in body["reason"]
    # The elimination trace is still sent for 5x5, so the page can show why.
    assert body["elimination"]["steps"] and len(body["elimination"]["start"]) == 25


def test_trace_only_up_to_seven(client):
    board = [0] * 81
    body = client.post("/api/lightsout/solve", json={"n": 9, "board": board}).json()
    assert body["elimination"] is None and body["presses"] == []
    body = client.post("/api/lightsout/solve", json={"n": 7, "board": [0] * 49}).json()
    assert body["elimination"] is not None


@pytest.mark.parametrize(
    "payload, status",
    [
        ({"n": 10, "board": [0] * 100}, 422),  # size out of range
        ({"n": 5, "board": [0] * 24}, 400),  # wrong cell count
        ({"n": 5, "board": [2] + [0] * 24}, 400),  # value other than 0 or 1
        ({"n": 5, "board": [0] * 82}, 422),  # longer than the largest board
    ],
)
def test_rejects_bad_boards(client, payload, status):
    assert client.post("/api/lightsout/solve", json=payload).status_code == status


def test_solve_is_rate_limited():
    with TestClient(make_app(rate=2)) as c:
        codes = [c.post("/api/lightsout/solve", json={"n": 3, "board": [0] * 9}).status_code for _ in range(3)]
    assert codes == [200, 200, 429]
