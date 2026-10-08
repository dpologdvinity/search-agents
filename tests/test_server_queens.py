import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from queens.board import brute_conflicts, is_solution
from server import queens_api
from server.limits import RateLimiter, SearchSlots


def make_app(rate: int = 30) -> FastAPI:
    """A minimal app with just this router, set up the way server.app's lifespan sets it up.

    The router is not registered in server.app here, so the tests mount it directly. Each test
    gets its own rate limiter, so one test's requests cannot throttle another.
    """
    app = FastAPI()
    app.include_router(queens_api.router)
    app.state.rate = RateLimiter(rate, per=60.0)
    app.state.slots = SearchSlots(2, wait_seconds=5.0)
    return app


@pytest.fixture
def client():
    with TestClient(make_app()) as c:
        yield c


def test_meta_lists_agents_and_limits(client):
    body = client.get("/api/queens/meta").json()
    assert set(body["agents"]) == {"backtrack", "hill", "anneal", "minconf"}
    assert body["agents"]["minconf"]["max_n"] == 10_000 and body["frames_max_n"] == 32


def test_minconf_solves_and_the_answer_is_checked(client):
    body = client.post("/api/queens/solve", json={"agent": "minconf", "n": 50, "seed": 3}).json()
    assert body["solved"] is True and body["reason"] == "solved"
    assert is_solution(body["cols"]) and brute_conflicts(body["cols"]) == 0
    assert body["trace_kind"] == "conflicts" and body["trace"][-1] == 0
    assert body["frames"] is None  # frames are only sent when asked for


def test_backtrack_reports_open_rows_not_conflicts(client):
    body = client.post("/api/queens/solve", json={"agent": "backtrack", "n": 8, "frames": True}).json()
    assert body["solved"] is True and body["trace_kind"] == "open_rows"
    assert body["trace"][0] == 8 and body["trace"][-1] == 0
    assert body["frames"][0] == [-1] * 8 and body["frames"][-1] == body["cols"]


def test_frames_come_back_for_small_boards_only(client):
    body = client.post("/api/queens/solve", json={"agent": "hill", "n": 8, "seed": 2, "frames": True}).json()
    assert body["frames"] and len(body["frames"][0]) == 8
    big = client.post("/api/queens/solve", json={"agent": "minconf", "n": 100, "frames": True}).json()
    assert big["frames"] is None


def test_backtrack_past_its_size_limit_is_refused(client):
    res = client.post("/api/queens/solve", json={"agent": "backtrack", "n": 1000})
    assert res.status_code == 400


@pytest.mark.parametrize(
    "payload",
    [
        {"agent": "genetic", "n": 8},  # unknown agent
        {"agent": "minconf", "n": 0},  # no board
        {"agent": "minconf", "n": 10_001},  # over the largest board
        {"agent": "minconf", "n": 8, "seed": -1},
    ],
)
def test_rejects_bad_requests(client, payload):
    assert client.post("/api/queens/solve", json=payload).status_code == 422


def test_solve_is_rate_limited():
    with TestClient(make_app(rate=2)) as c:
        codes = [c.post("/api/queens/solve", json={"agent": "minconf", "n": 8}).status_code for _ in range(3)]
    assert codes == [200, 200, 429]
