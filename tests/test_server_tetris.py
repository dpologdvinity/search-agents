import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from server import tetris_api
from server.limits import RateLimiter, SearchSlots
from tetris.board import FULL, H
from tetris.features import FEATURES, HAND_WEIGHTS
from tetris.search import best_move


def make_app(rate: int = 120) -> FastAPI:
    """A minimal app with just this router, set up the way server.app's lifespan sets it up.

    The router is not registered in server.app here, so the tests mount it directly. Each test gets its
    own rate limiter, so one test's requests cannot throttle another.
    """
    app = FastAPI()
    app.include_router(tetris_api.router)
    app.state.move_rate = RateLimiter(rate, per=60.0)
    app.state.slots = SearchSlots(2, wait_seconds=5.0)
    return app


@pytest.fixture
def client():
    with TestClient(make_app()) as c:
        yield c


def test_meta_describes_the_board_features_and_weights(client):
    body = client.get("/api/tetris/meta").json()
    assert body["width"] == 10 and body["height"] == H
    assert body["features"] == list(FEATURES) and len(body["tuned_weights"]) == 9
    assert body["hand_weights"] == list(HAND_WEIGHTS)
    assert isinstance(body["history"], list)


def test_choose_on_an_empty_board_lists_all_34_t_placements(client):
    body = client.post("/api/tetris/choose", json={"board": [0] * H, "piece": "T"}).json()
    assert len(body["candidates"]) == 34
    assert body["chosen"] is not None and body["game_over"] is False


def test_choose_matches_the_python_search(client):
    rows = [0] * H
    rows[0] = FULL & ~0b1
    rows[1] = 0b11
    body = client.post("/api/tetris/choose", json={"board": rows, "piece": "I", "weights": list(HAND_WEIGHTS)}).json()
    move, _ = best_move(tuple(rows), "I", HAND_WEIGHTS)
    assert (body["chosen"]["rot"], body["chosen"]["x"], body["chosen"]["y"]) == (move.rot, move.x, move.y)
    assert body["chosen"]["lines"] == move.lines


def test_lookahead_reports_a_total(client):
    req = {"board": [0] * H, "piece": "S", "preview": "Z", "lookahead": True}
    body = client.post("/api/tetris/choose", json=req).json()
    assert body["chosen"]["total"] != body["chosen"]["score"]


def test_a_blocked_board_is_game_over(client):
    board = [FULL] * (H - 1) + [0]
    body = client.post("/api/tetris/choose", json={"board": board, "piece": "O"}).json()
    assert body["game_over"] is True and body["chosen"] is None and body["candidates"] == []


def test_bad_requests_are_rejected(client):
    ok = [0] * H
    assert client.post("/api/tetris/choose", json={"board": ok[:-1], "piece": "T"}).status_code == 422
    assert client.post("/api/tetris/choose", json={"board": ok, "piece": "Q"}).status_code == 422
    bad_row = ok[:-1] + [FULL + 1]
    assert client.post("/api/tetris/choose", json={"board": bad_row, "piece": "T"}).status_code == 400
    heavy = [1000.0] + [0.0] * 8
    assert client.post("/api/tetris/choose", json={"board": ok, "piece": "T", "weights": heavy}).status_code == 400
    no_preview = {"board": ok, "piece": "T", "lookahead": True}
    assert client.post("/api/tetris/choose", json=no_preview).status_code == 400


def test_choose_is_rate_limited():
    with TestClient(make_app(rate=2)) as c:
        req = {"board": [0] * H, "piece": "T"}
        assert [c.post("/api/tetris/choose", json=req).status_code for _ in range(3)] == [200, 200, 429]

