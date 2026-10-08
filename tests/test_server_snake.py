from fastapi import FastAPI
from fastapi.testclient import TestClient

from server import snake_api
from server.limits import RateLimiter, SearchSlots
from snake.net import N_PARAMS


def make_app(rate: int = 30) -> FastAPI:
    """A minimal app with just this router, set up the way server.app's lifespan sets it up.

    The router is not registered in server.app here, so the tests mount it directly.
    """
    app = FastAPI()
    app.include_router(snake_api.router)
    app.state.move_rate = RateLimiter(rate, per=60.0)
    app.state.rate = RateLimiter(rate, per=60.0)
    app.state.slots = SearchSlots(2, wait_seconds=5.0)
    return app


def start_position(food=(9, 4)):
    """The opening position: a three-segment snake heading east, with food placed by the test."""
    return {"body": [[6, 6], [5, 6], [4, 6]], "heading": 1, "food": list(food)}


def test_meta_describes_the_net_and_the_champion():
    with TestClient(make_app()) as client:
        body = client.get("/api/snake/meta").json()
    assert body["board"] == 12 and body["hidden"] == 16
    assert len(body["input_names"]) == 17 and body["output_names"] == ["left", "straight", "right"]
    assert set(body["agents"]) == {"random", "greedy", "planner", "evolved"}
    assert "w1" not in body["champion"] and "history" not in body["champion"]
    assert body["champion"]["layers"] == [17, 16, 3]


def test_champion_weights_have_the_right_count():
    with TestClient(make_app()) as client:
        data = client.get("/api/snake/champion").json()
    flat = sum(len(r) for r in data["w1"]) + len(data["b1"]) + sum(len(r) for r in data["w2"]) + len(data["b2"])
    assert flat == N_PARAMS
    assert len(data["w1"]) == 17 and len(data["w1"][0]) == 16 and len(data["w2"]) == 16


def test_history_has_one_record_per_generation():
    with TestClient(make_app()) as client:
        history = client.get("/api/snake/history").json()["history"]
    assert history and history[0]["generation"] == 1
    assert {"best_fitness", "mean_fitness", "best_apples", "mean_apples"} <= set(history[0])


def test_hint_returns_the_planner_move_and_route():
    with TestClient(make_app()) as client:
        res = client.post("/api/snake/hint", json=start_position()).json()
    assert res["action"] in (0, 1, 2) and res["action_name"] in ("left", "straight", "right")
    assert res["route"] in ("food", "tail") and res["path"]
    assert res["path"][-1] == [9, 4] or res["route"] == "tail"


def test_hint_rejects_bad_positions():
    with TestClient(make_app()) as client:
        off_board = client.post("/api/snake/hint",
                                json={"body": [[12, 0], [11, 0], [10, 0]], "heading": 1, "food": None})
        broken = client.post("/api/snake/hint", json={"body": [[6, 6], [4, 6], [3, 6]], "heading": 1, "food": [0, 0]})
        on_body = client.post("/api/snake/hint", json={"body": [[6, 6], [5, 6], [4, 6]], "heading": 1, "food": [5, 6]})
        short = client.post("/api/snake/hint", json={"body": [[6, 6]], "heading": 1, "food": None})
    assert off_board.status_code == 400 and broken.status_code == 400
    assert on_body.status_code == 400 and short.status_code == 422


def test_hint_is_rate_limited():
    with TestClient(make_app(rate=2)) as client:
        codes = [client.post("/api/snake/hint", json=start_position()).status_code for _ in range(4)]
    assert codes[:2] == [200, 200] and 429 in codes[2:]
