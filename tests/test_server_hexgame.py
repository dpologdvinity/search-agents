from fastapi import FastAPI
from fastapi.testclient import TestClient

from hexgame.board import ACROSS, DOWN
from server import hexgame_api
from server.limits import RateLimiter, SearchSlots


def make_app(rate: int = 60) -> FastAPI:
    """A minimal app with just this router, set up the way server.app's lifespan sets it up.

    The router is not registered in server.app here, so the tests mount it directly.
    """
    app = FastAPI()
    app.include_router(hexgame_api.router)
    app.state.move_rate = RateLimiter(rate, per=60.0)
    app.state.slots = SearchSlots(2, wait_seconds=5.0)
    return app


def test_meta_lists_sizes_agents_and_levels():
    with TestClient(make_app()) as client:
        body = client.get("/api/hexgame/meta").json()
    assert body["sizes"] == list(range(5, 12)) and body["default_size"] == 7
    assert {a["name"] for a in body["agents"]} == {"rave", "uct", "shortest", "random", "chance"}
    assert body["chance"] == {"ring_weights": [4, 3, 2, 1]}
    assert set(body["levels"]) == {"1", "2", "3"}


def test_agent_move_is_legal_and_comes_with_its_search():
    with TestClient(make_app()) as client:
        body = client.post("/api/hexgame/move", json={"size": 5, "moves": [12], "agent": "rave", "level": 1}).json()
    assert body["winner"] is None and body["to_move"] == ACROSS
    assert body["move"] not in (12,) and 0 <= body["move"] < 25
    assert body["label"] and body["analysis"]["agent"] == "rave"
    assert len(body["analysis"]["visits"]) == 25
    assert body["analysis"]["visits"][12] == 0  # the taken cell is never a child
    assert body["analysis"]["pv_labels"][0] == body["label"]


def test_no_agent_means_no_move_and_the_position_is_reported():
    with TestClient(make_app()) as client:
        body = client.post("/api/hexgame/move", json={"size": 5, "moves": [0, 6]}).json()
    assert body["move"] is None and body["analysis"] is None
    assert body["to_move"] == DOWN and body["winner"] is None


def test_a_finished_game_reports_the_winning_chain_and_makes_no_move():
    # DOWN takes column 0 (cells 0, 5, 10, 15, 20) while ACROSS fills column 4.
    moves = [0, 4, 5, 9, 10, 14, 15, 19, 20]
    with TestClient(make_app()) as client:
        body = client.post("/api/hexgame/move", json={"size": 5, "moves": moves, "agent": "uct"}).json()
    assert body["winner"] == DOWN and body["chain"] == [0, 5, 10, 15, 20]
    assert body["move"] is None


def test_illegal_histories_are_refused():
    with TestClient(make_app()) as client:
        taken = client.post("/api/hexgame/move", json={"size": 5, "moves": [3, 3]})
        off = client.post("/api/hexgame/move", json={"size": 5, "moves": [25]})
        swap_without_rule = client.post("/api/hexgame/move", json={"size": 5, "moves": [3, -1]})
        bad_size = client.post("/api/hexgame/move", json={"size": 4, "moves": []})
    assert taken.status_code == 400 and "already taken" in taken.json()["detail"]
    assert off.status_code == 400
    assert swap_without_rule.status_code == 400
    assert bad_size.status_code == 422


def test_swap_is_accepted_with_the_swap_rule_on():
    with TestClient(make_app()) as client:
        body = client.post("/api/hexgame/move", json={"size": 5, "moves": [12, -1], "swap": True,
                                                       "agent": "random"}).json()
    assert body["to_move"] == DOWN and body["winner"] is None
    assert body["move"] in range(25) and body["move"] != 12


def test_rate_limit_answers_429():
    with TestClient(make_app(rate=1)) as client:
        first = client.post("/api/hexgame/move", json={"size": 5, "moves": []})
        second = client.post("/api/hexgame/move", json={"size": 5, "moves": []})
    assert first.status_code == 200
    assert second.status_code == 429


def test_chance_move_is_legal_and_reports_its_odds():
    with TestClient(make_app()) as client:
        r = client.post("/api/hexgame/move", json={"size": 7, "moves": [24], "agent": "chance"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["move"] not in (24,) and body["move"] != -1
    odds = body["analysis"]["odds"]
    assert len(odds) == 49 and odds[24] == 0 and abs(sum(odds) - 1) < 1e-3


def test_unknown_agent_name_is_rejected():
    with TestClient(make_app()) as client:
        r = client.post("/api/hexgame/move", json={"size": 7, "moves": [], "agent": "nope"})
    assert r.status_code == 422
