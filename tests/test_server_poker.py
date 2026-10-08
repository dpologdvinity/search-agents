from fastapi import FastAPI
from fastapi.testclient import TestClient

from server import poker_api
from server.limits import RateLimiter


def make_app(rate: int = 1000) -> FastAPI:
    """A minimal app with just this router, set up the way server.app's lifespan sets it up.

    The router is not registered in server.app here, so the tests mount it directly. Each test gets
    its own rate limiters, so one test's requests cannot throttle another.
    """
    app = FastAPI()
    app.include_router(poker_api.router)
    app.state.rate = RateLimiter(rate, per=60.0)
    app.state.move_rate = RateLimiter(rate, per=60.0)
    return app


def test_meta_and_training_and_strategy_endpoints():
    client = TestClient(make_app())
    meta = client.get("/api/poker/meta").json()
    assert meta["infosets"] == 288 and meta["big_blind"] == 2 and meta["bet_sizes"] == [2, 4]
    assert meta["algorithm"] == "CFR+"
    training = client.get("/api/poker/training").json()
    names = [r["algorithm"] for r in training["runs"]]
    assert names == ["CFR", "CFR+"]
    assert training["kuhn_check"]["cfr+"]["value_error"] < 1e-4
    table = client.get("/api/poker/strategy").json()
    assert len(table) == 288 and "K|-|" in table


def test_deal_starts_a_hand_with_you_to_act():
    client = TestClient(make_app())
    view = client.post("/api/poker/deal", json={"seed": 4}).json()
    assert view["to_act"] == 0 and view["history"] == "" and view["round"] == 1
    assert view["card"] in ("J", "Q", "K")
    assert view["public"] is None and view["bot_card"] is None  # the public and bot cards stay hidden
    assert view["legal"] == ["k", "b"] and view["hint"]["probs"]
    assert view["pot"] == 2


def test_the_bot_card_is_hidden_until_showdown():
    client = TestClient(make_app())
    view = client.post("/api/poker/deal", json={"seed": 9}).json()
    for _ in range(20):
        if view["terminal"]:
            break
        if view["to_act"] != 0:
            break
        action = "c" if "c" in view["legal"] else "k"
        view = client.post("/api/poker/act", json={"hand": view["hand"], "action": action}).json()
        if not view["terminal"]:
            assert view["bot_card"] is None
    if view["terminal"]:
        assert view["bot_card"] in ("J", "Q", "K")
        assert view["result"] in ("win", "lose", "split")


def test_a_full_hand_settles_and_the_id_cannot_be_replayed():
    client = TestClient(make_app())
    view = client.post("/api/poker/deal", json={"seed": 2}).json()
    hand_id = view["hand"]
    # Call what is owed and check otherwise, so the hand reaches a showdown.
    for _ in range(30):
        if view["terminal"]:
            break
        action = "c" if "c" in view["legal"] else "k"
        view = client.post("/api/poker/act", json={"hand": hand_id, "action": action}).json()
    assert view["terminal"] and view["payoff"] is not None and view["result"] is not None
    assert view["bot_card"] in ("J", "Q", "K") and view["public"] is not None
    again = client.post("/api/poker/act", json={"hand": hand_id, "action": "k"})
    assert again.status_code == 404  # settled hands are dropped


def test_illegal_and_out_of_turn_actions_are_refused():
    client = TestClient(make_app())
    view = client.post("/api/poker/deal", json={"seed": 5}).json()
    resp = client.post("/api/poker/act", json={"hand": view["hand"], "action": "c"})
    assert resp.status_code == 400  # nothing is owed, so call is not legal
    assert client.post("/api/poker/act", json={"hand": "nope-not-a-hand", "action": "k"}).status_code == 404
    assert client.post("/api/poker/act", json={"hand": view["hand"], "action": "x"}).status_code == 422


def test_bot_moves_carry_the_mix_it_played_from():
    client = TestClient(make_app())
    for seed in range(40):
        view = client.post("/api/poker/deal", json={"seed": seed}).json()
        if view["legal"] and "b" in view["legal"]:
            view = client.post("/api/poker/act", json={"hand": view["hand"], "action": "k"}).json()
            for move in view["bot_moves"]:
                assert move["action"] in move["probs"]
                assert abs(sum(move["probs"].values()) - 1.0) < 1e-6
            if view["bot_moves"]:
                return
    raise AssertionError("no hand where the bot acted after a check")


def test_the_deal_is_rate_limited():
    client = TestClient(make_app(rate=2))
    codes = [client.post("/api/poker/deal", json={"seed": i}).status_code for i in range(3)]
    assert codes == [200, 200, 429]
