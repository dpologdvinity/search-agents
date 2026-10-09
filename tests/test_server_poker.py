import json

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


def _play(client: TestClient, seed: int, fold_when_able: bool = False) -> list[dict]:
    """Play one hand to the end (calling what is owed, or folding on request) and return every response."""
    views = [client.post("/api/poker/deal", json={"seed": seed}).json()]
    for _ in range(30):
        view = views[-1]
        if view["terminal"]:
            break
        if fold_when_able and "f" in view["legal"]:
            action = "f"
        else:
            action = "c" if "c" in view["legal"] else "k"
        views.append(client.post("/api/poker/act", json={"hand": view["hand"], "action": action}).json())
    assert views[-1]["terminal"]
    return views


def test_bot_moves_carry_the_spot_until_showdown_and_the_mix_after():
    client = TestClient(make_app())
    bot_acted = 0
    for seed in range(40):
        views = _play(client, seed)
        public = [m for v in views[1:] for m in v["bot_moves"]]
        for move in public:
            assert set(move) == {"round", "spot", "action"}  # no probabilities before the showdown
            assert move["spot"].startswith("?")
        reveal = views[-1]["bot_reveal"]
        assert [d["action"] for d in reveal] == [m["action"] for m in public]
        for decision in reveal:
            assert decision["action"] in decision["probs"]
            assert abs(sum(decision["probs"].values()) - 1.0) < 1e-6
            assert decision["spot"] == "?" + decision["infoset"][1:]
        bot_acted += bool(public)
    assert bot_acted > 0


def test_no_response_before_showdown_carries_the_bot_card_or_its_mix():
    """Scan every field of every response before the showdown for the bot's information sets and mixes.

    The bot's information sets are read from the reveal in the last response of the same hand, which the
    server withholds until then, so the scan knows exactly which keys must stay hidden.
    """
    client = TestClient(make_app())
    for seed in range(30):
        views = _play(client, seed, fold_when_able=seed % 3 == 0)
        bot_infosets = [d["infoset"] for d in views[-1]["bot_reveal"]]
        for view in views[:-1]:
            assert view["bot_card"] is None and view["bot_reveal"] is None
            text = json.dumps({k: v for k, v in view.items() if k != "hint"})  # the hint is the human's own mix
            assert '"probs"' not in text and '"infoset"' not in text
            for key in bot_infosets:
                assert f'"{key}"' not in text
            for move in view["bot_moves"]:
                assert set(move) == {"round", "spot", "action"}


def test_the_deal_is_rate_limited():
    client = TestClient(make_app(rate=2))
    codes = [client.post("/api/poker/deal", json={"seed": i}).status_code for i in range(3)]
    assert codes == [200, 200, 429]
