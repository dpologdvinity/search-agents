"""API tests for the chance opponent: the deal option, the 422 for unknown opponents, /meta, and the hidden card.

The CFR+ opponent's API tests are in test_server_poker.py. Here every hand is played with chance in seat 1.
"""

import json

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from poker.chance import ACTION_ODDS, legal_weights
from server import poker_api
from server.limits import RateLimiter


def make_app(rate: int = 1000) -> FastAPI:
    """A minimal app with just the poker router, with its own rate limiters, as in test_server_poker.py."""
    app = FastAPI()
    app.include_router(poker_api.router)
    app.state.rate = RateLimiter(rate, per=60.0)
    app.state.move_rate = RateLimiter(rate, per=60.0)
    return app


def _play(client: TestClient, seed: int, opponent: str, fold_when_able: bool = False) -> list[dict]:
    """Play one hand to the end, calling what is owed (or folding on request), and return every response."""
    views = [client.post("/api/poker/deal", json={"seed": seed, "opponent": opponent}).json()]
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


def test_meta_lists_both_opponents_and_the_chance_table():
    meta = TestClient(make_app()).get("/api/poker/meta").json()
    by_name = {o["name"]: o for o in meta["opponents"]}
    assert set(by_name) == {"cfr", "chance"}
    assert by_name["chance"]["label"] == "Chance (fixed odds)"
    assert by_name["chance"]["action_odds"] == ACTION_ODDS
    assert "CFR+" in by_name["cfr"]["algorithm"] and "fixed" in by_name["chance"]["algorithm"]
    assert "action_odds" not in by_name["cfr"]
    assert meta["algorithm"] == "CFR+"  # the committed table is still the CFR+ one


def test_deal_defaults_to_cfr_and_accepts_chance():
    client = TestClient(make_app())
    assert client.post("/api/poker/deal", json={"seed": 4}).json()["opponent"] == "cfr"
    resp = client.post("/api/poker/deal", json={"seed": 4, "opponent": "chance"})
    assert resp.status_code == 200
    view = resp.json()
    assert view["opponent"] == "chance" and view["to_act"] == 0 and view["legal"] == ["k", "b"]


def test_an_unknown_opponent_is_refused_with_422():
    client = TestClient(make_app())
    assert client.post("/api/poker/deal", json={"seed": 4, "opponent": "minimax"}).status_code == 422
    assert client.post("/api/poker/deal", json={"seed": 4, "opponent": ""}).status_code == 422


def test_the_chance_deal_is_rate_limited():
    client = TestClient(make_app(rate=2))
    codes = [client.post("/api/poker/deal", json={"seed": i, "opponent": "chance"}).status_code for i in range(3)]
    assert codes == [200, 200, 429]


@pytest.mark.parametrize("fold_when_able", [False, True])
def test_the_chance_bot_card_is_hidden_until_showdown(fold_when_able):
    """Scan every field of every response before the showdown for the bot's card, mix, and information sets."""
    client = TestClient(make_app())
    for seed in range(30):
        views = _play(client, seed, "chance", fold_when_able=fold_when_able and seed % 2 == 0)
        bot_infosets = [d["infoset"] for d in views[-1]["bot_reveal"]]
        for view in views[:-1]:
            assert view["opponent"] == "chance"
            assert view["bot_card"] is None and view["bot_reveal"] is None
            text = json.dumps({k: v for k, v in view.items() if k != "hint"})  # the hint is the human's own mix
            assert '"probs"' not in text and '"infoset"' not in text
            for key in bot_infosets:
                assert f'"{key}"' not in text
            for move in view["bot_moves"]:
                assert set(move) == {"round", "spot", "action"}
                assert move["spot"].startswith("?")
        final = views[-1]
        if final["terminal"] and final["bot_card"] is not None:
            assert final["bot_card"] in ("J", "Q", "K")


def test_chance_moves_are_drawn_from_the_legal_weights_after_showdown():
    client = TestClient(make_app())
    bot_acted = 0
    for seed in range(40):
        reveal = _play(client, seed, "chance")[-1]["bot_reveal"]
        for decision in reveal:
            assert decision["action"] in decision["probs"]
            assert decision["probs"] == pytest.approx(legal_weights(tuple(decision["probs"])))
            assert decision["spot"] == "?" + decision["infoset"][1:]
        bot_acted += bool(reveal)
    assert bot_acted > 0
