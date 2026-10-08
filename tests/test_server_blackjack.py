import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from server import blackjack_api
from server.limits import RateLimiter, SearchSlots


@pytest.fixture
def client():
    # The router is not registered in server/app.py yet, so the test app carries the same state the lifespan sets.
    app = FastAPI()
    app.include_router(blackjack_api.router)
    app.state.move_rate = RateLimiter(1000, per=60.0)
    app.state.rate = RateLimiter(1000, per=60.0)
    app.state.slots = SearchSlots(2, wait_seconds=5.0)
    with TestClient(app) as c:
        yield c


def test_strategy_table_is_complete(client):
    body = client.get("/api/blackjack/strategy").json()
    assert len(body["rows"]) == 33
    assert all(len(row["cells"]) == 10 for row in body["rows"])
    assert 0.003 < body["house_edge"] < 0.007
    assert len(body["exact_actions"]) == 33 * 10
    assert {c["action"] for row in body["rows"] for c in row["cells"]} <= {"stand", "hit", "double", "split"}


def test_advice_for_hard_16_against_ten_is_hit(client):
    body = client.post("/api/blackjack/advice", json={"cards": [10, 6], "upcard": 10}).json()
    assert body["total"] == 16 and body["best"] == "hit" and not body["blackjack"]
    assert set(body["actions"]) == {"stand", "hit", "double"}  # two cards, not a pair


def test_advice_for_a_pair_offers_split(client):
    body = client.post("/api/blackjack/advice", json={"cards": [8, 8], "upcard": 10}).json()
    assert body["best"] == "split" and "split" in body["actions"]


def test_advice_for_a_natural(client):
    body = client.post("/api/blackjack/advice", json={"cards": [1, 13], "upcard": 6}).json()
    assert body["blackjack"] and body["best"] == "stand"
    assert body["actions"] == {"stand": 1.5}


def test_advice_for_a_hand_from_a_split_has_no_resplit(client):
    body = client.post("/api/blackjack/advice",
                       json={"cards": [8, 8], "upcard": 10, "from_split": True}).json()
    assert "split" not in body["actions"]


def test_advice_for_a_split_ace_is_stand_only(client):
    body = client.post("/api/blackjack/advice",
                       json={"cards": [1, 5], "upcard": 6, "from_split": True, "split_aces": True}).json()
    assert set(body["actions"]) == {"stand"}


def test_advice_rejects_bad_hands(client):
    assert client.post("/api/blackjack/advice", json={"cards": [10, 6, 9], "upcard": 10}).status_code == 400
    assert client.post("/api/blackjack/advice", json={"cards": [0, 6], "upcard": 10}).status_code == 422
    assert client.post("/api/blackjack/advice", json={"cards": [10, 6], "upcard": 14}).status_code == 422
    assert client.post("/api/blackjack/advice",
                       json={"cards": [1, 5], "upcard": 6, "split_aces": True}).status_code == 400


def test_learn_returns_snapshots_in_table_order(client):
    body = client.post("/api/blackjack/learn", json={"episodes": 1000, "seed": 2, "snapshots": 2}).json()
    assert [s["episodes"] for s in body["snapshots"]] == [500, 1000]
    assert all(len(s["actions"]) == 330 for s in body["snapshots"])
    assert len(body["exact"]) == 330
    assert 0.0 <= body["snapshots"][-1]["agreement"] <= 1.0


def test_learn_rejects_oversized_runs(client):
    assert client.post("/api/blackjack/learn", json={"episodes": 10**7}).status_code == 422
