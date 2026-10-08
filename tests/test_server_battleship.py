import time

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from battleship.board import CELLS, FLEET, Fleet, cell_name
from server import battleship_api
from server.limits import RateLimiter, SearchSlots

# A fixed fleet the tests can plant in the server store, so the game's outcome is known.
FIXED = [
    tuple(range(0, 5)),
    tuple(range(20, 24)),
    tuple(range(40, 43)),
    tuple(range(60, 63)),
    tuple(range(80, 82)),
]


@pytest.fixture
def client():
    # The router is not registered in server/app.py here; this app carries the same state the lifespan sets.
    app = FastAPI()
    app.include_router(battleship_api.router)
    app.state.move_rate = RateLimiter(10_000, per=60.0)
    app.state.slots = SearchSlots(2, wait_seconds=5.0)
    with TestClient(app) as c:
        yield c


def plant(fleet_cells) -> str:
    """Put a game with a known fleet into the store and return its id."""
    gid = f"test-{time.monotonic_ns()}"
    battleship_api._GAMES[gid] = battleship_api._Game(Fleet(fleet_cells), time.monotonic())
    return gid


def report(ship):
    """Observations for firing at every cell of one ship in order: hits, then a sunk report with the cells."""
    names = [cell_name(c) for c in ship]
    shots = [{"cell": n, "result": "hit"} for n in names[:-1]]
    shots.append({"cell": names[-1], "result": "sunk", "ship": {"cells": names}})
    return shots


def test_meta_describes_the_fleet(client):
    body = client.get("/api/battleship/meta").json()
    assert body["size"] == 10 and body["columns"] == "ABCDEFGHIJ"
    assert [f["length"] for f in body["fleet"]] == list(FLEET)
    assert body["agent"]["name"] == "probability"


def test_new_game_returns_an_id_and_the_store_is_bounded(client):
    body = client.post("/api/battleship/new").json()
    assert body["size"] == 10 and len(body["id"]) >= 16
    assert [f["length"] for f in body["fleet"]] == list(FLEET)
    for _ in range(battleship_api.MAX_GAMES + 5):
        battleship_api._store(Fleet(FIXED))
    assert len(battleship_api._GAMES) <= battleship_api.MAX_GAMES


def test_shots_report_miss_hit_and_sunk(client):
    gid = plant(FIXED)
    body = client.post("/api/battleship/shot", json={"id": gid, "cell": "f1"}).json()  # cell 5: a miss
    assert body["result"] == "miss" and body["cell"] == "F1" and body["won"] is False
    assert client.post("/api/battleship/shot", json={"id": gid, "cell": "a1"}).json()["result"] == "hit"
    last = None
    for c in FIXED[0][1:]:
        last = client.post("/api/battleship/shot", json={"id": gid, "cell": cell_name(c)}).json()
    assert last["result"] == "sunk" and last["ship"]["name"] == "Carrier" and last["ship"]["length"] == 5
    assert last["afloat"] == 4


def test_a_won_game_reveals_the_fleet_and_then_refuses_shots(client):
    gid = plant(FIXED)
    body = None
    for ship in FIXED:
        for c in ship:
            body = client.post("/api/battleship/shot", json={"id": gid, "cell": cell_name(c)}).json()
    assert body["won"] is True and body["afloat"] == 0
    assert [len(s["cells"]) for s in body["fleet"]] == list(FLEET)
    again = client.post("/api/battleship/shot", json={"id": gid, "cell": "J10"})
    assert again.status_code == 400


def test_shot_errors(client):
    gid = plant(FIXED)
    assert client.post("/api/battleship/shot", json={"id": "nope", "cell": "a1"}).status_code == 404
    assert client.post("/api/battleship/shot", json={"id": gid, "cell": "z9"}).status_code == 400
    assert client.post("/api/battleship/shot", json={"id": gid, "cell": "a1"}).status_code == 200
    assert client.post("/api/battleship/shot", json={"id": gid, "cell": "a1"}).status_code == 400  # repeat


def test_agent_shot_returns_odds_that_sum_to_the_ship_cells(client):
    body = client.post("/api/battleship/agent-shot", json={"shots": []}).json()
    assert len(body["probs"]) == CELLS
    assert sum(body["probs"]) == pytest.approx(sum(FLEET), abs=1e-3)
    assert body["remaining"] == list(FLEET)
    assert body["probability"] == body["top"][0]["probability"]  # the choice is the most likely cell


def test_agent_shot_never_repeats_a_cell_and_targets_next_to_a_hit(client):
    shots = [{"cell": "E5", "result": "hit"}]  # cell 44
    body = client.post("/api/battleship/agent-shot", json={"shots": shots}).json()
    assert body["choice"] != "E5"
    assert body["probs"][44] == 1.0
    # The cell chosen is the most likely unknown cell, and a neighbour of the hit is more likely than a corner.
    assert body["probability"] == max(p for c, p in enumerate(body["probs"]) if c != 44)
    assert body["probs"][54] > body["probs"][0]


def test_agent_shot_accepts_a_sunk_ship_and_rejects_bad_reports(client):
    ok = client.post("/api/battleship/agent-shot", json={"shots": report(FIXED[4])}).json()
    assert ok["remaining"] == [5, 4, 3, 3]
    assert sum(ok["probs"]) == pytest.approx(sum(FLEET), abs=0.02)  # the two sunk cells count as certain
    no_ship = [{"cell": "A1", "result": "sunk"}]
    assert client.post("/api/battleship/agent-shot", json={"shots": no_ship}).status_code == 400
    wrong = [{"cell": "A1", "result": "hit", "ship": {"cells": ["A1", "A2"]}}]
    assert client.post("/api/battleship/agent-shot", json={"shots": wrong}).status_code == 400
    twice = [{"cell": "A1", "result": "miss"}, {"cell": "A1", "result": "hit"}]
    assert client.post("/api/battleship/agent-shot", json={"shots": twice}).status_code == 400
    off_board = [{"cell": "K1", "result": "miss"}]
    assert client.post("/api/battleship/agent-shot", json={"shots": off_board}).status_code == 400
    too_long = [{"cell": "A1", "result": "sunk", "ship": {"cells": [f"{c}1" for c in "ABCDEFG"]}}]
    assert client.post("/api/battleship/agent-shot", json={"shots": too_long}).status_code == 422


def test_agent_shot_refuses_when_every_ship_is_sunk(client):
    shots = []
    for ship in FIXED:
        shots.extend(report(ship))
    assert client.post("/api/battleship/agent-shot", json={"shots": shots}).status_code == 400
