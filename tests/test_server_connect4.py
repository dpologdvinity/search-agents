import pytest
from fastapi.testclient import TestClient

from connect4.board import Board
from server.app import app


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


@pytest.mark.parametrize("agent", ["minimax", "mcts"])
def test_agent_takes_winning_move(client, agent):
    r = client.post("/api/connect4/move", json={"moves": "121212", "agent": agent, "level": 1})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["move"] == 0
    assert body["analysis"]["kind"] == agent


def test_minimax_analysis_reports_win(client):
    body = client.post("/api/connect4/move", json={"moves": "3355", "agent": "minimax", "level": 2}).json()
    assert body["analysis"]["best"]["result"] == "win"


@pytest.mark.parametrize("moves, fragment", [
    ("1111111", "full column"),
    ("1212121", "already over"),
    ("89", "digits 1-7"),
])
def test_rejects_bad_positions(client, moves, fragment):
    r = client.post("/api/connect4/move", json={"moves": moves, "agent": "minimax"})
    assert r.status_code in (400, 422)
    assert fragment in r.text


def test_alphazero_move_when_weights_exist(client):
    from connect4.net import WEIGHTS

    if not WEIGHTS.exists():
        r = client.post("/api/connect4/move", json={"moves": "", "agent": "alphazero", "level": 1})
        assert r.status_code == 503
        return
    body = client.post("/api/connect4/move", json={"moves": "44", "agent": "alphazero", "level": 1}).json()
    a = body["analysis"]
    assert Board.from_moves("44").can_play(body["move"])
    assert abs(sum(a["prior"]) - 1) < 1e-3
    assert sum(a["visits"]) == a["simulations"]
    assert 0 <= a["win_probability"] <= 1


def test_meta(client):
    names = {a["name"] for a in client.get("/api/connect4/meta").json()["agents"]}
    assert names == {"alphazero", "minimax", "mcts"}
