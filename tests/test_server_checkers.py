import pytest
from fastapi.testclient import TestClient

from checkers.board import START, legal_moves
from server.app import app


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


def test_legal_moves_from_start(client):
    body = client.post("/api/checkers/legal", json={"board": list(START), "turn": 1}).json()
    assert len(body["moves"]) == 7
    assert all(len(m["path"]) == 2 and not m["captured"] for m in body["moves"])


@pytest.mark.parametrize("agent", ["alphabeta", "minimax"])
def test_agent_move_is_legal_and_analysed(client, agent):
    body = client.post("/api/checkers/move", json={"board": list(START), "turn": -1, "agent": agent, "level": 1}).json()
    legal = {tuple(m.path) for m in legal_moves(START, -1)}
    assert tuple(body["move"]["path"]) in legal
    a = body["analysis"]
    assert a["nodes"] > 0 and a["depth"] >= 1 and len(a["root"]) == 7
    assert len(body["reply"]) > 0  # red has replies
    if agent == "minimax":
        assert a["cutoffs"] == 0


def test_rejects_bad_boards(client):
    assert client.post("/api/checkers/legal", json={"board": [0] * 31, "turn": 1}).status_code == 422
    assert client.post("/api/checkers/legal", json={"board": [5] + [0] * 31, "turn": 1}).status_code == 400
    over = [0] * 31 + [-1]
    assert client.post("/api/checkers/move", json={"board": over, "turn": 1}).status_code == 400
