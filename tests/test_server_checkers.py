import pytest
from fastapi.testclient import TestClient

from checkers.agents import AGENTS
from checkers.board import START, index, legal_moves, start_position
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


@pytest.mark.parametrize("size, other", [(8, 10), (10, 12), (12, 8)])
def test_size_must_match_the_board_length(client, size, other):
    # A correct board is accepted; the same squares sent with the wrong size are a 422.
    body = {"board": list(start_position(size)), "size": size}
    assert client.post("/api/checkers/legal", json=body).status_code == 200
    body = {"board": list(start_position(size)), "size": other}
    assert client.post("/api/checkers/legal", json=body).status_code == 422


def test_unsupported_size_and_bad_lengths_are_rejected(client):
    assert client.post("/api/checkers/legal", json={"board": [0] * 40, "size": 8}).status_code == 422
    assert client.post("/api/checkers/legal", json={"board": list(START), "size": 9}).status_code == 422
    assert client.post("/api/checkers/legal", json={"board": [0] * 80, "size": 12}).status_code == 422


def test_forced_flag_changes_the_legal_moves(client):
    # With a capture available, forced lists only captures; optional also lists the free moves.
    board = [0] * 32
    board[index(6, 1)] = 1
    board[index(5, 2)] = -1
    board[index(6, 5)] = 1
    body = {"board": board, "turn": 1}
    forced = client.post("/api/checkers/legal", json={**body, "forced": True}).json()["moves"]
    optional = client.post("/api/checkers/legal", json={**body, "forced": False}).json()["moves"]
    assert len(forced) == 1 and forced[0]["captured"]
    assert len(optional) == 4


@pytest.mark.parametrize("agent", AGENTS)
@pytest.mark.parametrize("size", [10, 12])
@pytest.mark.parametrize("forced", [True, False])
def test_every_agent_answers_on_bigger_boards(client, agent, size, forced):
    body = {"board": list(start_position(size)), "size": size, "turn": -1, "forced": forced,
            "agent": agent, "level": 1}
    r = client.post("/api/checkers/move", json=body)
    assert r.status_code == 200, r.text
    legal = {tuple(m.path) for m in legal_moves(start_position(size), -1, forced)}
    assert tuple(r.json()["move"]["path"]) in legal
    assert r.json()["analysis"]["agent"] == agent


def test_meta_lists_every_size_and_agent(client):
    body = client.get("/api/checkers/meta").json()
    assert [s["size"] for s in body["sizes"]] == [8, 10, 12]
    assert [s["squares"] for s in body["sizes"]] == [32, 50, 72]
    assert {a["name"] for a in body["agents"]} == set(AGENTS)
    assert body["forced"] is True


def test_chance_agent_and_describe_endpoint(client):
    r = client.post("/api/checkers/move", json={"board": list(START), "turn": -1, "agent": "chance"})
    assert r.status_code == 200
    assert tuple(r.json()["move"]["path"]) in {tuple(m.path) for m in legal_moves(START, -1)}
    text = client.get("/api/checkers/describe", params={"red": "mcts", "red_level": 3, "white": "alphabeta",
                                                        "white_level": 1, "size": 10}).json()["text"]
    assert text == "Red: MCTS (1,000 playouts) vs White: alpha-beta search (iterative deepening, 0.2 s)"
    human = client.get("/api/checkers/describe", params={"white": "chance"}).json()["text"]
    assert human == "White: Chance (fixed odds)"  # red defaults to a human
