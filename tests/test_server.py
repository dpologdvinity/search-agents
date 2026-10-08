import pytest
from fastapi.testclient import TestClient

from npuzzle import bfs
from server.app import app
from server.limits import RateLimiter
from tests.puzzles import solves


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


def solve(ws, request):
    ws.send_json(request)
    messages = []
    while True:
        msg = ws.receive_json()
        messages.append(msg)
        if msg["type"] in ("result", "error"):
            return messages


def test_health_and_meta(client):
    assert client.get("/api/health").json() == {"ok": True}
    meta = client.get("/api/npuzzle/meta").json()
    assert {a["name"] for a in meta["algorithms"]} >= {"bfs", "astar", "idastar", "bwas"}
    assert {h["name"] for h in meta["heuristics"]} >= {"manhattan", "pdb"}


def test_scramble(client):
    board = client.get("/api/npuzzle/scramble", params={"n": 4, "moves": 20, "seed": 1}).json()["board"]
    assert sorted(board) == list(range(16))
    assert client.get("/api/npuzzle/scramble", params={"n": 5}).status_code == 400


@pytest.mark.parametrize("algorithm", ["bfs", "astar", "idastar", "bibfs"])
def test_streams_search_and_result(client, algorithm):
    board = [3, 1, 2, 4, 7, 5, 6, 0, 8]
    with client.websocket_connect("/ws/npuzzle") as ws:
        messages = solve(ws, {"board": board, "algorithm": algorithm})
    assert messages[0]["type"] == "start"
    result = messages[-1]
    assert result["type"] == "result" and result["status"] == "solved"
    assert result["cost"] == bfs(tuple(board)).cost
    assert solves(tuple(board), result["path"])
    traced = sum(len(m["expanded"]) for m in messages if m["type"] == "progress")
    assert traced == min(result["expanded"], 3000)


def test_bwas_with_pdb_on_4x4(client):
    board = client.get("/api/npuzzle/scramble", params={"n": 4, "moves": 40, "seed": 2}).json()["board"]
    with client.websocket_connect("/ws/npuzzle") as ws:
        result = solve(ws, {"board": board, "algorithm": "bwas", "heuristic": "pdb"})[-1]
    assert result["status"] == "solved"
    assert solves(tuple(board), result["path"])


@pytest.mark.parametrize("request_body, fragment", [
    ({"board": [1, 2, 3], "algorithm": "bfs"}, "9 or 16"),
    ({"board": [0, 1, 2, 3, 4, 5, 6, 7, 8], "algorithm": "nope"}, "Input should be"),
    ({"board": [0, 1, 2, 3, 4, 5, 6, 7, 8], "algorithm": "astar", "heuristic": "pdb"}, "4x4"),
])
def test_rejects_bad_requests(client, request_body, fragment):
    with client.websocket_connect("/ws/npuzzle") as ws:
        msg = solve(ws, request_body)[-1]
    assert msg["type"] == "error" and fragment in msg["message"]


def test_unsolvable_board(client):
    with client.websocket_connect("/ws/npuzzle") as ws:
        result = solve(ws, {"board": [1, 0, 2, 3, 4, 5, 6, 8, 7], "algorithm": "astar"})[-1]
    assert result["status"] == "unsolvable"


def test_time_limit_and_cancel(client):
    hard = client.get("/api/npuzzle/scramble", params={"n": 4, "seed": 3}).json()["board"]
    with client.websocket_connect("/ws/npuzzle") as ws:
        result = solve(ws, {"board": hard, "algorithm": "bfs", "max_seconds": 0.5})[-1]
        assert result["status"] == "limit"
        ws.send_json({"board": hard, "algorithm": "bfs"})
        assert ws.receive_json()["type"] == "start"
        ws.send_json({"type": "cancel"})
        while (msg := ws.receive_json())["type"] != "result":
            pass
        assert msg["status"] == "cancelled"
        # The socket is still usable after a cancel.
        assert solve(ws, {"board": [3, 1, 2, 0, 4, 5, 6, 7, 8], "algorithm": "bfs"})[-1]["cost"] == 1


def test_rate_limiter():
    limiter = RateLimiter(rate=3, per=60)
    assert [limiter.allow("a") for _ in range(4)] == [True, True, True, False]
    assert limiter.allow("b")


def test_node_limits_bound_memory():
    from server.npuzzle_api import MAX_NODES, MAX_NODES_LINEAR_MEMORY, node_limit

    assert node_limit("astar") == node_limit("bfs") == node_limit("bwas") == MAX_NODES
    assert node_limit("idastar") == node_limit("ids") == MAX_NODES_LINEAR_MEMORY
