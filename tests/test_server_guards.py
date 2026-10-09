"""Request guards: body size cap, NaN/Infinity handling, client keys, inner list bounds, and WebSocket frames."""

import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from server.app import app
from server.limits import MAX_BODY_BYTES, BodyLimit, RateLimiter, client_key


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


def guarded_app(seen: list) -> FastAPI:
    """A tiny app behind BodyLimit. The route records the body it received, so tests can see whether it ran."""
    api = FastAPI()
    api.add_middleware(BodyLimit)

    @api.post("/echo")
    async def echo(request: Request):
        seen.append(await request.body())
        return {"ok": True}

    return api


# --- body size ---------------------------------------------------------------------------------------------------


def test_oversized_body_is_413_and_never_reaches_the_route():
    seen = []
    with TestClient(guarded_app(seen)) as c:
        r = c.post("/echo", content=b"x" * (MAX_BODY_BYTES + 1))
    assert r.status_code == 413
    assert seen == []


def test_body_at_the_limit_passes():
    seen = []
    with TestClient(guarded_app(seen)) as c:
        r = c.post("/echo", content=b"x" * MAX_BODY_BYTES)
    assert r.status_code == 200
    assert len(seen[0]) == MAX_BODY_BYTES


def test_chunked_oversized_body_is_413():
    # No Content-Length header: the limit has to be enforced on the bytes as they stream in.
    def stream():
        for _ in range(64):  # 64 x 64 KB = 4 MB, well over the limit
            yield b"x" * 65536

    seen = []
    with TestClient(guarded_app(seen)) as c:
        r = c.post("/echo", content=stream())
    assert r.status_code == 413
    assert seen == []


def test_oversized_json_on_a_real_route_is_413(client):
    r = client.post("/api/connect4/move", content=b'{"moves": "' + b"1" * (MAX_BODY_BYTES + 10) + b'"}')
    assert r.status_code == 413
    assert "256 KB" in r.json()["detail"]


# --- NaN and Infinity ----------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "path, body",
    [
        ("/api/connect4/move", b'{"moves": "", "level": NaN}'),
        ("/api/connect4/move", b'{"moves": "", "level": Infinity}'),
        ("/api/tetris/choose", b'{"board": [], "piece": "I", "weights": [NaN]}'),
        ("/api/routes/solve", b'{"cities": [[Infinity, 0], [0, 0], [1, 1]]}'),
        ("/api/routes/solve", b'{"cities": [[1e400, 0], [0, 0], [1, 1]]}'),  # overflows to inf in the parser
        ("/api/2048/move", b'{"grid": [[0, 0, 0, NaN], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}'),
    ],
)
def test_non_finite_json_is_422_not_500(client, path, body):
    r = client.post(path, content=body, headers={"content-type": "application/json"})
    assert r.status_code == 422
    assert r.json()["detail"][0]["loc"] == ["body"]
    assert "finite" in r.json()["detail"][0]["msg"] or "NaN" in r.json()["detail"][0]["msg"]


@pytest.mark.parametrize("query", ["nan", "inf", "-inf"])
def test_non_finite_query_number_is_422_not_500(client, query):
    # A bad query value fails validation, and FastAPI's default error echoes the NaN back, which used to give a 500.
    r = client.get("/api/rover/map", params={"size": 21, "density": query, "seed": 4})
    assert r.status_code == 422
    assert r.json()["detail"][0]["loc"] == ["query", "density"]


def test_finite_json_still_passes(client):
    r = client.post("/api/connect4/move", json={"moves": "", "level": 1})
    assert r.status_code == 200


# --- client keys ---------------------------------------------------------------------------------------------------


def _conn_scope(peer: str, headers: dict[str, str]) -> dict:
    return {
        "type": "http",
        "client": (peer, 40000),
        "headers": [(k.lower().encode(), v.encode()) for k, v in headers.items()],
    }


def test_fly_client_ip_is_believed_only_from_fly_proxy_peers():
    spoof = {"fly-client-ip": "203.0.113.9"}
    # Fly's private 6PN address and the 172.16/12 range may vouch for the header.
    assert client_key(Request(_conn_scope("fdaa:0:1234::5", spoof))) == "203.0.113.9"
    assert client_key(Request(_conn_scope("172.19.0.7", spoof))) == "203.0.113.9"
    # A direct client's header is ignored: the key stays the socket peer.
    assert client_key(Request(_conn_scope("198.51.100.4", spoof))) == "198.51.100.4"
    assert client_key(Request(_conn_scope("198.51.100.4", {}))) == "198.51.100.4"


def test_fly_proxy_without_the_header_uses_the_peer():
    assert client_key(Request(_conn_scope("fdaa::1", {}))) == "fdaa::1"


def test_rotated_spoofed_headers_from_one_direct_client_share_a_bucket():
    api = FastAPI()
    api.state.rate = RateLimiter(3, per=60.0)

    @api.get("/limited")
    async def limited(request: Request):
        return {"allowed": api.state.rate.allow(client_key(request))}

    with TestClient(api, client=("198.51.100.4", 50000)) as c:
        codes = [c.get("/limited", headers={"fly-client-ip": f"10.0.0.{i}"}).json()["allowed"] for i in range(10)]
    assert codes == [True, True, True] + [False] * 7


def test_rotated_spoofed_headers_from_a_fly_peer_are_each_a_client():
    # Behind Fly, the header is the real client, so each distinct client gets its own bucket.
    api = FastAPI()
    api.state.rate = RateLimiter(1, per=60.0)

    @api.get("/limited")
    async def limited(request: Request):
        return {"allowed": api.state.rate.allow(client_key(request))}

    with TestClient(api, client=("fdaa::1", 50000)) as c:
        codes = [c.get("/limited", headers={"fly-client-ip": f"10.0.0.{i}"}).json()["allowed"] for i in range(3)]
    assert codes == [True, True, True]


# --- inner list bounds ---------------------------------------------------------------------------------------------


def test_2048_row_longer_than_four_is_422(client):
    r = client.post("/api/2048/move", json={"grid": [[0, 0, 0, 0, 0], [0] * 4, [0] * 4, [0] * 4]})
    assert r.status_code == 422
    assert r.json()["detail"][0]["loc"][:3] == ["body", "grid", 0]


def test_2048_grid_of_four_by_four_is_accepted(client):
    r = client.post("/api/2048/move", json={"grid": [[0, 2, 0, 0], [0] * 4, [0] * 4, [0] * 4]})
    assert r.status_code == 200


def test_nonogram_clue_longer_than_a_25_line_can_hold_is_422(client):
    # A 25-cell line holds at most 13 runs.
    r = client.post("/api/nonogram/solve", json={"row_clues": [[1] * 14], "col_clues": [[1]]})
    assert r.status_code == 422
    assert r.json()["detail"][0]["loc"][:2] == ["body", "row_clues"]


def test_nonogram_grid_row_longer_than_the_largest_puzzle_is_422(client):
    r = client.post("/api/nonogram/hint", json={"row_clues": [[1]], "col_clues": [[1]], "grid": [[0] * 26]})
    assert r.status_code == 422
    assert r.json()["detail"][0]["loc"][:2] == ["body", "grid"]


def test_snake_body_cell_must_be_an_xy_pair(client):
    r = client.post("/api/snake/hint", json={"body": [[1, 2], [3, 4], [5, 6, 7]], "heading": 0})
    assert r.status_code == 422
    assert r.json()["detail"][0]["loc"][:3] == ["body", "body", 2]


# --- WebSocket frames ----------------------------------------------------------------------------------------------


def test_bad_websocket_frames_get_an_error_and_the_socket_stays_open(client):
    with client.websocket_connect("/ws/npuzzle") as ws:
        for frame in ("{bad json", '{"board":[1]} xx'):
            ws.send_text(frame)
            msg = ws.receive_json()
            assert msg["type"] == "error" and msg["message"].startswith("bad JSON")
        ws.send_bytes(b"\x00\x01\x02")
        assert ws.receive_json() == {"type": "error", "message": "send a JSON text message"}
        ws.send_text('{"board": [NaN]}')
        assert ws.receive_json()["type"] == "error"
        ws.send_text("[1, 2]")  # valid JSON, not a request: the model rejects it
        assert ws.receive_json()["type"] == "error"
        ws.send_text("x" * (MAX_BODY_BYTES + 1))
        assert ws.receive_json() == {"type": "error", "message": "message is too large"}
        # The socket still answers a real request after all of that.
        ws.send_json({"type": "cancel"})
        ws.send_json({"board": [8, 1, 3, 4, 5, 2, 6, 0, 7], "algorithm": "bfs"})
        msg = ws.receive_json()
        while msg["type"] not in ("result", "error"):  # the search streams start and progress messages first
            msg = ws.receive_json()
        assert msg["type"] == "result"


def test_npuzzle_board_longer_than_16_is_rejected_over_the_socket(client):
    with client.websocket_connect("/ws/npuzzle") as ws:
        ws.send_json({"board": list(range(17)), "algorithm": "bfs"})
        msg = ws.receive_json()
    assert msg["type"] == "error"
    assert "16" in msg["message"]
