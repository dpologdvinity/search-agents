from fastapi import FastAPI
from fastapi.testclient import TestClient

from endgame.rules import parse_square
from server import endgame_api
from server.limits import RateLimiter


def make_app(rate: int = 60) -> FastAPI:
    """A minimal app with just this router, with the move rate limiter set up as server.app's lifespan does.

    The router is not registered in server.app here, so the tests mount it directly.
    """
    app = FastAPI()
    app.include_router(endgame_api.router)
    app.state.move_rate = RateLimiter(rate, per=60.0)
    return app


def test_meta_reports_both_tables():
    with TestClient(make_app()) as c:
        body = c.get("/api/endgame/meta").json()
    q, r = body["pieces"]["Q"], body["pieces"]["R"]
    assert q["name"] == "KQK" and r["name"] == "KRK"
    assert q["max_win_moves"] == 10 and r["max_win_moves"] == 16
    assert q["legal"] == 368452 and r["legal"] == 399112
    assert sum(q["win_histogram"]) == q["won"] and len(r["loss_histogram"]) == 17


def test_random_position_is_wanted_and_analysed():
    with TestClient(make_app()) as c:
        body = c.get("/api/endgame/random", params={"piece": "R", "want": "strong_wins"}).json()
    assert body["piece"] == "R" and body["value"]["outcome"] in ("win", "loss")
    assert body["moves"] and all("san" in m and "uci" in m for m in body["moves"])
    assert body["best"] is not None


def test_random_draw_with_white_to_move_is_refused():
    with TestClient(make_app()) as c:
        r = c.get("/api/endgame/random", params={"piece": "Q", "want": "draw", "stm": 0})
    assert r.status_code == 400


def test_analyze_by_fen_and_by_squares_agree():
    fen = "8/8/8/5k2/8/8/1Q6/K7 w - - 0 1"
    squares = {"piece": "Q", "wk": parse_square("a1"), "wp": parse_square("b2"),
               "bk": parse_square("f5"), "stm": 0}
    with TestClient(make_app()) as c:
        by_fen = c.post("/api/endgame/analyze", json={"fen": fen}).json()
        by_sq = c.post("/api/endgame/analyze", json=squares).json()
    assert by_fen == by_sq
    assert by_fen["value"] == {"outcome": "win", "plies": 19, "moves": 10}
    best = next(m for m in by_fen["moves"] if m["uci"] == by_fen["best"]["uci"])
    assert best["mate_in"] == 10  # the table's move is one of the fastest wins
    assert any(m["outcome"] == "draw" for m in by_fen["moves"])  # Qe5 and Qf6 hang the queen


def test_analyze_rejects_illegal_positions():
    with TestClient(make_app()) as c:
        touching = c.post("/api/endgame/analyze", json={"piece": "Q", "wk": 36, "wp": 0, "bk": 37, "stm": 0})
        partial = c.post("/api/endgame/analyze", json={"piece": "Q", "wk": 36})
        bad_fen = c.post("/api/endgame/analyze", json={"fen": "8/8/8/8/8/8/8/8 w"})
    assert touching.status_code == 400 and partial.status_code == 400 and bad_fen.status_code == 400


def test_heatmap_has_64_squares_and_nulls_for_overlaps():
    with TestClient(make_app()) as c:
        body = c.get("/api/endgame/heatmap", params={"piece": "Q", "wk": 0, "wp": 9, "stm": 1}).json()
    assert len(body["outcome"]) == 64 and len(body["plies"]) == 64
    assert body["outcome"][0] is None and body["outcome"][9] is None  # the white king and piece squares
    # Black to move never wins in KQK. The queen on b2 is defended by the king on a1, so Black cannot take it
    # here, and every legal square is a loss for Black.
    assert set(body["outcome"]) == {None, "loss"}


def test_analyze_is_rate_limited():
    with TestClient(make_app(rate=2)) as c:
        codes = [c.post("/api/endgame/analyze", json={"piece": "R", "wk": 0, "wp": 9, "bk": 36, "stm": 1}).status_code
                 for _ in range(3)]
    assert codes[:2] == [200, 200] and codes[2] == 429
