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


def test_the_impossible_request_is_refused_at_once(monkeypatch):
    """The refusal is decided up front, so it never reaches the sampler (which used to spin for 200,000 tries)."""
    def no_sampling(*args, **kwargs):
        raise AssertionError("the sampler should not run for an impossible request")
    monkeypatch.setattr(endgame_api, "_table", no_sampling)
    with TestClient(make_app()) as c:
        for piece in ("Q", "R"):
            r = c.get("/api/endgame/random", params={"piece": piece, "want": "draw", "stm": 0})
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


def test_meta_lists_both_opponents_and_the_chance_odds():
    with TestClient(make_app()) as c:
        body = c.get("/api/endgame/meta").json()
    assert [o["key"] for o in body["opponents"]] == ["tablebase", "chance"]
    assert body["opponents"][1]["label"] == "Chance (fixed odds)"
    chance = body["chance"]
    assert chance["key"] == "chance"
    weights = {w["category"]: w["weight"] for w in chance["weights"]}
    assert weights == {"capture": 4, "check": 2, "king_centre": 2, "other": 1}


def test_analyze_marks_checks_and_captures():
    # White to move from "8/8/8/5k2/8/8/1Q6/K7": the queen has checks on the file and rank of the black king
    # on f5, and no move is a capture because the black king is not next to the queen.
    with TestClient(make_app()) as c:
        body = c.post("/api/endgame/analyze", json={"fen": "8/8/8/5k2/8/8/1Q6/K7 w - - 0 1"}).json()
    assert all("check" in m for m in body["moves"])
    assert any(m["check"] for m in body["moves"])
    assert not any(m["capture"] for m in body["moves"])
    mated = [m for m in body["moves"] if m["mate"]]
    assert all(m["check"] for m in mated)  # checkmate implies check


def test_analyze_validation_is_422_for_bad_fields_and_400_for_illegal_positions():
    with TestClient(make_app()) as c:
        bad_piece = c.post("/api/endgame/analyze", json={"piece": "K", "wk": 0, "wp": 9, "bk": 36, "stm": 0})
        bad_square = c.post("/api/endgame/analyze", json={"piece": "Q", "wk": 64, "wp": 9, "bk": 36, "stm": 0})
        bad_random = c.get("/api/endgame/random", params={"piece": "Q", "want": "lost"})
        illegal = c.post("/api/endgame/analyze", json={"piece": "Q", "wk": 0, "wp": 9, "bk": 9, "stm": 0})
    assert bad_piece.status_code == 422 and bad_square.status_code == 422 and bad_random.status_code == 422
    assert illegal.status_code == 400
