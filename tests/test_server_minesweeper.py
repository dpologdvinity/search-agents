import random

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from minesweeper.board import UNKNOWN, Game
from server import minesweeper_api
from server.limits import RateLimiter, SearchSlots


def make_app(rate: int = 120) -> FastAPI:
    """A minimal app with just this router, set up the way server.app's lifespan sets it up.

    The router is not registered in server.app here, so the tests mount it directly. Each test gets
    its own limiter, so one test's requests cannot throttle another.
    """
    app = FastAPI()
    app.include_router(minesweeper_api.router)
    app.state.move_rate = RateLimiter(rate, per=60.0)
    app.state.slots = SearchSlots(2, wait_seconds=5.0)
    return app


@pytest.fixture
def client():
    with TestClient(make_app()) as c:
        yield c


def opened_board(seed: int = 4, preset=(9, 9, 10)):
    """A real position: the first click made and the board as a player would see it."""
    rows, cols, mines = preset
    rng = random.Random(seed)
    g = Game(rows, cols, mines, rng)
    g.reveal(rng.randrange(rows * cols))
    return g


def payload(g: Game, level: int = 3) -> dict:
    v = g.view()
    return {"rows": v.rows, "cols": v.cols, "mines": v.mines, "cells": list(v.cells), "level": level}


def test_meta_lists_presets_and_agents(client):
    body = client.get("/api/minesweeper/meta").json()
    assert [p["name"] for p in body["presets"]] == ["beginner", "intermediate", "expert"]
    assert body["max_rows"] == 16 and body["max_cols"] == 30
    assert [a["name"] for a in body["agents"]] == ["random", "rules", "csp", "probability"]


def test_analyze_returns_proofs_and_probabilities(client):
    g = opened_board()
    body = client.post("/api/minesweeper/analyze", json=payload(g)).json()
    size = g.rows * g.cols
    assert len(body["probability"]) == size
    for i, v in enumerate(g.view().cells):
        if v != UNKNOWN:
            assert body["probability"][i] is None
    for c in body["safe"]:
        assert body["probability"][c] == 0.0
    for c in body["certain_mines"]:
        assert body["probability"][c] == 1.0
        assert c not in body["safe"]
    # Nothing on the page is a secret: the proofs are about the public view only, and they are sound.
    assert set(body["safe"]).isdisjoint(g.mine_set())
    assert set(body["certain_mines"]) <= g.mine_set()
    if not body["safe"]:
        assert body["guess"] is not None and body["guess"]["cell"] in range(size)
    assert all("layouts_log2" in c for c in body["components"])


def test_level_one_reports_no_probabilities_for_unproven_cells(client):
    g = opened_board(seed=9)
    body = client.post("/api/minesweeper/analyze", json=payload(g, level=1)).json()
    assert body["guess"] is None
    assert all(p is None or p in (0.0, 1.0) for p in body["probability"])


def test_inconsistent_board_is_a_bad_request(client):
    body = {"rows": 2, "cols": 2, "mines": 1, "cells": [-1, 2, -1, -1], "level": 3}
    r = client.post("/api/minesweeper/analyze", json=body)
    assert r.status_code == 400


@pytest.mark.parametrize(
    "patch, status",
    [
        ({"cells": [-1] * 80}, 400),  # wrong cell count
        ({"cells": [9] + [-1] * 80}, 400),  # a number above 8
        ({"cells": [-2] + [-1] * 80}, 400),  # below the covered marker
        ({"mines": 81}, 400),  # as many mines as cells
        ({"rows": 1}, 422),  # below the smallest board
        ({"level": 4}, 422),  # no such level
    ],
)
def test_rejects_bad_requests(client, patch, status):
    body = {"rows": 9, "cols": 9, "mines": 10, "cells": [-1] * 81, "level": 3}
    body.update(patch)
    assert client.post("/api/minesweeper/analyze", json=body).status_code == status


def test_analyze_is_rate_limited():
    body = {"rows": 3, "cols": 3, "mines": 1, "cells": [-1] * 9, "level": 1}
    with TestClient(make_app(rate=2)) as c:
        codes = [c.post("/api/minesweeper/analyze", json=body).status_code for _ in range(3)]
    assert codes == [200, 200, 429]
