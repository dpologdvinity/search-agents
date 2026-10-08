import pytest
from fastapi.testclient import TestClient

from server.app import app
from sudoku.solver import is_valid_solution, parse


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


GRID = [[2, 2, 0, 0], [0, 4, 0, 0], [0, 0, 8, 0], [0, 0, 0, 0]]


@pytest.mark.parametrize("agent", ["expectimax", "ntuple", "ntuple_search"])
def test_2048_move(client, agent):
    from game2048.ntuple import WEIGHTS

    r = client.post("/api/2048/move", json={"grid": GRID, "agent": agent})
    if agent.startswith("ntuple") and not WEIGHTS.exists():
        assert r.status_code == 503
        return
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["direction"] in body["values"]
    assert body["depth"] >= 1


@pytest.mark.parametrize("grid, fragment", [
    ([[2, 4, 2, 4], [4, 2, 4, 2], [2, 4, 2, 4], [4, 2, 4, 2]], "game is over"),
    ([[3, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], "invalid tile"),
])
def test_2048_rejects_bad_boards(client, grid, fragment):
    r = client.post("/api/2048/move", json={"grid": grid})
    assert r.status_code == 400 and fragment in r.text


@pytest.mark.parametrize("solver", ["plain", "mrv_fc", "propagate"])
def test_sudoku_solve_with_trace(client, solver):
    puzzle = client.get("/api/sudoku/puzzle", params={"set": "generated", "index": 3}).json()["puzzle"]
    body = client.post("/api/sudoku/solve", json={"puzzle": puzzle, "solver": solver}).json()
    if solver == "plain" and body["status"] == "limit":
        # Plain backtracking can exceed the server's node cap; that is the point of showing it.
        assert body["nodes"] > 0
        return
    assert body["status"] == "solved"
    assert is_valid_solution(parse(body["solution"]), parse(puzzle))
    from server.sudoku_api import TRACE_LIMIT

    if len(body["trace"]) < TRACE_LIMIT:
        assert sum(1 for t in body["trace"] if t[0] == "assign") == body["nodes"]
    else:  # long searches record only the first TRACE_LIMIT steps
        assert len(body["trace"]) == TRACE_LIMIT


def test_sudoku_hard_and_bad_input(client):
    hard = client.get("/api/sudoku/puzzle", params={"set": "hard", "name": "inkala_2012"}).json()
    assert len(hard["puzzle"]) == 81
    assert client.post("/api/sudoku/solve", json={"puzzle": "123"}).status_code == 400
