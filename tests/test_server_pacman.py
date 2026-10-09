"""HTTP tests for the Pac-Man API.

The app here mounts only the Pac-Man router, so the tests do not depend on server/app.py
registering it. The router reads the search-slot semaphore from app.state, set below as
the real app does.
"""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from pacman import mazes
from pacman.agents import make_agent
from pacman.engine import Game
from pacman.episode import run_episode
from pacman.features import FEATURE_NAMES
from pacman.mazes import ACTIONS
from server.limits import SearchSlots
from server.pacman_api import router


@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(router)
    app.state.slots = SearchSlots(2, wait_seconds=5.0)
    with TestClient(app) as c:
        yield c


def test_meta_lists_mazes_agents_features_and_weights(client):
    body = client.get("/api/pacman/meta").json()
    assert [m["name"] for m in body["mazes"]] == list(mazes.names())
    assert [a["name"] for a in body["agents"]] == ["q", "reflex", "random"]
    assert [f["name"] for f in body["features"]] == list(FEATURE_NAMES)
    assert body["weights"] is not None
    assert len(body["weights"]["weights"]) == len(FEATURE_NAMES)
    assert body["constants"]["max_turns"] >= 100


def test_q_episode_matches_the_in_process_game(client):
    body = client.post("/api/pacman/episode", json={"agent": "q", "maze": "vault", "seed": 3}).json()
    expected = run_episode(make_agent("q"), mazes.get("vault"), 3, "q")
    assert body["result"]["score"] == expected.score
    assert body["result"]["status"] == expected.final.status
    assert len(body["turns"]) == len(expected.turns)
    assert len(body["weights"]) == len(FEATURE_NAMES)


def test_episode_turns_move_one_step_and_carry_q_values(client):
    body = client.post("/api/pacman/episode", json={"agent": "q", "maze": "lanes", "seed": 1}).json()
    prev = body["start"]["pac"]
    m = mazes.get("lanes")
    for turn in body["turns"]:
        pac = turn["pac"]
        assert abs(pac[0] - prev[0]) + abs(pac[1] - prev[1]) == 1
        assert set(turn["q"]) <= {"N", "E", "S", "W"} and turn["q"][turn["action"]] == max(turn["q"].values())
        assert len(turn["features"][turn["action"]]) == len(FEATURE_NAMES)
        assert all(0 <= r < m.height and 0 <= c < m.width for r, c in (pac, *(g["pos"] for g in turn["ghosts"])))
        prev = pac


def test_random_episode_has_no_q_values(client):
    body = client.post("/api/pacman/episode", json={"agent": "random", "maze": "vault", "seed": 9}).json()
    assert body["turns"] and all(t["q"] is None for t in body["turns"])


def test_episode_respects_max_turns(client):
    req = {"agent": "random", "maze": "lanes", "seed": 2, "max_turns": 10}
    body = client.post("/api/pacman/episode", json=req).json()
    assert len(body["turns"]) <= 10


def test_play_replays_moves_and_reports_legal_next(client):
    start = client.post("/api/pacman/play", json={"maze": "lanes", "seed": 0, "actions": []}).json()
    assert start["last"] is None and start["state"]["turn"] == 0
    assert "E" in start["legal"]
    after = client.post("/api/pacman/play", json={"maze": "lanes", "seed": 0, "actions": ["E"]}).json()
    assert after["state"]["turn"] == 1
    assert after["state"]["pac"] == [start["state"]["pac"][0], start["state"]["pac"][1] + 1]
    assert after["last"]["action"] == "E"


def test_play_rejects_walls_and_bad_input(client):
    # East, then north, puts Pac-Man at row 2 column 8; west from there is a wall.
    res = client.post("/api/pacman/play", json={"maze": "lanes", "seed": 0, "actions": ["E", "N", "W"]})
    assert res.status_code == 400 and "move 3 (W)" in res.json()["detail"]
    assert client.post("/api/pacman/play", json={"maze": "lanes", "seed": 0, "actions": ["X"]}).status_code == 422
    assert client.post("/api/pacman/play", json={"maze": "nope", "seed": 0, "actions": []}).status_code == 400
    bad_agent = client.post("/api/pacman/episode", json={"agent": "ghost", "maze": "lanes", "seed": 0})
    assert bad_agent.status_code == 422


def test_meta_lists_both_ghost_policies_and_the_chance_odds(client):
    body = client.get("/api/pacman/meta").json()
    policies = {p["name"]: p for p in body["ghost_policies"]}
    assert set(policies) == {"ai", "chance"}
    assert policies["chance"]["label"] == "Chance (fixed odds)"
    assert policies["ai"]["label"] == "AI ghosts (A* routes)"
    assert body["chance_odds"] == {"straight": 0.6, "left": 0.15, "right": 0.15, "back": 0.1}
    assert sum(body["chance_odds"].values()) == pytest.approx(1.0)


def test_chance_episode_matches_the_in_process_game(client):
    req = {"agent": "random", "maze": "vault", "seed": 5, "ghosts": "chance"}
    body = client.post("/api/pacman/episode", json=req).json()
    expected = run_episode(make_agent("random"), mazes.get("vault"), 5, "random", ghosts="chance")
    assert body["ghosts"] == "chance" and body["ghosts_label"] == "Chance (fixed odds)"
    assert body["result"]["score"] == expected.score
    assert body["result"]["status"] == expected.final.status
    assert len(body["turns"]) == len(expected.turns)
    planned = [p for t in body["turns"] for p in t["plans"]]
    assert planned and all(p["target"] is None and p["path"] == [] for p in planned)


def test_episode_defaults_to_the_ai_ghosts(client):
    body = client.post("/api/pacman/episode", json={"agent": "random", "maze": "lanes", "seed": 2}).json()
    assert body["ghosts"] == "ai"
    assert any(t["plans"][0]["target"] is not None for t in body["turns"])


def test_play_replays_against_chance_ghosts(client):
    req = {"maze": "lanes", "seed": 0, "actions": ["E", "N"], "ghosts": "chance"}
    body = client.post("/api/pacman/play", json=req).json()
    game = Game(mazes.get("lanes"), seed=0, ghosts="chance")
    game.step(ACTIONS.index("E"))
    game.step(ACTIONS.index("N"))
    assert body["ghosts"] == "chance"
    assert body["state"]["ghosts"] == [
        {"pos": [g.pos // game.maze.width, g.pos % game.maze.width], "personality": g.personality, "scared": g.scared}
        for g in game.state.ghosts
    ]


@pytest.mark.parametrize("path, body", [
    ("/api/pacman/episode", {"agent": "random", "maze": "lanes", "seed": 0, "ghosts": "wild"}),
    ("/api/pacman/play", {"maze": "lanes", "seed": 0, "actions": [], "ghosts": "wild"}),
])
def test_unknown_ghost_policy_is_rejected(client, path, body):
    assert client.post(path, json=body).status_code == 422
