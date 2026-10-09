"""Chance, the fixed-odds shooter: its table, legal play, seeding, the empirical odds, the API and the race CLI."""

import math
import random
import time
from collections import Counter

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from battleship import match
from battleship.agents import AGENTS, CHANCE_PROFILE, CHANCE_WEIGHTS, ChanceAgent, chance_odds, make_agent
from battleship.board import CELLS, SIZE, Knowledge, random_fleet
from battleship.game import play_out
from server import battleship_api
from server.limits import RateLimiter, SearchSlots


@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(battleship_api.router)
    app.state.move_rate = RateLimiter(10_000, per=60.0)
    app.state.slots = SearchSlots(2, wait_seconds=5.0)
    with TestClient(app) as c:
        yield c


def test_the_table_has_one_weight_per_cell_and_is_built_from_the_profile():
    assert len(CHANCE_WEIGHTS) == CELLS
    assert all(w > 0 for w in CHANCE_WEIGHTS)
    for c in range(CELLS):
        assert CHANCE_WEIGHTS[c] == CHANCE_PROFILE[c // SIZE] * CHANCE_PROFILE[c % SIZE]
    # The middle of the board is weighted above the corners, as the docstring says.
    assert CHANCE_WEIGHTS[44] > CHANCE_WEIGHTS[0]


def test_the_odds_sum_to_one_over_the_unknown_cells():
    k = Knowledge()
    assert math.isclose(sum(chance_odds(k)), 1.0)
    for c in (0, 44, 99, 55):
        k.observe(c, "miss")
    odds = chance_odds(k)
    assert math.isclose(sum(odds), 1.0)
    assert all(odds[c] == 0.0 for c in (0, 44, 99, 55))


def test_only_legal_cells_are_chosen_over_a_whole_game():
    rng = random.Random(3)
    for _ in range(20):
        fleet = random_fleet(rng)
        shots = play_out(ChanceAgent(random.Random(rng.random())), fleet)
        assert 17 <= shots <= CELLS  # at least the 17 ship cells; play_out raises if a cell were fired twice


def test_a_seeded_chance_agent_repeats_its_shots_exactly():
    def shots(seed):
        agent = make_agent("chance", random.Random(seed))
        k = Knowledge()
        out = []
        for _ in range(30):
            c = agent.choose(k)
            out.append(c)
            k.observe(c, "miss")
        return out

    assert shots(11) == shots(11)
    assert shots(11) != shots(12)


def test_the_empirical_frequencies_match_the_table_within_four_standard_errors():
    draws = 200_000
    agent = ChanceAgent(random.Random(5))
    k = Knowledge()
    counts = Counter(agent.choose(k) for _ in range(draws))
    total = sum(CHANCE_WEIGHTS)
    for c in range(CELLS):
        p = CHANCE_WEIGHTS[c] / total
        se = math.sqrt(p * (1 - p) / draws)
        assert abs(counts[c] / draws - p) <= 4 * se, (c, counts[c] / draws, p)


def test_registered_under_the_chance_key_with_its_label():
    assert AGENTS["chance"] is ChanceAgent
    assert ChanceAgent.name == "chance"
    assert ChanceAgent.label == "Chance (fixed odds)"


def test_meta_returns_the_profile_and_the_rule(client):
    body = client.get("/api/battleship/meta").json()
    assert body["chance"]["name"] == "chance"
    assert body["chance"]["label"] == "Chance (fixed odds)"
    assert body["chance"]["profile"] == list(CHANCE_PROFILE)
    assert "profile[row] * profile[col]" in body["chance"]["rule"]


def test_agent_shot_with_chance_returns_a_legal_cell_and_the_fixed_odds(client):
    body = client.post("/api/battleship/agent-shot", json={"shots": [], "agent": "chance", "seed": 4}).json()
    assert 0 <= body["choice_index"] < CELLS
    assert body["method"] == "fixed odds"
    assert len(body["probs"]) == CELLS
    assert math.isclose(sum(body["probs"]), 1.0, abs_tol=1e-3)


def test_agent_shot_with_chance_is_deterministic_for_a_seed_and_history(client):
    shots = [{"cell": "A1", "result": "miss"}, {"cell": "F6", "result": "hit"}]
    first = client.post("/api/battleship/agent-shot", json={"shots": shots, "agent": "chance", "seed": 9}).json()
    again = client.post("/api/battleship/agent-shot", json={"shots": shots, "agent": "chance", "seed": 9}).json()
    assert first["choice_index"] == again["choice_index"]


def test_agent_shot_with_chance_never_picks_a_fired_cell(client):
    shots = [{"cell": cell, "result": "miss"} for cell in ("E5", "F5", "E6", "F6", "A1")]
    body = client.post("/api/battleship/agent-shot", json={"shots": shots, "agent": "chance", "seed": 1}).json()
    assert body["choice"] not in {"E5", "F5", "E6", "F6", "A1"}
    assert body["probs"][battleship_api.parse_cell("E5")] == 0.0


@pytest.mark.parametrize("payload", [
    {"shots": [], "agent": "hunt"},
    {"shots": [], "agent": "chance", "seed": -1},
    {"shots": [], "agent": "chance", "seed": 2**31},
    {"shots": [], "agent": "chance", "seed": "x"},
])
def test_agent_shot_rejects_bad_chance_options(client, payload):
    assert client.post("/api/battleship/agent-shot", json=payload).status_code == 422


def test_agent_shot_with_chance_still_refuses_a_finished_game(client):
    # Eight ships' worth of sunk reports (the full fleet) leaves nothing afloat, so the request is refused.
    shots = []
    for cells, _length in (([0, 1, 2, 3, 4], 5), ([10, 11, 12, 13], 4), ([20, 21, 22], 3),
                          ([30, 31, 32], 3), ([40, 41], 2)):
        names = [battleship_api.cell_name(c) for c in cells]
        shots += [{"cell": n, "result": "hit"} for n in names[:-1]]
        shots.append({"cell": names[-1], "result": "sunk", "ship": {"cells": names}})
    body = client.post("/api/battleship/agent-shot", json={"shots": shots, "agent": "chance"})
    assert body.status_code == 400


def test_the_race_is_decided_by_whichever_fleet_is_sunk_first():
    # The AI shoots first here, so when it wins it has taken one more shot than chance, and when chance wins
    # the two counts are equal (chance finished on its own shot, which came after the AI's).
    for seed in range(8):
        result = match.race("probability", random.Random(seed), ai_first=True)
        assert result.ai_shots >= 17 and result.chance_shots >= 17
        if result.winner == "ai":
            assert result.ai_shots == result.chance_shots + 1
        else:
            assert result.ai_shots == result.chance_shots


def test_the_race_cli_prints_the_algorithms_and_a_result_quickly(capsys):
    start = time.perf_counter()
    assert match.main(["--ai", "hunt", "--games", "20", "--seed", "1"]) == 0
    assert time.perf_counter() - start < 10
    out = capsys.readouterr().out
    assert "Chance (fixed odds)" in out
    assert "Hunt/target" in out
    assert "20 races, seed 1" in out


def test_the_race_is_reproducible_from_its_seed():
    a = match.match("hunt", 10, seed=3)
    b = match.match("hunt", 10, seed=3)
    assert a["wins"] == b["wins"] and a["ai_shots"] == b["ai_shots"] and a["chance_shots"] == b["chance_shots"]


def test_the_terminal_game_accepts_chance_as_its_opponent():
    # You quit on the first prompt, so the opponent only has to be constructible and named in the opening line.
    from battleship.__main__ import play

    lines = []
    outcome = play("chance", random.Random(2), read=lambda prompt: "q", write=lines.append)
    assert outcome == "quit"
    assert "chance" in lines[0]
