import json

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from bandits.agents import AGENTS, LINEUP
from bandits.env import Environment
from bandits.sim import play, score_pulls
from server import bandits_api
from server.limits import RateLimiter, SearchSlots


def make_app(rate: int = 30) -> FastAPI:
    """A minimal app with just this router, set up the way server.app's lifespan sets it up.

    The router is not registered in server.app here, so the tests mount it directly.
    Each test gets its own rate limiter, so one test's requests cannot throttle another.
    """
    app = FastAPI()
    app.include_router(bandits_api.router)
    app.state.rate = RateLimiter(rate, per=60.0)
    app.state.slots = SearchSlots(2, wait_seconds=5.0)
    return app


@pytest.fixture
def client():
    with TestClient(make_app()) as c:
        yield c


@pytest.fixture
def no_benchmark(tmp_path, monkeypatch):
    """Point the benchmark file at a path that does not exist, so the 404 branch is tested."""
    monkeypatch.setattr(bandits_api, "BENCHMARK", tmp_path / "missing.json")
    bandits_api._benchmark_bytes.cache_clear()
    yield
    bandits_api._benchmark_bytes.cache_clear()


def test_meta_lists_kinds_agents_and_constants(client):
    body = client.get("/api/bandits/meta").json()
    assert body["kinds"] == ["bernoulli", "gaussian", "drifting"]
    keys = [a["key"] for a in body["agents"]]
    assert keys == list(AGENTS)
    assert body["lineup"] == {kind: list(v) for kind, v in LINEUP.items()}
    assert body["constants"]["drift_period"] == 500 and body["constants"]["window"] == 200
    assert body["limits"]["pulls"] == [1, 1000]


def test_machines_are_the_same_as_the_python_casino(client):
    body = client.get("/api/bandits/machines", params={"kind": "drifting", "k": 4, "pulls": 1000, "seed": 9}).json()
    env = Environment("drifting", 4, 1000, 9)
    assert body["period"] == 500 and len(body["schedule"]) == 2
    assert [tuple(seg) for seg in body["schedule"]] == list(env.machines.schedule)
    assert body["best"] == [seg.index(max(seg)) for seg in env.machines.schedule]


def test_machines_validates_its_parameters(client):
    assert client.get("/api/bandits/machines", params={"kind": "roulette"}).status_code == 422
    assert client.get("/api/bandits/machines", params={"k": 1}).status_code == 422
    assert client.get("/api/bandits/machines", params={"pulls": 1001}).status_code == 422


def test_score_matches_the_python_reference(client):
    arms = [i % 5 for i in range(200)]
    body = client.post("/api/bandits/score", json={"kind": "bernoulli", "k": 5, "seed": 3, "arms": arms}).json()
    env = Environment("bernoulli", 5, 200, 3)
    mine = score_pulls(env, arms)
    assert body["pulls"] == 200
    assert body["player"]["regret"] == pytest.approx(round(mine.total_regret, 4))
    assert set(body["agents"]) == set(LINEUP["bernoulli"])
    ucb = play("ucb1", env)
    assert body["agents"]["ucb1"]["regret"] == pytest.approx(round(ucb.total_regret, 4))
    assert body["agents"]["ucb1"]["label"] == AGENTS["ucb1"].label


def test_score_rejects_machines_that_do_not_exist(client):
    response = client.post("/api/bandits/score", json={"kind": "bernoulli", "k": 3, "seed": 1, "arms": [0, 3]})
    assert response.status_code == 400 and "machines 0..2" in response.json()["detail"]


def test_score_rejects_bad_bodies(client):
    assert client.post("/api/bandits/score", json={"arms": []}).status_code == 422
    assert client.post("/api/bandits/score", json={"arms": [0] * 1001}).status_code == 422
    assert client.post("/api/bandits/score", json={"kind": "craps", "arms": [0]}).status_code == 422


def test_score_is_rate_limited():
    with TestClient(make_app(rate=2)) as c:
        body = {"kind": "bernoulli", "k": 3, "seed": 0, "arms": [0, 1]}
        assert c.post("/api/bandits/score", json=body).status_code == 200
        assert c.post("/api/bandits/score", json=body).status_code == 200
        assert c.post("/api/bandits/score", json=body).status_code == 429


def test_benchmark_is_404_until_it_has_been_run(client, no_benchmark):
    assert client.get("/api/bandits/benchmark").status_code == 404
    assert client.get("/api/bandits/meta").json()["benchmark"] is False


def test_benchmark_is_served_when_present(client, tmp_path, monkeypatch):
    path = tmp_path / "bandits_benchmark.json"
    path.write_text(json.dumps({"seeds": 1, "settings": {}}))
    monkeypatch.setattr(bandits_api, "BENCHMARK", path)
    bandits_api._benchmark_bytes.cache_clear()
    try:
        assert client.get("/api/bandits/benchmark").json() == {"seeds": 1, "settings": {}}
    finally:
        bandits_api._benchmark_bytes.cache_clear()


def test_shipped_benchmark_lives_inside_the_package():
    # The Docker image copies the bandits package, so the committed benchmark must sit under it, not in results/.
    assert bandits_api.BENCHMARK.is_file()
    assert bandits_api.BENCHMARK.parent.parent.name == "bandits"
