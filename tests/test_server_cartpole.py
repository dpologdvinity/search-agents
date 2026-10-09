import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from cartpole.agents import load_agent
from cartpole.env import INIT_RANGE, MAX_STEPS, CartPole, run_episode
from server import cartpole_api
from server.limits import RateLimiter, SearchSlots


def make_app(rate: int = 30) -> FastAPI:
    """A minimal app with just this router, set up the way server.app's lifespan sets it up.

    The router is not registered in server.app here, so the tests mount it directly. Each test
    gets its own limiter, so one test's requests cannot throttle another.
    """
    app = FastAPI()
    app.include_router(cartpole_api.router)
    app.state.rate = RateLimiter(rate, per=60.0)
    app.state.move_rate = RateLimiter(rate, per=60.0)
    app.state.slots = SearchSlots(2, wait_seconds=5.0)
    return app


@pytest.fixture
def client():
    with TestClient(make_app()) as c:
        yield c


def test_meta_lists_agents_and_physics(client):
    body = client.get("/api/cartpole/meta").json()
    assert [a["name"] for a in body["agents"]] == ["random", "pd", "reinforce", "actor_critic", "cem"]
    assert body["physics"]["max_steps"] == MAX_STEPS and body["physics"]["tau"] == 0.02
    assert len(body["obs_scale"]) == 4 and body["actions"] == [0, 1]


@pytest.mark.parametrize("name", ["reinforce", "actor_critic", "cem"])
def test_policy_weights_have_the_right_shapes(client, name):
    body = client.get(f"/api/cartpole/policy/{name}").json()
    w = body["weights"]
    if name == "cem":
        assert len(w["theta"]) == 5
    else:
        assert len(w["W1"]) == body["hidden"] and len(w["W1"][0]) == 4
        assert len(w["W2"]) == 2 and len(w["W2"][0]) == body["hidden"]
    if name == "actor_critic":
        assert len(w["v_W2"]) == 1 and "value_scale" in w


def test_unknown_policy_is_404(client):
    assert client.get("/api/cartpole/policy/random").status_code == 404
    assert client.get("/api/cartpole/policy/nope").status_code == 404


def test_curves_and_benchmark_are_served(client):
    curves = client.get("/api/cartpole/curves").json()
    assert {"reinforce", "actor_critic", "cem"} <= set(curves["algorithms"])
    bench = client.get("/api/cartpole/benchmark").json()
    assert bench["max_steps"] == MAX_STEPS and "reinforce" in bench["agents"]


def test_rollout_matches_the_python_physics(client):
    body = client.post("/api/cartpole/rollout", json={"agent": "reinforce", "seed": 12, "steps": 200}).json()
    expected_steps = run_episode(load_agent("reinforce"), 12, max_steps=200)
    assert body["steps"] == expected_steps == len(body["actions"])
    assert len(body["states"]) == body["steps"] + 1
    assert body["fell"] == (expected_steps < 200)


def test_rollout_starts_from_the_python_reset(client):
    # The page verifies from the server's states[0], so it must be the Python start state for that seed.
    body = client.post("/api/cartpole/rollout", json={"agent": "reinforce", "seed": 12, "steps": 5}).json()
    assert body["states"][0] == pytest.approx(list(CartPole(12).reset(12)))
    assert all(abs(v) <= INIT_RANGE for v in body["states"][0])


def test_rollout_rejects_bad_requests(client):
    assert client.post("/api/cartpole/rollout", json={"agent": "bogus"}).status_code == 400
    assert client.post("/api/cartpole/rollout", json={"steps": 501}).status_code == 422


def test_rollout_is_rate_limited():
    with TestClient(make_app(rate=2)) as c:
        codes = [c.post("/api/cartpole/rollout", json={"agent": "pd", "seed": i, "steps": 10}).status_code
                 for i in range(3)]
    assert codes == [200, 200, 429]


def test_rollout_reports_a_fall_on_the_last_allowed_step():
    """An episode that falls exactly on its final step still fell, so the flag comes from the state, not the count."""
    from server.cartpole_api import rollout_json
    n = rollout_json("random", 3, 500)["steps"]
    assert n < 500  # the random agent falls well before the cap
    exact = rollout_json("random", 3, n)
    assert exact["steps"] == n and exact["fell"] is True
    assert rollout_json("random", 3, n - 1)["fell"] is False
