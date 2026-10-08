import random

import pytest
from fastapi.testclient import TestClient

from server.app import app


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


def cities(n, seed=0):
    rng = random.Random(seed)
    return [[rng.random(), rng.random()] for _ in range(n)]


@pytest.mark.parametrize("algorithm", ["nearest_neighbor_2opt", "simulated_annealing", "genetic_algorithm"])
def test_solve_returns_tour_frames_and_optimum(client, algorithm):
    body = client.post("/api/routes/solve", json={
        "cities": cities(9), "algorithm": algorithm,
        "params": {"iterations": 5000, "generations": 40, "seed": 1},
    }).json()
    assert sorted(body["tour"]) == list(range(9))
    assert body["frames"] and body["optimal"]["length"] <= body["length"] + 1e-9


def test_no_optimum_for_large_maps(client):
    body = client.post("/api/routes/solve", json={"cities": cities(20), "algorithm": "nearest_neighbor_2opt"}).json()
    assert body["optimal"] is None


@pytest.mark.parametrize("payload", [
    {"cities": [[0.1, 0.1], [0.2, 0.2]]},                              # too few
    {"cities": cities(61)},                                            # too many
    {"cities": cities(5), "params": {"iterations": 10**7}},            # over the limit
])
def test_rejects_bad_requests(client, payload):
    assert client.post("/api/routes/solve", json=payload).status_code == 422


def test_rejects_out_of_range_coordinates(client):
    assert client.post("/api/routes/solve", json={"cities": [[0, 0], [2, 0], [0, 1]]}).status_code == 400
