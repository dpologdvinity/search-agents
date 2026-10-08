import numpy as np
import pytest

torch = pytest.importorskip("torch")

from npuzzle.batch import pdb_batch, scramble
from npuzzle.neural import NeuralHeuristic
from npuzzle.train import ResidualMLP, bellman_targets, export, heuristic


@pytest.fixture
def random_model():
    torch.manual_seed(0)
    model = ResidualMLP(hidden=64, blocks=2)
    for p in model.parameters():
        torch.nn.init.normal_(p, std=0.1)
    return model.eval()


def test_numpy_inference_matches_torch(random_model, tmp_path):
    path = tmp_path / "w.npz"
    export(random_model, path)
    rng = np.random.default_rng(0)
    boards = scramble(300, 4, rng.integers(0, 100, size=300), rng)
    with torch.no_grad():
        expected = heuristic(random_model, boards, pdb_batch(boards)).numpy()
    np.testing.assert_allclose(NeuralHeuristic(path)(boards), expected, atol=1e-4)


def test_heuristic_is_zero_at_goal_and_at_least_pdb(random_model, tmp_path):
    path = tmp_path / "w.npz"
    export(random_model, path)
    rng = np.random.default_rng(1)
    boards = scramble(200, 4, rng.integers(0, 60, size=200), rng)
    h = NeuralHeuristic(path)(boards)
    goal = (boards == np.arange(16)).all(axis=1)
    assert (h[goal] == 0).all()
    assert (h[~goal] >= pdb_batch(boards[~goal])).all()


def test_bellman_target_is_one_next_to_goal(random_model):
    rng = np.random.default_rng(2)
    boards = scramble(20, 4, np.ones(20, dtype=int), rng)
    np.testing.assert_allclose(bellman_targets(random_model, boards), 1.0)
