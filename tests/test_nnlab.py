"""Neural network lab: gradient check, learning, determinism, data and the CLI."""

import numpy as np
import pytest

from nnlab.cli import main
from nnlab.data import build_split, generate
from nnlab.mlp import ACTIVATIONS, Net, Optimizer, init_net, loss_and_grads, predict, train_epoch
from nnlab.rng import Rng
from nnlab.train import TrainConfig, train


def _random_net(act, sizes=(2, 3, 2, 1), seed=0):
    net = init_net(list(sizes), act, Rng(seed))
    # Zero biases would put some ReLU pre-activations exactly at the kink (z = 0), where the
    # derivative is undefined and central differences disagree with the convention a > 0.
    gen = np.random.default_rng(seed + 100)
    net.b = [gen.normal(scale=0.1, size=b.shape) for b in net.b]
    return net


@pytest.mark.parametrize("act", ACTIVATIONS)
def test_gradients_match_finite_differences(act):
    """Backprop must agree with central differences on every weight and bias, with L2 switched on.

    Central differences are accurate to about eps^2, so with eps = 1e-6 a relative error above 1e-6
    means a wrong formula, not rounding. The inputs are random, so ReLU kinks are not hit exactly.
    """
    rng = np.random.default_rng(1)
    X = rng.normal(size=(6, 2))
    y = rng.integers(0, 2, size=(6, 1)).astype(float)
    net = _random_net(act)
    l2 = 0.05
    _, _, gW, gb = loss_and_grads(net, X, y, l2)

    def total_loss():
        data, reg, _, _ = loss_and_grads(net, X, y, l2)
        return data + reg

    eps = 1e-6
    for li in range(len(net.W)):
        for arr, grad in ((net.W[li], gW[li]), (net.b[li], gb[li])):
            it = np.nditer(arr, flags=["multi_index"])
            for _ in it:
                idx = it.multi_index
                orig = arr[idx]
                arr[idx] = orig + eps
                up = total_loss()
                arr[idx] = orig - eps
                down = total_loss()
                arr[idx] = orig
                numeric = (up - down) / (2 * eps)
                assert abs(numeric - grad[idx]) <= 1e-6 * max(1.0, abs(numeric)), (act, li, idx)


def test_loss_decreases_under_gradient_descent():
    X, y, _, _ = build_split(generate("blobs", 80, 4), ["x", "y"], 0.0, 0.0)
    net = _random_net("tanh", sizes=(2, 4, 1), seed=2)
    opt = Optimizer("sgd", net, lr=0.5)
    first = loss_and_grads(net, X, y, 0.0)[0]
    for _ in range(50):
        _, _, gW, gb = loss_and_grads(net, X, y, 0.0)
        opt.step(net, gW, gb)
    assert loss_and_grads(net, X, y, 0.0)[0] < first


def test_learns_xor_to_high_accuracy():
    run = train(TrainConfig(data="xor", n=200, hidden=(4,), act="tanh", epochs=400, test_frac=0.0,
                            lr=0.03, batch=16))
    assert run.history.train_acc[-1] >= 0.98


def test_learns_circles_with_defaults():
    run = train(TrainConfig())  # the lab's default configuration
    assert run.history.test_acc[-1] >= 0.95
    assert run.history.train_loss[-1] < run.history.train_loss[0]


def test_same_seed_gives_identical_runs():
    cfg = TrainConfig(data="spiral", epochs=20, seed=7)
    a = train(cfg).history
    b = train(cfg).history
    assert a.train_loss == b.train_loss
    assert a.test_acc == b.test_acc


def test_different_seeds_give_different_weights():
    a = init_net([2, 4, 1], "tanh", Rng(1))
    b = init_net([2, 4, 1], "tanh", Rng(2))
    assert not np.allclose(a.W[0], b.W[0])


def test_init_scale_follows_fan_in_and_fan_out():
    net = init_net([2, 8, 1], "relu", Rng(3))
    bound_hidden = np.sqrt(6.0 / 2)  # He: fan_in = 2
    assert np.max(np.abs(net.W[0])) <= bound_hidden
    assert np.all(net.b[0] == 0.0)
    bound_out = np.sqrt(6.0 / (8 + 1))  # Glorot on the output layer
    assert np.max(np.abs(net.W[1])) <= bound_out


def test_presets_are_seeded_and_labelled():
    a = generate("moons", 50, seed=9)
    b = generate("moons", 50, seed=9)
    assert a == b
    assert {p["label"] for p in a} == {0, 1}
    assert len(generate("xor", 40, 0)) == 40
    with pytest.raises(ValueError):
        generate("nope", 10, 0)


def test_test_fraction_splits_cleanly():
    pts = generate("blobs", 100, 0)
    Xtr, ytr, Xte, yte = build_split(pts, ["x", "y"], 0.0, 0.0)
    assert Xte.shape[0] == 0 and Xtr.shape[0] == 100
    Xtr, ytr, Xte, yte = build_split(pts, ["x", "y"], 0.0, 1.0)
    assert Xtr.shape[0] == 0 and Xte.shape[0] == 100
    Xtr, ytr, Xte, yte = build_split(pts, ["x", "y"], 0.0, 0.3)
    assert Xtr.shape[0] + Xte.shape[0] == 100
    assert 0 < Xte.shape[0] < 100


def test_noise_moves_points_by_noise_times_jitter():
    p = generate("circles", 10, 0)
    X0, *_ = build_split(p, ["x", "y"], 0.0, 0.0)
    X1, *_ = build_split(p, ["x", "y"], 0.5, 0.0)
    expected = np.array([[q["x"] + 0.5 * q["jx"], q["y"] + 0.5 * q["jy"]] for q in p])
    assert np.allclose(X1, expected)
    assert not np.allclose(X0, X1)


def test_minibatch_sizes_cover_every_point_once():
    """One epoch with batch = n must equal one full-batch step, and a short last batch must not be dropped."""
    X, y, _, _ = build_split(generate("blobs", 10, 1), ["x", "y"], 0.0, 0.0)
    net_a = init_net([2, 3, 1], "tanh", Rng(5))
    net_b = init_net([2, 3, 1], "tanh", Rng(5))
    opt_a = Optimizer("sgd", net_a, 0.1)
    opt_b = Optimizer("sgd", net_b, 0.1)
    train_epoch(net_a, opt_a, X, y, Rng(6), 0, 0.0)  # full batch: one step
    _, _, gW, gb = loss_and_grads(net_b, X, y, 0.0)
    opt_b.step(net_b, gW, gb)
    assert np.allclose(net_a.W[0], net_b.W[0])
    assert opt_a.t == 1
    train_epoch(net_a, opt_a, X, y, Rng(6), 4, 0.0)  # batches of 4, 4, 2
    assert opt_a.t == 1 + 3


def test_predict_is_a_probability():
    net = _random_net("sigmoid")
    p = predict(net, np.random.default_rng(0).normal(size=(9, 2)))
    assert p.shape == (9, 1)
    assert np.all((p > 0) & (p < 1))


def test_cli_train_prints_curve_and_boundary(capsys):
    code = main(["train", "--data", "xor", "--epochs", "20", "--every", "10", "--layers", "4"])
    out = capsys.readouterr().out
    assert code == 0
    assert "train acc" in out and "test acc" in out
    assert "epoch" in out
    assert "#" in out or "+" in out  # the boundary drawing
    assert "O" in out or "o" in out  # the points


def test_cli_rejects_unknown_feature(capsys):
    with pytest.raises(SystemExit):
        main(["train", "--features", "x,z", "--epochs", "1"])


def test_cli_rejects_zero_width_layer(capsys):
    assert main(["train", "--layers", "0,4", "--epochs", "1", "--no-ascii"]) == 2


@pytest.mark.parametrize(
    "argv",
    [
        ["--n", "0"],
        ["--n", "1"],
        ["--test-frac", "1.0"],
        ["--test-frac", "0"],
        ["--test-frac", "nan"],
        ["--epochs", "-1"],
        ["--batch", "-2"],
    ],
)
def test_cli_rejects_splits_and_counts_that_cannot_train(argv, capsys):
    with pytest.raises(SystemExit) as exit_info:
        main(["train", *argv, "--epochs", "1", "--no-ascii"])
    assert exit_info.value.code == 2
    assert "error" in capsys.readouterr().err


def test_net_sizes_property():
    net = Net(W=[np.zeros((3, 2)), np.zeros((1, 3))], b=[np.zeros(3), np.zeros(1)], act="tanh")
    assert net.sizes == [2, 3, 1]
