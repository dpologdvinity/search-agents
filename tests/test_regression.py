"""Tests for the regression lab: solvers, gradient descent, logistic regression, presets, CLI, and JS parity.

The parity test runs web/js/regression-core.js in node on the same inputs as the Python reference and
compares every number to 1e-9. It is skipped when node is missing or the JavaScript file is absent.
The JS core exports camelCase names matching the Python ones, with matrices as arrays of rows:
polyFeatures, polyFeatures2d, solveNormal, solveQR, gradientDescent, logisticGD, makeRegression and
makeClassification. Descent results are {w, losses, diverged}; the presets return {x, y} and {xy, labels}.
"""

import json
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from bandits.rng import Rng
from regression.__main__ import main, monomial_names
from regression.core import (
    accuracy,
    gradient_descent,
    logistic_gd,
    logistic_gradient,
    logistic_loss,
    poly_features,
    poly_features_2d,
    regression_gradient,
    regression_loss,
    sigmoid,
    solve_normal,
    solve_qr,
    train_test_mask,
)
from regression.data import make_classification, make_regression

ROOT = Path(__file__).resolve().parent.parent
CORE_JS = ROOT / "web" / "js" / "regression-core.js"
NODE = shutil.which("node")


def _lines(argv):
    """Run the CLI and collect its output lines; returns (exit code, lines)."""
    out = []
    code = main(argv, out=out.append)
    return code, out


# Solvers -------------------------------------------------------------------------------------------


def test_exact_fit_on_collinear_points():
    x = np.linspace(-0.8, 0.8, 7)
    y = 2.0 * x + 1.0
    X = poly_features(x, 1)
    for solve in (solve_normal, solve_qr):
        np.testing.assert_allclose(solve(X, y), [1.0, 2.0], atol=1e-12)


@pytest.mark.parametrize("degree", [2, 3])
def test_exact_fit_on_polynomial_points(degree):
    # A polynomial of the fitted degree passes through every point, so the residual is zero.
    coef = np.array([0.5, -1.0, 0.25, 0.75])[: degree + 1]
    x = np.linspace(-0.9, 0.9, 9)
    y = poly_features(x, degree) @ coef
    X = poly_features(x, degree)
    for solve in (solve_normal, solve_qr):
        np.testing.assert_allclose(solve(X, y), coef, atol=1e-9)


def test_normal_equations_match_qr_and_lstsq():
    x, y = make_regression("noisy-sine", 30, 7)
    X = poly_features(x, 3)
    w_normal = solve_normal(X, y)
    w_qr = solve_qr(X, y)
    w_lstsq = np.linalg.lstsq(X, y, rcond=None)[0]
    np.testing.assert_allclose(w_normal, w_lstsq, atol=1e-9)
    np.testing.assert_allclose(w_qr, w_lstsq, atol=1e-9)


@pytest.mark.parametrize("ridge", [0.05, 0.5, 3.0])
def test_ridge_normal_equations_match_qr_and_closed_form(ridge):
    x, y = make_regression("quadratic", 30, 3)
    X = poly_features(x, 4)
    D = np.diag([0.0, 1.0, 1.0, 1.0, 1.0])
    closed = np.linalg.solve(X.T @ X + ridge * D, X.T @ y)
    np.testing.assert_allclose(solve_normal(X, y, ridge), closed, atol=1e-9)
    np.testing.assert_allclose(solve_qr(X, y, ridge), closed, atol=1e-9)


def test_huge_ridge_leaves_only_the_intercept_at_the_mean():
    # The intercept is not penalised, so as lam grows every other weight is driven to zero and the
    # intercept settles at the mean of y.
    x, y = make_regression("line", 30, 5)
    X = poly_features(x, 3)
    w = solve_qr(X, y, ridge=1e10)
    assert w[0] == pytest.approx(y.mean(), abs=1e-6)
    np.testing.assert_allclose(w[1:], 0.0, atol=1e-6)


def test_singular_normal_equations_raise():
    with pytest.raises(np.linalg.LinAlgError):
        solve_normal(np.zeros((4, 2)), np.ones(4))


def test_qr_refuses_a_column_that_is_numerically_dependent():
    # The third column is exactly twice the second, but rounding leaves a residual norm of about eps, not zero.
    # Dividing by it used to give enormous weights; the solver now refuses it as rank deficient.
    x = np.linspace(0.0, 1.0, 10)
    X = np.column_stack([np.ones_like(x), x, 2 * x])
    with pytest.raises(np.linalg.LinAlgError):
        solve_qr(X, x)


# Gradient descent ----------------------------------------------------------------------------------


def test_gradient_descent_converges_to_closed_form():
    x, y = make_regression("line", 30, 7)
    X = poly_features(x, 1)
    closed = solve_normal(X, y)
    result = gradient_descent(X, y, lr=0.4, steps=2000)
    assert not result.diverged
    np.testing.assert_allclose(result.w, closed, atol=1e-6)
    assert len(result.losses) == 2001
    assert result.losses[-1] <= result.losses[0]


def test_ridge_gradient_descent_converges_to_ridge_closed_form():
    x, y = make_regression("line", 30, 7)
    X = poly_features(x, 1)
    ridge = 0.5
    closed = solve_qr(X, y, ridge)
    result = gradient_descent(X, y, lr=0.4, steps=2000, ridge=ridge)
    np.testing.assert_allclose(result.w, closed, atol=1e-6)


def test_gradient_descent_reports_divergence():
    # Unnormalised cubic features with a large step overshoot and blow up.
    x, y = make_regression("noisy-sine", 30, 7)
    result = gradient_descent(poly_features(x, 3), y, lr=50.0, steps=500)
    assert result.diverged


def test_logistic_gradient_matches_finite_differences():
    rng = Rng(11)
    X = np.array([[1.0, rng.uniform() * 2 - 1, rng.uniform() * 2 - 1, rng.uniform()] for _ in range(15)])
    labels = np.array([float(rng.uniform() < 0.5) for _ in range(15)])
    w = np.array([0.3, -0.7, 0.4, 0.2])
    for ridge in (0.0, 0.7):
        grad = logistic_gradient(X, labels, w, ridge)
        numeric = np.zeros_like(w)
        h = 1e-6
        for k in range(len(w)):
            e = np.zeros_like(w)
            e[k] = h
            numeric[k] = (logistic_loss(X, labels, w + e, ridge) - logistic_loss(X, labels, w - e, ridge)) / (2 * h)
        np.testing.assert_allclose(grad, numeric, atol=1e-6)


def test_regression_gradient_matches_finite_differences():
    x, y = make_regression("quadratic", 20, 2)
    X = poly_features(x, 3)
    w = np.array([0.1, -0.2, 0.3, 0.4])
    grad = regression_gradient(X, y, w, ridge=0.2)
    numeric = np.zeros_like(w)
    h = 1e-6
    for k in range(len(w)):
        e = np.zeros_like(w)
        e[k] = h
        numeric[k] = (regression_loss(X, y, w + e, 0.2) - regression_loss(X, y, w - e, 0.2)) / (2 * h)
    np.testing.assert_allclose(grad, numeric, atol=1e-6)


def test_logistic_gd_reduces_log_loss_on_separable_blobs():
    xy, labels = make_classification("blobs", 60, 7)
    X = poly_features_2d(xy, 1)
    result = logistic_gd(X, labels, lr=0.5, steps=500)
    assert not result.diverged
    assert result.losses[-1] < result.losses[0]
    assert accuracy(X, labels, result.w) > 0.95


# Features, splits, and presets ---------------------------------------------------------------------


def test_poly_features_2d_order_and_names():
    xy = np.array([[0.5, -2.0]])
    np.testing.assert_allclose(poly_features_2d(xy, 2), [[1.0, 0.5, -2.0, 0.25, -1.0, 4.0]])
    assert monomial_names(2) == ["1", "x1", "x2", "x1^2", "x1x2", "x2^2"]
    assert poly_features_2d(xy, 3).shape[1] == len(monomial_names(3)) == 10


def test_train_test_mask_holds_out_every_fourth_point():
    mask = train_test_mask(12)
    assert list(np.flatnonzero(mask)) == [3, 7, 11]


def test_sigmoid_is_stable_for_large_arguments():
    z = np.array([-1000.0, 0.0, 1000.0])
    p = sigmoid(z)
    assert np.all(np.isfinite(p))
    np.testing.assert_allclose(p, [0.0, 0.5, 1.0], atol=1e-12)


def test_logistic_loss_is_finite_for_confident_wrong_predictions():
    X = np.array([[1.0, 800.0]])
    assert np.isfinite(logistic_loss(X, [0.0], [0.0, 1.0]))


def test_presets_are_reproducible_and_seed_dependent():
    a = make_regression("noisy-sine", 30, 7)
    b = make_regression("noisy-sine", 30, 7)
    c = make_regression("noisy-sine", 30, 8)
    np.testing.assert_array_equal(a[0], b[0])
    np.testing.assert_array_equal(a[1], b[1])
    assert not np.allclose(a[1], c[1])
    for name in ("moons", "blobs"):
        xy1, l1 = make_classification(name, 40, 7)
        xy2, l2 = make_classification(name, 40, 7)
        np.testing.assert_array_equal(xy1, xy2)
        np.testing.assert_array_equal(l1, l2)
        assert list(l1[:4]) == [0, 1, 0, 1]


def test_presets_reject_unknown_names():
    with pytest.raises(ValueError):
        make_regression("cubic", 10, 1)
    with pytest.raises(ValueError):
        make_classification("spiral", 10, 1)


# Command line --------------------------------------------------------------------------------------


def test_cli_fit_prints_coefficients_errors_and_plot():
    code, out = _lines(["fit", "--data", "noisy-sine", "--degree", "5", "--method", "qr"])
    assert code == 0
    text = "\n".join(out)
    for k in range(6):
        assert f"w{k} = " in text
    assert "train MSE" in text and "test MSE" in text
    assert any(line.startswith("+") for line in out)


def test_cli_fit_is_deterministic():
    argv = ["fit", "--data", "quadratic", "--degree", "2", "--method", "gd", "--steps", "100"]
    assert _lines(argv) == _lines(argv)


def test_cli_gd_reports_steps_and_divergence():
    _, out = _lines(["fit", "--method", "gd", "--steps", "50", "--lr", "0.1"])
    assert any(line.startswith("gd: steps 50") and "diverged no" in line for line in out)


def test_cli_logistic_prints_accuracy_and_map():
    code, out = _lines(["logistic", "--data", "moons", "--steps", "500"])
    assert code == 0
    text = "\n".join(out)
    assert "train accuracy" in text and "test accuracy" in text
    assert any("#" in line for line in out) and any("." in line for line in out)
    assert _lines(["logistic", "--data", "moons", "--steps", "500"]) == (code, out)


def test_cli_refuses_fits_with_more_coefficients_than_training_points():
    # n = 5 leaves 4 training points (every fourth point is held out), so degree 6 is underdetermined.
    code, out = _lines(["fit", "--n", "5", "--degree", "6", "--method", "normal"])
    assert code == 2
    assert out == [
        "error: degree 6 needs 7 coefficients but there are only 4 training points; use a degree of at most 3"
    ]


def test_cli_sweep_checks_the_degree_before_printing_the_table():
    code, out = _lines(["sweep", "--n", "4", "--max-degree", "12"])
    assert code == 2
    assert len(out) == 1 and out[0].startswith("error:")


@pytest.mark.parametrize("argv", [["fit", "--degree", "-1"], ["sweep", "--max-degree", "0"], ["fit", "--n", "3"]])
def test_cli_rejects_negative_or_empty_settings(argv, capsys):
    with pytest.raises(SystemExit) as exit_info:
        main(argv)
    assert exit_info.value.code == 2
    assert "must be at least" in capsys.readouterr().err


def test_cli_sweep_shows_overfitting_at_high_degree():
    _, out = _lines(["sweep", "--data", "noisy-sine", "--max-degree", "10"])
    rows = [line.split() for line in out[2:]]
    assert [int(r[0]) for r in rows] == list(range(1, 11))
    train = [float(r[1]) for r in rows]
    test = [float(r[2]) for r in rows]
    # Polynomial models are nested, so least-squares train error can only fall as the degree rises.
    assert all(b <= a + 1e-12 for a, b in zip(train, train[1:], strict=False))
    # Past the sweet spot the test error climbs back up while the train error keeps falling.
    assert test[-1] > 1.2 * min(test)
    assert test[-1] > 3 * train[-1]


# JavaScript parity ---------------------------------------------------------------------------------

NODE_PROGRAM = r"""
import fs from 'node:fs';
const src = fs.readFileSync(process.env.REG_CORE, 'utf8');
const core = await import('data:text/javascript;base64,' + Buffer.from(src).toString('base64'));
const c = JSON.parse(process.env.REG_CASE);
const res = {};
res.polyFeatures = core.polyFeatures(c.x, 3);
res.polyFeatures2d = core.polyFeatures2d(c.xy, 2);
res.solveNormal = core.solveNormal(res.polyFeatures, c.y, 0);
res.solveQR = core.solveQR(res.polyFeatures, c.y, 0);
res.solveNormalRidge = core.solveNormal(res.polyFeatures, c.y, 0.3);
res.solveQRRidge = core.solveQR(res.polyFeatures, c.y, 0.3);
const gd = core.gradientDescent(res.polyFeatures, c.y, 0.1, 200, 0.1);
res.gdW = gd.w; res.gdLosses = gd.losses; res.gdDiverged = gd.diverged;
const lg = core.logisticGD(core.polyFeatures2d(c.xy, 2), c.labels, 0.5, 100, 0.01);
res.logW = lg.w; res.logLosses = lg.losses;
res.regressionPreset = core.makeRegression('noisy-sine', 30, 7, 0.15);
res.classPreset = core.makeClassification('moons', 40, 7, 0.1);
res.blobPreset = core.makeClassification('blobs', 40, 7, 0.22);
console.log(JSON.stringify(res));
"""


def _flat(v):
    """Flatten nested lists of numbers (and booleans) into a float array for comparison."""
    return np.asarray(v, dtype=float).ravel()


@pytest.mark.skipif(NODE is None, reason="node is not installed")
@pytest.mark.skipif(not CORE_JS.exists(), reason="web/js/regression-core.js is not present")
def test_js_core_matches_python_to_1e9():
    x = np.linspace(-0.9, 0.9, 12)
    y = 0.8 * np.sin(2.0 * np.pi * x) + 0.05 * np.cos(9 * x)
    xy, labels = make_classification("moons", 40, 7, 0.1)
    case = {"x": x.tolist(), "y": y.tolist(), "xy": xy.tolist(), "labels": labels.tolist()}

    X = poly_features(x, 3)
    gd = gradient_descent(X, y, 0.1, 200, ridge=0.1)
    lg = logistic_gd(poly_features_2d(xy, 2), labels, 0.5, 100, ridge=0.01)
    xr, yr = make_regression("noisy-sine", 30, 7, 0.15)
    xc, lc = make_classification("blobs", 40, 7, 0.22)
    expected = {
        "polyFeatures": X,
        "polyFeatures2d": poly_features_2d(xy, 2),
        "solveNormal": solve_normal(X, y, 0.0),
        "solveQR": solve_qr(X, y, 0.0),
        "solveNormalRidge": solve_normal(X, y, 0.3),
        "solveQRRidge": solve_qr(X, y, 0.3),
        "gdW": gd.w,
        "gdLosses": gd.losses,
        "logW": lg.w,
        "logLosses": lg.losses,
    }
    env = dict(os.environ, REG_CORE=str(CORE_JS), REG_CASE=json.dumps(case))
    proc = subprocess.run(
        [NODE, "--input-type=module", "-e", NODE_PROGRAM],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, proc.stderr
    js = json.loads(proc.stdout)

    for key, value in expected.items():
        np.testing.assert_allclose(_flat(js[key]), _flat(value), rtol=0, atol=1e-9, err_msg=key)
    assert js["gdDiverged"] is gd.diverged
    np.testing.assert_allclose(_flat(js["regressionPreset"]["x"]), xr, rtol=0, atol=1e-12)
    np.testing.assert_allclose(_flat(js["regressionPreset"]["y"]), yr, rtol=0, atol=1e-12)
    np.testing.assert_allclose(_flat(js["blobPreset"]["xy"]), xc.ravel(), rtol=0, atol=1e-12)
    assert list(js["blobPreset"]["labels"]) == list(lc)
