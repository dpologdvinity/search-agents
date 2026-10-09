"""Regression from the command line: fit a polynomial, classify with a sigmoid, or sweep the degree.

    python -m regression fit --data noisy-sine --degree 5 --method qr
    python -m regression fit --data noisy-sine --degree 5 --method gd --lr 0.1 --steps 500 --ridge 0.01
    python -m regression logistic --data moons --steps 500
    python -m regression sweep --data noisy-sine --max-degree 10

Every run is deterministic: the seed fixes the points, and nothing else is random. The output shows the
coefficients, the train and test errors, and an ASCII plot, so the overfitting gap is visible in the
terminal. Every fourth point (indices 3, 7, 11, ...) is held out as the test set, so the test points are spread
across the whole x range (see train_test_mask).
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from .core import (
    accuracy,
    gradient_descent,
    logistic_gd,
    logistic_loss,
    mse,
    poly_features,
    poly_features_2d,
    solve_normal,
    solve_qr,
    train_test_mask,
)
from .data import (
    CLASSIFICATION_PRESETS,
    REGRESSION_NOISE,
    REGRESSION_PRESETS,
    make_classification,
    make_regression,
)
from .plot import curve_plot, decision_plot

METHODS = {"normal": solve_normal, "qr": solve_qr}


def monomial_names(degree: int) -> list[str]:
    """Names of the columns of poly_features_2d, in the same order: 1, x1, x2, x1^2, x1x2, x2^2, ..."""
    names = []
    for t in range(degree + 1):
        for i in range(t, -1, -1):
            j = t - i
            if t == 0:
                names.append("1")
                continue
            parts = []
            if i:
                parts.append("x1" if i == 1 else f"x1^{i}")
            if j:
                parts.append("x2" if j == 1 else f"x2^{j}")
            names.append("".join(parts))
    return names


def _int_at_least(low: int):
    """argparse type factory: an integer of at least low. Negative degrees gave tracebacks, and fewer than four
    points leave no train or test split."""

    def parse(text: str) -> int:
        value = int(text)
        if value < low:
            raise argparse.ArgumentTypeError(f"must be at least {low}, got {value}")
        return value

    return parse


def _too_many_coefficients(degree: int, n_train: int) -> str | None:
    """A message when a polynomial of this degree has more coefficients than training points, else None.

    Such a fit is underdetermined: the solver gives weights that fit the training points exactly and mean nothing
    (the old output printed twenty coefficients for five points), so the command refuses it instead."""
    if degree + 1 <= n_train:
        return None
    return (
        f"error: degree {degree} needs {degree + 1} coefficients but there are only {n_train} training points; "
        f"use a degree of at most {n_train - 1}"
    )


def _fmt(v: float) -> str:
    """A signed number with six decimals, so two runs can be compared by eye."""
    return f"{v: .6f}"


def cmd_fit(args, out) -> int:
    """Fit a polynomial of one variable on the training split and report both errors."""
    x, y = make_regression(args.data, args.n, args.seed, args.noise)
    mask = train_test_mask(args.n)
    X = poly_features(x, args.degree)
    Xtr, ytr = X[~mask], y[~mask]
    problem = _too_many_coefficients(args.degree, len(ytr))
    if problem:
        out(problem)
        return 2
    if args.method == "gd":
        result = gradient_descent(Xtr, ytr, args.lr, args.steps, ridge=args.ridge)
        w = result.w
    else:
        w = METHODS[args.method](Xtr, ytr, args.ridge)
        result = None

    out(f"regression fit: data {args.data}, degree {args.degree}, method {args.method}, ridge {args.ridge:g}")
    out(
        f"points: n {args.n} (train {int((~mask).sum())}, test {int(mask.sum())}), seed {args.seed}, "
        f"noise {args.noise:g}"
    )
    for k, wk in enumerate(w):
        out(f"w{k} = {_fmt(wk)}")
    out(f"train MSE {_fmt(mse(Xtr, ytr, w))}")
    out(f"test MSE  {_fmt(mse(X[mask], y[mask], w))}")
    if result is not None:
        final = result.losses[-1] if result.losses else float("nan")
        out(
            f"gd: steps {args.steps}, lr {args.lr:g}, final loss {final:.6f}, "
            f"diverged {'yes' if result.diverged else 'no'}"
        )
    for line in curve_plot(x, y, mask, w, args.degree):
        out(line)
    return 0


def cmd_logistic(args, out) -> int:
    """Fit a sigmoid classifier by gradient descent on the log-loss and report accuracy and the map."""
    xy, labels = make_classification(args.data, args.n, args.seed, args.noise)
    mask = train_test_mask(args.n)
    X = poly_features_2d(xy, args.degree)
    Xtr, ltr = X[~mask], labels[~mask]
    result = logistic_gd(Xtr, ltr, args.lr, args.steps, ridge=args.ridge)
    w = result.w

    out(f"logistic regression: data {args.data}, degree {args.degree}, ridge {args.ridge:g}")
    out(f"points: n {args.n} (train {int((~mask).sum())}, test {int(mask.sum())}), seed {args.seed}")
    names = monomial_names(args.degree)
    for name, wk in zip(names, w, strict=True):
        out(f"{name:>6} = {_fmt(wk)}")
    out(f"train log-loss {logistic_loss(Xtr, ltr, w, args.ridge):.6f}")
    out(f"train accuracy {accuracy(Xtr, ltr, w):.3f}")
    out(f"test accuracy  {accuracy(X[mask], labels[mask], w):.3f}")
    out(
        f"gd: steps {args.steps}, lr {args.lr:g}, final loss {result.losses[-1]:.6f}, "
        f"diverged {'yes' if result.diverged else 'no'}"
    )
    for line in decision_plot(xy, labels, w, args.degree):
        out(line)
    return 0


def cmd_sweep(args, out) -> int:
    """Train and test MSE for every degree from 1 to max-degree: the overfitting curve in one table."""
    x, y = make_regression(args.data, args.n, args.seed, args.noise)
    mask = train_test_mask(args.n)
    problem = _too_many_coefficients(args.max_degree, int((~mask).sum()))
    if problem:
        out(problem)
        return 2  # checked before the table starts, so a bad --max-degree prints nothing partial
    out(
        f"degree sweep: data {args.data}, n {args.n} (train {int((~mask).sum())}, test {int(mask.sum())}), "
        f"seed {args.seed}, ridge {args.ridge:g}"
    )
    out(f"{'degree':>6}  {'train MSE':>12}  {'test MSE':>12}")
    for degree in range(1, args.max_degree + 1):
        X = poly_features(x, degree)
        w = solve_qr(X[~mask], y[~mask], args.ridge)
        out(f"{degree:>6}  {mse(X[~mask], y[~mask], w):>12.6f}  {mse(X[mask], y[mask], w):>12.6f}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    """The three subcommands and their options. The defaults match the page's starting values."""
    parser = argparse.ArgumentParser(prog="python -m regression", description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)

    fit = sub.add_parser("fit", help="least squares or gradient descent on a polynomial")
    fit.add_argument("--data", choices=REGRESSION_PRESETS, default="noisy-sine")
    fit.add_argument("--degree", type=_int_at_least(0), default=3)
    fit.add_argument("--method", choices=("normal", "qr", "gd"), default="qr")
    fit.add_argument("--n", type=_int_at_least(4), default=30, help="points, at least 4 (every fourth is a test point)")
    fit.add_argument("--seed", type=int, default=7)
    fit.add_argument("--noise", type=float, default=REGRESSION_NOISE)
    fit.add_argument("--ridge", type=float, default=0.0)
    fit.add_argument("--lr", type=float, default=0.1)
    fit.add_argument("--steps", type=int, default=500)
    fit.set_defaults(run=cmd_fit)

    lg = sub.add_parser("logistic", help="sigmoid classifier by gradient descent on the log-loss")
    lg.add_argument("--data", choices=CLASSIFICATION_PRESETS, default="moons")
    lg.add_argument("--n", type=int, default=60)
    lg.add_argument("--seed", type=int, default=7)
    lg.add_argument("--noise", type=float, default=None, help="default: 0.1 for moons, 0.22 for blobs")
    lg.add_argument("--degree", type=_int_at_least(0), default=1)
    lg.add_argument("--ridge", type=float, default=0.0)
    lg.add_argument("--lr", type=float, default=0.5)
    lg.add_argument("--steps", type=int, default=500)
    lg.set_defaults(run=cmd_logistic)

    sw = sub.add_parser("sweep", help="train and test MSE against polynomial degree")
    sw.add_argument("--data", choices=REGRESSION_PRESETS, default="noisy-sine")
    sw.add_argument("--max-degree", type=_int_at_least(1), default=10)
    sw.add_argument("--n", type=_int_at_least(4), default=30, help="points, at least 4 (every fourth is a test point)")
    sw.add_argument("--seed", type=int, default=7)
    sw.add_argument("--noise", type=float, default=REGRESSION_NOISE)
    sw.add_argument("--ridge", type=float, default=0.0)
    sw.set_defaults(run=cmd_sweep)
    return parser


def main(argv=None, out=print) -> int:
    """Parse argv and run one subcommand, writing each output line through `out`. Returns an exit code.

    A singular design (for example the normal equations with more parameters than distinct x values)
    is reported as an error line, not a traceback.
    """
    args = build_parser().parse_args(argv)
    try:
        return args.run(args, out)
    except np.linalg.LinAlgError as err:
        out(f"error: {err}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
