"""Command line: train a network on a preset and print loss and accuracy, then the decision boundary.

    python -m nnlab train --data spiral --layers 8,8 --act tanh --epochs 500

Options mirror the browser lab: --layers is the list of hidden layer widths (the browser caps it at
4 layers of 8 neurons), --features picks the inputs, --noise jitters the points, and --test-frac holds
points out for the test accuracy.
"""

from __future__ import annotations

import argparse
import sys

from .ascii import render
from .config import DEFAULTS, OPTIMIZERS
from .data import FEATURES, PRESETS
from .mlp import ACTIVATIONS
from .train import TrainConfig, train


def _ints(text: str) -> tuple[int, ...]:
    if text.strip() in ("", "0", "none"):
        return ()
    return tuple(int(t) for t in text.split(","))


def _at_least(low: int):
    """argparse type factory: an integer of at least low (a split of fewer than two points, or negative epochs,
    would give NaN losses or no training at all)."""

    def parse(text: str) -> int:
        value = int(text)
        if value < low:
            raise argparse.ArgumentTypeError(f"must be at least {low}, got {value}")
        return value

    return parse


def _test_fraction(text: str) -> float:
    """argparse type for --test-frac: strictly between 0 and 1, so both the train and test sets are non-empty."""
    value = float(text)
    if not 0 < value < 1:  # also rejects NaN
        raise argparse.ArgumentTypeError(f"must be between 0 and 1 (exclusive), got {text}")
    return value


def _features(text: str) -> tuple[str, ...]:
    names = [t.strip() for t in text.split(",") if t.strip()]
    bad = [t for t in names if t not in FEATURES]
    if bad or not names:
        raise argparse.ArgumentTypeError(f"features must be a list from {', '.join(FEATURES)}")
    return tuple(f for f in FEATURES if f in names)  # canonical order, as in the browser


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m nnlab", description="Neural network playground in the terminal.")
    sub = p.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train", help="train on a preset dataset and report the result")
    t.add_argument("--data", choices=PRESETS, default=DEFAULTS["data"], help="preset dataset")
    t.add_argument("--n", type=_at_least(2), default=DEFAULTS["n"], help="number of points, at least 2")
    t.add_argument("--layers", type=_ints, default=tuple(DEFAULTS["hidden"]),
                   help="hidden layer widths, comma separated, e.g. 8,8 (0 for none)")
    t.add_argument("--act", choices=ACTIVATIONS, default=DEFAULTS["act"], help="hidden activation")
    t.add_argument("--features", type=_features, default=tuple(DEFAULTS["features"]),
                   help="input features, comma separated from " + ", ".join(FEATURES))
    t.add_argument("--epochs", type=_at_least(0), default=DEFAULTS["epochs"])
    t.add_argument("--lr", type=float, default=DEFAULTS["lr"], help="learning rate")
    t.add_argument("--batch", type=_at_least(0), default=DEFAULTS["batch"], help="mini-batch size (0 for full batch)")
    t.add_argument("--l2", type=float, default=DEFAULTS["l2"], help="L2 weight penalty (lambda)")
    t.add_argument("--optimizer", choices=OPTIMIZERS, default=DEFAULTS["optimizer"])
    t.add_argument("--noise", type=float, default=DEFAULTS["noise"], help="jitter added to the points")
    t.add_argument("--test-frac", type=_test_fraction, default=DEFAULTS["test_frac"],
                   help="fraction held out for test, strictly between 0 and 1")
    t.add_argument("--seed", type=int, default=DEFAULTS["seed"])
    t.add_argument("--every", type=int, default=0, help="print a line every N epochs (default: about 10 lines)")
    t.add_argument("--no-ascii", action="store_true", help="skip the decision boundary drawing")
    return p


def cmd_train(args: argparse.Namespace) -> int:
    if any(w < 1 for w in args.layers):
        print("error: layer widths must be >= 1", file=sys.stderr)
        return 2
    cfg = TrainConfig(
        data=args.data, n=args.n, seed=args.seed, noise=args.noise, features=args.features,
        test_frac=args.test_frac, hidden=args.layers, act=args.act, lr=args.lr, batch=args.batch,
        l2=args.l2, optimizer=args.optimizer, epochs=args.epochs,
    )
    every = args.every if args.every > 0 else max(1, cfg.epochs // 10)

    print(f"data={cfg.data} n={cfg.n} features={','.join(cfg.features)} layers={list(cfg.hidden)} "
          f"act={cfg.act} opt={cfg.optimizer} lr={cfg.lr} batch={cfg.batch} l2={cfg.l2} seed={cfg.seed}")
    print(f"{'epoch':>6}  {'train loss':>10}  {'train acc':>9}  {'test loss':>10}  {'test acc':>8}")

    def show(epoch, history):
        if epoch % every == 0 or epoch == cfg.epochs:
            tl, ta = history.train_loss[-1], history.train_acc[-1]
            vl, va = history.test_loss[-1], history.test_acc[-1]
            test = f"{vl:10.4f}  {va:8.3f}" if va == va else f"{'-':>10}  {'-':>8}"
            print(f"{epoch:6d}  {tl:10.4f}  {ta:9.3f}  {test}", flush=True)

    run = train(cfg, on_epoch=show)
    if not args.no_ascii:
        print()
        print(render(run.net, run.points, cfg.features, cfg.noise))
        print("O pink class, o cyan class; # and + predict pink, . and : predict cyan")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.cmd == "train":
        return cmd_train(args)
    return 2
