"""Command line for the lab: grow one CART tree, or train a random forest, on a seeded preset.

    python -m treelab grow --data xor --depth 4                 # the tree as rules, accuracy, region map
    python -m treelab grow --data nested --criterion entropy --prune 0.02
    python -m treelab forest --data spiral --trees 50           # OOB and test accuracy, feature importances

Train and test rows come from one seeded shuffle, so the same flags print the same numbers every time.
"""

from __future__ import annotations

import argparse

import numpy as np

from .data import FEATURE_NAMES, PRESETS, expand, make_preset, train_test_split
from .forest import RandomForest
from .tree import CRITERIA, DecisionTree, n_leaves, prune, rules_text

CLASS_SYMBOLS = "ABCD"
GRID_COLS, GRID_ROWS = 48, 20


def _at_least(low: int):
    """argparse type factory: an integer of at least low. Zero points, a zero-leaf minimum or zero trees gave
    IndexError, ValueError or NaN output."""

    def parse(text: str) -> int:
        value = int(text)
        if value < low:
            raise argparse.ArgumentTypeError(f"must be at least {low}, got {value}")
        return value

    return parse


def _float_at_least(low: float):
    """argparse type factory: a number of at least low (NaN is rejected too)."""

    def parse(text: str) -> float:
        value = float(text)
        if not value >= low:
            raise argparse.ArgumentTypeError(f"must be at least {low:g}, got {text}")
        return value

    return parse


def _test_fraction(text: str) -> float:
    """argparse type for --test: strictly between 0 and 1, so the train and test sets are both non-empty."""
    value = float(text)
    if not 0 < value < 1:  # also rejects NaN
        raise argparse.ArgumentTypeError(f"must be between 0 and 1 (exclusive), got {text}")
    return value


def _data(args: argparse.Namespace):
    """Return expanded train and test matrices with their labels, and the number of classes."""
    X, y = make_preset(args.data, args.n, args.seed)
    tr, te = train_test_split(len(y), args.test, seed=args.seed)
    F = expand(X, extras=not args.no_extras)
    return F[tr], y[tr], F[te], y[te], PRESETS[args.data]


def _names(args: argparse.Namespace) -> tuple[str, ...]:
    return FEATURE_NAMES if not args.no_extras else FEATURE_NAMES[:2]


def _region_map(predict, extras: bool) -> str:
    """ASCII picture of the plane: each cell shows the class letter the model predicts there."""
    xs = (np.arange(GRID_COLS) + 0.5) / GRID_COLS
    ys = 1.0 - (np.arange(GRID_ROWS) + 0.5) / GRID_ROWS  # top row is y near 1
    gx, gy = np.meshgrid(xs, ys)
    labels = predict(expand(np.column_stack([gx.ravel(), gy.ravel()]), extras=extras))
    rows = [("".join(CLASS_SYMBOLS[c] for c in labels[r * GRID_COLS:(r + 1) * GRID_COLS]))
            for r in range(GRID_ROWS)]
    return "\n".join("|" + row + "|" for row in rows)


def cmd_grow(args: argparse.Namespace) -> None:
    Xtr, ytr, Xte, yte, k = _data(args)
    tree = DecisionTree(args.criterion, args.depth, args.min_leaf).fit(Xtr, ytr, k)
    root = tree.root
    if args.prune > 0:
        root = prune(root, args.prune)
    names = _names(args)
    print(f"preset {args.data}  seed {args.seed}  train {len(ytr)}  test {len(yte)}  "
          f"criterion {args.criterion}  depth<={args.depth}  min_leaf {args.min_leaf}")
    print(rules_text(root, names, args.criterion))
    shown = DecisionTree()
    shown.root, shown.n_classes = root, k
    leaves = n_leaves(root)
    print()
    print(f"leaves {leaves}  (unpruned {tree.n_leaves()})" if args.prune > 0 else f"leaves {leaves}")
    print(f"train accuracy {np.mean(shown.predict(Xtr) == ytr):.3f}")
    print(f"test accuracy  {np.mean(shown.predict(Xte) == yte):.3f}")
    print()
    print("region map (letters are the predicted class):")
    print(_region_map(shown.predict, not args.no_extras))


def cmd_forest(args: argparse.Namespace) -> None:
    Xtr, ytr, Xte, yte, k = _data(args)
    forest = RandomForest(args.trees, criterion=args.criterion, max_depth=args.depth,
                          min_samples_leaf=args.min_leaf, seed=args.seed).fit(Xtr, ytr, k)
    single = DecisionTree(args.criterion, 12, 1).fit(Xtr, ytr, k)
    print(f"preset {args.data}  seed {args.seed}  trees {args.trees}  train {len(ytr)}  test {len(yte)}  "
          f"criterion {args.criterion}  depth<={args.depth}  min_leaf {args.min_leaf}")
    print(f"OOB accuracy   {forest.oob_accuracy:.3f}" if forest.oob_accuracy is not None
          else "OOB accuracy   n/a")
    print(f"test accuracy  {np.mean(forest.predict(Xte) == yte):.3f}")
    print(f"single tree (depth 12, min_leaf 1) test accuracy  {np.mean(single.predict(Xte) == yte):.3f}")
    print("feature importances:")
    for name, value in zip(_names(args), forest.feature_importances_):
        bar = "#" * int(round(value * 40))
        print(f"  {name:<8} {value:6.3f} {bar}")
    print()
    print("region map (letters are the forest's averaged prediction):")
    print(_region_map(forest.predict, not args.no_extras))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m treelab", description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)

    def common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--data", choices=sorted(PRESETS), default="xor")
        p.add_argument("--n", type=_at_least(2), default=240, help="points in the preset, at least 2 (default 240)")
        p.add_argument("--seed", type=int, default=1)
        p.add_argument("--test", type=_test_fraction, default=0.3,
                       help="test fraction, strictly between 0 and 1 (default 0.3)")
        p.add_argument("--criterion", choices=CRITERIA, default="gini")
        p.add_argument("--depth", type=_at_least(0), default=5, help="maximum depth (default 5)")
        p.add_argument("--min-leaf", type=_at_least(1), default=3, help="minimum samples per leaf (default 3)")
        p.add_argument("--no-extras", action="store_true", help="split on x and y only (no x*y, x^2+y^2)")

    grow = sub.add_parser("grow", help="grow one CART tree and print it")
    common(grow)
    grow.add_argument("--prune", type=_float_at_least(0), default=0.0,
                      help="cost-complexity alpha (default 0 = no pruning)")
    grow.set_defaults(func=cmd_grow)

    forest = sub.add_parser("forest", help="train a random forest and report it")
    common(forest)
    forest.add_argument("--trees", type=_at_least(1), default=25, help="number of trees, at least 1 (default 25)")
    forest.set_defaults(func=cmd_forest)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)
