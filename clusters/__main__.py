"""Command line for the clustering lab.

    python -m clusters run --algo kmeans --k 4 --data blobs --seed 2
    python -m clusters run --algo dbscan --eps 0.06 --min-pts 5 --data moons
    python -m clusters run --algo gmm --k 3 --data aniso --seed 1
    python -m clusters elbow --data blobs --seed 2 --k-max 10

`run` prints iterations, inertia (k-means), log-likelihood (GMM), silhouette, and an ASCII scatter with one
digit per cluster ('x' for DBSCAN noise, '*' for centres). Colour is used on a terminal; --no-color turns it off.
"""

import argparse
import sys

import numpy as np

from clusters.data import PRESETS, make_dataset
from clusters.dbscan import NOISE, dbscan
from clusters.gmm import gmm_em
from clusters.kmeans import elbow, kmeans
from clusters.metrics import silhouette

PALETTE = ["36", "35", "33", "32", "34", "31", "96", "95", "93", "92"]  # ANSI colours, one per cluster
DIGITS = "0123456789"


def _positive_int(text: str) -> int:
    """argparse type for counts and sizes: an integer of at least 1 (zero would crash or print nothing)."""
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError(f"must be at least 1, got {value}")
    return value


def _non_negative_float(text: str) -> float:
    """argparse type for DBSCAN's eps: a radius cannot be negative."""
    value = float(text)
    if not value >= 0:  # also rejects NaN
        raise argparse.ArgumentTypeError(f"must be zero or more, got {text}")
    return value


def ascii_scatter(
    X: np.ndarray,
    labels: np.ndarray,
    centres: np.ndarray | None = None,
    width: int = 60,
    height: int = 22,
    color: bool = False,
) -> str:
    """Render points in the unit square as a character grid, one symbol per cluster."""
    grid = [[" "] * width for _ in range(height)]
    paint = [[None] * width for _ in range(height)]

    def put(x: float, y: float, ch: str, code: str | None) -> None:
        c = min(width - 1, max(0, int(x * (width - 1) + 0.5)))
        r = min(height - 1, max(0, int((1.0 - y) * (height - 1) + 0.5)))  # y grows upward
        grid[r][c] = ch
        paint[r][c] = code

    for (x, y), lab in zip(X, labels, strict=True):
        if lab == NOISE:
            put(x, y, "x", "90")
        else:
            put(x, y, DIGITS[int(lab) % len(DIGITS)], PALETTE[int(lab) % len(PALETTE)])
    if centres is not None:
        for x, y in centres:
            put(x, y, "*", "97")
    lines = []
    for r in range(height):
        row = []
        for c in range(width):
            ch, code = grid[r][c], paint[r][c]
            row.append(f"\x1b[{code}m{ch}\x1b[0m" if color and code else ch)
        lines.append("".join(row).rstrip())
    return "\n".join(lines)


def _run(args: argparse.Namespace) -> int:
    X, truth = make_dataset(args.data, args.seed, args.n)
    print(f"data {args.data}  n={len(X)}  seed={args.seed}  true components={len(set(truth.tolist()))}")
    centres = None
    if args.algo == "kmeans":
        res = kmeans(X, args.k, seed=args.seed, init=args.init, max_iter=args.max_iter)
        labels, centres = res.labels, res.centers
        print(f"algorithm k-means ({args.init}) k={args.k}")
        print(f"iterations {res.iterations} ({'converged' if res.converged else 'stopped at max-iter'})")
        print(f"inertia {res.inertia:.6f}")
    elif args.algo == "dbscan":
        labels, core = dbscan(X, args.eps, args.min_pts)
        ids = sorted(set(labels.tolist()) - {NOISE})
        print(f"algorithm dbscan eps={args.eps} min_pts={args.min_pts}")
        print(f"clusters {len(ids)}  core {int(core.sum())}  border {int((~core & (labels >= 0)).sum())}"
              f"  noise {int((labels == NOISE).sum())}")
    else:
        res = gmm_em(X, args.k, seed=args.seed, init=args.init, max_iter=args.max_iter)
        labels = res.labels
        centres = res.means
        print(f"algorithm gmm-em ({args.init}) k={args.k}")
        print(f"iterations {res.iterations} ({'converged' if res.converged else 'stopped at max-iter'})")
        print(f"log-likelihood {res.loglik:.6f} per point")
    if args.algo != "dbscan":
        print(f"clusters {len(set(labels.tolist()))}")
    score = silhouette(X, labels)
    print(f"silhouette {'n/a' if score is None else f'{score:.4f}'}")
    print(ascii_scatter(X, labels, centres, args.width, args.height, color=args.color))
    return 0


def _elbow(args: argparse.Namespace) -> int:
    X, _ = make_dataset(args.data, args.seed, args.n)
    print(f"elbow data {args.data}  n={len(X)}  seed={args.seed}")
    print(f"{'k':>3}  {'inertia':>12}")
    for k, inertia in elbow(X, args.k_max, seed=args.seed):
        print(f"{k:>3}  {inertia:>12.6f}")
    return 0


def main(argv: list[str] | None = None) -> int:
    """Entry point: parse the subcommand and run it. Returns the process exit code."""
    p = argparse.ArgumentParser(prog="python -m clusters", description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)

    run = sub.add_parser("run", help="run one algorithm on a preset and print the result")
    run.add_argument("--algo", choices=["kmeans", "dbscan", "gmm"], default="kmeans")
    run.add_argument("--k", type=_positive_int, default=3, help="clusters (k-means and GMM)")
    run.add_argument("--init", choices=["kmeans++", "random"], default="kmeans++")
    run.add_argument("--eps", type=_non_negative_float, default=0.06, help="DBSCAN neighbourhood radius")
    run.add_argument("--min-pts", type=_positive_int, default=5, help="DBSCAN core threshold")
    run.add_argument("--data", choices=PRESETS, default="blobs")
    run.add_argument("--n", type=_positive_int, default=300, help="points")
    run.add_argument("--seed", type=int, default=0)
    run.add_argument("--max-iter", type=_positive_int, default=100)
    run.add_argument("--width", type=_positive_int, default=60)
    run.add_argument("--height", type=_positive_int, default=22)
    run.add_argument("--no-color", action="store_true", help="plain characters even on a terminal")

    el = sub.add_parser("elbow", help="k-means inertia for k = 1..k-max")
    el.add_argument("--data", choices=PRESETS, default="blobs")
    el.add_argument("--n", type=_positive_int, default=300)
    el.add_argument("--seed", type=int, default=0)
    el.add_argument("--k-max", type=_positive_int, default=10)

    args = p.parse_args(argv)
    if args.cmd == "run":
        args.color = sys.stdout.isatty() and not args.no_color
        return _run(args)
    return _elbow(args)


if __name__ == "__main__":
    raise SystemExit(main())
