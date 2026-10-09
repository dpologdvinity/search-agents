"""Command line for the optimizer race.

    python -m optlab race --surface rosenbrock --start -1.5,2 --steps 500
    python -m optlab surfaces

`race` prints a table (final loss, steps to tolerance, divergence) and an ASCII contour with the trails.
`--lr` takes one number for every optimizer, or a list such as `sgd=0.05,adam=0.1`. A negative start
(`-1.5,2`) works with `--start -1.5,2` as written: the argument is joined to its option before parsing.
"""

from __future__ import annotations

import argparse
import sys

from . import optimizers, surfaces
from .optimizers import Hyper
from .race import BLOW, DEFAULT_LR, GTOL, Race
from .render import contour_ascii, table


def _bounded(low: float, high: float, *, low_open: bool = False, high_open: bool = False, integer: bool = False):
    """argparse type factory for a number between low and high; each end is open or closed as named. NaN fails
    both comparisons, so it is rejected too. A beta of 1 or more, or a step count of 0, used to print nonsense."""

    def parse(text: str):
        value = int(text) if integer else float(text)
        above = value > low if low_open else value >= low
        below = value < high if high_open else value <= high
        if not (above and below):
            left = "(" if low_open else "["
            right = ")" if high_open else "]"
            raise argparse.ArgumentTypeError(f"must be in {left}{low:g}, {high:g}{right}, got {text}")
        return value

    return parse


def _start(text: str) -> tuple[float, float]:
    parts = text.split(",")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("start must be x,y")
    try:
        return float(parts[0]), float(parts[1])
    except ValueError:
        raise argparse.ArgumentTypeError("start must be two numbers, e.g. -1.5,2") from None


def _lrs(text: str) -> dict[str, float]:
    """`0.05` sets every optimizer; `sgd=0.05,adam=0.1` sets some (the rest keep their defaults)."""
    if "=" not in text:
        value = float(text)
        return {name: value for name in optimizers.NAMES}
    out = {}
    for item in text.split(","):
        name, _, value = item.partition("=")
        if name not in optimizers.NAMES:
            raise argparse.ArgumentTypeError(f"unknown optimizer {name!r}")
        out[name] = float(value)
    return out


def _names(text: str) -> list[str]:
    names = [n.strip() for n in text.split(",") if n.strip()]
    for n in names:
        if n not in optimizers.NAMES:
            raise argparse.ArgumentTypeError(f"unknown optimizer {n!r}; choose from {', '.join(optimizers.NAMES)}")
    return names


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m optlab", description="Race gradient-based optimizers on 2D landscapes.")
    sub = p.add_subparsers(dest="command", required=True)

    r = sub.add_parser("race", help="run every optimizer from one start and print the results")
    r.add_argument("--surface", default="rosenbrock", choices=sorted(surfaces.SURFACES))
    r.add_argument("--start", type=_start, default=None, help="x,y (default: the surface's start)")
    r.add_argument("--steps", type=_bounded(1, float("inf"), integer=True), default=500)
    r.add_argument("--optimizers", type=_names, default=list(optimizers.NAMES), help="comma-separated names")
    r.add_argument("--lr", type=_lrs, default=None, help="one value for all, or sgd=0.05,adam=0.1")
    decay = _bounded(0, 1, high_open=True)  # decays and betas must be in [0, 1) for the averages to settle
    r.add_argument("--momentum", type=decay, default=Hyper.momentum, help="beta for momentum and Nesterov, in [0, 1)")
    r.add_argument("--beta1", type=decay, default=Hyper.beta1, help="Adam first-moment decay, in [0, 1)")
    r.add_argument("--beta2", type=decay, default=Hyper.beta2, help="Adam second-moment decay, in [0, 1)")
    r.add_argument("--eps", type=_bounded(0, float("inf"), low_open=True), default=Hyper.eps,
                   help="epsilon under the square root, above 0")
    r.add_argument("--rho", type=decay, default=Hyper.rho, help="RMSProp squared-gradient decay, in [0, 1)")
    r.add_argument("--noise", type=_bounded(0, float("inf")), default=0.0,
                   help="std of the gradient noise (0 = exact gradients)")
    r.add_argument("--seed", type=int, default=0, help="seed for the gradient noise")
    r.add_argument("--tol", type=_bounded(0, float("inf")), default=GTOL,
                   help="gradient norm below which a track has reached a minimum")
    r.add_argument("--width", type=_bounded(1, float("inf"), integer=True), default=64,
                   help="columns in the ASCII map, at least 1")
    r.add_argument("--no-map", action="store_true", help="print the table only")

    sub.add_parser("surfaces", help="list the built-in landscapes")
    return p


def _join_negative_start(argv: list[str]) -> list[str]:
    """argparse reads `--start -1.5,2` as an option; rejoin it to `--start=-1.5,2`."""
    out = []
    i = 0
    while i < len(argv):
        if argv[i] == "--start" and i + 1 < len(argv) and argv[i + 1].startswith("-"):
            out.append(f"--start={argv[i + 1]}")
            i += 2
            continue
        out.append(argv[i])
        i += 1
    return out


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(_join_negative_start(list(sys.argv[1:] if argv is None else argv)))
    if args.command == "surfaces":
        for s in surfaces.SURFACES.values():
            print(f"{s.name:<12} domain {s.domain}  start {s.start}  {s.note}")
        return 0

    surface = surfaces.get(args.surface)
    start = args.start if args.start is not None else surface.start
    lrs = dict(DEFAULT_LR)
    if args.lr:
        lrs.update(args.lr)
    hyper = Hyper(momentum=args.momentum, beta1=args.beta1, beta2=args.beta2, eps=args.eps, rho=args.rho)
    race = Race(
        surface,
        start,
        names=args.optimizers,
        lrs=lrs,
        hyper=hyper,
        noise=args.noise,
        seed=args.seed,
        gtol=args.tol,
        blow=BLOW,
    ).run(args.steps)
    print(
        f"surface {surface.name}  start ({start[0]:g}, {start[1]:g})  steps {args.steps}  "
        f"noise {args.noise:g} (seed {args.seed})  tol {args.tol:g}"
    )
    print(table(race))
    if not args.no_map:
        print()
        print(contour_ascii(race, width=args.width))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
