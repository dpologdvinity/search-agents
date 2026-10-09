"""Watch a robot lose and find itself on a floor plan.

    python -m localize                       # autopilot drives; the particle cloud is drawn on the floor
    python -m localize --drive               # you drive with w/a/s/d (q quits); needs a terminal
    python -m localize --kidnap-at 40        # the robot is teleported at step 40; watch the cloud re-spread
    python -m localize --layout vault --seed 3 --particles 500 --delay 0.1
    python -m localize benchmark             # the seeded benchmark; writes results/localize_benchmark.md

In the picture, '#' is a wall, 'R' the true robot, 'E' the estimate when it is a different cell, and the dots
show where particles are: ' ' none, '.' a few, ':' some, '*' many (the density of the cloud per cell).
The header reports the error of the estimate in metres, the effective sample size, and how many separate
places the particles sit in ("modes"). Several modes mean the robot could be in more than one place.
"""

from __future__ import annotations

import argparse
import math
import select
import sys
import time
from pathlib import Path

from .particles import ParticleFilter
from .sim import Autopilot, Sim, count_modes
from .world import CELL_M, LAYOUTS, make_floor

ROOT = Path(__file__).resolve().parent.parent
CLEAR = "\033[H\033[2J"
DRIVE_KEYS = {"w": (0.0, 0.3), "s": (0.0, -0.2), "a": (-0.2, 0.0), "d": (0.2, 0.0)}


def render(floor, pf: ParticleFilter, sim: Sim, est) -> str:
    """The floor as text: walls, particle density per cell, the true robot, and the estimate."""
    hist = pf.position_histogram()
    total = float(pf.w.sum()) or 1.0
    grid = []
    for y in range(floor.h):
        row = []
        for x in range(floor.w):
            if floor.walls[y, x]:
                ch = "#"
            else:
                # The cell's share of the total weight, bucketed: a few particles show as '.', a crowd as '*'.
                share = hist.get((x, y), 0.0) / total
                ch = " " if share == 0 else ("." if share < 0.02 else (":" if share < 0.1 else "*"))
            row.append(ch)
        grid.append(row)
    tx, ty = int(math.floor(sim.x)), int(math.floor(sim.y))
    grid[ty][tx] = "R"
    ex, ey = int(math.floor(est["x"])), int(math.floor(est["y"]))
    if (ex, ey) != (tx, ty) and grid[ey][ex] != "#":
        grid[ey][ex] = "E"
    return "\n".join("".join(r) for r in grid)


def status(step: int, floor, sim: Sim, pf: ParticleFilter, est) -> str:
    err = math.hypot(est["x"] - sim.x, est["y"] - sim.y) * CELL_M
    modes = count_modes(pf.position_histogram(), 0.02)
    return (f"{floor.title}  seed {floor.seed}  step {step:4d}  error {err:5.2f} m  "
            f"ESS {pf.ess:7.1f}/{pf.n}  modes {modes}  injected {pf.injected}  "
            f"(R = robot, E = estimate)")


def read_key(timeout: float) -> str:
    """One key from the terminal, or '' after `timeout` seconds. Needs the terminal in cbreak mode."""
    r, _, _ = select.select([sys.stdin], [], [], timeout)
    return sys.stdin.read(1) if r else ""


def play(layout: str, seed: int, particles: int, rays: int, sigma: float, scale: float, steps: int, delay: float,
         kidnap_at: int | None, drive: bool, clear: bool, out=sys.stdout) -> int:
    """Animate one run. Each frame shows the cloud and the true robot; the header holds the error."""
    floor = make_floor(layout, seed)
    sim = Sim(floor, seed + 7, rays=rays, sigma=sigma, motion_scale=scale)
    ap = Autopilot(floor, seed + 9)
    pf = ParticleFilter(floor, particles, rays, sigma, 0.05, scale, seed + 11)
    tty = clear and out.isatty()
    old = None
    if drive:
        import termios
        import tty as ttymod
        old = termios.tcgetattr(sys.stdin)
        ttymod.setcbreak(sys.stdin.fileno())
    try:
        z = sim.z
        pf.step(0.0, 0.0, z)  # the first frame already has one set of beams
        for t in range(steps):
            if kidnap_at is not None and t == kidnap_at:
                z = sim.kidnap()
                ap.route = []
                rot, fwd = 0.0, 0.0
            elif drive:
                key = read_key(delay)
                if key == "q":
                    break
                rot, fwd = DRIVE_KEYS.get(key, (0.0, 0.0))
                z = sim.drive(rot, fwd)
            else:
                rot, fwd = ap.command(sim.x, sim.y, sim.th)
                z = sim.drive(rot, fwd)
            pf.step(rot, fwd, z)
            est = pf.estimate()
            frame = [status(t + 1, floor, sim, pf, est), render(floor, pf, sim, est)]
            if tty:
                out.write(CLEAR)
            out.write("\n".join(frame) + "\n")
            out.flush()
            if delay and tty and not drive:
                time.sleep(delay)
    finally:
        if old is not None:
            import termios
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old)
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m localize",
        description="Lost Robot: a robot on a known floor plan localises itself with a grid filter and particles.")
    parser.add_argument("command", nargs="?", default="play", choices=["play", "benchmark"],
                        help="play (default) animates one run; benchmark runs the seeded comparison")
    parser.add_argument("--layout", choices=sorted(LAYOUTS), default="halls", help="floor plan (default halls)")
    parser.add_argument("--seed", type=int, default=1, help="seed for the pillars and the robot (default 1)")
    parser.add_argument("--particles", type=int, default=2000, help="particle count, 10..20000 (default 2000)")
    parser.add_argument("--rays", type=int, default=8, help="range beams per scan, 4..16 (default 8)")
    parser.add_argument("--sigma", type=float, default=0.3, help="sensor noise in cells, 0.05..1.5 (default 0.3)")
    parser.add_argument("--scale", type=float, default=1.0, help="odometry noise multiplier, 0..3 (default 1)")
    parser.add_argument("--steps", type=int, default=200, help="steps to run (default 200)")
    parser.add_argument("--delay", type=float, default=0.12, help="seconds between frames on a terminal (default 0.12)")
    parser.add_argument("--kidnap-at", type=int, default=None, help="teleport the robot at this step")
    parser.add_argument("--drive", action="store_true", help="drive with w/a/s/d; q quits (needs a terminal)")
    parser.add_argument("--no-clear", action="store_true", help="print frames one after another, without clearing")
    parser.add_argument("--out", default=str(ROOT / "results"), help="benchmark: directory for the results file")
    args = parser.parse_args(argv)

    if args.command == "benchmark":
        from . import benchmark

        result = benchmark.run()
        path = benchmark.write(result, Path(args.out))
        print(benchmark.markdown(result))
        print(f"wrote {path}")
        return 0

    if not 10 <= args.particles <= 20000:
        parser.error("--particles must be between 10 and 20000")
    if not 4 <= args.rays <= 16:
        parser.error("--rays must be between 4 and 16")
    if not 0.05 <= args.sigma <= 1.5:
        parser.error("--sigma must be between 0.05 and 1.5")
    if args.drive and not sys.stdin.isatty():
        parser.error("--drive needs a terminal")
    play(args.layout, args.seed, args.particles, args.rays, args.sigma, args.scale, args.steps, args.delay,
         args.kidnap_at, args.drive, not args.no_clear)
    return 0


if __name__ == "__main__":
    sys.exit(main())
