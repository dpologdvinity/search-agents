"""Seeded benchmark: how fast and how reliably each filter finds the robot.

Every trial is a (floor, simulator seed) pair: 2 layouts x 4 floor seeds x 2 robot seeds = 16 trials. A trial runs the
autopilot for STEPS steps and the filter alongside it. The score is the first step at which the estimate is within
REACH_M (0.5 m) of the true robot. A trial that never gets there is a failure. Times are wall-clock per filter step
on the machine that ran the benchmark, so they are useful as ratios, not as absolute speeds.

Sections:
  particles   100, 500 and 2000 particles, default noise
  noise       sensor noise (sigma) and odometry noise (scale) sweeps, 2000 particles
  filters     the grid filter and the 2000-particle filter side by side
  kidnap      the robot teleports at KIDNAP_AT; augmented MCL against plain MCL, and the grid filter
"""

from __future__ import annotations

import math
import statistics
import time
from pathlib import Path

from .grid import GridFilter
from .particles import ParticleFilter
from .sim import Autopilot, Sim, count_modes
from .world import CELL_M, make_floor

REACH_M = 0.5
STEPS = 100
KIDNAP_AT = 40
CHECKPOINTS = (10, 25, 50, 100)
DEFAULTS = dict(n=2000, rays=8, sigma=0.3, pdrop=0.05, scale=1.0)
TRIALS = [(key, fseed, sseed) for key in ("halls", "vault") for fseed in (1, 2, 3, 4) for sseed in (11, 12)]


def _err(est_x: float, est_y: float, sim: Sim) -> float:
    """Distance from the estimate to the true robot, in metres."""
    return math.hypot(est_x - sim.x, est_y - sim.y) * CELL_M


def run_pf(floor, sseed: int, n: int, rays: int, sigma: float, pdrop: float, scale: float,
           steps: int = STEPS, kidnap_at: int | None = None, augmented: bool = True) -> dict:
    """One particle-filter trial: per-step errors (m), per-step filter time (s), and the modes seen after step 2."""
    sim = Sim(floor, sseed, rays, sigma, pdrop, scale)
    ap = Autopilot(floor, sseed + 1000)
    pf = ParticleFilter(floor, n, rays, sigma, pdrop, scale, sseed + 2000, augmented=augmented)
    errs, times, modes2, inj = [], [], None, []
    for t in range(steps):
        if kidnap_at is not None and t == kidnap_at:
            z = sim.kidnap()
            ap.route = []
            rot, fwd = 0.0, 0.0
        else:
            rot, fwd = ap.command(sim.x, sim.y, sim.th)
            z = sim.drive(rot, fwd)
        t0 = time.perf_counter()
        pf.step(rot, fwd, z)
        e = pf.estimate()
        times.append(time.perf_counter() - t0)
        if t == 1:
            modes2 = count_modes(pf.position_histogram(), 0.02)
        errs.append(_err(e["x"], e["y"], sim))
        inj.append(pf.injected / n)
    return {"errs": errs, "times": times, "modes": modes2, "inj": inj, "kidnap_at": kidnap_at}


def run_grid(floor, sseed: int, rays: int, sigma: float, pdrop: float, scale: float, steps: int = STEPS,
             kidnap_at: int | None = None) -> dict:
    """One grid-filter trial. Building the grid (the sensor table) is timed separately as `init`."""
    sim = Sim(floor, sseed, rays, sigma, pdrop, scale)
    ap = Autopilot(floor, sseed + 1000)
    t0 = time.perf_counter()
    g = GridFilter(floor, rays, sigma, pdrop, scale)
    init = time.perf_counter() - t0
    errs, times = [], []
    for t in range(steps):
        if kidnap_at is not None and t == kidnap_at:
            z = sim.kidnap()
            ap.route = []
            rot, fwd = 0.0, 0.0
        else:
            rot, fwd = ap.command(sim.x, sim.y, sim.th)
            z = sim.drive(rot, fwd)
        t0 = time.perf_counter()
        g.predict(rot, fwd)
        g.update(z)
        ex, ey, _, _ = g.estimate()
        times.append(time.perf_counter() - t0)
        errs.append(_err(ex, ey, sim))
    return {"errs": errs, "times": times, "init": init}


HOLD = 10  # the estimate must stay within REACH_M for this many steps, so a lucky single hit does not count


def first_within(errs: list, start: int = 0):
    """The first step (1-based) from `start` on at which the estimate is within REACH_M and stays there for HOLD
    steps. None if no such window exists before the run ends."""
    run_len = 0
    for t in range(start, len(errs)):
        run_len = run_len + 1 if errs[t] <= REACH_M else 0
        if run_len >= HOLD:
            return t - HOLD + 2
    return None


def summarise(runs: list, label: str, kidnap_at: int | None = None) -> dict:
    """Aggregate trials: median steps to converge (among the ones that did), failure rate, mean error at the
    checkpoints, and mean filter time per step."""
    steps_to = []
    failures = 0
    for r in runs:
        if kidnap_at is None:
            c = first_within(r["errs"])
        else:
            c = first_within(r["errs"], kidnap_at)
            c = None if c is None else c - kidnap_at
        if c is None:
            failures += 1
        else:
            steps_to.append(c)
    out = {
        "label": label,
        "runs": len(runs),
        "median": statistics.median(steps_to) if steps_to else None,
        "fail": failures,
        "ms": 1000.0 * statistics.mean(t for r in runs for t in r["times"]),
    }
    for c in CHECKPOINTS:
        out[f"err{c}"] = statistics.mean(r["errs"][c - 1] for r in runs if len(r["errs"]) >= c)
    return out


def run(quick: bool = False) -> dict:
    """Run every section. `quick` shrinks the trials for a smoke test; the committed results use the full set."""
    trials = TRIALS[:4] if quick else TRIALS
    steps = STEPS  # the checkpoints run to step 100, so quick mode only drops trials
    rows = {"particles": [], "noise": [], "filters": [], "kidnap": []}
    maps = {}

    def floor_for(key, fseed):
        if (key, fseed) not in maps:
            maps[(key, fseed)] = make_floor(key, fseed)
        return maps[(key, fseed)]

    d = DEFAULTS
    # Particle counts, default noise.
    for n in (100, 500, 2000):
        runs = [run_pf(floor_for(k, f), s, n, d["rays"], d["sigma"], d["pdrop"], d["scale"], steps=steps)
                for k, f, s in trials]
        row = summarise(runs, f"{n} particles")
        row["modes2"] = statistics.mean(r["modes"] for r in runs)
        rows["particles"].append(row)
    # Sensor noise, then odometry noise, at 2000 particles.
    for sig in (0.15, 0.3, 0.6):
        runs = [run_pf(floor_for(k, f), s, d["n"], d["rays"], sig, d["pdrop"], 1.0, steps=steps) for k, f, s in trials]
        rows["noise"].append(summarise(runs, f"sensor sigma {sig} cells ({sig * CELL_M:.2f} m)"))
    for sc in (0.5, 2.0):
        runs = [run_pf(floor_for(k, f), s, d["n"], d["rays"], d["sigma"], d["pdrop"], sc, steps=steps)
                for k, f, s in trials]
        rows["noise"].append(summarise(runs, f"odometry scale {sc}"))
    # Grid filter and particle filter, same trials.
    grid_runs = [run_grid(floor_for(k, f), s, d["rays"], d["sigma"], d["pdrop"], d["scale"], steps=steps)
                 for k, f, s in trials]
    pf_runs = [run_pf(floor_for(k, f), s, d["n"], d["rays"], d["sigma"], d["pdrop"], d["scale"], steps=steps)
               for k, f, s in trials]
    rows["filters"].append(summarise(grid_runs, "grid filter (16 headings, 0.25 m bins)"))
    rows["filters"][-1]["init_ms"] = 1000.0 * statistics.mean(r["init"] for r in grid_runs)
    rows["filters"].append(summarise(pf_runs, f"particle filter, {d['n']} particles"))
    # Kidnapping: the robot teleports at KIDNAP_AT; time to re-converge within REACH_M.
    kstart = 30 if quick else KIDNAP_AT
    kidnap_steps = steps
    aug = [run_pf(floor_for(k, f), s, d["n"], d["rays"], d["sigma"], d["pdrop"], d["scale"], steps=kidnap_steps,
                  kidnap_at=kstart, augmented=True) for k, f, s in trials]
    plain = [run_pf(floor_for(k, f), s, d["n"], d["rays"], d["sigma"], d["pdrop"], d["scale"], steps=kidnap_steps,
                    kidnap_at=kstart, augmented=False) for k, f, s in trials]
    gkid = [run_grid(floor_for(k, f), s, d["rays"], d["sigma"], d["pdrop"], d["scale"], steps=kidnap_steps,
                     kidnap_at=kstart) for k, f, s in trials]
    rows["kidnap"].append(summarise(aug, "augmented MCL (injection)", kidnap_at=kstart))
    rows["kidnap"].append(summarise(plain, "plain MCL (no injection)", kidnap_at=kstart))
    rows["kidnap"].append(summarise(gkid, "grid filter (exact Bayes)", kidnap_at=kstart))
    # False alarms: particles injected per step before the kidnap, with no kidnap in the run. Post-kidnap: after it.
    calm = [run_pf(floor_for(k, f), s, d["n"], d["rays"], d["sigma"], d["pdrop"], d["scale"], steps=kstart,
                   augmented=True) for k, f, s in trials]
    rows["kidnap"][0]["false_alarm"] = statistics.mean(v for r in calm for v in r["inj"])
    rows["kidnap"][0]["post"] = statistics.mean(v for r in aug for v in r["inj"][kstart:])
    rows["kidnap"][0]["kidnap_at"] = kstart
    return {"trials": len(trials), "steps": steps, "reach_m": REACH_M, "rows": rows, "defaults": d}


def markdown(result: dict) -> str:
    """The results as markdown tables, the same tables the README and the page quote."""
    r = result["rows"]
    lines = [
        "# Lost Robot benchmark",
        "",
        f"{result['trials']} trials per row (2 layouts x 4 floor seeds x 2 robot seeds), {result['steps']} steps each. "
        f"Steps counts the first step with the estimate within {REACH_M} m of the robot. Time is per filter step "
        "in Python on one core; the page runs the same algorithms in JavaScript.",
        "",
    ]

    def table(title, rows, extra=None):
        head = "| setting | median steps to 0.5 m | failures | err @10 (m) | err @25 | err @50 | err @100 | ms/step |"
        rule = "|---|---:|---:|---:|---:|---:|---:|---:|"
        if extra:
            head += " " + extra[0] + " |"
            rule += "---:|"
        lines.extend([f"## {title}", "", head, rule])
        for row in rows:
            med = "-" if row["median"] is None else f"{row['median']:.0f}"
            cells = [row["label"], med, f"{row['fail']}/{row['runs']}", f"{row['err10']:.2f}",
                     f"{row['err25']:.2f}", f"{row['err50']:.2f}", f"{row['err100']:.2f}", f"{row['ms']:.1f}"]
            if extra:
                cells.append(extra[1](row))
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")

    table("Particle count (default noise)", r["particles"],
          extra=("belief modes after step 2", lambda row: f"{row['modes2']:.1f}"))
    table("Noise (2000 particles)", r["noise"])
    table("Grid filter vs particle filter (same trials)", r["filters"],
          extra=("init ms", lambda row: f"{row.get('init_ms', 0):.0f}" if "init_ms" in row else "-"))
    kid = r["kidnap"]
    a = kid[0]
    lines.extend([f"## Kidnapped robot (teleported at step {a['kidnap_at']})", "",
                  "| filter | median steps to recover | failures | ms/step |",
                  "|---|---:|---:|---:|"])
    for row in kid:
        med = "-" if row["median"] is None else f"{row['median']:.0f}"
        lines.append(f"| {row['label']} | {med} | {row['fail']}/{row['runs']} | {row['ms']:.1f} |")
    lines.extend(["", f"Augmented MCL injects {100 * a['false_alarm']:.2f}% of particles per step on average when "
                  f"nothing is wrong, and {100 * a['post']:.1f}% per step after the kidnap.", ""])
    return "\n".join(lines)


def write(result: dict, out_dir: Path) -> Path:
    """Write the markdown results to <out_dir>/localize_benchmark.md and return the path."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "localize_benchmark.md"
    path.write_text(markdown(result) + "\n")
    return path
