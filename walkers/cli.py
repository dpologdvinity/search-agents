"""Command line for EVOLVING WALKERS: `python -m walkers <command>`.

evolve     run a genetic algorithm and print one line per generation (--json writes the history)
replay     animate one creature (a w1 text code) as ASCII; --no-animate prints only the result line
benchmark  time the batch physics and one generation, and print the preset and sample distances
study      rerun the committed study (results/walkers_evolution.*) and rebuild the presets (slow: minutes)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from . import study
from .core import DEFAULT_WORLD, GA_DEFAULTS, TERRAINS, Evolver, Mulberry32, decode, random_genome, simulate
from .render import replay_frames, world_for

PRESETS_PATH = Path(__file__).resolve().parent / "data" / "champions.json"


def _cmd_evolve(args: argparse.Namespace) -> int:
    """Run the GA. Prints gen, best/mean/median fitness, species count and the number of exploded creatures."""
    world = world_for(args.terrain)
    ev = Evolver(seed=args.seed, pop_size=args.pop, mutation_rate=args.mutation, world=world)
    print(
        f"# evolve seed={args.seed} pop={args.pop} gens={args.generations} "
        f"terrain={args.terrain} mutation={args.mutation}"
    )
    history = []
    for _ in range(args.generations):
        s = ev.step()
        line = (
            f"gen {s['gen']:4d}  best {s['best']:8.3f}  mean {s['mean']:8.3f}  median {s['median']:8.3f}"
            f"  species {s['species']:3d}  exploded {s['exploded']:3d}"
        )
        print(line, flush=True)
        history.append(
            {
                "gen": s["gen"],
                "best": s["best"],
                "mean": s["mean"],
                "median": s["median"],
                "species": s["species"],
                "exploded": s["exploded"],
                "champion": {
                    "fitness": s["champion"]["fitness"],
                    "distance": s["champion"]["distance"],
                    "code": s["champion"]["code"],
                },
            }
        )
    champion = history[-1]["champion"]["code"] if history else None
    print(f"champion {champion}")
    if args.json:
        payload = {
            "seed": args.seed,
            "pop": args.pop,
            "terrain": args.terrain,
            "mutation": args.mutation,
            "generations": history,
            "champion": champion,
        }
        Path(args.json).write_text(json.dumps(payload, indent=1) + "\n")
    return 0


def _cmd_replay(args: argparse.Namespace) -> int:
    """Animate one run: clear-screen frames at 15 fps on a terminal, a frame every 20 steps otherwise."""
    try:
        g = decode(args.code)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    world = world_for(args.terrain)
    tty = sys.stdout.isatty()
    if args.no_animate:
        r = simulate([g], world, 0)[0]
        print(
            f"distance={r['distance']:.4f} fitness={r['fitness']:.4f} "
            f"fallen={int(r['fallen'])} exploded={int(r['exploded'])}"
        )
        return 0
    sample = 16 if tty else 20
    frames_run = simulate([g], world, sample)[0]
    texts = replay_frames(g, frames_run["frames"], sample, args.terrain)
    if tty:
        delay = sample / 240
        try:
            for i, text in enumerate(texts):
                sys.stdout.write("\033[2J\033[H")
                sys.stdout.write(f"EVOLVING WALKERS  t={i * sample / 240:5.2f}s  terrain={args.terrain}\n")
                sys.stdout.write(text + "\n")
                sys.stdout.flush()
                time.sleep(delay)
        except KeyboardInterrupt:
            pass
    else:
        for i, text in enumerate(texts):
            print(f"-- frame {i} t={i * sample / 240:.2f}s")
            print(text)
    print(f"distance={frames_run['distance']:.4f} fitness={frames_run['fitness']:.4f}")
    return 0


def _cmd_benchmark(args: argparse.Namespace) -> int:
    """Time a batch of 80 random creatures, one full generation, and report presets and sample distances."""
    rng = Mulberry32(args.seed)
    batch = [random_genome(rng) for _ in range(80)]
    t0 = time.time()
    simulate(batch, DEFAULT_WORLD, 0)
    batch_s = time.time() - t0
    print(f"batch of 80 creatures: {batch_s:.3f} s  ({80 / batch_s:.0f} creatures/s)")
    ev = Evolver(seed=args.seed, pop_size=80)
    t0 = time.time()
    ev.step()
    print(f"one generation (pop 80): {time.time() - t0:.3f} s")
    doc = json.loads(PRESETS_PATH.read_text())
    for p in doc["presets"]:
        r = simulate([decode(p["code"])], {**doc["world"], "terrain": p["terrain"]}, 0)[0]
        print(f"preset {p['id']:9s} distance {r['distance']:.4f} (recorded {p['distance']:.4f})")
    samples = [simulate([g], DEFAULT_WORLD, 0)[0]["distance"] for g in (random_genome(rng) for _ in range(20))]
    print(f"random creatures (seed {args.seed}): " + " ".join(f"{d:.3f}" for d in samples))
    return 0


def _cmd_study(args: argparse.Namespace) -> int:
    """Rerun the committed study, write results/walkers_evolution.json, and rebuild the presets on both sides.

    With --reuse PATH the runs are read from an existing results file instead of evolved again, so only the
    preset selection (the behaviour checks) runs. The selection is deterministic, so both paths give the same presets.
    """
    if args.reuse:
        runs = json.loads(Path(args.reuse).read_text())["runs"]
        print(f"reusing {len(runs)} runs from {args.reuse}")
    else:
        runs = _evolve_study_runs()
    picks = study.select_presets(runs)
    doc = study.presets_document(picks)
    results = {
        "config": {
            "runs": [{"seed": s, "terrain": t} for s, t in study.STUDY_RUNS],
            "generations": study.STUDY_GENERATIONS,
            "pop": study.STUDY_POP,
            "mutation": GA_DEFAULTS["mutationRate"],
            "world": DEFAULT_WORLD,
            "dt": 1 / 240,
        },
        "runs": runs,
        "presets": doc["presets"],
    }
    study.write_json(study.RESULTS_DIR / "walkers_evolution.json", results)
    study.write_json(study.DATA_FILE, doc)
    study.write_json(study.WEB_DATA_FILE, doc)
    print("presets:", ", ".join(p["id"] for p in doc["presets"]))
    return 0


def _evolve_study_runs() -> list:
    """Evolve every study run in order, printing each generation."""
    runs = []
    for seed, terrain in study.STUDY_RUNS:
        print(f"run seed={seed} terrain={terrain}", flush=True)
        run = study.run_evolution(
            seed,
            study.STUDY_GENERATIONS,
            study.STUDY_POP,
            terrain,
            progress=lambda e: print(f"  gen {e['gen']:3d} best {e['best']:.3f} mean {e['mean']:.3f}", flush=True),
        )
        runs.append(run)
    return runs


def _probability(text: str) -> float:
    """argparse type for a mutation rate: a probability from 0 to 1 (NaN is rejected too)."""
    value = float(text)
    if not 0 <= value <= 1:
        raise argparse.ArgumentTypeError(f"must be between 0 and 1, got {text}")
    return value


def build_parser() -> argparse.ArgumentParser:
    """The argument parser for all commands."""
    parser = argparse.ArgumentParser(
        prog="python -m walkers", description="EVOLVING WALKERS: soft bodies learn to walk"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    ev = sub.add_parser("evolve", help="run the genetic algorithm")
    ev.add_argument("--generations", type=int, default=50)
    ev.add_argument("--pop", type=int, default=GA_DEFAULTS["popSize"])
    ev.add_argument("--seed", type=int, default=1)
    ev.add_argument("--terrain", choices=TERRAINS, default="flat")
    ev.add_argument("--mutation", type=_probability, default=GA_DEFAULTS["mutationRate"])
    ev.add_argument("--json", metavar="PATH", help="write the per-generation history and the champion code")
    ev.set_defaults(func=_cmd_evolve)

    rp = sub.add_parser("replay", help="animate one creature")
    rp.add_argument("code", help="a w1|... creature string")
    rp.add_argument("--terrain", choices=TERRAINS, default="flat")
    rp.add_argument("--no-animate", action="store_true", help="print only the final distance line")
    rp.set_defaults(func=_cmd_replay)

    bm = sub.add_parser("benchmark", help="time the physics and one generation")
    bm.add_argument("--seed", type=int, default=1)
    bm.set_defaults(func=_cmd_benchmark)

    st = sub.add_parser("study", help="rerun the committed study and rebuild the presets (minutes)")
    st.add_argument("--reuse", metavar="PATH", help="reuse the runs in an existing results file (presets only)")
    st.set_defaults(func=_cmd_study)
    return parser


def main(argv: list | None = None) -> int:
    """Entry point: parse arguments and run the chosen command. Returns the process exit code."""
    args = build_parser().parse_args(argv)
    return args.func(args)
