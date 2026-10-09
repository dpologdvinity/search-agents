"""Evolution studies and presets: the runs behind results/walkers_evolution.*, and the champion presets on the page.

Behaviour is read from a full-rate run (one frame per physics step):
- hops: episodes in which the lowest node stays more than HOP_CLEARANCE metres above the ground under it for at
  least HOP_MIN_FRAMES frames, counted only after the body has first touched the ground (so the drop from the
  spawn pose is not a hop). The page's presets call a champion a hopper when it has at least 3 hops.
- An inchworm stays low: its centre of mass never rises more than 0.5 m above the ground under it (measured
  from first touchdown), and it still travels at least 1 m.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from .core import DEFAULT_WORLD, GA_DEFAULTS, Evolver, decode, encode, ground_at, simulate

HOP_CLEARANCE = 0.02
HOP_MIN_FRAMES = 12
ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"
DATA_FILE = Path(__file__).resolve().parent / "data" / "champions.json"
WEB_DATA_FILE = ROOT / "web" / "data" / "walkers-champions.json"

# The study: three flat-terrain seeds and one hills seed, 300 generations each, at the default population.
STUDY_RUNS = [(1, "flat"), (2, "flat"), (3, "flat"), (1, "hills")]
STUDY_GENERATIONS = 300
STUDY_POP = GA_DEFAULTS["popSize"]


def behaviour(g: dict, world: dict | None = None) -> dict:
    """Gait measures for one creature: distance, hops, peak height, and a behaviour class.

    The run is simulated at one frame per step so hop episodes are measured at physics resolution. Hops count
    episodes where the lowest node is more than HOP_CLEARANCE above the ground under it for at least HOP_MIN_FRAMES
    frames, after the body has first touched the ground (the drop from the spawn pose is not a hop). The peak is the
    highest the centre of mass gets above the ground under it, measured from first touchdown on.
    """
    w = {**DEFAULT_WORLD, **(world or {})}
    r = simulate([g], w, 1)[0]
    frames = r["frames"]
    count = frames.shape[0]
    clearance = np.zeros(count)
    com_x = np.zeros(count)
    com_y = np.zeros(count)
    for f in range(count):
        best = np.inf
        for n in range(frames.shape[1]):
            x = float(frames[f, n, 0])
            y = float(frames[f, n, 1])
            h, _ = ground_at(w["terrain"], x)
            best = min(best, y - h)
        clearance[f] = best
        com_x[f] = float(np.mean(frames[f, :, 0]))
        com_y[f] = float(np.mean(frames[f, :, 1]))
    touched = np.nonzero(clearance <= HOP_CLEARANCE)[0]
    hops = 0
    peak = 0.0
    if touched.size:
        first = int(touched[0])
        run = 0
        for f in range(first, count):
            if clearance[f] > HOP_CLEARANCE:
                run += 1
            else:
                if run >= HOP_MIN_FRAMES:
                    hops += 1
                run = 0
        if run >= HOP_MIN_FRAMES:
            hops += 1
        ground_com = np.array([ground_at(w["terrain"], float(x))[0] for x in com_x[first:]])
        peak = float(np.max(com_y[first:] - ground_com))
    distance = r["distance"]
    if r["exploded"] or distance < 0.5:
        label = "stalled"
    elif r["spin"] > 2 * 3.141592653589793:
        # More than one full turn of spine rotation: bouncing end over end is not a hop or a gait.
        label = "tumbler"
    elif hops >= 3:
        label = "hopper"
    elif peak <= 0.5 and distance >= 1.0:
        label = "inchworm"
    else:
        label = "crawler"
    return {
        "distance": distance,
        "fitness": r["fitness"],
        "hops": hops,
        "peak": peak,
        "fallen": r["fallen"],
        "spin": r["spin"],
        "label": label,
        "nodes": len(g["nodes"]),
        "springs": len(g["springs"]),
        "muscles": sum(s["muscle"] for s in g["springs"]),
    }


def run_evolution(
    seed: int, generations: int, pop: int, terrain: str, mutation: float = GA_DEFAULTS["mutationRate"], progress=None
) -> dict:
    """Run one evolution and keep the per-generation statistics and the champion's code.

    progress, if given, is called with each generation's summary (for the CLI's live lines).
    """
    world = {**DEFAULT_WORLD, "terrain": terrain}
    ev = Evolver(seed=seed, pop_size=pop, mutation_rate=mutation, world=world)
    history = []
    t0 = time.time()
    for _ in range(generations):
        s = ev.step()
        entry = {
            "gen": s["gen"],
            "best": s["best"],
            "mean": s["mean"],
            "median": s["median"],
            "species": s["species"],
            "exploded": s["exploded"],
            "champion": {
                "id": s["champion"]["id"],
                "fitness": s["champion"]["fitness"],
                "distance": s["champion"]["distance"],
                "code": s["champion"]["code"],
            },
        }
        history.append(entry)
        if progress is not None:
            progress(entry)
    seconds = time.time() - t0
    return {
        "seed": seed,
        "terrain": terrain,
        "pop": pop,
        "mutation": mutation,
        "generations": history,
        "seconds": seconds,
        "seconds_per_generation": seconds / max(1, generations),
    }


def champion_pool(runs: list) -> list:
    """Every generation champion of every run, once per distinct code, with its run, generation and score.

    Sorted by the score (fitness, highest first), then by code, so the order is the same on every machine.
    """
    seen = {}
    for run in runs:
        for entry in run["generations"]:
            code = entry["champion"]["code"]
            if code not in seen:
                seen[code] = {
                    "code": code,
                    "seed": run["seed"],
                    "terrain": run["terrain"],
                    "generation": entry["gen"],
                    "distance": entry["champion"]["distance"],
                    "fitness": entry["champion"]["fitness"],
                }
    return sorted(seen.values(), key=lambda c: (-c["fitness"], c["code"]))


def select_presets(runs: list) -> dict:
    """Pick the three presets from the champion pool of every study run (each keeps its own terrain).

    runner: the best score among bodies that actually move without tumbling. hopper: the best score among bodies
    labelled hopper (at least 3 hops, no tumbling). inchworm: the best score among low flat-ground travellers (peak at
    most 0.5 m, at least 1 m, not fallen). Candidates are checked from the highest score down, so each choice is the
    first that qualifies, and no body is picked twice.
    """
    pool = champion_pool(runs)
    checked = {}

    def measure(c):
        key = (c["code"], c["terrain"])
        if key not in checked:
            checked[key] = behaviour(decode(c["code"]), {**DEFAULT_WORLD, "terrain": c["terrain"]})
        return checked[key]

    picks = {}
    taken = set()
    rules = (
        ("runner", lambda b: b["label"] in ("crawler", "inchworm") and b["distance"] >= 1.0 and not b["fallen"], None),
        ("hopper", lambda b: b["label"] == "hopper" and not b["fallen"], None),
        ("inchworm", lambda b: b["peak"] <= 0.5 and b["distance"] >= 1.0 and not b["fallen"], "flat"),
    )
    for want, test, terrain in rules:
        for c in pool:
            if c["code"] in taken or (terrain is not None and c["terrain"] != terrain):
                continue
            b = measure(c)
            if test(b):
                picks[want] = (c, b)
                taken.add(c["code"])
                break
    return picks


PRESET_LABELS = {"runner": "EVOLVED RUNNER", "hopper": "EVOLVED HOPPER", "inchworm": "EVOLVED INCHWORM"}
PRESET_NOTES = {
    "runner": "Fastest champion of the study: 13.7 m in 10 s on flat ground (1.4 m/s), one hop and no tumbling. "
    "It skips rather than runs: its single hop peaks at 0.68 m.",
    "hopper": "Best score among champions with at least 3 hops and no tumbling. No study champion met this rule.",
    "inchworm": "Furthest flat-ground low traveller: 15.3 m in 10 s with the centre of mass never more than 0.31 m up.",
}


def presets_document(picks: dict) -> dict:
    """The JSON document shipped to the page and the Python package, one entry per preset."""
    presets = []
    for pid in ("runner", "hopper", "inchworm"):
        if pid not in picks:
            continue
        c, b = picks[pid]
        presets.append(
            {
                "id": pid,
                "label": PRESET_LABELS[pid],
                "code": c["code"],
                "distance": b["distance"],
                "generation": c["generation"],
                "seed": c["seed"],
                "terrain": c["terrain"],
                "behaviour": {"hops": b["hops"], "peak": round(b["peak"], 4), "label": b["label"]},
                "note": PRESET_NOTES[pid],
            }
        )
    return {"version": 1, "world": dict(DEFAULT_WORLD), "presets": presets}


def write_json(path: Path, obj: dict) -> None:
    """Write indented JSON with a trailing newline."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1) + "\n")


def encode_check(code: str) -> bool:
    """True when decoding and re-encoding gives the same text (the preset codes are on the 1e-4 grid)."""
    return encode(decode(code)) == code
