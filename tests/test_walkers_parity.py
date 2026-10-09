"""Parity: the JavaScript core (web/js/walkers-core.js) must reproduce the Python port (walkers/core.py).

Physics: fixed creatures (seeded random ones and the two presets that are not the runner) on every terrain. Positions
are compared frame by frame in float64 (the JS side runs with float64 frames), and fitness and distance too, to 1e-9.

Evolution: a short run with a fixed seed. Every member's genome text must match in every generation, and so must
the best fitness (to 1e-9). The test is skipped when node is not installed.
"""

import json
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from walkers.core import DEFAULT_WORLD, Evolver, Mulberry32, decode, encode, random_genome, simulate

ROOT = Path(__file__).resolve().parent.parent
CORE = ROOT / "web" / "js" / "walkers-core.js"
PRESETS = ROOT / "walkers" / "data" / "champions.json"
NODE = shutil.which("node")
TERRAINS = ["flat", "hills", "steps", "gaps"]
TOL = 1e-9

# The node program: import the core through a data: URL (no package.json, so no module-type guessing), run every
# case with float64 frames, then the GA run, and print one JSON document.
NODE_PROGRAM = r"""
import fs from 'node:fs';
const src = fs.readFileSync(process.env.WALKERS_CORE, 'utf8');
const core = await import('data:text/javascript;base64,' + Buffer.from(src).toString('base64'));
const cases = JSON.parse(process.env.WALKERS_CASES);
const sims = cases.map(({ code, terrain, sample }) => {
  const g = core.decode(code);
  const r = core.simulate(g, { ...core.DEFAULT_WORLD, terrain }, { sampleEvery: sample, float64: true });
  return {
    fitness: r.fitness, distance: r.distance, exploded: r.exploded, fallen: r.fallen, spin: r.spin,
    frameCount: r.frameCount, N: r.N, frames: r.frames ? Array.from(r.frames) : null,
  };
});
const cfg = JSON.parse(process.env.WALKERS_GA);
const world = { ...core.DEFAULT_WORLD, terrain: cfg.terrain };
const ev = new core.Evolver({ seed: cfg.seed, popSize: cfg.pop, world });
const ga = [];
for (let g = 0; g < cfg.gens; g++) {
  const codes = ev.population.map((m) => core.encode(m.genome));
  const s = ev.step(0);
  ga.push({ codes, best: s.best, mean: s.mean, median: s.median, species: s.species, champion: s.champion.code });
}
console.log(JSON.stringify({ sims, ga }));
"""


def fixed_creatures():
    """Six seeded random creatures, and the presets other than the runner, which the distance test covers."""
    rng = Mulberry32(3)
    codes = [encode(random_genome(rng)) for _ in range(6)]
    doc = json.loads(PRESETS.read_text())
    codes += [p["code"] for p in doc["presets"] if p["id"] != "runner"]
    return codes


def run_node(cases, ga):
    env = {
        **os.environ,
        "WALKERS_CORE": str(CORE),
        "WALKERS_CASES": json.dumps(cases),
        "WALKERS_GA": json.dumps(ga),
    }
    proc = subprocess.run(
        [NODE, "--input-type=module", "-e", NODE_PROGRAM], env=env, capture_output=True, text=True, timeout=600
    )
    assert proc.returncode == 0, proc.stderr
    return json.loads(proc.stdout.strip().splitlines()[-1])


@pytest.mark.skipif(NODE is None, reason="node is not installed")
def test_js_physics_matches_python_on_every_terrain():
    codes = fixed_creatures()
    cases = [{"code": c, "terrain": t, "sample": 24} for t in TERRAINS for c in codes]
    js = run_node(cases, {"seed": 1, "pop": 2, "gens": 0, "terrain": "flat"})["sims"]
    assert len(js) == len(cases)
    worst_frames = 0.0
    worst_scalar = 0.0
    for case, got in zip(cases, js):
        world = {**DEFAULT_WORLD, "terrain": case["terrain"]}
        want = simulate([decode(case["code"])], world, case["sample"])[0]
        assert got["exploded"] == want["exploded"] and got["fallen"] == want["fallen"], case
        worst_scalar = max(worst_scalar, abs(got["fitness"] - want["fitness"]), abs(got["distance"] - want["distance"]))
        assert got["frameCount"] == want["frameCount"], case
        js_frames = np.array(got["frames"]).reshape(got["frameCount"], got["N"], 2)
        worst_frames = max(worst_frames, float(np.abs(js_frames - want["frames"]).max()))
    assert worst_scalar <= TOL, worst_scalar
    assert worst_frames <= TOL, worst_frames


@pytest.mark.skipif(NODE is None, reason="node is not installed")
def test_js_evolution_makes_the_same_decisions_as_python():
    cfg = {"seed": 5, "pop": 12, "gens": 3, "terrain": "flat"}
    js = run_node([], cfg)["ga"]
    ev = Evolver(seed=cfg["seed"], pop_size=cfg["pop"], world={**DEFAULT_WORLD, "terrain": cfg["terrain"]})
    for gen in range(cfg["gens"]):
        codes = [encode(m["genome"]) for m in ev.population]
        summary = ev.step(0)
        want = js[gen]
        assert codes == want["codes"], f"generation {gen + 1}: genomes differ"
        assert abs(summary["best"] - want["best"]) <= TOL
        assert abs(summary["mean"] - want["mean"]) <= TOL
        assert summary["species"] == want["species"]
        assert summary["champion"]["code"] == want["champion"]
