"""Parity: the JavaScript port (web/js/optlab-core.js) must reproduce the Python reference (optlab/).

Each scenario runs twice: here in Python, and in node by loading the JS module. The random stream, the
divergence steps and the steps-to-tolerance must match exactly. Positions and losses must agree to 1e-9
(relative to the magnitude once a value exceeds 1). The test is skipped when node is not installed.
"""

import json
import math
import shutil
import subprocess
from pathlib import Path

import pytest

from optlab import optimizers, surfaces
from optlab.race import Race
from optlab.rng import Rng

ROOT = Path(__file__).resolve().parent.parent
CORE = ROOT / "web" / "js" / "optlab-core.js"
NODE = shutil.which("node")
TOL = 1e-9

# Scenarios: a spread of surfaces, noise levels and seeds. Rastrigin uses cos and sin, where node and CPython can
# differ in the last bit. Its curvature makes the dynamics chaotic at lr 0.05: the two ports agree to 1e-9 for
# about 50 steps and then separate (measured). Its horizon is 30 steps, so a different libm does not flip a track
# inside the test. Every other scenario agrees to ~1e-16
# over its full length.
SCENARIOS = [
    {"surface": "bowl", "start": [2.4, 1.6], "steps": 200, "noise": 0.0, "seed": 0},
    {"surface": "ravine", "start": [3.2, 0.9], "steps": 150, "noise": 0.3, "seed": 7},
    {"surface": "rosenbrock", "start": [-1.5, 2.0], "steps": 60, "noise": 0.0, "seed": 0},
    {"surface": "himmelblau", "start": [-4.0, 4.5], "steps": 200, "noise": 0.05, "seed": 3},
    {"surface": "saddle", "start": [1.8, 0.6], "steps": 150, "noise": 0.1, "seed": 11},
    {"surface": "rastrigin", "start": [4.1, 2.9], "steps": 30, "noise": 0.1, "seed": 1},
    {"surface": "custom", "start": [-2.2, -1.6], "steps": 150, "noise": 0.0, "seed": 0},
]

NODE_PROGRAM = r"""
import fs from 'node:fs';
const src = fs.readFileSync(process.env.OPTLAB_CORE, 'utf8');
const core = await import('data:text/javascript;base64,' + Buffer.from(src).toString('base64'));
const scenarios = JSON.parse(process.env.OPTLAB_SCENARIOS);
const rng = new core.Rng(12345);
const draws = [];
for (let i = 0; i < 200; i++) draws.push(rng.uniform());
const normals = [];
for (let i = 0; i < 50; i++) normals.push(rng.normal());
const out = scenarios.map((sc) => {
  const s = sc.surface === 'custom' ? core.customSurface() : core.getSurface(sc.surface);
  const race = new core.Race({ surface: s, start: sc.start, noise: sc.noise, seed: sc.seed }).run(sc.steps);
  return race.tracks.map((tr) => ({
    name: tr.name,
    traj: tr.traj,
    losses: tr.losses,
    steps_to_tol: tr.stepsToTol,
    diverged_at: tr.divergedAt,
  }));
});
console.log(JSON.stringify({ draws, normals, runs: out }));
"""


def _python_runs():
    runs = []
    for sc in SCENARIOS:
        s = surfaces.custom() if sc["surface"] == "custom" else surfaces.get(sc["surface"])
        race = Race(s, sc["start"], noise=sc["noise"], seed=sc["seed"]).run(sc["steps"])
        runs.append(
            [
                {
                    "name": tr.name,
                    "traj": [list(p) for p in tr.traj],
                    "losses": tr.losses,
                    "steps_to_tol": tr.steps_to_tol,
                    "diverged_at": tr.diverged_at,
                }
                for tr in race.tracks
            ]
        )
    return runs


def _close(a, b):
    """Equal to TOL, absolute below 1 and relative above it. Non-finite values must match exactly."""
    if not (math.isfinite(a) and math.isfinite(b)):
        return a == b
    return abs(a - b) <= TOL * max(1.0, abs(a), abs(b))


@pytest.mark.skipif(NODE is None, reason="node is not installed")
def test_js_port_matches_python_trajectories(tmp_path):
    env = {"OPTLAB_CORE": str(CORE), "OPTLAB_SCENARIOS": json.dumps(SCENARIOS), "PATH": "/usr/bin:/bin"}
    proc = subprocess.run(
        [NODE, "--input-type=module", "-e", NODE_PROGRAM],
        capture_output=True,
        text=True,
        env=env,
        cwd=tmp_path,
        timeout=120,
    )
    assert proc.returncode == 0, proc.stderr
    js = json.loads(proc.stdout)

    # The random stream is integer arithmetic: it must match exactly.
    rng = Rng(12345)
    assert [rng.uniform() for _ in range(200)] == js["draws"]
    rng = Rng(12345)
    for _ in range(200):  # the JS program draws 200 uniforms first, then 50 normals
        rng.uniform()
    normals_py = [rng.normal() for _ in range(50)]
    assert normals_py == js["normals"]

    py_runs = _python_runs()
    assert len(py_runs) == len(js["runs"]) == len(SCENARIOS)
    for sc, py_tracks, js_tracks in zip(SCENARIOS, py_runs, js["runs"]):
        assert [t["name"] for t in py_tracks] == [t["name"] for t in js_tracks] == list(optimizers.NAMES)
        for pt, jt in zip(py_tracks, js_tracks):
            where = (sc["surface"], pt["name"])
            assert pt["steps_to_tol"] == jt["steps_to_tol"], where
            assert pt["diverged_at"] == jt["diverged_at"], where
            assert len(pt["traj"]) == len(jt["traj"]), where
            assert len(pt["losses"]) == len(jt["losses"]), where
            for (px, py), (jx, jy) in zip(pt["traj"], jt["traj"]):
                assert _close(px, jx) and _close(py, jy), where
            for pl, jl in zip(pt["losses"], jt["losses"]):
                assert _close(pl, jl), where
