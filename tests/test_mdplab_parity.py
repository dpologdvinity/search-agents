"""Parity: web/js/mdplab-core.js must reproduce the Python mdplab package.

Each scenario runs value iteration, policy iteration and both learners in Python and, through node, in the
JavaScript port. Values must agree to 1e-9, policies and trajectories exactly. The preset lists must match
too, so the page and the command line teach the same grids. Skipped when node is not installed.
"""

import dataclasses
import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from mdplab import Grid, GridMDP, PolicyIterator, QLearner, ValueIterator
from mdplab.grid import PRESETS, Params

ROOT = Path(__file__).resolve().parent.parent
CORE = ROOT / "web" / "js" / "mdplab-core.js"
NODE = shutil.which("node")

NODE_PROGRAM = r"""
import fs from 'node:fs';
const src = fs.readFileSync(process.env.MDPLAB_CORE, 'utf8');
const core = await import('data:text/javascript;base64,' + Buffer.from(src).toString('base64'));
const scenarios = JSON.parse(process.env.MDPLAB_SCENARIOS);
const out = scenarios.map((s) => {
  const m = new core.GridModel(s.rows, s.params);
  const vi = new core.ValueIterator(m);
  for (let i = 0; i < s.vi_sweeps; i++) vi.sweep();
  const pi = new core.PolicyIterator(m, s.pi_seed);
  const summary = pi.solve(s.eps);
  const learned = {};
  for (const algo of ['q', 'sarsa']) {
    const agent = new core.QLearner(m, { algo, seed: s.seed });
    const returns = agent.run(s.episodes);
    learned[algo] = { returns, Q: agent.Q.flat(), eps: agent.eps, path: agent.last.path };
  }
  return {
    vi: vi.V,
    pi: { rounds: pi.rounds, evalSweeps: pi.evalSweeps, stable: summary.stable, capped: summary.capped,
          V: pi.V, pi: pi.pi },
    learned,
  };
});
console.log(JSON.stringify({ scenarios: out, presets: core.PRESETS }));
"""

SCENARIOS = [
    {"rows": list(p.rows), "params": dataclasses.asdict(p.params), "vi_sweeps": 20, "pi_seed": 1,
     "eps": 1e-9, "seed": 7, "episodes": 150}
    for p in PRESETS.values()
] + [
    {"rows": ["#####", "#S.-#", "#..+#", "#####"],
     "params": dataclasses.asdict(Params(gamma=0.9, slip=0.25, living=-0.05)),
     "vi_sweeps": 12, "pi_seed": 4, "eps": 1e-9, "seed": 3, "episodes": 200},
]


def python_result(s):
    mdp = GridMDP(Grid(tuple(s["rows"])), Params(**s["params"]))
    vi = ValueIterator(mdp)
    for _ in range(s["vi_sweeps"]):
        vi.sweep()
    pi = PolicyIterator(mdp, seed=s["pi_seed"])
    summary = pi.solve(s["eps"])
    learned = {}
    for algo in ("q", "sarsa"):
        agent = QLearner(mdp, algo=algo, seed=s["seed"])
        returns = agent.run(s["episodes"])
        learned[algo] = {
            "returns": returns,
            "Q": [q for row in agent.Q for q in row],
            "eps": agent.eps,
            "path": agent.last.path,
        }
    return {
        "vi": vi.V,
        "pi": {"rounds": pi.rounds, "evalSweeps": pi.eval_sweeps, "stable": summary["stable"],
               "capped": summary["capped"], "V": pi.V, "pi": pi.pi},
        "learned": learned,
    }


def node_results():
    proc = subprocess.run(
        [NODE, "--input-type=module", "-e", NODE_PROGRAM],
        env={**os.environ, "MDPLAB_CORE": str(CORE), "MDPLAB_SCENARIOS": json.dumps(SCENARIOS)},
        capture_output=True, text=True, timeout=300,
    )
    assert proc.returncode == 0, proc.stderr
    return json.loads(proc.stdout.strip().splitlines()[-1])


def close_list(a, b, tol=1e-9):
    assert len(a) == len(b)
    for x, y in zip(a, b):
        assert abs(x - y) <= tol, (x, y)


@pytest.mark.skipif(NODE is None, reason="node is not installed")
def test_js_presets_equal_the_python_presets():
    js = node_results()["presets"]
    py = [{"key": p.key, "label": p.label, "blurb": p.blurb, "rows": list(p.rows),
           "params": dataclasses.asdict(p.params)} for p in PRESETS.values()]
    got = [{"key": p["key"], "label": p["label"], "blurb": p["blurb"], "rows": p["rows"], "params": p["params"]}
           for p in js]
    assert got == py


@pytest.mark.skipif(NODE is None, reason="node is not installed")
def test_js_port_matches_python_on_every_scenario():
    js = node_results()["scenarios"]
    assert len(js) == len(SCENARIOS)
    for s, got in zip(SCENARIOS, js):
        want = python_result(s)
        close_list(got["vi"], want["vi"])

        gp, wp = got["pi"], want["pi"]
        assert gp["rounds"] == wp["rounds"]
        assert gp["evalSweeps"] == wp["evalSweeps"]
        assert gp["stable"] == wp["stable"] and gp["capped"] == wp["capped"]
        assert gp["pi"] == wp["pi"]
        close_list(gp["V"], wp["V"])

        for algo in ("q", "sarsa"):
            gl, wl = got["learned"][algo], want["learned"][algo]
            close_list(gl["returns"], wl["returns"])
            close_list(gl["Q"], wl["Q"])
            assert gl["path"] == wl["path"]
            assert abs(gl["eps"] - wl["eps"]) <= 1e-12
