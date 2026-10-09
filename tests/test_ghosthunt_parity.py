"""Parity: the JavaScript port (web/js/ghosthunt-core.js) must reproduce the Python reference.

Each fixed scenario is played twice, once here and once in node by loading the core as a data: URL. The
autopilot plays every game to its end. Maze walls, every ghost's true path, every sonar reading, the Viterbi
paths and the final belief of ghost 0 must match. Cells, readings and bust turns are integers and must be
identical; the belief values are floats and are compared to 1e-9 (exp() may differ in the last bit between
V8 and libm). The test is skipped when node is not installed.
"""

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from ghosthunt import agent
from ghosthunt.game import Game
from ghosthunt.maze import generate

ROOT = Path(__file__).resolve().parent.parent
CORE = ROOT / "web" / "js" / "ghosthunt-core.js"
NODE = shutil.which("node")

# (size, ghosts, motion, sigma, seed, filter, particles)
SCENARIOS = [
    (11, 2, "random", 1.0, 3, "exact", 0),
    (15, 2, "lurker", 0.5, 4, "exact", 0),
    (15, 2, "patrol", 2.0, 5, "exact", 0),
    (15, 1, "random", 1.0, 6, "particles", 60),
    (21, 3, "lurker", 1.0, 7, "particles", 100),
    (11, 2, "patrol", 0.5, 8, "particles", 30),
]

NODE_PROGRAM = r"""
import fs from 'node:fs';
const src = fs.readFileSync(process.env.GH_CORE, 'utf8');
const core = await import('data:text/javascript;base64,' + Buffer.from(src).toString('base64'));
const scenarios = JSON.parse(process.env.GH_SCENARIOS);
const out = scenarios.map(([size, ghosts, model, sigma, seed, filt, particles]) => {
  const g = new core.Game({ size, ghosts, model, sigma, seed, filt, particles });
  const walls = Array.from(g.maze.walls);
  while (!g.done) g.act(core.choose(g));
  const traces = [];
  for (let i = 0; i < ghosts; i++) traces.push(g.trace(i));
  return {
    walls,
    t: g.t,
    busted: g.busted,
    bust_turns: g.ghosts.map((gh) => gh.bustTurn),
    true: traces.map((tr) => tr.true),
    readings: traces.map((tr) => tr.readings),
    viterbi: traces.map((tr) => tr.viterbi),
    marg0: g.marginal(0),
  };
});
console.log(JSON.stringify(out));
"""


def python_result(size, ghosts, model, sigma, seed, filt, particles):
    g = Game(size=size, ghosts=ghosts, model=model, sigma=sigma, seed=seed, filt=filt, particles=particles)
    walls = list(g.maze.walls)
    while not g.done:
        g.act(agent.choose(g))
    traces = [g.trace(i) for i in range(ghosts)]
    return {
        "walls": walls,
        "t": g.t,
        "busted": g.busted,
        "bust_turns": [gh.bust_turn for gh in g.ghosts],
        "true": [tr["true"] for tr in traces],
        "readings": [tr["readings"] for tr in traces],
        "viterbi": [tr["viterbi"] for tr in traces],
        "marg0": g.marginal(0),
    }


def node_results(scenarios):
    proc = subprocess.run(
        [NODE, "--input-type=module", "-e", NODE_PROGRAM],
        env={**os.environ, "GH_CORE": str(CORE), "GH_SCENARIOS": json.dumps(scenarios)},
        capture_output=True, text=True, timeout=300,
    )
    assert proc.returncode == 0, proc.stderr
    return json.loads(proc.stdout.strip().splitlines()[-1])


@pytest.mark.skipif(NODE is None, reason="node is not installed")
def test_js_port_matches_python_on_every_scenario():
    js = node_results([list(s) for s in SCENARIOS])
    assert len(js) == len(SCENARIOS)
    for scenario, got in zip(SCENARIOS, js):
        want = python_result(*scenario)
        for key in ("walls", "t", "busted", "bust_turns", "true", "readings", "viterbi"):
            assert got[key] == want[key], f"scenario {scenario}: {key} differs"
        assert len(got["marg0"]) == len(want["marg0"])
        for a, b in zip(got["marg0"], want["marg0"]):
            assert a == pytest.approx(b, abs=1e-9), f"scenario {scenario}: belief differs"


@pytest.mark.skipif(NODE is None, reason="node is not installed")
def test_js_maze_is_the_python_maze():
    js = node_results([[21, 1, "random", 1.0, 42, "exact", 0]])
    assert js[0]["walls"] == list(generate(21, 42))
