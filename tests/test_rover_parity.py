"""Parity: the JavaScript port (web/js/rover_core.js) must reproduce the Python reference exactly.

Each fixed scenario is run twice: here in Python, and in node by loading the JS module. Maps, step counts,
replans, expansion counts (initial and total), cross-check mismatches and the final map must be identical.
The test is skipped when node is not installed.
"""

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from rover.world import generate
from tests.test_rover import PARITY_SCENARIOS, run_with_edits

ROOT = Path(__file__).resolve().parent.parent
CORE = ROOT / "web" / "js" / "rover_core.js"
NODE = shutil.which("node")

# Scenarios beyond the golden two: a spread of sizes, densities, sensor radii and drivers.
EXTRA = [
    (9, 0.2, 11, 1, "dstar"),
    (9, 0.35, 12, 2, "astar"),
    (17, 0.1, 5, 3, "dstar"),
    (17, 0.3, 21, 1, "astar"),
    (25, 0.25, 8, 2, "dstar"),
]

# The node program: import the core as a data: URL (no package.json, so no module-type guessing), run each
# scenario with the same edit schedule as tests.test_rover.run_with_edits, and print one JSON line.
NODE_PROGRAM = r"""
import fs from 'node:fs';
const src = fs.readFileSync(process.env.ROVER_CORE, 'utf8');
const core = await import('data:text/javascript;base64,' + Buffer.from(src).toString('base64'));
const scenarios = JSON.parse(process.env.ROVER_SCENARIOS);
const out = scenarios.map(([n, density, seed, radius, driver]) => {
  const walls = core.generate(n, density, seed);
  let indexSum = 0;
  for (let c = 0; c < walls.length; c++) if (walls[c]) indexSum += c;
  const world = { n, walls, start: 0, goal: n * n - 1 };
  const ex = new core.Explorer(world, radius, driver);
  const total = n * n;
  let t = 0;
  while (!ex.done && t < 20 * total) {
    if (t % 6 === 2) {
      const cell = (t * 37 + 11) % total;
      ex.edit(cell, !world.walls[cell]);
    }
    ex.tick();
    t++;
  }
  let finalWalls = 0;
  for (const w of world.walls) finalWalls += w;
  return {
    mapIndexSum: indexSum,
    ticks: t,
    steps: ex.steps,
    replans: ex.replans,
    init_dstar: ex.initExpansions.dstar,
    init_astar: ex.initExpansions.astar,
    expansions_dstar: ex.expansionsOf('dstar'),
    expansions_astar: ex.expansionsOf('astar'),
    mismatches: ex.mismatches,
    done: ex.done,
    walls: finalWalls,
  };
});
console.log(JSON.stringify(out));
"""


def python_result(n, density, seed, radius, driver):
    """The same fields as the node program, from the Python reference."""
    world_walls = generate(n, density, seed)
    index_sum = sum(c for c, w in enumerate(world_walls) if w)
    ex, ticks = run_with_edits(n, density, seed, radius, driver=driver)
    return {
        "mapIndexSum": index_sum,
        "ticks": ticks,
        "steps": ex.steps,
        "replans": ex.replans,
        "init_dstar": ex.init_expansions["dstar"],
        "init_astar": ex.init_expansions["astar"],
        "expansions_dstar": ex.expansions("dstar"),
        "expansions_astar": ex.expansions("astar"),
        "mismatches": ex.mismatches,
        "done": ex.done,
        "walls": sum(ex.world.walls),
    }


def node_results(scenarios):
    proc = subprocess.run(
        [NODE, "--input-type=module", "-e", NODE_PROGRAM],
        env={**os.environ, "ROVER_CORE": str(CORE), "ROVER_SCENARIOS": json.dumps(scenarios)},
        capture_output=True, text=True, timeout=120,
    )
    assert proc.returncode == 0, proc.stderr
    return json.loads(proc.stdout.strip().splitlines()[-1])


@pytest.mark.skipif(NODE is None, reason="node is not installed")
def test_js_port_matches_python_on_all_scenarios():
    scenarios = [[*case, "dstar"] for case, _ in PARITY_SCENARIOS] + [list(s) for s in EXTRA]
    js = node_results(scenarios)
    assert len(js) == len(scenarios)
    for scenario, got in zip(scenarios, js):
        want = python_result(*scenario)
        assert got == want, f"scenario {scenario}: js {got} != python {want}"
        assert got["mismatches"] == 0 and got["done"]


@pytest.mark.skipif(NODE is None, reason="node is not installed")
def test_js_port_matches_the_golden_numbers_of_the_python_scenarios():
    scenarios = [[*case, "dstar"] for case, _ in PARITY_SCENARIOS]
    js = node_results(scenarios)
    for (case, golden), got in zip(PARITY_SCENARIOS, js):
        for key, value in golden.items():
            assert got[key] == value, f"{case}: {key} is {got[key]}, golden {value}"
