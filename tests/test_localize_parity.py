"""Parity: web/js/localize-core.js must reproduce the Python package (localize/) on the same seeds.

One node program runs the JavaScript core on a fixed scenario and prints JSON: the floors, the robot's true trajectory
(with a kidnap), and the per-step estimates of the grid filter and the particle filter. The same scenario is run here
in Python, and the two must agree. The random streams are bit-identical, so the floors, the trajectory and the beams
match to floating-point round-off; the filters are compared with a small tolerance because numpy and V8 round
transcendental functions (exp, log, cos) in the last bit, and the particle filter's resampling and sorting can
amplify that. The test is skipped when node is not installed.
"""

import json
import math
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from localize.grid import GridFilter
from localize.particles import ParticleFilter
from localize.sim import Autopilot, Sim
from localize.world import LAYOUTS, make_floor, motion

ROOT = Path(__file__).resolve().parent.parent
CORE = ROOT / "web" / "js" / "localize-core.js"
NODE = shutil.which("node")

SCENARIO = {
    "floors": [["halls", 1], ["halls", 2], ["vault", 1], ["vault", 3]],
    "sim": {"key": "halls", "floor_seed": 1, "seed": 11, "auto_seed": 12, "rays": 8, "sigma": 0.3, "pdrop": 0.05,
            "scale": 1.0, "steps": 26, "kidnap_at": 14},
    "grid": {"rays": 8, "sigma": 0.3, "pdrop": 0.05, "scale": 1.0},
    "pf": {"n": 300, "seed": 13, "rays": 8, "sigma": 0.3, "pdrop": 0.05, "scale": 1.0},
}

# The node program. It imports the core from a data: URL (so no package.json or module-type guessing), runs the
# scenario, and prints one JSON object on stdout.
NODE_PROGRAM = r"""
import fs from 'node:fs';
const src = fs.readFileSync(process.env.LOC_CORE, 'utf8');
const core = await import('data:text/javascript;base64,' + Buffer.from(src).toString('base64'));
const sc = JSON.parse(process.env.LOC_SCENARIO);
const out = { floors: [], traj: [], grid: [], pf: [] };
for (const [key, seed] of sc.floors) {
  const f = core.makeFloor(key, seed);
  out.floors.push({ walls: Array.from(f.walls), region: Array.from(f.region), pillars: f.pillars });
}
const s = sc.sim;
const f = core.makeFloor(s.key, s.floor_seed);
const sim = new core.Sim(f, s.seed, s.rays, s.sigma, s.pdrop, s.scale);
const ap = new core.Autopilot(f, s.auto_seed);
const cmds = [];
out.traj.push({ x: sim.x, y: sim.y, th: sim.th, z: Array.from(sim.z) });
for (let t = 0; t < s.steps; t++) {
  let rot = 0, fwd = 0, z;
  if (t === s.kidnap_at) { z = sim.kidnap(); ap.route = []; }
  else { [rot, fwd] = ap.command(sim.x, sim.y, sim.th); z = sim.drive(rot, fwd); }
  cmds.push([rot, fwd]);
  out.traj.push({ x: sim.x, y: sim.y, th: sim.th, z: Array.from(z) });
}
const g = new core.GridFilter(f, sc.grid.rays, sc.grid.sigma, sc.grid.pdrop, sc.grid.scale);
for (let t = 0; t < cmds.length; t++) {
  g.predict(cmds[t][0], cmds[t][1]);
  g.update(out.traj[t + 1].z);
  const e = g.estimate();
  out.grid.push({ x: e.x, y: e.y, th: e.th, mass: e.mass });
}
const p = sc.pf;
const pf = new core.ParticleFilter(f, p.n, p.rays, p.sigma, p.pdrop, p.scale, p.seed, true);
for (let t = 0; t < cmds.length; t++) {
  pf.step(cmds[t][0], cmds[t][1], out.traj[t + 1].z);
  const e = pf.estimate();
  out.pf.push({ x: e.x, y: e.y, th: e.th, mass: e.mass, ess: pf.ess, injected: pf.injected, resampled: pf.resampled });
}
process.stdout.write(JSON.stringify(out));
"""


def run_node() -> dict:
    env = dict(os.environ, LOC_CORE=str(CORE), LOC_SCENARIO=json.dumps(SCENARIO))
    proc = subprocess.run([NODE, "--input-type=module", "-e", NODE_PROGRAM], capture_output=True, text=True,
                          env=env, timeout=600)
    assert proc.returncode == 0, proc.stderr
    return json.loads(proc.stdout)


@pytest.fixture(scope="module")
def js():
    if NODE is None:
        pytest.skip("node is not installed")
    return run_node()


def test_layouts_match_the_javascript_copy(js):
    """The two copies of the layouts are the same data, so the page draws the floor the Python package studies."""
    text = CORE.read_text()
    for key, lay in LAYOUTS.items():
        assert f"title: '{lay['title']}'" in text, key
        assert f"w: {lay['w']}, h: {lay['h']}" in text, key
        for rect in lay["walls"]:
            assert f"[{', '.join(str(v) for v in rect)}]" in text, (key, rect)
        for slot in lay["slots"]:
            assert f"[{slot[0]}, {slot[1]}]" in text, (key, slot)


def test_floors_are_identical(js):
    for (key, seed), jf in zip(SCENARIO["floors"], js["floors"]):
        f = make_floor(key, seed)
        assert jf["walls"] == [int(v) for v in f.walls.ravel()], (key, seed)
        assert jf["region"] == [int(v) for v in f.region.ravel()], (key, seed)
        assert [list(p) for p in f.pillars] == jf["pillars"], (key, seed)


def test_true_robot_trajectory_matches(js):
    """The simulator, the odometry noise, the beams, the dropouts and the autopilot all run the same way."""
    s = SCENARIO["sim"]
    f = make_floor(s["key"], s["floor_seed"])
    sim = Sim(f, s["seed"], s["rays"], s["sigma"], s["pdrop"], s["scale"])
    ap = Autopilot(f, s["auto_seed"])
    traj = [(sim.x, sim.y, sim.th, sim.z)]
    cmds = []
    for t in range(s["steps"]):
        if t == s["kidnap_at"]:
            z = sim.kidnap()
            ap.route = []
            rot, fwd = 0.0, 0.0
        else:
            rot, fwd = ap.command(sim.x, sim.y, sim.th)
            z = sim.drive(rot, fwd)
        cmds.append((rot, fwd))
        traj.append((sim.x, sim.y, sim.th, z))
    for (x, y, th, z), jt in zip(traj, js["traj"]):
        assert math.isclose(x, jt["x"], abs_tol=1e-9)
        assert math.isclose(y, jt["y"], abs_tol=1e-9)
        assert math.isclose(th, jt["th"], abs_tol=1e-9)
        assert np.allclose(z, jt["z"], atol=1e-9)


def test_grid_filter_matches_step_by_step(js):
    """Same commands and beams (taken from the JavaScript run): the belief's estimate agrees at every step."""
    g_cfg = SCENARIO["grid"]
    f = make_floor(SCENARIO["sim"]["key"], SCENARIO["sim"]["floor_seed"])
    g = GridFilter(f, g_cfg["rays"], g_cfg["sigma"], g_cfg["pdrop"], g_cfg["scale"])
    cmds = js_commands(js)
    for t, (rot, fwd) in enumerate(cmds):
        g.predict(rot, fwd)
        g.update(np.array(js["traj"][t + 1]["z"]))
        x, y, th, mass = g.estimate()
        want = js["grid"][t]
        assert math.isclose(x, want["x"], abs_tol=1e-9), t
        assert math.isclose(y, want["y"], abs_tol=1e-9), t
        assert math.isclose(th, want["th"], abs_tol=1e-9), t
        assert math.isclose(mass, want["mass"], rel_tol=1e-6, abs_tol=1e-12), t


def test_particle_filter_matches_step_by_step(js):
    """Particles, resampling and injection replay the same random draws as the JavaScript port."""
    # The scenario must exercise both resampling and kidnap injection, or the comparison proves less than it says.
    assert any(r["resampled"] for r in js["pf"])
    assert any(r["injected"] > 0 for r in js["pf"])
    p = SCENARIO["pf"]
    f = make_floor(SCENARIO["sim"]["key"], SCENARIO["sim"]["floor_seed"])
    pf = ParticleFilter(f, p["n"], p["rays"], p["sigma"], p["pdrop"], p["scale"], p["seed"])
    cmds = js_commands(js)
    for t, (rot, fwd) in enumerate(cmds):
        pf.step(rot, fwd, np.array(js["traj"][t + 1]["z"]))
        e = pf.estimate()
        want = js["pf"][t]
        assert math.isclose(e["x"], want["x"], abs_tol=1e-6), t
        assert math.isclose(e["y"], want["y"], abs_tol=1e-6), t
        assert math.isclose(pf.ess, want["ess"], rel_tol=1e-6), t
        assert pf.injected == want["injected"], t
        assert pf.resampled == want["resampled"], t


def js_commands(js) -> list:
    """The commands the JavaScript run issued: recovered from the trajectory (a kidnap step has no drive)."""
    s = SCENARIO["sim"]
    f = make_floor(s["key"], s["floor_seed"])
    ap = Autopilot(f, s["auto_seed"])
    cmds = []
    for t in range(s["steps"]):
        if t == s["kidnap_at"]:
            ap.route = []
            cmds.append((0.0, 0.0))
        else:
            cmds.append(ap.command(js["traj"][t]["x"], js["traj"][t]["y"], js["traj"][t]["th"]))
    return cmds


# One deterministic predict per case, from a single bin, with the odometry noise turned off (scale 0). A case is a bin
# index and a commanded (turn, drive); it is a bump when the drive would end in a wall.
BUMP_PROGRAM = r"""
import fs from 'node:fs';
const src = fs.readFileSync(process.env.LOC_CORE, 'utf8');
const core = await import('data:text/javascript;base64,' + Buffer.from(src).toString('base64'));
const sc = JSON.parse(process.env.LOC_BUMP);
const f = core.makeFloor(sc.key, sc.seed);
const g = new core.GridFilter(f, sc.rays, sc.sigma, sc.pdrop, 0);
const out = [];
for (const c of sc.cases) {
  g.belief = new Float64Array(g.size);
  g.belief[c.idx] = 1;
  g.predict(c.rot, c.fwd);
  out.push(Array.from(g.belief));
}
process.stdout.write(JSON.stringify(out));
"""


def bump_cases(f, g) -> list:
    """Three bins whose drive hits a wall and two that move freely, for each of two turns."""
    cases = []
    for rot, fwd in ((0.5, 0.45), (-0.5, 0.45)):
        counts = {"bump": 0, "free": 0}
        for idx in range(0, g.size, 11):
            x2, y2, _ = motion(f.walls, np.array(g.x[idx]), np.array(g.y[idx]), np.array(g.th[idx]),
                               np.array(rot), np.array(fwd))
            kind = "bump" if x2 == g.x[idx] and y2 == g.y[idx] else "free"
            if counts[kind] < (3 if kind == "bump" else 2):
                counts[kind] += 1
                cases.append((idx, rot, fwd))
        assert counts == {"bump": 3, "free": 2}, rot
    return cases


def test_grid_bump_matches_javascript():
    """A drive into a wall keeps the position and applies the turn, in the browser as in Python."""
    if NODE is None:
        pytest.skip("node is not installed")
    f = make_floor("halls", 1)
    g = GridFilter(f, 8, 0.3, 0.05, motion_scale=0.0)
    cases = bump_cases(f, g)
    want = []
    for idx, rot, fwd in cases:
        g.belief = np.zeros(g.size)
        g.belief[idx] = 1.0
        g.predict(rot, fwd)
        want.append(g.belief)
    payload = {"key": "halls", "seed": 1, "rays": 8, "sigma": 0.3, "pdrop": 0.05,
               "cases": [{"idx": idx, "rot": rot, "fwd": fwd} for idx, rot, fwd in cases]}
    env = dict(os.environ, LOC_CORE=str(CORE), LOC_BUMP=json.dumps(payload))
    proc = subprocess.run([NODE, "--input-type=module", "-e", BUMP_PROGRAM], capture_output=True, text=True,
                          env=env, timeout=600)
    assert proc.returncode == 0, proc.stderr
    got = json.loads(proc.stdout)
    for (idx, rot, fwd), w, jb in zip(cases, want, got):
        assert np.allclose(w, jb, atol=1e-12, rtol=0), (idx, rot, fwd)
