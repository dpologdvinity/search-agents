"""The browser port (web/js/pathfind-core.js) must match the Python package step for step.

The test runs tests/js/pathfind_parity.mjs under node on fixed seeds, and compares the maze cells,
random numbers, expansion order (with discovered cells and g values), paths and costs with the Python
results. It is skipped when node is not installed.
"""

from __future__ import annotations

import json
import pathlib
import shutil
import subprocess

import pytest

from pathfind.mazes import make_maze
from pathfind.rng import Rng
from pathfind.search import ALGOS, run

NODE = shutil.which("node")
SCRIPT = pathlib.Path(__file__).resolve().parent / "js" / "pathfind_parity.mjs"
HEURISTICS = ("manhattan", "euclidean", "octile")


def _runs():
    out = []
    for kind in ("recursive", "prim", "scatter", "rooms", "open"):
        for size in (15, 25):
            for seed in (1, 2):
                swamp = 15 if kind == "open" or seed == 2 else 0
                for algo in ALGOS:
                    for diagonal in (False, True):
                        for heur in HEURISTICS:
                            out.append(
                                {
                                    "maze": kind,
                                    "size": size,
                                    "seed": seed,
                                    "density": 30,
                                    "swamp": swamp,
                                    "algo": algo,
                                    "diagonal": diagonal,
                                    "heur": heur,
                                    "weight": 1.5,
                                }
                            )
    return out


def _python_result(c):
    maze = make_maze(c["maze"], c["size"], c["size"], c["seed"], density=c["density"], swamp=c["swamp"])
    res = run(c["algo"], maze.grid, maze.start, maze.goal, diagonal=c["diagonal"], heur=c["heur"], weight=c["weight"])
    steps = [[u, side, [[v, g] for v, g in new]] for u, side, new in res.steps]
    return {
        "cells": maze.grid.cells,
        "start": maze.start,
        "goal": maze.goal,
        "found": res.found,
        "path": res.path,
        "cost": res.cost,
        "steps": steps,
        "generated": res.generated,
    }


@pytest.mark.skipif(NODE is None, reason="node is not installed")
def test_js_port_matches_python_step_for_step():
    runs = _runs()
    request = {"rng": [0, 1, 42, 2**31 + 5, 2**32 - 1], "u32": 12, "runs": runs}
    proc = subprocess.run([NODE, str(SCRIPT)], input=json.dumps(request), capture_output=True, text=True, timeout=300)
    assert proc.returncode == 0, proc.stderr
    js = json.loads(proc.stdout)

    for seed, got in zip(request["rng"], js["rng"], strict=True):
        r = Rng(seed)
        assert got == [r.u32() for _ in range(12)], seed

    assert len(js["runs"]) == len(runs)
    for c, got in zip(runs, js["runs"], strict=True):
        want = _python_result(c)
        for key in ("cells", "start", "goal", "found", "path", "cost", "steps", "generated"):
            assert got[key] == want[key], (c, key)


def test_python_side_is_deterministic_for_parity_seeds():
    # Without node this still pins the Python values: the same request twice gives the same output.
    for c in _runs()[:40]:
        assert _python_result(c) == _python_result(c)
