"""Parity: the page's evaluation-function agent (web/js/snake-core.js) must match snake/evaluator.py.

Python plays seeded games with a fixed weight vector and records, at every position, each move's features,
whether it ends the game, its score, and the chosen move. Node loads the JavaScript core and computes the same
for the same positions. Features must agree to 1e-12, scores to 1e-9, and the chosen move exactly (ties
within 1e-9 are accepted, since the two languages may add the eight products in a different order).
The test is skipped when node is not installed.
"""

import json
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from snake.board import Game
from snake.evaluator import choose, move_table

ROOT = Path(__file__).resolve().parent.parent
CORE = ROOT / "web" / "js" / "snake-core.js"
NODE = shutil.which("node")

# Fixed weights for the parity check: not the committed ones, so the check does not depend on training.
WEIGHTS = [1.0, 0.5, 0.2, 0.5, 0.3, 0.5, 0.1, -0.2]
PARITY_SEEDS = [70_000, 70_001, 70_002, 70_003]

# The node program: import the core as a data: URL, then evaluate every recorded position once.
NODE_PROGRAM = r"""
import fs from 'node:fs';
const src = fs.readFileSync(process.env.SNAKE_CORE, 'utf8');
const core = await import('data:text/javascript;base64,' + Buffer.from(src).toString('base64'));
const cases = JSON.parse(fs.readFileSync(process.env.SNAKE_CASES, 'utf8'));
const out = cases.map((c) => {
  const game = core.gameFrom(c.size, c.body, c.heading, c.food);
  const rows = core.moveTable(game, c.weights).map((r) => ({
    action: r.action, dies: r.dies, score: r.score, features: Array.from(r.features),
  }));
  return { rows, action: core.chooseEval(c.weights, game) };
});
console.log(JSON.stringify(out));
"""


def recorded_positions():
    """Every position on the parity games, with Python's move table and choice."""
    cases = []
    for seed in PARITY_SEEDS:
        game = Game(seed)
        while game.alive:
            rows = move_table(game, WEIGHTS)
            cases.append({
                "size": game.size,
                "body": [list(c) for c in game.body],
                "heading": game.heading,
                "food": list(game.food) if game.food is not None else None,
                "weights": WEIGHTS,
                "rows": rows,
                "action": choose(game, WEIGHTS),
            })
            game.step(choose(game, WEIGHTS))
    return cases


@pytest.mark.skipif(NODE is None, reason="node is not installed")
def test_js_evaluator_matches_python_on_recorded_games(tmp_path):
    cases = recorded_positions()
    assert len(cases) > 100  # several games' worth of positions, including ones where the snake eats
    payload = [{k: c[k] for k in ("size", "body", "heading", "food", "weights")} for c in cases]
    # The positions go through a file: together they are too long for an environment variable.
    cases_path = tmp_path / "cases.json"
    cases_path.write_text(json.dumps(payload))
    env = {**os.environ, "SNAKE_CORE": str(CORE), "SNAKE_CASES": str(cases_path)}
    proc = subprocess.run([NODE, "--input-type=module", "-e", NODE_PROGRAM], env=env,
                          capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, proc.stderr
    js = json.loads(proc.stdout)
    assert len(js) == len(cases)
    for py, jsr in zip(cases, js):
        assert len(jsr["rows"]) == len(py["rows"]) == 3
        for prow, jrow in zip(py["rows"], jsr["rows"]):
            assert jrow["action"] == prow["action"] and jrow["dies"] == prow["dies"]
            assert np.allclose(jrow["features"], prow["features"], rtol=0, atol=1e-12)
            if prow["dies"]:
                assert jrow["score"] is None
            else:
                assert jrow["score"] == pytest.approx(prow["score"], abs=1e-9)
        # The chosen move: the same, unless the two best safe scores are a tie within rounding.
        if jsr["action"] != py["action"]:
            safe = sorted((r["score"] for r in py["rows"] if not r["dies"]), reverse=True)
            assert len(safe) >= 2 and safe[0] - safe[1] < 1e-9
