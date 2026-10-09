"""Parity: the page's chance opponent (web/js/endgame-core.js) must reproduce endgame/chance.py.

The Python reference and the JavaScript port are fed the same positions and the same uniform numbers. The
odds table must be identical, every move's category and weight must match, probabilities must agree to 1e-12,
and each uniform number must pick the same move in both languages. Positions come from the tablebase itself,
so the move lists are the ones the page would receive from /api/endgame/analyze. The test is skipped when
node is not installed.
"""

import json
import os
import random
import shutil
import subprocess
from pathlib import Path

import pytest

from endgame import chance
from endgame.tablebase import load

ROOT = Path(__file__).resolve().parent.parent
CORE = ROOT / "web" / "js" / "endgame-core.js"
NODE = shutil.which("node")

# The node program: import the core as a data: URL (no package.json, so no module-type guessing), score every
# case, and print one JSON object. Each case is a position with its move list in the analyze reply's shape.
NODE_PROGRAM = r"""
import fs from 'node:fs';
const src = fs.readFileSync(process.env.ENDGAME_CORE, 'utf8');
const core = await import('data:text/javascript;base64,' + Buffer.from(src).toString('base64'));
const cases = JSON.parse(process.env.ENDGAME_CASES);
const out = {
  weights: core.CHANCE_WEIGHTS,
  cases: cases.map((c) => ({
    categories: c.moves.map((mv) => core.moveCategory(mv, c.wp)),
    weights: core.weights(c.moves, c.wp),
    probabilities: core.probabilities(c.moves, c.wp),
    picks: c.us.map((u) => {
      const mv = core.pickAt(c.moves, c.wp, u);
      return c.moves.indexOf(mv);
    }),
  })),
};
process.stdout.write(JSON.stringify(out));
"""


def _cases(n_per_piece: int = 12):
    """Positions for both pieces and both sides to move, with the move list the analyze reply would carry."""
    cases = []
    for piece in ("Q", "R"):
        tb = load(piece)
        rng = random.Random(77 if piece == "Q" else 78)
        made = 0
        while made < n_per_piece:
            pos = tb.random_position(rng, want="any")
            infos = tb.move_infos(pos)
            if len(infos) < 2:
                continue
            wp = pos[1]
            moves = [
                {"mover": i.move[0], "from": i.move[1], "to": i.move[2], "check": i.check,
                 "capture": i.move[0] == "k" and i.move[2] == wp}
                for i in infos
            ]
            us = [round(rng.random(), 12) for _ in range(25)] + [0.0, 0.999999999999]
            cases.append({"piece": piece, "wp": wp, "moves": moves, "us": us, "infos": infos,
                          "pos": pos})
            made += 1
    return cases


@pytest.mark.skipif(NODE is None, reason="node is not installed")
def test_js_chance_matches_the_python_reference(tmp_path):
    cases = _cases()
    payload = [{"wp": c["wp"], "moves": c["moves"], "us": c["us"]} for c in cases]
    env = {**os.environ, "ENDGAME_CORE": str(CORE), "ENDGAME_CASES": json.dumps(payload)}
    # Run node from an empty directory so no stray package.json or module file can change how it loads.
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

    # The table itself.
    assert js["weights"] == chance.CHANCE_WEIGHTS

    for case, jc in zip(cases, js["cases"], strict=True):
        pos, infos = case["pos"], case["infos"]
        py_cats = [chance.move_category(i, pos) for i in infos]
        assert jc["categories"] == py_cats
        assert jc["weights"] == chance.weights(infos, pos)
        assert jc["probabilities"] == pytest.approx(chance.probabilities(infos, pos), abs=1e-12)
        py_picks = [infos.index(chance.pick_at(infos, pos, u)) for u in case["us"]]
        assert jc["picks"] == py_picks


def test_python_and_js_share_one_table_in_source():
    # Runs without node: the JS table literal must spell out the same weights as the Python constant.
    src = CORE.read_text()
    for cat, w in chance.CHANCE_WEIGHTS.items():
        assert f"{cat}: {w}" in src
