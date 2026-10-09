"""Parity: web/js/markov-core.js must reproduce markov/model.py exactly.

Each case (corpus, level, order, temperature, seed) is generated here in Python, and again in node by importing the
JavaScript core. Tokens, the prompt length, restarts, every pick (position, token and probability) and the copy
statistics must match. The test is skipped when node is not installed.
"""

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from bandits.rng import Rng
from markov import NGramModel, copy_report, generate, load_corpus, ngram_sets, tokenize

ROOT = Path(__file__).resolve().parent.parent
CORE = ROOT / "web" / "js" / "markov-core.js"
NODE = shutil.which("node")

COUNT = 60
CASES = [
    # corpus, level, order, temperature, seed. Temperatures 1 and 0.5 keep the powers exact (c**1, c**2).
    ("alice", "word", 1, 1.0, 1),
    ("alice", "word", 2, 1.0, 1),
    ("alice", "word", 3, 1.0, 7),
    ("alice", "word", 4, 0.5, 12345),
    ("alice", "word", 5, 1.0, 99),
    ("sonnets", "word", 2, 0.5, 3),
    ("sonnets", "word", 3, 1.0, 42),
    ("sonnets", "char", 4, 1.0, 5),
    ("sonnets", "char", 5, 0.5, 2024),
    ("constitution", "word", 3, 1.0, 1),
    ("constitution", "word", 5, 1.0, 8),
    ("pride", "word", 2, 1.0, 2147483647),
    ("pride", "char", 2, 1.0, 0),
]

# Runs long enough on a short corpus to reach a dead end and exercise the restart jump.
RESTART_CASES = [("sonnets", "word", 4, 1.0, 3, 400), ("constitution", "word", 3, 1.0, 1, 400)]

NODE_PROGRAM = r"""
import fs from 'node:fs';
const src = fs.readFileSync(process.env.MARKOV_CORE, 'utf8');
const core = await import('data:text/javascript;base64,' + Buffer.from(src).toString('base64'));
const bodies = JSON.parse(process.env.MARKOV_BODIES);
const cases = JSON.parse(process.env.MARKOV_CASES);
const out = cases.map(([name, level, order, temperature, seed, count]) => {
  const tokens = core.tokenize(bodies[name], level);
  const model = core.buildModel(tokens, order);
  const rng = new core.Rng(seed);
  const gen = core.generate(model, rng, count, temperature);
  const rep = core.copyReport(gen, core.ngramSets(tokens), order);
  return {
    tokens: gen.tokens, prompt_len: gen.promptLen, restarts: gen.restarts,
    picks: gen.picks.map((p) => [p.index, p.token, p.prob]),
    copied_pct: rep.copiedPct, longest: rep.longestRun,
    vocab: model.vocabSize, contexts: model.contextCount, ngrams: model.ngramCount,
  };
});
process.stdout.write(JSON.stringify(out));
"""


def py_case(body, name, level, order, temperature, seed, count):
    tokens = tokenize(body, level)
    model = NGramModel(tokens, order)
    gen = generate(model, Rng(seed), count, temperature)
    rep = copy_report(gen, ngram_sets(tokens), order)
    return {
        "tokens": gen.tokens,
        "prompt_len": gen.prompt_len,
        "restarts": gen.restarts,
        "picks": [[p.index, p.token, p.prob] for p in gen.picks],
        "copied_pct": rep.copied_pct,
        "longest": rep.longest_run,
        "vocab": model.vocab_size,
        "contexts": model.context_count,
        "ngrams": model.ngram_count,
    }


def node_results(cases, bodies):
    env = dict(os.environ, MARKOV_CORE=str(CORE), MARKOV_BODIES=json.dumps(bodies), MARKOV_CASES=json.dumps(cases))
    proc = subprocess.run([NODE, "--input-type=module", "-e", NODE_PROGRAM], env=env, capture_output=True,
                          text=True, timeout=120)
    assert proc.returncode == 0, proc.stderr
    return json.loads(proc.stdout)


def _bodies():
    return {name: load_corpus(name).body for name in ("alice", "sonnets", "constitution", "pride")}


@pytest.mark.skipif(NODE is None, reason="node is not installed")
def test_js_generation_matches_python_token_for_token():
    bodies = _bodies()
    cases = [c if len(c) == 6 else (*c, COUNT) for c in CASES + RESTART_CASES]
    js = node_results(cases, bodies)
    for case, got in zip(cases, js):
        name, level, order, temperature, seed, count = case
        ref = py_case(bodies[name], name, level, order, temperature, seed, count)
        label = f"{name} {level} order {order} T {temperature} seed {seed}"
        assert got["tokens"] == ref["tokens"], label
        assert got["prompt_len"] == ref["prompt_len"], label
        assert got["restarts"] == ref["restarts"], label
        assert [p[:2] for p in got["picks"]] == [p[:2] for p in ref["picks"]], label
        for gp, rp in zip(got["picks"], ref["picks"]):
            assert gp[2] == pytest.approx(rp[2], rel=1e-12, abs=0), label
        assert got["copied_pct"] == pytest.approx(ref["copied_pct"]), label
        assert got["longest"] == ref["longest"], label
        assert (got["vocab"], got["contexts"], got["ngrams"]) == (ref["vocab"], ref["contexts"], ref["ngrams"]), label


@pytest.mark.skipif(NODE is None, reason="node is not installed")
def test_restart_path_is_exercised_by_the_parity_cases():
    bodies = _bodies()
    seen = 0
    for name, level, order, temperature, seed, count in RESTART_CASES:
        seen += py_case(bodies[name], name, level, order, temperature, seed, count)["restarts"]
    assert seen > 0
