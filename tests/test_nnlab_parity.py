"""Parity: web/js/nnlab-core.js must reproduce nnlab (Python) on the same seeds, to 1e-9.

The check covers the random numbers, the preset points, the features and split, the initial weights, one
full-batch gradient, and three epochs of mini-batch training for several activations and for SGD. Node runs
the real core file (loaded as a data: URL, so no package.json is needed). The test is skipped without node.
"""

import json
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from nnlab.data import PRESETS, build_split, featurize, generate
from nnlab.mlp import Optimizer, evaluate, init_net, loss_and_grads, predict, train_epoch
from nnlab.rng import Rng

ROOT = Path(__file__).resolve().parent.parent
CORE = ROOT / "web" / "js" / "nnlab-core.js"
NODE = shutil.which("node")

CASE = {
    "nPoints": 12,
    "seed": 3,
    "data": "blobs",
    "n": 12,
    "noise": 0.1,
    "features": ["x", "y", "xy"],
    "testFrac": 0.5,
    "hidden": [3, 2],
    "lr": 0.05,
    "batch": 3,
    "l2": 0.01,
    "epochs": 3,
    "runs": [["tanh", "adam"], ["relu", "adam"], ["sigmoid", "adam"], ["tanh", "sgd"]],
    "grid": [[-1.0, 0.5], [0.0, 0.0], [0.7, -0.4]],
}

# The node program: run every step of the lab on the case and print one JSON object.
NODE_PROGRAM = r"""
import fs from 'node:fs';
const src = fs.readFileSync(process.env.NNLAB_CORE, 'utf8');
const core = await import('data:text/javascript;base64,' + Buffer.from(src).toString('base64'));
const cfg = JSON.parse(process.env.NNLAB_CASE);
const out = { points: {}, runs: [] };
for (const name of core.PRESETS) {
  out.points[name] = core.generatePoints(name, cfg.nPoints, cfg.seed).map(
    (p) => [p.x, p.y, p.label, p.jx, p.jy, p.u]);
}
const pts = core.generatePoints(cfg.data, cfg.n, cfg.seed);
const s = core.buildSplit(pts, cfg.features, cfg.noise, cfg.testFrac);
out.split = { ntr: s.ntr, nte: s.nte, Xtr: Array.from(s.Xtr), ytr: Array.from(s.ytr),
              Xte: Array.from(s.Xte), yte: Array.from(s.yte) };
const sizes = [s.F, ...cfg.hidden, 1];
for (const [act, kind] of cfg.runs) {
  const net = core.initNet(sizes, act, new core.Rng(cfg.seed + 1));
  const init = { W: net.W.map((w) => Array.from(w)), b: net.b.map((v) => Array.from(v)) };
  const g = core.lossAndGrads(net, s.Xtr, s.ytr, s.ntr, cfg.l2);
  const grads = { data: g.data, reg: g.reg, gW: g.gW.map((w) => Array.from(w)), gb: g.gb.map((v) => Array.from(v)) };
  const opt = new core.Optimizer(kind, net, cfg.lr);
  const rng = new core.Rng(cfg.seed + 2);
  const curve = [];
  for (let e = 0; e < cfg.epochs; e++) {
    core.trainEpoch(net, opt, s.Xtr, s.ytr, s.ntr, rng, cfg.batch, cfg.l2);
    curve.push(core.evaluate(net, s.Xtr, s.ytr, s.ntr).loss);
  }
  const trainEval = core.evaluate(net, s.Xtr, s.ytr, s.ntr);
  const testEval = core.evaluate(net, s.Xte, s.yte, s.nte);
  const grid = Float64Array.from(cfg.grid.flatMap(([x, y]) => core.featurize(x, y, cfg.features)));
  const pred = Array.from(core.predict(net, grid, cfg.grid.length));
  out.runs.push({ act, kind, init, grads, curve,
                  final: { W: net.W.map((w) => Array.from(w)), b: net.b.map((v) => Array.from(v)) },
                  trainEval, testEval, pred });
}
process.stdout.write(JSON.stringify(out));
"""


def _node_output():
    env = dict(os.environ, NNLAB_CORE=str(CORE), NNLAB_CASE=json.dumps(CASE))
    res = subprocess.run([NODE, "--input-type=module", "-e", NODE_PROGRAM], env=env, capture_output=True,
                         text=True, timeout=120)
    assert res.returncode == 0, res.stderr
    return json.loads(res.stdout)


def _python_reference():
    """The same steps, in Python, with the same seeds and order."""
    out = {"points": {}, "runs": []}
    for name in PRESETS:
        out["points"][name] = [[p["x"], p["y"], p["label"], p["jx"], p["jy"], p["u"]]
                               for p in generate(name, CASE["nPoints"], CASE["seed"])]
    pts = generate(CASE["data"], CASE["n"], CASE["seed"])
    Xtr, ytr, Xte, yte = build_split(pts, CASE["features"], CASE["noise"], CASE["testFrac"])
    out["split"] = {"ntr": Xtr.shape[0], "nte": Xte.shape[0], "Xtr": Xtr.ravel(), "ytr": ytr.ravel(),
                    "Xte": Xte.ravel(), "yte": yte.ravel()}
    sizes = [len(CASE["features"]), *CASE["hidden"], 1]
    # The probe points go through the same feature map as the browser's grid.
    grid_X = np.array([featurize(x, y, CASE["features"]) for x, y in CASE["grid"]], dtype=np.float64)
    for act, kind in CASE["runs"]:
        net = init_net(sizes, act, Rng(CASE["seed"] + 1))
        init = {"W": [w.ravel() for w in net.W], "b": [v.copy() for v in net.b]}
        data, reg, gW, gb = loss_and_grads(net, Xtr, ytr, CASE["l2"])
        grads = {"data": data, "reg": reg, "gW": [g.ravel() for g in gW], "gb": [g.copy() for g in gb]}
        opt = Optimizer(kind, net, CASE["lr"])
        rng = Rng(CASE["seed"] + 2)
        curve = []
        for _ in range(CASE["epochs"]):
            train_epoch(net, opt, Xtr, ytr, rng, CASE["batch"], CASE["l2"])
            curve.append(evaluate(net, Xtr, ytr)[0])
        tl, ta = evaluate(net, Xtr, ytr)
        vl, va = evaluate(net, Xte, yte)
        pred = predict(net, grid_X).ravel()
        out["runs"].append({
            "act": act, "kind": kind, "init": init, "grads": grads, "curve": curve,
            "final": {"W": [w.ravel() for w in net.W], "b": [v.copy() for v in net.b]},
            "trainEval": {"loss": tl, "acc": ta}, "testEval": {"loss": vl, "acc": va}, "pred": pred,
        })
    return out


@pytest.mark.skipif(NODE is None, reason="node is not installed")
def test_js_core_matches_python_reference():
    js = _node_output()
    py = _python_reference()

    for name in PRESETS:
        assert np.allclose(np.array(js["points"][name]), np.array(py["points"][name]), rtol=0, atol=1e-9), name

    sp = js["split"]
    assert (sp["ntr"], sp["nte"]) == (py["split"]["ntr"], py["split"]["nte"])
    assert sp["nte"] > 0  # the test-set metrics below must be real numbers, not NaN
    for key in ("Xtr", "ytr", "Xte", "yte"):
        assert np.allclose(np.array(sp[key], dtype=float), py["split"][key], rtol=0, atol=1e-9), key

    for jr, pr in zip(js["runs"], py["runs"]):
        tag = (jr["act"], jr["kind"])
        assert jr["act"] == pr["act"] and jr["kind"] == pr["kind"]
        for li in range(len(pr["init"]["W"])):
            assert np.allclose(jr["init"]["W"][li], pr["init"]["W"][li], rtol=0, atol=1e-9), tag
            assert np.allclose(jr["init"]["b"][li], pr["init"]["b"][li], rtol=0, atol=1e-9), tag
            assert np.allclose(jr["grads"]["gW"][li], pr["grads"]["gW"][li], rtol=0, atol=1e-9), tag
            assert np.allclose(jr["grads"]["gb"][li], pr["grads"]["gb"][li], rtol=0, atol=1e-9), tag
            assert np.allclose(jr["final"]["W"][li], pr["final"]["W"][li], rtol=0, atol=1e-9), tag
            assert np.allclose(jr["final"]["b"][li], pr["final"]["b"][li], rtol=0, atol=1e-9), tag
        assert abs(jr["grads"]["data"] - pr["grads"]["data"]) <= 1e-9, tag
        assert abs(jr["grads"]["reg"] - pr["grads"]["reg"]) <= 1e-9, tag
        assert np.allclose(jr["curve"], pr["curve"], rtol=0, atol=1e-9), tag
        assert abs(jr["trainEval"]["acc"] - pr["trainEval"]["acc"]) <= 1e-9, tag
        assert abs(jr["testEval"]["loss"] - pr["testEval"]["loss"]) <= 1e-9, tag
        assert np.allclose(jr["pred"], pr["pred"], rtol=0, atol=1e-9), tag
