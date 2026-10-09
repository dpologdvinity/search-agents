"""Parity: web/js/treelab-core.js must reproduce the Python reference (treelab/) exactly.

Each fixed case runs twice: here in Python, and in node by importing the JS module. Structure (node ids, split
features, thresholds, counts, leaf status), grid predictions, pruned trees, forest out-of-bag accuracy, feature
importances and bootstrap rows must agree. Thresholds, counts and labels must match exactly; floats such as gains
and importances are compared to 1e-9 (the JavaScript log2 may differ from NumPy's by one rounding step). The test
is skipped when node is not installed.
"""

import json
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from treelab.data import PRESETS, expand, make_preset, train_test_split
from treelab.forest import RandomForest
from treelab.tree import DecisionTree, preorder, prune

ROOT = Path(__file__).resolve().parent.parent
CORE = ROOT / "web" / "js" / "treelab-core.js"
NODE = shutil.which("node")

CASES = [
    {"kind": "tree", "name": "blobs", "n": 160, "seed": 3, "extras": True, "criterion": "gini", "depth": 4,
     "minLeaf": 2, "alphas": [0.0, 0.005, 0.02, 0.1]},
    {"kind": "tree", "name": "xor", "n": 160, "seed": 4, "extras": True, "criterion": "entropy", "depth": 5,
     "minLeaf": 3, "alphas": [0.0, 0.01, 0.05]},
    {"kind": "tree", "name": "checkerboard", "n": 200, "seed": 5, "extras": False, "criterion": "gini",
     "depth": 6, "minLeaf": 1, "alphas": [0.0, 0.002]},
    {"kind": "tree", "name": "spiral", "n": 200, "seed": 6, "extras": True, "criterion": "entropy", "depth": 7,
     "minLeaf": 2, "alphas": [0.0, 0.003, 0.02]},
    {"kind": "tree", "name": "nested", "n": 180, "seed": 7, "extras": True, "criterion": "gini", "depth": 5,
     "minLeaf": 4, "alphas": [0.0, 0.01]},
    {"kind": "tree", "name": "diagonal", "n": 220, "seed": 8, "extras": True, "criterion": "entropy", "depth": 8,
     "minLeaf": 1, "alphas": [0.0, 0.004, 0.03]},
    {"kind": "forest", "name": "diagonal", "n": 200, "seed": 9, "extras": True, "criterion": "gini", "depth": 5,
     "minLeaf": 2, "trees": 12, "forestSeed": 5, "splitSeed": 2},
    {"kind": "forest", "name": "spiral", "n": 180, "seed": 10, "extras": True, "criterion": "entropy", "depth": 6,
     "minLeaf": 1, "trees": 8, "forestSeed": 11, "splitSeed": 3},
]

GRID = [[i / 20.0, j / 20.0] for i in range(21) for j in range(21)]

# Node program: import the core as a data: URL (no package.json needed), run every case, print one JSON line.
NODE_PROGRAM = r"""
import fs from 'node:fs';
const src = fs.readFileSync(process.env.TREELAB_CORE, 'utf8');
const core = await import('data:text/javascript;base64,' + Buffer.from(src).toString('base64'));
const spec = JSON.parse(process.env.TREELAB_SPEC);
const grid = spec.grid;
const ser = (root) => {
  const nodes = core.preorder(root).sort((a, b) => a.id - b.id);
  return nodes.map((n) => [n.id, n.depth, n.counts, n.status, n.feature, n.threshold, n.gain, n.impurity,
    n.left ? n.left.id : -1, n.right ? n.right.id : -1]);
};
const out = spec.cases.map((c) => {
  const { X, y } = core.makePreset(c.name, c.n, c.seed);
  const K = core.PRESETS[c.name];
  const F = core.expand(X, c.extras);
  const gridF = core.expand(grid, c.extras);
  if (c.kind === 'tree') {
    const root = core.fitTree(F, y, K, { criterion: c.criterion, maxDepth: c.depth, minLeaf: c.minLeaf });
    const pruned = c.alphas.map((a) => {
      const p = core.prune(root, a);
      return { alpha: a, tree: ser(p), labels: gridF.map((x) => core.nodeLabel(core.leafOf(p, x))) };
    });
    return { X, y, tree: ser(root), labels: gridF.map((x) => core.nodeLabel(core.leafOf(root, x))), pruned };
  }
  const { train, test } = core.trainTestSplit(y.length, 0.3, c.splitSeed);
  const Ftr = train.map((i) => F[i]);
  const ytr = train.map((i) => y[i]);
  const forest = core.fitForest(Ftr, ytr, K, { nTrees: c.trees, criterion: c.criterion, maxDepth: c.depth,
    minLeaf: c.minLeaf, seed: c.forestSeed });
  const probs = gridF.map((x) => core.forestProba(forest, x));
  return {
    X, y, train, test,
    oob: forest.oob,
    importances: forest.importances,
    trees: forest.trees.map((t) => ({ tree: ser(t.root), inBag: Array.from(t.inBag) })),
    labels: probs.map((p) => core.argmax(p)),
    probs,
  };
});
console.log(JSON.stringify(out));
"""


def _ser(root):
    """Same layout as the JavaScript ser(): one row per node, ordered by id."""
    rows = []
    for node in sorted(preorder(root), key=lambda nd: nd.id):
        rows.append([node.id, node.depth, list(node.counts), node.status, node.feature, node.threshold,
                     node.gain, node.impurity, node.left.id if node.left else -1,
                     node.right.id if node.right else -1])
    return rows


def _python_tree_case(c):
    X, y = make_preset(c["name"], c["n"], c["seed"])
    K = PRESETS[c["name"]]
    F = expand(X, c["extras"])
    gridF = expand(np.array(GRID), c["extras"])
    root = DecisionTree(c["criterion"], c["depth"], c["minLeaf"]).fit(F, y, K).root
    pruned = []
    for a in c["alphas"]:
        p = prune(root, a)
        shown = DecisionTree()
        shown.root, shown.n_classes = p, K
        pruned.append({"alpha": a, "tree": _ser(p), "labels": shown.predict(gridF).tolist()})
    shown = DecisionTree()
    shown.root, shown.n_classes = root, K
    return {"X": X, "y": y, "tree": _ser(root), "labels": shown.predict(gridF).tolist(), "pruned": pruned}


def _python_forest_case(c):
    X, y = make_preset(c["name"], c["n"], c["seed"])
    K = PRESETS[c["name"]]
    F = expand(X, c["extras"])
    gridF = expand(np.array(GRID), c["extras"])
    tr, te = train_test_split(len(y), 0.3, c["splitSeed"])
    f = RandomForest(c["trees"], criterion=c["criterion"], max_depth=c["depth"], min_samples_leaf=c["minLeaf"],
                     seed=c["forestSeed"]).fit(F[tr], y[tr], K)
    probs = f.predict_proba(gridF)
    return {
        "X": X, "y": y, "train": tr.tolist(), "test": te.tolist(), "oob": f.oob_accuracy,
        "importances": f.feature_importances_.tolist(),
        "trees": [{"tree": _ser(t.root), "inBag": m.astype(int).tolist()} for t, m in zip(f.trees, f.in_bag)],
        "labels": probs.argmax(axis=1).tolist(), "probs": probs.tolist(),
    }


def _approx_eq(a, b, tol=1e-9):
    if isinstance(a, float) or isinstance(b, float):
        return a == pytest.approx(b, abs=tol)
    return a == b


def _compare_tree(py_rows, js_rows, where):
    assert len(py_rows) == len(js_rows), where
    for pr, jr in zip(py_rows, js_rows):
        # id, depth, counts, status, feature, left, right: exact. threshold, gain, impurity: tolerance. The two
        # languages compute the features with different rounding, so a midpoint threshold can differ by an ulp.
        assert pr[0] == jr[0] and pr[1] == jr[1], where
        assert pr[2] == jr[2], f"{where}: counts at node {pr[0]}"
        assert pr[3] == jr[3], f"{where}: status at node {pr[0]}"
        assert pr[4] == jr[4] and pr[8] == jr[8] and pr[9] == jr[9], f"{where}: structure at node {pr[0]}"
        assert _approx_eq(pr[5], jr[5]), f"{where}: threshold at node {pr[0]} ({pr[5]!r} vs {jr[5]!r})"
        assert _approx_eq(pr[6], jr[6]), f"{where}: gain at node {pr[0]}"
        assert _approx_eq(pr[7], jr[7]), f"{where}: impurity at node {pr[0]}"


@pytest.fixture(scope="module")
def js_results():
    if NODE is None:
        pytest.skip("node is not installed")
    spec = {"cases": CASES, "grid": GRID}
    env = dict(os.environ, TREELAB_CORE=str(CORE), TREELAB_SPEC=json.dumps(spec))
    proc = subprocess.run([NODE, "--input-type=module", "-e", NODE_PROGRAM], env=env, capture_output=True,
                          text=True, timeout=120)
    assert proc.returncode == 0, proc.stderr
    return json.loads(proc.stdout)


@pytest.mark.parametrize("index", range(len(CASES)))
def test_js_matches_python(index, js_results):
    c = CASES[index]
    js = js_results[index]
    py = _python_tree_case(c) if c["kind"] == "tree" else _python_forest_case(c)
    where = f"{c['kind']} {c['name']}"
    # Data: the same seeded draws give the same points (trig may differ in the last bit).
    assert np.allclose(np.array(py["X"]), np.array(js["X"]), rtol=0, atol=1e-12), where
    assert np.array_equal(np.array(py["y"]), np.array(js["y"])), where
    if c["kind"] == "tree":
        _compare_tree(py["tree"], js["tree"], where)
        assert py["labels"] == js["labels"], where
        for pp, jp in zip(py["pruned"], js["pruned"]):
            _compare_tree(pp["tree"], jp["tree"], f"{where} alpha {pp['alpha']}")
            assert pp["labels"] == jp["labels"], f"{where} alpha {pp['alpha']}"
    else:
        assert py["train"] == js["train"] and py["test"] == js["test"], where
        assert _approx_eq(py["oob"], js["oob"]), where
        assert np.allclose(py["importances"], js["importances"], rtol=0, atol=1e-9), where
        for i, (pt, jt) in enumerate(zip(py["trees"], js["trees"])):
            assert pt["inBag"] == jt["inBag"], f"{where} tree {i} bootstrap rows"
            _compare_tree(pt["tree"], jt["tree"], f"{where} tree {i}")
        assert py["labels"] == js["labels"], where
        assert np.allclose(np.array(py["probs"]), np.array(js["probs"]), rtol=0, atol=1e-12), where
