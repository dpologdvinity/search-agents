"""Parity: the JavaScript port (web/js/clusters-core.js) must reproduce the Python reference (clusters/).

Each fixed scenario runs twice: here in Python, and in node by loading the JS module. Presets must match to
rounding (the two languages' cos and log can differ in the last bit), k-means and GMM labels must be identical,
centres and log-likelihoods must agree to 1e-9, and DBSCAN labels and core flags must be identical. The test is
skipped when node is not installed.
"""

import json
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from clusters.data import make_dataset
from clusters.dbscan import dbscan
from clusters.gmm import gmm_em
from clusters.kmeans import elbow, kmeans
from clusters.metrics import silhouette

ROOT = Path(__file__).resolve().parent.parent
CORE = ROOT / "web" / "js" / "clusters-core.js"
NODE = shutil.which("node")

PRESET_CASES = [
    ("blobs", 0, 120), ("aniso", 3, 90), ("rings", 0, 150), ("moons", 2, 100), ("uniform", 1, 80), ("smiley", 5, 140),
]
KMEANS_CASES = [
    ("blobs", 0, 3, "kmeans++"), ("aniso", 1, 3, "kmeans++"), ("smiley", 5, 4, "random"), ("moons", 2, 2, "kmeans++"),
]
DBSCAN_CASES = [("rings", 0, 0.09, 5), ("blobs", 4, 0.06, 4), ("moons", 2, 0.07, 3)]
GMM_CASES = [
    ("blobs", 6, 4, "kmeans++", 200), ("aniso", 1, 3, "kmeans++", 200), ("moons", 2, 2, "random", 200),
    ("aniso", 1, 3, "kmeans++", 2),  # cut short by max_iter: the returned parameters must match resp and loglik
]
ELBOW_CASE = ("blobs", 2, 120, 6)

# The node program: import the core as a data: URL (no package.json, so no module-type guessing), run every
# scenario named in CASES, and print one JSON object.
NODE_PROGRAM = r"""
import fs from 'node:fs';
const src = fs.readFileSync(process.env.CLUSTERS_CORE, 'utf8');
const core = await import('data:text/javascript;base64,' + Buffer.from(src).toString('base64'));
const cases = JSON.parse(process.env.CLUSTERS_CASES);
const flat = (a) => Array.from(a);
const out = {};
out.presets = cases.presets.map(([name, seed, n]) => {
  const d = core.makeDataset(name, seed, n);
  return { points: flat(d.points), truth: flat(d.truth) };
});
out.kmeans = cases.kmeans.map(([name, seed, k, init]) => {
  const { points } = core.makeDataset(name, seed, 150);
  const r = core.kmeans(points, k, { seed: 7, init });
  return { labels: flat(r.labels), centers: flat(r.centers), inertia: r.inertia, iterations: r.iterations,
           converged: r.converged, history: r.history };
});
out.dbscan = cases.dbscan.map(([name, seed, eps, minPts]) => {
  const { points } = core.makeDataset(name, seed, 150);
  const r = core.dbscan(points, eps, minPts);
  return { labels: flat(r.labels), core: flat(r.core), clusters: r.clusters };
});
out.gmm = cases.gmm.map(([name, seed, k, init, maxIter]) => {
  const { points } = core.makeDataset(name, seed, 150);
  const r = core.gmmEM(points, k, { seed: 3, init, maxIter });
  return { labels: flat(r.labels), loglik: r.loglik, iterations: r.iterations, converged: r.converged,
           weights: flat(r.weights), means: flat(r.means), history: r.history };
});
const [en, es, en2, ek] = cases.elbow;
{
  const { points } = core.makeDataset(en, es, en2);
  out.elbow = core.elbow(points, ek, 0).map(([k, v]) => [k, v]);
}
out.silhouette = cases.kmeans.map(([name, seed, k, init]) => {
  const { points } = core.makeDataset(name, seed, 150);
  const r = core.kmeans(points, k, { seed: 7, init });
  return core.silhouette(points, r.labels);
});
process.stdout.write(JSON.stringify(out));
"""


def _python_results():
    out = {"presets": [], "kmeans": [], "dbscan": [], "gmm": [], "silhouette": []}
    for name, seed, n in PRESET_CASES:
        X, truth = make_dataset(name, seed, n)
        out["presets"].append((X, truth))
    for name, seed, k, init in KMEANS_CASES:
        X, _ = make_dataset(name, seed, 150)
        out["kmeans"].append(kmeans(X, k, seed=7, init=init))
        out["silhouette"].append(silhouette(X, out["kmeans"][-1].labels))
    for name, seed, eps, mp in DBSCAN_CASES:
        X, _ = make_dataset(name, seed, 150)
        out["dbscan"].append(dbscan(X, eps, mp))
    for name, seed, k, init, max_iter in GMM_CASES:
        X, _ = make_dataset(name, seed, 150)
        out["gmm"].append(gmm_em(X, k, seed=3, init=init, max_iter=max_iter))
    name, seed, n, kmax = ELBOW_CASE
    X, _ = make_dataset(name, seed, n)
    out["elbow"] = elbow(X, kmax, seed=0)
    return out


@pytest.mark.skipif(NODE is None, reason="node is not installed")
def test_js_core_matches_python_reference(tmp_path):
    cases = {
        "presets": [list(c) for c in PRESET_CASES],
        "kmeans": [list(c) for c in KMEANS_CASES],
        "dbscan": [list(c) for c in DBSCAN_CASES],
        "gmm": [list(c) for c in GMM_CASES],
        "elbow": list(ELBOW_CASE),
    }
    env = {**os.environ, "CLUSTERS_CORE": str(CORE), "CLUSTERS_CASES": json.dumps(cases)}
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
    py = _python_results()

    for (X, truth), jp in zip(py["presets"], js["presets"], strict=True):
        assert np.allclose(X.reshape(-1), np.array(jp["points"]), atol=1e-9, rtol=0)
        assert np.array_equal(truth, np.array(jp["truth"]))

    for ref, jk in zip(py["kmeans"], js["kmeans"], strict=True):
        assert np.array_equal(ref.labels, np.array(jk["labels"]))
        assert np.allclose(ref.centers.reshape(-1), jk["centers"], atol=1e-9)
        assert ref.iterations == jk["iterations"] and ref.converged == jk["converged"]
        assert ref.inertia == pytest.approx(jk["inertia"], rel=1e-9)
        assert np.allclose(ref.history, jk["history"], rtol=1e-9)

    for ref, jd in zip(py["dbscan"], js["dbscan"], strict=True):
        labels, core = ref
        assert np.array_equal(labels, np.array(jd["labels"]))
        assert np.array_equal(core.astype(int), np.array(jd["core"]))

    for ref, jg in zip(py["gmm"], js["gmm"], strict=True):
        assert np.array_equal(ref.labels, np.array(jg["labels"]))
        assert ref.loglik == pytest.approx(jg["loglik"], rel=1e-9)
        assert ref.iterations == jg["iterations"] and ref.converged == jg["converged"]
        assert np.allclose(ref.weights, jg["weights"], atol=1e-9)
        assert np.allclose(ref.means.reshape(-1), jg["means"], atol=1e-9)

    assert np.allclose([v for _, v in py["elbow"]], [v for _, v in js["elbow"]], rtol=1e-9)
    assert [k for k, _ in py["elbow"]] == [k for k, _ in js["elbow"]]
    for ref, jsv in zip(py["silhouette"], js["silhouette"], strict=True):
        assert ref == pytest.approx(jsv, abs=1e-9)
