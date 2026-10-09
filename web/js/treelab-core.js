// Decision tree and random forest math for the Tree lab. Pure functions, no DOM, so node can run them.
// This file is a line-by-line port of treelab/{rng,data,tree,forest}.py. The same operations run in the same
// order, the same seeds draw the same numbers, and ties break the same way, so the trees agree node for node
// (tests/test_treelab_parity.py checks this). Change both sides together.

export const EPS = 1e-12;            // gains and risks closer than this count as ties
export const CRITERIA = ['gini', 'entropy'];
export const PRESETS = {             // name -> number of classes
  blobs: 3, xor: 2, checkerboard: 2, spiral: 2, nested: 3, diagonal: 2,
};
export const FLIP = 0.15;
const BLOB_CENTRES = [[0.25, 0.3], [0.7, 0.35], [0.5, 0.75]];
const BLOB_RADIUS = 0.16;
const RING_R2 = [0.04, 0.13];

// Feature names, in the order expand() produces them. The Python names are for the CLI; these are for the page.
export const FEATURE_NAMES = ['x', 'y', 'x·y', 'x²+y²'];

/** Mulberry32 (same constants and bit tricks as treelab/rng.py). next() is in [0, 1). */
export function mulberry32(seed) {
  let a = seed >>> 0;
  return {
    next() {
      a = (a + 0x6D2B79F5) >>> 0;
      let t = a;
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    },
    randInt(n) {
      return Math.floor(this.next() * n);
    },
  };
}

/** Gini 1 - sum p^2, or entropy -sum p log2 p, of a class-count array. Zero for an empty or pure node. */
export function impurity(counts, criterion = 'gini') {
  let n = 0;
  for (const c of counts) n += c;
  if (n === 0) return 0;
  let acc = 0;
  if (criterion === 'gini') {
    for (const c of counts) {
      const p = c / n;
      acc = acc + p * p;
    }
    return 1 - acc;
  }
  if (criterion !== 'entropy') throw new Error(`unknown criterion ${criterion}`);
  for (const c of counts) {
    if (c > 0) {
      const p = c / n;
      acc = acc - p * Math.log2(p);
    }
  }
  return acc;
}

/**
 * Exhaustive best split of the rows idx over the candidate features (same rule as tree.best_split):
 * sort on each feature, test the midpoint of every distinct adjacent pair, keep both sides >= minLeaf.
 * Within a feature, the first candidate within EPS of the best gain wins; across features, the first feature
 * within EPS of the best wins. Returns {feature, threshold, gain} or null.
 */
export function bestSplit(X, y, idx, features, criterion, minLeaf, K) {
  const m = idx.length;
  const total = new Array(K).fill(0);
  for (const r of idx) total[y[r]] += 1;
  const parent = impurity(total, criterion);
  let bestF = -1;
  let bestGain = 0;
  let bestT = 0;
  for (const f of features) {
    const vals = idx.map((r) => X[r][f]);
    const order = vals.map((_, i) => i);
    order.sort((a, b) => (vals[a] - vals[b]) || (a - b)); // stable: ties keep row order
    const v = order.map((i) => vals[i]);
    const ys = order.map((i) => y[idx[i]]);
    const cands = []; // [gain, threshold]
    const cl = new Array(K).fill(0);
    for (let pos = 0; pos < m - 1; pos++) {
      cl[ys[pos]] += 1;
      if (!(v[pos] < v[pos + 1])) continue;
      const nl = pos + 1;
      const nr = m - nl;
      if (nl < minLeaf || nr < minLeaf) continue;
      const cr = total.map((c, k) => c - cl[k]);
      const gl = impurity(cl, criterion);
      const gr = impurity(cr, criterion);
      const gain = parent - (nl / m) * gl - (nr / m) * gr;
      const lo = v[pos];
      const hi = v[pos + 1];
      let thr = (lo + hi) / 2;
      if (!(lo < thr)) thr = hi;
      cands.push([gain, thr]);
    }
    if (!cands.length) continue;
    let gmax = -Infinity;
    for (const [g] of cands) if (g > gmax) gmax = g;
    let k = 0;
    while (!(cands[k][0] >= gmax - EPS)) k++;
    const g = cands[k][0];
    const ok = bestF < 0 ? g > EPS : g > bestGain + EPS;
    if (ok) {
      bestF = f;
      bestGain = g;
      bestT = cands[k][1];
    }
  }
  if (bestF < 0) return null;
  return { feature: bestF, threshold: bestT, gain: bestGain };
}

/**
 * Breadth-first CART builder (tree.Builder). step() splits the next splittable node of the FIFO frontier and
 * returns true, or returns false when the frontier is empty. grow() runs to the end.
 * Nodes: {id, depth, counts, impurity, feature, threshold, gain, left, right, status, idx}.
 * status is 'pending' (in the frontier), 'split', or 'leaf'.
 */
export function makeBuilder({ X, y, K, criterion = 'gini', maxDepth = 4, minLeaf = 1, rng = null,
  maxFeatures = null, idx = null }) {
  const p = X.length ? X[0].length : 0;
  const mf = maxFeatures == null ? null : Math.min(maxFeatures, p);
  if (mf != null && mf < p && !rng) throw new Error('feature subsampling needs an rng');
  let nextId = 0;
  const make = (rows, depth) => {
    const counts = new Array(K).fill(0);
    for (const r of rows) counts[y[r]] += 1;
    return {
      id: nextId++, depth, counts, impurity: impurity(counts, criterion),
      feature: -1, threshold: 0, gain: 0, left: null, right: null, status: 'pending', idx: rows,
    };
  };
  const rows0 = idx == null ? Array.from({ length: y.length }, (_, i) => i) : Array.from(idx);
  const root = make(rows0, 0);
  const queue = [root];

  // All features, or a random subset of mf drawn by a partial Fisher-Yates shuffle.
  const candidates = () => {
    if (mf == null || mf >= p) return Array.from({ length: p }, (_, i) => i);
    const arr = Array.from({ length: p }, (_, i) => i);
    for (let i = 0; i < mf; i++) {
      const j = i + rng.randInt(p - i);
      const t = arr[i]; arr[i] = arr[j]; arr[j] = t;
    }
    return arr.slice(0, mf);
  };

  const step = () => {
    while (queue.length) {
      const node = queue.shift();
      const rows = node.idx;
      node.idx = null;
      if (node.impurity <= 0 || node.depth >= maxDepth || rows.length < 2 * minLeaf) {
        node.status = 'leaf';
        continue;
      }
      const best = bestSplit(X, y, rows, candidates(), criterion, minLeaf, K);
      if (!best) {
        node.status = 'leaf';
        continue;
      }
      node.feature = best.feature;
      node.threshold = best.threshold;
      node.gain = best.gain;
      node.status = 'split';
      const left = [];
      const right = [];
      for (const r of rows) (X[r][best.feature] < best.threshold ? left : right).push(r);
      node.left = make(left, node.depth + 1);
      node.right = make(right, node.depth + 1);
      queue.push(node.left, node.right);
      return true;
    }
    return false;
  };
  const grow = () => {
    while (step()) { /* keep splitting */ }
    return root;
  };
  return { root, queue, step, grow };
}

/** Grow a whole tree; returns the root. */
export function fitTree(X, y, K, opts = {}) {
  return makeBuilder({ X, y, K, ...opts }).grow();
}

/** The leaf a feature row lands in. */
export function leafOf(root, x) {
  let node = root;
  while (node.left) node = x[node.feature] < node.threshold ? node.left : node.right;
  return node;
}

/** Leaf class distribution, counts / n (same as predict_proba of a leaf). */
export function leafProba(node) {
  let n = 0;
  for (const c of node.counts) n += c;
  return node.counts.map((c) => c / n);
}

/** Majority class of a node; ties go to the lowest index. */
export function nodeLabel(node) {
  let best = 0;
  for (let c = 1; c < node.counts.length; c++) if (node.counts[c] > node.counts[best]) best = c;
  return best;
}

/** Pre-order (left first) list of the nodes of a tree. */
export function preorder(root) {
  const out = [];
  const stack = [root];
  while (stack.length) {
    const node = stack.pop();
    out.push(node);
    if (node.right) { stack.push(node.right); stack.push(node.left); }
  }
  return out;
}

export function nLeaves(root) {
  let n = 0;
  for (const node of preorder(root)) if (!node.left) n++;
  return n;
}

export function treeDepth(root) {
  let d = 0;
  for (const node of preorder(root)) if (node.depth > d) d = node.depth;
  return d;
}

/** Deep copy of the structure, without the frontier rows. */
export function copyTree(root) {
  const clone = (node) => {
    const c = {
      id: node.id, depth: node.depth, counts: node.counts.slice(), impurity: node.impurity,
      feature: node.feature, threshold: node.threshold, gain: node.gain, left: null, right: null,
      status: node.status, idx: null,
    };
    if (node.left) {
      c.left = clone(node.left);
      c.right = clone(node.right);
    }
    return c;
  };
  return clone(root);
}

function subtreeRisk(node, total) {
  if (!node.left) {
    let mx = 0;
    for (const c of node.counts) if (c > mx) mx = c;
    let n = 0;
    for (const c of node.counts) n += c;
    return [(n - mx) / total, 1];
  }
  const [lr, ll] = subtreeRisk(node.left, total);
  const [rr, rl] = subtreeRisk(node.right, total);
  return [lr + rr, ll + rl];
}

/** Cost-complexity pruning (weakest link). Returns a pruned copy; the input is not changed. See prune() in tree.py. */
export function prune(root, alpha) {
  const t = copyTree(root);
  let total = 0;
  for (const c of t.counts) total += c;
  for (;;) {
    let weakest = null;
    let bestG = null;
    for (const node of preorder(t)) {
      if (!node.left) continue;
      let mx = 0;
      let n = 0;
      for (const c of node.counts) { n += c; if (c > mx) mx = c; }
      const rNode = (n - mx) / total;
      const [rSub, leaves] = subtreeRisk(node, total);
      const g = (rNode - rSub) / (leaves - 1);
      if (bestG === null || g < bestG) { weakest = node; bestG = g; }
    }
    if (!weakest || bestG > alpha) return t;
    weakest.left = null;
    weakest.right = null;
    weakest.status = 'leaf';
    weakest.feature = -1;
  }
}

/** g(t) for every internal node in pre-order (for picking alpha slider values). */
export function linkStrengths(root) {
  let total = 0;
  for (const c of root.counts) total += c;
  const out = [];
  for (const node of preorder(root)) {
    if (!node.left) continue;
    let mx = 0;
    let n = 0;
    for (const c of node.counts) { n += c; if (c > mx) mx = c; }
    const rNode = (n - mx) / total;
    const [rSub, leaves] = subtreeRisk(node, total);
    out.push((rNode - rSub) / (leaves - 1));
  }
  return out;
}

/** Expand [x, y] rows to features: [x, y] plus, when extras, [x*y, x*x + y*y]. */
export function expand(points, extras = true) {
  return points.map(([x, y]) => (extras ? [x, y, x * y, x * x + y * y] : [x, y]));
}

function disc(rng) {
  for (;;) {
    const u = 2 * rng.next() - 1;
    const v = 2 * rng.next() - 1;
    if (u * u + v * v <= 1) return [u, v];
  }
}

/** Seeded 2-D preset: {X: [[x, y]...] in [0, 1]^2, y: [labels]}. Same draws and formulas as data.make_preset. */
export function makePreset(name, n = 240, seed = 1) {
  if (!(name in PRESETS)) throw new Error(`unknown preset ${name}`);
  const rng = mulberry32(seed);
  const X = [];
  const y = [];
  for (let i = 0; i < n; i++) {
    if (name === 'blobs') {
      const c = i % 3;
      const [u, v] = disc(rng);
      X.push([BLOB_CENTRES[c][0] + BLOB_RADIUS * u, BLOB_CENTRES[c][1] + BLOB_RADIUS * v]);
      y.push(c);
    } else if (name === 'xor') {
      const x = rng.next();
      const yy = rng.next();
      X.push([x, yy]);
      y.push((x > 0.5) !== (yy > 0.5) ? 1 : 0);
    } else if (name === 'checkerboard') {
      const x = rng.next();
      const yy = rng.next();
      X.push([x, yy]);
      y.push((Math.floor(x * 4) + Math.floor(yy * 4)) % 2);
    } else if (name === 'spiral') {
      const arm = i % 2;
      const t = rng.next();
      const theta = 3.5 * Math.PI * t + arm * Math.PI;
      const rad = 0.04 + 0.4 * t;
      const jx = 0.02 * (2 * rng.next() - 1);
      const jy = 0.02 * (2 * rng.next() - 1);
      X.push([0.5 + rad * Math.cos(theta) + jx, 0.5 + rad * Math.sin(theta) + jy]);
      y.push(arm);
    } else if (name === 'nested') {
      const x = rng.next();
      const yy = rng.next();
      const d2 = (x - 0.5) * (x - 0.5) + (yy - 0.5) * (yy - 0.5);
      X.push([x, yy]);
      y.push(d2 < RING_R2[0] ? 0 : (d2 < RING_R2[1] ? 1 : 2));
    } else { // diagonal
      const x = rng.next();
      const yy = rng.next();
      let lab = x > yy ? 1 : 0;
      if (rng.next() < FLIP) lab = 1 - lab;
      X.push([x, yy]);
      y.push(lab);
    }
  }
  return { X, y };
}

/** Seeded shuffle of 0..n-1; the first round(frac * n) indices are the test set. Returns {train, test}. */
export function trainTestSplit(n, testFraction, seed = 1) {
  const rng = mulberry32(seed);
  const perm = Array.from({ length: n }, (_, i) => i);
  for (let i = n - 1; i > 0; i--) {
    const j = rng.randInt(i + 1);
    const t = perm[i]; perm[i] = perm[j]; perm[j] = t;
  }
  const nTest = Math.floor(testFraction * n + 0.5);
  return { train: perm.slice(nTest), test: perm.slice(0, nTest) };
}

/**
 * Random forest (forest.RandomForest). Each tree's seed comes from a master stream; the tree draws its bootstrap
 * rows first, then one feature subset per split it considers.
 * Returns {trees: [{root, inBag}], nFeatures, K, oob, importances}.
 */
export function fitForest(X, y, K, { nTrees = 25, criterion = 'gini', maxDepth = 5, minLeaf = 1,
  maxFeatures = null, bootstrap = true, seed = 1 } = {}) {
  const n = y.length;
  const p = X.length ? X[0].length : 0;
  const mf = maxFeatures != null ? maxFeatures : Math.max(1, Math.floor(Math.sqrt(p)));
  const master = mulberry32(seed);
  const trees = [];
  for (let t = 0; t < nTrees; t++) {
    const rng = mulberry32(Math.floor(master.next() * 4294967296));
    const idx = [];
    if (bootstrap) for (let i = 0; i < n; i++) idx.push(rng.randInt(n));
    else for (let i = 0; i < n; i++) idx.push(i);
    const inBag = new Uint8Array(n);
    for (const r of idx) inBag[r] = 1;
    const root = makeBuilder({ X, y, K, criterion, maxDepth, minLeaf, rng, maxFeatures: mf, idx }).grow();
    trees.push({ root, inBag });
  }
  const forest = { trees, nFeatures: p, K, nTrees, oob: null, importances: null };
  scoreOob(forest, X, y);
  forest.importances = importancesOf(forest);
  return forest;
}

/** Averaged leaf distribution of the forest for one feature row: sum over trees in order, then / T. */
export function forestProba(forest, x) {
  const total = new Array(forest.K).fill(0);
  for (const { root } of forest.trees) {
    const pr = leafProba(leafOf(root, x));
    for (let c = 0; c < forest.K; c++) total[c] = total[c] + pr[c];
  }
  return total.map((v) => v / forest.trees.length);
}

export function argmax(arr) {
  let best = 0;
  for (let c = 1; c < arr.length; c++) if (arr[c] > arr[best]) best = c;
  return best;
}

function scoreOob(forest, X, y) {
  const n = y.length;
  const K = forest.K;
  const acc = Array.from({ length: n }, () => new Array(K).fill(0));
  const seen = new Array(n).fill(0);
  for (const { root, inBag } of forest.trees) {
    for (let i = 0; i < n; i++) {
      if (inBag[i]) continue;
      const pr = leafProba(leafOf(root, X[i]));
      for (let c = 0; c < K; c++) acc[i][c] = acc[i][c] + pr[c];
      seen[i] += 1;
    }
  }
  let scored = 0;
  let correct = 0;
  for (let i = 0; i < n; i++) {
    if (!seen[i]) continue;
    scored++;
    const avg = acc[i].map((v) => v / seen[i]);
    if (argmax(avg) === y[i]) correct++;
  }
  forest.oob = scored ? correct / scored : null;
}

function importancesOf(forest) {
  const total = new Array(forest.nFeatures).fill(0);
  for (const { root } of forest.trees) {
    const imp = new Array(forest.nFeatures).fill(0);
    for (const node of preorder(root)) {
      if (node.left) imp[node.feature] += nodeN(node) * node.gain;
    }
    let s = 0;
    for (const v of imp) s += v;
    for (let f = 0; f < forest.nFeatures; f++) total[f] += s > 0 ? imp[f] / s : imp[f];
  }
  return total.map((v) => v / forest.trees.length);
}

/** Number of training rows (bootstrap copies included) that reached a node. */
export function nodeN(node) {
  let n = 0;
  for (const c of node.counts) n += c;
  return n;
}
