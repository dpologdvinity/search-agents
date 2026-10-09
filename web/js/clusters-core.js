// Clustering core: a line-for-line port of clusters/ (rng, data, kmeans, dbscan, gmm, metrics).
//
// Pure JavaScript with no DOM access, so the page and node (tests/test_clusters_parity.py) can both load it.
// Points are flat Float64Arrays: point i is (P[2i], P[2i+1]), centres are flat arrays of length 2k. The
// same mulberry32 stream drives presets and seeding, and every tie goes to the lower index, so the results
// match the Python reference to floating-point rounding. Python is the reference: fix this file if they differ.
//
// Generators (kmeansSteps, dbscanSteps, gmmSteps) yield each step so the page can animate them; the
// plain functions (kmeans, dbscan, gmmEM) run the same generators to the end.

// ── seeded PRNG ─────────────────────────────────────────────────────────

// mulberry32, with Math.imul keeping the low 32 bits of each product like Python's `& 0xFFFFFFFF`.
export class Mulberry32 {
  constructor(seed) {
    this.a = seed >>> 0;
  }
  random() {
    this.a = (this.a + 0x6d2b79f5) >>> 0;
    let t = this.a;
    t = Math.imul(t ^ (t >>> 15), t | 1) >>> 0;
    t = (t ^ ((t + (Math.imul(t ^ (t >>> 7), t | 61) >>> 0)) >>> 0)) >>> 0;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  }
  uniform(lo, hi) {
    return lo + (hi - lo) * this.random();
  }
  // Box-Muller; always two uniforms per normal, as in clusters/rng.py.
  normal() {
    let u1 = this.random();
    const u2 = this.random();
    if (u1 < 1e-12) u1 = 1e-12;
    return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
  }
  index(n) {
    const j = Math.floor(this.random() * n);
    return j < n ? j : n - 1;
  }
}

// ── presets (clusters/data.py) ──────────────────────────────────────────

export const PRESETS = ['blobs', 'aniso', 'rings', 'moons', 'uniform', 'smiley'];
const BLOB_CENTRES = [[0.25, 0.3], [0.75, 0.25], [0.3, 0.75], [0.72, 0.72]];
const SHEAR = [[0.6, -0.6], [-0.4, 0.8]];

function split(n, parts) {
  const base = Math.floor(n / parts);
  const extra = n % parts;
  const out = [];
  for (let i = 0; i < parts; i++) out.push(base + (i < extra ? 1 : 0));
  return out;
}

const clamp01 = (v) => (v < 0 ? 0 : v > 1 ? 1 : v);

/** Points in the unit square and the generating component of each point. */
export function makeDataset(name, seed = 0, n = 300) {
  if (!PRESETS.includes(name)) throw new Error(`unknown preset ${name}`);
  const rng = new Mulberry32(seed);
  const P = new Float64Array(2 * n);
  const truth = new Int32Array(n);
  let m = 0;
  const emit = (comp, x, y) => {
    P[2 * m] = clamp01(x);
    P[2 * m + 1] = clamp01(y);
    truth[m] = comp;
    m++;
  };
  if (name === 'blobs' || name === 'aniso') {
    let centres;
    if (name === 'blobs') {
      centres = BLOB_CENTRES.map(([cx, cy]) => [cx + rng.uniform(-0.04, 0.04), cy + rng.uniform(-0.04, 0.04)]);
    } else {
      centres = [[0.25, 0.3], [0.7, 0.3], [0.5, 0.75]];
    }
    const counts = split(n, centres.length);
    centres.forEach(([cx, cy], comp) => {
      for (let t = 0; t < counts[comp]; t++) {
        let gx = rng.normal() * 0.05;
        let gy = rng.normal() * 0.05;
        if (name === 'aniso') {
          const [[a, b], [c, d]] = SHEAR;
          [gx, gy] = [a * gx + b * gy, c * gx + d * gy];
        }
        emit(comp, cx + gx, cy + gy);
      }
    });
  } else if (name === 'rings') {
    const counts = split(n, 2);
    [0.2, 0.42].forEach((radius, comp) => {
      for (let t = 0; t < counts[comp]; t++) {
        const a = rng.uniform(0, 2 * Math.PI);
        const r = radius + rng.normal() * 0.02;
        emit(comp, 0.5 + r * Math.cos(a), 0.5 + r * Math.sin(a));
      }
    });
  } else if (name === 'moons') {
    const counts = split(n, 2);
    for (let comp = 0; comp < 2; comp++) {
      for (let t = 0; t < counts[comp]; t++) {
        const a = rng.uniform(0, Math.PI);
        let x;
        let y;
        if (comp === 0) {
          x = 0.35 + 0.28 * Math.cos(a);
          y = 0.45 + 0.28 * Math.sin(a);
        } else {
          x = 0.65 - 0.28 * Math.cos(a);
          y = 0.55 - 0.28 * Math.sin(a);
        }
        emit(comp, x + rng.normal() * 0.025, y + rng.normal() * 0.025);
      }
    }
  } else if (name === 'uniform') {
    for (let i = 0; i < n; i++) emit(0, rng.random(), rng.random());
  } else {
    const counts = split(n, 4);
    for (let t = 0; t < counts[0]; t++) emit(0, 0.36 + rng.normal() * 0.025, 0.66 + rng.normal() * 0.025);
    for (let t = 0; t < counts[1]; t++) emit(1, 0.64 + rng.normal() * 0.025, 0.66 + rng.normal() * 0.025);
    for (let t = 0; t < counts[2]; t++) {
      const a = rng.uniform(0.15 * Math.PI, 0.85 * Math.PI);
      emit(2, 0.5 + 0.22 * Math.cos(a) + rng.normal() * 0.012, 0.5 - 0.22 * Math.sin(a) + rng.normal() * 0.012);
    }
    for (let t = 0; t < counts[3]; t++) {
      const a = rng.uniform(0, 2 * Math.PI);
      emit(3, 0.5 + 0.42 * Math.cos(a) + rng.normal() * 0.012, 0.5 + 0.42 * Math.sin(a) + rng.normal() * 0.012);
    }
  }
  return { points: P, truth };
}

// ── k-means (clusters/kmeans.py) ────────────────────────────────────────

/** Nearest centre per point (squared distance; ties go to the lower index). */
export function assign(P, C, k) {
  const n = P.length >> 1;
  const lab = new Int32Array(n);
  for (let i = 0; i < n; i++) {
    const x = P[2 * i];
    const y = P[2 * i + 1];
    let best = 0;
    let bd = Infinity;
    for (let j = 0; j < k; j++) {
      const dx = x - C[2 * j];
      const dy = y - C[2 * j + 1];
      const d = dx * dx + dy * dy;
      if (d < bd) {
        bd = d;
        best = j;
      }
    }
    lab[i] = best;
  }
  return lab;
}

/** Each centre moves to the mean of its points; an empty centre stays put. */
export function update(P, labels, C, k) {
  const out = Float64Array.from(C);
  const n = labels.length;
  for (let j = 0; j < k; j++) {
    let sx = 0;
    let sy = 0;
    let count = 0;
    for (let i = 0; i < n; i++) {
      if (labels[i] === j) {
        sx += P[2 * i];
        sy += P[2 * i + 1];
        count++;
      }
    }
    if (count) {
      out[2 * j] = sx / count;
      out[2 * j + 1] = sy / count;
    }
  }
  return out;
}

/** Within-cluster sum of squared errors (the inertia). */
export function sse(P, labels, C) {
  let s = 0;
  for (let i = 0; i < labels.length; i++) {
    const j = labels[i];
    const dx = P[2 * i] - C[2 * j];
    const dy = P[2 * i + 1] - C[2 * j + 1];
    s += dx * dx + dy * dy;
  }
  return s;
}

/** k-means++: first centre uniform, each next one drawn with probability proportional to D^2. */
export function kmeansPP(P, k, rng) {
  const n = P.length >> 1;
  const C = new Float64Array(2 * k);
  const first = rng.index(n);
  C[0] = P[2 * first];
  C[1] = P[2 * first + 1];
  const d2 = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    const dx = P[2 * i] - C[0];
    const dy = P[2 * i + 1] - C[1];
    d2[i] = dx * dx + dy * dy;
  }
  for (let c = 1; c < k; c++) {
    let total = 0;
    const cum = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      total += d2[i];
      cum[i] = total;
    }
    let j;
    if (total <= 0) {
      j = rng.index(n);
    } else {
      const r = rng.random() * total;
      j = 0;
      while (j < n - 1 && !(cum[j] > r)) j++; // first index with cum > r (searchsorted side='right')
    }
    C[2 * c] = P[2 * j];
    C[2 * c + 1] = P[2 * j + 1];
    for (let i = 0; i < n; i++) {
      const dx = P[2 * i] - P[2 * j];
      const dy = P[2 * i + 1] - P[2 * j + 1];
      const d = dx * dx + dy * dy;
      if (d < d2[i]) d2[i] = d;
    }
  }
  return C;
}

/** k distinct data points from a seeded Fisher-Yates shuffle. */
export function randomInit(P, k, rng) {
  const n = P.length >> 1;
  const order = Array.from({ length: n }, (_, i) => i);
  for (let i = n - 1; i > 0; i--) {
    const j = rng.index(i + 1);
    [order[i], order[j]] = [order[j], order[i]];
  }
  const C = new Float64Array(2 * k);
  for (let j = 0; j < k; j++) {
    C[2 * j] = P[2 * order[j]];
    C[2 * j + 1] = P[2 * order[j] + 1];
  }
  return C;
}

/** Generator: init, then assign/update pairs until an assign leaves labels unchanged (or maxIter). */
export function* kmeansSteps(P, k, { seed = 0, init = 'kmeans++', centers = null, maxIter = 100 } = {}) {
  const rng = new Mulberry32(seed);
  let C;
  if (centers) C = Float64Array.from(centers);
  else if (init === 'kmeans++') C = kmeansPP(P, k, rng);
  else C = randomInit(P, k, rng);
  yield { phase: 'init', iteration: 0, labels: null, centers: Float64Array.from(C), inertia: null };
  let labels = null;
  for (let it = 1; it <= maxIter; it++) {
    const next = assign(P, C, k);
    yield { phase: 'assign', iteration: it, labels: next, centers: Float64Array.from(C), inertia: sse(P, next, C) };
    if (labels && next.every((v, i) => v === labels[i])) return;
    labels = next;
    C = update(P, labels, C, k);
    yield { phase: 'update', iteration: it, labels, centers: Float64Array.from(C), inertia: sse(P, labels, C) };
  }
}

/** Run k-means to the end; the same result as clusters.kmeans.kmeans(). */
export function kmeans(P, k, opts = {}) {
  let last = null;
  const history = [];
  for (const step of kmeansSteps(P, k, opts)) {
    if (step.phase === 'assign') history.push(step.inertia);
    last = step;
  }
  const converged = last.phase === 'assign';
  let labels;
  const centers = last.centers;
  let inertia;
  if (converged) {
    labels = last.labels;
    inertia = history[history.length - 1];
  } else {
    labels = assign(P, centers, k);
    inertia = sse(P, labels, centers);
  }
  return { labels, centers, inertia, iterations: history.length, converged, history };
}

/** Inertia for k = 1..kMax, all with the same seed. */
export function elbow(P, kMax = 10, seed = 0) {
  const out = [];
  for (let k = 1; k <= kMax; k++) out.push([k, kmeans(P, k, { seed }).inertia]);
  return out;
}

// ── DBSCAN (clusters/dbscan.py) ─────────────────────────────────────────

export const NOISE = -1;

/** For each point, the ascending indices of all points within eps (self included). */
export function neighbourhoods(P, eps) {
  const n = P.length >> 1;
  const limit = eps * eps;
  const out = [];
  for (let i = 0; i < n; i++) {
    const nb = [];
    const x = P[2 * i];
    const y = P[2 * i + 1];
    for (let j = 0; j < n; j++) {
      const dx = P[2 * j] - x;
      const dy = P[2 * j + 1] - y;
      if (dx * dx + dy * dy <= limit) nb.push(j);
    }
    out.push(nb);
  }
  return out;
}

/** Generator form of DBSCAN: yields each point as it joins a cluster; returns {labels, core, clusters}. */
export function* dbscanSteps(P, eps, minPts) {
  const nbrs = neighbourhoods(P, eps);
  const n = nbrs.length;
  const core = new Uint8Array(n);
  for (let i = 0; i < n; i++) core[i] = nbrs[i].length >= minPts ? 1 : 0;
  const labels = new Int32Array(n).fill(NOISE);
  let cid = 0;
  for (let i = 0; i < n; i++) {
    if (!core[i] || labels[i] !== NOISE) continue;
    labels[i] = cid;
    yield { i, cluster: cid, border: false, labels };
    const queue = [i];
    let head = 0;
    while (head < queue.length) { // flood fill over core points only
      const p = queue[head++];
      for (const q of nbrs[p]) {
        if (core[q] && labels[q] === NOISE) {
          labels[q] = cid;
          queue.push(q);
          yield { i: q, cluster: cid, border: false, labels };
        }
      }
    }
    cid++;
  }
  for (let i = 0; i < n; i++) {
    if (core[i]) continue;
    for (const q of nbrs[i]) {
      if (core[q]) {
        labels[i] = labels[q];
        yield { i, cluster: labels[i], border: true, labels };
        break;
      }
    }
  }
  return { labels, core, clusters: cid };
}

export function dbscan(P, eps, minPts) {
  const gen = dbscanSteps(P, eps, minPts);
  let r = gen.next();
  while (!r.done) r = gen.next();
  return r.value;
}

// ── Gaussian mixture by EM (clusters/gmm.py) ────────────────────────────

const LOG_2PI = Math.log(2 * Math.PI);

/** log N(x | mu, S) for every point; S is [a, b, c, d] row-major, inverted in closed form. */
function logPdf(P, mx, my, S) {
  const [a, b, c, d] = S;
  const det = a * d - b * c;
  const ia = d / det;
  const ib = -b / det;
  const ic = -c / det;
  const id = a / det;
  const n = P.length >> 1;
  const out = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    const dx = P[2 * i] - mx;
    const dy = P[2 * i + 1] - my;
    const maha = dx * (ia * dx + ib * dy) + dy * (ic * dx + id * dy);
    out[i] = -0.5 * (2 * LOG_2PI + Math.log(det) + maha);
  }
  return out;
}

/** Weights, means and covariances from responsibilities (resp is row-major n x k). */
function mStep(P, resp, k, reg) {
  const n = P.length >> 1;
  const weights = new Float64Array(k);
  const means = new Float64Array(2 * k);
  const covs = new Float64Array(4 * k);
  for (let j = 0; j < k; j++) {
    let nk = 0;
    let sx = 0;
    let sy = 0;
    for (let i = 0; i < n; i++) {
      const r = resp[i * k + j];
      nk += r;
      sx += r * P[2 * i];
      sy += r * P[2 * i + 1];
    }
    nk = Math.max(nk, 1e-10);
    weights[j] = nk / n;
    const mx = sx / nk;
    const my = sy / nk;
    means[2 * j] = mx;
    means[2 * j + 1] = my;
    let a = 0;
    let b = 0;
    let d = 0;
    for (let i = 0; i < n; i++) {
      const r = resp[i * k + j];
      const dx = P[2 * i] - mx;
      const dy = P[2 * i + 1] - my;
      a += r * dx * dx;
      b += r * dx * dy;
      d += r * dy * dy;
    }
    covs[4 * j] = a / nk + reg;
    covs[4 * j + 1] = b / nk;
    covs[4 * j + 2] = b / nk;
    covs[4 * j + 3] = d / nk + reg;
  }
  return { weights, means, covs };
}

/** Generator: each E step yields the responsibilities under the current parameters, then each M step. */
export function* gmmSteps(P, k, { seed = 0, init = 'kmeans++', maxIter = 200, tol = 1e-6, reg = 1e-6 } = {}) {
  const n = P.length >> 1;
  const rng = new Mulberry32(seed);
  const centres = init === 'kmeans++' ? kmeansPP(P, k, rng) : randomInit(P, k, rng);
  const hard = assign(P, centres, k);
  const resp = new Float64Array(n * k);
  for (let i = 0; i < n; i++) resp[i * k + hard[i]] = 1;
  let { weights, means, covs } = mStep(P, resp, k, reg);
  let prev = -Infinity;
  let converged = false;
  let iterations = 0;
  const history = [];
  for (let it = 1; it <= maxIter; it++) {
    const logp = new Float64Array(n * k);
    for (let j = 0; j < k; j++) {
      const lp = logPdf(P, means[2 * j], means[2 * j + 1], covs.subarray(4 * j, 4 * j + 4));
      for (let i = 0; i < n; i++) logp[i * k + j] = Math.log(weights[j]) + lp[i];
    }
    const lognorm = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      let m = -Infinity;
      for (let j = 0; j < k; j++) if (logp[i * k + j] > m) m = logp[i * k + j];
      let s = 0;
      for (let j = 0; j < k; j++) s += Math.exp(logp[i * k + j] - m);
      lognorm[i] = m + Math.log(s);
      for (let j = 0; j < k; j++) resp[i * k + j] = Math.exp(logp[i * k + j] - lognorm[i]);
    }
    let total = 0;
    for (let i = 0; i < n; i++) total += lognorm[i];
    const ll = total / n;
    history.push(ll);
    iterations = it;
    yield { phase: 'e', iteration: it, resp, weights: Float64Array.from(weights), means: Float64Array.from(means),
            covs: Float64Array.from(covs), loglik: ll };
    if (ll - prev < tol && it > 1) {
      converged = true;
      break;
    }
    prev = ll;
    // Out of iterations: stop before the M step, so the returned parameters are the ones that produced resp,
    // labels and loglik. An M step here would leave them one update ahead (same rule as clusters/gmm.py).
    if (it === maxIter) break;
    ({ weights, means, covs } = mStep(P, resp, k, reg));
    yield { phase: 'm', iteration: it, resp, weights: Float64Array.from(weights), means: Float64Array.from(means),
            covs: Float64Array.from(covs), loglik: ll };
  }
  const labels = new Int32Array(n);
  for (let i = 0; i < n; i++) {
    let best = 0;
    for (let j = 1; j < k; j++) if (resp[i * k + j] > resp[i * k + best]) best = j;
    labels[i] = best;
  }
  return { weights, means, covs, resp, labels, loglik: history[history.length - 1], iterations, converged, history };
}

export function gmmEM(P, k, opts = {}) {
  const gen = gmmSteps(P, k, opts);
  let r = gen.next();
  while (!r.done) r = gen.next();
  return r.value;
}

/** Semi-axes of the sigma-contour and the angle of a 2x2 covariance [a, b, c, d], for drawing ellipses. */
export function ellipse(cov, sigma = 2) {
  const [a, b, , d] = cov;
  const tr = a + d;
  const det = a * d - b * b;
  const disc = Math.sqrt(Math.max((tr * tr) / 4 - det, 0));
  const l1 = tr / 2 + disc;
  const l2 = tr / 2 - disc;
  return {
    major: sigma * Math.sqrt(Math.max(l1, 0)),
    minor: sigma * Math.sqrt(Math.max(l2, 0)),
    angle: 0.5 * Math.atan2(2 * b, a - d),
  };
}

// ── silhouette (clusters/metrics.py) ────────────────────────────────────

/** Mean silhouette over non-noise points, or null with fewer than two clusters. */
export function silhouette(P, labels) {
  const idx = [];
  for (let i = 0; i < labels.length; i++) if (labels[i] >= 0) idx.push(i);
  const lab = idx.map((i) => labels[i]);
  const ids = [...new Set(lab)];
  if (ids.length < 2) return null;
  const n = idx.length;
  const D = new Float64Array(n * n);
  for (let a = 0; a < n; a++) {
    for (let b = 0; b < n; b++) {
      const dx = P[2 * idx[a]] - P[2 * idx[b]];
      const dy = P[2 * idx[a] + 1] - P[2 * idx[b] + 1];
      D[a * n + b] = Math.sqrt(dx * dx + dy * dy);
    }
  }
  let total = 0;
  for (let i = 0; i < n; i++) {
    let sameSum = 0;
    let nSame = 0;
    for (let j = 0; j < n; j++) {
      if (lab[j] === lab[i]) {
        sameSum += D[i * n + j];
        nSame++;
      }
    }
    if (nSame <= 1) continue; // singleton scores 0
    const a = sameSum / (nSame - 1);
    let b = Infinity;
    for (const id of ids) {
      if (id === lab[i]) continue;
      let s = 0;
      let c = 0;
      for (let j = 0; j < n; j++) {
        if (lab[j] === id) {
          s += D[i * n + j];
          c++;
        }
      }
      const mean = s / c;
      if (mean < b) b = mean;
    }
    const m = Math.max(a, b);
    total += m > 0 ? (b - a) / m : 0;
  }
  return total / n;
}
