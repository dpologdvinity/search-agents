// Neural network lab: the pure maths, with no DOM.
//
// This file is a port of the nnlab Python package. Random numbers (mulberry32), dataset generators,
// features, the MLP forward pass, hand-written backprop, Adam and SGD, and the epoch loop all follow
// the Python code in the same order of operations, so a seed gives the same points, weights and
// curves in both places. tests/test_nnlab_parity.py runs this file under node and checks the two
// agree to 1e-9.
//
// Layout: a weight matrix for layer l is a Float64Array of shape out x in, row-major, so the weight
// from input i to unit j is W[l][j * in + i]. Activations for n samples are n x dim, row-major.

export const PRESETS = ['blobs', 'xor', 'circles', 'spiral', 'moons'];
export const FEATURES = ['x', 'y', 'x2', 'y2', 'xy', 'sinx', 'siny'];
export const ACTIVATIONS = ['tanh', 'relu', 'sigmoid'];
export const OPTIMIZERS = ['adam', 'sgd'];

/** Defaults, as in nnlab/config.py. */
export const DEFAULTS = {
  data: 'circles', n: 200, seed: 0, noise: 0, features: ['x', 'y'], testFrac: 0.2,
  hidden: [6, 6], act: 'tanh', lr: 0.03, batch: 16, l2: 0, optimizer: 'adam', epochs: 300,
};

const TWO_PI = 2 * Math.PI;

/**
 * mulberry32: a 32-bit generator. Math.imul gives the low 32 bits of a product, and >>> 0 reads a
 * value as unsigned, so the bits match Python's masked integer arithmetic exactly.
 */
export class Rng {
  constructor(seed) { this.a = seed >>> 0; }

  random() {
    this.a = (this.a + 0x6d2b79f5) >>> 0;
    let t = this.a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  }

  /** Standard normal by Box-Muller. Uses two uniforms and discards the second Box-Muller value. */
  normal() {
    const u1 = Math.max(this.random(), 1e-12);
    const u2 = this.random();
    return Math.sqrt(-2 * Math.log(u1)) * Math.cos(TWO_PI * u2);
  }
}

// ── Datasets ──────────────────────────────────────────────────────────────

/** Base position and label of point i of a preset. Labels alternate (0 cyan, 1 pink), except XOR. */
function basePoint(name, i, rng) {
  if (name === 'blobs') {
    const label = i % 2;
    const c = label ? 0.45 : -0.45;
    const x = c + 0.25 * rng.normal();
    const y = c + 0.25 * rng.normal();
    return [x, y, label];
  }
  if (name === 'xor') {
    const cluster = i % 4;
    const sx = (cluster & 1) ? 0.5 : -0.5;
    const sy = (cluster & 2) ? 0.5 : -0.5;
    const label = ((sx > 0) === (sy > 0)) ? 1 : 0;
    const x = sx + 0.2 * rng.normal();
    const y = sy + 0.2 * rng.normal();
    return [x, y, label];
  }
  if (name === 'circles') {
    const label = i % 2;
    const r = (label ? 0.85 : 0.4) + 0.05 * rng.normal();
    const a = TWO_PI * rng.random();
    return [r * Math.cos(a), r * Math.sin(a), label];
  }
  if (name === 'spiral') {
    const label = i % 2;
    const t = rng.random();
    const a = 3.2 * Math.PI * t + label * Math.PI;
    const r = 0.1 + 0.8 * t;
    const x = r * Math.cos(a) + 0.03 * rng.normal();
    const y = r * Math.sin(a) + 0.03 * rng.normal();
    return [x, y, label];
  }
  if (name === 'moons') {
    const label = i % 2;
    const t = Math.PI * rng.random();
    let x, y;
    if (label === 0) { x = 0.6 * Math.cos(t) - 0.3; y = 0.6 * Math.sin(t) - 0.15; }
    else { x = -0.6 * Math.cos(t) + 0.3; y = -0.6 * Math.sin(t) + 0.15; }
    return [x + 0.06 * rng.normal(), y + 0.06 * rng.normal(), label];
  }
  throw new Error(`unknown dataset ${name}`);
}

/**
 * n points of a preset as {x, y, label, jx, jy, u} objects. Per point: base position, then jitter
 * (two normals), then the split draw u. The effective position is x + noise * jx.
 */
export function generatePoints(name, n, seed) {
  const rng = new Rng(seed);
  const points = [];
  for (let i = 0; i < n; i++) {
    const [x, y, label] = basePoint(name, i, rng);
    const jx = rng.normal();
    const jy = rng.normal();
    const u = rng.random();
    points.push({ x, y, label, jx, jy, u });
  }
  return points;
}

/** The chosen input features of one point, in the order given. */
export function featurize(x, y, names) {
  const table = {
    x, y,
    x2: x * x, y2: y * y, xy: x * y,
    sinx: Math.sin(Math.PI * x), siny: Math.sin(Math.PI * y),
  };
  return names.map((k) => table[k]);
}

/**
 * Training and test matrices from the points. A point is a test point when u < testFrac.
 * Returns flat Float64Arrays: Xtr (ntr x F), ytr (ntr), Xte (nte x F), yte (nte).
 */
export function buildSplit(points, features, noise, testFrac) {
  const F = features.length;
  const tr = [], trY = [], te = [], teY = [];
  for (const p of points) {
    const row = featurize(p.x + noise * p.jx, p.y + noise * p.jy, features);
    if (p.u < testFrac) { te.push(...row); teY.push(p.label); }
    else { tr.push(...row); trY.push(p.label); }
  }
  return {
    F,
    ntr: trY.length, Xtr: Float64Array.from(tr), ytr: Float64Array.from(trY),
    nte: teY.length, Xte: Float64Array.from(te), yte: Float64Array.from(teY),
  };
}

// ── Network ───────────────────────────────────────────────────────────────

/**
 * Initial weights: Glorot uniform, or He's rule for ReLU hidden layers (sqrt(6 / fan_in)), and zero
 * biases. Same order of draws as the Python init: output j outer, input i inner.
 */
export function initNet(sizes, act, rng) {
  const W = [], b = [];
  const L = sizes.length - 1;
  for (let l = 0; l < L; l++) {
    const fin = sizes[l], fout = sizes[l + 1];
    const isHidden = l < L - 1;
    const scale = (act === 'relu' && isHidden) ? Math.sqrt(6 / fin) : Math.sqrt(6 / (fin + fout));
    const w = new Float64Array(fout * fin);
    for (let j = 0; j < fout; j++) {
      for (let i = 0; i < fin; i++) w[j * fin + i] = scale * (2 * rng.random() - 1);
    }
    W.push(w);
    b.push(new Float64Array(fout));
  }
  return { sizes: sizes.slice(), W, b, act };
}

function activate(name, z) {
  if (name === 'tanh') return Math.tanh(z);
  if (name === 'relu') return z > 0 ? z : 0;
  return 1 / (1 + Math.exp(-z)); // sigmoid
}

/** f'(z) from the output a = f(z): tanh 1 - a^2, sigmoid a(1 - a), relu 1 when a > 0. */
function activationGrad(name, a) {
  if (name === 'tanh') return 1 - a * a;
  if (name === 'sigmoid') return a * (1 - a);
  return a > 0 ? 1 : 0;
}

const sigmoid = (z) => 1 / (1 + Math.exp(-z));

/**
 * Forward pass for n samples. Returns {A, Z}: A[l] is the input to layer l (A[0] = X, the last entry
 * holds the predicted probabilities), Z[l] the pre-activations of layer l.
 */
export function forward(net, X, n) {
  const A = [X], Z = [];
  const L = net.W.length;
  for (let l = 0; l < L; l++) {
    const din = net.sizes[l], dout = net.sizes[l + 1];
    const a = A[l], w = net.W[l], bias = net.b[l];
    const z = new Float64Array(n * dout);
    const out = new Float64Array(n * dout);
    const last = l === L - 1;
    for (let s = 0; s < n; s++) {
      for (let j = 0; j < dout; j++) {
        let acc = 0;
        for (let i = 0; i < din; i++) acc += w[j * din + i] * a[s * din + i];
        const zv = acc + bias[j];
        z[s * dout + j] = zv;
        out[s * dout + j] = last ? sigmoid(zv) : activate(net.act, zv);
      }
    }
    Z.push(z);
    A.push(out);
  }
  return { A, Z };
}

/** Predicted probability of the pink class for n samples. */
export function predict(net, X, n) {
  const { A } = forward(net, X, n);
  return A[A.length - 1];
}

/**
 * Mean cross-entropy from the logit (stable form), the L2 term, and the gradient with respect to
 * every weight and bias. The chain rule runs from the output back, as in nnlab/mlp.py:
 *   dz_out = (p - y) / n;  dW_l = dz_l^T a_l + l2 W_l;  db_l = column sums of dz_l;
 *   da_l = dz_l W_l;  dz_{l-1} = da_l * f'(z_{l-1}).
 * Returns {data, reg, gW, gb}.
 */
export function lossAndGrads(net, X, y, n, l2) {
  const { A, Z } = forward(net, X, n);
  const L = net.W.length;
  const zo = Z[L - 1];
  let data = 0;
  for (let s = 0; s < n; s++) {
    const z = zo[s];
    data += Math.max(z, 0) - z * y[s] + Math.log1p(Math.exp(-Math.abs(z)));
  }
  data /= n;
  let sq = 0;
  for (const w of net.W) for (let k = 0; k < w.length; k++) sq += w[k] * w[k];
  const reg = 0.5 * l2 * sq;

  const gW = new Array(L), gb = new Array(L);
  let dz = new Float64Array(n);
  for (let s = 0; s < n; s++) dz[s] = (sigmoid(zo[s]) - y[s]) / n;
  for (let l = L - 1; l >= 0; l--) {
    const din = net.sizes[l], dout = net.sizes[l + 1];
    const a = A[l], w = net.W[l];
    const gw = new Float64Array(dout * din);
    const gbl = new Float64Array(dout);
    for (let j = 0; j < dout; j++) {
      let sumB = 0;
      for (let s = 0; s < n; s++) sumB += dz[s * dout + j];
      gbl[j] = sumB;
      for (let i = 0; i < din; i++) {
        let sumW = 0;
        for (let s = 0; s < n; s++) sumW += dz[s * dout + j] * a[s * din + i];
        gw[j * din + i] = sumW + l2 * w[j * din + i];
      }
    }
    gW[l] = gw;
    gb[l] = gbl;
    if (l > 0) {
      const prev = new Float64Array(n * din);
      for (let s = 0; s < n; s++) {
        for (let i = 0; i < din; i++) {
          let da = 0;
          for (let j = 0; j < dout; j++) da += dz[s * dout + j] * w[j * din + i];
          prev[s * din + i] = da * activationGrad(net.act, a[s * din + i]);
        }
      }
      dz = prev;
    }
  }
  return { data, reg, gW, gb };
}

/** Data loss (no L2 term) and accuracy at the 0.5 threshold. An empty set gives NaN for both. */
export function evaluate(net, X, y, n) {
  if (n === 0) return { loss: NaN, acc: NaN };
  const { Z } = forward(net, X, n);
  const zo = Z[Z.length - 1];
  let loss = 0, hits = 0;
  for (let s = 0; s < n; s++) {
    const z = zo[s];
    loss += Math.max(z, 0) - z * y[s] + Math.log1p(Math.exp(-Math.abs(z)));
    if ((z >= 0) === (y[s] >= 0.5)) hits++;
  }
  return { loss: loss / n, acc: hits / n };
}

/** SGD or Adam over the flat parameter arrays, with the same update rule as nnlab/mlp.py. */
export class Optimizer {
  constructor(kind, net, lr, b1 = 0.9, b2 = 0.999, eps = 1e-8) {
    this.kind = kind; this.lr = lr; this.b1 = b1; this.b2 = b2; this.eps = eps; this.t = 0;
    const zeros = (arrs) => arrs.map((a) => new Float64Array(a.length));
    this.mW = zeros(net.W); this.vW = zeros(net.W);
    this.mb = zeros(net.b); this.vb = zeros(net.b);
  }

  step(net, gW, gb) {
    this.t++;
    for (let l = 0; l < net.W.length; l++) {
      this._update(net.W[l], gW[l], this.mW[l], this.vW[l]);
      this._update(net.b[l], gb[l], this.mb[l], this.vb[l]);
    }
  }

  _update(p, g, m, v) {
    const { lr, b1, b2, eps } = this;
    if (this.kind === 'sgd') {
      for (let k = 0; k < p.length; k++) p[k] = p[k] - lr * g[k];
      return;
    }
    const c1 = 1 - Math.pow(b1, this.t), c2 = 1 - Math.pow(b2, this.t);
    for (let k = 0; k < p.length; k++) {
      m[k] = b1 * m[k] + (1 - b1) * g[k];
      v[k] = b2 * v[k] + (1 - b2) * g[k] * g[k];
      const mhat = m[k] / c1;
      const vhat = v[k] / c2;
      p[k] = p[k] - lr * mhat / (Math.sqrt(vhat) + eps);
    }
  }
}

/** Fisher-Yates over 0..n-1, walking from the end, as in nnlab/mlp.py. */
export function shuffledIndices(n, rng) {
  const idx = new Array(n);
  for (let k = 0; k < n; k++) idx[k] = k;
  for (let i = n - 1; i > 0; i--) {
    const j = Math.floor(rng.random() * (i + 1));
    const tmp = idx[i]; idx[i] = idx[j]; idx[j] = tmp;
  }
  return idx;
}

/**
 * One epoch: shuffle, then one optimiser step per mini-batch. batch <= 0 or >= n means full batch.
 * The last batch may be smaller. Returns nothing; net and opt are updated in place.
 */
export function trainEpoch(net, opt, X, y, n, rng, batch, l2) {
  if (n === 0) return;
  const F = net.sizes[0];
  const order = shuffledIndices(n, rng);
  const bs = (batch <= 0 || batch >= n) ? n : batch;
  for (let start = 0; start < n; start += bs) {
    const idx = order.slice(start, Math.min(n, start + bs));
    const m = idx.length;
    const Xb = new Float64Array(m * F), yb = new Float64Array(m);
    for (let k = 0; k < m; k++) {
      const s = idx[k];
      for (let i = 0; i < F; i++) Xb[k * F + i] = X[s * F + i];
      yb[k] = y[s];
    }
    const { gW, gb } = lossAndGrads(net, Xb, yb, m, l2);
    opt.step(net, gW, gb);
  }
}

/**
 * Activations of one hidden neuron over n samples, for the per-neuron heat map.
 * layer is the index into net.W (0 = first hidden layer); unit is the neuron in that layer.
 */
export function neuronActivations(net, X, n, layer, unit) {
  const { A } = forward(net, X, n);
  const dout = net.sizes[layer + 1];
  const out = new Float64Array(n);
  for (let s = 0; s < n; s++) out[s] = A[layer + 1][s * dout + unit];
  return out;
}
