// Regression core: least squares, ridge, gradient descent and logistic regression, ported from the
// Python package `regression`. This file has no DOM code, so the same numbers come out of Python and
// JavaScript for the same data. The ports keep the same operations in the same order: mulberry32 with
// Math.imul, the Marsaglia polar normal, the Gaussian elimination and Householder QR written by hand,
// and the gradient formulas exactly as written in the docstrings below.
//
// Conventions shared with Python:
//   * the intercept (weight 0) is never penalised: ridge uses D = diag(0, 1, ..., 1)
//   * linear objective   J(w) = (1/n) (sum (y - Xw)^2 + lam * sum_{k>=1} w_k^2)
//   * logistic objective J(w) = (1/n) (sum log-loss + lam * sum_{k>=1} w_k^2), labels in {0, 1}
//   * a degree-d polynomial uses columns x^0 .. x^d; a 2-D one uses every monomial with i + j <= d

const TWO32 = 4294967296;
const PI = Math.PI;

/** Seeded uniforms and normals. Same stream as bandits/rng.py (mulberry32, polar normal). */
export class Rng {
  constructor(seed) {
    this.state = seed >>> 0;
  }

  /** A double in [0, 1). Each call advances the state by one mixing step. */
  uniform() {
    this.state = (this.state + 0x6D2B79F5) >>> 0;
    let t = this.state;
    // The multiplies keep the low 32 bits, like the Python `& MASK`.
    t = Math.imul(t ^ (t >>> 15), 1 | t) >>> 0;
    t = (((t + (Math.imul(t ^ (t >>> 7), 61 | t) >>> 0)) >>> 0) ^ t) >>> 0;
    return ((t ^ (t >>> 14)) >>> 0) / TWO32;
  }

  /** A standard normal by the polar method: draw in the unit disc and keep one coordinate. */
  normal() {
    let u, v, s;
    for (;;) {
      u = 2 * this.uniform() - 1;
      v = 2 * this.uniform() - 1;
      s = u * u + v * v;
      if (s > 0 && s < 1) break;
    }
    return u * Math.sqrt(-2 * Math.log(s) / s);
  }
}

// ---------------------------------------------------------------- presets

const REG_FN = {
  line: (x) => 0.6 * x - 0.2,
  'noisy-sine': (x) => 0.8 * Math.sin(2 * PI * x),
  quadratic: (x) => 1.2 * x * x - 0.6,
};

/** Regression preset: n points with x uniform in [-1, 1] and y = f(x) + noise * N(0, 1). */
export function makeRegression(name, n, seed, noise) {
  const f = REG_FN[name];
  if (!f) throw new Error(`unknown regression preset: ${name}`);
  const rng = new Rng(seed);
  const x = [], y = [];
  for (let i = 0; i < n; i++) {
    const xi = -1 + 2 * rng.uniform();
    const eps = noise * rng.normal();
    x.push(xi);
    y.push(f(xi) + eps);
  }
  return { x, y };
}

/**
 * Classification preset. Returns { xy: [[x1, x2], ...], labels: [0 | 1, ...] }.
 * 'moons': two interleaved half circles, class c = i % 2, angle t uniform in [0, pi].
 * 'blobs': two Gaussian clouds centred at (-0.45, -0.35) and (0.45, 0.35), no angle draw.
 */
export function makeClassification(name, n, seed, noise) {
  // No noise given: the same per-preset defaults as regression/data.py (MOONS_NOISE 0.1, BLOBS_NOISE 0.22).
  if (noise === null || noise === undefined) noise = name === 'moons' ? 0.1 : 0.22;
  const rng = new Rng(seed);
  const xy = [], labels = [];
  for (let i = 0; i < n; i++) {
    const c = i % 2;
    let px, py;
    if (name === 'moons') {
      const t = PI * rng.uniform();
      if (c === 0) { px = Math.cos(t); py = Math.sin(t); }
      else { px = 1 - Math.cos(t); py = 0.5 - Math.sin(t); }
      px = (px - 0.5) * 0.6;
      py = (py - 0.25) * 1.0;
      px += noise * rng.normal();
      py += noise * rng.normal();
    } else if (name === 'blobs') {
      const [cx, cy] = c === 0 ? [-0.45, -0.35] : [0.45, 0.35];
      px = cx + noise * rng.normal();
      py = cy + noise * rng.normal();
    } else {
      throw new Error(`unknown classification preset: ${name}`);
    }
    xy.push([px, py]);
    labels.push(c);
  }
  return { xy, labels };
}

/** Deterministic split: point i is a test point when i % 4 == 3 (a quarter of the points). */
export function testMask(n) {
  const mask = new Array(n);
  for (let i = 0; i < n; i++) mask[i] = i % 4 === 3;
  return mask;
}

// ---------------------------------------------------------------- features

/** Rows [x^0, x^1, ..., x^d] for each scalar x. Powers by repeated multiplication. */
export function polyFeatures(x, degree) {
  return x.map((xi) => {
    const row = new Array(degree + 1);
    row[0] = 1;
    for (let k = 1; k <= degree; k++) row[k] = row[k - 1] * xi;
    return row;
  });
}

/**
 * Rows of every monomial x1^i x2^j with i + j <= degree, ordered by t = i + j from 0 to d and,
 * within each t, by i from t down to 0. Degree 1 gives [1, x1, x2].
 */
export function polyFeatures2d(xy, degree) {
  return xy.map(([x1, x2]) => {
    const row = [];
    for (let t = 0; t <= degree; t++) {
      for (let i = t; i >= 0; i--) {
        const j = t - i;
        row.push(powInt(x1, i) * powInt(x2, j));
      }
    }
    return row;
  });
}

function powInt(a, k) {
  let r = 1;
  for (let i = 0; i < k; i++) r *= a;
  return r;
}

// ---------------------------------------------------------------- losses

/** Predictions X w. */
export function predict(X, w) {
  return X.map((row) => dot(row, w));
}

function dot(a, b) {
  let s = 0;
  for (let k = 0; k < a.length; k++) s += a[k] * b[k];
  return s;
}

/** Mean squared error with no penalty. */
export function mse(X, y, w) {
  let s = 0;
  for (let i = 0; i < X.length; i++) {
    const r = y[i] - dot(X[i], w);
    s += r * r;
  }
  return s / X.length;
}

/** Linear objective J(w) = (1/n)(sum r^2 + lam * sum_{k>=1} w_k^2). */
export function linearLoss(X, y, w, ridge = 0) {
  let s = 0;
  for (let i = 0; i < X.length; i++) {
    const r = y[i] - dot(X[i], w);
    s += r * r;
  }
  for (let k = 1; k < w.length; k++) s += ridge * w[k] * w[k];
  return s / X.length;
}

/** Logistic function, computed on the side where exp cannot overflow. */
export function sigmoid(z) {
  if (z >= 0) return 1 / (1 + Math.exp(-z));
  const e = Math.exp(z);
  return e / (1 + e);
}

/** softplus(z) = log(1 + e^z) without overflow; log-loss of one point is softplus(z) - y z. */
function softplus(z) {
  return Math.max(z, 0) + Math.log1p(Math.exp(-Math.abs(z)));
}

/** Logistic objective J(w) = (1/n)(sum log-loss + lam * sum_{k>=1} w_k^2). Labels in {0, 1}. */
export function logLoss(X, labels, w, ridge = 0) {
  let s = 0;
  for (let i = 0; i < X.length; i++) {
    const z = dot(X[i], w);
    s += softplus(z) - labels[i] * z;
  }
  for (let k = 1; k < w.length; k++) s += ridge * w[k] * w[k];
  return s / X.length;
}

/** Fraction of points where the model's p >= 0.5 matches the label. */
export function accuracy(X, labels, w) {
  let hit = 0;
  for (let i = 0; i < X.length; i++) {
    const p = sigmoid(dot(X[i], w));
    if ((p >= 0.5 ? 1 : 0) === labels[i]) hit++;
  }
  return hit / X.length;
}

// ---------------------------------------------------------------- exact solvers

/**
 * Normal equations (X^T X + lam D) w = X^T y, solved by Gaussian elimination with partial pivoting
 * (the largest entry in the column is the pivot; the first one wins a tie).
 */
export function solveNormal(X, y, ridge = 0) {
  const n = X.length, p = X[0].length;
  // Build A = X^T X + lam D and b = X^T y.
  const A = [];
  const b = new Array(p).fill(0);
  for (let j = 0; j < p; j++) {
    A.push(new Array(p).fill(0));
    for (let k = 0; k < p; k++) {
      let s = 0;
      for (let i = 0; i < n; i++) s += X[i][j] * X[i][k];
      A[j][k] = s + (j === k && j > 0 ? ridge : 0);
    }
    let t = 0;
    for (let i = 0; i < n; i++) t += X[i][j] * y[i];
    b[j] = t;
  }
  // Forward elimination with partial pivoting; the augmented column is b.
  for (let c = 0; c < p; c++) {
    let piv = c;
    for (let r = c + 1; r < p; r++) if (Math.abs(A[r][c]) > Math.abs(A[piv][c])) piv = r;
    if (piv !== c) {
      [A[c], A[piv]] = [A[piv], A[c]];
      [b[c], b[piv]] = [b[piv], b[c]];
    }
    for (let r = c + 1; r < p; r++) {
      const f = A[r][c] / A[c][c];
      if (f === 0) continue;
      for (let k = c; k < p; k++) A[r][k] -= f * A[c][k];
      b[r] -= f * b[c];
    }
  }
  // Back substitution.
  const w = new Array(p).fill(0);
  for (let r = p - 1; r >= 0; r--) {
    let s = b[r];
    for (let k = r + 1; k < p; k++) s -= A[r][k] * w[k];
    w[r] = s / A[r][r];
  }
  return w;
}

/**
 * Least squares by Householder QR of the augmented matrix [X; sqrt(lam) E] against [y; 0], where E
 * holds the rows e_k for k >= 1. Ridge becomes extra rows, so QR never forms X^T X and stays accurate
 * when the polynomial columns are nearly collinear.
 */
export function solveQR(X, y, ridge = 0) {
  const n = X.length, p = X[0].length;
  const A = X.map((row) => row.slice());
  const b = y.slice();
  if (ridge > 0) {
    const s = Math.sqrt(ridge);
    for (let k = 1; k < p; k++) {
      const row = new Array(p).fill(0);
      row[k] = s;
      A.push(row);
      b.push(0);
    }
  }
  const m = A.length;
  // Rank rule of regression/core.py: a column left with a norm below max(m, p) * eps * (largest column norm) depends
  // on the columns before it. The page reads non-finite weights as singular, so that case returns NaNs.
  let largest = 0;
  for (let j = 0; j < p; j++) {
    let s2 = 0;
    for (let i = 0; i < m; i++) s2 += A[i][j] * A[i][j];
    largest = Math.max(largest, Math.sqrt(s2));
  }
  const tol = Math.max(m, p) * Number.EPSILON * largest;
  for (let k = 0; k < p; k++) {
    // Householder vector for column k below the diagonal.
    let norm2 = 0;
    for (let i = k; i < m; i++) norm2 += A[i][k] * A[i][k];
    const norm = Math.sqrt(norm2);
    if (norm <= tol) return new Array(p).fill(NaN);
    const sign = A[k][k] >= 0 ? 1 : -1;
    const alpha = -sign * norm; // the reflected column becomes alpha e_k
    const v = new Array(m).fill(0);
    for (let i = k; i < m; i++) v[i] = A[i][k];
    v[k] -= alpha;
    let vv = 0;
    for (let i = k; i < m; i++) vv += v[i] * v[i];
    if (vv === 0) continue;
    // Apply H = I - 2 v v^T / (v^T v) to the remaining columns and to b.
    for (let j = k; j < p; j++) {
      let d = 0;
      for (let i = k; i < m; i++) d += v[i] * A[i][j];
      const f = (2 * d) / vv;
      for (let i = k; i < m; i++) A[i][j] -= f * v[i];
    }
    let d = 0;
    for (let i = k; i < m; i++) d += v[i] * b[i];
    const f = (2 * d) / vv;
    for (let i = k; i < m; i++) b[i] -= f * v[i];
  }
  // R is the top p x p block of A; solve R w = (Q^T b)[:p] by back substitution.
  const w = new Array(p).fill(0);
  for (let r = p - 1; r >= 0; r--) {
    let s = b[r];
    for (let k = r + 1; k < p; k++) s -= A[r][k] * w[k];
    w[r] = s / A[r][r];
  }
  return w;
}

// ---------------------------------------------------------------- gradient descent

/**
 * Plain gradient descent on the linear objective. Gradient: (2/n)(X^T (Xw - y) + lam D w).
 * losses[t] is J at the iterate before step t, and the last entry is J at the final w.
 * Stops early (diverged = true) if J becomes non-finite or exceeds 1e12.
 */
export function gradientDescent(X, y, lr, steps, ridge = 0, w0 = null) {
  return descend(X, y, lr, steps, ridge, w0, false);
}

/**
 * Gradient descent on the logistic objective. Gradient: (1/n)(X^T (p - y) + 2 lam D w), p = sigmoid(Xw).
 * Same loss bookkeeping and divergence rule as gradientDescent.
 */
export function logisticGD(X, labels, lr, steps, ridge = 0, w0 = null) {
  return descend(X, labels, lr, steps, ridge, w0, true);
}

function descend(X, y, lr, steps, ridge, w0, logistic) {
  const n = X.length, p = X[0].length;
  const w = w0 ? w0.slice() : new Array(p).fill(0);
  const losses = [];
  let diverged = false;
  const J = (wt) => (logistic ? logLoss(X, y, wt, ridge) : linearLoss(X, y, wt, ridge));
  for (let t = 0; t < steps; t++) {
    const loss = J(w);
    losses.push(loss);
    if (!Number.isFinite(loss) || loss > 1e12) { diverged = true; break; }
    // Residual-type term: r_i = Xw - y for linear, p - y for logistic.
    const g = new Array(p).fill(0);
    for (let i = 0; i < n; i++) {
      const z = dot(X[i], w);
      const r = logistic ? sigmoid(z) - y[i] : z - y[i];
      for (let k = 0; k < p; k++) g[k] += X[i][k] * r;
    }
    // Overall factor: 2/n for the squared loss, 1/n for log-loss. The penalty inside the bracket is
    // lam * w_k for the linear objective and 2 lam * w_k for the logistic one (the derivative of w^2 is 2w).
    const scale = logistic ? 1 / n : 2 / n;
    const penalty = logistic ? 2 * ridge : ridge;
    for (let k = 0; k < p; k++) {
      const grad = scale * (g[k] + (k > 0 ? penalty * w[k] : 0));
      w[k] -= lr * grad;
    }
  }
  if (!diverged) {
    const loss = J(w);
    losses.push(loss);
    if (!Number.isFinite(loss) || loss > 1e12) diverged = true;
  }
  return { w, losses, diverged };
}
