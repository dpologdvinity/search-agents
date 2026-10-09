// Optimizer race: the pure maths behind the lab. No DOM here, so node can load it for the parity test.
//
// This is a port of optlab/ (the Python reference). Every formula is written with the same operations in the
// same order, the random stream is mulberry32 with the same integer steps, and normal noise is the sum of 12
// uniforms minus 6, so a race run in node reproduces the Python trajectories to rounding error.

export const NAMES = ['sgd', 'momentum', 'nesterov', 'rmsprop', 'adam', 'adagrad'];
export const LABELS = {
  sgd: 'SGD', momentum: 'Momentum', nesterov: 'Nesterov', rmsprop: 'RMSProp', adam: 'Adam', adagrad: 'AdaGrad',
};
export const GTOL = 1e-3; // gradient norm below which a track has reached a minimum
export const BLOW = 1e6; // a position this large counts as divergence
export const DEFAULT_LR = {
  sgd: 0.05, momentum: 0.02, nesterov: 0.02, rmsprop: 0.05, adam: 0.1, adagrad: 0.5,
};
export const DEFAULT_HYPER = { momentum: 0.9, beta1: 0.9, beta2: 0.999, eps: 1e-8, rho: 0.9 };

const TAU = 2 * Math.PI;
const HESSIAN_STEP = 1e-5;
const M32 = 0xffffffff;

// ---------- random numbers ----------

/** mulberry32 with the same integer steps as optlab/rng.py. */
export class Rng {
  constructor(seed = 0) {
    this.a = seed >>> 0;
  }

  uniform() {
    this.a = (this.a + 0x6d2b79f5) | 0;
    let t = Math.imul(this.a ^ (this.a >>> 15), 1 | this.a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  }

  /** Sum of 12 uniforms minus 6: mean 0, variance 1, bounded in [-6, 6]. */
  normal() {
    let s = 0;
    for (let i = 0; i < 12; i++) s += this.uniform();
    return s - 6;
  }
}

// ---------- surfaces ----------

const bowlF = (x, y) => x * x + y * y;
const bowlG = (x, y) => [2 * x, 2 * y];
const ravineF = (x, y) => 0.5 * (x * x + 25 * y * y);
const ravineG = (x, y) => [x, 25 * y];
const saddleF = (x, y) => x * x - y * y + 0.25 * (y * y * y * y);
const saddleG = (x, y) => [2 * x, -2 * y + y * y * y];
const rosenF = (x, y) => (1 - x) * (1 - x) + 100 * ((y - x * x) * (y - x * x));
const rosenG = (x, y) => {
  const r = y - x * x;
  return [-2 * (1 - x) - 400 * x * r, 200 * r];
};
const himmelF = (x, y) => {
  const a = x * x + y - 11;
  const b = x + y * y - 7;
  return a * a + b * b;
};
const himmelG = (x, y) => {
  const a = x * x + y - 11;
  const b = x + y * y - 7;
  return [4 * x * a + 2 * b, 2 * a + 4 * y * b];
};
const rastriginF = (x, y) => 20 + (x * x - 10 * Math.cos(TAU * x)) + (y * y - 10 * Math.cos(TAU * y));
const rastriginG = (x, y) => [2 * x + 10 * TAU * Math.sin(TAU * x), 2 * y + 10 * TAU * Math.sin(TAU * y)];

const HIMMEL_MINIMA = [[3, 2], [-2.805118, 3.131312], [-3.779310, -3.283186], [3.584428, -1.848126]];

function surface(name, label, f, grad, domain, start, fmin, minima, note) {
  return { name, label, f, grad, domain, start, fmin, minima, note };
}

export const SURFACES = {
  bowl: surface('bowl', 'Bowl', bowlF, bowlG, [-3, 3, -3, 3], [2.4, 1.6], 0, [[0, 0]],
    'x^2 + y^2. Well conditioned: every direction has the same curvature.'),
  ravine: surface('ravine', 'Ravine', ravineF, ravineG, [-4, 4, -4, 4], [3.2, 0.9], 0, [[0, 0]],
    '0.5 (x^2 + 25 y^2). Curvature 25 across the ravine, 1 along it: ill conditioned.'),
  saddle: surface('saddle', 'Saddle', saddleF, saddleG, [-2.5, 2.5, -2.5, 2.5], [1.8, 0.6], -1,
    [[0, Math.sqrt(2)], [0, -Math.sqrt(2)]],
    'x^2 - y^2 + y^4/4. The origin is a saddle; the two minima sit at y = +-sqrt(2).'),
  rosenbrock: surface('rosenbrock', 'Rosenbrock', rosenF, rosenG, [-2, 2, -1, 3], [-1.5, 2], 0, [[1, 1]],
    '(1 - x)^2 + 100 (y - x^2)^2. A curved banana valley to one minimum.'),
  himmelblau: surface('himmelblau', 'Himmelblau', himmelF, himmelG, [-5, 5, -5, 5], [-4, 4.5], 0, HIMMEL_MINIMA,
    '(x^2 + y - 11)^2 + (x + y^2 - 7)^2. Four global minima, all at zero loss.'),
  rastrigin: surface('rastrigin', 'Rastrigin', rastriginF, rastriginG, [-5.12, 5.12, -5.12, 5.12], [4.1, 2.9], 0,
    [[0, 0]], '20 + sum (x_i^2 - 10 cos 2 pi x_i). A grid of local minima around one global one.'),
};

export const CUSTOM_DEFAULT_BUMPS = [[0.6, -0.4, 3.0, 0.7], [-1.2, 1.1, -2.5, 0.9]];
export const CUSTOM_BASE = 0.1;
export const CUSTOM_DOMAIN = [-3, 3, -3, 3];
export const CUSTOM_START = [-2.2, -1.6];
export const MAX_BUMPS = 12;

/** Paint-your-own landscape: CUSTOM_BASE (x^2 + y^2) plus Gaussian bumps [cx, cy, amp, sigma]. */
export function customSurface(bumps = CUSTOM_DEFAULT_BUMPS) {
  const bs = bumps.map(([cx, cy, a, s]) => [Number(cx), Number(cy), Number(a), Number(s)]);
  if (bs.length > MAX_BUMPS) throw new Error(`at most ${MAX_BUMPS} bumps`);
  for (const b of bs) if (!(b[3] > 0)) throw new Error('bump sigma must be positive');
  const f = (x, y) => {
    let v = CUSTOM_BASE * (x * x + y * y);
    for (const [cx, cy, a, s] of bs) {
      const dx = x - cx;
      const dy = y - cy;
      v += a * Math.exp(-(dx * dx + dy * dy) / (2 * s * s));
    }
    return v;
  };
  const grad = (x, y) => {
    let gx = 2 * CUSTOM_BASE * x;
    let gy = 2 * CUSTOM_BASE * y;
    for (const [cx, cy, a, s] of bs) {
      const dx = x - cx;
      const dy = y - cy;
      const e = a * Math.exp(-(dx * dx + dy * dy) / (2 * s * s)) / (s * s);
      gx -= e * dx;
      gy -= e * dy;
    }
    return [gx, gy];
  };
  return surface('custom', 'Paint your own', f, grad, CUSTOM_DOMAIN, CUSTOM_START, null, [],
    'A bowl with your own hills (+) and valleys (-).');
}

export function getSurface(name) {
  if (name === 'custom') return customSurface();
  const s = SURFACES[name];
  if (!s) throw new Error(`unknown surface ${name}`);
  return s;
}

/** Symmetric Hessian [hxx, hxy, hyy] from central differences of the gradient (as optlab.surfaces.hessian). */
export function hessian(grad, x, y, h = HESSIAN_STEP) {
  const gxp = grad(x + h, y);
  const gxm = grad(x - h, y);
  const gyp = grad(x, y + h);
  const gym = grad(x, y - h);
  const hxx = (gxp[0] - gxm[0]) / (2 * h);
  const hyx = (gxp[1] - gxm[1]) / (2 * h);
  const hxy = (gyp[0] - gym[0]) / (2 * h);
  const hyy = (gyp[1] - gym[1]) / (2 * h);
  return [hxx, 0.5 * (hxy + hyx), hyy];
}

/** Gradient norm below gtol and a positive definite Hessian (a minimum, not a saddle). */
export function isMinimum(s, x, y, gtol) {
  const [gx, gy] = s.grad(x, y);
  if (!(Math.sqrt(gx * gx + gy * gy) < gtol)) return false;
  const [hxx, hxy, hyy] = hessian(s.grad, x, y);
  return hxx * hyy - hxy * hxy > 0 && hxx + hyy > 0;
}

// ---------- optimizers ----------

/** Each optimizer: query(x) says where to take the gradient, update(x, g) gives the next position. */
export class SGD {
  constructor(lr) { this.lr = Number(lr); }
  query(x) { return [x[0], x[1]]; }
  update(x, g) { return [x[0] - this.lr * g[0], x[1] - this.lr * g[1]]; }
}

export class Momentum {
  constructor(lr, hyper) { this.lr = Number(lr); this.h = hyper; this.v = [0, 0]; }
  query(x) { return [x[0], x[1]]; }
  update(x, g) {
    const b = this.h.momentum;
    const out = [0, 0];
    for (let i = 0; i < 2; i++) {
      this.v[i] = b * this.v[i] - this.lr * g[i];
      out[i] = x[i] + this.v[i];
    }
    return out;
  }
}

export class Nesterov extends Momentum {
  query(x) {
    const b = this.h.momentum;
    return [x[0] + b * this.v[0], x[1] + b * this.v[1]];
  }
}

export class RMSProp {
  constructor(lr, hyper) { this.lr = Number(lr); this.h = hyper; this.s = [0, 0]; }
  query(x) { return [x[0], x[1]]; }
  update(x, g) {
    const { rho, eps } = this.h;
    const out = [0, 0];
    for (let i = 0; i < 2; i++) {
      this.s[i] = rho * this.s[i] + (1 - rho) * (g[i] * g[i]);
      out[i] = x[i] - this.lr * g[i] / (Math.sqrt(this.s[i]) + eps);
    }
    return out;
  }
}

export class AdaGrad {
  constructor(lr, hyper) { this.lr = Number(lr); this.h = hyper; this.G = [0, 0]; }
  query(x) { return [x[0], x[1]]; }
  update(x, g) {
    const eps = this.h.eps;
    const out = [0, 0];
    for (let i = 0; i < 2; i++) {
      this.G[i] = this.G[i] + g[i] * g[i];
      out[i] = x[i] - this.lr * g[i] / (Math.sqrt(this.G[i]) + eps);
    }
    return out;
  }
}

export class Adam {
  constructor(lr, hyper) {
    this.lr = Number(lr);
    this.h = hyper;
    this.m = [0, 0];
    this.v = [0, 0];
    this.b1t = 1;
    this.b2t = 1;
  }
  query(x) { return [x[0], x[1]]; }
  update(x, g) {
    const { beta1: b1, beta2: b2, eps } = this.h;
    this.b1t *= b1;
    this.b2t *= b2;
    const out = [0, 0];
    for (let i = 0; i < 2; i++) {
      this.m[i] = b1 * this.m[i] + (1 - b1) * g[i];
      this.v[i] = b2 * this.v[i] + (1 - b2) * (g[i] * g[i]);
      const mh = this.m[i] / (1 - this.b1t);
      const vh = this.v[i] / (1 - this.b2t);
      out[i] = x[i] - this.lr * mh / (Math.sqrt(vh) + eps);
    }
    return out;
  }
}

const CLASSES = { sgd: SGD, momentum: Momentum, nesterov: Nesterov, rmsprop: RMSProp, adam: Adam, adagrad: AdaGrad };

export function makeOptimizer(name, lr, hyper = DEFAULT_HYPER) {
  const C = CLASSES[name];
  if (!C) throw new Error(`unknown optimizer ${name}`);
  return new C(lr, hyper);
}

// ---------- the race ----------

function inRange(x, y, blow) {
  return Number.isFinite(x) && Number.isFinite(y) && Math.abs(x) <= blow && Math.abs(y) <= blow;
}

/**
 * Several optimizers on one surface, advanced together by step(). Mirrors optlab/race.py: all tracks share each
 * step's noise draw, a track stops at its first non-finite or out-of-range position (diverged), and reaching a
 * minimum is recorded once, at the first step where isMinimum holds.
 */
export class Race {
  constructor({ surface: s, start, names = NAMES, lrs = null, hyper = DEFAULT_HYPER, noise = 0, seed = 0,
    gtol = GTOL, blow = BLOW }) {
    const lr = { ...DEFAULT_LR, ...(lrs || {}) };
    this.surface = s;
    this.hyper = hyper;
    this.noise = Number(noise);
    this.gtol = gtol;
    this.blow = blow;
    this.rng = new Rng(seed);
    this.t = 0;
    const x0 = Number(start[0]);
    const y0 = Number(start[1]);
    this.tracks = names.map((name) => {
      const l = Number(lr[name]);
      const tr = {
        name, lr: l, opt: makeOptimizer(name, l, hyper), x: x0, y: y0,
        traj: [[x0, y0]], losses: [s.f(x0, y0)], stepsToTol: null, divergedAt: null, divergedBy: null,
      };
      if (isMinimum(s, x0, y0, gtol)) tr.stepsToTol = 0;
      return tr;
    });
  }

  step() {
    this.t += 1;
    const nx = this.rng.normal();
    const ny = this.rng.normal();
    const s = this.surface;
    for (const tr of this.tracks) {
      if (tr.divergedAt !== null) continue;
      const x = [tr.x, tr.y];
      const q = tr.opt.query(x);
      const [gx, gy] = s.grad(q[0], q[1]);
      const g = [gx + this.noise * nx, gy + this.noise * ny];
      const nxt = tr.opt.update(x, g);
      if (!inRange(nxt[0], nxt[1], this.blow)) {
        // A finite position past blow is a blow-up; a NaN or infinite position is a NaN.
        tr.divergedAt = this.t;
        tr.divergedBy = Number.isFinite(nxt[0]) && Number.isFinite(nxt[1]) ? 'blowup' : 'nan';
        continue;
      }
      const loss = s.f(nxt[0], nxt[1]);
      if (!Number.isFinite(loss)) {
        tr.divergedAt = this.t;
        tr.divergedBy = 'nan';
        continue;
      }
      tr.x = nxt[0];
      tr.y = nxt[1];
      tr.traj.push([tr.x, tr.y]);
      tr.losses.push(loss);
      if (tr.stepsToTol === null && isMinimum(s, tr.x, tr.y, this.gtol)) tr.stepsToTol = this.t;
    }
  }

  run(steps) {
    for (let i = 0; i < steps; i++) this.step();
    return this;
  }
}
