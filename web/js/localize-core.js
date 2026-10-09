// Lost Robot core: the floor plans, the sensor and motion models, the histogram (grid) Bayes filter, and the
// particle filter with augmented-MCL injection. A line-by-line port of the Python package localize/, so the
// page and the CLI run the same algorithm. tests/test_localize_parity.py runs this file in node and compares it
// with the Python results on the same seeds.
//
// Units: a grid cell is 0.5 m. Poses are continuous: x, y in cells (x right, y down), th in radians.
// Random numbers come from mulberry32 (the same stream as localize/prng.py and bandits/rng.py), so the page, the
// CLI and the tests draw the same floors, the same odometry noise and the same beams for the same seeds.

export const CELL_M = 0.5;
export const MAX_RANGE = 8.0;     // cells (4 m): a ray that hits nothing reads this
export const RAY_STEP = 0.25;     // cells per ray-march step
export const PILLAR_PROB = 0.5;
export const OUTLIER = 0.02;
export const TAU = 2 * Math.PI;
export const SUB = 2;             // grid filter: bins per cell along x and y
export const HEADS = 16;          // grid filter: heading bins
const ROT_NOISE = [0.05, 0.15];
const FWD_NOISE = [0.03, 0.10];
const ALPHA_FAST = 0.1;
const ALPHA_SLOW = 0.01;
const KIDNAP_NATS = 2.5;
const CLUSTER_R = 1.5;
const CLUSTER_ANGLE = 0.6;
const MAX_TURN = 0.25;
const SPEED = 0.3;
const REACH = 0.35;
const FACE = 0.5;
const GH_NODES = [-Math.sqrt(3.0), 0.0, Math.sqrt(3.0)];
const GH_WEIGHTS = [1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0];
// Same order as localize/sim.py NEIGHBOURS: N, E, S, W.
const NEIGHBOURS = [[0, -1], [1, 0], [0, 1], [-1, 0]];

// The layouts are duplicated from localize/world.py; the parity test checks that they agree.
export const LAYOUTS = {
  halls: {
    title: 'TWIN HALLS', w: 24, h: 16,
    walls: [
      [0, 0, 23, 0], [0, 15, 23, 15], [0, 0, 0, 15], [23, 0, 23, 15],
      [1, 6, 2, 6], [4, 6, 8, 6], [10, 6, 14, 6], [16, 6, 19, 6], [21, 6, 22, 6],
      [1, 9, 2, 9], [4, 9, 8, 9], [10, 9, 14, 9], [16, 9, 19, 9], [21, 9, 22, 9],
      [6, 1, 6, 5], [12, 1, 12, 5], [18, 1, 18, 5],
      [6, 10, 6, 14], [12, 10, 12, 14], [18, 10, 18, 14],
    ],
    slots: [[21, 3], [6, 7], [16, 8], [15, 12]],
  },
  vault: {
    title: 'THE VAULT', w: 24, h: 16,
    walls: [
      [0, 0, 23, 0], [0, 15, 23, 15], [0, 0, 0, 15], [23, 0, 23, 15],
      [5, 1, 5, 6], [10, 4, 10, 11], [15, 1, 15, 8], [19, 6, 19, 14],
      [5, 10, 8, 10], [12, 13, 14, 13],
    ],
    slots: [[2, 3], [7, 8], [13, 2], [17, 11], [21, 4]],
  },
};

// mulberry32, bit for bit the generator of localize/prng.py. next() is one uniform; normal() is the polar method,
// which consumes uniforms in pairs exactly as the batched numpy version does.
export class Mulberry32 {
  constructor(seed) {
    this.a = seed >>> 0;
  }
  uniform() {
    this.a = (this.a + 0x6d2b79f5) >>> 0;
    let t = this.a;
    t = Math.imul(t ^ (t >>> 15), 1 | t) >>> 0;
    t = (((t + (Math.imul(t ^ (t >>> 7), 61 | t) >>> 0)) >>> 0) ^ t) >>> 0;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  }
  index(k) {
    return Math.min(k - 1, Math.floor(this.uniform() * k));
  }
  normal() {
    for (;;) {
      const u = 2 * this.uniform() - 1;
      const v = 2 * this.uniform() - 1;
      const s = u * u + v * v;
      if (s > 0 && s < 1) return u * Math.sqrt(-2 * Math.log(s) / s);
    }
  }
}

// Mask of the largest 4-connected free area. Scanned row by row, so a tie goes to the area found first.
function largestComponent(walls, w, h) {
  const seen = new Uint8Array(w * h);
  let best = [];
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      if (walls[y * w + x] || seen[y * w + x]) continue;
      const comp = [];
      const stack = [[x, y]];
      seen[y * w + x] = 1;
      while (stack.length) {
        const [cx, cy] = stack.pop();
        comp.push([cx, cy]);
        for (const [dx, dy] of [[0, -1], [1, 0], [0, 1], [-1, 0]]) {
          const nx = cx + dx, ny = cy + dy;
          if (nx >= 0 && nx < w && ny >= 0 && ny < h && !walls[ny * w + nx] && !seen[ny * w + nx]) {
            seen[ny * w + nx] = 1;
            stack.push([nx, ny]);
          }
        }
      }
      if (comp.length > best.length) best = comp;
    }
  }
  const mask = new Uint8Array(w * h);
  for (const [cx, cy] of best) mask[cy * w + cx] = 1;
  return mask;
}

// One floor. Pillars are drawn from the seed: one uniform per slot, in slot order (same as make_floor).
export function makeFloor(key, seed = 1) {
  const lay = LAYOUTS[key];
  const { w, h } = lay;
  const walls = new Uint8Array(w * h);
  for (const [x0, y0, x1, y1] of lay.walls) {
    for (let y = y0; y <= y1; y++) for (let x = x0; x <= x1; x++) walls[y * w + x] = 1;
  }
  const rng = new Mulberry32(seed);
  const pillars = [];
  for (const [x, y] of lay.slots) {
    if (rng.uniform() < PILLAR_PROB) {
      walls[y * w + x] = 1;
      pillars.push([x, y]);
    }
  }
  return { key, title: lay.title, seed, w, h, walls, pillars, region: largestComponent(walls, w, h) };
}

export function freeCells(floor) {
  const out = [];
  for (let y = 0; y < floor.h; y++) for (let x = 0; x < floor.w; x++) if (!floor.walls[y * floor.w + x]) out.push([x, y]);
  return out;
}

export function regionCells(floor) {
  const out = [];
  for (let y = 0; y < floor.h; y++) for (let x = 0; x < floor.w; x++) if (floor.region[y * floor.w + x]) out.push([x, y]);
  return out;
}

export function wrap(a) {
  return a - TAU * Math.floor((a + Math.PI) / TAU);
}

export function blocked(walls, w, h, x, y) {
  const i = Math.floor(x), j = Math.floor(y);
  if (i < 0 || i >= w || j < 0 || j >= h) return true;
  return walls[j * w + i] === 1;
}

export function noiseSd(rot, fwd, scale) {
  const sr = scale * (ROT_NOISE[0] + ROT_NOISE[1] * Math.abs(rot));
  const st = scale * (FWD_NOISE[0] + FWD_NOISE[1] * Math.abs(fwd));
  return [sr, st];
}

// One odometry step: drive along the current heading, then turn. A drive that ends in a wall is cancelled.
export function motion(floor, x, y, th, rot, fwd) {
  const nx = x + fwd * Math.cos(th);
  const ny = y + fwd * Math.sin(th);
  const bump = blocked(floor.walls, floor.w, floor.h, nx, ny);
  return [bump ? x : nx, bump ? y : ny, wrap(th + rot)];
}

// Normal upper tail helper: erfc for x >= 0 (Numerical Recipes erfcc, the same polynomial as world.erfc_pos).
export function erfcPos(x) {
  const t = 1.0 / (1.0 + 0.5 * x);
  const poly = -1.26551223 + t * (1.00002368 + t * (0.37409196 + t * (0.09678418 + t * (-0.18628806 + t * (
    0.27886807 + t * (-1.13520398 + t * (1.48851587 + t * (-0.82215223 + t * 0.17087277))))))));
  return t * Math.exp(-x * x + poly);
}

// Distance from (x, y) along angle a to the first wall, capped at rmax. The same marching loop as world.cast.
export function castOne(walls, w, h, x, y, a, rmax = MAX_RANGE, step = RAY_STEP) {
  const dx = Math.cos(a) * step;
  const dy = Math.sin(a) * step;
  let px = x, py = y;
  const n = Math.round(rmax / step);
  for (let k = 1; k <= n; k++) {
    px = px + dx;
    py = py + dy;
    const i = Math.floor(px), j = Math.floor(py);
    if (i < 0 || i >= w || j < 0 || j >= h || walls[j * w + i]) return k * step;
  }
  return rmax;
}

// Log-likelihood of a measured range z given the expected range e (world.ray_loglik).
export function rayLoglik(z, e, sigma, pdrop, rmax = MAX_RANGE) {
  const gauss = (1.0 - pdrop) * Math.exp(-0.5 * ((z - e) / sigma) ** 2) / (sigma * Math.sqrt(2.0 * Math.PI))
    + OUTLIER / rmax;
  const tail = 0.5 * erfcPos((rmax - e) / (sigma * Math.sqrt(2.0)));
  const maxp = pdrop + (1.0 - pdrop) * tail + OUTLIER / rmax;
  return Math.log(z >= rmax ? maxp : gauss);
}

// Beams from a pose: `rays` directions evenly spread from the robot's heading, read through the same marcher.
export function expectedBeams(floor, x, y, th, rays) {
  const out = new Float64Array(rays);
  for (let j = 0; j < rays; j++) out[j] = castOne(floor.walls, floor.w, floor.h, x, y, th + (TAU * j) / rays);
  return out;
}

// The sensor, on the true robot: all dropout uniforms first, then all noise normals, in beam order.
export function sense(floor, x, y, th, rays, sigma, pdrop, rng) {
  const d = expectedBeams(floor, x, y, th, rays);
  const u = new Float64Array(rays);
  for (let j = 0; j < rays; j++) u[j] = rng.uniform();
  const z = new Float64Array(rays);
  for (let j = 0; j < rays; j++) {
    const v = Math.min(MAX_RANGE, Math.max(0.0, d[j] + sigma * rng.normal()));
    z[j] = u[j] < pdrop ? MAX_RANGE : v;
  }
  return z;
}

// The hidden robot. Its motion obeys the same odometry noise as the filters assume.
export class Sim {
  constructor(floor, seed, rays = 8, sigma = 0.3, pdrop = 0.05, scale = 1.0) {
    this.floor = floor;
    this.rays = rays;
    this.sigma = sigma;
    this.pdrop = pdrop;
    this.scale = scale;
    this.rng = new Mulberry32(seed);
    this.x = 0; this.y = 0; this.th = 0;
    this.placeRandom();
    this.z = sense(floor, this.x, this.y, this.th, rays, sigma, pdrop, this.rng);
  }
  placeRandom() {
    const cells = regionCells(this.floor);
    const u0 = this.rng.uniform(), u1 = this.rng.uniform(), u2 = this.rng.uniform(), u3 = this.rng.uniform();
    const k = Math.min(cells.length - 1, Math.floor(u0 * cells.length));
    this.x = cells[k][0] + u1;
    this.y = cells[k][1] + u2;
    this.th = u3 * TAU - Math.PI;
  }
  kidnap() {
    this.placeRandom();
    this.z = sense(this.floor, this.x, this.y, this.th, this.rays, this.sigma, this.pdrop, this.rng);
    return this.z;
  }
  drive(rot, fwd) {
    const n0 = this.rng.normal(), n1 = this.rng.normal();
    const [sr, st] = noiseSd(rot, fwd, this.scale);
    [this.x, this.y, this.th] = motion(this.floor, this.x, this.y, this.th, rot + sr * n0, fwd + st * n1);
    this.z = sense(this.floor, this.x, this.y, this.th, this.rays, this.sigma, this.pdrop, this.rng);
    return this.z;
  }
}

// Shortest 4-connected path between free cells, excluding the start. Same neighbour order and tie-break as Python.
export function bfsPath(floor, start, goal) {
  if (start[0] === goal[0] && start[1] === goal[1]) return [];
  const { w, h } = floor;
  const key = (c) => c[0] + ',' + c[1];
  const prev = new Map([[key(start), null]]);
  const queue = [start];
  let qi = 0;
  while (qi < queue.length) {
    const c = queue[qi++];
    for (const [dx, dy] of NEIGHBOURS) {
      const n = [c[0] + dx, c[1] + dy];
      if (n[0] < 0 || n[0] >= w || n[1] < 0 || n[1] >= h || floor.walls[n[1] * w + n[0]] || prev.has(key(n))) continue;
      prev.set(key(n), c);
      if (n[0] === goal[0] && n[1] === goal[1]) {
        // Walk back through the parents until the one that is the start (which is excluded from the path).
        const path = [n];
        for (;;) {
          const parent = prev.get(key(path[path.length - 1]));
          if (parent === null || key(parent) === key(start)) break;
          path.push(parent);
        }
        return path.reverse();
      }
      queue.push(n);
    }
  }
  return [];
}

// The robot's controller: follow BFS routes to random destinations. Steers by the true pose; never by the filters.
export class Autopilot {
  constructor(floor, seed) {
    this.floor = floor;
    this.rng = new Mulberry32(seed);
    this.route = [];
  }
  newRoute(x, y) {
    const cells = regionCells(this.floor);
    const start = [Math.floor(x), Math.floor(y)];
    for (let t = 0; t < 8; t++) {
      const goal = cells[this.rng.index(cells.length)];
      const path = bfsPath(this.floor, start, goal);
      if (path.length) {
        this.route = path;
        return;
      }
    }
    this.route = [];
  }
  command(x, y, th) {
    if (!this.route.length) {
      this.newRoute(x, y);
      if (!this.route.length) return [0, 0];
    }
    const tx = this.route[0][0] + 0.5, ty = this.route[0][1] + 0.5;
    if (Math.hypot(tx - x, ty - y) < REACH) {
      this.route.shift();
      return [0, 0];
    }
    const err = wrap(Math.atan2(ty - y, tx - x) - th);
    const rot = Math.max(-MAX_TURN, Math.min(MAX_TURN, err));
    const fwd = Math.abs(err) < FACE ? SPEED : 0;
    return [rot, fwd];
  }
}

// Groups of 1-cell bins that each hold at least `threshold` of the weight (4-connected). Keys are "bx,by".
export function countModes(hist, threshold = 0.05) {
  const live = new Set();
  for (const [k, v] of hist) if (v >= threshold) live.add(k);
  const seen = new Set();
  let groups = 0;
  for (const c of live) {
    if (seen.has(c)) continue;
    groups++;
    const stack = [c];
    seen.add(c);
    while (stack.length) {
      const [cx, cy] = stack.pop().split(',').map(Number);
      for (const [dx, dy] of NEIGHBOURS) {
        const n = (cx + dx) + ',' + (cy + dy);
        if (live.has(n) && !seen.has(n)) {
          seen.add(n);
          stack.push(n);
        }
      }
    }
  }
  return groups;
}

// The histogram filter. belief[(j * w2 + i) * HEADS + k] is the probability of position bin (i, j), heading bin k.
export class GridFilter {
  constructor(floor, rays, sigma, pdrop, scale = 1.0) {
    this.floor = floor;
    this.rays = rays;
    this.sigma = sigma;
    this.pdrop = pdrop;
    this.scale = scale;
    this.w2 = floor.w * SUB;
    this.h2 = floor.h * SUB;
    this.dh = TAU / HEADS;
    const s = this.h2 * this.w2 * HEADS;
    this.size = s;
    this.x = new Float64Array(s);
    this.y = new Float64Array(s);
    this.th = new Float64Array(s);
    this.cosTh = new Float64Array(s);
    this.sinTh = new Float64Array(s);
    this.belief = new Float64Array(s);
    let total = 0;
    for (let j = 0; j < this.h2; j++) {
      for (let i = 0; i < this.w2; i++) {
        const free = !floor.walls[Math.floor(j / SUB) * floor.w + Math.floor(i / SUB)];
        for (let k = 0; k < HEADS; k++) {
          const idx = (j * this.w2 + i) * HEADS + k;
          this.x[idx] = (i + 0.5) / SUB;
          this.y[idx] = (j + 0.5) / SUB;
          this.th[idx] = k * this.dh;
          this.cosTh[idx] = Math.cos(this.th[idx]);
          this.sinTh[idx] = Math.sin(this.th[idx]);
          this.belief[idx] = free ? 1 : 0;
          total += this.belief[idx];
        }
      }
    }
    for (let idx = 0; idx < s; idx++) this.belief[idx] /= total;
    // Expected beams from every bin centre, computed once per map (as in grid.py).
    this.expected = new Float64Array(s * rays);
    for (let idx = 0; idx < s; idx++) {
      for (let j = 0; j < rays; j++) {
        this.expected[idx * rays + j] = castOne(floor.walls, floor.w, floor.h, this.x[idx], this.y[idx],
          this.th[idx] + (TAU * j) / rays);
      }
    }
    this.steps = 0;
  }
  predict(rot, fwd) {
    const [sr, st] = noiseSd(rot, fwd, this.scale);
    const s = this.size;
    const w = this.floor.w, h = this.floor.h;
    const dest = new Int32Array(9 * s);
    const weight = new Float64Array(9);
    let q = 0;
    for (let ia = 0; ia < 3; ia++) {
      for (let ib = 0; ib < 3; ib++) {
        // Quadrature point (ia, ib): turn noise node ia, drive noise node ib; weights multiply.
        weight[q] = GH_WEIGHTS[ia] * GH_WEIGHTS[ib];
        const r = rot + GH_NODES[ia] * sr;
        const f = fwd + GH_NODES[ib] * st;
        for (let idx = 0; idx < s; idx++) {
          const nx = this.x[idx] + f * this.cosTh[idx];
          const ny = this.y[idx] + f * this.sinTh[idx];
          const nth = wrap(this.th[idx] + r);
          const k2 = ((Math.floor(nth / this.dh + 0.5) % HEADS) + HEADS) % HEADS;
          if (blocked(this.floor.walls, w, h, nx, ny)) {
            // A bump keeps the position but still turns, as world.motion does: this bin's position at the new heading.
            dest[q * s + idx] = idx - (idx % HEADS) + k2;
          } else {
            const i2 = Math.min(this.w2 - 1, Math.max(0, Math.floor(nx * SUB)));
            const j2 = Math.min(this.h2 - 1, Math.max(0, Math.floor(ny * SUB)));
            dest[q * s + idx] = (j2 * this.w2 + i2) * HEADS + k2;
          }
        }
        q++;
      }
    }
    const out = new Float64Array(s);
    for (let qq = 0; qq < 9; qq++) {
      for (let idx = 0; idx < s; idx++) out[dest[qq * s + idx]] += weight[qq] * this.belief[idx];
    }
    this.belief = out;
    this.steps++;
  }
  update(z) {
    const s = this.size, rays = this.rays;
    const logl = new Float64Array(s);
    let top = -Infinity;
    for (let idx = 0; idx < s; idx++) {
      if (this.belief[idx] <= 0) {
        logl[idx] = -Infinity;
        continue;
      }
      let acc = 0;
      for (let j = 0; j < rays; j++) acc += rayLoglik(z[j], this.expected[idx * rays + j], this.sigma, this.pdrop);
      logl[idx] = acc;
      if (acc > top) top = acc;
    }
    let total = 0;
    for (let idx = 0; idx < s; idx++) {
      this.belief[idx] = this.belief[idx] > 0 ? this.belief[idx] * Math.exp(logl[idx] - top) : 0;
      total += this.belief[idx];
    }
    for (let idx = 0; idx < s; idx++) this.belief[idx] /= total;
  }
  estimate() {
    let best = 0;
    for (let idx = 1; idx < this.size; idx++) if (this.belief[idx] > this.belief[best]) best = idx;
    return { x: this.x[best], y: this.y[best], th: this.th[best], mass: this.belief[best] };
  }
  // Belief summed over heading: a Float64Array of h2 * w2, the heat map the page draws.
  positionMap() {
    const out = new Float64Array(this.h2 * this.w2);
    for (let idx = 0; idx < this.size; idx++) out[Math.floor(idx / HEADS)] += this.belief[idx];
    return out;
  }
}

// Python's round(): halves go to the even neighbour. Matters only for the injection count.
function roundHalfEven(v) {
  const f = Math.floor(v);
  const d = v - f;
  if (d > 0.5) return f + 1;
  if (d < 0.5) return f;
  return f % 2 === 0 ? f : f + 1;
}

// Monte Carlo localisation with systematic resampling and augmented-MCL injection (particles.py).
export class ParticleFilter {
  constructor(floor, n, rays, sigma, pdrop, scale, seed, augmented = true) {
    this.floor = floor;
    this.n = n;
    this.rays = rays;
    this.sigma = sigma;
    this.pdrop = pdrop;
    this.scale = scale;
    this.augmented = augmented;
    this.rng = new Mulberry32(seed);
    const cells = freeCells(floor);
    this.cells = cells;
    this.x = new Float64Array(n);
    this.y = new Float64Array(n);
    this.th = new Float64Array(n);
    this.w = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      const u0 = this.rng.uniform(), u1 = this.rng.uniform(), u2 = this.rng.uniform(), u3 = this.rng.uniform();
      const k = Math.min(cells.length - 1, Math.floor(u0 * cells.length));
      this.x[i] = cells[k][0] + u1;
      this.y[i] = cells[k][1] + u2;
      this.th[i] = u3 * TAU - Math.PI;
      this.w[i] = 1.0 / n;
    }
    this.ess = n;
    this.pInject = 0;
    this.injected = 0;
    this.replaced = new Int32Array(0);
    this.resampled = false;
    this.ls = null;
    this.lf = null;
    this.steps = 0;
  }
  // k indices chosen uniformly at random without replacement: a partial Fisher-Yates shuffle over the stream.
  chooseReplaced(k) {
    const n = this.n;
    const perm = Array.from({ length: n }, (_, i) => i);
    for (let i = 0; i < k; i++) {
      const j = i + Math.min(n - 1 - i, Math.floor(this.rng.uniform() * (n - i)));
      const t = perm[i]; perm[i] = perm[j]; perm[j] = t;
    }
    return perm.slice(0, k);
  }
  sampleCells(m, idx) {
    for (let k = 0; k < m; k++) {
      const u0 = this.rng.uniform(), u1 = this.rng.uniform(), u2 = this.rng.uniform(), u3 = this.rng.uniform();
      const kk = Math.min(this.cells.length - 1, Math.floor(u0 * this.cells.length));
      const i = idx[k];
      this.x[i] = this.cells[kk][0] + u1;
      this.y[i] = this.cells[kk][1] + u2;
      this.th[i] = u3 * TAU - Math.PI;
    }
  }
  step(rot, fwd, z) {
    const n = this.n, rays = this.rays, floor = this.floor;
    // 1. Motion: the turn noise then the drive noise, for each particle in turn.
    const [sr, st] = noiseSd(rot, fwd, this.scale);
    for (let i = 0; i < n; i++) {
      const nr = this.rng.normal(), nf = this.rng.normal();
      const [x, y, th] = motion(floor, this.x[i], this.y[i], this.th[i], rot + sr * nr, fwd + st * nf);
      this.x[i] = x; this.y[i] = y; this.th[i] = th;
    }
    // 2. Sensing: the log-likelihood of the beams under each particle's ray cast.
    const lik = new Float64Array(n);
    const logl = new Float64Array(n);
    let top = -Infinity;
    for (let i = 0; i < n; i++) {
      let acc = 0;
      for (let j = 0; j < rays; j++) {
        const e = castOne(floor.walls, floor.w, floor.h, this.x[i], this.y[i], this.th[i] + (TAU * j) / rays);
        acc += rayLoglik(z[j], e, this.sigma, this.pdrop);
      }
      logl[i] = acc;
      if (acc > top) top = acc;
    }
    let sumLik = 0;
    for (let i = 0; i < n; i++) {
      lik[i] = Math.exp(logl[i] - top);
      sumLik += lik[i];
    }
    const lavg = top + Math.log(sumLik / n);
    if (this.ls === null) {
      this.ls = lavg;
      this.lf = lavg;
    } else {
      this.ls += ALPHA_SLOW * (lavg - this.ls);
      this.lf += ALPHA_FAST * (lavg - this.lf);
    }
    const drop = this.lf - this.ls;
    let p = this.augmented && drop < -KIDNAP_NATS ? 1.0 - Math.exp(drop) : 0.0;
    p = Math.min(1.0, Math.max(0.0, p));
    this.pInject = p;
    let total = 0;
    for (let i = 0; i < n; i++) {
      this.w[i] = this.w[i] * lik[i];
      total += this.w[i];
    }
    for (let i = 0; i < n; i++) this.w[i] /= total;
    let sq = 0;
    for (let i = 0; i < n; i++) sq += this.w[i] * this.w[i];
    this.ess = 1.0 / sq;
    this.resampled = this.ess < n / 2.0;
    if (this.resampled) this.systematicResample();
    // 4. Recovery: a random subset of the particles is replaced by uniform draws (chooseReplaced, as particles.py).
    this.injected = roundHalfEven(p * n);
    if (this.injected > 0) {
      const idx = this.chooseReplaced(this.injected);
      this.replaced = idx;
      this.sampleCells(this.injected, idx);
      for (const i of idx) this.w[i] = 1.0 / n;
      let t2 = 0;
      for (let i = 0; i < n; i++) t2 += this.w[i];
      for (let i = 0; i < n; i++) this.w[i] /= t2;
    }
    this.steps++;
  }
  systematicResample() {
    const n = this.n;
    const u0 = this.rng.uniform() / n;
    const cum = new Float64Array(n);
    let acc = 0;
    for (let i = 0; i < n; i++) {
      acc += this.w[i];
      cum[i] = acc;
    }
    cum[n - 1] = 1.0;
    const idx = new Int32Array(n);
    let j = 0;
    for (let i = 0; i < n; i++) {
      const pointer = u0 + i / n;
      while (j < n - 1 && cum[j] < pointer) j++;
      idx[i] = j;
    }
    const x = new Float64Array(n), y = new Float64Array(n), th = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      x[i] = this.x[idx[i]];
      y[i] = this.y[idx[i]];
      th[i] = this.th[idx[i]];
    }
    this.x = x; this.y = y; this.th = th;
    this.w.fill(1.0 / n);
  }
  estimate() {
    const n = this.n;
    let best = 0;
    for (let i = 1; i < n; i++) if (this.w[i] > this.w[best]) best = i;
    let mass = 0, sx = 0, sy = 0, sc = 0, ss = 0;
    const inCluster = new Uint8Array(n);
    for (let i = 0; i < n; i++) {
      const dist = Math.hypot(this.x[i] - this.x[best], this.y[i] - this.y[best]);
      const dang = Math.abs(wrap(this.th[i] - this.th[best]));
      if (dist < CLUSTER_R && dang < CLUSTER_ANGLE) {
        inCluster[i] = 1;
        mass += this.w[i];
      }
    }
    for (let i = 0; i < n; i++) {
      if (!inCluster[i]) continue;
      const wi = this.w[i] / mass;
      sx += wi * this.x[i];
      sy += wi * this.y[i];
      sc += wi * Math.cos(this.th[i]);
      ss += wi * Math.sin(this.th[i]);
    }
    let cxx = 0, cxy = 0, cyy = 0;
    for (let i = 0; i < n; i++) {
      if (!inCluster[i]) continue;
      const wi = this.w[i] / mass;
      const dx = this.x[i] - sx, dy = this.y[i] - sy;
      cxx += wi * dx * dx;
      cxy += wi * dx * dy;
      cyy += wi * dy * dy;
    }
    return { x: sx, y: sy, th: Math.atan2(ss, sc), mass, cov: [[cxx, cxy], [cxy, cyy]], heaviest: best };
  }
  // Weight per 1-cell bin, keyed "bx,by" (the mode count and the density drawing use it).
  positionHistogram() {
    const out = new Map();
    for (let i = 0; i < this.n; i++) {
      const k = Math.floor(this.x[i]) + ',' + Math.floor(this.y[i]);
      out.set(k, (out.get(k) || 0) + this.w[i]);
    }
    return out;
  }
}
