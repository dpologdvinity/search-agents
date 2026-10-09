// EVOLVING WALKERS: pure physics and genetic algorithm.
//
// No DOM and no imports, so the same file runs on the page, in the Web Worker, and under node for the parity
// test. The Python port (walkers/) follows this file operation for operation: same order of floating-point
// operations, same order of random draws. Change one, change the other, and run tests/test_walkers.py.
//
// Units: metres, seconds, kilograms (every node has mass 1).

// ---------------------------------------------------------------- constants
export const DT = 1 / 240; // fixed physics step, seconds
export const NODE_MIN = 2;
export const NODE_MAX = 12;
export const SPRING_MAX = 40; // soft cap: mutation and crossover respect it, repair does not truncate
export const REST_MIN = 0.08;
export const REST_MAX = 1.6;
export const K_MIN = 20;
export const K_MAX = 600;
export const C_MIN = 0.2;
export const C_MAX = 10;
export const AMP_MAX = 0.4;
export const FREQ_MIN = 0.4;
export const FREQ_MAX = 4.0;
export const POS_MAX = 1.5; // genome node coordinates (metres, relative to the body centre)
export const VMAX = 30; // velocity clamp, m/s: an explosion guard, counted but never fatal by itself
export const AIR_DRAG = 0.1; // 1/s, linear drag on every node
export const REST_SPEED = 0.5; // m/s: impacts slower than this are inelastic (no bounce jitter)
export const STATIC_RATIO = 1.2; // static friction coefficient = friction * STATIC_RATIO
export const FALL_Y = -1.5; // the body's centre of mass below this has fallen into a gap: scoring stops
export const SPIN_FREE = 1.0; // radians of accumulated spine rotation that go unpunished
export const SPIN_PENALTY = 1.0; // metres per radian beyond SPIN_FREE (tumbling must cost more than it earns)
export const SPEED_BONUS = 0.5; // seconds: fitness gains this times the mean forward speed of the second half
export const SPAWN_X = 2.0;
export const SPAWN_LIFT = 0.45; // lowest node starts this high above the floor
export const HOLE = -1000; // ground height where a gap has no floor
export const DEFAULT_WORLD = { terrain: 'flat', gravity: 9.8, friction: 0.8, restitution: 0.2, duration: 10 };
export const TERRAINS = ['flat', 'hills', 'steps', 'gaps'];
export const GA_DEFAULTS = {
  popSize: 80,
  mutationRate: 0.5,
  crossoverRate: 0.6,
  eliteMin: 5, // species of at least this size keep their best unchanged
  tournament: 3,
  survivalFrac: 0.5,
  speciesThreshold: 1.2,
};

const PI = 3.141592653589793;
export const TWO_PI = 6.283185307179586;
const HALF_PI = 1.5707963267948966;

// ---------------------------------------------------------------- numbers
// Sine from basic arithmetic only. V8's Math.sin and NumPy's sin can differ in the last bit, which chaotic
// bodies amplify, so the actuators use this polynomial and the two languages agree bit for bit.
export function wsin(x) {
  let r = x - TWO_PI * Math.floor((x + PI) / TWO_PI);
  if (r > HALF_PI) r = PI - r;
  else if (r < -HALF_PI) r = -PI - r;
  const r2 = r * r;
  return r * (1 - (r2 / 6) * (1 - (r2 / 20) * (1 - (r2 / 42) * (1 - (r2 / 72) * (1 - (r2 / 110) * (1 - (r2 / 156) * (1 - r2 / 210)))))));
}

export function wcos(x) {
  return wsin(x + HALF_PI);
}

// Round to the 1e-4 grid. Genes live on this grid, so the text encoding is exact.
export function q4(v) {
  return Math.floor(v * 10000 + 0.5) / 10000;
}

export function clamp(v, lo, hi) {
  return v < lo ? lo : v > hi ? hi : v;
}

// mulberry32, the same generator in Python. Returns an object with next() in [0, 1).
export function mulberry32(seed) {
  let a = seed >>> 0;
  const next = () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
  return {
    next,
    int: (n) => Math.floor(next() * n), // 0..n-1
    range: (lo, hi) => lo + (hi - lo) * next(),
  };
}

// ---------------------------------------------------------------- terrain
// groundAt writes the ground height and slope under x into G (no allocation in the hot loop).
// hills: a ramp-in from x=3 to x=5, then two sine waves. steps: 0.15 m risers every 1.2 m from x=3 (8 max).
// gaps: flat floor with 0.9 m holes every 3.2 m from x=4.
const G = { h: 0, s: 0 };
export function groundAt(terrain, x) {
  if (terrain === 'hills') {
    let ramp = (x - 3) / 2;
    ramp = ramp < 0 ? 0 : ramp > 1 ? 1 : ramp;
    const dramp = x > 3 && x < 5 ? 0.5 : 0;
    const H = 0.35 * wsin(0.7 * x) + 0.12 * wsin(1.9 * x + 1);
    const dH = 0.35 * 0.7 * wcos(0.7 * x) + 0.12 * 1.9 * wcos(1.9 * x + 1);
    G.h = ramp * H;
    G.s = ramp * dH + dramp * H;
  } else if (terrain === 'steps') {
    const k = x > 3 ? Math.min(8, Math.floor((x - 3) / 1.2)) : 0;
    G.h = 0.15 * k;
    G.s = 0;
  } else if (terrain === 'gaps') {
    G.s = 0;
    if (x < 4) G.h = 0;
    else {
      const u = x - 4;
      const r = u - 3.2 * Math.floor(u / 3.2);
      G.h = r < 0.9 ? HOLE : 0;
    }
  } else {
    G.h = 0;
    G.s = 0;
  }
  return G;
}

// ---------------------------------------------------------------- genome
// A genome is plain data, JSON-compatible:
//   nodes:   [{x, y}]                               rest positions (metres, body frame)
//   springs: [{a, b, rest, k, c, muscle, amp, freq, phase}]   a < b (node indices)
// rest: rest length (m). k: stiffness (N/m). c: damping (N s/m). muscle: 0 or 1 (flag).
// A muscle's rest length oscillates: rest * (1 + amp * sin(2 pi freq t + phase)).
export function cloneGenome(g) {
  return {
    nodes: g.nodes.map((n) => ({ x: n.x, y: n.y })),
    springs: g.springs.map((s) => ({
      a: s.a, b: s.b, rest: s.rest, k: s.k, c: s.c, muscle: s.muscle ? 1 : 0,
      amp: s.amp, freq: s.freq, phase: s.phase,
    })),
  };
}

function makeSpring(a, b, rest, muscle, rng) {
  // Draw order is fixed (k, c, amp, freq, phase) so the Python port consumes the same random numbers.
  return {
    a: Math.min(a, b),
    b: Math.max(a, b),
    rest: q4(clamp(rest, REST_MIN, REST_MAX)),
    k: q4(rng.range(60, 250)),
    c: q4(rng.range(0.5, 6)),
    muscle: muscle ? 1 : 0,
    amp: q4(rng.range(0.1, 0.35)),
    freq: q4(rng.range(0.8, 3.0)),
    phase: q4(rng.range(0, TWO_PI)),
  };
}

function dist(nodes, i, j) {
  const dx = nodes[j].x - nodes[i].x;
  const dy = nodes[j].y - nodes[i].y;
  return Math.sqrt(dx * dx + dy * dy);
}

export function randomGenome(rng) {
  const n = 3 + rng.int(5); // 3 to 7 nodes
  const nodes = [];
  for (let i = 0; i < n; i++) {
    nodes.push({ x: q4(rng.range(-0.35, 0.35)), y: q4(rng.range(0, 0.35)) });
  }
  const g = { nodes, springs: [] };
  const has = (a, b) => g.springs.some((s) => s.a === Math.min(a, b) && s.b === Math.max(a, b));
  // Spanning tree first (every node attaches to an earlier one), then short extra links.
  for (let i = 1; i < n; i++) {
    const j = rng.int(i);
    g.springs.push(makeSpring(i, j, dist(nodes, i, j), rng.next() < 0.6, rng));
  }
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      if (has(i, j)) continue;
      const d = dist(nodes, i, j);
      if (d < 0.8 && rng.next() < 0.3) g.springs.push(makeSpring(i, j, d, rng.next() < 0.6, rng));
    }
  }
  if (!g.springs.some((s) => s.muscle)) g.springs[rng.int(g.springs.length)].muscle = 1;
  return repair(g);
}

// Deterministic repair: quantise and clamp, drop bad or duplicate springs, keep at least one spring and one
// muscle, and connect every node to the body. Unreached nodes join through the nearest reached node
// (lowest index first on ties). Consumes no random numbers.
export function repair(g) {
  const nodes = g.nodes.slice(0, NODE_MAX).map((n) => ({
    x: q4(clamp(n.x, -POS_MAX, POS_MAX)),
    y: q4(clamp(n.y, -POS_MAX, POS_MAX)),
  }));
  while (nodes.length < NODE_MIN) nodes.push({ x: q4(0.1 * nodes.length), y: 0.1 });
  const n = nodes.length;
  const seen = new Set();
  const springs = [];
  for (const s of g.springs) {
    const a = Math.min(s.a, s.b);
    const b = Math.max(s.a, s.b);
    if (a === b || a < 0 || b >= n) continue;
    const key = a * 64 + b;
    if (seen.has(key)) continue;
    seen.add(key);
    springs.push({
      a,
      b,
      rest: q4(clamp(s.rest, REST_MIN, REST_MAX)),
      k: q4(clamp(s.k, K_MIN, K_MAX)),
      c: q4(clamp(s.c, C_MIN, C_MAX)),
      muscle: s.muscle ? 1 : 0,
      amp: q4(clamp(s.amp, 0, AMP_MAX)),
      freq: q4(clamp(s.freq, FREQ_MIN, FREQ_MAX)),
      phase: q4(s.phase),
    });
  }
  const out = { nodes, springs };
  // Connectivity: grow the reached set from node 0 until every node is reached.
  for (;;) {
    const reached = new Array(n).fill(false);
    reached[0] = true;
    let grew = true;
    while (grew) {
      grew = false;
      for (const s of out.springs) {
        if (reached[s.a] !== reached[s.b]) {
          reached[s.a] = true;
          reached[s.b] = true;
          grew = true;
        }
      }
    }
    const u = reached.indexOf(false);
    if (u < 0) break;
    let best = -1;
    let bestD = Infinity;
    for (let v = 0; v < n; v++) {
      if (!reached[v]) continue;
      const d = dist(nodes, u, v);
      if (d < bestD) {
        bestD = d;
        best = v;
      }
    }
    out.springs.push({
      a: Math.min(u, best),
      b: Math.max(u, best),
      rest: q4(clamp(bestD, REST_MIN, REST_MAX)),
      k: 150,
      c: 2,
      muscle: 0,
      amp: 0,
      freq: 1,
      phase: 0,
    });
  }
  if (out.springs.length === 0) {
    out.springs.push({ a: 0, b: 1, rest: q4(clamp(dist(nodes, 0, 1), REST_MIN, REST_MAX)), k: 150, c: 2, muscle: 0, amp: 0, freq: 1, phase: 0 });
  }
  if (!out.springs.some((s) => s.muscle)) out.springs[0].muscle = 1;
  return out;
}

export function isValid(g) {
  const n = g.nodes.length;
  if (n < NODE_MIN || n > NODE_MAX || g.springs.length < 1 || g.springs.length > SPRING_MAX) return false;
  if (!g.springs.some((s) => s.muscle)) return false;
  const seen = new Set();
  for (const s of g.springs) {
    if (!(s.a < s.b) || s.a < 0 || s.b >= n) return false;
    const key = s.a * 64 + s.b;
    if (seen.has(key)) return false;
    seen.add(key);
    if (s.rest < REST_MIN || s.rest > REST_MAX || s.k < K_MIN || s.k > K_MAX || s.c < C_MIN || s.c > C_MAX) return false;
  }
  const reached = new Array(n).fill(false);
  reached[0] = true;
  let grew = true;
  while (grew) {
    grew = false;
    for (const s of g.springs) {
      if (reached[s.a] !== reached[s.b]) {
        reached[s.a] = true;
        reached[s.b] = true;
        grew = true;
      }
    }
  }
  return reached.every(Boolean);
}

// ---------------------------------------------------------------- encoding
// Text form, shared with the Python port and the editor: "w1|x,y;x,y;...|a,b,rest,k,c,muscle,amp,freq,phase;..."
// Numbers are written with up to four decimals and no trailing zeros. Decoding is exact for genomes on the grid.
function fmt4(v) {
  let s = v.toFixed(4);
  if (s.indexOf('.') >= 0) s = s.replace(/0+$/, '').replace(/\.$/, '');
  return s;
}

export function encode(g) {
  const nodes = g.nodes.map((n) => `${fmt4(n.x)},${fmt4(n.y)}`).join(';');
  const springs = g.springs
    .map((s) => [s.a, s.b, fmt4(s.rest), fmt4(s.k), fmt4(s.c), s.muscle ? 1 : 0, fmt4(s.amp), fmt4(s.freq), fmt4(s.phase)].join(','))
    .join(';');
  return `w1|${nodes}|${springs}`;
}

export function decode(text) {
  const parts = String(text).trim().split('|');
  if (parts.length !== 3 || parts[0] !== 'w1') throw new Error('not a walkers creature (expected w1|nodes|springs)');
  const nodes = parts[1].split(';').map((p) => {
    const [x, y] = p.split(',').map(Number);
    return { x, y };
  });
  const springs = parts[2].split(';').map((p) => {
    const f = p.split(',').map(Number);
    if (f.length !== 9 || f.some((v) => !Number.isFinite(v))) throw new Error('bad spring: ' + p);
    return { a: f[0], b: f[1], rest: f[2], k: f[3], c: f[4], muscle: f[5] ? 1 : 0, amp: f[6], freq: f[7], phase: f[8] };
  });
  const g = { nodes, springs };
  if (nodes.some((n) => !Number.isFinite(n.x) || !Number.isFinite(n.y))) throw new Error('bad node');
  return g;
}

// ---------------------------------------------------------------- body and physics
// buildBody places the genome above the floor: centred on SPAWN_X, lowest node SPAWN_LIFT up. The spine is the
// pair of nodes farthest apart at spawn (first pair on ties); its rotation is the "flip" measure.
export function buildBody(g) {
  const N = g.nodes.length;
  const S = g.springs.length;
  let sx = 0;
  let minY = Infinity;
  for (const n of g.nodes) {
    sx += n.x;
    if (n.y < minY) minY = n.y;
  }
  const mx = sx / N;
  const px = new Float64Array(N);
  const py = new Float64Array(N);
  for (let i = 0; i < N; i++) {
    px[i] = SPAWN_X + (g.nodes[i].x - mx);
    py[i] = g.nodes[i].y - minY + SPAWN_LIFT;
  }
  const ia = new Int32Array(S);
  const ib = new Int32Array(S);
  const rest = new Float64Array(S);
  const k = new Float64Array(S);
  const c = new Float64Array(S);
  const mus = new Uint8Array(S);
  const amp = new Float64Array(S);
  const w = new Float64Array(S);
  const ph = new Float64Array(S);
  for (let s = 0; s < S; s++) {
    const sp = g.springs[s];
    ia[s] = sp.a;
    ib[s] = sp.b;
    rest[s] = sp.rest;
    k[s] = sp.k;
    c[s] = sp.c;
    mus[s] = sp.muscle ? 1 : 0;
    amp[s] = sp.amp;
    w[s] = TWO_PI * sp.freq;
    ph[s] = sp.phase;
  }
  let sa = 0;
  let sb = N > 1 ? 1 : 0;
  let bestD2 = -1;
  for (let i = 0; i < N; i++) {
    for (let j = i + 1; j < N; j++) {
      const dx = px[j] - px[i];
      const dy = py[j] - py[i];
      const d2 = dx * dx + dy * dy;
      if (d2 > bestD2) {
        bestD2 = d2;
        sa = i;
        sb = j;
      }
    }
  }
  return { N, S, px, py, vx: new Float64Array(N), vy: new Float64Array(N), ia, ib, rest, k, c, mus, amp, w, ph, sa, sb };
}

// simulate runs one creature for world.duration seconds at DT, semi-implicit Euler.
// Per step: (1) spring and muscle forces from the positions at the start of the step, accumulated in two
// ordered passes (all "a" ends in spring order, then all "b" ends); (2) velocities, gravity, drag and the
// velocity clamp; (3) positions; (4) ground contact; (5) bookkeeping.
// Ground contact: a node below the ground is put on it. Normal speed reverses with restitution only for impacts
// faster than REST_SPEED. Tangential speed is cut by a Coulomb limit: it sticks if |vt| <= static * (1+e)|vn|,
// otherwise it loses kinetic * (1+e)|vn|. Friction is the kinetic coefficient; the static one is STATIC_RATIO times that.
// Fitness is the centre-of-mass travel in x, less SPIN_PENALTY per radian of spine rotation beyond SPIN_FREE, plus
// SPEED_BONUS times the mean forward speed over the second half of the run (a body that keeps going scores more than
// one that lunges and stops). A body that fell gets no speed bonus.
// A body that falls below FALL_Y is scored where it fell. A NaN makes the fitness 0 and flags the run.
export function simulate(g, world = DEFAULT_WORLD, opts = {}) {
  const b = buildBody(g);
  const { N, S, px, py, vx, vy, ia, ib, rest, k, c, mus, amp, w, ph, sa, sb } = b;
  const steps = Math.round(world.duration / DT);
  const grav = world.gravity;
  const muK = world.friction;
  const muS = world.friction * STATIC_RATIO;
  const e = world.restitution;
  const terrain = world.terrain;
  const damp = 1 - AIR_DRAG * DT;
  const fx = new Float64Array(N);
  const fy = new Float64Array(N);
  const Fx = new Float64Array(S);
  const Fy = new Float64Array(S);
  const sampleEvery = opts.sampleEvery || 0;
  const frameCount = sampleEvery ? Math.floor(steps / sampleEvery) + 1 : 0;
  // Float32 keeps the transfer small for the page; opts.float64 keeps full precision for the parity tests.
  const FrameArray = opts.float64 ? Float64Array : Float32Array;
  const frames = sampleEvery ? new FrameArray(frameCount * N * 2) : null;
  let fr = 0;
  const record = () => {
    for (let i = 0; i < N; i++) {
      frames[fr * N * 2 + 2 * i] = px[i];
      frames[fr * N * 2 + 2 * i + 1] = py[i];
    }
    fr++;
  };
  if (frames) record();

  let com0 = 0;
  for (let i = 0; i < N; i++) com0 += px[i];
  com0 /= N;
  let ux = px[sb] - px[sa];
  let uy = py[sb] - py[sa];
  let ul = Math.sqrt(ux * ux + uy * uy);
  if (ul < 1e-9) ul = 1e-9;
  ux /= ul;
  uy /= ul;

  let fallen = false;
  let distFall = 0;
  let spinFall = 0;
  let spin = 0;
  let exploded = false;
  let clamps = 0;
  let stepsRun = 0;
  let com = com0;
  const half = Math.floor(steps / 2);
  let comMid = com0;

  for (let step = 0; step < steps; step++) {
    const t = step * DT;
    for (let s = 0; s < S; s++) {
      const a = ia[s];
      const bb = ib[s];
      const dx = px[bb] - px[a];
      const dy = py[bb] - py[a];
      let L = Math.sqrt(dx * dx + dy * dy);
      if (L < 1e-9) L = 1e-9;
      const nx = dx / L;
      const ny = dy / L;
      const r = mus[s] ? rest[s] * (1 + amp[s] * wsin(w[s] * t + ph[s])) : rest[s];
      const f = k[s] * (L - r) + c[s] * ((vx[bb] - vx[a]) * nx + (vy[bb] - vy[a]) * ny);
      Fx[s] = f * nx;
      Fy[s] = f * ny;
    }
    fx.fill(0);
    fy.fill(0);
    for (let s = 0; s < S; s++) {
      fx[ia[s]] += Fx[s];
      fy[ia[s]] += Fy[s];
    }
    for (let s = 0; s < S; s++) {
      fx[ib[s]] -= Fx[s];
      fy[ib[s]] -= Fy[s];
    }
    for (let i = 0; i < N; i++) {
      let vxi = (vx[i] + fx[i] * DT) * damp;
      let vyi = (vy[i] + (fy[i] - grav) * DT) * damp;
      if (vxi > VMAX) {
        vxi = VMAX;
        clamps++;
      } else if (vxi < -VMAX) {
        vxi = -VMAX;
        clamps++;
      }
      if (vyi > VMAX) {
        vyi = VMAX;
        clamps++;
      } else if (vyi < -VMAX) {
        vyi = -VMAX;
        clamps++;
      }
      vx[i] = vxi;
      vy[i] = vyi;
      px[i] += vxi * DT;
      py[i] += vyi * DT;
    }
    for (let i = 0; i < N; i++) {
      if (!Number.isFinite(px[i]) || !Number.isFinite(py[i])) {
        exploded = true;
        break;
      }
      groundAt(terrain, px[i]);
      const h = G.h;
      if (py[i] < h) {
        const sl = G.s;
        const inv = 1 / Math.sqrt(1 + sl * sl);
        const nx = -sl * inv;
        const ny = inv;
        const tx = ny;
        const ty = -nx;
        py[i] = h;
        const vn = vx[i] * nx + vy[i] * ny;
        if (vn < 0) {
          const vt = vx[i] * tx + vy[i] * ty;
          const speed = -vn;
          const ee = speed > REST_SPEED ? e : 0;
          const vnNew = -ee * vn;
          const lim = (1 + ee) * speed;
          let vtNew;
          if (Math.abs(vt) <= muS * lim) vtNew = 0;
          else vtNew = vt > 0 ? vt - muK * lim : vt + muK * lim;
          vx[i] = vnNew * nx + vtNew * tx;
          vy[i] = vnNew * ny + vtNew * ty;
        }
      }
    }
    if (exploded) break;
    stepsRun = step + 1;
    if (!fallen) {
      com = 0;
      let comY = 0;
      for (let i = 0; i < N; i++) {
        com += px[i];
        comY += py[i];
      }
      com /= N;
      comY /= N;
      let dxs = px[sb] - px[sa];
      let dys = py[sb] - py[sa];
      let dl = Math.sqrt(dxs * dxs + dys * dys);
      if (dl < 1e-9) dl = 1e-9;
      dxs /= dl;
      dys /= dl;
      spin += Math.abs(ux * dys - uy * dxs);
      ux = dxs;
      uy = dys;
      if (comY < FALL_Y) {
        fallen = true;
        distFall = com - com0;
        spinFall = spin;
      }
      if (stepsRun === half) comMid = com;
    }
    if (frames && stepsRun % sampleEvery === 0) record();
  }

  let distance = 0;
  let spinUsed = spin;
  if (!exploded) {
    if (fallen) {
      distance = distFall;
      spinUsed = spinFall;
    } else {
      distance = com - com0;
    }
  }
  // Mean forward speed over the second half: (centre of mass at the end - at the midpoint) / (second half's seconds).
  let speedBonus = 0;
  if (!exploded && !fallen && steps > half) {
    speedBonus = (SPEED_BONUS * (com - comMid)) / ((steps - half) * DT);
  }
  const fitness = exploded ? 0 : distance - SPIN_PENALTY * Math.max(0, spinUsed - SPIN_FREE) + speedBonus;
  return {
    fitness,
    distance: exploded ? 0 : distance,
    exploded,
    fallen,
    spin: spinUsed,
    speedBonus: exploded ? 0 : speedBonus,
    clamps,
    steps: stepsRun,
    frames,
    frameCount: frames ? fr : 0,
    N,
  };
}

// The actuation level of a muscle at time t, in [-amp, amp]: the rest length is rest * (1 + level).
export function actuation(s, t) {
  return s.amp * wsin(TWO_PI * s.freq * t + s.phase);
}

// ---------------------------------------------------------------- genetic operators
function edgeKeys(g) {
  const out = new Set();
  for (const s of g.springs) out.add(s.a * 64 + s.b);
  return out;
}

// Compatibility distance for speciation: node and spring counts, muscle fraction, and the Jaccard distance of
// the edge sets (node indices). Weights 0.4 per node, 0.3 per spring, 1 for muscle fraction and 1 for edges.
export function compatibility(a, b) {
  const sa = a.springs.length;
  const sb = b.springs.length;
  const ma = a.springs.reduce((acc, s) => acc + s.muscle, 0) / sa;
  const mb = b.springs.reduce((acc, s) => acc + s.muscle, 0) / sb;
  const ea = edgeKeys(a);
  const eb = edgeKeys(b);
  let inter = 0;
  for (const key of ea) if (eb.has(key)) inter++;
  const uni = ea.size + eb.size - inter;
  const jac = uni === 0 ? 1 : inter / uni;
  return 0.4 * Math.abs(a.nodes.length - b.nodes.length) + 0.3 * Math.abs(sa - sb) + Math.abs(ma - mb) + (1 - jac);
}

function tournament(list, k, rng) {
  let best = list[rng.int(list.length)];
  for (let i = 1; i < k; i++) {
    const cand = list[rng.int(list.length)];
    if (cand.fitness > best.fitness) best = cand;
  }
  return best;
}

// Crossover aligned by node index. The fitter parent (ties: the first) is dominant: it sets the node count.
// Each node index the other parent also has takes its position from either parent with probability 1/2. Springs
// present in both parents (same endpoint pair) take either parent's genes with probability 1/2. Springs only the
// dominant parent has are kept; springs only the other parent has are dropped (NEAT's rule for disjoint genes).
// The result is repaired, so it is always connected.
export function crossover(a, b, rng, fa, fb) {
  const [d, o] = fb > fa ? [b, a] : [a, b];
  const nN = d.nodes.length;
  const nodes = [];
  for (let i = 0; i < nN; i++) {
    if (i < o.nodes.length && rng.next() < 0.5) nodes.push({ x: o.nodes[i].x, y: o.nodes[i].y });
    else nodes.push({ x: d.nodes[i].x, y: d.nodes[i].y });
  }
  const oBy = new Map();
  for (const s of o.springs) oBy.set(s.a * 64 + s.b, s);
  const springs = [];
  for (const sd of d.springs) {
    const other = oBy.get(sd.a * 64 + sd.b);
    if (other && rng.next() < 0.5) springs.push({ ...other });
    else springs.push({ ...sd });
  }
  return repair({ nodes, springs: springs.map((s) => ({ ...s })) });
}

function nearest(g, i, exclude) {
  let best = -1;
  let bestD = Infinity;
  for (let j = 0; j < g.nodes.length; j++) {
    if (j === i || j === exclude) continue;
    const d = dist(g.nodes, i, j);
    if (d < bestD) {
      bestD = d;
      best = j;
    }
  }
  return best;
}

function hasSpring(g, a, b) {
  const lo = Math.min(a, b);
  const hi = Math.max(a, b);
  return g.springs.some((s) => s.a === lo && s.b === hi);
}

function addNode(g, rng) {
  const i = rng.int(g.nodes.length);
  const base = g.nodes[i];
  const x = q4(clamp(base.x + rng.range(-0.3, 0.3), -POS_MAX, POS_MAX));
  const y = q4(clamp(base.y + rng.range(-0.3, 0.3), -POS_MAX, POS_MAX));
  g.nodes.push({ x, y });
  const idx = g.nodes.length - 1;
  g.springs.push(makeSpring(i, idx, dist(g.nodes, i, idx), rng.next() < 0.5, rng));
  const j = nearest(g, idx, i);
  if (j >= 0) g.springs.push(makeSpring(j, idx, dist(g.nodes, j, idx), rng.next() < 0.5, rng));
}

function removeNode(g, rng) {
  const r = rng.int(g.nodes.length);
  g.nodes.splice(r, 1);
  g.springs = g.springs.filter((s) => s.a !== r && s.b !== r).map((s) => ({
    ...s,
    a: s.a > r ? s.a - 1 : s.a,
    b: s.b > r ? s.b - 1 : s.b,
  }));
}

function addSpring(g, rng) {
  const n = g.nodes.length;
  for (let attempt = 0; attempt < 12; attempt++) {
    const a = rng.int(n);
    const b = rng.int(n);
    if (a === b || hasSpring(g, a, b)) continue;
    g.springs.push(makeSpring(a, b, dist(g.nodes, a, b), rng.next() < 0.5, rng));
    return;
  }
}

function removeSpring(g, rng) {
  g.springs.splice(rng.int(g.springs.length), 1);
}

function toggleMuscle(g, rng) {
  const s = g.springs[rng.int(g.springs.length)];
  s.muscle = s.muscle ? 0 : 1;
}

// Mutation: each node and each spring has a rate/2 chance of a parameter jitter; four structural operators each
// fire with their own chance (add node 0.15 x rate, remove node 0.1, add spring 0.2, remove spring 0.15), and
// toggle muscle 0.2. The result is repaired. Operators run in this fixed order.
export function mutate(g0, rng, rate) {
  const g = cloneGenome(g0);
  for (const n of g.nodes) {
    if (rng.next() < rate * 0.5) {
      n.x = q4(clamp(n.x + rng.range(-0.06, 0.06), -POS_MAX, POS_MAX));
      n.y = q4(clamp(n.y + rng.range(-0.06, 0.06), -POS_MAX, POS_MAX));
    }
  }
  for (const s of g.springs) {
    if (rng.next() < rate * 0.5) {
      s.rest = q4(clamp(s.rest * (1 + rng.range(-0.1, 0.1)), REST_MIN, REST_MAX));
      s.k = q4(clamp(s.k * (1 + rng.range(-0.2, 0.2)), K_MIN, K_MAX));
      s.c = q4(clamp(s.c * (1 + rng.range(-0.2, 0.2)), C_MIN, C_MAX));
      s.amp = q4(clamp(s.amp + rng.range(-0.05, 0.05), 0, AMP_MAX));
      s.freq = q4(clamp(s.freq * (1 + rng.range(-0.2, 0.2)), FREQ_MIN, FREQ_MAX));
      s.phase = q4(s.phase + rng.range(-0.5, 0.5));
    }
  }
  if (rng.next() < rate * 0.15 && g.nodes.length < NODE_MAX) addNode(g, rng);
  if (rng.next() < rate * 0.1 && g.nodes.length > NODE_MIN) removeNode(g, rng);
  if (rng.next() < rate * 0.2 && g.springs.length < SPRING_MAX) addSpring(g, rng);
  if (rng.next() < rate * 0.15 && g.springs.length > 1) removeSpring(g, rng);
  // A body can lose all its springs when a node is removed; the draw still happens so the streams stay aligned.
  if (rng.next() < rate * 0.2 && g.springs.length > 0) toggleMuscle(g, rng);
  return repair(g);
}

// ---------------------------------------------------------------- the evolver
// One object drives the whole run: the worker holds it, and the Python CLI uses the same class in spirit.
// Population members: {id, genome, parents, gen, user, fitness, distance, exploded, fallen, species}.
// The lineage record for each evaluated member is appended to `records` (its code is the text encoding).
export class Evolver {
  constructor({ seed = 1, popSize = GA_DEFAULTS.popSize, mutationRate = GA_DEFAULTS.mutationRate, world = DEFAULT_WORLD } = {}) {
    this.rng = mulberry32(seed);
    this.popSize = popSize;
    this.mutationRate = mutationRate;
    this.world = { ...world };
    this.generation = 1;
    this.nextId = 1;
    this.nextSpeciesId = 1;
    this.species = [];
    this.injected = [];
    this.records = [];
    this.population = [];
    for (let i = 0; i < popSize; i++) this.population.push(this.member(randomGenome(this.rng), []));
  }

  member(genome, parents, user = false) {
    return { id: this.nextId++, genome, parents, gen: this.generation, user, fitness: 0, distance: 0, exploded: false, fallen: false, species: 0 };
  }

  // Queue a user creature. It joins the next generation in place of the weakest slot.
  inject(genome) {
    this.injected.push(repair(genome));
  }

  evaluate() {
    for (const m of this.population) {
      const r = simulate(m.genome, this.world);
      m.fitness = r.fitness;
      m.distance = r.distance;
      m.exploded = r.exploded;
      m.fallen = r.fallen;
      m.record = {
        id: m.id,
        gen: m.gen,
        parents: m.parents,
        fitness: r.fitness,
        distance: r.distance,
        exploded: r.exploded,
        fallen: r.fallen,
        user: m.user,
        species: 0,
        code: encode(m.genome),
      };
      this.records.push(m.record);
    }
  }

  // Speciation: each member joins the first species whose representative is within the threshold. A new species
  // is founded by the first member that matches none. After the sorting, each species picks a random member as its
  // next representative (one draw per species, in species order).
  speciate(threshold = GA_DEFAULTS.speciesThreshold) {
    for (const sp of this.species) sp.members = [];
    const created = [];
    for (const m of this.population) {
      let sp = this.species.find((s) => compatibility(m.genome, s.rep) < threshold);
      if (!sp) {
        sp = { id: this.nextSpeciesId++, rep: m.genome, members: [], created: this.generation };
        this.species.push(sp);
        created.push(sp.id);
      }
      sp.members.push(m);
      m.species = sp.id;
      m.record.species = sp.id;
    }
    this.species = this.species.filter((s) => s.members.length > 0);
    for (const sp of this.species) sp.rep = sp.members[this.rng.int(sp.members.length)].genome;
    return created;
  }

  // Breeding. Shifted fitness (fitness - min + 1) is divided by species size (fitness sharing), and each species
  // gets a share of the population proportional to its sum. Shares are floored, and the remainder goes to the
  // largest fractional parts (earlier species first on ties). Within a species: an elite (its best, unchanged)
  // when it has at least eliteMin members or holds the global best; the rest are children of tournament picks
  // from the top half, by crossover (rate crossoverRate) or cloning, then mutation.
  breed() {
    const P = this.popSize;
    let minF = Infinity;
    let bestMember = null;
    for (const m of this.population) {
      if (m.fitness < minF) minF = m.fitness;
      if (!bestMember || m.fitness > bestMember.fitness) bestMember = m;
    }
    const shares = this.species.map((sp) => {
      let s = 0;
      for (const m of sp.members) s += (m.fitness - minF + 1) / sp.members.length;
      return s;
    });
    const total = shares.reduce((a, b) => a + b, 0);
    const exact = shares.map((s) => (P * s) / total);
    const quota = exact.map((x) => Math.floor(x));
    let rem = P - quota.reduce((a, b) => a + b, 0);
    const order = exact.map((x, i) => i).sort((i, j) => exact[j] - quota[j] - (exact[i] - quota[i]) || i - j);
    for (let r = 0; r < rem; r++) quota[order[r % order.length]]++;

    const next = [];
    this.species.forEach((sp, si) => {
      const q = quota[si];
      if (q <= 0) return;
      const sorted = sp.members.slice().sort((a, b) => b.fitness - a.fitness || a.id - b.id);
      const keep = Math.max(1, Math.ceil(sorted.length * GA_DEFAULTS.survivalFrac));
      const survivors = sorted.slice(0, keep);
      let made = 0;
      const hasBest = sp.members.includes(bestMember);
      if (sp.members.length >= GA_DEFAULTS.eliteMin || hasBest) {
        next.push(this.member(cloneGenome(sorted[0].genome), [sorted[0].id]));
        made++;
      }
      while (made < q) {
        let child;
        let parents;
        if (survivors.length >= 2 && this.rng.next() < GA_DEFAULTS.crossoverRate) {
          const a = tournament(survivors, GA_DEFAULTS.tournament, this.rng);
          const b = tournament(survivors, GA_DEFAULTS.tournament, this.rng);
          child = crossover(a.genome, b.genome, this.rng, a.fitness, b.fitness);
          parents = [a.id, b.id];
        } else {
          const a = tournament(survivors, GA_DEFAULTS.tournament, this.rng);
          child = cloneGenome(a.genome);
          parents = [a.id];
        }
        next.push(this.member(mutate(child, this.rng, this.mutationRate), parents));
        made++;
      }
    });
    // Injected user creatures take the last slots (replacing children, never the elites at the front).
    const slots = Math.min(this.injected.length, next.length);
    for (let k = 0; k < slots; k++) {
      next[next.length - 1 - k] = this.member(cloneGenome(this.injected[k]), [], true);
    }
    this.injected = this.injected.slice(slots);
    // Quotas sum to P by construction; trim or pad defensively so the population size never drifts.
    while (next.length > P) next.pop();
    while (next.length < P) next.push(this.member(randomGenome(this.rng), []));
    return next;
  }

  // One generation: evaluate, summarise, speciate, breed. Returns the summary of the evaluated population.
  // With topK > 0 the summary also carries `top`: frames of the topK evaluated members (taken before breeding,
  // so they show the population that was scored, not its children).
  step(topK = 0) {
    this.evaluate();
    // Speciate before summarising, so the champion's species is set. Speciation and the summary draw no random
    // numbers, so this order does not change the run.
    const created = this.speciate();
    const summary = this.summary();
    summary.newSpecies = created;
    summary.speciesSizes = this.species.map((sp) => ({ id: sp.id, size: sp.members.length }));
    if (topK > 0) summary.top = this.topFrames(topK, 4);
    // The generation counter moves on before breeding, so the children carry the new generation number.
    this.generation++;
    this.population = this.breed();
    return summary;
  }

  summary() {
    const pop = this.population;
    const fits = pop.map((m) => m.fitness).sort((a, b) => a - b);
    const n = fits.length;
    const mean = fits.reduce((a, b) => a + b, 0) / n;
    const median = n % 2 ? fits[(n - 1) / 2] : (fits[n / 2 - 1] + fits[n / 2]) / 2;
    let champ = pop[0];
    for (const m of pop) if (m.fitness > champ.fitness || (m.fitness === champ.fitness && m.id < champ.id)) champ = m;
    const user = pop.filter((m) => m.user).map((m) => ({ id: m.id, fitness: m.fitness, distance: m.distance }));
    return {
      gen: this.generation,
      best: champ.fitness,
      mean,
      median,
      exploded: pop.filter((m) => m.exploded).length,
      champion: { id: champ.id, fitness: champ.fitness, distance: champ.distance, parents: champ.parents, species: champ.species, code: champ.record.code },
      user,
      species: this.species.length,
    };
  }

  // Frames for the k fittest members, for the live view: positions every `sampleEvery` steps (default 4, 60 Hz).
  topFrames(k = 4, sampleEvery = 4) {
    const ranked = this.population.slice().sort((a, b) => b.fitness - a.fitness || a.id - b.id).slice(0, k);
    return ranked.map((m) => {
      const r = simulate(m.genome, this.world, { sampleEvery });
      return { id: m.id, fitness: r.fitness, N: r.N, frameCount: r.frameCount, sampleEvery, frames: r.frames };
    });
  }
}
