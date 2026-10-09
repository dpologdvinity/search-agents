// Grid MDP lab page. Paint a slippery grid world, then watch value iteration, policy iteration and a
// Q-learning agent solve it side by side, or one at a time.
//
// Every number comes from mdplab-core.js, which the parity test checks against the Python package. This file
// wires the controls, runs the three solvers a little at a time each frame, animates the agent along its last
// walk, and draws the boards and charts on canvas.

import {
  DEFAULT_PARAMS, GridModel, PolicyIterator, QLearner, ValueIterator, greedyPolicy, presetByKey, qTable, reachable,
} from './mdplab-core.js';
import { banner, burst, pop, shake } from './fx.js';

const $ = (id) => document.getElementById(id);
const REDUCED = matchMedia('(prefers-reduced-motion: reduce)').matches;
const DIRS = [[0, -1], [1, 0], [0, 1], [-1, 0]];
const COL = { navy: [10, 18, 48], cyan: [0, 245, 255], pink: [255, 0, 160], green: [0, 255, 136], yellow: [255, 230, 0] };

// One speed control drives all three. Planner sweeps and agent episodes are per frame or per second, and the walk
// animation runs at stepMs per cell, so TURBO fast-forwards thousands of episodes while the orb keeps moving.
const SPEEDS = [
  { name: 'SLOW', vi: 1, pi: 4, eps: 3, stepMs: 160 },
  { name: 'NORMAL', vi: 2, pi: 25, eps: 40, stepMs: 90 },
  { name: 'FAST', vi: 8, pi: 120, eps: 400, stepMs: 40 },
  { name: 'TURBO', vi: 40, pi: 900, eps: 4000, stepMs: 15 },
];
const CAP = { viSweeps: 20000, piEval: 5000, optSweeps: 20000, stepsPerFrame: 60000, episodesPerFrame: 5000 };
const HIST = { viMax: 4000, piMax: 2500, retMax: 4000, agreeMax: 1500 };

const app = {
  key: 'rooms',
  grid: [],            // editable layout: an array of arrays of cell characters
  params: { ...DEFAULT_PARAMS },
  seed: 1,
  conv: 1e-4,
  numbers: false,
  compare: false,
  algo: 'q',
  tool: 'reward',
  speed: 1,
  dirty: true,
  model: null,
  opt: null,           // the reference optimal policy and Q, from value iteration to 1e-10
  vi: null,
  pi: null,
  q: null,
  viRun: true,
  piRun: true,
  qRun: true,
  viDone: false,
  piDone: false,
  qDone: false,
  viHist: [],
  piHist: [],          // {r} for an evaluation sweep, {mark, changed} for an improvement step
  piPhase: 'eval',
  piEvalCount: 0,
  piFlash: new Map(),  // cell -> time of change, for the ring drawn where an action flipped
  viFlash: new Map(),
  prevVI: null,
  qReturns: [],
  qAgree: [],
  qAcc: 0,
  qAgreeRun: 0,        // consecutive samples in which every reachable cell matched the optimal policy
  qFlags: null,
  qAgreeFrac: null,
  orbCell: -1,
  walk: { path: null, t: 0, ep: null, pending: null, trails: [], boom: null },
};

// ── helpers ──────────────────────────────────────────────────────────────

const fmtExp = (x) => (Number.isFinite(x) ? x.toExponential(1) : '—');
const fmt = (x, d = 2) => x.toFixed(d).replace('-', '−');
const pct = (x) => `${Math.round(x * 100)}%`;
const rowsOf = (grid) => grid.map((r) => r.join(''));
const gridOf = (rows) => rows.map((r) => r.split(''));

function maxAbs(values, states) {
  let m = 0;
  for (const s of states) m = Math.max(m, Math.abs(values[s]));
  return m > 1e-9 ? m : 1;
}

// Cell colour for a value: navy at zero, cyan above and pink below, brighter with magnitude.
function heat(v, scale, alpha = 1) {
  const t = Math.min(1, Math.abs(v) / scale);
  const to = v >= 0 ? COL.cyan : COL.pink;
  const k = Math.pow(t, 0.7);
  const c = COL.navy.map((a, i) => a + (to[i] - a) * k);
  return `rgba(${c[0] | 0},${c[1] | 0},${c[2] | 0},${alpha})`;
}

// Canvas sized to its CSS width; the board's height follows its aspect ratio. Returns the 2D context in CSS px.
function prepare(canvas, W, H) {
  const dpr = Math.min(2, window.devicePixelRatio || 1);
  const bw = Math.round(W * dpr);
  const bh = Math.round(H * dpr);
  if (canvas.width !== bw || canvas.height !== bh) {
    canvas.width = bw;
    canvas.height = bh;
  }
  const ctx = canvas.getContext('2d');
  ctx.setTransform(bw / W, 0, 0, bh / H, 0, 0);
  ctx.clearRect(0, 0, W, H);
  return ctx;
}

// Centre of cell i in pixels, for a board whose cells are `size` px wide.
function cellCenter(m, size, i) {
  return [(i % m.W + 0.5) * size, (Math.floor(i / m.W) + 0.5) * size];
}

// ── presets, sliders, and the model ─────────────────────────────────────

function loadPreset(key) {
  const p = presetByKey(key);
  app.key = key;
  app.grid = gridOf(p.rows);
  app.params = { ...p.params };
  syncSliders();
  $('preset-blurb').textContent = p.blurb;
  document.querySelectorAll('[data-preset]').forEach((b) => b.setAttribute('aria-pressed', String(b.dataset.preset === key)));
  app.dirty = true;
}

function clearWorld() {
  const H = app.grid.length;
  const W = app.grid[0].length;
  app.grid = Array.from({ length: H }, () => Array(W).fill('.'));
  app.grid[H - 1][0] = 'S';
  app.grid[0][W - 1] = '+';
  app.key = 'custom';
  $('preset-blurb').textContent = 'A blank room with the start in the bottom-left and a reward in the top-right. Paint your own.';
  document.querySelectorAll('[data-preset]').forEach((b) => b.setAttribute('aria-pressed', 'false'));
  app.dirty = true;
}

const SLIDERS = [
  // id, params key, sign, decimals, on change: 'model' rebuilds the world, 'agent' restarts Q-learning
  ['gamma', 'gamma', 1, 2, 'model'],
  ['slip', 'slip', 1, 2, 'model'],
  ['living', 'living', 1, 2, 'model'],
  ['reward', 'reward', 1, 1, 'model'],
  ['pit', 'pit', -1, 1, 'model'],
  ['alpha', 'alpha', 1, 2, 'agent'],
  ['epsilon', 'epsilon', 1, 2, 'agent'],
  ['decay', 'decay', 1, 4, 'agent'],
];

function syncSliders() {
  for (const [id, key, sign, dec] of SLIDERS) {
    const v = sign * app.params[key];
    $(id).value = String(v);
    $(`${id}-val`).textContent = fmt(v, dec);
  }
  $('seed').value = String(app.seed);
  $('conv').value = String(app.conv);
  $('q-algo').value = app.algo;
  $('numbers').checked = app.numbers;
  $('q-compare').checked = app.compare;
  $('seed-val').textContent = String(app.seed);
}

function rebuild() {
  app.dirty = false;
  try {
    app.model = new GridModel(rowsOf(app.grid), app.params);
  } catch (e) {
    $('world-hint').textContent = `That layout is not valid: ${e.message}`;
    return;
  }
  $('world-hint').textContent = 'Paint with the selected tool. Any change restarts all three solvers.';
  app.opt = solveOptimal(app.model);
  $('chip-world').textContent = app.key === 'custom' ? 'CUSTOM' : presetByKey(app.key).label.toUpperCase();
  resetVI();
  resetPI();
  resetQ();
}

// The reference the agent is judged against: value iteration run to 1e-10, then its greedy policy.
function solveOptimal(m) {
  const vi = new ValueIterator(m);
  vi.solve(1e-10, CAP.optSweeps);
  return { V: vi.V, pi: greedyPolicy(m, vi.V), Q: qTable(m, vi.V) };
}

// A reset starts the solver again, so it also re-arms the run flag (a converged solver had cleared it).
function resetVI() {
  app.vi = new ValueIterator(app.model);
  app.viHist = [];
  app.viDone = false;
  app.viRun = true;
  app.viFlash = new Map();
  app.prevVI = null;
}

function resetPI() {
  app.pi = new PolicyIterator(app.model, app.seed);
  app.piHist = [];
  app.piPhase = 'eval';
  app.piEvalCount = 0;
  app.piDone = false;
  app.piRun = true;
  app.piFlash = new Map();
}

function resetQ() {
  const p = app.params;
  app.q = new QLearner(app.model, {
    algo: app.algo, seed: app.seed, alpha: p.alpha, epsilon: p.epsilon, decay: p.decay, eps_min: p.eps_min,
    max_steps: p.max_steps,
  });
  app.qReturns = [];
  app.qAgree = [];
  app.qAcc = 0;
  app.qDone = false;
  app.qRun = true;
  app.qAgreeRun = 0;
  app.qFlags = null;
  app.qAgreeFrac = null;
  app.walk = { path: null, t: 0, ep: null, pending: null, trails: [], boom: null };
  app.orbCell = app.model.start;
}

// ── value iteration ──────────────────────────────────────────────────────

function stepVI(sweeps) {
  for (let k = 0; k < sweeps && !app.viDone; k++) {
    app.vi.sweep();
    app.viHist.push(app.vi.residual);
    if (app.viHist.length > HIST.viMax) app.viHist.splice(0, app.viHist.length - HIST.viMax);
    if (app.vi.residual < app.conv || app.vi.sweeps >= CAP.viSweeps) {
      app.viDone = app.vi.residual < app.conv;
      app.viRun = false;
    }
  }
}

// ── policy iteration ─────────────────────────────────────────────────────

// One frame's worth of policy iteration: evaluation sweeps until the residual is under the threshold, then one
// improvement step. Improvement is a single step, so the arrows flip in one frame and the chart marks it.
function stepPI(budget) {
  const pi = app.pi;
  let left = budget;
  while (left > 0 && !app.piDone) {
    if (app.piPhase === 'eval') {
      if (app.piEvalCount > 0 && pi.residual < app.conv) {
        app.piPhase = 'improve';
      } else if (app.piEvalCount >= CAP.piEval) {
        app.piPhase = 'improve';
      } else {
        pi.evalSweep();
        app.piEvalCount += 1;
        app.piHist.push({ r: pi.residual });
        left -= 1;
      }
      continue;
    }
    const before = pi.pi.slice();
    const changed = pi.improve();
    const now = performance.now();
    for (let s = 0; s < before.length; s++) if (before[s] !== pi.pi[s] && before[s] >= 0) app.piFlash.set(s, now);
    app.piHist.push({ mark: true, changed });
    app.piPhase = 'eval';
    app.piEvalCount = 0;
    pi.residual = Infinity;
    left -= 1;
    if (pi.stable) {
      app.piDone = true;
      app.piRun = false;
      banner('POLICY STABLE ✓', `${pi.rounds} improvement rounds, ${pi.evalSweeps} evaluation sweeps`, '#00ff88');
    }
  }
  if (app.piHist.length > HIST.piMax) app.piHist.splice(0, app.piHist.length - HIST.piMax);
}

// ── Q-learning and the walk ──────────────────────────────────────────────

function recordEpisodes(n, cap) {
  let last = null;
  let steps = 0;
  for (let i = 0; i < n && steps < cap; i++) {
    const ep = app.q.episode();
    steps += ep.steps;
    app.qReturns.push(ep.ret);
    last = ep;
  }
  if (app.qReturns.length > HIST.retMax) app.qReturns.splice(0, app.qReturns.length - HIST.retMax);
  return last;
}

function afterEpisodes(last) {
  if (!last) return;
  app.walk.pending = last; // the orb walks the newest episode; older ones it has not reached are skipped
  const ag = app.q.agreement(app.opt.pi, app.opt.Q);
  app.qFlags = ag.flags;
  app.qAgreeFrac = ag.frac;
  app.qAgree.push(ag.frac);
  if (app.qAgree.length > HIST.agreeMax) app.qAgree.splice(0, app.qAgree.length - HIST.agreeMax);
  // Converged means the match holds across a run of samples and at least 300 episodes, not one lucky check:
  // on a near-deterministic route, a few dozen episodes already get every cell right once.
  app.qAgreeRun = ag.frac >= 1 ? app.qAgreeRun + 1 : 0;
  if (!app.qDone && app.q.episodes >= 300 && app.qAgreeRun >= 20) {
    app.qDone = true;
    app.qRun = false;
    banner('POLICY CONVERGED', 'Q-learning matches the optimal policy on every reachable cell', '#ffe600');
  }
}

function stepQ(dt) {
  app.qAcc += (SPEEDS[app.speed].eps * dt) / 1000;
  const n = Math.min(CAP.episodesPerFrame, Math.floor(app.qAcc));
  app.qAcc -= n;
  if (n > 0) afterEpisodes(recordEpisodes(n, CAP.stepsPerFrame));
}

function fastForward(n) {
  afterEpisodes(recordEpisodes(n, Infinity));
}

function startWalk(ep) {
  app.walk.path = ep.path;
  app.walk.ep = ep;
  app.walk.t = 0;
}

function advanceWalk(dt, now) {
  const w = app.walk;
  if (!w.path) {
    if (!w.pending) return;
    startWalk(w.pending);
    w.pending = null;
  }
  w.t += dt / SPEEDS[app.speed].stepMs;
  if (w.t >= w.path.length - 1) finishWalk(now);
}

function finishWalk(now) {
  const w = app.walk;
  const end = w.path[w.path.length - 1];
  const m = app.model;
  if (w.ep.fell) {
    w.boom = { cell: end, t: now, kind: 'pit' };
    const at = clientOfCell(app.boards.q, end);
    if (at) {
      burst(at, { count: 70, colors: ['#ff00a0', '#ff3b3b', '#ffe600', '#00f5ff'] });
      shake($('panel-q'));
    }
    pop(at || $('panel-q'), '−' + Math.abs(m.params.pit).toFixed(0), '#ff00a0');
  } else if (w.ep.reachedTerminal) {
    w.boom = { cell: end, t: now, kind: 'reward' };
    const at = clientOfCell(app.boards.q, end);
    if (at) burst(at, { count: 40, colors: ['#00ff88', '#ffe600'] });
    pop(at || $('panel-q'), '+' + m.params.reward.toFixed(0), '#00ff88');
  }
  w.trails.push(w.path);
  if (w.trails.length > 6) w.trails.shift();
  app.orbCell = end;
  w.path = null;
  w.ep = null;
  w.t = 0;
  if (w.pending) {
    startWalk(w.pending);
    w.pending = null;
  }
}

function clientOfCell(canvas, cell) {
  const m = app.model;
  if (!canvas || !m || canvas.clientWidth === 0) return null;
  const r = canvas.getBoundingClientRect();
  const size = canvas.clientWidth / m.W;
  const [x, y] = cellCenter(m, size, cell);
  return [r.left + x, r.top + y];
}

// ── drawing: boards ──────────────────────────────────────────────────────

function drawArrow(ctx, cx, cy, a, len, color, lw) {
  const [dx, dy] = DIRS[a];
  const x1 = cx - dx * len * 0.5;
  const y1 = cy - dy * len * 0.5;
  const x2 = cx + dx * len * 0.5;
  const y2 = cy + dy * len * 0.5;
  ctx.strokeStyle = color;
  ctx.fillStyle = color;
  ctx.lineWidth = lw;
  ctx.beginPath();
  ctx.moveTo(x1, y1);
  ctx.lineTo(x2, y2);
  ctx.stroke();
  // Arrowhead: a triangle at the tip, along the direction of travel.
  const hs = len * 0.3;
  const px = -dy;
  const py = dx;
  ctx.beginPath();
  ctx.moveTo(x2, y2);
  ctx.lineTo(x2 - dx * hs + px * hs * 0.6, y2 - dy * hs + py * hs * 0.6);
  ctx.lineTo(x2 - dx * hs - px * hs * 0.6, y2 - dy * hs - py * hs * 0.6);
  ctx.closePath();
  ctx.fill();
}

// Draws one board. `o` may carry: values (heat per cell), scale, arrows (action per state), arrowColors,
// qTable with qScale (triangles), dim (flags, true = in scope), flash (Map cell -> time), trails, orb, boom,
// numbers (values to print), and nums (Q table to print in the triangles).
function drawBoard(canvas, m, o, now) {
  const W = canvas.clientWidth;
  if (!W) return;
  const cell = W / m.W;
  const H = cell * m.H;
  canvas.style.height = `${H}px`;
  const ctx = prepare(canvas, W, H);
  const pulse = REDUCED ? 0.5 : 0.5 + 0.5 * Math.sin(now / 260);

  // Cells: base colour, then the kind's own mark.
  for (let i = 0; i < m.n; i++) {
    const x = (i % m.W) * cell;
    const y = Math.floor(i / m.W) * cell;
    const k = m.kinds[i];
    if (k === '#') {
      ctx.fillStyle = '#070d1c';
      ctx.fillRect(x, y, cell, cell);
      ctx.strokeStyle = 'rgba(0,245,255,0.22)';
      ctx.lineWidth = 1;
      ctx.strokeRect(x + 2.5, y + 2.5, cell - 5, cell - 5);
      continue;
    }
    ctx.fillStyle = m.isState[i] && o.values ? heat(o.values[i], o.scale, 0.85) : '#0a1226';
    if (o.qTable) ctx.fillStyle = '#0a1226';
    ctx.fillRect(x, y, cell, cell);
    if (o.dim && m.isState[i] && !o.dim[i]) {
      ctx.fillStyle = 'rgba(2,6,16,0.62)';
      ctx.fillRect(x, y, cell, cell);
    }
    if (k === '+') {
      ctx.fillStyle = `rgba(0,255,136,${0.16 + 0.22 * pulse})`;
      ctx.fillRect(x, y, cell, cell);
      ctx.shadowColor = '#00ff88';
      ctx.shadowBlur = 6 + 10 * pulse;
      ctx.strokeStyle = '#00ff88';
      ctx.lineWidth = 2;
      ctx.strokeRect(x + 2, y + 2, cell - 4, cell - 4);
      ctx.shadowBlur = 0;
      if (cell >= 22) label(ctx, `+${m.cellPay[i].toFixed(1).replace(/\.0$/, '')}`, x + cell / 2, y + cell / 2, cell * 0.28, '#00ff88');
    } else if (k === '-') {
      ctx.fillStyle = 'rgba(255,59,59,0.28)';
      ctx.fillRect(x, y, cell, cell);
      ctx.strokeStyle = '#ff3b3b';
      ctx.lineWidth = 2;
      ctx.strokeRect(x + 2, y + 2, cell - 4, cell - 4);
      if (cell >= 22) label(ctx, '−', x + cell / 2, y + cell / 2, cell * 0.38, '#ff8080');
    } else if (k === 'C') {
      ctx.save();
      ctx.beginPath();
      ctx.rect(x, y, cell, cell);
      ctx.clip();
      ctx.strokeStyle = 'rgba(255,230,0,0.55)';
      ctx.lineWidth = 3;
      for (let d = -cell; d < cell * 2; d += cell * 0.5) {
        ctx.beginPath();
        ctx.moveTo(x + d, y + cell);
        ctx.lineTo(x + d + cell, y);
        ctx.stroke();
      }
      ctx.restore();
      ctx.strokeStyle = '#ffe600';
      ctx.lineWidth = 1.5;
      ctx.strokeRect(x + 1.5, y + 1.5, cell - 3, cell - 3);
    } else if (k === 'S') {
      ctx.strokeStyle = '#00f5ff';
      ctx.lineWidth = 2;
      ctx.strokeRect(x + 3, y + 3, cell - 6, cell - 6);
      if (cell >= 22) label(ctx, 'S', x + cell / 2, y + cell / 2, cell * 0.3, '#00f5ff');
    }
  }

  // Grid lines, faint.
  ctx.strokeStyle = 'rgba(0,245,255,0.07)';
  ctx.lineWidth = 1;
  for (let c = 1; c < m.W; c++) {
    ctx.beginPath();
    ctx.moveTo(c * cell, 0);
    ctx.lineTo(c * cell, H);
    ctx.stroke();
  }
  for (let r = 1; r < m.H; r++) {
    ctx.beginPath();
    ctx.moveTo(0, r * cell);
    ctx.lineTo(W, r * cell);
    ctx.stroke();
  }

  // Q triangles: one per direction, coloured by that Q value and sharing the cell's corners.
  if (o.qTable) {
    for (const s of m.states) {
      const x = (s % m.W) * cell;
      const y = Math.floor(s / m.W) * cell;
      const c = [x + cell / 2, y + cell / 2];
      const corners = [[x, y], [x + cell, y], [x + cell, y + cell], [x, y + cell]];
      const tri = [[0, 1], [1, 2], [2, 3], [3, 0]];
      for (let a = 0; a < 4; a++) {
        const [p, q2] = [corners[tri[a][0]], corners[tri[a][1]]];
        ctx.beginPath();
        ctx.moveTo(c[0], c[1]);
        ctx.lineTo(p[0], p[1]);
        ctx.lineTo(q2[0], q2[1]);
        ctx.closePath();
        ctx.fillStyle = heat(o.qTable[s][a], o.qScale, 0.9);
        ctx.fill();
      }
      ctx.strokeStyle = 'rgba(2,6,16,0.8)';
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(x, y);
      ctx.lineTo(x + cell, y + cell);
      ctx.moveTo(x + cell, y);
      ctx.lineTo(x, y + cell);
      ctx.stroke();
      if (o.nums && cell >= 30) {
        const sz = cell * 0.17;
        const spots = [[c[0], y + cell * 0.22], [x + cell * 0.8, c[1]], [c[0], y + cell * 0.8], [x + cell * 0.2, c[1]]];
        for (let a = 0; a < 4; a++) label(ctx, fmt(o.qTable[s][a], 1), spots[a][0], spots[a][1], sz, '#02060f');
      }
    }
  }

  // Values in the corner of each cell, when numbers are on.
  if (o.values && o.nums && cell >= 30) {
    for (const s of m.states) {
      const x = (s % m.W) * cell;
      const y = Math.floor(s / m.W) * cell;
      label(ctx, fmt(o.values[s]), x + cell * 0.5, y + cell * 0.2, cell * 0.17, '#e8fbff', 'left');
    }
  }

  // Arrows: the policy, one per state.
  if (o.arrows) {
    for (const s of m.states) {
      const a = o.arrows[s];
      if (a < 0) continue;
      const x = (s % m.W) * cell + cell / 2;
      const y = Math.floor(s / m.W) * cell + cell / 2;
      const color = o.arrowColors ? o.arrowColors[s] : '#f2fdff';
      drawArrow(ctx, x, y, a, cell * 0.52, color, Math.max(1.5, cell * 0.07));
    }
  }

  // Flashes where an action just changed.
  if (o.flash) {
    for (const [s, t] of o.flash) {
      const age = now - t;
      if (age > 900) continue;
      const x = (s % m.W) * cell + cell / 2;
      const y = Math.floor(s / m.W) * cell + cell / 2;
      ctx.strokeStyle = `rgba(255,230,0,${(1 - age / 900) * 0.9})`;
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.arc(x, y, cell * (0.35 + 0.25 * (age / 900)), 0, Math.PI * 2);
      ctx.stroke();
    }
  }

  // The agent's last few walks, fading, and the orb itself.
  if (o.trails) {
    o.trails.forEach((path, idx) => {
      const alpha = 0.12 + 0.5 * ((idx + 1) / o.trails.length);
      ctx.strokeStyle = `rgba(0,245,255,${alpha})`;
      ctx.lineWidth = Math.max(1.5, cell * 0.08);
      ctx.beginPath();
      path.forEach((s, j) => {
        const [px, py] = cellCenter(m, cell, s);
        if (j === 0) ctx.moveTo(px, py);
        else ctx.lineTo(px, py);
      });
      ctx.stroke();
    });
  }
  if (o.orb) {
    const [ox, oy] = o.orb;
    const r = cell * 0.26;
    const g = ctx.createRadialGradient(ox, oy, 0, ox, oy, r * 2.2);
    g.addColorStop(0, 'rgba(255,255,255,0.95)');
    g.addColorStop(0.35, 'rgba(0,245,255,0.85)');
    g.addColorStop(1, 'rgba(0,245,255,0)');
    ctx.fillStyle = g;
    ctx.beginPath();
    ctx.arc(ox, oy, r * 2.2, 0, Math.PI * 2);
    ctx.fill();
  }
  if (o.boom && now - o.boom.t < 650) {
    const age = (now - o.boom.t) / 650;
    const [bx, by] = cellCenter(m, cell, o.boom.cell);
    ctx.strokeStyle = o.boom.kind === 'pit' ? `rgba(255,0,160,${1 - age})` : `rgba(0,255,136,${1 - age})`;
    ctx.lineWidth = 3;
    ctx.beginPath();
    ctx.arc(bx, by, cell * (0.3 + 0.9 * age), 0, Math.PI * 2);
    ctx.stroke();
  }
}

function label(ctx, text, x, y, size, color, align = 'center') {
  ctx.fillStyle = color;
  ctx.font = `${size.toFixed(1)}px "JetBrains Mono", monospace`;
  ctx.textAlign = align;
  ctx.textBaseline = 'middle';
  ctx.fillText(text, x, y);
}

// ── drawing: charts ──────────────────────────────────────────────────────

// A line chart on a fixed frame. `series` is [{ys, color, width, alpha}] with ys in plotted units. Options:
// yMin/yMax (fixed range), hline (value and color), marks (x indices for vertical ticks), labels (top, bottom).
function drawChart(canvas, series, o = {}) {
  const W = canvas.clientWidth;
  const H = canvas.clientHeight;
  if (!W || !H) return;
  const ctx = prepare(canvas, W, H);
  const pad = 6;
  let n = 0;
  let lo = Infinity;
  let hi = -Infinity;
  for (const s of series) {
    n = Math.max(n, s.ys.length);
    for (const y of s.ys) {
      if (!Number.isFinite(y)) continue;
      if (y < lo) lo = y;
      if (y > hi) hi = y;
    }
  }
  if (o.yMin !== undefined) lo = o.yMin;
  if (o.yMax !== undefined) hi = o.yMax;
  if (!Number.isFinite(lo) || !Number.isFinite(hi)) { lo = 0; hi = 1; }
  if (hi - lo < 1e-9) { hi = lo + 1; }
  const X = (i) => pad + ((W - 2 * pad) * i) / Math.max(1, n - 1);
  const Y = (y) => H - pad - ((H - 2 * pad) * (y - lo)) / (hi - lo);

  if (o.hline) {
    ctx.strokeStyle = o.hline.color;
    ctx.setLineDash([4, 4]);
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(pad, Y(o.hline.y));
    ctx.lineTo(W - pad, Y(o.hline.y));
    ctx.stroke();
    ctx.setLineDash([]);
  }
  if (o.marks) {
    ctx.strokeStyle = 'rgba(255,230,0,0.5)';
    ctx.lineWidth = 1;
    for (const i of o.marks) {
      ctx.beginPath();
      ctx.moveTo(X(i), pad);
      ctx.lineTo(X(i), H - pad);
      ctx.stroke();
    }
  }
  for (const s of series) {
    if (s.ys.length < 2) continue;
    ctx.strokeStyle = s.color;
    ctx.globalAlpha = s.alpha ?? 1;
    ctx.lineWidth = s.width ?? 1.5;
    ctx.beginPath();
    let started = false;
    for (let i = 0; i < s.ys.length; i++) {
      const y = s.ys[i];
      if (!Number.isFinite(y)) continue;
      if (!started) { ctx.moveTo(X(i), Y(y)); started = true; } else ctx.lineTo(X(i), Y(y));
    }
    ctx.stroke();
    ctx.globalAlpha = 1;
  }
  if (o.labels) {
    ctx.fillStyle = 'rgba(74,122,155,0.95)';
    ctx.font = '10px "JetBrains Mono", monospace';
    ctx.textAlign = 'left';
    ctx.textBaseline = 'top';
    ctx.fillText(o.labels[0], pad + 2, pad);
    ctx.textBaseline = 'bottom';
    ctx.fillText(o.labels[1], pad + 2, H - pad);
  }
}

function movingAverage(xs, win) {
  const out = [];
  let sum = 0;
  for (let i = 0; i < xs.length; i++) {
    sum += xs[i];
    if (i >= win) sum -= xs[i - win];
    out.push(sum / Math.min(i + 1, win));
  }
  return out;
}

// ── frame: run the solvers, animate, redraw ─────────────────────────────

const boards = {};
let lastFrame = performance.now();
let lastText = 0;

function frame(now) {
  const dt = Math.min(100, now - lastFrame);
  lastFrame = now;
  if (app.dirty) rebuild();
  const sp = SPEEDS[app.speed];
  if (app.model) {
    if (app.viRun && !app.viDone) stepVI(sp.vi);
    if (app.piRun && !app.piDone) stepPI(sp.pi);
    if (app.qRun && !app.qDone) stepQ(dt);
    advanceWalk(dt, now);
    draw(now);
    if (now - lastText > 110) {
      updateText();
      lastText = now;
    }
  }
  requestAnimationFrame(frame);
}

function draw(now) {
  const m = app.model;
  // The world editor shows the layout alone.
  drawBoard(boards.world, m, {}, now);

  // Value iteration: the values as heat, and the greedy policy of those values.
  const vi = app.vi;
  const viArrows = greedyPolicy(m, vi.V);
  const viFlash = new Map();
  if (app.prevVI) {
    for (const s of m.states) if (app.prevVI[s] !== viArrows[s] && app.prevVI[s] >= 0) viFlash.set(s, now);
  }
  app.prevVI = viArrows;
  drawBoard(boards.vi, m, {
    values: vi.V, scale: maxAbs(vi.V, m.states), arrows: viArrows, flash: viFlash, numbers: app.numbers, nums: app.numbers,
  }, now);
  drawChart(boards.viChart, [{ ys: app.viHist.map((r) => Math.log10(Math.max(r, 1e-12))), color: '#00f5ff', width: 1.6 }], {
    hline: { y: Math.log10(app.conv), color: 'rgba(0,255,136,0.7)' }, labels: ['residual 1e+0', `ε ${fmtExp(app.conv)}`],
    yMin: -8, yMax: 1,
  });

  // Policy iteration: the values of the current policy, and the policy itself. The chart plots evaluation
  // residuals; improvement steps are NaN points (skipped) and vertical ticks in the same index space.
  const pi = app.pi;
  drawBoard(boards.pi, m, {
    values: pi.V, scale: maxAbs(pi.V, m.states), arrows: pi.pi, flash: app.piFlash, numbers: app.numbers,
    nums: app.numbers,
  }, now);
  const piYs = [];
  const marks = [];
  app.piHist.forEach((h, i) => {
    if (h.mark) marks.push(i);
    piYs.push(h.mark ? NaN : Math.log10(Math.max(h.r, 1e-12)));
  });
  drawChart(boards.piChart, [{ ys: piYs, color: '#00ff88', width: 1.6 }], {
    marks, labels: ['residual 1e+0', `ε ${fmtExp(app.conv)}`], yMin: -8, yMax: 1,
    hline: { y: Math.log10(app.conv), color: 'rgba(0,255,136,0.5)' },
  });

  // Q-learning: Q per direction as triangles, the orb, and the learned policy when comparing.
  const q = app.q;
  const qVals = [];
  for (const s of m.states) for (const v of q.Q[s]) qVals.push(v);
  let qScale = 1;
  for (const v of qVals) qScale = Math.max(qScale, Math.abs(v));
  // The orb sits between the two cells of the walk step it is on, in pixels.
  const w = app.walk;
  const size = boards.q.clientWidth / m.W;
  let orbPx = cellCenter(m, size, app.orbCell >= 0 ? app.orbCell : m.start);
  if (w.path) {
    const idx = Math.min(w.path.length - 2, Math.floor(w.t));
    const frac = Math.min(1, w.t - idx);
    const [ax, ay] = cellCenter(m, size, w.path[idx]);
    const [bx, by] = cellCenter(m, size, w.path[idx + 1]);
    orbPx = [ax + (bx - ax) * frac, ay + (by - ay) * frac];
  }
  const qOpts = {
    qTable: q.Q, qScale, nums: app.numbers, trails: w.trails, orb: orbPx,
    boom: w.boom && now - w.boom.t < 650 ? w.boom : null,
  };
  if (app.compare && app.qFlags) {
    const pol = q.policy();
    const colors = pol.map((_, s) => (app.qFlags[s] ? '#00ff88' : '#ff00a0'));
    Object.assign(qOpts, { arrows: pol, arrowColors: colors, dim: reachableFlags(m) });
  }
  drawBoard(boards.q, m, qOpts, now);
  drawChart(boards.qReturns, [
    { ys: app.qReturns.slice(-1500), color: 'rgba(0,245,255,0.25)', width: 1 },
    { ys: movingAverage(app.qReturns, 50).slice(-1500), color: '#00f5ff', width: 1.8 },
  ], { labels: ['return', ''] });
  drawChart(boards.qAgree, [{ ys: app.qAgree, color: '#00ff88', width: 1.6 }], {
    yMin: 0, yMax: 1, hline: { y: 1, color: 'rgba(255,230,0,0.5)' }, labels: ['100%', '0%'],
  });
}

// Cells the optimal policy reaches: the same scope the agreement score counts. Cached per model.
let reachCache = null;
function reachableFlags(m) {
  if (reachCache && reachCache.model === m) return reachCache.flags;
  const flags = new Array(m.n).fill(false);
  for (const s of reachable(m, app.opt.pi)) flags[s] = true;
  reachCache = { model: m, flags };
  return flags;
}

function updateText() {
  const m = app.model;
  const vi = app.vi;
  const pi = app.pi;
  const q = app.q;

  $('vi-status').innerHTML = app.viDone
    ? `sweep <b>${vi.sweeps}</b> · residual <b>${fmtExp(vi.residual)}</b> · converged below ${fmtExp(app.conv)}`
    : `sweep <b>${vi.sweeps}</b> · residual ${fmtExp(vi.residual)} · target ${fmtExp(app.conv)}`;
  const viBadge = $('vi-badge');
  viBadge.textContent = app.viDone ? 'CONVERGED ✓' : app.viRun ? 'RUNNING' : 'PAUSED';
  viBadge.className = `mdp-badge ${app.viDone ? 'is-done' : app.viRun ? '' : 'is-paused'}`;
  $('vi-run').querySelector('.btn-txt').textContent = app.viRun ? '❚❚ PAUSE' : '▶ RUN';

  const changed = pi.lastChanged;
  if (app.piDone) {
    $('pi-status').innerHTML = `stable after <b>${pi.rounds}</b> rounds and <b>${pi.evalSweeps}</b> evaluation sweeps`;
  } else if (app.piPhase === 'eval') {
    $('pi-status').innerHTML = `round <b>${pi.rounds + 1}</b> · evaluation sweep ${app.piEvalCount} · residual ${fmtExp(pi.residual)} · changed last round ${changed < 0 ? '—' : changed}`;
  } else {
    $('pi-status').innerHTML = `round <b>${pi.rounds + 1}</b> · improving: every cell takes its best action`;
  }
  const piBadge = $('pi-badge');
  piBadge.textContent = app.piDone ? 'POLICY STABLE ✓' : app.piPhase === 'eval' ? 'EVALUATING' : 'IMPROVING';
  piBadge.className = `mdp-badge ${app.piDone ? 'is-done' : app.piRun ? '' : 'is-paused'}`;
  $('pi-run').querySelector('.btn-txt').textContent = app.piRun ? '❚❚ PAUSE' : '▶ RUN';

  const last = q.last ? q.last.ret : null;
  const agree = app.qAgreeFrac;
  $('q-status').innerHTML = q.episodes === 0
    ? 'no episodes yet'
    : `episode <b>${q.episodes}</b> · return ${last === null ? '—' : fmt(last)} · ε ${q.eps.toFixed(3)} · optimal on <b>${agree === null ? '—' : pct(agree)}</b> of reachable cells`;
  const qBadge = $('q-badge');
  qBadge.textContent = app.qDone ? 'POLICY CONVERGED ✓' : app.qRun ? 'WALKING' : 'PAUSED';
  qBadge.className = `mdp-badge ${app.qDone ? 'is-done' : app.qRun ? '' : 'is-paused'}`;
  $('q-run').querySelector('.btn-txt').textContent = app.qRun ? '❚❚ PAUSE' : '▶ RUN';

  $('chip-vi').textContent = String(vi.sweeps);
  $('chip-pi').textContent = String(pi.rounds);
  $('chip-q').textContent = String(q.episodes);
  $('chip-agree').textContent = agree === null ? '—' : pct(agree);
  void m;
}

// ── wiring ───────────────────────────────────────────────────────────────

function cellAt(evt) {
  const canvas = boards.world;
  const r = canvas.getBoundingClientRect();
  const W = app.grid[0].length;
  const H = app.grid.length;
  const x = Math.floor(((evt.clientX - r.left) / r.width) * W);
  const y = Math.floor(((evt.clientY - r.top) / r.height) * H);
  if (x < 0 || y < 0 || x >= W || y >= H) return null;
  return [x, y];
}

// Paints one cell. The start moves rather than duplicates, and the start cell cannot be overwritten by another tool.
function paintAt(x, y) {
  const cur = app.grid[y][x];
  // Any edit makes the world a custom one, so the header and the preset highlight stop naming a preset.
  if (app.key !== 'custom') {
    app.key = 'custom';
    document.querySelectorAll('[data-preset]').forEach((b) => b.setAttribute('aria-pressed', 'false'));
    $('preset-blurb').textContent = 'A hand-painted world. Presets reset it to a lesson.';
  }
  if (app.tool === 'start') {
    if (cur === 'S') return false;
    for (const row of app.grid) for (let c = 0; c < row.length; c++) if (row[c] === 'S') row[c] = '.';
    app.grid[y][x] = 'S';
    app.dirty = true;
    return true;
  }
  if (cur === 'S') return false;
  const next = { wall: '#', reward: '+', pit: '-', cliff: 'C', erase: '.' }[app.tool];
  if (next === cur) return false;
  app.grid[y][x] = next;
  app.dirty = true;
  return true;
}

function setupWorld() {
  const canvas = boards.world;
  let painting = false;
  let last = null;
  canvas.addEventListener('pointerdown', (e) => {
    const c = cellAt(e);
    if (!c) return;
    painting = true;
    canvas.setPointerCapture(e.pointerId);
    last = c.join(',');
    paintAt(c[0], c[1]);
  });
  canvas.addEventListener('pointermove', (e) => {
    if (!painting) return;
    const c = cellAt(e);
    if (!c || c.join(',') === last) return;
    last = c.join(',');
    paintAt(c[0], c[1]);
  });
  const end = () => { painting = false; last = null; };
  canvas.addEventListener('pointerup', end);
  canvas.addEventListener('pointercancel', end);
}

function wire() {
  document.querySelectorAll('[data-preset]').forEach((b) => b.addEventListener('click', () => loadPreset(b.dataset.preset)));
  $('btn-clear').addEventListener('click', clearWorld);

  const setTool = (tool) => {
    app.tool = tool;
    document.querySelectorAll('[data-tool]').forEach((b) => b.setAttribute('aria-pressed', String(b.dataset.tool === tool)));
  };
  document.querySelectorAll('[data-tool]').forEach((b) => b.addEventListener('click', () => setTool(b.dataset.tool)));
  setTool(app.tool);

  for (const [id, key, sign, dec, kind] of SLIDERS) {
    $(id).addEventListener('input', (e) => {
      app.params[key] = sign * Number(e.target.value);
      $(`${id}-val`).textContent = fmt(sign * Number(e.target.value), dec);
      if (kind === 'model') app.dirty = true;
      else resetQ();
    });
  }

  $('seed').addEventListener('change', (e) => {
    const v = Math.max(1, Math.min(99999, Math.round(Number(e.target.value)) || 1));
    app.seed = v;
    $('seed').value = String(v);
    $('seed-val').textContent = String(v);
    resetPI();
    resetQ();
  });
  $('btn-seed').addEventListener('click', () => {
    app.seed = 1 + Math.floor(Math.random() * 99998);
    $('seed').value = String(app.seed);
    $('seed-val').textContent = String(app.seed);
    resetPI();
    resetQ();
  });
  $('conv').addEventListener('change', (e) => {
    app.conv = Number(e.target.value);
    resetVI();
    resetPI();
  });
  $('numbers').addEventListener('change', (e) => { app.numbers = e.target.checked; });
  $('q-compare').addEventListener('change', (e) => { app.compare = e.target.checked; });
  $('q-algo').addEventListener('change', (e) => {
    app.algo = e.target.value;
    resetQ();
  });

  $('vi-run').addEventListener('click', () => {
    if (app.viDone) resetVI();
    app.viRun = !app.viRun;
  });
  $('vi-step').addEventListener('click', () => {
    if (!app.viDone) stepVI(1);
  });
  $('vi-reset').addEventListener('click', () => { resetVI(); });

  $('pi-run').addEventListener('click', () => {
    if (app.piDone) resetPI();
    app.piRun = !app.piRun;
  });
  $('pi-step').addEventListener('click', () => {
    if (!app.piDone) stepPI(1);
  });
  $('pi-reset').addEventListener('click', () => { resetPI(); });

  $('q-run').addEventListener('click', () => {
    if (app.qDone) resetQ();
    app.qRun = !app.qRun;
  });
  $('q-ff').addEventListener('click', () => { fastForward(1000); });
  $('q-reset').addEventListener('click', () => { resetQ(); });

  document.querySelectorAll('[data-speed]').forEach((b) => b.addEventListener('click', () => {
    app.speed = Number(b.dataset.speed);
    document.querySelectorAll('[data-speed]').forEach((x) => x.setAttribute('aria-pressed', String(x === b)));
    $('speed-note').textContent = `Planners: ${SPEEDS[app.speed].vi} to ${SPEEDS[app.speed].pi} sweeps per frame. Agent: ${SPEEDS[app.speed].eps} episodes per second; the orb walks one episode at a time.`;
  }));
  document.querySelectorAll('[data-speed]').forEach((x) => x.setAttribute('aria-pressed', String(Number(x.dataset.speed) === app.speed)));

  document.querySelectorAll('.mdp-views [data-view]').forEach((b) => b.addEventListener('click', () => {
    $('trio').dataset.view = b.dataset.view;
    document.querySelectorAll('.mdp-views [data-view]').forEach((x) => x.setAttribute('aria-pressed', String(x === b)));
  }));
}

function init() {
  boards.world = $('world');
  boards.vi = $('vi-board');
  boards.pi = $('pi-board');
  boards.q = $('q-board');
  boards.viChart = $('vi-chart');
  boards.piChart = $('pi-chart');
  boards.qReturns = $('q-returns');
  boards.qAgree = $('q-agree');
  app.boards = boards;
  wire();
  setupWorld();
  loadPreset(app.key);
  rebuild();
  requestAnimationFrame((t) => { lastFrame = t; requestAnimationFrame(frame); });
}

init();
