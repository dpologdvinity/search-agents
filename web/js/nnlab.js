// Neural network lab page: a small multilayer perceptron that learns to separate pink and cyan points, live.
//
// The maths is in nnlab-core.js (no DOM); this file draws and wires the controls. Each frame it runs as many
// epochs as the speed slider asks for (capped, so the page stays responsive), then redraws:
//   THE PLANE    the network's probability over a grid of the plane, as a glowing heat map, with the points
//   THE NETWORK  neurons lit by their activation for one data point, edges coloured by weight sign
//   DETECTOR     a hovered (or pinned) hidden neuron's own heat map over the plane
//   LOSS CURVE   training loss (solid) and held-out loss (dashed) on a log scale
// Architecture and feature changes rebuild the network; data, noise, split and training settings take effect
// on the running network.

import {
  Optimizer, Rng, buildSplit, evaluate, featurize, forward, generatePoints, initNet, neuronActivations,
  predict, trainEpoch, DEFAULTS, FEATURES, PRESETS,
} from './nnlab-core.js';
import { banner, burst } from './fx.js';

const $ = (id) => document.getElementById(id);
const D = 1.25;               // the plane runs from -D to D on both axes
const G = 56;                 // heat map resolution (G x G cells, stretched over the canvas)
const POINTS_PER_PRESET = 200;
const MAX_EPOCHS_PER_FRAME = 12;
const PINK = [255, 0, 160];
const CYAN = [0, 245, 255];
const REDUCED = matchMedia('(prefers-reduced-motion: reduce)').matches;
const FEATURE_NAMES = { x: 'x', y: 'y', x2: 'x²', y2: 'y²', xy: 'xy', sinx: 'sin πx', siny: 'sin πy' };

const S = {
  points: [],               // {x, y, label, jx, jy, u}: base position, jitter, label (1 pink, 0 cyan), split draw
  preset: 'circles',        // a preset name, or 'custom' once points are edited, or 'none'
  seed: DEFAULTS.seed,
  drawClass: 1,
  brush: 'add',
  features: [...DEFAULTS.features],
  layers: DEFAULTS.hidden.length,
  neurons: DEFAULTS.hidden[0],
  act: DEFAULTS.act,
  lr: DEFAULTS.lr,
  batch: DEFAULTS.batch,
  l2: DEFAULTS.l2,
  optimizer: DEFAULTS.optimizer,
  noise: DEFAULTS.noise,
  testFrac: DEFAULTS.testFrac,
  speed: 20,
  playing: true,
  split: null,
  splitDirty: true,
  net: null,
  opt: null,
  shuffle: null,
  addRng: null,
  epochs: 0,
  metrics: { trL: NaN, trA: NaN, teL: NaN, teA: NaN },
  hist: { trL: [], teL: [], trA: [], teA: [] },
  milestone: false,
  budget: 0,
  pulse: 0,
  sample: 0,
  frame: 0,
  painting: false,          // pointer is down on the plane
  lastPaint: null,          // last point laid down while painting, to space the drops
  hover: null,              // [x, y] under the pointer on the plane, or null
  neuron: null,             // {c, j}: the neuron shown in the detector (column c, index j)
  pinned: null,
  layout: null,             // node positions from the last network draw, for hit tests
  gridKey: '',
  gridX: null,              // features of each heat-map cell, rebuilt when the features change
  uiKey: '',
};

// ── Canvas helpers ────────────────────────────────────────────────────────

/** Size a canvas to its CSS box at the device pixel ratio. Returns null while the canvas is hidden. */
function setupCanvas(cv) {
  const w = cv.clientWidth, h = cv.clientHeight;
  if (!w || !h) return null;
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const W = Math.round(w * dpr), H = Math.round(h * dpr);
  if (cv.width !== W || cv.height !== H) { cv.width = W; cv.height = H; }
  const ctx = cv.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return { ctx, w, h };
}

const toScreen = (x, y, w, h) => [((x + D) / (2 * D)) * w, ((D - y) / (2 * D)) * h];
const toPlane = (sx, sy, w, h) => [(sx / w) * 2 * D - D, D - (sy / h) * 2 * D];
const rgb = (c, a = 1) => `rgba(${c[0] | 0},${c[1] | 0},${c[2] | 0},${a})`;
const mix = (t) => [CYAN[0] + (PINK[0] - CYAN[0]) * t, CYAN[1] + (PINK[1] - CYAN[1]) * t, CYAN[2] + (PINK[2] - CYAN[2]) * t];
const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));
const pct = (v) => (Number.isFinite(v) ? `${(100 * v).toFixed(1)}%` : '—');

/** Centre of cell (gx, gy) of the heat-map grid. Row 0 is the top, so y decreases going down. */
const cellCoord = (gx, gy) => [((gx + 0.5) / G) * 2 * D - D, D - ((gy + 0.5) / G) * 2 * D];

/** Effective position of a point: the base position moved by the noise slider. */
const effective = (p) => [p.x + S.noise * p.jx, p.y + S.noise * p.jy];

// ── Data and network ──────────────────────────────────────────────────────

function gridFeatures() {
  const key = S.features.join(',');
  if (key !== S.gridKey) {
    const F = S.features.length;
    const arr = new Float64Array(G * G * F);
    for (let gy = 0; gy < G; gy++) {
      for (let gx = 0; gx < G; gx++) {
        const [x, y] = cellCoord(gx, gy);
        const row = featurize(x, y, S.features);
        const k = (gy * G + gx) * F;
        for (let i = 0; i < F; i++) arr[k + i] = row[i];
      }
    }
    S.gridX = arr;
    S.gridKey = key;
  }
  return S.gridX;
}

function rebuildSplit() {
  S.split = buildSplit(S.points, S.features, S.noise, S.testFrac);
  S.splitDirty = false;
}

/** Probability of pink for every point, at its effective position. */
function pointProbs() {
  const n = S.points.length;
  if (n === 0) return new Float64Array(0);
  const F = S.features.length;
  const X = new Float64Array(n * F);
  S.points.forEach((p, i) => {
    const [x, y] = effective(p);
    const row = featurize(x, y, S.features);
    for (let k = 0; k < F; k++) X[i * F + k] = row[k];
  });
  return predict(S.net, X, n);
}

function recordMetrics() {
  if (S.splitDirty) rebuildSplit();
  const sp = S.split;
  const tr = evaluate(S.net, sp.Xtr, sp.ytr, sp.ntr);
  const te = evaluate(S.net, sp.Xte, sp.yte, sp.nte);
  S.metrics = { trL: tr.loss, trA: tr.acc, teL: te.loss, teA: te.acc };
  const H = S.hist;
  H.trL.push(tr.loss); H.teL.push(te.loss); H.trA.push(tr.acc); H.teA.push(te.acc);
  if (H.trL.length > 6000) for (const k of Object.keys(H)) H[k].splice(0, 1000);
  // The milestone fires once per network: the first time the held-out (or training) accuracy reaches 95%.
  const acc = Number.isFinite(te.acc) ? te.acc : tr.acc;
  if (!S.milestone && S.epochs >= 20 && acc >= 0.95) {
    S.milestone = true;
    const what = Number.isFinite(te.acc) ? 'test' : 'training';
    banner('BOUNDARY LOCKED', `${(100 * acc).toFixed(0)}% ${what} accuracy after ${S.epochs} epochs`, '#00ff88');
    const cv = $('plane').getBoundingClientRect();
    burst([cv.left + cv.width / 2, cv.top + cv.height / 2], { count: 90 });
  }
}

/** Build a fresh network from the architecture and the seed, and restart the curves. */
function buildNet() {
  if (S.splitDirty) rebuildSplit();
  const sizes = [S.features.length, ...Array(S.layers).fill(S.neurons), 1];
  S.net = initNet(sizes, S.act, new Rng(S.seed + 1));
  S.opt = new Optimizer(S.optimizer, S.net, S.lr);
  S.shuffle = new Rng(S.seed + 2);
  S.epochs = 0;
  S.hist = { trL: [], teL: [], trA: [], teA: [] };
  S.milestone = false;
  S.pinned = null;
  S.neuron = null;
  recordMetrics();
}

/** One epoch over the training points, then the metrics. */
function runEpoch() {
  if (S.splitDirty) rebuildSplit();
  const sp = S.split;
  if (sp.ntr === 0) return;
  S.opt.lr = S.lr;
  trainEpoch(S.net, S.opt, sp.Xtr, sp.ytr, sp.ntr, S.shuffle, S.batch, S.l2);
  S.epochs++;
  S.pulse = 1;
  recordMetrics();
}

function markDataChanged() {
  S.splitDirty = true;
  S.preset = 'custom';
}

function loadPreset(name) {
  S.preset = name;
  S.points = generatePoints(name, POINTS_PER_PRESET, S.seed);
  S.addRng = new Rng(S.seed + 99);
  S.splitDirty = true;
  buildNet();
}

function addPoint(x, y) {
  const r = S.addRng;
  const jx = r.normal(), jy = r.normal(), u = r.random();
  // Store the base position so that the point lands where clicked at the current noise level.
  S.points.push({ x: x - S.noise * jx, y: y - S.noise * jy, label: S.drawClass, jx, jy, u });
  markDataChanged();
}

function erasePoint(x, y) {
  let best = -1, bestD = 0.14;
  S.points.forEach((p, i) => {
    const [px, py] = effective(p);
    const d = Math.hypot(px - x, py - y);
    if (d < bestD) { bestD = d; best = i; }
  });
  if (best >= 0) { S.points.splice(best, 1); markDataChanged(); }
}

// ── Plane ─────────────────────────────────────────────────────────────────

const heatCv = document.createElement('canvas');
heatCv.width = G; heatCv.height = G;
const heatCtx = heatCv.getContext('2d');
const heatImg = heatCtx.createImageData(G, G);

/**
 * Paint the probability grid. The colour runs from cyan (0) to pink (1). Confidence fades the colour
 * toward black near the boundary, and a white glow marks the 0.5 contour.
 */
function paintHeat(P) {
  const d = heatImg.data;
  for (let k = 0; k < G * G; k++) {
    const p = P[k];
    const conf = Math.abs(2 * p - 1);                 // 0 on the boundary, 1 far from it
    const edge = Math.exp(-(((p - 0.5) / 0.06) ** 2)); // bright along the boundary
    const base = mix(p);
    const k4 = 4 * k;
    const f = 0.14 + 0.5 * conf;
    d[k4] = base[0] * f + 255 * edge * 0.85;
    d[k4 + 1] = base[1] * f + 255 * edge * 0.85;
    d[k4 + 2] = base[2] * f + 255 * edge * 0.85;
    d[k4 + 3] = 255;
  }
  heatCtx.putImageData(heatImg, 0, 0);
}

function nearestPoint(x, y, radius) {
  let best = -1, bestD = radius;
  S.points.forEach((p, i) => {
    const [px, py] = effective(p);
    const d = Math.hypot(px - x, py - y);
    if (d < bestD) { bestD = d; best = i; }
  });
  return best;
}

function drawPlane() {
  const c = setupCanvas($('plane'));
  if (!c) return;
  const { ctx, w, h } = c;
  paintHeat(predict(S.net, gridFeatures(), G * G));
  ctx.imageSmoothingEnabled = true;
  ctx.drawImage(heatCv, 0, 0, w, h);

  // Axes through the origin.
  ctx.strokeStyle = 'rgba(255,255,255,0.12)';
  ctx.lineWidth = 1;
  ctx.beginPath();
  const [ox, oy] = toScreen(0, 0, w, h);
  ctx.moveTo(ox, 0); ctx.lineTo(ox, h); ctx.moveTo(0, oy); ctx.lineTo(w, oy);
  ctx.stroke();

  const probs = pointProbs();
  const hoverIdx = S.hover ? nearestPoint(S.hover[0], S.hover[1], 0.1) : -1;
  S.points.forEach((p, i) => {
    const [x, y] = effective(p);
    const [sx, sy] = toScreen(x, y, w, h);
    const col = p.label ? PINK : CYAN;
    const isTest = S.testFrac > 0 && p.u < S.testFrac;
    const correct = (probs[i] >= 0.5) === (p.label === 1);
    ctx.shadowColor = rgb(col);
    ctx.shadowBlur = 12;
    if (isTest) {
      // Held-out points are hollow rings, so they can be told apart from training points.
      ctx.strokeStyle = rgb(col);
      ctx.lineWidth = 2;
      ctx.beginPath(); ctx.arc(sx, sy, 5, 0, Math.PI * 2); ctx.stroke();
    } else {
      ctx.fillStyle = rgb(col);
      ctx.beginPath(); ctx.arc(sx, sy, 4.5, 0, Math.PI * 2); ctx.fill();
    }
    ctx.shadowBlur = 0;
    if (!correct) {
      // A white dashed ring marks a point on the wrong side of the boundary.
      ctx.strokeStyle = 'rgba(255,255,255,0.9)';
      ctx.lineWidth = 1.2;
      ctx.setLineDash([3, 3]);
      ctx.beginPath(); ctx.arc(sx, sy, 9, 0, Math.PI * 2); ctx.stroke();
      ctx.setLineDash([]);
    }
    if (i === hoverIdx) {
      ctx.strokeStyle = '#ffe600';
      ctx.lineWidth = 2;
      ctx.beginPath(); ctx.arc(sx, sy, 13, 0, Math.PI * 2); ctx.stroke();
    }
  });

  // Neon frame, brighter on each epoch.
  const glow = REDUCED ? 0 : S.pulse;
  ctx.strokeStyle = rgb(CYAN, 0.35 + 0.6 * glow);
  ctx.shadowColor = rgb(CYAN);
  ctx.shadowBlur = 8 + 18 * glow;
  ctx.lineWidth = 2;
  ctx.strokeRect(1, 1, w - 2, h - 2);
  ctx.shadowBlur = 0;

  // Readout at the pointer: the coordinates and the network's answer there.
  if (S.hover) {
    const [hx, hy] = S.hover;
    const row = featurize(hx, hy, S.features);
    const { A } = forward(S.net, Float64Array.from(row), 1);
    const p = A[A.length - 1][0];
    ctx.font = '11px "JetBrains Mono", monospace';
    ctx.fillStyle = 'rgba(2,6,16,0.75)';
    ctx.fillRect(8, 8, 226, 22);
    ctx.fillStyle = p >= 0.5 ? rgb(PINK) : rgb(CYAN);
    ctx.fillText(`x ${hx.toFixed(2)}  y ${hy.toFixed(2)}  p(pink) ${p.toFixed(2)}`, 14, 23);
  }
}

// ── Network diagram ───────────────────────────────────────────────────────

/** Diverging value in [-1, 1] for an activation: negative cyan, positive pink. */
function diverging(act, a) {
  if (act === 'sigmoid') return 2 * a - 1;
  if (act === 'relu') return a / (1 + a);
  return a; // tanh is already in (-1, 1)
}

/** Activations of every column for one input: A[0] is the input, A[last] the probability. */
function columnActivations(x) {
  return forward(S.net, Float64Array.from(x), 1).A;
}

function displayInput() {
  if (S.hover) return featurize(S.hover[0], S.hover[1], S.features);
  if (S.points.length) {
    const p = S.points[S.sample % S.points.length];
    const [x, y] = effective(p);
    return featurize(x, y, S.features);
  }
  return featurize(0, 0, S.features);
}

function drawNet() {
  const c = setupCanvas($('net'));
  if (!c) return;
  const { ctx, w, h } = c;
  const sizes = S.net.sizes;
  const L = sizes.length;
  const padL = 62, padR = 46, padY = 30;
  const colX = (ci) => padL + (ci * (w - padL - padR)) / (L - 1);
  const maxN = Math.max(...sizes);
  const sp = maxN > 1 ? Math.min(44, (h - 2 * padY) / (maxN - 1)) : 0;
  const nodeY = (ci, k) => h / 2 + (k - (sizes[ci] - 1) / 2) * sp;
  const pos = sizes.map((n, ci) => Array.from({ length: n }, (_, k) => [colX(ci), nodeY(ci, k)]));
  S.layout = pos;

  ctx.clearRect(0, 0, w, h);
  const acts = columnActivations(displayInput());

  // Edges: pink for a positive weight, cyan for a negative one. Thickness and opacity grow with |w|, up to 1.
  for (let li = 0; li < L - 1; li++) {
    const din = sizes[li], dout = sizes[li + 1], W = S.net.W[li];
    for (let j = 0; j < dout; j++) {
      const focus = S.neuron && S.neuron.c === li + 1 && S.neuron.j === j;
      for (let i = 0; i < din; i++) {
        const wv = W[j * din + i];
        const m = Math.min(1, Math.abs(wv));
        const [x0, y0] = pos[li][i];
        const [x1, y1] = pos[li + 1][j];
        ctx.strokeStyle = rgb(wv >= 0 ? PINK : CYAN, focus ? 0.95 : 0.1 + 0.7 * m);
        ctx.lineWidth = (focus ? 1.2 : 0.4) + 2 * m;
        ctx.beginPath(); ctx.moveTo(x0, y0); ctx.lineTo(x1, y1); ctx.stroke();
      }
    }
  }

  // Neurons: colour from the diverging activation, brightness from its size, glow on the layer the pulse is at.
  const scanPos = REDUCED ? -10 : ((S.epochs * 0.3) % Math.max(1, L - 1));
  const kind = S.act;
  for (let ci = 0; ci < L; ci++) {
    for (let k = 0; k < sizes[ci]; k++) {
      const [x, y] = pos[ci][k];
      let d;
      if (ci === 0) d = clamp(acts[0][k], -1, 1);
      else if (ci === L - 1) d = 2 * acts[ci][0] - 1;
      else d = diverging(kind, acts[ci][k]);
      const t = (d + 1) / 2;
      const col = mix(t);
      const strength = 0.35 + 0.65 * Math.abs(d);
      const scan = Math.exp(-(((ci - scanPos) ** 2) * 2.5));
      ctx.shadowColor = rgb(col);
      ctx.shadowBlur = 6 + 14 * strength + 10 * scan;
      ctx.fillStyle = rgb(col, strength);
      ctx.beginPath(); ctx.arc(x, y, 9, 0, Math.PI * 2); ctx.fill();
      ctx.shadowBlur = 0;
      ctx.strokeStyle = rgb(col, 0.9);
      ctx.lineWidth = 1.5;
      ctx.beginPath(); ctx.arc(x, y, 9, 0, Math.PI * 2); ctx.stroke();
      if (S.neuron && S.neuron.c === ci && S.neuron.j === k) {
        ctx.strokeStyle = '#ffe600';
        ctx.lineWidth = 2;
        ctx.beginPath(); ctx.arc(x, y, 14, 0, Math.PI * 2); ctx.stroke();
      }
    }
  }

  // Column captions and the input feature names.
  ctx.font = '10px "JetBrains Mono", monospace';
  ctx.textAlign = 'center';
  ctx.fillStyle = 'rgba(255,255,255,0.55)';
  for (let ci = 0; ci < L; ci++) {
    const label = ci === 0 ? 'IN' : ci === L - 1 ? 'OUT' : `H${ci}`;
    ctx.fillText(label, colX(ci), 14);
  }
  ctx.textAlign = 'right';
  ctx.fillStyle = 'rgba(0,245,255,0.85)';
  S.features.forEach((f, k) => ctx.fillText(FEATURE_NAMES[f], pos[0][k][0] - 14, pos[0][k][1] + 3));
  // The output probability sits above the output node, so it stays inside the canvas.
  ctx.textAlign = 'center';
  ctx.fillStyle = 'rgba(255,0,160,0.9)';
  ctx.fillText(`p=${acts[L - 1][0].toFixed(2)}`, pos[L - 1][0][0], pos[L - 1][0][1] - 16);
  ctx.textAlign = 'start';
}

/** The hidden neuron under a pointer event, or null. Input and output columns are not targets. */
function neuronAt(e) {
  const cv = $('net');
  const r = cv.getBoundingClientRect();
  const sx = e.clientX - r.left, sy = e.clientY - r.top;
  if (!S.layout) return null;
  let best = null, bestD = 16;
  for (let ci = 1; ci < S.layout.length - 1; ci++) {
    S.layout[ci].forEach(([x, y], j) => {
      const d = Math.hypot(x - sx, y - sy);
      if (d < bestD) { bestD = d; best = { c: ci, j }; }
    });
  }
  return best;
}

// ── Neuron detector ───────────────────────────────────────────────────────

const detailCv = document.createElement('canvas');
detailCv.width = G; detailCv.height = G;
const detailCtx = detailCv.getContext('2d');
const detailImg = detailCtx.createImageData(G, G);

function drawDetail() {
  const c = setupCanvas($('detail'));
  if (!c) return;
  const { ctx, w, h } = c;
  ctx.clearRect(0, 0, w, h);
  const title = $('detail-title');
  const note = $('detail-note');
  if (!S.neuron) {
    setText(title, 'NEURON DETECTOR');
    setText(note, 'Hover a hidden neuron in the network to see its own heat map. Pink is where it fires positive, cyan where it fires negative. Click to pin it.');
    ctx.fillStyle = 'rgba(255,255,255,0.35)';
    ctx.font = '11px "JetBrains Mono", monospace';
    ctx.textAlign = 'center';
    ctx.fillText('no neuron', w / 2, h / 2);
    ctx.textAlign = 'start';
    return;
  }
  const { c: col, j } = S.neuron;
  const vals = neuronActivations(S.net, gridFeatures(), G * G, col - 1, j);
  const d = detailImg.data;
  let pos = 0;
  for (let k = 0; k < G * G; k++) {
    const dv = diverging(S.act, vals[k]);
    const base = mix((dv + 1) / 2);
    const f = 0.08 + 0.92 * Math.min(1, Math.abs(dv));
    const k4 = 4 * k;
    d[k4] = base[0] * f; d[k4 + 1] = base[1] * f; d[k4 + 2] = base[2] * f; d[k4 + 3] = 255;
    if (dv > 0) pos++;
  }
  detailCtx.putImageData(detailImg, 0, 0);
  ctx.imageSmoothingEnabled = true;
  ctx.drawImage(detailCv, 0, 0, w, h);
  // The data points, faint, for orientation.
  for (const p of S.points) {
    const [x, y] = effective(p);
    const [sx, sy] = toScreen(x, y, w, h);
    ctx.fillStyle = p.label ? 'rgba(255,0,160,0.8)' : 'rgba(0,245,255,0.8)';
    ctx.fillRect(sx - 1.5, sy - 1.5, 3, 3);
  }
  const share = Math.round((100 * pos) / (G * G));
  setText(title, `HIDDEN ${col} · NEURON ${j + 1}${S.pinned ? ' (PINNED)' : ''}`);
  setText(note, `Fires positive on ${share}% of the plane (pink) and negative on ${100 - share}% (cyan). ` +
    'Its weights into the next layer decide how it votes on the answer.');
}

function setText(el, text) {
  if (el && el.textContent !== text) el.textContent = text;
}

// ── Loss curve ────────────────────────────────────────────────────────────

function drawLoss() {
  const c = setupCanvas($('loss'));
  if (!c) return;
  const { ctx, w, h } = c;
  ctx.clearRect(0, 0, w, h);
  const H = S.hist;
  const n = H.trL.length;
  const i0 = Math.max(0, n - 1200);
  const ly = (v) => Math.log10(Math.max(v, 1e-6));
  let lo = Infinity, hi = -Infinity;
  for (let i = i0; i < n; i++) {
    for (const arr of [H.trL, H.teL]) {
      const v = arr[i];
      if (Number.isFinite(v)) { lo = Math.min(lo, ly(v)); hi = Math.max(hi, ly(v)); }
    }
  }
  const pad = { l: 46, r: 12, t: 12, b: 20 };
  if (!Number.isFinite(lo)) {
    ctx.fillStyle = 'rgba(255,255,255,0.35)';
    ctx.font = '11px "JetBrains Mono", monospace';
    ctx.fillText('waiting for data', pad.l, h / 2);
    return;
  }
  lo = Math.floor(Math.min(lo, hi - 0.5));
  hi = Math.ceil(Math.max(hi, lo + 0.5));
  const span = hi - lo;
  const X = (i) => pad.l + ((i - i0) / Math.max(1, n - 1 - i0)) * (w - pad.l - pad.r);
  const Y = (lg) => pad.t + (1 - (lg - lo) / span) * (h - pad.t - pad.b);

  // Decade gridlines and labels.
  ctx.font = '10px "JetBrains Mono", monospace';
  ctx.fillStyle = 'rgba(255,255,255,0.45)';
  ctx.strokeStyle = 'rgba(255,255,255,0.08)';
  ctx.lineWidth = 1;
  for (let k = lo; k <= hi; k++) {
    const y = Y(k);
    ctx.beginPath(); ctx.moveTo(pad.l, y); ctx.lineTo(w - pad.r, y); ctx.stroke();
    ctx.fillText(`1e${k}`, 4, y + 3);
  }

  const plot = (arr, col, dash) => {
    ctx.strokeStyle = rgb(col);
    ctx.shadowColor = rgb(col);
    ctx.shadowBlur = 8;
    ctx.lineWidth = 2;
    ctx.setLineDash(dash);
    ctx.beginPath();
    let open = false;
    for (let i = i0; i < n; i++) {
      const v = arr[i];
      if (!Number.isFinite(v)) { open = false; continue; }
      const x = X(i), y = Y(ly(v));
      if (open) ctx.lineTo(x, y); else ctx.moveTo(x, y);
      open = true;
    }
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.shadowBlur = 0;
  };
  plot(H.trL, PINK, []);
  plot(H.teL, CYAN, [5, 4]);

  ctx.textAlign = 'right';
  ctx.fillStyle = rgb(PINK);
  ctx.fillText('TRAIN', w - pad.r - 52, pad.t + 12);
  ctx.fillStyle = rgb(CYAN);
  ctx.fillText('TEST', w - pad.r, pad.t + 12);
  ctx.textAlign = 'start';
}

// ── Readouts and controls ─────────────────────────────────────────────────

function updateUI(force = false) {
  const m = S.metrics;
  const key = [S.epochs, S.playing, S.points.length, S.preset, m.trA, m.teA, m.trL, S.neuron && S.neuron.c, S.neuron && S.neuron.j].join('|');
  if (!force && key === S.uiKey) return;
  S.uiKey = key;
  const pinks = S.points.filter((p) => p.label === 1).length;
  setText($('chip-status'), S.playing ? 'LIVE' : 'PAUSED');
  setText($('chip-epoch'), String(S.epochs));
  setText($('chip-train'), pct(m.trA));
  setText($('chip-test'), S.testFrac > 0 ? pct(m.teA) : '—');
  setText($('chip-loss'), Number.isFinite(m.trL) ? m.trL.toFixed(3) : '—');
  setText($('chip-points'), `${S.points.length} (${pinks} pink)`);
  setText($('seed-val'), String(S.seed));
  const play = $('btn-play');
  setText(play.querySelector('.btn-txt'), S.playing ? '❚❚ PAUSE' : '▶ PLAY');
  play.setAttribute('aria-pressed', String(S.playing));
  document.querySelectorAll('[data-preset]').forEach((b) => b.setAttribute('aria-pressed', String(b.dataset.preset === S.preset)));
  document.querySelectorAll('[data-class]').forEach((b) => b.setAttribute('aria-pressed', String(Number(b.dataset.class) === S.drawClass)));
  document.querySelectorAll('[data-brush]').forEach((b) => b.setAttribute('aria-pressed', String(b.dataset.brush === S.brush)));
  document.querySelectorAll('[data-feature]').forEach((b) => b.setAttribute('aria-pressed', String(S.features.includes(b.dataset.feature))));
  $('plane').classList.toggle('erase', S.brush === 'erase');
  setText($('layers-val'), String(S.layers));
  setText($('neurons-val'), String(S.neurons));
  setText($('noise-val'), S.noise.toFixed(2));
  setText($('speed-val'), String(S.speed));
}

function bindControls() {
  $('plane').addEventListener('pointerdown', (e) => {
    const cv = e.currentTarget;
    cv.setPointerCapture(e.pointerId);
    S.painting = true;
    const [x, y] = pointerPlane(e);
    S.lastPaint = [x, y];
    applyBrush(x, y);
    if (S.brush === 'add') burst([e.clientX, e.clientY], { count: 10, speed: 4 });
  });
  $('plane').addEventListener('pointermove', (e) => {
    const [x, y] = pointerPlane(e);
    S.hover = [x, y];
    if (S.painting && S.brush === 'add' && S.lastPaint && Math.hypot(x - S.lastPaint[0], y - S.lastPaint[1]) > 0.09) {
      addPoint(x, y);
      S.lastPaint = [x, y];
    } else if (S.painting && S.brush === 'erase') {
      erasePoint(x, y);
    }
  });
  const stop = () => { S.painting = false; };
  $('plane').addEventListener('pointerup', stop);
  $('plane').addEventListener('pointercancel', stop);
  $('plane').addEventListener('pointerleave', () => { S.hover = null; });

  const net = $('net');
  net.addEventListener('pointermove', (e) => { S.neuron = neuronAt(e) || S.pinned; });
  net.addEventListener('pointerleave', () => { S.neuron = S.pinned; });
  net.addEventListener('click', (e) => {
    const hit = neuronAt(e);
    if (!hit) return;
    const same = S.pinned && S.pinned.c === hit.c && S.pinned.j === hit.j;
    S.pinned = same ? null : hit;
    S.neuron = S.pinned || hit;
  });

  document.querySelectorAll('[data-preset]').forEach((b) => b.addEventListener('click', () => loadPreset(b.dataset.preset)));
  $('btn-clear').addEventListener('click', () => { S.points = []; S.addRng = new Rng(S.seed + 99); markDataChanged(); buildNet(); });
  document.querySelectorAll('[data-class]').forEach((b) => b.addEventListener('click', () => { S.drawClass = Number(b.dataset.class); }));
  document.querySelectorAll('[data-brush]').forEach((b) => b.addEventListener('click', () => { S.brush = b.dataset.brush; }));

  document.querySelectorAll('[data-feature]').forEach((b) => b.addEventListener('click', () => {
    const f = b.dataset.feature;
    const on = S.features.includes(f);
    if (on && S.features.length === 1) return; // at least one input
    const picked = on ? S.features.filter((k) => k !== f) : [...S.features, f];
    S.features = FEATURES.filter((k) => picked.includes(k));
    S.splitDirty = true;
    buildNet();
  }));

  $('layers').addEventListener('input', (e) => { S.layers = Number(e.target.value); buildNet(); });
  $('neurons').addEventListener('input', (e) => { S.neurons = Number(e.target.value); buildNet(); });
  $('act').addEventListener('change', (e) => { S.act = e.target.value; buildNet(); });
  $('lr').addEventListener('change', (e) => { S.lr = Number(e.target.value); });
  $('batch').addEventListener('change', (e) => { S.batch = Number(e.target.value); });
  $('l2').addEventListener('change', (e) => { S.l2 = Number(e.target.value); });
  $('opt').addEventListener('change', (e) => {
    S.optimizer = e.target.value;
    S.opt = new Optimizer(S.optimizer, S.net, S.lr);
  });
  $('noise').addEventListener('input', (e) => { S.noise = Number(e.target.value); S.splitDirty = true; });
  $('split').addEventListener('change', (e) => { S.testFrac = Number(e.target.value); S.splitDirty = true; });
  $('speed').addEventListener('input', (e) => { S.speed = Number(e.target.value); });

  $('btn-play').addEventListener('click', () => { S.playing = !S.playing; });
  $('btn-step').addEventListener('click', () => { S.playing = false; runEpoch(); });
  $('btn-reset').addEventListener('click', () => buildNet());
  $('btn-seed').addEventListener('click', () => {
    S.seed = Math.floor(Math.random() * 100000);
    S.addRng = new Rng(S.seed + 99);
    if (PRESETS.includes(S.preset)) loadPreset(S.preset);
    else buildNet();
  });

  window.addEventListener('keydown', (e) => {
    if (e.code !== 'Space' || /INPUT|SELECT|TEXTAREA|BUTTON/.test(document.activeElement?.tagName || '')) return;
    e.preventDefault();
    S.playing = !S.playing;
  });
}

function pointerPlane(e) {
  const r = $('plane').getBoundingClientRect();
  const [x, y] = toPlane(e.clientX - r.left, e.clientY - r.top, r.width, r.height);
  return [clamp(x, -D, D), clamp(y, -D, D)];
}

function applyBrush(x, y) {
  if (S.brush === 'add') addPoint(x, y);
  else erasePoint(x, y);
}

// ── Frame loop ────────────────────────────────────────────────────────────

let last = performance.now();

function frame(now) {
  const dt = clamp((now - last) / 1000, 0, 0.1);
  last = now;
  if (S.playing) {
    // Accumulate fractional epochs so that any speed works at any frame rate.
    S.budget += S.speed * dt;
    let steps = 0;
    while (S.budget >= 1 && steps < MAX_EPOCHS_PER_FRAME) { runEpoch(); S.budget -= 1; steps++; }
    if (steps === MAX_EPOCHS_PER_FRAME) S.budget = 0;
  } else {
    S.budget = 0;
  }
  S.pulse *= 0.86;
  S.frame++;
  if (S.frame % 40 === 0 && S.points.length) S.sample = (S.sample + 1) % S.points.length;
  drawPlane();
  drawNet();
  drawDetail();
  drawLoss();
  updateUI(S.frame % 6 === 0);
  requestAnimationFrame(frame);
}

loadPreset(S.preset);
bindControls();
updateUI(true);
requestAnimationFrame(frame);
