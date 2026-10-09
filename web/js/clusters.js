// Cluster lab page: paint or load points, then watch k-means, DBSCAN or an EM-fitted Gaussian mixture work.
//
// The maths lives in clusters-core.js (a port of the Python package clusters/). This file owns the page: the
// spray brush, the centroid drag, the step loop (each algorithm is a generator, advanced by the speed slider or
// the STEP button), the canvas drawing, the convergence and elbow charts, and the readouts.

import { banner, burst } from './fx.js';
import {
  assign, dbscanSteps, ellipse, gmmSteps, kmeans, kmeansSteps, makeDataset, neighbourhoods, silhouette, sse,
} from './clusters-core.js';

const $ = (id) => document.getElementById(id);
const PALETTE = ['#00f5ff', '#ff00a0', '#00ff88', '#ffe600', '#9b00ff', '#ff6a00', '#4dd2ff', '#ff4d6d', '#b4ff00', '#ff8ae2'];
const GREY = '#5a6478';
const MAX_POINTS = 1200; // keeps the O(n^2) neighbour and silhouette passes instant
const PAD = 18;          // margin around the unit square, in CSS pixels
const VN = 96;           // Voronoi resolution (cells per side)
const REDUCED = matchMedia('(prefers-reduced-motion: reduce)').matches;

const hexRgb = (h) => [1, 3, 5].map((i) => parseInt(h.slice(i, i + 2), 16));
const RGB = PALETTE.map(hexRgb);
const clamp01 = (v) => (v < 0 ? 0 : v > 1 ? 1 : v);

const cvs = $('cl-canvas');
const ctx = cvs.getContext('2d');
const conv = $('conv-canvas');
const cctx = conv.getContext('2d');
const elb = $('elbow-canvas');
const ectx = elb.getContext('2d');
const vor = document.createElement('canvas');
vor.width = VN;
vor.height = VN;
const vctx = vor.getContext('2d');

const view = { dpr: 1, size: 300 };

const S = {
  pts: [],          // flat [x0, y0, x1, y1, ...] in the unit square; the source of truth while painting
  P: new Float64Array(0),
  preset: 'blobs',
  seed: 2,
  npts: 300,
  algo: 'kmeans',
  k: 4,
  init: 'kmeans++',
  eps: 0.08,
  minPts: 5,
  brush: 0.06,
  rate: 3,          // points sprayed per frame while the pointer is down
  perSec: 3,        // algorithm steps per second while playing
  playing: true,
  painting: false,
  paintAt: null,
  hover: null,
  run: null,
  custom: null,     // centroids the user dragged; the next k-means run starts there
  dragIdx: -1,
  sil: null,
  silAt: 0,
  vorDirty: true,
  elbowTok: 0,
  elbowVals: [],
  elbowTimer: 0,
  last: 0,
};

// ── small helpers ───────────────────────────────────────────────────────

function log(msg, cls = 'log-info') {
  const el = document.createElement('div');
  el.className = cls;
  el.textContent = msg;
  const box = $('log');
  box.appendChild(el);
  while (box.childElementCount > 60) box.firstElementChild.remove();
  box.scrollTop = box.scrollHeight;
}

function setPhase(text) {
  $('cl-phase').textContent = text;
}

function setPlayLabel() {
  const run = S.run && !S.run.done;
  $('btn-toggle').querySelector('.btn-txt').textContent = S.playing && run ? '❚❚ PAUSE' : '▶ PLAY';
}

const ALGO_NAME = { kmeans: 'K-MEANS', dbscan: 'DBSCAN', gmm: 'EM' };

function nPoints() {
  return S.P.length >> 1;
}

/** Map a pointer event to unit-square coordinates. */
function toData(e) {
  const r = cvs.getBoundingClientRect();
  const span = view.size - 2 * PAD;
  const x = (e.clientX - r.left - PAD) / span;
  const y = 1 - (e.clientY - r.top - PAD) / span;
  return [clamp01(x), clamp01(y)];
}

/** Unit-square coordinates to canvas pixels. */
const sx = (x) => PAD + x * (view.size - 2 * PAD);
const sy = (y) => PAD + (1 - y) * (view.size - 2 * PAD);

// ── data ────────────────────────────────────────────────────────────────

/** Rebuild the typed-array copy and everything that depends on the points. */
function dataChanged({ keepCustom = false } = {}) {
  S.P = Float64Array.from(S.pts);
  if (!keepCustom) S.custom = null;
  S.run = null;
  S.sil = null;
  renderReadouts();
  scheduleElbow();
}

function loadPreset(name) {
  S.preset = name;
  S.pts = Array.from(makeDataset(name, S.seed, S.npts).points);
  highlightPreset();
  dataChanged();
  log(`preset ${name} · ${S.npts} points · seed ${S.seed}`, 'log-move');
  startRun();
}

function highlightPreset() {
  document.querySelectorAll('#presets [data-preset]').forEach((b) => {
    b.classList.toggle('is-on', b.dataset.preset === S.preset);
  });
}

function spray() {
  if (!S.paintAt) return;
  const [cx, cy] = S.paintAt;
  for (let i = 0; i < S.rate && S.pts.length < 2 * MAX_POINTS; i++) {
    const a = Math.random() * 2 * Math.PI;
    const r = Math.sqrt(Math.random()) * S.brush; // uniform over the disc
    S.pts.push(clamp01(cx + r * Math.cos(a)), clamp01(cy + r * Math.sin(a)));
  }
  S.P = Float64Array.from(S.pts);
}

// ── runs ────────────────────────────────────────────────────────────────

/** Start a fresh run of the selected algorithm on the current points. */
function startRun({ centers = null } = {}) {
  const n = nPoints();
  if (n < 2) {
    S.run = null;
    setPhase('ADD AT LEAST TWO POINTS');
    renderReadouts();
    return;
  }
  const k = Math.min(S.k, n);
  const run = {
    algo: S.algo, k, n, done: false, converged: false, acc: 0, iter: 0, phase: 'init',
    history: [], trails: Array.from({ length: k }, () => []), centers: null, labels: null, resp: null, weights: null, means: null, covs: null,
    loglik: null, focus: -1, core: null, nbrs: null, budget: 1, result: null,
  };
  if (S.algo === 'kmeans') {
    const start = centers || (S.custom && S.custom.length === 2 * k ? S.custom : null);
    run.gen = kmeansSteps(S.P, k, { seed: S.seed, init: S.init, centers: start, maxIter: 100 });
    run.label = `k-means ${start ? '(dragged centroids)' : S.init} · k=${k}`;
  } else if (S.algo === 'dbscan') {
    run.gen = dbscanSteps(S.P, S.eps, S.minPts);
    run.labels = new Int32Array(n).fill(-1);
    run.nbrs = neighbourhoods(S.P, S.eps);
    run.core = run.nbrs.map((nb) => nb.length >= S.minPts);
    run.budget = Math.max(2, Math.round(n / 15)); // a step handles a slice of points, so the whole run is about 15 steps
    run.label = `DBSCAN · eps=${S.eps.toFixed(3)} · minPts=${S.minPts}`;
  } else {
    run.gen = gmmSteps(S.P, k, { seed: S.seed, init: S.init, maxIter: 200 });
    run.label = `EM · k=${k} · ${S.init}`;
  }
  S.run = run;
  S.vorDirty = true;
  S.playing = true;
  $('chip-algo').textContent = ALGO_NAME[S.algo];
  log(`run: ${run.label} on ${n} points`, 'log-move');
  setPhase(S.algo === 'dbscan' ? 'GROWING CLUSTERS' : 'INITIALISING');
  setPlayLabel();
  renderReadouts();
}

/** Record one yielded step from the generator. */
function absorb(run, v) {
  if (run.algo === 'kmeans') {
    run.phase = v.phase;
    run.centers = v.centers;
    if (v.labels) run.labels = v.labels;
    if (v.phase === 'assign') {
      run.iter = v.iteration;
      run.history.push(v.inertia);
    } else {
      pushTrail(run, v.centers);
    }
    S.vorDirty = true;
  } else if (run.algo === 'dbscan') {
    run.labels = v.labels; // the generator's own array, updated in place as points are claimed
    run.focus = v.i;
  } else {
    run.phase = v.phase === 'e' ? 'E' : 'M';
    run.iter = v.iteration;
    run.resp = v.resp;
    run.weights = v.weights;
    run.means = v.means;
    run.covs = v.covs;
    run.loglik = v.loglik;
    if (v.phase === 'e') run.history.push(v.loglik);
  }
}

/** Keep the last 14 positions of every centroid, so its path can be drawn as a fading trail. */
function pushTrail(run, C) {
  for (let j = 0; j < run.k; j++) {
    const t = run.trails[j];
    t.push([C[2 * j], C[2 * j + 1]]);
    if (t.length > 14) t.shift();
  }
}

/** Finish a run: store its result, the silhouette, and celebrate convergence. */
function finish(run, value) {
  run.done = true;
  if (run.algo === 'kmeans') {
    const converged = run.phase === 'assign';
    run.converged = converged;
    if (!converged) {
      run.labels = assign(S.P, run.centers, run.k);
    }
    run.inertia = sse(S.P, run.labels, run.centers);
  } else if (run.algo === 'dbscan') {
    run.labels = value.labels;
    run.core = value.core;
    run.clustersFound = value.clusters;
    run.converged = true;
  } else {
    run.labels = value.labels;
    run.converged = value.converged;
    run.loglik = value.loglik;
    run.iter = value.iterations;
    run.weights = value.weights;
    run.means = value.means;
    run.covs = value.covs;
    run.resp = value.resp;
  }
  S.sil = silhouette(S.P, run.labels);
  S.silAt = performance.now();
  S.playing = false;
  setPlayLabel();
  const found = clusterCount(run);
  const tail = run.algo === 'kmeans'
    ? `${run.iter} iterations, inertia ${run.inertia.toFixed(3)}`
    : run.algo === 'dbscan'
      ? `${found} clusters, ${noiseCount(run)} noise`
      : `${run.iter} E steps, log-likelihood ${run.loglik.toFixed(3)} per point`;
  if (run.algo === 'kmeans' && !run.converged) {
    setPhase(`STOPPED AT ${run.iter} ITERATIONS (NOT CONVERGED)`);
    log(`max iterations reached · ${tail}`, 'log-err');
  } else {
    const verb = run.algo === 'dbscan' ? 'DONE' : 'CONVERGED';
    setPhase(`${verb} · ${tail}`);
    log(`${verb.toLowerCase()} · ${tail}`, 'log-best');
    const color = PALETTE[0];
    banner(verb, `${ALGO_NAME[run.algo]} · ${found} clusters`, color);
    const r = cvs.getBoundingClientRect();
    burst([r.left + r.width / 2, r.top + r.height / 2], { count: 90 });
  }
  renderReadouts();
  drawConv();
}

/** Advance the run by one step (a slice of the points for DBSCAN). */
function stepOnce(run) {
  if (!run || run.done) return;
  const budget = run.algo === 'dbscan' ? run.budget : 1;
  for (let b = 0; b < budget; b++) {
    const r = run.gen.next();
    if (r.done) {
      finish(run, r.value);
      return;
    }
    absorb(run, r.value);
  }
  if (run.algo === 'dbscan') run.history.push(run.labels.reduce((c, v) => c + (v >= 0 ? 1 : 0), 0));
  if (run.algo === 'kmeans' && run.phase === 'assign') {
    log(`iteration ${run.iter} · inertia ${run.history[run.history.length - 1].toFixed(3)}`);
    setPhase('ASSIGN · each point joins its nearest centroid');
  } else if (run.algo === 'kmeans') {
    setPhase('UPDATE · centroids move to their cluster means');
  } else if (run.algo === 'gmm') {
    setPhase(run.phase === 'E' ? 'E STEP · responsibilities for each point' : 'M STEP · ellipses refit to the points they own');
  }
  renderReadouts();
  drawConv();
}

function clusterCount(run) {
  if (!run) return 0;
  if (run.algo === 'kmeans') {
    if (!run.labels) return 0;
    return new Set(run.labels).size;
  }
  if (run.algo === 'dbscan') {
    if (!run.labels) return 0;
    return new Set(Array.from(run.labels).filter((v) => v >= 0)).size;
  }
  return run.labels ? new Set(run.labels).size : 0;
}

function noiseCount(run) {
  if (!run || run.algo !== 'dbscan' || !run.labels) return 0;
  let c = 0;
  for (const v of run.labels) if (v < 0) c++;
  return c;
}

// ── readouts ────────────────────────────────────────────────────────────

function renderReadouts() {
  const run = S.run;
  const n = nPoints();
  const fmt = (v, d = 3) => (v === null || v === undefined || Number.isNaN(v) ? '—' : Number(v).toFixed(d));
  const set = (id, text) => {
    $(id).textContent = text;
  };
  if (!run) {
    set('ro-k', '—'); set('ro-iter', '—'); set('ro-inertia', '—'); set('ro-ll', '—');
    set('ro-noise', '—'); set('ro-conv', '—');
    set('chip-k', '0'); set('chip-iter', '0');
  } else {
    const found = clusterCount(run);
    const iter = run.algo === 'dbscan' ? '—' : String(run.iter);
    set('ro-k', String(found));
    set('ro-iter', iter);
    set('chip-k', String(found));
    set('chip-iter', run.algo === 'dbscan' ? '—' : String(run.iter));
    set('ro-inertia', run.algo === 'kmeans' ? fmt(run.done ? run.inertia : run.history[run.history.length - 1], 2) : '—');
    set('ro-ll', run.algo === 'gmm' ? fmt(run.loglik, 4) : '—');
    set('ro-noise', run.algo === 'dbscan' ? `${noiseCount(run)} / ${n}` : '—');
    set('ro-conv', run.done ? (run.converged ? 'yes' : 'no, max iterations') : 'running');
  }
  set('ro-sil', S.sil === null ? '—' : fmt(S.sil, 3));
  set('chip-sil', S.sil === null ? '—' : fmt(S.sil, 2));
  const notes = {
    kmeans: 'Drag a centroid to restart from there. Shaded regions are the Voronoi cells of the centroids.',
    dbscan: 'Hover a point to see its eps circle. Hollow rings are core points not yet reached; grey crosses are noise.',
    gmm: 'Ellipses are the 2-sigma contours of each Gaussian; colours blend by responsibility.',
  };
  $('ro-note').textContent = S.hoverNote && S.algo === 'dbscan' && S.hover ? S.hoverNote : notes[S.algo];
}

// ── elbow (background sweep) ────────────────────────────────────────────

function scheduleElbow() {
  clearTimeout(S.elbowTimer);
  S.elbowTimer = setTimeout(startElbow, 250);
}

/** Run k-means for k = 1..10 a step at a time, so the page stays responsive while the curve fills in. */
function startElbow() {
  const tok = ++S.elbowTok;
  const P = S.P;
  const n = P.length >> 1;
  S.elbowVals = [];
  if (n < 2) {
    $('elbow-note').textContent = 'Add points to see the elbow.';
    drawElbow();
    return;
  }
  const kMax = Math.min(10, n);
  $('elbow-note').textContent = 'Running k = 1 to 10 in the background…';
  let k = 1;
  const next = () => {
    if (tok !== S.elbowTok) return;
    if (k > kMax) {
      $('elbow-note').textContent = `Inertia for k = 1 to ${kMax}, same seed. Look for the bend.`;
      return;
    }
    S.elbowVals.push(kmeans(P, k, { seed: S.seed }).inertia);
    drawElbow();
    k++;
    setTimeout(next, 0);
  };
  setTimeout(next, 0);
}

// ── charts ──────────────────────────────────────────────────────────────

function sizeCanvases() {
  view.dpr = Math.min(window.devicePixelRatio || 1, 2);
  const size = Math.round(cvs.getBoundingClientRect().width) || 300;
  view.size = size;
  cvs.width = Math.round(size * view.dpr);
  cvs.height = Math.round(size * view.dpr);
  for (const c of [conv, elb]) {
    const r = c.getBoundingClientRect();
    c.width = Math.round(r.width * view.dpr) || 300;
    c.height = Math.round(r.height * view.dpr) || 170;
  }
  S.vorDirty = true;
}

/** Plot values as a glowing polyline with axes; `marker` draws a dashed vertical line at x. */
function plot(c, g, { xs, ys, color, marker = null, xTitle = '', fixedY = null, xRange = null }) {
  const W = c.width / view.dpr;
  const H = c.height / view.dpr;
  g.setTransform(view.dpr, 0, 0, view.dpr, 0, 0);
  g.clearRect(0, 0, W, H);
  const L = 42, R = 10, T = 10, B = 22;
  let lo = fixedY ? fixedY[0] : Infinity;
  let hi = fixedY ? fixedY[1] : -Infinity;
  if (!fixedY) for (const y of ys) { lo = Math.min(lo, y); hi = Math.max(hi, y); }
  if (!Number.isFinite(lo)) { lo = 0; hi = 1; }
  if (hi - lo < 1e-9) { hi = lo + 1; }
  // Enough decimals that the tick labels differ: 2 for big ranges, more for small ones.
  const decimals = hi - lo >= 10 ? 0 : hi - lo >= 1 ? 2 : hi - lo >= 0.01 ? 3 : 5;
  const [xMin, xMax] = xRange || [xs.length ? xs[0] : 0, xs.length ? xs[xs.length - 1] : 1];
  const X = (x) => L + ((x - xMin) / Math.max(1e-9, xMax - xMin)) * (W - L - R);
  const Y = (y) => T + (1 - (y - lo) / (hi - lo)) * (H - T - B);
  g.strokeStyle = 'rgba(0,245,255,0.12)';
  g.fillStyle = 'rgba(160,190,210,0.8)';
  g.font = '10px JetBrains Mono, monospace';
  for (let i = 0; i <= 3; i++) {
    const y = T + (i / 3) * (H - T - B);
    g.beginPath(); g.moveTo(L, y); g.lineTo(W - R, y); g.stroke();
    const v = hi - (i / 3) * (hi - lo);
    g.fillText(v.toFixed(decimals), 2, y + 3);
  }
  g.fillText(xTitle, L, H - 6);
  if (marker !== null && xs.length) {
    g.setLineDash([4, 4]);
    g.strokeStyle = 'rgba(255,230,0,0.6)';
    g.beginPath(); g.moveTo(X(marker), T); g.lineTo(X(marker), H - B); g.stroke();
    g.setLineDash([]);
  }
  if (!ys.length) return;
  g.shadowColor = color; g.shadowBlur = 8;
  g.strokeStyle = color; g.lineWidth = 2;
  g.beginPath();
  ys.forEach((y, i) => (i ? g.lineTo(X(xs[i]), Y(y)) : g.moveTo(X(xs[i]), Y(y))));
  g.stroke();
  g.shadowBlur = 0;
  g.fillStyle = color;
  ys.forEach((y, i) => { g.beginPath(); g.arc(X(xs[i]), Y(y), 2.5, 0, 2 * Math.PI); g.fill(); });
  const last = ys.length - 1;
  g.beginPath(); g.arc(X(xs[last]), Y(ys[last]), 5, 0, 2 * Math.PI); g.fillStyle = '#ffffff'; g.fill();
}

function drawConv() {
  const run = S.run;
  const title = $('conv-title');
  if (!run || !run.history.length) {
    plot(conv, cctx, { xs: [], ys: [], color: PALETTE[0], xTitle: 'waiting for the first step' });
    title.textContent = run ? (run.algo === 'kmeans' ? 'Inertia per iteration' : 'Progress') : 'Inertia per iteration';
    return;
  }
  if (run.algo === 'kmeans') {
    title.textContent = 'Inertia (SSE) per iteration';
    plot(conv, cctx, { xs: run.history.map((_, i) => i + 1), ys: run.history, color: PALETTE[0], xTitle: 'iteration' });
  } else if (run.algo === 'dbscan') {
    title.textContent = 'Points claimed by clusters';
    plot(conv, cctx, { xs: run.history.map((_, i) => i + 1), ys: run.history, color: PALETTE[2], xTitle: 'step', fixedY: [0, nPoints()] });
  } else {
    title.textContent = 'Average log-likelihood per point';
    plot(conv, cctx, { xs: run.history.map((_, i) => i + 1), ys: run.history, color: PALETTE[1], xTitle: 'E step' });
  }
}

function drawElbow() {
  const vals = S.elbowVals;
  plot(elb, ectx, {
    xs: vals.map((_, i) => i + 1),
    ys: vals,
    color: PALETTE[3],
    marker: S.k,
    xRange: [1, 10],
    xTitle: 'k (dashed line: current k)',
  });
}

// ── drawing the plot ────────────────────────────────────────────────────

function drawGrid(g) {
  g.strokeStyle = 'rgba(0,245,255,0.07)';
  g.lineWidth = 1;
  for (let i = 0; i <= 10; i++) {
    const t = i / 10;
    g.beginPath(); g.moveTo(sx(t), sy(0)); g.lineTo(sx(t), sy(1)); g.stroke();
    g.beginPath(); g.moveTo(sx(0), sy(t)); g.lineTo(sx(1), sy(t)); g.stroke();
  }
  g.strokeStyle = 'rgba(0,245,255,0.35)';
  g.strokeRect(sx(0), sy(1), sx(1) - sx(0), sy(0) - sy(1));
}

/** Nearest-centroid colouring of the square, drawn once per centroid change at VN x VN resolution. */
function renderVoronoi(C, k) {
  const img = vctx.createImageData(VN, VN);
  const d = img.data;
  for (let gy = 0; gy < VN; gy++) {
    const y = 1 - (gy + 0.5) / VN;
    for (let gx = 0; gx < VN; gx++) {
      const x = (gx + 0.5) / VN;
      let best = 0;
      let bd = Infinity;
      for (let j = 0; j < k; j++) {
        const dx = x - C[2 * j];
        const dy = y - C[2 * j + 1];
        const v = dx * dx + dy * dy;
        if (v < bd) { bd = v; best = j; }
      }
      const o = (gy * VN + gx) * 4;
      const c = RGB[best % RGB.length];
      d[o] = c[0]; d[o + 1] = c[1]; d[o + 2] = c[2]; d[o + 3] = 34;
    }
  }
  vctx.putImageData(img, 0, 0);
}

function draw(now) {
  const W = view.size;
  const run = S.run;
  const P = S.P;
  const n = P.length >> 1;
  ctx.setTransform(view.dpr, 0, 0, view.dpr, 0, 0);
  ctx.fillStyle = '#03060f';
  ctx.fillRect(0, 0, W, W);
  drawGrid(ctx);

  // Voronoi regions of the k-means centroids
  if (run && run.algo === 'kmeans' && run.centers) {
    if (S.vorDirty) {
      renderVoronoi(run.centers, run.k);
      S.vorDirty = false;
    }
    ctx.imageSmoothingEnabled = true;
    ctx.drawImage(vor, sx(0), sy(1), sx(1) - sx(0), sy(0) - sy(1));
  }

  // GMM: 2-sigma ellipses, drawn as device-space polylines so the stroke width stays uniform.
  // They breathe while the fit is still moving and settle once it converges.
  if (run && run.algo === 'gmm' && run.resp) {
    const span = view.size - 2 * PAD;
    const breath = run.done ? 0.01 : 0.05;
    for (let j = 0; j < run.k; j++) {
      const { major, minor, angle } = ellipse(run.covs.subarray(4 * j, 4 * j + 4), 2);
      const m = 1 + breath * Math.sin(now / 380 + j * 1.3);
      const cx = sx(run.means[2 * j]);
      const cy = sy(run.means[2 * j + 1]);
      const colour = PALETTE[j % PALETTE.length];
      ctx.save();
      ctx.shadowColor = colour;
      ctx.shadowBlur = 12;
      ctx.strokeStyle = colour;
      ctx.globalAlpha = Math.min(1, 0.45 + run.weights[j] * 1.2);
      ctx.lineWidth = 1.6;
      ctx.beginPath();
      for (let t = 0; t <= 64; t++) {
        const th = (t / 64) * 2 * Math.PI;
        const u = major * m * Math.cos(th);
        const v = minor * m * Math.sin(th);
        const dx = Math.cos(angle) * u - Math.sin(angle) * v;
        const dy = Math.sin(angle) * u + Math.cos(angle) * v;
        const px = cx + dx * span;
        const py = cy - dy * span; // data y points up, canvas y down
        if (t === 0) ctx.moveTo(px, py);
        else ctx.lineTo(px, py);
      }
      ctx.stroke();
      ctx.restore();
    }
  }

  // Points. The colour is what the algorithm currently says about each point.
  const colourOf = (i) => {
    if (!run) return GREY;
    if (run.algo === 'kmeans') return run.labels ? PALETTE[run.labels[i] % PALETTE.length] : GREY;
    return GREY;
  };

  // DBSCAN: eps circle under the cursor
  if (S.algo === 'dbscan' && S.hover && !S.painting) {
    const r = S.eps * (view.size - 2 * PAD);
    const hx = sx(S.hover[0]);
    const hy = sy(S.hover[1]);
    ctx.save();
    ctx.fillStyle = 'rgba(0,245,255,0.06)';
    ctx.strokeStyle = 'rgba(0,245,255,0.7)';
    ctx.lineWidth = 1.2;
    ctx.beginPath(); ctx.arc(hx, hy, r, 0, 2 * Math.PI); ctx.fill(); ctx.stroke();
    ctx.restore();
  }

  for (let i = 0; i < n; i++) {
    const x = sx(P[2 * i]);
    const y = sy(P[2 * i + 1]);
    if (run && run.algo === 'dbscan') {
      const lab = run.labels[i];
      if (lab >= 0) {
        const c = RGB[lab % RGB.length];
        ctx.fillStyle = `rgb(${c[0]},${c[1]},${c[2]})`;
        if (run.core[i]) {
          ctx.beginPath(); ctx.arc(x, y, 3.4, 0, 2 * Math.PI); ctx.fill();
        } else {
          ctx.globalAlpha = 0.6;
          ctx.beginPath(); ctx.arc(x, y, 2.6, 0, 2 * Math.PI); ctx.fill();
          ctx.globalAlpha = 1;
        }
      } else if (run.done) {
        ctx.strokeStyle = GREY; ctx.lineWidth = 1.2;
        ctx.beginPath(); ctx.moveTo(x - 3, y - 3); ctx.lineTo(x + 3, y + 3);
        ctx.moveTo(x + 3, y - 3); ctx.lineTo(x - 3, y + 3); ctx.stroke();
      } else if (run.core[i]) {
        ctx.strokeStyle = 'rgba(0,245,255,0.55)'; ctx.lineWidth = 1;
        ctx.beginPath(); ctx.arc(x, y, 3.2, 0, 2 * Math.PI); ctx.stroke();
      } else {
        ctx.fillStyle = GREY;
        ctx.beginPath(); ctx.arc(x, y, 2, 0, 2 * Math.PI); ctx.fill();
      }
    } else if (run && run.algo === 'gmm' && run.resp) {
      let r = 0; let g = 0; let b = 0;
      const k = run.k;
      for (let j = 0; j < k; j++) {
        const w = run.resp[i * k + j];
        const c = RGB[j % RGB.length];
        r += w * c[0]; g += w * c[1]; b += w * c[2];
      }
      ctx.fillStyle = `rgb(${r | 0},${g | 0},${b | 0})`;
      ctx.beginPath(); ctx.arc(x, y, 3, 0, 2 * Math.PI); ctx.fill();
    } else {
      ctx.fillStyle = colourOf(i);
      ctx.globalAlpha = 0.9;
      ctx.beginPath(); ctx.arc(x, y, 2.8, 0, 2 * Math.PI); ctx.fill();
      ctx.globalAlpha = 1;
    }
  }

  // Focus marker: the DBSCAN point being claimed right now
  if (run && run.algo === 'dbscan' && run.focus >= 0 && !run.done) {
    const x = sx(P[2 * run.focus]);
    const y = sy(P[2 * run.focus + 1]);
    const pulse = 5 + 2 * Math.sin(now / 90);
    ctx.strokeStyle = '#ffffff'; ctx.lineWidth = 1.5;
    ctx.beginPath(); ctx.arc(x, y, pulse, 0, 2 * Math.PI); ctx.stroke();
  }

  // k-means: trails and glowing centroids
  if (run && run.algo === 'kmeans' && run.centers) {
    for (let j = 0; j < run.k; j++) {
      const trail = run.trails[j];
      const colour = PALETTE[j % PALETTE.length];
      ctx.strokeStyle = colour;
      for (let t = 1; t < trail.length; t++) {
        ctx.globalAlpha = (t / trail.length) * 0.6;
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        ctx.moveTo(sx(trail[t - 1][0]), sy(trail[t - 1][1]));
        ctx.lineTo(sx(trail[t][0]), sy(trail[t][1]));
        ctx.stroke();
      }
      ctx.globalAlpha = 1;
      const cx = sx(run.centers[2 * j]);
      const cy = sy(run.centers[2 * j + 1]);
      ctx.save();
      ctx.shadowColor = colour; ctx.shadowBlur = 18;
      ctx.fillStyle = colour;
      ctx.beginPath(); ctx.arc(cx, cy, 6, 0, 2 * Math.PI); ctx.fill();
      ctx.restore();
      ctx.strokeStyle = '#ffffff'; ctx.lineWidth = 1.5;
      ctx.beginPath(); ctx.arc(cx, cy, 9, 0, 2 * Math.PI); ctx.stroke();
      ctx.fillStyle = colour;
      ctx.font = '11px JetBrains Mono, monospace';
      ctx.fillText(`${j + 1}`, cx + 11, cy - 8);
    }
  }

  // GMM: means as glowing cores
  if (run && run.algo === 'gmm' && run.means) {
    for (let j = 0; j < run.k; j++) {
      const colour = PALETTE[j % PALETTE.length];
      const cx = sx(run.means[2 * j]);
      const cy = sy(run.means[2 * j + 1]);
      ctx.save();
      ctx.shadowColor = colour; ctx.shadowBlur = 16;
      ctx.fillStyle = '#ffffff';
      ctx.beginPath(); ctx.arc(cx, cy, 3.5, 0, 2 * Math.PI); ctx.fill();
      ctx.restore();
    }
  }
}

// ── loop ────────────────────────────────────────────────────────────────

function frame(now) {
  const dt = Math.min(100, now - S.last || 16);
  S.last = now;
  if (S.painting) spray();
  const run = S.run;
  if (run && !run.done && S.playing && S.dragIdx < 0) {
    run.acc += (dt * S.perSec) / 1000;
    let guard = 0;
    while (run.acc >= 1 && !run.done && guard++ < 6) {
      run.acc -= 1;
      stepOnce(run);
    }
  }
  if (run && !run.done && now - S.silAt > 300 && run.labels) {
    S.silAt = now;
    S.sil = silhouette(S.P, run.labels);
    renderReadouts();
  }
  draw(now);
  requestAnimationFrame(frame);
}

// ── pointer: spray, drag centroids, hover ───────────────────────────────

function nearestCentre(x, y) {
  const run = S.run;
  if (!run || run.algo !== 'kmeans' || !run.centers) return -1;
  const span = view.size - 2 * PAD;
  let best = -1;
  let bd = 14 * 14;
  for (let j = 0; j < run.k; j++) {
    const dx = (x - run.centers[2 * j]) * span;
    const dy = (y - run.centers[2 * j + 1]) * span;
    const d = dx * dx + dy * dy;
    if (d < bd) { bd = d; best = j; }
  }
  return best;
}

cvs.addEventListener('pointerdown', (e) => {
  cvs.setPointerCapture(e.pointerId);
  const [x, y] = toData(e);
  const j = nearestCentre(x, y);
  if (j >= 0) {
    S.dragIdx = j;
    S.custom = Float64Array.from(S.run.centers);
    S.playing = false;
    setPlayLabel();
    setPhase(`DRAGGING CENTROID ${j + 1} · release to restart from here`);
    return;
  }
  S.painting = true;
  S.paintAt = [x, y];
  S.run = null;
  S.sil = null;
  $('cl-hint').textContent = 'Spraying… release to start the algorithm on your points.';
  setPhase('SPRAYING');
  renderReadouts();
  spray();
});

cvs.addEventListener('pointermove', (e) => {
  const [x, y] = toData(e);
  S.hover = [x, y];
  if (S.dragIdx >= 0 && S.run) {
    S.custom[2 * S.dragIdx] = x;
    S.custom[2 * S.dragIdx + 1] = y;
    S.run.centers = Float64Array.from(S.custom);
    S.vorDirty = true;
  } else if (S.painting) {
    S.paintAt = [x, y];
  }
  if (S.algo === 'dbscan' && S.P.length && !S.painting) {
    const { count } = countNeighbours(x, y);
    S.hoverNote = `Under the cursor: ${count} point${count === 1 ? '' : 's'} within eps (minPts ${S.minPts}): `
      + `${count >= S.minPts ? 'a core point' : 'not a core point'}.`;
    $('ro-note').textContent = S.hoverNote;
  }
  cvs.style.cursor = S.dragIdx >= 0 || nearestCentre(x, y) >= 0 ? 'grab' : 'crosshair';
});

cvs.addEventListener('pointerleave', () => {
  S.hover = null;
  S.hoverNote = null;
  renderReadouts();
});

function countNeighbours(x, y) {
  let count = 0;
  const r2 = S.eps * S.eps;
  for (let i = 0; i < nPoints(); i++) {
    const dx = S.P[2 * i] - x;
    const dy = S.P[2 * i + 1] - y;
    if (dx * dx + dy * dy <= r2) count++;
  }
  return { count };
}

function endPointer() {
  if (S.dragIdx >= 0) {
    const j = S.dragIdx;
    S.dragIdx = -1;
    log(`centroid ${j + 1} moved · restarting k-means from there`, 'log-adv');
    startRun({ centers: S.custom });
    return;
  }
  if (S.painting) {
    S.painting = false;
    S.paintAt = null;
    $('cl-hint').textContent = 'Hold and drag on the plot to spray points. Drag a centroid to restart from there.';
    S.preset = null;
    highlightPreset();
    dataChanged();
    log(`${nPoints()} points painted`, 'log-move');
    startRun();
  }
}
cvs.addEventListener('pointerup', endPointer);
cvs.addEventListener('pointercancel', endPointer);

// ── controls ────────────────────────────────────────────────────────────

function syncPanels() {
  const kOnly = S.algo !== 'dbscan';
  $('p-k').hidden = !kOnly;
  $('p-init').hidden = !kOnly;
  $('p-dbscan').hidden = kOnly;
  $('chip-algo').textContent = ALGO_NAME[S.algo];
}

function bindControls() {
  document.querySelectorAll('#presets [data-preset]').forEach((b) => {
    b.addEventListener('click', () => loadPreset(b.dataset.preset));
  });

  const range = (id, fn) => $(id).addEventListener('input', () => fn($(id).valueAsNumber));

  range('n', (v) => { S.npts = v; $('n-val').textContent = String(v); });
  $('n').addEventListener('change', () => { if (S.preset) loadPreset(S.preset); });

  range('brush', (v) => { S.brush = v / 100; $('brush-val').textContent = S.brush.toFixed(2); });
  range('rate', (v) => { S.rate = v; $('rate-val').textContent = String(v); });
  range('speed', (v) => { S.perSec = v; $('speed-val').textContent = `${v} steps/s`; });
  range('k', (v) => { S.k = v; $('k-val').textContent = String(v); });
  range('eps', (v) => { S.eps = v; $('eps-val').textContent = v.toFixed(3); });
  range('minpts', (v) => { S.minPts = v; $('minpts-val').textContent = String(v); });
  $('k').addEventListener('change', () => { S.custom = null; startRun(); });
  $('init').addEventListener('change', () => { S.init = $('init').value; S.custom = null; startRun(); });
  $('eps').addEventListener('change', () => startRun());
  $('minpts').addEventListener('change', () => startRun());
  $('algo').addEventListener('change', () => {
    S.algo = $('algo').value;
    syncPanels();
    startRun();
  });

  $('seed').addEventListener('change', () => {
    const v = Math.max(0, Math.min(2147483647, Math.floor($('seed').valueAsNumber) || 0));
    S.seed = v;
    $('seed').value = String(v);
    if (S.preset) loadPreset(S.preset);
    else { S.custom = null; startRun(); }
    scheduleElbow();
  });
  $('btn-seed').addEventListener('click', () => {
    $('seed').value = String(Math.floor(Math.random() * 99999));
    $('seed').dispatchEvent(new Event('change'));
  });

  $('btn-run').addEventListener('click', () => {
    if (!S.P.length) return;
    startRun();
  });
  $('btn-toggle').addEventListener('click', () => {
    if (!S.run || S.run.done) {
      if (S.P.length) startRun();
      return;
    }
    S.playing = !S.playing;
    setPlayLabel();
  });
  $('btn-step').addEventListener('click', () => {
    if (!S.run || S.run.done) {
      if (!S.P.length) return;
      startRun();
    }
    S.playing = false;
    setPlayLabel();
    stepOnce(S.run);
  });
  $('btn-clear').addEventListener('click', () => {
    S.pts = [];
    S.preset = null;
    highlightPreset();
    dataChanged();
    log('cleared', 'log-err');
    setPhase('CLEARED · spray some points');
  });
}

function init() {
  sizeCanvases();
  window.addEventListener('resize', () => {
    sizeCanvases();
    drawConv();
    drawElbow();
  });
  bindControls();
  $('n-val').textContent = String(S.npts);
  $('brush-val').textContent = S.brush.toFixed(2);
  $('rate-val').textContent = String(S.rate);
  $('speed-val').textContent = `${S.perSec} steps/s`;
  $('k-val').textContent = String(S.k);
  $('eps-val').textContent = S.eps.toFixed(3);
  $('minpts-val').textContent = String(S.minPts);
  syncPanels();
  loadPreset(S.preset);
  renderReadouts();
  drawConv();
  drawElbow();
  setPlayLabel();
  requestAnimationFrame(frame);
}

init();
