// Regression lab page: drag points on a neon plane and watch the fit snap. Linear or polynomial
// least squares (normal equations or QR), gradient descent creeping toward the same answer, ridge,
// a train/test split with a degree sweep, and logistic regression with a probability map.
// The maths lives in regression-core.js; this file draws, handles input and keeps the state.

import {
  polyFeatures, polyFeatures2d, solveNormal, solveQR, gradientDescent, logisticGD,
  linearLoss, logLoss, mse, predict, sigmoid, accuracy, makeRegression, makeClassification,
} from './regression-core.js';
import { burst, shake } from './fx.js';

// ---------------------------------------------------------------- geometry

// Logical plane size. The canvas is scaled to its column by CSS; pointer positions map back to it.
const W = 720, H = 480, M = 36;
const PW = W - 2 * M, PH = H - 2 * M;
// Pixels per data unit (the data range is [-1, 1] on both axes). A residual of r is r * SY pixels tall.
const SY = PH / 2;

const COL = {
  pink: '#ff00a0', cyan: '#00f5ff', green: '#00ff88', yellow: '#ffe600', purple: '#9b00ff',
  grid: 'rgba(0,245,255,0.07)', axis: 'rgba(0,245,255,0.28)', dim: '#6b7a90',
};

const toPx = (x) => M + ((x + 1) / 2) * PW;
const toPy = (y) => H - M - ((y + 1) / 2) * PH;
const toX = (px) => ((px - M) / PW) * 2 - 1;
const toY = (py) => ((H - M - py) / PH) * 2 - 1;
const clamp = (v) => Math.max(-1, Math.min(1, v));

const REG_PRESETS = ['noisy-sine', 'line', 'quadratic'];
const LOG_PRESETS = ['moons', 'blobs'];
const REG_N = 30, LOG_N = 60;
const HEAT_W = 144, HEAT_H = 96;

// ---------------------------------------------------------------- state

const S = {
  mode: 'reg',
  pts: [],                     // { id, x, y, cls }; cls is 0 or 1 (logistic only)
  nextId: 0,
  seed: 7,                     // matches the CLI default, so `python -m regression fit` reproduces the page
  reg: { preset: 'noisy-sine', degree: 3, method: 'qr', holdout: false },
  log: { preset: 'moons', degree: 1, cls: 1 },
  ridge: 0,
  lr: 0.1,                     // gradient descent step size
  steps: 4,                    // descent steps per animation frame while running
  running: false,
  gd: { key: '', w: null, losses: [], diverged: false, total: 0 },
  history: [],                 // JSON snapshots of pts for undo
  drag: null,                  // { p, moved } while a point is held
  lastTap: null,               // { id, t } for double-tap delete
  lastSSE: null,               // previous squared-error total, to detect improvements
  flashUntil: 0,
  dirty: true,
};

const $ = (id) => document.getElementById(id);
const plane = $('plane'), pctx = plane.getContext('2d');
const lossCv = $('loss-chart'), lctx = lossCv.getContext('2d');
const sweepCv = $('sweep-chart'), sctx = sweepCv.getContext('2d');
const heat = document.createElement('canvas');
heat.width = HEAT_W; heat.height = HEAT_H;
const hctx = heat.getContext('2d');

// Test points are every fourth point by id, the same rule as regression.core.test_mask.
const isTest = (p) => p.id % 4 === 3;
const touch = () => { S.dirty = true; };

// ---------------------------------------------------------------- data

function genData(mode, preset, seed) {
  if (mode === 'reg') {
    const d = makeRegression(preset, REG_N, seed, 0.15);
    return d.x.map((x, i) => ({ id: i, x, y: d.y[i], cls: 0 }));
  }
  const d = makeClassification(preset, LOG_N, seed, preset === 'blobs' ? 0.22 : 0.1);
  return d.xy.map(([x, y], i) => ({ id: i, x, y, cls: d.labels[i] }));
}

/** Replace the points with a preset generated from `seed`; the descent restarts. */
function loadData(seed) {
  const cfg = S.mode === 'reg' ? S.reg : S.log;
  S.seed = seed;
  S.pts = genData(S.mode, cfg.preset, seed);
  S.nextId = S.pts.length;
  S.history = [];
  S.lastSSE = null;
  $('seed-val').textContent = `seed ${seed}`;
  resetGD();
}

function resetGD() {
  S.gd = { key: '', w: null, losses: [], diverged: false, total: 0 };
  touch();
}

// ---------------------------------------------------------------- fits

/** Regression fit. Returns the displayed model plus the exact solution for comparison. */
function regFit() {
  const deg = S.reg.degree, lam = S.ridge;
  const fitPts = S.reg.holdout ? S.pts.filter((p) => !isTest(p)) : S.pts;
  const out = { deg, w: null, exact: null, exactJ: null, sse: null, status: '' };
  if (fitPts.length === 0) { out.status = 'ADD POINTS'; return out; }
  if (lam === 0 && fitPts.length < deg + 1) { out.status = `NEEDS ${deg + 1} POINTS OR RIDGE`; return out; }
  const X = polyFeatures(fitPts.map((p) => p.x), deg);
  const y = fitPts.map((p) => p.y);
  const wQR = solveQR(X, y, lam);
  if (!wQR.every(Number.isFinite)) { out.status = 'SINGULAR: ADD RIDGE'; return out; }
  // The normal equations are solved too, so their error against QR is visible in the HUD.
  const wN = solveNormal(X, y, lam);
  out.exact = wQR;
  out.exactJ = linearLoss(X, y, wQR, lam);
  out.normalGap = Math.max(...wQR.map((v, k) => Math.abs(wN[k] - v)));
  if (S.reg.method === 'normal') out.w = wN;
  else if (S.reg.method === 'qr') out.w = wQR;
  else out.w = gdStep(X, y, false, deg);
  return out;
}

/** Logistic fit: gradient descent on every point. */
function logFit() {
  const deg = S.log.degree;
  const out = { deg, w: null, X: null, labels: null, acc: null, status: '' };
  if (S.pts.length === 0) { out.status = 'ADD POINTS'; return out; }
  out.X = polyFeatures2d(S.pts.map((p) => [p.x, p.y]), deg);
  out.labels = S.pts.map((p) => p.cls);
  out.w = gdStep(out.X, out.labels, true, deg);
  return out;
}

/**
 * Advance the descent by one frame's worth of steps, if it is running. The state restarts from zero
 * when the model shape changes (mode or degree), and warm-starts when only the points move.
 */
function gdStep(X, y, logistic, deg) {
  const key = `${logistic ? 'log' : 'reg'}|${deg}`;
  if (S.gd.key !== key || !S.gd.w) {
    // One weight per feature column: deg + 1 for a polynomial, (d+1)(d+2)/2 for the 2-D logistic features.
    S.gd = { key, w: new Array(X[0].length).fill(0), losses: [], diverged: false, total: 0 };
  }
  if (S.running && !S.gd.diverged) {
    const k = S.steps;
    const r = logistic
      ? logisticGD(X, y, S.lr, k, S.ridge, S.gd.w)
      : gradientDescent(X, y, S.lr, k, S.ridge, S.gd.w);
    S.gd.w = r.w;
    // Each call returns the loss before every step plus the final one; that final one starts the next call.
    for (let i = 0; i < r.losses.length - 1; i++) S.gd.losses.push(r.losses[i]);
    S.gd.losses.push(r.losses[r.losses.length - 1]);
    S.gd.total += k;
    if (S.gd.losses.length > 1500) S.gd.losses.splice(0, S.gd.losses.length - 1500);
    if (r.diverged) {
      S.gd.diverged = true;
      S.running = false;
      shake($('plane-wrap'), 'big');
    }
  }
  return S.gd.w;
}

/** Degree sweep: fit on the non-test points, then measure train and test MSE for degrees 1 to 10. */
function sweepFit() {
  const train = S.pts.filter((p) => !isTest(p));
  const test = S.pts.filter(isTest);
  const rows = [];
  for (let d = 1; d <= 10; d++) {
    if (train.length === 0 || (S.ridge === 0 && train.length < d + 1)) { rows.push(null); continue; }
    const X = polyFeatures(train.map((p) => p.x), d);
    const y = train.map((p) => p.y);
    const w = solveQR(X, y, S.ridge);
    if (!w.every(Number.isFinite)) { rows.push(null); continue; }
    rows.push({
      d,
      train: mse(X, y, w),
      test: test.length ? mse(polyFeatures(test.map((p) => p.x), d), test.map((p) => p.y), w) : null,
    });
  }
  return { rows, hasTest: test.length > 0 };
}

// ---------------------------------------------------------------- drawing

function gridPlane(ctx) {
  ctx.fillStyle = 'rgba(2,6,16,0.9)';
  ctx.fillRect(0, 0, W, H);
  ctx.strokeStyle = COL.grid;
  ctx.lineWidth = 1;
  ctx.beginPath();
  for (let i = -1; i <= 1; i += 0.25) {
    ctx.moveTo(toPx(i), toPy(-1)); ctx.lineTo(toPx(i), toPy(1));
    ctx.moveTo(toPx(-1), toPy(i)); ctx.lineTo(toPx(1), toPy(i));
  }
  ctx.stroke();
  ctx.strokeStyle = COL.axis;
  ctx.beginPath();
  ctx.moveTo(toPx(0), toPy(-1)); ctx.lineTo(toPx(0), toPy(1));
  ctx.moveTo(toPx(-1), toPy(0)); ctx.lineTo(toPx(1), toPy(0));
  ctx.stroke();
  ctx.strokeRect(M, M, PW, PH);
}

/** A point glows and swells while it is dragged and for a moment after; pulse is 0 to 1. */
function pulseOf(p, now) {
  if (!p.pulse) return 0;
  const age = now - p.pulse;
  if (age > 700) return 0;
  return (1 - age / 700) * (0.6 + 0.4 * Math.sin(now / 70));
}

function drawPoint(ctx, p, color, now) {
  const g = pulseOf(p, now);
  ctx.save();
  ctx.shadowColor = color;
  ctx.shadowBlur = 10 + 18 * g;
  ctx.fillStyle = color;
  ctx.beginPath();
  ctx.arc(toPx(p.x), toPy(p.y), 6 + 3 * g, 0, Math.PI * 2);
  ctx.fill();
  ctx.restore();
}

function curveFrom(ctx, grid, gx, w, color, dashed, glow) {
  ctx.save();
  ctx.beginPath();
  const yhat = predict(gx, w);
  grid.forEach((xv, i) => {
    const px = toPx(xv), py = toPy(yhat[i]);
    if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
  });
  ctx.strokeStyle = color;
  ctx.lineWidth = dashed ? 1.5 : 2.6;
  ctx.shadowColor = color;
  ctx.shadowBlur = glow;
  ctx.setLineDash(dashed ? [6, 6] : []);
  ctx.stroke();
  ctx.restore();
}

const GRID_X = Array.from({ length: 181 }, (_, i) => -1 + (2 * i) / 180);

function drawReg(now, fit) {
  gridPlane(pctx);
  if (!fit.w) {
    pctx.fillStyle = COL.yellow;
    pctx.font = '15px monospace';
    pctx.fillText(fit.status, M + 12, M + 22);
    S.pts.forEach((p) => drawPoint(pctx, p, isTest(p) ? COL.green : COL.pink, now));
    return;
  }
  const deg = fit.deg;
  const pred = predict(polyFeatures(S.pts.map((p) => p.x), deg), fit.w);
  const flash = now < S.flashUntil;
  // Big squares can run past the plot; clip them (and the curve) to the plot rectangle.
  pctx.save();
  pctx.beginPath();
  pctx.rect(M, M, PW, PH);
  pctx.clip();
  // Squared errors: a square of side |residual| hangs off each residual segment, so its area is r^2.
  let sse = 0;
  S.pts.forEach((p, i) => {
    const r = p.y - pred[i];
    sse += r * r;
    const side = Math.abs(r) * SY;
    if (side < 0.5) return;
    const col = isTest(p) ? COL.green : COL.pink;
    pctx.save();
    pctx.shadowColor = col;
    pctx.shadowBlur = flash ? 26 : 12;
    pctx.fillStyle = flash ? 'rgba(255,255,255,0.45)' : (isTest(p) ? 'rgba(0,255,136,0.16)' : 'rgba(255,0,160,0.16)');
    pctx.strokeStyle = col;
    pctx.lineWidth = flash ? 2.2 : 1.2;
    const x0 = toPx(p.x), y0 = Math.min(toPy(p.y), toPy(pred[i]));
    pctx.fillRect(x0, y0, side, side);
    pctx.strokeRect(x0, y0, side, side);
    pctx.restore();
  });
  // An improvement of more than 1% in the squared-error total flashes the squares.
  if (S.lastSSE !== null && sse < S.lastSSE * 0.99 && !S.drag) S.flashUntil = now + 420;
  S.lastSSE = sse;
  fit.sse = sse;
  // While the descent creeps, the dashed purple curve is the exact answer it is heading for.
  const gx = polyFeatures(GRID_X, deg);
  if (S.reg.method === 'gd' && fit.exact) curveFrom(pctx, GRID_X, gx, fit.exact, COL.purple, true, 0);
  curveFrom(pctx, GRID_X, gx, fit.w, COL.cyan, false, 14);
  pctx.restore();
  pctx.fillStyle = COL.dim;
  pctx.font = '15px monospace';
  const m = S.reg.method === 'gd' ? 'GRADIENT DESCENT' : S.reg.method === 'normal' ? 'NORMAL EQUATIONS' : 'QR';
  pctx.fillText(`${m} · DEGREE ${deg}${S.ridge > 0 ? ` · RIDGE ${S.ridge.toFixed(2)}` : ''}`, M + 12, M + 20);
  S.pts.forEach((p) => drawPoint(pctx, p, isTest(p) ? COL.green : COL.pink, now));
}

/** Logistic heat map on a coarse grid, stretched to the plane. The p = 0.5 band is drawn in white. */
function drawLog(now, fit) {
  gridPlane(pctx);
  if (fit.X) {
    const img = hctx.createImageData(HEAT_W, HEAT_H);
    const w = fit.w;
    // Monomial exponents in the same order as polyFeatures2d, so w[k] pairs with x1^i x2^j.
    const exps = [];
    for (let t = 0; t <= fit.deg; t++) for (let i = t; i >= 0; i--) exps.push([i, t - i]);
    for (let j = 0; j < HEAT_H; j++) {
      const py = M + ((j + 0.5) / HEAT_H) * PH;
      for (let i = 0; i < HEAT_W; i++) {
        const px = M + ((i + 0.5) / HEAT_W) * PW;
        const x1 = toX(px), x2 = toY(py);
        let z = 0;
        for (let k = 0; k < exps.length; k++) z += w[k] * x1 ** exps[k][0] * x2 ** exps[k][1];
        const p = sigmoid(z);
        const o = (j * HEAT_W + i) * 4;
        const band = Math.abs(p - 0.5) < 0.025;
        img.data[o] = band ? 255 : 255 * (1 - p);
        img.data[o + 1] = band ? 255 : 245 * p;
        img.data[o + 2] = band ? 255 : 160 * (1 - p) + 255 * p;
        img.data[o + 3] = band ? 255 : 40 + 150 * Math.abs(p - 0.5) * 2;
      }
    }
    hctx.putImageData(img, 0, 0);
    pctx.save();
    pctx.imageSmoothingEnabled = true;
    pctx.globalAlpha = 0.8;
    pctx.drawImage(heat, M, M, PW, PH);
    pctx.restore();
    fit.acc = accuracy(fit.X, fit.labels, fit.w);
  }
  // Points: class 0 pink, class 1 cyan. A yellow ring marks a point the boundary gets wrong.
  const pr = fit.X ? predict(fit.X, fit.w) : [];
  S.pts.forEach((p, i) => {
    drawPoint(pctx, p, p.cls === 1 ? COL.cyan : COL.pink, now);
    if (fit.X && (sigmoid(pr[i]) >= 0.5 ? 1 : 0) !== p.cls) {
      pctx.beginPath();
      pctx.strokeStyle = COL.yellow;
      pctx.lineWidth = 2;
      pctx.arc(toPx(p.x), toPy(p.y), 10, 0, Math.PI * 2);
      pctx.stroke();
    }
  });
  pctx.fillStyle = COL.dim;
  pctx.font = '15px monospace';
  pctx.fillText(`LOGISTIC · DEGREE ${fit.deg} · ${fit.w ? fit.w.length : 0} FEATURES`, M + 12, M + 20);
  if (!fit.X) { pctx.fillStyle = COL.yellow; pctx.fillText('ADD POINTS', M + 12, M + 42); }
}

/** Plain line chart on its own canvas. Series are point lists, or { ref } for a dashed horizontal line. */
function chart(ctx, cv, { series, xmin, xmax, ymin, ymax, xlabel, ylabel, marker, empty }) {
  const w = cv.width, h = cv.height;
  const L = 58, R = w - 14, T = 14, B = h - 30;
  ctx.fillStyle = 'rgba(2,6,16,0.9)';
  ctx.fillRect(0, 0, w, h);
  ctx.strokeStyle = COL.grid;
  ctx.strokeRect(L, T, R - L, B - T);
  const X = (v) => L + ((v - xmin) / (xmax - xmin || 1)) * (R - L);
  const Y = (v) => B - ((v - ymin) / (ymax - ymin || 1)) * (B - T);
  ctx.fillStyle = COL.dim;
  ctx.font = '12px monospace';
  ctx.fillText(ylabel, 6, T + 12);
  ctx.fillText(ymax.toFixed(2), 6, T + 26);
  ctx.fillText(ymin.toFixed(2), 6, B);
  ctx.fillText(xlabel, R - ctx.measureText(xlabel).width, h - 8);
  if (marker !== undefined && marker !== null) {
    ctx.strokeStyle = COL.yellow;
    ctx.setLineDash([3, 4]);
    ctx.beginPath(); ctx.moveTo(X(marker), T); ctx.lineTo(X(marker), B); ctx.stroke();
    ctx.setLineDash([]);
  }
  if (empty) { ctx.fillStyle = COL.yellow; ctx.fillText(empty, L + 12, (T + B) / 2); return; }
  for (const s of series) {
    if (s.ref !== undefined) {
      ctx.strokeStyle = s.color;
      ctx.setLineDash([6, 5]);
      ctx.beginPath(); ctx.moveTo(L, Y(s.ref)); ctx.lineTo(R, Y(s.ref)); ctx.stroke();
      ctx.setLineDash([]);
      continue;
    }
    ctx.strokeStyle = s.color;
    ctx.lineWidth = 2;
    ctx.shadowColor = s.color;
    ctx.shadowBlur = 8;
    ctx.beginPath();
    let started = false;
    for (const [xv, yv] of s.points) {
      if (!Number.isFinite(yv)) { started = false; continue; }
      if (started) ctx.lineTo(X(xv), Y(yv)); else { ctx.moveTo(X(xv), Y(yv)); started = true; }
    }
    ctx.stroke();
    ctx.shadowBlur = 0;
    if (s.dots) {
      ctx.fillStyle = s.color;
      for (const [xv, yv] of s.points) {
        if (!Number.isFinite(yv)) continue;
        ctx.beginPath(); ctx.arc(X(xv), Y(yv), 3, 0, Math.PI * 2); ctx.fill();
      }
    }
  }
}

const log10 = (v) => Math.log10(Math.max(v, 1e-12));

/** Log10 of the objective over the descent's steps (last 400), with the exact optimum as a dashed line. */
function drawLossChart(fit) {
  const losses = S.gd.losses.slice(-400);
  const start = S.gd.total - losses.length;
  const pts = losses.map((v, i) => [start + i, log10(v)]);
  const ref = S.mode === 'reg' && fit.exactJ !== null && fit.exactJ !== undefined ? log10(fit.exactJ) : null;
  const ys = pts.map((p) => p[1]).concat(ref === null ? [] : [ref]).filter(Number.isFinite);
  if (!ys.length) {
    chart(lctx, lossCv, { series: [], xmin: 0, xmax: 1, ymin: 0, ymax: 1, xlabel: 'step', ylabel: 'log10 J', empty: 'RUN THE DESCENT TO DRAW ITS LOSS' });
    return;
  }
  const series = [{ points: pts, color: COL.cyan }];
  if (ref !== null) series.push({ ref, color: COL.purple });
  chart(lctx, lossCv, {
    series, xmin: Math.max(0, start), xmax: Math.max(S.gd.total, start + 1),
    ymin: Math.min(...ys) - 0.1, ymax: Math.max(...ys) + 0.1, xlabel: 'step', ylabel: 'log10 J',
  });
}

function drawSweep(sw) {
  const tr = sw.rows.filter(Boolean).map((r) => [r.d, log10(r.train)]);
  const te = sw.rows.filter((r) => r && r.test !== null).map((r) => [r.d, log10(r.test)]);
  const all = tr.map((p) => p[1]).concat(te.map((p) => p[1]));
  if (!all.length) {
    chart(sctx, sweepCv, { series: [], xmin: 1, xmax: 10, ymin: 0, ymax: 1, xlabel: 'degree', ylabel: 'log10 MSE', empty: 'ADD POINTS' });
    return;
  }
  chart(sctx, sweepCv, {
    series: [
      { points: tr, color: COL.cyan, dots: true },
      { points: te, color: COL.green, dots: true },
    ],
    xmin: 1, xmax: 10, ymin: Math.min(...all) - 0.15, ymax: Math.max(...all) + 0.15,
    xlabel: 'degree', ylabel: 'log10 MSE', marker: S.reg.degree,
    empty: sw.hasTest ? null : 'ADD 4+ POINTS FOR A TEST SET',
  });
}

// ---------------------------------------------------------------- HUD

function setText(id, v) {
  const el = $(id);
  if (el && el.textContent !== v) el.textContent = v;
}

function updateHud(fit) {
  if (S.mode === 'reg') {
    const pr = fit.w ? predict(polyFeatures(S.pts.map((p) => p.x), fit.deg), fit.w) : null;
    const mseOf = (test) => {
      if (!pr) return '—';
      const rs = S.pts.map((p, i) => ({ r: p.y - pr[i], t: isTest(p) })).filter((o) => o.t === test);
      return rs.length ? (rs.reduce((s, o) => s + o.r * o.r, 0) / rs.length).toFixed(4) : '—';
    };
    setText('chip-mode', 'POLY REGRESSION');
    setText('chip-deg', String(S.reg.degree));
    setText('chip-n', String(S.pts.length));
    setText('chip-loss', fit.sse !== null && fit.sse !== undefined ? fit.sse.toFixed(3) : '—');
    setText('chip-test', `${mseOf(false)} / ${mseOf(true)}`);
    setText('chip-status', fit.w ? (S.gd.diverged ? 'DIVERGED' : 'FITTED') : fit.status);
  } else {
    setText('chip-mode', 'LOGISTIC');
    setText('chip-deg', String(S.log.degree));
    setText('chip-n', String(S.pts.length));
    const ll = fit.X ? logLoss(fit.X, fit.labels, fit.w, 0) : NaN;
    setText('chip-loss', Number.isFinite(ll) ? ll.toFixed(4) : '—');
    setText('chip-test', fit.acc !== null && fit.acc !== undefined ? `${(fit.acc * 100).toFixed(1)}%` : '—');
    setText('chip-status', S.gd.diverged ? 'DIVERGED' : (S.running ? 'DESCENDING' : 'PAUSED'));
  }
  const last = S.gd.losses.length ? S.gd.losses[S.gd.losses.length - 1] : null;
  setText('gd-status', S.gd.diverged
    ? 'DIVERGED: LOWER THE STEP SIZE, THEN RESET'
    : `STEP ${S.gd.total}${last === null ? '' : ` · J ${last.toPrecision(4)}`}`);
  if (fit.normalGap !== undefined) setText('gap-val', fit.normalGap.toExponential(1));
  const runLabel = S.running ? '❚❚ PAUSE' : '▶ RUN';
  setText('gd-run-label', runLabel);
}

// ---------------------------------------------------------------- frame loop

function render(now) {
  if (S.mode === 'reg') {
    const fit = regFit();
    drawReg(now, fit);
    drawLossChart(fit);
    drawSweep(sweepFit());
    updateHud(fit);
  } else {
    const fit = logFit();
    drawLog(now, fit);
    drawLossChart(fit);
    updateHud(fit);
  }
}

function frame(now) {
  requestAnimationFrame(frame);
  const pulsing = S.drag !== null || S.pts.some((p) => p.pulse && now - p.pulse < 700);
  const descending = S.running && !S.gd.diverged;
  if (!(S.dirty || pulsing || descending || now < S.flashUntil)) return;
  S.dirty = false;
  render(now);
}

// ---------------------------------------------------------------- input

function localPoint(e) {
  const r = plane.getBoundingClientRect();
  return [((e.clientX - r.left) * W) / r.width, ((e.clientY - r.top) * H) / r.height];
}

/** Client (screen) coordinates of a data point, for placing bursts. */
function clientOf(p) {
  const r = plane.getBoundingClientRect();
  return [r.left + (toPx(p.x) * r.width) / W, r.top + (toPy(p.y) * r.height) / H];
}

function nearest(px, py, radius) {
  let best = null, bd = radius * radius;
  for (const p of S.pts) {
    const dx = toPx(p.x) - px, dy = toPy(p.y) - py;
    const d = dx * dx + dy * dy;
    if (d <= bd) { bd = d; best = p; }
  }
  return best;
}

function snapshot() {
  S.history.push(JSON.stringify(S.pts));
  if (S.history.length > 60) S.history.shift();
}

function removePoint(p) {
  snapshot();
  burst(clientOf(p), { count: 18, speed: 3 });
  S.pts = S.pts.filter((q) => q !== p);
  touch();
}

plane.addEventListener('pointerdown', (e) => {
  if (e.button === 2) return;
  e.preventDefault();
  const [px, py] = localPoint(e);
  const hit = nearest(px, py, e.pointerType === 'touch' ? 26 : 16);
  const now = performance.now();
  if (hit) {
    // A second press on the same point within 350 ms deletes it.
    if (S.lastTap && S.lastTap.id === hit.id && now - S.lastTap.t < 350) {
      S.lastTap = null;
      removePoint(hit);
      return;
    }
    S.lastTap = { id: hit.id, t: now };
    snapshot();
    S.drag = { p: hit };
    hit.pulse = now;
    plane.setPointerCapture(e.pointerId);
  } else {
    // Empty space adds a point; in logistic mode the active class colours it.
    snapshot();
    S.pts.push({ id: S.nextId++, x: clamp(toX(px)), y: clamp(toY(py)), cls: S.mode === 'log' ? S.log.cls : 0, pulse: now });
  }
  touch();
});

plane.addEventListener('pointermove', (e) => {
  const [px, py] = localPoint(e);
  if (!S.drag) {
    plane.style.cursor = nearest(px, py, e.pointerType === 'touch' ? 26 : 16) ? 'grab' : 'crosshair';
    return;
  }
  S.drag.p.x = clamp(toX(px));
  S.drag.p.y = clamp(toY(py));
  S.drag.p.pulse = performance.now();
  touch();
});

function endDrag(e) {
  if (S.drag) {
    S.drag.p.pulse = performance.now();
    S.drag = null;
    touch();
  }
  if (e && plane.hasPointerCapture && plane.hasPointerCapture(e.pointerId)) plane.releasePointerCapture(e.pointerId);
}
plane.addEventListener('pointerup', endDrag);
plane.addEventListener('pointercancel', endDrag);

// Right click deletes the nearest point on desktop.
plane.addEventListener('contextmenu', (e) => {
  e.preventDefault();
  const [px, py] = localPoint(e);
  const hit = nearest(px, py, 16);
  if (hit) removePoint(hit);
});

// ---------------------------------------------------------------- controls

function setMode(mode) {
  S.mode = mode;
  document.body.dataset.mode = mode;
  $('mode-reg').setAttribute('aria-pressed', String(mode === 'reg'));
  $('mode-log').setAttribute('aria-pressed', String(mode === 'log'));
  const list = mode === 'reg' ? REG_PRESETS : LOG_PRESETS;
  const cfg = mode === 'reg' ? S.reg : S.log;
  $('preset').innerHTML = list.map((p) => `<option value="${p}">${p}</option>`).join('');
  $('preset').value = cfg.preset;
  // Logistic regression always descends; linear regression descends only when the method says so.
  S.running = mode === 'log' ? true : S.reg.method === 'gd';
  loadData(S.seed);
}

function setClass(c) {
  S.log.cls = c;
  $('cls0').setAttribute('aria-pressed', String(c === 0));
  $('cls1').setAttribute('aria-pressed', String(c === 1));
}

function bindControls() {
  $('mode-reg').addEventListener('click', () => setMode('reg'));
  $('mode-log').addEventListener('click', () => setMode('log'));
  $('preset').addEventListener('change', (e) => {
    (S.mode === 'reg' ? S.reg : S.log).preset = e.target.value;
    loadData(S.seed);
  });
  $('btn-new').addEventListener('click', () => loadData(Math.floor(Math.random() * 900000) + 1));
  $('btn-undo').addEventListener('click', () => {
    if (!S.history.length) return;
    S.pts = JSON.parse(S.history.pop());
    touch();
  });
  $('btn-clear').addEventListener('click', () => {
    snapshot();
    S.pts = [];
    resetGD();
  });
  $('degree').addEventListener('input', (e) => {
    S.reg.degree = +e.target.value;
    $('degree-val').textContent = e.target.value;
    S.lastSSE = null;
    resetGD();
  });
  $('ldeg').addEventListener('input', (e) => {
    S.log.degree = +e.target.value;
    $('ldeg-val').textContent = e.target.value;
    resetGD();
  });
  $('method').addEventListener('change', (e) => {
    S.reg.method = e.target.value;
    S.running = S.reg.method === 'gd';
    S.lastSSE = null;
    touch();
  });
  $('holdout').addEventListener('change', (e) => { S.reg.holdout = e.target.checked; touch(); });
  $('ridge').addEventListener('input', (e) => {
    S.ridge = +e.target.value;
    $('ridge-val').textContent = S.ridge.toFixed(2);
    touch();
  });
  $('lr').addEventListener('input', (e) => {
    S.lr = Math.pow(10, +e.target.value);
    $('lr-val').textContent = S.lr.toPrecision(2);
  });
  $('steps').addEventListener('input', (e) => {
    S.steps = +e.target.value;
    $('steps-val').textContent = e.target.value;
  });
  $('gd-run').addEventListener('click', () => {
    if (S.gd.diverged) return;
    S.running = !S.running;
    touch();
  });
  $('gd-reset').addEventListener('click', () => {
    S.running = S.mode === 'log' ? true : S.reg.method === 'gd';
    resetGD();
  });
  $('cls0').addEventListener('click', () => setClass(0));
  $('cls1').addEventListener('click', () => setClass(1));
}

function initControls() {
  $('degree').value = S.reg.degree;
  $('degree-val').textContent = S.reg.degree;
  $('ldeg').value = S.log.degree;
  $('ldeg-val').textContent = S.log.degree;
  $('method').value = S.reg.method;
  $('ridge').value = S.ridge;
  $('ridge-val').textContent = S.ridge.toFixed(2);
  $('lr').value = Math.log10(S.lr);
  $('lr-val').textContent = S.lr.toPrecision(2);
  $('steps').value = S.steps;
  $('steps-val').textContent = S.steps;
  setClass(S.log.cls);
}

// ---------------------------------------------------------------- boot

function boot() {
  // The plane is drawn at device resolution; all coordinates stay in the logical W x H space.
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  plane.width = W * dpr;
  plane.height = H * dpr;
  pctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  bindControls();
  initControls();
  setMode('reg');
  requestAnimationFrame(frame);
}

boot();
