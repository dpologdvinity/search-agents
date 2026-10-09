// Optimizer race page. The maths is in optlab-core.js (no DOM, shared with the node parity test); this file
// draws the landscape as a neon heat map with contour lines (or a tilted wireframe), runs the race on the
// animation frame, and wires the controls. Effects come from fx.js: bursts and banners for divergence,
// floating labels for settling.

import {
  CUSTOM_DEFAULT_BUMPS, CUSTOM_START, DEFAULT_HYPER, DEFAULT_LR, LABELS, MAX_BUMPS, NAMES, Race, customSurface,
  getSurface,
} from './optlab-core.js';
import { banner, burst, pop, shake } from './fx.js';

const $ = (id) => document.getElementById(id);
const REDUCED = matchMedia('(prefers-reduced-motion: reduce)').matches;
const COLORS = {
  sgd: '#00f5ff', momentum: '#00ff88', nesterov: '#ffe600', rmsprop: '#ff00a0', adam: '#b56bff', adagrad: '#ff8a3d',
};
const LETTER = { sgd: 'S', momentum: 'M', nesterov: 'N', rmsprop: 'R', adam: 'A', adagrad: 'D' };
const TRAIL = 240; // points drawn per comet trail
const GRID = 220; // heat map resolution along the longer side
const MESH = 24; // cells per side of the tilted wireframe
const MAX_STEPS = 3000; // the race pauses here
const PALETTE = [[0, [0, 255, 225]], [0.18, [0, 170, 255]], [0.42, [95, 20, 230]], [0.7, [55, 0, 115]], [1, [6, 2, 18]]];
// Tilted view: the base plane's centre height and the horizontal, depth and height scales, as fractions of the frame.
const TILT_X = 0.42;
const TILT_Y = 0.62;
const TILT_V = 0.28;
const TILT_Z = 0.3;
const CHART_MIN = -10; // log10 of the gap to the minimum, bottom of the chart
const CHART_MAX = 4;
const GREEN = '#00ff88';
const PINK = '#ff00a0';

const state = {
  surfaceName: 'rosenbrock',
  surface: null,
  bumps: CUSTOM_DEFAULT_BUMPS.map((b) => [...b]),
  start: [0, 0],
  enabled: Object.fromEntries(NAMES.map((n) => [n, true])),
  lr: { ...DEFAULT_LR },
  shared: false,
  sharedLr: 0.05,
  hyper: { ...DEFAULT_HYPER },
  noise: 0,
  seed: 7,
  speed: 12,
  running: false,
  acc: 0,
  tilt: false,
  paint: 'start',
  hover: null,
  race: null,
  heat: null,
  fx: [],
  events: new Set(),
  view: { w: 0, h: 0 },
  dpr: 1,
};

const canvas = $('ol-canvas');
const ctx = canvas.getContext('2d');
const chart = $('ol-chart');
const cctx = chart.getContext('2d');

// ---------- small helpers ----------

const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));
const fmt = (v) => (Number.isFinite(v) ? v.toFixed(3) : String(v));
function fmtLoss(v) {
  if (!Number.isFinite(v)) return '—';
  const a = Math.abs(v);
  return a >= 1e4 || (a < 1e-3 && a !== 0) ? v.toExponential(2) : v.toFixed(4);
}
const fmtLr = (lr) => Number(lr.toPrecision(3)).toString();

function log(text, cls = 'log-info') {
  const box = $('ol-log');
  const row = document.createElement('div');
  row.className = cls;
  row.textContent = `> ${text}`;
  box.appendChild(row);
  while (box.childElementCount > 200) box.firstElementChild.remove();
  box.scrollTop = box.scrollHeight;
}

function paletteAt(t) {
  const u = clamp(t, 0, 1);
  for (let i = 1; i < PALETTE.length; i++) {
    const [t1, c1] = PALETTE[i];
    const [t0, c0] = PALETTE[i - 1];
    if (u <= t1) {
      const k = (u - t0) / (t1 - t0);
      return [c0[0] + (c1[0] - c0[0]) * k, c0[1] + (c1[1] - c0[1]) * k, c0[2] + (c1[2] - c0[2]) * k];
    }
  }
  return PALETTE[PALETTE.length - 1][1];
}

// ---------- landscape: heat map and tilted mesh ----------

/** Evaluate the surface on a grid once (after any change of surface or bumps): colours, contours, mesh heights. */
function buildHeat() {
  const s = state.surface;
  const [x0, x1, y0, y1] = s.domain;
  const w = GRID;
  const h = Math.max(8, Math.round(GRID * (y1 - y0) / (x1 - x0)));
  const raw = new Float64Array(w * h);
  for (let r = 0; r < h; r++) {
    const y = y1 - ((r + 0.5) / h) * (y1 - y0);
    for (let c = 0; c < w; c++) raw[r * w + c] = s.f(x0 + ((c + 0.5) / w) * (x1 - x0), y);
  }
  // The floor is the known minimum, or the lowest sampled value for paint-your-own landscapes.
  let floor = s.fmin;
  if (floor === null || floor === undefined) {
    floor = Infinity;
    for (const v of raw) if (v < floor) floor = v;
  }
  const vals = new Float32Array(w * h); // log10 of the loss above the floor (+0.01 keeps it finite)
  let lo = Infinity;
  let hi = -Infinity;
  for (let i = 0; i < raw.length; i++) {
    const v = Math.log10(Math.max(raw[i] - floor, 0) + 0.01);
    vals[i] = v;
    if (v < lo) lo = v;
    if (v > hi) hi = v;
  }
  if (!(hi > lo)) hi = lo + 1;

  const off = document.createElement('canvas');
  off.width = w;
  off.height = h;
  const octx = off.getContext('2d');
  const img = octx.createImageData(w, h);
  for (let i = 0; i < vals.length; i++) {
    let [R, G, B] = paletteAt((vals[i] - lo) / (hi - lo));
    // Contour lines every quarter decade of loss, brightened where the value is close to a multiple of 0.25.
    const k = vals[i] * 4;
    const frac = k - Math.floor(k);
    if (Math.min(frac, 1 - frac) < 0.04) {
      R += (255 - R) * 0.35;
      G += (255 - G) * 0.35;
      B += (255 - B) * 0.35;
    }
    img.data[4 * i] = R;
    img.data[4 * i + 1] = G;
    img.data[4 * i + 2] = B;
    img.data[4 * i + 3] = 255;
  }
  octx.putImageData(img, 0, 0);

  // Mesh heights for the tilted view, on a coarser grid, from the same normalisation.
  const z = new Float64Array((MESH + 1) * (MESH + 1));
  for (let j = 0; j <= MESH; j++) {
    for (let i = 0; i <= MESH; i++) {
      const x = x0 + (i / MESH) * (x1 - x0);
      const y = y0 + (j / MESH) * (y1 - y0);
      z[j * (MESH + 1) + i] = heightOf(s.f(x, y), { floor, lo, hi });
    }
  }
  state.heat = { w, h, lo, hi, floor, canvas: off, z };
}

/** Normalised height in [0, 1] for a loss value: 0 at the valley floor, 1 at the highest loss on the grid. */
function heightOf(f, heat = state.heat) {
  const v = Math.log10(Math.max(f - heat.floor, 0) + 0.01);
  return clamp((v - heat.lo) / (heat.hi - heat.lo), 0, 1);
}

/** World point to screen pixels. In the tilted view, z lifts the point off the base plane. */
function proj(x, y, z = 0) {
  const [x0, x1, y0, y1] = state.surface.domain;
  const { w, h } = state.view;
  if (!state.tilt) return [((x - x0) / (x1 - x0)) * w, ((y1 - y) / (y1 - y0)) * h];
  const u = ((x - x0) / (x1 - x0)) * 2 - 1;
  const v = ((y - y0) / (y1 - y0)) * 2 - 1;
  return [w / 2 + u * w * TILT_X, h * TILT_Y - v * h * TILT_V - z * h * TILT_Z];
}

/** Screen pixels to world, on the base plane in the tilted view. Null outside the domain. */
function toWorld(px, py) {
  const [x0, x1, y0, y1] = state.surface.domain;
  const { w, h } = state.view;
  let x;
  let y;
  if (state.tilt) {
    const u = (px - w / 2) / (w * TILT_X);
    const v = (h * TILT_Y - py) / (h * TILT_V);
    x = x0 + ((u + 1) / 2) * (x1 - x0);
    y = y0 + ((v + 1) / 2) * (y1 - y0);
  } else {
    x = x0 + (px / w) * (x1 - x0);
    y = y1 - (py / h) * (y1 - y0);
  }
  if (x < x0 || x > x1 || y < y0 || y > y1) return null;
  return [x, y];
}

function localPx(e) {
  const r = canvas.getBoundingClientRect();
  return [(e.clientX - r.left) * (state.view.w / r.width), (e.clientY - r.top) * (state.view.h / r.height)];
}

function viewportOf(world) {
  const r = canvas.getBoundingClientRect();
  const [px, py] = proj(world[0], world[1], heightOf(state.surface.f(world[0], world[1])));
  return [r.left + px * (r.width / state.view.w), r.top + py * (r.height / state.view.h)];
}

function resize() {
  const frame = $('ol-frame');
  const [x0, x1, y0, y1] = state.surface.domain;
  const avail = Math.max(200, frame.clientWidth - 18);
  const w = Math.min(avail, 720);
  const h = Math.round(w * (y1 - y0) / (x1 - x0));
  state.dpr = Math.min(window.devicePixelRatio || 1, 2);
  state.view = { w, h };
  canvas.style.width = `${w}px`;
  canvas.style.height = `${h}px`;
  canvas.width = Math.round(w * state.dpr);
  canvas.height = Math.round(h * state.dpr);
  const cw = chart.clientWidth || 600;
  chart.width = Math.round(cw * state.dpr);
  chart.height = Math.round(180 * state.dpr);
}

// ---------- drawing ----------

function drawArrow(from, to, color, label) {
  ctx.strokeStyle = color;
  ctx.fillStyle = color;
  ctx.lineWidth = 2;
  ctx.shadowColor = color;
  ctx.shadowBlur = 10;
  ctx.beginPath();
  ctx.moveTo(from[0], from[1]);
  ctx.lineTo(to[0], to[1]);
  ctx.stroke();
  const ang = Math.atan2(to[1] - from[1], to[0] - from[0]);
  ctx.beginPath();
  ctx.moveTo(to[0], to[1]);
  ctx.lineTo(to[0] - 9 * Math.cos(ang - 0.4), to[1] - 9 * Math.sin(ang - 0.4));
  ctx.lineTo(to[0] - 9 * Math.cos(ang + 0.4), to[1] - 9 * Math.sin(ang + 0.4));
  ctx.closePath();
  ctx.fill();
  ctx.shadowBlur = 0;
  if (label) {
    ctx.font = "11px 'JetBrains Mono', monospace";
    ctx.fillText(label, to[0] + 8, to[1] + 4);
  }
}

/** Wireframe of the landscape with each edge coloured by its height (the tilted view). */
function drawMesh() {
  const [x0, x1, y0, y1] = state.surface.domain;
  const z = state.heat.z;
  const stride = MESH + 1;
  const wx = (i) => x0 + (i / MESH) * (x1 - x0);
  const wy = (j) => y0 + (j / MESH) * (y1 - y0);
  ctx.lineWidth = 1;
  const edge = (a, b, zz) => {
    const [R, G, B] = paletteAt(zz);
    ctx.strokeStyle = `rgba(${R | 0},${G | 0},${B | 0},0.7)`;
    ctx.beginPath();
    ctx.moveTo(a[0], a[1]);
    ctx.lineTo(b[0], b[1]);
    ctx.stroke();
  };
  for (let j = 0; j <= MESH; j++) {
    for (let i = 0; i <= MESH; i++) {
      const zi = z[j * stride + i];
      const p = proj(wx(i), wy(j), zi);
      if (i < MESH) edge(p, proj(wx(i + 1), wy(j), z[j * stride + i + 1]), zi);
      if (j < MESH) edge(p, proj(wx(i), wy(j + 1), z[(j + 1) * stride + i]), zi);
    }
  }
}

function drawTrails() {
  for (const tr of state.race.tracks) {
    const pts = tr.traj;
    const n = pts.length;
    if (n < 2) continue;
    const color = COLORS[tr.name];
    const from = Math.max(0, n - TRAIL);
    ctx.strokeStyle = color;
    ctx.lineCap = 'round';
    for (let i = from + 1; i < n; i++) {
      const k = (i - from) / (n - from - 1); // 0 at the tail, 1 at the head
      const a = proj(pts[i - 1][0], pts[i - 1][1], heightOf(tr.losses[i - 1]));
      const b = proj(pts[i][0], pts[i][1], heightOf(tr.losses[i]));
      ctx.globalAlpha = 0.08 + 0.92 * k * k;
      ctx.lineWidth = 0.8 + 2.6 * k;
      ctx.shadowBlur = i > n - 25 ? 12 : 0;
      ctx.shadowColor = color;
      ctx.beginPath();
      ctx.moveTo(a[0], a[1]);
      ctx.lineTo(b[0], b[1]);
      ctx.stroke();
    }
  }
  ctx.globalAlpha = 1;
  ctx.shadowBlur = 0;
}

function drawHeads(now) {
  for (const tr of state.race.tracks) {
    const n = tr.traj.length;
    const last = tr.traj[n - 1];
    const lossLast = tr.losses[n - 1];
    const [px, py] = proj(last[0], last[1], heightOf(lossLast));
    const color = COLORS[tr.name];
    if (tr.divergedAt !== null) {
      ctx.strokeStyle = PINK;
      ctx.lineWidth = 2;
      ctx.shadowColor = PINK;
      ctx.shadowBlur = 10;
      ctx.beginPath();
      ctx.moveTo(px - 6, py - 6);
      ctx.lineTo(px + 6, py + 6);
      ctx.moveTo(px + 6, py - 6);
      ctx.lineTo(px - 6, py + 6);
      ctx.stroke();
      ctx.shadowBlur = 0;
      continue;
    }
    if (tr.stepsToTol !== null) {
      const pulse = REDUCED ? 0 : (Math.sin(now / 240) + 1) * 2;
      ctx.strokeStyle = GREEN;
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.arc(px, py, 9 + pulse, 0, Math.PI * 2);
      ctx.stroke();
    }
    ctx.fillStyle = color;
    ctx.shadowColor = color;
    ctx.shadowBlur = 14;
    ctx.beginPath();
    ctx.arc(px, py, 4.5, 0, Math.PI * 2);
    ctx.fill();
    ctx.shadowBlur = 0;
    ctx.fillStyle = color;
    ctx.font = "bold 10px 'JetBrains Mono', monospace";
    ctx.fillText(LETTER[tr.name], px + 8, py - 7);
  }
}

function drawStart() {
  const [sx, sy] = state.start;
  const [px, py] = proj(sx, sy, heightOf(state.surface.f(sx, sy)));
  ctx.strokeStyle = 'rgba(255,255,255,0.85)';
  ctx.lineWidth = 1;
  ctx.setLineDash([4, 4]);
  ctx.beginPath();
  ctx.arc(px, py, 12, 0, Math.PI * 2);
  ctx.stroke();
  ctx.setLineDash([]);
  ctx.fillStyle = 'rgba(255,255,255,0.9)';
  ctx.font = "bold 10px 'JetBrains Mono', monospace";
  ctx.fillText('o', px - 3, py + 3);
}

function drawMinima() {
  const s = state.surface;
  ctx.strokeStyle = 'rgba(255,255,255,0.7)';
  ctx.lineWidth = 1;
  for (const [mx, my] of s.minima) {
    const [px, py] = proj(mx, my, heightOf(s.f(mx, my)));
    ctx.beginPath();
    ctx.moveTo(px - 5, py);
    ctx.lineTo(px + 5, py);
    ctx.moveTo(px, py - 5);
    ctx.lineTo(px, py + 5);
    ctx.stroke();
  }
}

function drawBumps() {
  const s = state.surface;
  const [x0, x1] = s.domain;
  const pxPerX = state.view.w / (x1 - x0);
  for (const [cx, cy, a, sig] of state.bumps) {
    const z = heightOf(s.f(cx, cy));
    const [px, py] = proj(cx, cy, z);
    const color = a > 0 ? PINK : '#00f5ff';
    ctx.strokeStyle = color;
    ctx.globalAlpha = 0.5;
    ctx.setLineDash([3, 4]);
    ctx.beginPath();
    ctx.arc(px, py, sig * pxPerX, 0, Math.PI * 2);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.globalAlpha = 1;
    ctx.fillStyle = color;
    ctx.font = "bold 12px 'JetBrains Mono', monospace";
    ctx.fillText(a > 0 ? '+' : '−', px - 4, py + 4);
  }
}

function drawHover() {
  if (!state.hover) return;
  const [x, y] = state.hover;
  const s = state.surface;
  const [gx, gy] = s.grad(x, y);
  const gn = Math.hypot(gx, gy);
  if (!(gn > 0) || !Number.isFinite(gn)) return;
  const [x0, x1, y0, y1] = s.domain;
  const [px, py] = proj(x, y, heightOf(s.f(x, y)));
  // Gradient in screen space (y points down), scaled so steep slopes give longer arrows, capped near the view.
  const sx = gx * (state.view.w / (x1 - x0));
  const sy = -gy * (state.view.h / (y1 - y0));
  const sn = Math.hypot(sx, sy);
  const len = 18 + (70 * gn) / (1 + gn);
  drawArrow([px, py], [px + (sx / sn) * len, py + (sy / sn) * len], PINK, '∇f');
}

function drawFx(now) {
  state.fx = state.fx.filter((e) => now - e.t0 < e.dur);
  for (const e of state.fx) {
    // The frame time can be a little earlier than the time the effect started, so clamp the age to [0, 1].
    const age = clamp((now - e.t0) / e.dur, 0, 1);
    const [px, py] = proj(e.x, e.y, heightOf(state.surface.f(e.x, e.y)));
    ctx.strokeStyle = e.color;
    ctx.globalAlpha = 1 - age;
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.arc(px, py, 6 + 60 * age, 0, Math.PI * 2);
    ctx.stroke();
  }
  ctx.globalAlpha = 1;
}

function drawMain(now) {
  const { w, h } = state.view;
  ctx.setTransform(state.dpr, 0, 0, state.dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = '#020610';
  ctx.fillRect(0, 0, w, h);
  if (state.tilt) drawMesh();
  else ctx.drawImage(state.heat.canvas, 0, 0, w, h);
  drawMinima();
  if (state.surfaceName === 'custom') drawBumps();
  if (state.race) {
    drawTrails();
    drawHeads(now);
  }
  drawStart();
  drawHover();
  drawFx(now);
}

function drawChart() {
  const { width: W, height: H } = chart;
  const d = state.dpr;
  cctx.setTransform(d, 0, 0, d, 0, 0);
  const w = W / d;
  const h = H / d;
  cctx.clearRect(0, 0, w, h);
  const padL = 34;
  const padR = 8;
  const padT = 8;
  const padB = 18;
  const race = state.race;
  const steps = race ? race.t : 0;
  const xmax = Math.max(100, steps * 1.05);
  const yOf = (lg) => padT + ((CHART_MAX - clamp(lg, CHART_MIN, CHART_MAX)) / (CHART_MAX - CHART_MIN)) * (h - padT - padB);
  const xOf = (s) => padL + (s / xmax) * (w - padL - padR);
  cctx.font = "9px 'JetBrains Mono', monospace";
  cctx.textBaseline = 'middle';
  for (let lg = CHART_MIN; lg <= CHART_MAX; lg += 2) {
    const y = yOf(lg);
    cctx.strokeStyle = 'rgba(0,245,255,0.12)';
    cctx.lineWidth = 1;
    cctx.beginPath();
    cctx.moveTo(padL, y);
    cctx.lineTo(w - padR, y);
    cctx.stroke();
    cctx.fillStyle = 'rgba(74,122,155,1)';
    cctx.fillText(`1e${lg}`, 2, y);
  }
  cctx.fillStyle = 'rgba(74,122,155,1)';
  cctx.fillText('step 0', padL, h - 6);
  cctx.textAlign = 'right';
  cctx.fillText(`${Math.round(xmax)}`, w - padR, h - 6);
  cctx.textAlign = 'left';
  if (!race) return;
  const floor = gapFloor();
  for (const tr of race.tracks) {
    if (tr.losses.length < 2) continue;
    cctx.strokeStyle = COLORS[tr.name];
    cctx.lineWidth = 1.6;
    cctx.shadowColor = COLORS[tr.name];
    cctx.shadowBlur = 6;
    cctx.beginPath();
    tr.losses.forEach((loss, i) => {
      const gap = Math.max(loss - floor, 1e-12);
      const x = xOf(i);
      const y = yOf(Math.log10(gap));
      if (i === 0) cctx.moveTo(x, y);
      else cctx.lineTo(x, y);
    });
    cctx.stroke();
    cctx.shadowBlur = 0;
    if (tr.divergedAt !== null) {
      const gap = Math.max(tr.losses[tr.losses.length - 1] - floor, 1e-12);
      cctx.fillStyle = PINK;
      cctx.fillText('✗', xOf(tr.losses.length - 1) - 4, yOf(Math.log10(gap)));
    }
  }
}

/** The chart's zero: the known minimum, or the lowest loss any track has reached. */
function gapFloor() {
  const s = state.surface;
  if (s.fmin !== null && s.fmin !== undefined) return s.fmin;
  let m = Infinity;
  for (const tr of state.race.tracks) for (const v of tr.losses) if (v < m) m = v;
  return Number.isFinite(m) ? m : 0;
}

function renderLegend() {
  const box = $('ol-legend');
  box.innerHTML = '';
  if (!state.race) return;
  for (const tr of state.race.tracks) {
    const item = document.createElement('span');
    item.innerHTML = `<i class="ol-sw" style="background:${COLORS[tr.name]};color:${COLORS[tr.name]}"></i>${LABELS[tr.name]}`;
    box.appendChild(item);
  }
}

// ---------- race state, events and stats ----------

function buildRace() {
  const names = NAMES.filter((n) => state.enabled[n]);
  const lrs = {};
  for (const n of names) lrs[n] = state.shared ? state.sharedLr : state.lr[n];
  return new Race({
    surface: state.surface, start: state.start, names, lrs, hyper: { ...state.hyper }, noise: state.noise,
    seed: state.seed,
  });
}

function resetRace() {
  state.race = buildRace();
  state.fx = [];
  state.events = new Set();
  state.acc = 0;
  renderLegend();
  const [sx, sy] = state.start;
  log(`new race on ${state.surface.label} from (${fmt(sx)}, ${fmt(sy)}), seed ${state.seed}, σ ${state.noise.toFixed(2)}`);
  if (!state.race.tracks.length) log('no optimizer is enabled', 'log-err');
  updateBadges();
}

const allDone = () => state.race.tracks.length > 0
  && state.race.tracks.every((t) => t.divergedAt !== null || t.stepsToTol !== null);

function addFx(world, color, dur) {
  if (REDUCED) return;
  state.fx.push({ x: world[0], y: world[1], color, dur, t0: performance.now() });
}

function processEvents() {
  const race = state.race;
  for (const tr of race.tracks) {
    const n = tr.traj.length;
    const last = tr.traj[n - 1];
    const lossLast = tr.losses[n - 1];
    if (tr.stepsToTol !== null && !state.events.has(`ok:${tr.name}`)) {
      state.events.add(`ok:${tr.name}`);
      log(`${LABELS[tr.name]} settled at a minimum after ${tr.stepsToTol} steps (loss ${fmtLoss(lossLast)})`, 'log-best');
      addFx(last, GREEN, 900);
      if (!REDUCED) {
        const [vx, vy] = viewportOf(last);
        pop([vx, vy - 14], '✓ SETTLED', GREEN);
      }
    }
    if (tr.divergedAt !== null && !state.events.has(`boom:${tr.name}`)) {
      state.events.add(`boom:${tr.name}`);
      // A finite position that ran past the box is a BLOW-UP; only a non-finite value is a NaN DETONATION.
      const title = tr.divergedBy === 'blowup' ? 'BLOW-UP' : 'NaN DETONATION';
      log(`${LABELS[tr.name]} diverged at step ${tr.divergedAt}: ${title}`, 'log-err');
      addFx(last, PINK, 1100);
      const [vx, vy] = viewportOf(last);
      burst([vx, vy], { count: 90, colors: [PINK, COLORS[tr.name], '#ffe600', '#ffffff'], speed: 9 });
      shake($('ol-frame'), 'big');
      banner(title, `${LABELS[tr.name]} diverged at step ${tr.divergedAt}`, PINK);
    }
  }
  if (allDone() && !state.events.has('done')) {
    state.events.add('done');
    setRunning(false);
    log('every track has settled or diverged; race paused', 'log-info');
  }
}

function advance() {
  state.race.step();
  processEvents();
}

function setRunning(on) {
  state.running = on;
  $('ol-run').querySelector('.btn-txt').textContent = on ? '❚❚ PAUSE' : '▶ RUN RACE';
}

const BADGE_TEXT = {
  off: ['OFF', ''],
  run: ['RUNNING', 'run'],
  ok: ['SETTLED ✓', 'ok'],
  boom: ['DIVERGED ✗', 'boom'],
};

function updateBadges() {
  for (const n of NAMES) {
    const row = document.querySelector(`.ol-opt[data-name="${n}"]`);
    if (!row) continue;
    const badge = row.querySelector('[data-role="badge"]');
    const tr = state.race && state.race.tracks.find((t) => t.name === n);
    let key = 'off';
    let text = BADGE_TEXT.off[0];
    if (tr) {
      if (tr.divergedAt !== null) {
        key = 'boom';
        text = `DIVERGED ✗ @${tr.divergedAt}`;
      } else if (tr.stepsToTol !== null) {
        key = 'ok';
        text = `SETTLED ✓ @${tr.stepsToTol}`;
      } else {
        key = 'run';
        text = BADGE_TEXT.run[0];
      }
    }
    const cls = `ol-badge ${BADGE_TEXT[key][1]}`.trim();
    if (badge.dataset.key !== key || badge.dataset.text !== text) {
      badge.dataset.key = key;
      badge.dataset.text = text;
      badge.textContent = text;
      badge.className = cls;
    }
  }
}

function updateStats() {
  const race = state.race;
  if (!race) return;
  $('chip-step').textContent = String(race.t);
  let best = Infinity;
  let settled = 0;
  for (const tr of race.tracks) {
    const l = tr.losses[tr.losses.length - 1];
    if (tr.divergedAt === null && l < best) best = l;
    if (tr.stepsToTol !== null) settled += 1;
  }
  $('chip-best').textContent = Number.isFinite(best) ? fmtLoss(best) : '—';
  $('chip-reached').textContent = `${settled}/${race.tracks.length}`;
  updateBadges();
}

// ---------- the animation loop ----------

let lastNow = performance.now();

function frame(now) {
  requestAnimationFrame(frame); // schedule first, so an error in one frame cannot stop the loop
  const dt = Math.min(0.1, Math.max(0, (now - lastNow) / 1000));
  lastNow = now;
  if (state.running && state.race) {
    state.acc += state.speed * dt;
    let budget = 40;
    while (state.acc >= 1 && state.running && budget-- > 0) {
      advance();
      state.acc -= 1;
      if (state.race.t >= MAX_STEPS) {
        setRunning(false);
        log(`stopped at ${MAX_STEPS} steps; press RESET to race again`);
      }
    }
  }
  if (state.race) {
    drawMain(now);
    drawChart();
    updateStats();
  }
}

// ---------- controls ----------

function setSurface(name) {
  state.surfaceName = name;
  state.surface = name === 'custom' ? customSurface(state.bumps) : getSurface(name);
  state.start = [...(name === 'custom' ? CUSTOM_START : state.surface.start)];
  $('chip-surface').textContent = state.surface.label.toUpperCase();
  $('ol-note').textContent = state.surface.note;
  $('ol-paint-row').hidden = name !== 'custom';
  $('ol-hint').textContent = name === 'custom'
    ? 'Pick HILL or VALLEY, then click the map to paint a bump. Click START to move the start point.'
    : 'Click the map to drop a start point. Hover to read the gradient at the cursor.';
  updateBumpsLabel();
  buildHeat();
  resize();
  resetRace();
}

function updateBumpsLabel() {
  $('ol-bumps').textContent = `Bumps: ${state.bumps.length} of ${MAX_BUMPS}`;
}

function rebuildCustom() {
  state.surface = customSurface(state.bumps);
  updateBumpsLabel();
  buildHeat();
  resetRace();
}

function addBump(world, kind) {
  if (state.bumps.length >= MAX_BUMPS) {
    log(`at most ${MAX_BUMPS} bumps; press UNDO or CLEAR`, 'log-err');
    return;
  }
  state.bumps.push([world[0], world[1], kind === 'hill' ? 2.4 : -2.0, 0.55]);
  log(`${kind === 'hill' ? 'hill' : 'valley'} painted at (${fmt(world[0])}, ${fmt(world[1])})`);
  rebuildCustom();
}

/** Apply a changed hyperparameter to the live race without restarting it. */
function applyHyper() {
  if (!state.race) return;
  for (const tr of state.race.tracks) tr.opt.h = { ...state.hyper };
}

function applyLr() {
  if (!state.race) return;
  for (const tr of state.race.tracks) {
    const lr = state.shared ? state.sharedLr : state.lr[tr.name];
    tr.opt.lr = lr;
    tr.lr = lr;
  }
}

function renderRows() {
  $('ol-opts').innerHTML = NAMES.map((n) => `
    <div class="ol-opt" data-name="${n}">
      <label class="ol-opt-name"><input type="checkbox" data-role="on" checked aria-label="race ${LABELS[n]}"><i class="ol-sw" style="background:${COLORS[n]};color:${COLORS[n]}"></i>${LABELS[n]}</label>
      <span class="ol-badge" data-role="badge">OFF</span>
      <div class="ol-opt-lr">
        <input type="range" min="-4" max="0.7" step="0.01" value="${Math.log10(DEFAULT_LR[n]).toFixed(2)}" data-role="lr" aria-label="${LABELS[n]} learning rate">
        <span class="ol-lr-val" data-role="lrval">${fmtLr(DEFAULT_LR[n])}</span>
      </div>
    </div>`).join('');
}

function bindControls() {
  renderRows();
  const rows = $('ol-opts');
  rows.addEventListener('change', (e) => {
    const row = e.target.closest('.ol-opt');
    if (!row || e.target.dataset.role !== 'on') return;
    const n = row.dataset.name;
    state.enabled[n] = e.target.checked;
    row.classList.toggle('off', !e.target.checked);
    resetRace();
  });
  rows.addEventListener('input', (e) => {
    const row = e.target.closest('.ol-opt');
    if (!row || e.target.dataset.role !== 'lr') return;
    const n = row.dataset.name;
    const lr = Number(10 ** Number(e.target.value)) ;
    state.lr[n] = Number(lr.toPrecision(3));
    row.querySelector('[data-role="lrval"]').textContent = fmtLr(state.lr[n]);
    applyLr();
  });

  $('ol-shared').addEventListener('change', (e) => {
    state.shared = e.target.checked;
    $('ol-shared-lr-row').hidden = !state.shared;
    rows.classList.toggle('ol-shared', state.shared);
    applyLr();
  });
  $('ol-shared-lr').addEventListener('input', (e) => {
    state.sharedLr = Number((10 ** Number(e.target.value)).toPrecision(3));
    $('ol-shared-lr-val').textContent = `= ${fmtLr(state.sharedLr)}`;
    applyLr();
  });
  $('ol-shared-lr').value = Math.log10(state.sharedLr).toFixed(2);
  $('ol-shared-lr-val').textContent = `= ${fmtLr(state.sharedLr)}`;

  const slider = (id, key, digits, label) => {
    const el = $(id);
    el.value = String(state.hyper[key]);
    const show = () => { $(`${id}-val`).textContent = label(state.hyper[key]); };
    el.addEventListener('input', () => {
      state.hyper[key] = Number(Number(el.value).toFixed(digits));
      show();
      applyHyper();
    });
    show();
  };
  slider('ol-momentum', 'momentum', 2, (v) => `β = ${v.toFixed(2)}`);
  slider('ol-beta1', 'beta1', 3, (v) => `β₁ = ${v.toFixed(3)}`);
  slider('ol-beta2', 'beta2', 4, (v) => `β₂ = ${v.toFixed(4)}`);
  slider('ol-rho', 'rho', 3, (v) => `ρ = ${v.toFixed(3)}`);
  const epsEl = $('ol-eps');
  epsEl.value = String(Math.round(Math.log10(state.hyper.eps)));
  const showEps = () => { $('ol-eps-val').textContent = `ε = 1e${Math.round(Math.log10(state.hyper.eps))}`; };
  epsEl.addEventListener('input', () => {
    state.hyper.eps = 10 ** Number(epsEl.value);
    showEps();
    applyHyper();
  });
  showEps();

  const noiseEl = $('ol-noise');
  const showNoise = () => { $('ol-noise-val').textContent = `σ = ${state.noise.toFixed(2)}`; };
  noiseEl.addEventListener('input', () => {
    state.noise = Number(noiseEl.value);
    showNoise();
    if (state.race) state.race.noise = state.noise;
  });
  showNoise();

  const speedEl = $('ol-speed');
  speedEl.value = String(state.speed);
  const showSpeed = () => { $('ol-speed-val').textContent = `${state.speed} steps/s`; };
  speedEl.addEventListener('input', () => {
    state.speed = Number(speedEl.value);
    showSpeed();
  });
  showSpeed();

  const seedEl = $('ol-seed');
  seedEl.addEventListener('change', () => {
    const v = Math.max(0, Math.min(2147483647, Math.floor(Number(seedEl.value) || 0)));
    seedEl.value = String(v);
    state.seed = v;
    resetRace();
  });
  $('ol-seed-rnd').addEventListener('click', () => {
    state.seed = Math.floor(Math.random() * 2147483647);
    seedEl.value = String(state.seed);
    resetRace();
  });

  $('ol-surface').addEventListener('change', (e) => setSurface(e.target.value));
  for (const r of document.querySelectorAll('input[name="ol-paint"]')) {
    r.addEventListener('change', () => { if (r.checked) state.paint = r.value; });
  }
  $('ol-undo').addEventListener('click', () => {
    if (!state.bumps.length) return;
    state.bumps.pop();
    log('last bump removed');
    rebuildCustom();
  });
  $('ol-clear').addEventListener('click', () => {
    state.bumps = [];
    log('all bumps cleared');
    rebuildCustom();
  });

  $('ol-run').addEventListener('click', () => setRunning(!state.running));
  $('ol-step').addEventListener('click', () => {
    if (!state.race) return;
    setRunning(false);
    advance();
  });
  $('ol-reset').addEventListener('click', () => {
    setRunning(false);
    resetRace();
  });
  $('ol-view').addEventListener('click', (e) => {
    state.tilt = !state.tilt;
    e.currentTarget.querySelector('.btn-txt').textContent = state.tilt ? 'VIEW: 3D' : 'VIEW: 2D';
    e.currentTarget.setAttribute('aria-pressed', String(state.tilt));
    $('ol-hint').textContent = state.tilt
      ? 'Tilted view: the height is the loss. Clicks land on the base plane.'
      : 'Click the map to drop a start point. Hover to read the gradient at the cursor.';
  });

  canvas.addEventListener('pointermove', (e) => {
    const [px, py] = localPx(e);
    const w = toWorld(px, py);
    state.hover = w;
    const readout = $('ol-readout');
    if (!w) {
      readout.textContent = 'hover the map';
      return;
    }
    const [x, y] = w;
    const f = state.surface.f(x, y);
    const [gx, gy] = state.surface.grad(x, y);
    readout.textContent = `x ${fmt(x)}  y ${fmt(y)}  f ${fmtLoss(f)}  ∇f (${fmt(gx)}, ${fmt(gy)})`;
  });
  canvas.addEventListener('pointerleave', () => {
    state.hover = null;
    $('ol-readout').textContent = 'hover the map';
  });
  canvas.addEventListener('click', (e) => {
    const [px, py] = localPx(e);
    const w = toWorld(px, py);
    if (!w) return;
    if (state.surfaceName === 'custom' && state.paint !== 'start') {
      addBump(w, state.paint);
      return;
    }
    state.start = w;
    log(`start moved to (${fmt(w[0])}, ${fmt(w[1])})`);
    resetRace();
  });

  document.addEventListener('keydown', (e) => {
    if (e.key !== ' ' || e.target.closest('input, select, textarea, button')) return;
    e.preventDefault();
    setRunning(!state.running);
  });

  const frameEl = $('ol-frame');
  if ('ResizeObserver' in window) new ResizeObserver(() => resize()).observe(frameEl);
  else addEventListener('resize', resize);
}

function init() {
  bindControls();
  $('ol-shared-lr-row').hidden = true;
  setSurface(state.surfaceName);
  log('the race is ready: press RUN, or STEP to advance one step at a time');
  requestAnimationFrame(frame);
}

init();
