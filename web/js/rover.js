// Rover page.
//
// The planners and the rover loop run here, in rover_core.js, so the page needs no server round trips.
// This file draws the map and the chart, runs the timer that advances the rover, and turns clicks into
// wall edits. The text in "What the rover is thinking" is written from the same numbers the planners
// report: the expansions each one spent on a replan and the cost each one reports from the rover's cell.
//
// Cell colours: unseen cells are dark, seen-free cells are faint cyan, walls found by the sensor are pink.
// A pink dashed outline marks a wall you dropped that the sensor has not reached yet. The heat map shows
// D* Lite's g or rhs for every cell. Green outlines show the cells the last replan expanded.

import { DStarLite, Explorer, INF, generate } from './rover_core.js';
import { banner } from './fx.js';
import { expand } from './sfx.js';

const $ = (id) => document.getElementById(id);
const COL = { cyan: '#00f5ff', pink: '#ff00a0', yellow: '#ffe600', green: '#00ff88', white: '#e8fdff' };
const FLASH_MS = 1400; // how long the green expansion outline stays visible
const CHART_POINTS = 4000;

const state = {
  size: 21, density: 0.2, seed: 7, radius: 2,
  driver: 'dstar', view: 'dstar', heat: 'none', delay: 180,
  ex: null,
  running: false,
  timer: 0,
  ghosts: new Set(),         // cells the user made walls that the sensor has not seen yet
  flash: { cells: [], t0: 0 }, // expansions of the latest replan, drawn in green
  paths: { dstar: [], astar: [] },
  series: [],                // one point per step: cumulative expansions of each planner
  hover: -1,
  cell: 10,                  // css pixels per map cell
  dirty: true,
};

const canvas = $('rv-map');
const ctx = canvas.getContext('2d');
const chart = $('rv-chart');
const cctx = chart.getContext('2d');

// ── small helpers ────────────────────────────────────────────────────────

const labelOf = (cell, n) => `(${cell % n},${Math.floor(cell / n)})`;
const plural = (k, word) => `${k} ${word}${k === 1 ? '' : 's'}`;
const totals = () => ({ d: state.ex.expansionsOf('dstar'), a: state.ex.expansionsOf('astar') });

function log(text, cls = 'log-info') {
  const line = document.createElement('div');
  line.className = cls;
  line.textContent = text;
  $('log').appendChild(line);
  $('log').scrollTop = $('log').scrollHeight;
}

// Mix two hex colours: t = 0 gives a, t = 1 gives b. Returns an rgb() string.
function mix(a, b, t) {
  const pa = [1, 3, 5].map((i) => parseInt(a.slice(i, i + 2), 16));
  const pb = [1, 3, 5].map((i) => parseInt(b.slice(i, i + 2), 16));
  const c = pa.map((v, i) => Math.round(v + (pb[i] - v) * t));
  return `rgb(${c[0]},${c[1]},${c[2]})`;
}

// ── the map ──────────────────────────────────────────────────────────────

function fitCanvas() {
  const css = Math.max(200, Math.min(640, Math.round(canvas.getBoundingClientRect().width)));
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  canvas.width = css * dpr;
  canvas.height = css * dpr;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  state.cell = css / state.size;
  state.dirty = true;
}

// Refresh the cached routes. Walking a route is O(length), so do it once per change, not once per frame.
function refreshPaths() {
  const ex = state.ex;
  state.paths = { dstar: ex.path('dstar'), astar: ex.path('astar') };
}

function cellCentre(cell) {
  const n = state.size, cs = state.cell;
  return [((cell % n) + 0.5) * cs, (Math.floor(cell / n) + 0.5) * cs];
}

function drawMap(now) {
  const ex = state.ex;
  if (!ex) return;
  const n = ex.n, cs = state.cell, N = n * n;
  const W = cs * n;
  ctx.fillStyle = '#02060c';
  ctx.fillRect(0, 0, W, W);

  // Heat map: D* Lite's g or rhs, normalised over the finite values. Cyan is near the goal, pink far away.
  if (state.heat !== 'none') {
    const vals = state.heat === 'g' ? ex.planners.dstar.g : ex.planners.dstar.rhs;
    let lo = INF, hi = -INF;
    for (let i = 0; i < N; i++) {
      const v = vals[i];
      if (v < INF) { if (v < lo) lo = v; if (v > hi) hi = v; }
    }
    const span = hi > lo ? hi - lo : 1;
    for (let i = 0; i < N; i++) {
      const v = vals[i];
      if (v === INF) continue;
      const r = Math.floor(i / n), c = i % n;
      ctx.globalAlpha = ex.seen[i] ? 0.5 : 0.22;
      ctx.fillStyle = mix(COL.cyan, COL.pink, (v - lo) / span);
      ctx.fillRect(c * cs, r * cs, cs + 0.5, cs + 0.5);
    }
    ctx.globalAlpha = 1;
  }

  // Cells: unseen dark, seen-free faint, walls found pink with a glow (one path, one shadow pass).
  const walls = new Path2D();
  const free = new Path2D();
  const unseen = new Path2D();
  for (let i = 0; i < N; i++) {
    const r = Math.floor(i / n), c = i % n;
    const target = ex.belief[i] ? walls : ex.seen[i] ? free : unseen;
    target.rect(c * cs, r * cs, cs, cs);
  }
  if (state.heat === 'none') {
    ctx.fillStyle = '#050d1a';
    ctx.fill(unseen);
  }
  ctx.fillStyle = 'rgba(0,245,255,0.08)';
  ctx.fill(free);
  ctx.save();
  ctx.shadowColor = COL.pink;
  ctx.shadowBlur = Math.max(4, cs * 0.9);
  ctx.fillStyle = COL.pink;
  ctx.fill(walls);
  ctx.restore();

  // Grid lines, faint: they make a single cell readable at 390 px.
  if (cs >= 6) {
    ctx.strokeStyle = 'rgba(0,245,255,0.06)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    for (let k = 0; k <= n; k++) {
      ctx.moveTo(k * cs, 0); ctx.lineTo(k * cs, W);
      ctx.moveTo(0, k * cs); ctx.lineTo(W, k * cs);
    }
    ctx.stroke();
  }

  // Wall drops the sensor has not reached: dashed pink outlines, so the player knows what is there.
  ctx.setLineDash([3, 3]);
  ctx.strokeStyle = COL.pink;
  ctx.lineWidth = 1.2;
  for (const i of state.ghosts) {
    if (ex.seen[i] || !ex.world.walls[i]) continue;
    const r = Math.floor(i / n), c = i % n;
    ctx.strokeRect(c * cs + 1, r * cs + 1, cs - 2, cs - 2);
  }
  ctx.setLineDash([]);

  // Expansions of the latest replan, fading out over FLASH_MS.
  const age = now - state.flash.t0;
  if (state.flash.cells.length && age < FLASH_MS) {
    const a = 1 - age / FLASH_MS;
    ctx.strokeStyle = `rgba(0,255,136,${a})`;
    ctx.fillStyle = `rgba(0,255,136,${0.22 * a})`;
    ctx.lineWidth = 1.5;
    for (const i of state.flash.cells) {
      const r = Math.floor(i / n), c = i % n;
      ctx.fillRect(c * cs, r * cs, cs, cs);
      ctx.strokeRect(c * cs + 0.75, r * cs + 0.75, cs - 1.5, cs - 1.5);
    }
  }

  // Sensor: the square the rover can see right now, with a slow pulse on its edge.
  const [rx, ry] = cellCentre(ex.pos);
  const rad = ex.radius;
  const pr = Math.floor(ex.pos / n), pc = ex.pos % n;
  const x0 = Math.max(0, pc - rad) * cs, y0 = Math.max(0, pr - rad) * cs;
  const x1 = (Math.min(n - 1, pc + rad) + 1) * cs, y1 = (Math.min(n - 1, pr + rad) + 1) * cs;
  const pulse = 0.5 + 0.5 * Math.sin(now / 320);
  ctx.fillStyle = 'rgba(0,245,255,0.05)';
  ctx.fillRect(x0, y0, x1 - x0, y1 - y0);
  ctx.save();
  ctx.strokeStyle = `rgba(0,245,255,${0.3 + 0.35 * pulse})`;
  ctx.shadowColor = COL.cyan;
  ctx.shadowBlur = 10;
  ctx.lineWidth = 1.5;
  ctx.strokeRect(x0 + 0.5, y0 + 0.5, x1 - x0 - 1, y1 - y0 - 1);
  ctx.restore();

  // Routes: A* in yellow dashed underneath, D* Lite in cyan on top. The route shown is drawn thicker.
  const drawRoute = (cells, name) => {
    if (cells.length < 2) return;
    const thick = state.view === name;
    ctx.save();
    ctx.beginPath();
    cells.forEach((cell, k) => {
      const [x, y] = cellCentre(cell);
      if (k) ctx.lineTo(x, y); else ctx.moveTo(x, y);
    });
    if (name === 'astar') {
      ctx.setLineDash([cs * 0.35, cs * 0.25]);
      ctx.strokeStyle = COL.yellow;
      ctx.lineWidth = thick ? Math.max(2, cs * 0.22) : Math.max(1.2, cs * 0.12);
    } else {
      ctx.strokeStyle = COL.cyan;
      ctx.shadowColor = COL.cyan;
      ctx.shadowBlur = 8;
      ctx.lineWidth = thick ? Math.max(2, cs * 0.25) : Math.max(1.2, cs * 0.12);
    }
    ctx.lineJoin = 'round';
    ctx.stroke();
    ctx.restore();
  };
  drawRoute(state.paths.astar, 'astar');
  drawRoute(state.paths.dstar, 'dstar');

  // Goal: a pulsing green diamond.
  const goalCell = ex.world.goal;
  const [gx, gy] = cellCentre(goalCell);
  const gr = cs * (0.36 + 0.06 * pulse);
  ctx.save();
  ctx.fillStyle = COL.green;
  ctx.shadowColor = COL.green;
  ctx.shadowBlur = 14;
  ctx.beginPath();
  ctx.moveTo(gx, gy - gr); ctx.lineTo(gx + gr, gy); ctx.lineTo(gx, gy + gr); ctx.lineTo(gx - gr, gy);
  ctx.closePath();
  ctx.fill();
  ctx.restore();

  // The rover: a white core with a yellow ring and a cyan glow.
  ctx.save();
  ctx.shadowColor = COL.cyan;
  ctx.shadowBlur = 16;
  ctx.fillStyle = COL.white;
  ctx.beginPath();
  ctx.arc(rx, ry, cs * 0.34, 0, Math.PI * 2);
  ctx.fill();
  ctx.lineWidth = Math.max(1.5, cs * 0.12);
  ctx.strokeStyle = COL.yellow;
  ctx.beginPath();
  ctx.arc(rx, ry, cs * 0.5, 0, Math.PI * 2);
  ctx.stroke();
  ctx.restore();

  // Hovered cell: a bright outline.
  if (state.hover >= 0) {
    const r = Math.floor(state.hover / n), c = state.hover % n;
    ctx.strokeStyle = COL.white;
    ctx.lineWidth = 1.5;
    ctx.strokeRect(c * cs + 0.75, r * cs + 0.75, cs - 1.5, cs - 1.5);
  }
}

function drawChart() {
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const cssW = Math.round(chart.getBoundingClientRect().width) || 640;
  const cssH = Math.round(cssW * 220 / 640);
  chart.width = cssW * dpr;
  chart.height = cssH * dpr;
  cctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  cctx.fillStyle = 'rgba(0,0,0,0.35)';
  cctx.fillRect(0, 0, cssW, cssH);
  const pad = { l: 46, r: 12, t: 12, b: 26 };
  const w = cssW - pad.l - pad.r, h = cssH - pad.t - pad.b;
  const s = state.series;
  const last = s[s.length - 1] || { steps: 0, d: 0, a: 0 };
  const xMax = Math.max(10, last.steps);
  const yMax = Math.max(10, last.d, last.a) * 1.08;
  const X = (v) => pad.l + (v / xMax) * w;
  const Y = (v) => pad.t + h - (v / yMax) * h;

  cctx.strokeStyle = 'rgba(0,245,255,0.12)';
  cctx.lineWidth = 1;
  cctx.font = '10px JetBrains Mono, monospace';
  cctx.fillStyle = 'rgba(160,190,200,0.7)';
  for (let k = 0; k <= 4; k++) {
    const y = pad.t + (h * k) / 4;
    cctx.beginPath(); cctx.moveTo(pad.l, y); cctx.lineTo(pad.l + w, y); cctx.stroke();
    cctx.fillText(Math.round(yMax * (1 - k / 4)).toString(), 6, y + 3);
  }
  cctx.fillText('steps', pad.l + w - 30, cssH - 8);
  cctx.fillText('0', pad.l - 4, cssH - 8);
  cctx.fillText(String(Math.round(xMax)), pad.l + w - 18, cssH - 8);

  const line = (key, color, dash) => {
    if (s.length < 2) return;
    cctx.save();
    cctx.strokeStyle = color;
    cctx.shadowColor = color;
    cctx.shadowBlur = 6;
    cctx.lineWidth = 2;
    cctx.setLineDash(dash);
    cctx.beginPath();
    s.forEach((p, k) => {
      if (k) cctx.lineTo(X(p.steps), Y(p[key])); else cctx.moveTo(X(p.steps), Y(p[key]));
    });
    cctx.stroke();
    cctx.restore();
  };
  line('a', COL.yellow, [6, 4]);
  line('d', COL.cyan, []);
}

// ── panels ───────────────────────────────────────────────────────────────

function updateChips() {
  const ex = state.ex;
  const t = totals();
  const status = ex.done ? 'GOAL REACHED' : ex.stuck ? 'NO KNOWN ROUTE' : state.running ? 'EXPLORING' : 'PAUSED';
  $('chip-status').textContent = status;
  $('chip-steps').textContent = String(ex.steps);
  $('chip-replans').textContent = String(ex.replans);
  $('chip-dstar').textContent = String(t.d);
  $('chip-astar').textContent = String(t.a);
  $('chip-ratio').textContent = t.a ? (t.d / t.a).toFixed(2) + 'x' : '—';
  $('chip-check').textContent = ex.mismatches === 0 ? 'OK' : `${ex.mismatches} MISMATCH`;
  $('chip-check').style.color = ex.mismatches === 0 ? '' : COL.pink;
}

// Explain the latest event in words, using the counters the planners report.
function think(text, { flash = false, warn = false } = {}) {
  const el = $('think-body');
  el.innerHTML = text;
  el.classList.toggle('warn', warn);
  if (flash) {
    el.classList.remove('flash');
    void el.offsetWidth;
    el.classList.add('flash');
  }
}

function routeCost() {
  const c = state.ex.planners.dstar.costToGoal(state.ex.pos);
  return c === INF ? null : c;
}

// The lead sentence says what happened (a sensor reading, or the player's edit). Then both planners'
// work on this replan, and the cost each one reports from the rover's cell.
function describeReplan(changes, dExp, aExp, lead, { withParts = true } = {}) {
  const walls = changes.filter(([, w]) => w).length;
  const clears = changes.length - walls;
  const parts = [];
  if (walls) parts.push(`found <span class="rv-warn">${plural(walls, 'wall')}</span>`);
  if (clears) parts.push(`cleared ${plural(clears, 'cell')}`);
  const cost = routeCost();
  const costText = cost === null ? 'no route' : `<span class="rv-num">${cost}</span> steps`;
  const detail = withParts && parts.length ? ` It ${parts.join(' and ')}.` : '';
  return `<b>${lead}</b>${detail}<br>` +
    `D* Lite repaired its plan with <span class="rv-num">${dExp}</span> expansions. ` +
    `A* searched again from scratch and expanded <span class="rv-num">${aExp}</span>.<br>` +
    `Both now report ${costText} to the goal from here.`;
}

function setStatusText() {
  const ex = state.ex;
  if (ex.done) return;
  if (ex.stuck) {
    think('<b>No known route.</b> Every path to the goal runs through a wall the rover has found. It waits: lift a wall, or the map changes.', { warn: true });
    return;
  }
  const cost = routeCost();
  if (cost !== null) think(`Following the cyan route. <span class="rv-num">${cost}</span> steps to the goal.`);
}

// ── stepping ─────────────────────────────────────────────────────────────

// Record one point on the chart and a line in the log when the rover has moved or replanned.
function afterStep(before, { user = false } = {}) {
  const ex = state.ex;
  const t = totals();
  const last = state.series[state.series.length - 1];
  if (!last || last.steps !== ex.steps || last.d !== t.d || last.a !== t.a) {
    if (state.series.length < CHART_POINTS) state.series.push({ steps: ex.steps, d: t.d, a: t.a });
  }
  for (const c of [...state.ghosts]) if (ex.seen[c]) state.ghosts.delete(c);
  updateChips();
  refreshPaths();
  if (ex.done) finish();
  state.dirty = true;
}

// Advance one step. Returns nothing; updates the panels, the chart, and the log.
function stepOnce() {
  const ex = state.ex;
  if (ex.done) return;
  const before = { ...totals(), replans: ex.replans, pos: ex.pos };
  ex.tick();
  if (ex.replans !== before.replans) {
    const now = totals();
    const dExp = now.d - before.d, aExp = now.a - before.a;
    const cells = (state.view === 'astar' ? ex.planners.astar : ex.planners.dstar).lastExpanded;
    state.flash = { cells: cells.slice(0, 4000), t0: performance.now() };
    const changes = ex.lastChanges;
    think(describeReplan(changes, dExp, aExp, `Sensor update at ${labelOf(before.pos, state.size)}.`), { flash: true });
    expand(Math.min(1, (dExp + aExp) / 300)); // one micro-blip per replan, higher for more expansions
    const walls = changes.filter(([, w]) => w).length;
    log(`step ${ex.steps}: sensed ${walls ? plural(walls, 'wall') : 'a clear cell'} near ${labelOf(before.pos, state.size)}. ` +
        `replan cost D* ${dExp}, A* ${aExp}.`, walls ? 'log-warn' : 'log-info');
    if (ex.mismatches) log('cost check failed: the planners disagree', 'log-warn');
  } else {
    setStatusText();
  }
  afterStep(before);
}

function finish() {
  const ex = state.ex;
  stop();
  const t = totals();
  const ratio = t.a ? (t.d / t.a).toFixed(2) : '—';
  log(`goal reached in ${ex.steps} steps with ${ex.replans} replans. D* Lite ${t.d} expansions, A* ${t.a} (ratio ${ratio}).`, 'log-win');
  banner('GOAL REACHED', `${ex.steps} steps · D* ${t.d} · A* ${t.a} expansions`, COL.green);
  think(`<b>Goal reached</b> in <span class="rv-num">${ex.steps}</span> steps. Both planners agree on every replan cost.`);
}

function stop() {
  state.running = false;
  clearTimeout(state.timer);
  $('btn-run').setAttribute('aria-pressed', 'false');
  $('btn-run').querySelector('.btn-txt').textContent = '▶ RUN';
}

function runLoop() {
  if (!state.running) return;
  const ex = state.ex;
  if (ex.done) { stop(); return; }
  stepOnce();
  if (!ex.done && state.running) state.timer = setTimeout(runLoop, state.delay);
}

function start() {
  if (state.ex.done) newMap(state.seed);
  state.running = true;
  $('btn-run').setAttribute('aria-pressed', 'true');
  $('btn-run').querySelector('.btn-txt').textContent = '❚❚ PAUSE';
  runLoop();
}

// ── maps and clicks ──────────────────────────────────────────────────────

function newMap(seed) {
  stop();
  state.seed = seed;
  $('seed').value = String(seed);
  const n = state.size;
  const world = { n, walls: generate(n, state.density, seed), start: 0, goal: n * n - 1 };
  state.ex = new Explorer(world, state.radius, state.driver);
  state.ex.refresh();            // sense the start: the first replan happens before the first step
  state.ghosts = new Set();
  state.flash = { cells: [], t0: 0 };
  const t = totals();
  state.series = [{ steps: 0, d: t.d, a: t.a }];
  fitCanvas();
  refreshPaths();
  $('log').innerHTML = '';
  log(`new maze ${n}x${n}, ${Math.round(state.density * 100)}% walls, seed ${seed}. Both planners know only the start.`);
  think('<b>Ready.</b> The rover starts with an empty map. Every cell is assumed free until the sensor covers it. Press RUN or STEP.');
  updateChips();
  state.dirty = true;
}

// Click or tap: drop a wall on a free cell, or lift one. The rover senses it if the cell is in view.
function toggleAt(cell) {
  const ex = state.ex;
  const n = state.size;
  if (cell === ex.pos) { think('The rover is standing there. Pick another cell.', { warn: true }); return; }
  if (cell === ex.world.goal) { think('The goal cannot be walled off in this game. Pick another cell.', { warn: true }); return; }
  const wasWall = ex.world.walls[cell] === 1;
  const before = { ...totals(), replans: ex.replans, pos: ex.pos };
  ex.edit(cell, !wasWall);
  const dropped = !wasWall;
  const inView = ex.seen[cell] === 1;
  if (dropped && !inView) state.ghosts.add(cell);
  if (!dropped) state.ghosts.delete(cell);
  const where = labelOf(cell, n);
  if (ex.replans !== before.replans) {
    const now = totals();
    state.flash = { cells: (state.view === 'astar' ? ex.planners.astar : ex.planners.dstar).lastExpanded.slice(0, 4000), t0: performance.now() };
    const lead = dropped
      ? `You dropped a wall at ${where}, inside the sensor.`
      : `You lifted the wall at ${where}, inside the sensor.`;
    think(describeReplan(ex.lastChanges, now.d - before.d, now.a - before.a, lead, { withParts: false }), { flash: true });
    log(`${dropped ? 'dropped' : 'lifted'} a wall at ${where} in sight: replan D* ${now.d - before.d}, A* ${now.a - before.a}.`, dropped ? 'log-warn' : 'log-info');
  } else if (dropped) {
    think(`You dropped a wall at ${where}, outside the sensor. The rover does not know it yet; it will replan when it gets close.`, { flash: true });
    log(`dropped a wall at ${where}, out of sight.`, 'log-warn');
  } else {
    think(`You lifted a wall at ${where}, outside the sensor. Nothing changes for the rover until it looks.`);
    log(`lifted a wall at ${where}, out of sight.`);
  }
  afterStep(before, { user: true });
}

// Map a pointer position to a cell index, or -1 outside the grid.
function cellAt(evt) {
  const rect = canvas.getBoundingClientRect();
  const x = evt.clientX - rect.left, y = evt.clientY - rect.top;
  if (x < 0 || y < 0 || x >= rect.width || y >= rect.height) return -1;
  const n = state.size;
  const c = Math.min(n - 1, Math.floor((x / rect.width) * n));
  const r = Math.min(n - 1, Math.floor((y / rect.height) * n));
  return r * n + c;
}

function describe(cell) {
  const ex = state.ex;
  const n = state.size;
  const where = labelOf(cell, n);
  let what;
  if (cell === ex.pos) what = 'the rover';
  else if (ex.belief[cell]) what = 'wall (found)';
  else if (ex.seen[cell]) what = 'free (sensed)';
  else if (state.ghosts.has(cell)) what = 'your wall, not sensed yet';
  else what = 'unseen';
  const g = ex.planners.dstar.g[cell], rhs = ex.planners.dstar.rhs[cell];
  const vals = ex.belief[cell] ? '' : ` · D* g ${g === INF ? '∞' : g}, rhs ${rhs === INF ? '∞' : rhs}`;
  return `${where}: ${what}${vals}`;
}

canvas.addEventListener('click', (e) => {
  const cell = cellAt(e);
  if (cell >= 0) toggleAt(cell);
});
canvas.addEventListener('pointermove', (e) => {
  const cell = cellAt(e);
  if (cell !== state.hover) {
    state.hover = cell;
    state.dirty = true;
    $('rv-info').textContent = cell >= 0 ? describe(cell) : 'Hover a cell for its D* Lite values. Click a cell to drop or remove a wall.';
  }
});
canvas.addEventListener('pointerleave', () => {
  state.hover = -1;
  state.dirty = true;
});

// ── controls ─────────────────────────────────────────────────────────────

$('btn-run').addEventListener('click', () => (state.running ? stop() : start()));
$('btn-step').addEventListener('click', () => { stop(); stepOnce(); });
$('btn-new').addEventListener('click', () => newMap(Math.floor(Math.random() * 999999) + 1));
$('btn-reset').addEventListener('click', () => newMap(state.seed));
$('btn-shuffle').addEventListener('click', () => newMap(Math.floor(Math.random() * 999999) + 1));
$('size').addEventListener('change', (e) => {
  state.size = Number(e.target.value);
  newMap(state.seed);
});
$('density').addEventListener('change', (e) => {
  state.density = Number(e.target.value);
  newMap(state.seed);
});
$('seed').addEventListener('change', (e) => {
  const v = Math.max(0, Math.min(4294967295, Math.floor(Number(e.target.value) || 0)));
  newMap(v);
});
$('radius').addEventListener('input', (e) => {
  state.radius = Number(e.target.value);
  $('radius-val').textContent = String(state.radius);
  state.ex.radius = state.radius;
  state.ex.refresh();            // a wider sensor may reveal cells right away
  refreshPaths();
  updateChips();
  state.dirty = true;
});
$('speed').addEventListener('input', (e) => {
  state.delay = Number(e.target.value);
  $('speed-val').textContent = String(state.delay);
});
function setDriver(name) {
  state.driver = name;
  state.ex.driver = name;
  $('drv-dstar').setAttribute('aria-pressed', String(name === 'dstar'));
  $('drv-astar').setAttribute('aria-pressed', String(name === 'astar'));
  log(`${name === 'dstar' ? 'D* Lite' : 'A*'} now steers the rover from here.`);
  refreshPaths();
  state.dirty = true;
}
$('drv-dstar').addEventListener('click', () => setDriver('dstar'));
$('drv-astar').addEventListener('click', () => setDriver('astar'));
$('view').addEventListener('change', (e) => {
  state.view = e.target.value;
  state.dirty = true;
});
$('heat').addEventListener('change', (e) => {
  state.heat = e.target.value;
  state.dirty = true;
});

// The scale changes with the viewport (the layout stacks on phones), so refit when it does.
let resizeTimer = 0;
window.addEventListener('resize', () => {
  clearTimeout(resizeTimer);
  resizeTimer = setTimeout(() => { fitCanvas(); drawChart(); }, 120);
});

// ── animation ────────────────────────────────────────────────────────────

// A frame loop that draws at full rate while something changes, and at a slow pulse when idle.
let lastDraw = 0;
let chartDirty = true;
function frame(now) {
  const active = state.dirty || (state.flash.cells.length && now - state.flash.t0 < FLASH_MS);
  if (active || now - lastDraw > 90) {
    drawMap(now);
    lastDraw = now;
    state.dirty = false;
  }
  if (chartDirty || active) {
    drawChart();
    chartDirty = false;
  }
  requestAnimationFrame(frame);
}

// ── boot ─────────────────────────────────────────────────────────────────

$('size').value = String(state.size);
$('density').value = String(state.density);
$('seed').value = String(state.seed);
$('radius').value = String(state.radius);
$('radius-val').textContent = String(state.radius);
$('speed').value = String(state.delay);
$('speed-val').textContent = String(state.delay);
newMap(state.seed);
requestAnimationFrame(frame);
