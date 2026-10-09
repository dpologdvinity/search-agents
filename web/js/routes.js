// Route planner page.
//
// The browser owns the map (city positions) and drawing; the server runs the
// solvers and returns each run's progress as "frames" (snapshots of the best
// route so far). Animating a run means drawing those frames one after another.

import { getJSON, postJSON } from './api.js';
import { banner, burst, shake } from './fx.js';
import { expand } from './sfx.js';

const $ = (id) => document.getElementById(id);
const REDUCED = matchMedia('(prefers-reduced-motion: reduce)').matches;
const NAMES = {
  nearest_neighbor_2opt: 'Greedy + 2-opt',
  simulated_annealing: 'Simulated annealing',
  genetic_algorithm: 'Genetic algorithm',
};
const RACE_COLORS = { nearest_neighbor_2opt: '#00f5ff', simulated_annealing: '#00ff88', genetic_algorithm: '#9b00ff' };

const state = {
  cities: [],        // [[x, y], ...] in [0, 1]
  shown: null,       // route currently drawn on the main map
  shownColor: '#00ff88',
  best: null,        // shortest route any agent has found on this map
  optimal: null,     // exact optimum from the server (maps of 12 cities or fewer)
  drawing: false,    // true while the player is clicking out their own route
  yours: [],         // the player's route so far
  animation: 0,      // id of the running animation timer, so a new run can cancel it
  meta: null,
};

// ── Geometry ────────────────────────────────────────────────────────────

const dist = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1]);

/** Length of a closed route over the current cities, in map units (the map is 1 x 1). */
function routeLength(route) {
  let total = 0;
  for (let k = 0; k < route.length; k++) total += dist(state.cities[route[k]], state.cities[route[(k + 1) % route.length]]);
  return total;
}

/** Lengths are shown x100 so a full-width map edge reads as 100 units. */
const fmt = (len) => (len * 100).toFixed(1);

// ── Drawing ─────────────────────────────────────────────────────────────

function setupCanvas(canvas, cssSize) {
  // Scale the backing store for sharp lines on high-DPI screens.
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const w = cssSize ?? canvas.clientWidth, h = cssSize ?? canvas.clientHeight;
  canvas.width = Math.round(w * dpr);
  canvas.height = Math.round(h * dpr);
  const ctx = canvas.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return { ctx, w, h };
}

/** Draw cities and, optionally, a closed route on any canvas. */
function drawMap(canvas, route, color, { yours = null } = {}) {
  const { ctx, w, h } = setupCanvas(canvas);
  const pad = Math.max(10, w * 0.04);
  const X = (c) => pad + c[0] * (w - 2 * pad), Y = (c) => pad + c[1] * (h - 2 * pad);

  ctx.fillStyle = '#020610';
  ctx.fillRect(0, 0, w, h);
  // Faint street grid for the "city" feel.
  ctx.strokeStyle = 'rgba(0,245,255,0.06)';
  for (let k = 0; k <= 10; k++) {
    const t = pad + (k / 10) * (w - 2 * pad);
    ctx.beginPath(); ctx.moveTo(t, pad); ctx.lineTo(t, h - pad); ctx.moveTo(pad, t); ctx.lineTo(w - pad, t); ctx.stroke();
  }

  const path = (r, closed) => {
    ctx.beginPath();
    r.forEach((i, k) => (k ? ctx.lineTo(X(state.cities[i]), Y(state.cities[i])) : ctx.moveTo(X(state.cities[i]), Y(state.cities[i]))));
    if (closed) ctx.closePath();
    ctx.stroke();
  };
  if (route && route.length > 1) {
    ctx.save();
    ctx.strokeStyle = color; ctx.lineWidth = Math.max(1.5, w / 260); ctx.shadowColor = color; ctx.shadowBlur = 12;
    path(route, true);
    ctx.restore();
  }
  if (yours && yours.length > 1) {
    // The player's route: dashed pink, closed only once every city is visited.
    ctx.save();
    ctx.strokeStyle = '#ff00a0'; ctx.lineWidth = 2.5; ctx.setLineDash([8, 6]); ctx.shadowColor = '#ff00a0'; ctx.shadowBlur = 10;
    path(yours, yours.length === state.cities.length);
    ctx.restore();
  }
  // Cities: glowing squares; the player's visited cities get a ring.
  const r = Math.max(2.5, w / 110);
  state.cities.forEach((c, i) => {
    ctx.save();
    ctx.fillStyle = i === 0 ? '#ffe600' : '#c0e8ff';
    ctx.shadowColor = ctx.fillStyle; ctx.shadowBlur = 10;
    ctx.fillRect(X(c) - r, Y(c) - r, 2 * r, 2 * r);
    ctx.restore();
    if (yours && yours.includes(i)) {
      ctx.strokeStyle = '#ff00a0'; ctx.lineWidth = 2;
      ctx.beginPath(); ctx.arc(X(c), Y(c), r * 2.2, 0, Math.PI * 2); ctx.stroke();
    }
  });
}

function redraw() {
  drawMap($('map'), state.drawing ? null : state.shown, state.shownColor, { yours: state.yours });
  $('chip-cities').textContent = String(state.cities.length);
  $('chip-best').textContent = state.best ? fmt(state.best.length) : '—';
}

/** Line chart of each series ([{label, color, values}]) against step. */
function drawChart(series, title) {
  const canvas = $('chart');
  const { ctx, w, h } = setupCanvas(canvas);
  ctx.clearRect(0, 0, w, h);
  const pad = 30, all = series.flatMap((s) => s.values.filter((v) => v != null));
  if (!all.length) return;
  const lo = Math.min(...all), hi = Math.max(...all), span = hi - lo || 1;
  ctx.strokeStyle = 'rgba(0,245,255,0.25)';
  ctx.beginPath(); ctx.moveTo(pad, 6); ctx.lineTo(pad, h - pad); ctx.lineTo(w - 6, h - pad); ctx.stroke();
  for (const s of series) {
    ctx.save();
    ctx.strokeStyle = s.color; ctx.lineWidth = 2; ctx.shadowColor = s.color; ctx.shadowBlur = 6;
    ctx.beginPath();
    s.values.forEach((v, k) => {
      const x = pad + (k / Math.max(1, s.values.length - 1)) * (w - pad - 10);
      const y = h - pad - ((v - lo) / span) * (h - pad - 14);
      k ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
    });
    ctx.stroke();
    ctx.restore();
  }
  $('chart-title').textContent = title;
  $('chart-legend').innerHTML = series.map((s) =>
    `<span class="field-hint"><span class="legend-dot" style="background:${s.color}"></span>${s.label}</span>`).join('');
}

// ── Map editing ─────────────────────────────────────────────────────────

function randomMap(n) {
  state.cities = Array.from({ length: n }, () => [Math.random(), Math.random()]);
  resetResults();
}

function resetResults() {
  cancelAnimationFrame(state.animation);
  Object.assign(state, { shown: null, best: null, optimal: null, yours: [] });
  $('race-body').innerHTML = '';
  $('chip-yours').textContent = '—';
  setDrawing(false);
  redraw();
}

/** Nearest city to a click, if the click is close enough to count. */
function cityAt(x, y) {
  let best = -1, bestD = 0.04;
  state.cities.forEach((c, i) => { const d = dist(c, [x, y]); if (d < bestD) { best = i; bestD = d; } });
  return best;
}

function onMapClick(e) {
  const rect = $('map').getBoundingClientRect();
  const pad = Math.max(10, rect.width * 0.04);
  const x = (e.clientX - rect.left - pad) / (rect.width - 2 * pad);
  const y = (e.clientY - rect.top - pad) / (rect.height - 2 * pad);
  if (x < 0 || x > 1 || y < 0 || y > 1) return;

  if (!state.drawing) {
    if (state.cities.length >= state.meta.max_cities) return;
    state.cities.push([x, y]);
    resetResults();
    return;
  }
  // Drawing mode: append the clicked city to the player's route.
  const i = cityAt(x, y);
  if (i < 0 || state.yours.includes(i)) return;
  state.yours.push(i);
  redraw();
  if (state.yours.length === state.cities.length) finishYourRoute();
}

function setDrawing(on) {
  state.drawing = on;
  $('btn-draw').setAttribute('aria-pressed', String(on));
  $('btn-draw').querySelector('.btn-txt').textContent = on ? '✕ STOP DRAWING' : '✎ DRAW YOUR ROUTE';
  $('map-hint').textContent = on ? 'Click every city once, in the order you would visit them. The route closes itself.'
    : 'Click the map to add a city.';
}

function finishYourRoute() {
  const mine = routeLength(state.yours);
  $('chip-yours').textContent = fmt(mine);
  setDrawing(false);
  state.shown = state.best ? state.best.route : null;
  redraw();
  if (!state.best) { banner('ROUTE DONE', `your route: ${fmt(mine)} — now run an agent and compare`, '#ff00a0'); return; }
  const diff = ((mine / state.best.length - 1) * 100).toFixed(1);
  if (mine <= state.best.length + 1e-9) {
    burst($('map'), { count: 160 });
    banner('YOU BEAT THE AGENTS', `your route ${fmt(mine)} vs ${fmt(state.best.length)}`, '#00ff88');
  } else {
    shake($('map'));
    banner(`${diff}% LONGER`, `your route ${fmt(mine)} vs the agents' best ${fmt(state.best.length)}`, '#ff00a0');
  }
}

// ── Running agents ──────────────────────────────────────────────────────

function params() {
  const cool = Number($('cooling').value);
  return {
    iterations: Number($('iterations').value),
    // Slider 0 means "auto"; otherwise map 1..100 onto cooling rates 0.9990..0.99999.
    cooling: cool === 0 ? null : 0.999 + (cool / 100) * 0.00099,
    population_size: Number($('population').value),
    generations: Number($('generations').value),
    mutation_rate: Number($('mutation').value) / 100,
  };
}

const solve = (algorithm) => postJSON('/api/routes/solve', { cities: state.cities, algorithm, params: params() });

/** Remember the shortest route seen on this map, across all agents and runs. */
function consider(res) {
  if (res.optimal) state.optimal = res.optimal.length;
  if (!state.best || res.length < state.best.length) state.best = { route: res.tour, length: res.length, by: res.algorithm };
}

/** Animate frames onto a canvas; resolves when the last frame is drawn. */
function animate(frames, onFrame, perFrameMs = 25) {
  return new Promise((resolve) => {
    if (REDUCED) { onFrame(frames[frames.length - 1], frames.length - 1); resolve(); return; }
    let k = 0, last = 0;
    const step = (t) => {
      if (t - last >= perFrameMs) { onFrame(frames[k], k); expand(k / frames.length); k++; last = t; }
      if (k < frames.length) state.animation = requestAnimationFrame(step); else resolve();
    };
    state.animation = requestAnimationFrame(step);
  });
}

function chartFor(res, upTo) {
  const fr = res.frames.slice(0, upTo + 1);
  const series = [{ label: 'best route so far', color: '#00ff88', values: fr.map((f) => f.best_length) }];
  if (res.algorithm === 'simulated_annealing') {
    series.push({ label: 'current route (wanders while hot)', color: '#ff00a0', values: fr.map((f) => f.current_length) });
  }
  if (res.algorithm === 'genetic_algorithm') {
    series.push({ label: 'population average', color: '#9b00ff', values: fr.map((f) => f.mean_length) });
  }
  const f = fr[fr.length - 1];
  const extra = res.algorithm === 'simulated_annealing' ? ` · temperature ${f.temperature.toPrecision(2)}` : '';
  return [series, `${NAMES[res.algorithm]} · step ${f.step.toLocaleString()} · best ${fmt(f.best_length)}${extra}`];
}

async function runOne() {
  if (state.cities.length < 3) { banner('ADD CITIES', 'place at least 3 cities first', '#ffe600'); return; }
  const alg = $('algorithm').value;
  setBusy(true, `${NAMES[alg].toUpperCase()} RUNNING`);
  try {
    const res = await solve(alg);
    consider(res);
    state.shownColor = RACE_COLORS[alg];
    await animate(res.frames, (f, k) => {
      state.shown = f.tour;
      redraw();
      drawChart(...chartFor(res, k));
    });
    state.shown = res.tour;
    redraw();
    burst($('map'), { count: 80, colors: [RACE_COLORS[alg], '#ffe600'] });
    const gap = state.optimal ? ` · ${gapText(res.length)}` : '';
    banner('ROUTE FOUND', `${NAMES[alg]} · length ${fmt(res.length)} in ${res.seconds.toFixed(2)} s${gap}`, RACE_COLORS[alg]);
  } catch (err) {
    banner('ERROR', err.message, '#ff3b3b');
  }
  setBusy(false);
}

function gapText(length) {
  const gap = (length / state.optimal - 1) * 100;
  return gap < 0.05 ? 'the best possible route' : `${gap.toFixed(1)}% longer than the best possible`;
}

async function race() {
  if (state.cities.length < 3) { banner('ADD CITIES', 'place at least 3 cities first', '#ffe600'); return; }
  setBusy(true, 'RACING');
  const algs = Object.keys(NAMES);
  try {
    // Solve all three on the server at once, then animate them side by side.
    const results = await Promise.all(algs.map(solve));
    results.forEach(consider);
    const canvases = Object.fromEntries([...document.querySelectorAll('#race canvas')].map((c) => [c.dataset.alg, c]));
    await Promise.all(results.map((res) => animate(res.frames, (f) => drawMap(canvases[res.algorithm], f.tour, RACE_COLORS[res.algorithm]), 30)));
    results.forEach((res) => drawMap(canvases[res.algorithm], res.tour, RACE_COLORS[res.algorithm]));

    const ranked = [...results].sort((a, b) => a.length - b.length);
    $('race-body').innerHTML = ranked.map((res, k) => `
      <tr class="${k === 0 ? 'best' : ''}">
        <td>${NAMES[res.algorithm]}</td>
        <td class="num">${fmt(res.length)}</td>
        <td class="num">${state.optimal ? gapText(res.length) : 'too many cities to check'}</td>
        <td class="num">${res.steps.toLocaleString()}</td>
        <td class="num">${res.seconds.toFixed(2)} s</td>
      </tr>`).join('');
    state.shown = ranked[0].tour;
    state.shownColor = RACE_COLORS[ranked[0].algorithm];
    redraw();
    burst($('race'), { count: 120 });
    banner(`${NAMES[ranked[0].algorithm].toUpperCase()} WINS`, `length ${fmt(ranked[0].length)}`, RACE_COLORS[ranked[0].algorithm]);
  } catch (err) {
    banner('ERROR', err.message, '#ff3b3b');
  }
  setBusy(false);
}

function setBusy(busy, status = 'READY') {
  for (const id of ['btn-run', 'btn-race', 'btn-draw', 'btn-random', 'btn-clear', 'algorithm']) $(id).disabled = busy;
  $('chip-status').textContent = status;
}

// ── Wiring ──────────────────────────────────────────────────────────────

function syncLabels() {
  $('n-lbl').textContent = $('n').value;
  $('it-lbl').textContent = Number($('iterations').value).toLocaleString();
  const p = params();
  $('cool-lbl').textContent = p.cooling ? p.cooling.toFixed(5) : 'auto';
  $('pop-lbl').textContent = $('population').value;
  $('gen-lbl').textContent = $('generations').value;
  $('mut-lbl').textContent = (Number($('mutation').value) / 100).toFixed(2);
  const alg = $('algorithm').value;
  $('sa-params').classList.toggle('hidden', alg !== 'simulated_annealing');
  $('ga-params').classList.toggle('hidden', alg !== 'genetic_algorithm');
  const info = state.meta?.algorithms.find((a) => a.name === alg);
  $('alg-desc').textContent = info ? info.description : '';
}

async function init() {
  $('map').addEventListener('click', onMapClick);
  $('btn-run').onclick = runOne;
  $('btn-race').onclick = race;
  $('btn-draw').onclick = () => {
    if (state.cities.length < 3) return;
    state.yours = [];
    setDrawing(!state.drawing);
    redraw();
  };
  $('btn-random').onclick = () => randomMap(Number($('n').value));
  $('btn-clear').onclick = () => { state.cities = []; resetResults(); };
  for (const id of ['n', 'iterations', 'cooling', 'population', 'generations', 'mutation', 'algorithm']) $(id).oninput = syncLabels;
  addEventListener('resize', redraw);
  try {
    state.meta = await getJSON('/api/routes/meta');
  } catch (err) {
    banner('SERVER UNREACHABLE', err.message, '#ff3b3b');
    return;
  }
  syncLabels();
  randomMap(20);
}

init();
