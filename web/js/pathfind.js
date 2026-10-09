// Pathfinding lab page. The search and the maps run here, in pathfind-core.js, which is a port of the
// Python package pathfind/. This file draws the map editor, plays the race back one expansion at a
// time, and explains each cell when you hover it.
//
// Racing works like this: a race runs every checked search to completion at once (the algorithms are
// fast on these grids), then playback advances all of them by the same number of expansions per frame,
// so the panels stay synchronised. A cell's state is kept as a small array (none, frontier, expanded),
// updated as the playback moves, so each frame draws only what changed since the last one.

import { getJSON } from './api.js';
import { banner, burst, shake } from './fx.js';
import {
  ALGOS, DEFAULT_WEIGHT, LABELS, MUD, OPEN, SWAMP, WALL,
  frontierSizes, heuristic, makeMaze, run, uniformCost,
} from './pathfind-core.js';

const $ = (id) => document.getElementById(id);
const REDUCED = matchMedia('(prefers-reduced-motion: reduce)').matches;
const EPS = 1e-9;
const RECENT = 6; // expansions that pulse after they are visited
const LASER_MS = 900; // the path laser's sweep time, plus a little per cell below
const COLORS = { frontier: '0,245,255', expanded: '155,0,255', path: '#ff00a0', start: '#00ff88', goal: '#ffe600' };
const TOOL_HINT = {
  wall: 'Drag to draw walls. Start and goal stay open.',
  swamp: 'Drag to paint swamp: entering a swamp cell costs 5, so routes go around it when they can.',
  mud: 'Drag to paint mud: entering a mud cell costs 3.',
  erase: 'Drag to clear walls and terrain back to open ground (cost 1).',
  start: 'Drag the start marker anywhere on the map.',
  goal: 'Drag the goal marker anywhere on the map.',
};
const ALGO_TEXT = {
  bfs: 'Expands cells in rings by number of steps. Shortest in steps; on swamp it can pay more than the cheapest route.',
  dfs: 'Follows one branch as far as it goes, then backs up. Fast and light, but the first route found is rarely short.',
  ucs: 'Expands the cheapest cell from the start (g). Always optimal. The reference the others are judged against.',
  greedy: 'Expands the cell that looks closest to the goal (h only). Fast, ignores cost paid, can walk through swamp.',
  astar: 'Expands the lowest f = g + h. Optimal with an admissible h (octile is admissible on 8-way moves).',
  wastar: 'Expands the lowest f = g + w·h. With w = 2 the path is at most twice the optimum, and far fewer cells are expanded.',
  bidi: 'Two breadth-first searches, one from each end, until they meet. Short in steps, not necessarily cheap.',
};
const KIND_NOTE = {
  recursive: 'Perfect maze, carved by a recursive backtracker: long winding corridors, one route between any two cells.',
  prim: "Perfect maze grown by Prim's algorithm: many short branches, one route between any two cells.",
  scatter: 'Random walls at 28%. Connected only by luck; the lab says so when no route exists.',
  rooms: 'Rectangular rooms joined by corridors, with a few loops.',
  open: 'Empty ground. Only the costs and the walls you draw matter.',
};

const S = {
  grid: null,
  start: 0,
  goal: 0,
  map: { kind: 'prim', size: 41, seed: 3, swamp: 0, density: 28 },
  tool: 'wall',
  drag: null, // 'paint' | 'start' | 'goal' | null
  lastCell: -1,
  version: 0, // bumped on every map edit, so cached base layers are rebuilt
  racers: [],
  optimal: null,
  state: 'idle', // idle | running | paused | done
  mode: 'panels',
  focus: 'astar',
  rate: 8,
  rafId: 0,
  frames: 0,
  presets: [],
  editor: null,
  focusView: null,
};

// ── Views: a canvas plus a cached base layer (walls, terrain, endpoints) ──

function makeView(canvas, kind, racer = null) {
  return { canvas, ctx: canvas.getContext('2d'), base: document.createElement('canvas'), key: '', cs: 0, kind, racer, hoverCell: -1 };
}

// Size the canvas to its CSS width (times device pixel ratio, up to 2x) and rebuild the base layer when
// the map, the size or the pixel ratio changed.
function ensureSize(view, css) {
  const N = S.grid.width;
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const key = `${S.version}|${N}|${css}|${dpr}`;
  if (view.key === key) return;
  view.key = key;
  view.cs = css / N;
  const px = Math.round(css * dpr);
  view.canvas.width = px;
  view.canvas.height = px;
  view.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  view.base.width = px;
  view.base.height = px;
  const bctx = view.base.getContext('2d');
  bctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  paintBase(bctx, css, N, view.cs);
}

// Snap a cell to whole pixels so neighbouring cells never leave a hairline gap.
function cellRect(cs, i, N) {
  const x = i % N, y = Math.floor(i / N);
  const x0 = Math.floor(x * cs), y0 = Math.floor(y * cs);
  return [x0, y0, Math.floor((x + 1) * cs) - x0, Math.floor((y + 1) * cs) - y0];
}

function paintBase(bctx, css, N, cs) {
  const { cells } = S.grid;
  bctx.fillStyle = '#020610';
  bctx.fillRect(0, 0, css, css);
  bctx.strokeStyle = 'rgba(0,245,255,0.07)';
  bctx.lineWidth = 1;
  bctx.beginPath();
  for (let k = 0; k <= N; k++) {
    const p = Math.round(k * cs) + 0.5;
    bctx.moveTo(p, 0); bctx.lineTo(p, css);
    bctx.moveTo(0, p); bctx.lineTo(css, p);
  }
  bctx.stroke();
  for (let i = 0; i < cells.length; i++) {
    const [x, y, w, h] = cellRect(cs, i, N);
    const v = cells[i];
    if (v === WALL) {
      bctx.fillStyle = 'rgba(0,245,255,0.45)';
      bctx.fillRect(x, y, w, h);
      if (w > 3) {
        bctx.fillStyle = '#050a18';
        bctx.fillRect(x + 1, y + 1, w - 2, h - 2);
      }
    } else if (v === SWAMP || v === MUD) {
      bctx.fillStyle = v === SWAMP ? 'rgba(0,255,136,0.18)' : 'rgba(255,230,0,0.12)';
      bctx.fillRect(x, y, w, h);
      if (cs >= 12) {
        bctx.fillStyle = v === SWAMP ? 'rgba(0,255,136,0.7)' : 'rgba(255,230,0,0.7)';
        bctx.font = `${Math.floor(cs * 0.5)}px "JetBrains Mono", monospace`;
        bctx.textAlign = 'center';
        bctx.textBaseline = 'middle';
        bctx.fillText(String(v), x + w / 2, y + h / 2);
      }
    }
  }
  // Endpoints last, so they sit on top of any terrain under them.
  const [sx, sy, sw, sh] = cellRect(cs, S.start, N);
  const [gx, gy, gw, gh] = cellRect(cs, S.goal, N);
  bctx.save();
  bctx.shadowColor = COLORS.start;
  bctx.shadowBlur = 10;
  bctx.fillStyle = COLORS.start;
  bctx.beginPath();
  bctx.arc(sx + sw / 2, sy + sh / 2, Math.max(2, Math.min(sw, sh) * 0.36), 0, Math.PI * 2);
  bctx.fill();
  bctx.shadowColor = COLORS.goal;
  bctx.fillStyle = COLORS.goal;
  bctx.fillRect(gx + gw * 0.2, gy + gh * 0.2, gw * 0.6, gh * 0.6);
  bctx.restore();
}

// ── Drawing a view: base layer, then the racer's states, pulses and path laser ──

function drawView(view, now) {
  const css = view.canvas.clientWidth;
  if (!css) return;
  ensureSize(view, css);
  const { ctx, cs } = view;
  const N = S.grid.width;
  ctx.clearRect(0, 0, css, css);
  ctx.drawImage(view.base, 0, 0, css, css);
  const r = view.racer;
  if (r) paintRacer(ctx, r, cs, N, now, view.kind);
  if (view.hoverCell >= 0) {
    const [x, y, w, h] = cellRect(cs, view.hoverCell, N);
    ctx.strokeStyle = '#ffffff';
    ctx.lineWidth = 1.5;
    ctx.strokeRect(x + 0.75, y + 0.75, w - 1.5, h - 1.5);
  }
}

// Adds one cell to the current path. Filling one path per state is far cheaper than a fillRect per cell.
function addCellRect(ctx, cs, i, N) {
  const x = i % N, y = (i - x) / N;
  const x0 = Math.floor(x * cs), y0 = Math.floor(y * cs);
  ctx.rect(x0, y0, Math.floor((x + 1) * cs) - x0, Math.floor((y + 1) * cs) - y0);
}

function paintRacer(ctx, r, cs, N, now, kind) {
  const st = r.state;
  const pulse = 0.55 + 0.25 * Math.sin(now / 180);
  ctx.beginPath();
  for (let i = 0; i < st.length; i++) if (st[i] === 2) addCellRect(ctx, cs, i, N);
  ctx.fillStyle = `rgba(${COLORS.expanded},0.38)`;
  ctx.fill();
  ctx.beginPath();
  for (let i = 0; i < st.length; i++) if (st[i] === 1) addCellRect(ctx, cs, i, N);
  ctx.fillStyle = `rgba(${COLORS.frontier},${pulse.toFixed(3)})`;
  ctx.fill();
  // The last few expansions flash and shrink back, so the eye can follow the search front.
  r.recent.forEach((cell, k) => {
    const age = r.recent.length - 1 - k;
    const f = 1 - age / RECENT;
    const [x, y, w, h] = cellRect(cs, cell, N);
    const grow = cs * 0.4 * f;
    ctx.fillStyle = `rgba(${COLORS.frontier},${(0.6 * f).toFixed(3)})`;
    ctx.fillRect(x - grow, y - grow, w + 2 * grow, h + 2 * grow);
  });
  if (r.done && r.res.found) paintLaser(ctx, r, cs, N, now, kind);
}

// The path is drawn as a laser that sweeps from start to goal once the racer finishes. A bright dashed
// core runs along it while the sweep is moving, then the path settles into a steady pink line.
function paintLaser(ctx, r, cs, N, now, kind) {
  const path = r.res.path;
  if (path.length < 2) return;
  // The frame time can be a little earlier than doneAt (it is taken at the start of the frame), so clamp.
  const elapsed = Math.max(0, now - r.doneAt);
  const span = LASER_MS + path.length * 12;
  const sweeping = !REDUCED && elapsed < span;
  const progress = REDUCED ? 1 : Math.min(1, elapsed / span);
  const pts = path.map((i) => [((i % N) + 0.5) * cs, (Math.floor(i / N) + 0.5) * cs]);
  const pos = progress * (pts.length - 1);
  const whole = Math.floor(pos);
  const frac = pos - whole;
  const tip = whole < pts.length - 1
    ? [pts[whole][0] + (pts[whole + 1][0] - pts[whole][0]) * frac, pts[whole][1] + (pts[whole + 1][1] - pts[whole][1]) * frac]
    : pts[pts.length - 1];
  const trace = () => {
    ctx.beginPath();
    ctx.moveTo(pts[0][0], pts[0][1]);
    for (let k = 1; k <= whole; k++) ctx.lineTo(pts[k][0], pts[k][1]);
    ctx.lineTo(tip[0], tip[1]);
  };
  ctx.save();
  ctx.lineJoin = 'round';
  ctx.lineCap = 'round';
  trace();
  ctx.strokeStyle = COLORS.path;
  ctx.lineWidth = Math.max(2, cs * 0.5);
  ctx.shadowColor = COLORS.path;
  ctx.shadowBlur = kind === 'focus' ? 16 : 8;
  ctx.stroke();
  ctx.shadowBlur = 0;
  if (sweeping) {
    trace();
    ctx.setLineDash([cs * 0.6, cs * 0.9]);
    ctx.lineDashOffset = -now / 25;
    ctx.strokeStyle = 'rgba(255,255,255,0.9)';
    ctx.lineWidth = Math.max(1, cs * 0.15);
    ctx.stroke();
    ctx.setLineDash([]);
  }
  ctx.fillStyle = '#ffffff';
  ctx.shadowColor = COLORS.path;
  ctx.shadowBlur = 10;
  ctx.beginPath();
  ctx.arc(tip[0], tip[1], Math.max(2, cs * 0.35), 0, Math.PI * 2);
  ctx.fill();
  ctx.restore();
}

function laserActive(now) {
  return !REDUCED && S.racers.some((r) => r.done && r.res.found && now - r.doneAt < LASER_MS + r.res.path.length * 12);
}

function drawAll(now) {
  if (!S.grid) return; // the first map is still loading
  drawView(S.editor, now);
  if (S.mode === 'panels') {
    for (const r of S.racers) if (r.view) drawView(r.view, now);
  } else if (S.focusView && S.focusView.racer) {
    drawView(S.focusView, now);
  }
}

// One frame: advance the race while it runs, redraw, and keep going while the laser sweeps.
// The leaderboard is refreshed every few frames, not every frame.
function frame(now) {
  S.rafId = 0;
  if (S.state === 'running') {
    advanceAll(S.rate);
    if (++S.frames % 8 === 0) renderBoard();
  }
  drawAll(now);
  if (S.state === 'running' || laserActive(now)) S.rafId = requestAnimationFrame(frame);
}

function requestDraw() {
  if (!S.rafId) S.rafId = requestAnimationFrame(frame);
}

// ── Race state ─────────────────────────────────────────────────────────────

// Apply expansions up to index `to`. Each expansion marks its cell expanded, and each discovered cell
// becomes frontier (unless it already has a state); g is kept at its latest value.
function advance(r, to) {
  const { steps } = r.res;
  while (r.idx < to) {
    const [cell, , fresh] = steps[r.idx];
    r.state[cell] = 2;
    r.order[cell] = r.idx + 1;
    r.recent.push(cell);
    if (r.recent.length > RECENT) r.recent.shift();
    for (const [v, g] of fresh) {
      r.g[v] = g;
      if (r.state[v] === 0) r.state[v] = 1;
    }
    r.generated += fresh.length;
    r.idx++;
    r.frontier = r.sizes[r.idx - 1];
    if (r.frontier > r.peak) r.peak = r.frontier;
  }
  if (!r.done && r.idx >= steps.length) {
    r.done = true;
    r.doneAt = performance.now();
  }
}

function advanceAll(n) {
  let finishedNow = false;
  for (const r of S.racers) {
    if (r.done) continue;
    advance(r, Math.min(r.res.steps.length, r.idx + n));
    updateCounters(r);
    if (r.done) finishedNow = true;
  }
  updateFocusStats();
  if (finishedNow) renderBoard();
  if (S.racers.length && S.racers.every((r) => r.done)) finishRace();
}

function buildRace() {
  const algos = checkedAlgos();
  if (!algos.length) {
    setHint('Tick at least one search in SEARCH to race it.');
    return false;
  }
  const opts = raceOptions();
  const ref = uniformCost(S.grid, S.start, S.goal, opts.diagonal);
  S.optimal = ref.found ? ref.cost : null;
  const n = S.grid.length;
  S.racers = algos.map((algo) => {
    const res = run(algo, S.grid, S.start, S.goal, opts);
    return {
      algo,
      res,
      sizes: frontierSizes(res),
      idx: 0,
      state: new Uint8Array(n),
      g: new Float64Array(n),
      order: new Int32Array(n),
      recent: [],
      generated: 0,
      peak: 0,
      frontier: 0,
      done: false,
      doneAt: 0,
      view: null,
    };
  });
  buildPanels();
  if (S.focusView) S.focusView.racer = S.racers.find((r) => r.algo === S.focus) || null;
  S.frames = 0;
  return true;
}

function resetRace() {
  S.state = 'idle';
  if (S.rafId) cancelAnimationFrame(S.rafId);
  S.rafId = 0;
  S.racers = [];
  S.optimal = null;
  $('pf-panels').innerHTML = '';
  if (S.focusView) S.focusView.racer = null;
  $('pf-focus-stats').innerHTML = '';
  renderBoard();
  setStatus('IDLE');
  refreshButtons();
  requestDraw();
}

function raceOptions() {
  return { diagonal: $('diag').checked, heur: $('heur').value, weight: Number($('weight').value) };
}

function checkedAlgos() {
  return ALGOS.filter((a) => document.querySelector(`#algos input[value="${a}"]`).checked);
}

// Panels: one canvas and a few counters per racer. The focus view reuses one large canvas.
function buildPanels() {
  const wrap = $('pf-panels');
  wrap.innerHTML = S.racers.map((r) => `
    <article class="pf-panel" data-algo="${r.algo}">
      <div class="pf-panel-head"><span class="pf-panel-title">${LABELS[r.algo]}</span><span class="pf-badge wait">WAIT</span></div>
      <canvas aria-label="${LABELS[r.algo]} search"></canvas>
      <dl class="pf-stats">
        <div><dt>EXPANDED</dt><dd data-k="exp">0</dd></div>
        <div><dt>FRONTIER</dt><dd data-k="front">0</dd></div>
        <div><dt>GENERATED</dt><dd data-k="gen">0</dd></div>
        <div><dt>COST</dt><dd data-k="cost">—</dd></div>
      </dl>
    </article>`).join('');
  wrap.querySelectorAll('.pf-panel').forEach((el, k) => {
    const r = S.racers[k];
    const canvas = el.querySelector('canvas');
    r.view = makeView(canvas, 'panel', r);
    canvas.addEventListener('pointermove', (e) => hoverRacer(r, r.view, e));
    canvas.addEventListener('pointerleave', () => hideTip(r.view));
  });
  updateStatChips();
  renderBoard();
}

function updateCounters(r) {
  if (!r.view) return;
  const el = r.view.canvas.closest('.pf-panel');
  el.querySelector('[data-k="exp"]').textContent = String(r.idx);
  el.querySelector('[data-k="front"]').textContent = String(r.frontier);
  el.querySelector('[data-k="gen"]').textContent = String(r.generated);
  if (r.done) {
    const badge = el.querySelector('.pf-badge');
    el.querySelector('[data-k="cost"]').textContent = r.res.found ? r.res.cost.toFixed(2) : '—';
    const verdict = verdictOf(r);
    badge.className = `pf-badge ${verdict.cls}`;
    badge.textContent = verdict.text;
  } else {
    el.querySelector('.pf-badge').className = 'pf-badge wait';
    el.querySelector('.pf-badge').textContent = 'SEARCHING';
  }
}

function verdictOf(r) {
  if (!r.res.found) return { cls: 'bad', text: 'NO PATH ✗' };
  if (S.optimal !== null && r.res.cost <= S.optimal + EPS) return { cls: 'ok', text: 'OPTIMAL ✓' };
  return { cls: 'bad', text: `+${(r.res.cost - S.optimal).toFixed(2)} ✗` };
}

function finishRace() {
  S.state = 'done';
  const reachable = S.racers.some((r) => r.res.found);
  setStatus(reachable ? 'DONE' : 'NO PATH');
  refreshButtons();
  const winners = S.racers.filter((r) => r.res.found && (S.optimal === null || r.res.cost <= S.optimal + EPS));
  winners.sort((a, b) => a.idx - b.idx || a.res.cost - b.res.cost);
  const best = winners[0];
  if (best) {
    best.view.canvas.closest('.pf-panel').classList.add('is-best');
    burst(best.view.canvas);
    banner(`${LABELS[best.algo].toUpperCase()} WINS`, `${best.idx} expansions · cost ${best.res.cost.toFixed(2)}`, '#00ff88');
  } else {
    shake($('pf-grid'));
    banner('NO ROUTE', 'the start and goal are not connected on this map', '#ff00a0');
  }
  renderBoard();
  requestDraw();
}

function setStatus(text) {
  $('chip-status').textContent = text;
}

function refreshButtons() {
  const running = S.state === 'running';
  $('btn-pause').disabled = !(running || S.state === 'paused');
  $('btn-pause').querySelector('.btn-txt').textContent = S.state === 'paused' ? '▶ RESUME' : '❚❚ PAUSE';
  $('btn-race').querySelector('.btn-txt').textContent = S.state === 'paused' ? '▶ RESUME' : '▶ RACE';
}

function updateStatChips() {
  $('chip-map').textContent = `${S.grid.width}×${S.grid.width}`;
  $('chip-racers').textContent = String(S.racers.length || checkedAlgos().length);
}

// Leaderboard: optimal finishers first (fewest expansions wins), then suboptimal, then no path, then
// the racers still searching.
function renderBoard() {
  const body = $('pf-board').querySelector('tbody');
  if (!S.racers.length) {
    body.innerHTML = '<tr><td colspan="6" class="field-hint">No race yet. Press RACE.</td></tr>';
    return;
  }
  const rank = (r) => {
    if (!r.done) return 3;
    if (!r.res.found) return 2;
    return S.optimal !== null && r.res.cost <= S.optimal + EPS ? 0 : 1;
  };
  const rows = [...S.racers].sort((a, b) => rank(a) - rank(b) || a.idx - b.idx || (a.res.cost ?? 0) - (b.res.cost ?? 0));
  let place = 0;
  body.innerHTML = rows.map((r) => {
    const finished = r.done;
    const verdict = finished ? verdictOf(r) : null;
    if (finished && rank(r) !== 3) place++;
    const cost = r.res.found ? r.res.cost.toFixed(2) : '—';
    const cls = verdict ? verdict.cls : 'wait';
    const text = verdict ? verdict.text : (r.done ? 'NO PATH ✗' : 'SEARCHING');
    return `<tr><td class="num">${finished && rank(r) !== 3 ? place : '·'}</td><td>${LABELS[r.algo]}</td>` +
      `<td class="num">${r.idx}</td><td class="num">${r.peak}</td><td class="num">${cost}</td>` +
      `<td><span class="pf-badge ${cls}">${text}</span></td></tr>`;
  }).join('');
  const note = S.optimal === null
    ? 'No route exists on this map, so nothing can be optimal.'
    : `Optimal cost on this map: ${S.optimal.toFixed(2)} (the Dijkstra reference). Ties go to fewer expansions.`;
  $('pf-board-note').textContent = note;
}

// Step: advance every racer by one expansion. Builds the race first if there is none.
function step() {
  if (S.state === 'done' || (S.state === 'idle' && !S.racers.length)) {
    if (S.state === 'done') resetRace();
    if (!buildRace()) return;
  }
  if (S.state === 'running') pause();
  S.state = 'paused';
  setStatus('PAUSED');
  refreshButtons();
  advanceAll(1);
  renderBoard();
  requestDraw();
}

function race() {
  if (S.state === 'done') resetRace();
  if (S.state === 'idle') {
    if (!buildRace()) return;
  }
  S.state = 'running';
  setStatus('RACING');
  refreshButtons();
  requestDraw();
}

function pause() {
  if (S.state !== 'running') return;
  S.state = 'paused';
  setStatus('PAUSED');
  refreshButtons();
}

function togglePlay() {
  if (S.state === 'running') pause();
  else if (S.state === 'paused' || S.state === 'idle' || S.state === 'done') race();
}

// ── Editing the map ────────────────────────────────────────────────────────

function markChanged() {
  S.version++;
  if (S.racers.length || S.state !== 'idle') resetRace();
  requestDraw();
}

function setHint(text) {
  $('pf-hint').textContent = text;
}

function generate(kind, size, seed, swamp) {
  try {
    const maze = makeMaze(kind, size, size, seed, { density: S.map.density, swamp });
    S.grid = maze.grid;
    S.start = maze.start;
    S.goal = maze.goal;
  } catch (err) {
    setHint(`Could not build that map: ${err.message}. Try a bigger size or another seed.`);
    return;
  }
  Object.assign(S.map, { kind, size, seed, swamp });
  $('map-note').textContent = KIND_NOTE[kind];
  markChanged();
  updateStatChips();
}

function paintCell(cell) {
  if (S.drag === 'start' || S.drag === 'goal') {
    if (S.grid.cells[cell] === WALL) S.grid.cells[cell] = OPEN;
    if (S.drag === 'start') S.start = cell; else S.goal = cell;
    markChanged();
    return;
  }
  const value = { wall: WALL, swamp: SWAMP, mud: MUD, erase: OPEN }[S.tool];
  if (value === WALL && (cell === S.start || cell === S.goal)) return;
  if (S.grid.cells[cell] === value) return;
  S.grid.cells[cell] = value;
  markChanged();
}

// Every cell on the segment between two points, so a fast drag leaves no gaps.
function lineCells(a, b) {
  const N = S.grid.width;
  const ax = a % N, ay = Math.floor(a / N), bx = b % N, by = Math.floor(b / N);
  const steps = Math.max(Math.abs(bx - ax), Math.abs(by - ay), 1);
  const out = [];
  for (let t = 0; t <= steps; t++) {
    const x = Math.round(ax + ((bx - ax) * t) / steps);
    const y = Math.round(ay + ((by - ay) * t) / steps);
    out.push(y * N + x);
  }
  return out;
}

function cellFromEvent(canvas, e) {
  const N = S.grid.width;
  const rect = canvas.getBoundingClientRect();
  const x = Math.floor(((e.clientX - rect.left) / rect.width) * N);
  const y = Math.floor(((e.clientY - rect.top) / rect.height) * N);
  if (x < 0 || y < 0 || x >= N || y >= N) return -1;
  return y * N + x;
}

function terrainName(v) {
  if (v === WALL) return 'wall';
  if (v === SWAMP) return 'swamp (cost 5)';
  if (v === MUD) return 'mud (cost 3)';
  return 'open (cost 1)';
}

// ── Hover: the explain-this-step card ─────────────────────────────────────

function showTip(e, html) {
  const tip = $('pf-tip');
  tip.innerHTML = html;
  tip.hidden = false;
  const w = tip.offsetWidth || 240;
  const x = e.clientX + 16 + w > window.innerWidth ? e.clientX - w - 12 : e.clientX + 16;
  tip.style.left = `${Math.max(4, x)}px`;
  tip.style.top = `${e.clientY + 16}px`;
}

function hideTip(view) {
  $('pf-tip').hidden = true;
  if (view && view.hoverCell !== -1) {
    view.hoverCell = -1;
    requestDraw();
  }
}

function setHover(view, cell) {
  if (view.hoverCell === cell) return;
  view.hoverCell = cell;
  requestDraw();
}

function hoverEditor(e) {
  const cell = cellFromEvent(S.editor.canvas, e);
  setHover(S.editor, cell);
  if (cell < 0) return hideTip();
  const N = S.grid.width;
  const x = cell % N, y = Math.floor(cell / N);
  const gx = S.goal % N, gy = Math.floor(S.goal / N);
  const h = heuristic($('heur').value, x, y, gx, gy);
  showTip(e, `<b>cell (${x}, ${y})</b><br>terrain: ${terrainName(S.grid.cells[cell])}<br>` +
    `h to goal (${$('heur').value}): ${h.toFixed(2)}`);
}

function hoverRacer(r, view, e) {
  const cell = cellFromEvent(view.canvas, e);
  setHover(view, cell);
  if (cell < 0) return hideTip();
  const N = S.grid.width;
  const x = cell % N, y = Math.floor(cell / N);
  const gx = S.goal % N, gy = Math.floor(S.goal / N);
  const hName = $('heur').value;
  const w = Number($('weight').value);
  const usesH = r.algo === 'greedy' || r.algo === 'astar' || r.algo === 'wastar';
  const h = heuristic(hName, x, y, gx, gy);
  const g = r.state[cell] ? r.g[cell] : null;
  let state;
  if (r.state[cell] === 0) state = 'not reached yet';
  else if (r.state[cell] === 1) state = 'in the frontier, waiting';
  else state = `expanded at step ${r.order[cell]}`;
  let fLine;
  if (r.algo === 'astar') fLine = g === null ? '—' : `f = g + h = ${(g + h).toFixed(2)}`;
  else if (r.algo === 'wastar') fLine = g === null ? '—' : `f = g + w·h = ${(g + w * h).toFixed(2)}`;
  else if (r.algo === 'greedy') fLine = `priority = h = ${h.toFixed(2)}`;
  else if (r.algo === 'ucs') fLine = g === null ? '—' : `priority = g = ${g.toFixed(2)}`;
  else fLine = 'no priority: this search orders by its own rule';
  const hLine = usesH ? `h (${hName}): ${h.toFixed(2)}` : `h (${hName}): ${h.toFixed(2)} (not used by ${LABELS[r.algo]})`;
  showTip(e, `<b>${LABELS[r.algo]}</b> · cell (${x}, ${y})<br>${state}<br>` +
    `g (cost so far): ${g === null ? '—' : g.toFixed(2)}<br>${hLine}<br>${fLine}` +
    `<span class="pf-tip-note">Lower f (or priority) is expanded first. Ties go to the cell pushed first.</span>`);
}

// ── Controls ───────────────────────────────────────────────────────────────

function settingChanged() {
  if (S.state !== 'idle') resetRace();
  requestDraw();
}

function updateHeurNote() {
  const diag = $('diag').checked;
  const heur = $('heur').value;
  const note = $('heur-note');
  if (heur === 'manhattan' && diag) {
    note.textContent = 'Manhattan overestimates a diagonal step (2 vs √2), so A* may return a longer path. The race will say so.';
    note.classList.add('warn');
  } else {
    note.textContent = heur === 'manhattan' ? 'Admissible on 4-way moves.' : 'Admissible on both 4- and 8-way moves.';
    note.classList.remove('warn');
  }
}

function setTool(tool) {
  S.tool = tool;
  document.querySelectorAll('.pf-tool').forEach((b) => b.setAttribute('aria-pressed', String(b.dataset.tool === tool)));
  setHint(TOOL_HINT[tool]);
}

function setMode(mode) {
  S.mode = mode;
  $('mode-panels').setAttribute('aria-pressed', String(mode === 'panels'));
  $('mode-focus').setAttribute('aria-pressed', String(mode === 'focus'));
  $('pf-panels').hidden = mode !== 'panels';
  $('pf-focus').hidden = mode !== 'focus';
  if (mode === 'focus') setFocus($('focus-algo').value);
  requestDraw();
}

function setFocus(algo) {
  S.focus = algo;
  $('focus-algo').value = algo;
  $('algo-desc').textContent = ALGO_TEXT[algo];
  if (!S.focusView) S.focusView = makeView($('pf-focus-canvas'), 'focus');
  S.focusView.racer = S.racers.find((r) => r.algo === algo) || null;
  const stats = $('pf-focus-stats');
  if (!S.focusView.racer) {
    stats.innerHTML = checkedAlgos().includes(algo)
      ? '<span class="field-hint">Press RACE to start this search.</span>'
      : '<span class="field-hint">This search is not in the race. Tick it under SEARCH.</span>';
  } else {
    stats.innerHTML = `<span><b>${LABELS[algo]}</b></span>`;
    updateFocusStats();
  }
  requestDraw();
}

function updateFocusStats() {
  const r = S.focusView && S.focusView.racer;
  if (!r) return;
  const v = r.done ? verdictOf(r) : null;
  $('pf-focus-stats').innerHTML = `<span>EXPANDED <b>${r.idx}</b></span><span>FRONTIER <b>${r.frontier}</b></span>` +
    `<span>GENERATED <b>${r.generated}</b></span><span>COST <b>${r.done && r.res.found ? r.res.cost.toFixed(2) : '—'}</b></span>` +
    `<span class="pf-badge ${v ? v.cls : 'wait'}">${v ? v.text : 'SEARCHING'}</span><span><b>${LABELS[r.algo]}</b></span>`;
}

function applyPreset(p) {
  $('kind').value = p.maze;
  $('size').value = String(p.size);
  $('seed').value = String(p.seed);
  $('swamp').value = String(p.swamp);
  $('swamp-val').textContent = `${p.swamp}%`;
  $('diag').checked = Boolean(p.diagonal);
  S.map.density = p.density ?? 28;
  $('preset-note').textContent = p.note || '';
  updateHeurNote();
  generate(p.maze, p.size, p.seed, p.swamp);
}

function randomSeed() {
  return 1 + Math.floor(Math.random() * 99998);
}

async function loadPresets() {
  try {
    const body = await getJSON('/api/pathfind/presets');
    S.presets = body.presets || [];
  } catch {
    S.presets = [];
  }
  const sel = $('preset');
  if (!S.presets.length) {
    sel.innerHTML = '<option value="">presets unavailable</option>';
    sel.disabled = true;
    generate('prim', 41, 3, 0);
    return;
  }
  sel.innerHTML = S.presets.map((p) => `<option value="${p.id}">${p.name}</option>`).join('');
  applyPreset(S.presets[0]);
  sel.value = S.presets[0].id;
}

// ── Wiring ─────────────────────────────────────────────────────────────────

function wire() {
  const canvas = $('pf-grid');
  S.editor = makeView(canvas, 'editor');
  canvas.addEventListener('pointerdown', (e) => {
    const cell = cellFromEvent(canvas, e);
    if (cell < 0) return;
    canvas.setPointerCapture(e.pointerId);
    S.drag = S.tool === 'start' || S.tool === 'goal' ? S.tool : 'paint';
    S.lastCell = cell;
    paintCell(cell);
    hoverEditor(e);
  });
  canvas.addEventListener('pointermove', (e) => {
    const cell = cellFromEvent(canvas, e);
    if (S.drag && cell >= 0) {
      if (S.drag === 'paint') {
        for (const c of lineCells(S.lastCell < 0 ? cell : S.lastCell, cell)) paintCell(c);
      } else {
        paintCell(cell);
      }
      S.lastCell = cell;
    }
    hoverEditor(e);
  });
  const end = () => {
    S.drag = null;
    S.lastCell = -1;
  };
  canvas.addEventListener('pointerup', end);
  canvas.addEventListener('pointercancel', end);
  canvas.addEventListener('pointerleave', () => {
    end();
    hideTip(S.editor);
  });

  document.querySelectorAll('.pf-tool').forEach((b) => b.addEventListener('click', () => setTool(b.dataset.tool)));
  $('btn-gen').addEventListener('click', () => {
    generate($('kind').value, Number($('size').value), Number($('seed').value) || 0, Number($('swamp').value));
  });
  $('btn-seed').addEventListener('click', () => {
    $('seed').value = String(randomSeed());
  });
  $('swamp').addEventListener('input', () => {
    $('swamp-val').textContent = `${$('swamp').value}%`;
  });
  $('preset').addEventListener('change', () => {
    const p = S.presets.find((x) => x.id === $('preset').value);
    if (p) applyPreset(p);
  });

  $('diag').addEventListener('change', () => {
    updateHeurNote();
    settingChanged();
  });
  $('heur').addEventListener('change', () => {
    updateHeurNote();
    settingChanged();
  });
  $('weight').addEventListener('input', () => {
    $('weight-val').textContent = Number($('weight').value).toFixed(1);
  });
  $('weight').addEventListener('change', settingChanged);
  document.querySelectorAll('#algos input').forEach((cb) => cb.addEventListener('change', settingChanged));

  $('btn-race').addEventListener('click', race);
  $('btn-pause').addEventListener('click', togglePlay);
  $('btn-step').addEventListener('click', step);
  $('btn-reset').addEventListener('click', resetRace);

  $('mode-panels').addEventListener('click', () => setMode('panels'));
  $('mode-focus').addEventListener('click', () => setMode('focus'));
  $('focus-algo').addEventListener('change', (e) => setFocus(e.target.value));
  $('speed').addEventListener('input', () => {
    S.rate = Math.max(1, Math.round(2 ** (Number($('speed').value) / 12)));
    $('speed-val').textContent = `${S.rate} per frame`;
  });
  window.addEventListener('keydown', (e) => {
    if (e.code === 'Space' && e.target === document.body) {
      e.preventDefault();
      togglePlay();
    }
  });
  window.addEventListener('resize', requestDraw);
}

function init() {
  wire();
  $('weight-val').textContent = DEFAULT_WEIGHT.toFixed(1);
  $('weight').value = String(DEFAULT_WEIGHT);
  $('speed').dispatchEvent(new Event('input'));
  $('swamp-val').textContent = '0%';
  $('heur').value = 'octile';
  updateHeurNote();
  setTool('wall');
  setFocus(S.focus);
  setMode('panels');
  refreshButtons();
  loadPresets();
}

init();
