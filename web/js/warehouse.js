// Warehouse robots page.
//
// The server plans; this page owns the map, the robots, and the drawing. A plan is one path per
// robot: paths[i][t] is robot i's [x, y] at time step t, and it stays on its goal after arriving.
// Playback draws the plan at a fractional time tau, interpolating between steps so motion looks
// smooth. Conflict-Based Search also returns a trace of its constraint tree. The tree is laid out
// once and revealed one expansion at a time, so its growth shows without any further requests.
//
// Coordinates are [x, y] with x to the right and y down, the same as the server.

import { formatSeconds, getJSON, postJSON } from './api.js';
import { banner, burst, shake } from './fx.js';

const $ = (id) => document.getElementById(id);
const REDUCED = matchMedia('(prefers-reduced-motion: reduce)').matches;
const TAU = Math.PI * 2;
const PALETTE = ['#00f5ff', '#ff00a0', '#00ff88', '#ffe600', '#9b00ff', '#ff8a00', '#b6ff00', '#58a6ff'];
const LETTERS = 'ABCDEFGH';           // one letter per robot; also the cap on robots (PALETTE has 8)
const RED = '#ff3b3b';
const PINK = '#ff00a0';
const STEPS_PER_SECOND = 15;          // 4 animation frames per time step at 60 fps
const SOLVE_LIMITS = { max_nodes: 2000, max_seconds: 2.0 };  // within the server's caps
const TREE_STEP_MS = 380;             // delay between revealed constraint-tree expansions
const PLANNERS = ['cbs', 'prioritized', 'independent'];
const NODE_W = 100, NODE_H = 46, GAP_X = 112, GAP_Y = 84, PAD = 20;

const MODE_HINTS = {
  shelves: 'Click a cell to toggle a shelf. Cells holding a robot or goal stay open.',
  start: 'Click a free cell to add a robot. Click a robot start to remove that robot.',
  goal: 'Click a free cell to set a goal: the selected robot first, else the first robot without one. ' +
        'Click a goal to remove it, or click a robot start to select that robot.',
};
const STATUS_LABEL = {
  collisions: 'COLLISIONS', failed: 'NO ROUTE', node_limit: 'NODE LIMIT', time_limit: 'TIME LIMIT',
  no_solution: 'NO PLAN EXISTS',
};
const STATUS_HELP = {
  node_limit: 'CBS hit its node budget before finding a collision-free plan, so there is no plan to show.',
  time_limit: 'CBS ran out of time before finding a collision-free plan. Try fewer robots or a smaller map.',
  no_solution: 'The constraint tree was exhausted: no collision-free plan exists for these starts and goals.',
  failed: 'Prioritized planning could not route a robot around the paths planned before it.',
  collisions: 'Independent paths ignore each other, so they collide. Each red flash is a rule the plan breaks.',
};

const state = {
  meta: null,
  layoutName: null,
  rows: [],              // map rows: '.' is floor, '#' is a shelf
  robots: [],            // [{start: [x, y] | null, goal: [x, y] | null}], robot i is letter LETTERS[i]
  mode: 'start',         // 'shelves' | 'start' | 'goal'
  selected: null,        // robot chosen in goal mode, so the next free cell becomes its goal
  hover: null,           // cell under the pointer while editing
  planner: 'cbs',
  result: null,          // the plan shown on the map: {planner, status, paths, stats, conflicts, trace}
  race: null,            // {cbs, prioritized, independent} results from the last race
  busy: false,
  tau: 0,                // playback position in time steps; fractional while playing
  tMax: 0,               // last time step of the plan
  playing: false,
  speed: 1,
  lastFrame: 0,
  tree: null,            // laid-out constraint tree for the current result
  reveal: 0,             // how many trace entries (expansions) are shown in the tree
  revealTimer: 0,
  treeKey: null,         // node picked in the tree
};

// ── Small helpers ──────────────────────────────────────────────────────

const letter = (i) => LETTERS[i] ?? String(i + 1);
const colorOf = (i) => PALETTE[i % PALETTE.length];
const eq = (a, b) => !!a && !!b && a[0] === b[0] && a[1] === b[1];
const cellW = () => state.rows[0].length;
const cellH = () => state.rows.length;
const isShelf = (x, y) => state.rows[y][x] === '#';
const maxRobots = () => Math.min(state.meta?.limits?.max_robots ?? 8, LETTERS.length);
const ctr = (cell, cs) => [(cell[0] + 0.5) * cs, (cell[1] + 0.5) * cs];
const cellText = (c) => `(${c[0]},${c[1]})`;

/** Show a message in the banner above the map. kind is '', 'warn', or 'err'. */
let bannerTimer = 0;
function notice(text, kind = '') {
  const el = $('banner');
  el.textContent = text;
  el.className = kind ? `banner ${kind}` : 'banner';
  el.hidden = false;
  clearTimeout(bannerTimer);
  bannerTimer = setTimeout(() => { el.hidden = true; }, 7000);
}

function setBusy(on) {
  state.busy = on;
  for (const id of ['btn-solve', 'btn-race', 'btn-random']) $(id).disabled = on;
}

// ── Plan results ───────────────────────────────────────────────────────

/** Label for the header and tree panel. CBS "solved" means optimal; the others only mean a plan exists. */
function statusLabel(planner, status) {
  if (status === 'solved') return planner === 'cbs' ? 'OPTIMAL' : 'PLAN FOUND';
  return STATUS_LABEL[status] ?? status.toUpperCase();
}

function statusClass(status) {
  if (status === 'solved') return 'wh-ok';
  if (status === 'node_limit' || status === 'time_limit') return 'wh-warn';
  return 'wh-bad';
}

/** Show a plan on the map and in the panels, and start the constraint-tree reveal. */
function setResult(res) {
  state.result = res;
  state.planner = res.planner;
  state.tMax = res.paths ? Math.max(0, ...res.paths.map((p) => p.length - 1)) : 0;
  state.tau = 0;
  state.playing = !!res.paths && !REDUCED;
  state.tree = res.planner === 'cbs' && res.trace?.length ? layoutTree(res.trace) : null;
  state.treeKey = null;
  startReveal();
  refreshAll();
}

/** Forget the plan (the map or the robots changed, so it no longer matches the inputs). */
function clearResult() {
  stopReveal();
  state.result = null;
  state.race = null;
  state.tree = null;
  state.tau = 0;
  state.tMax = 0;
  state.playing = false;
  state.treeKey = null;
  refreshAll();
}

function announce(res) {
  const stats = res.stats;
  if (res.status === 'solved') {
    if (res.planner === 'cbs') banner('OPTIMAL PLAN', `sum of costs ${stats.sum_of_costs}, makespan ${stats.makespan}`, '#00ff88');
    burst($('map'), { count: 70 });
    notice(`${statusLabel(res.planner, res.status)}: sum of costs ${stats.sum_of_costs}, makespan ${stats.makespan}, ${formatSeconds(stats.seconds)}`);
    return;
  }
  const kind = res.status === 'node_limit' || res.status === 'time_limit' ? 'warn' : 'err';
  if (res.status === 'collisions') shake($('map'));
  notice(STATUS_HELP[res.status] ?? statusLabel(res.planner, res.status), kind);
}

// ── Map editing ────────────────────────────────────────────────────────

function resetLayout(name) {
  const layout = state.meta.layouts.find((l) => l.name === name);
  if (!layout) return;
  state.layoutName = name;
  state.rows = [...layout.rows];
  state.robots = [];
  state.selected = null;
  clearResult();
}

/** Cell under a pointer event, or null if it is outside the map. */
function cellAt(ev) {
  const r = $('map').getBoundingClientRect();
  const x = Math.floor(((ev.clientX - r.left) / r.width) * cellW());
  const y = Math.floor(((ev.clientY - r.top) / r.height) * cellH());
  if (x < 0 || y < 0 || x >= cellW() || y >= cellH()) return null;
  return [x, y];
}

function onMapClick(ev) {
  if (state.busy) return;
  const cell = cellAt(ev);
  if (!cell) return;
  if (state.mode === 'shelves') toggleShelf(cell);
  else if (state.mode === 'start') placeStart(cell);
  else placeGoal(cell);
}

function toggleShelf([x, y]) {
  // A shelf cannot appear under a robot or goal, or the plan would start or end inside a rack.
  if (state.robots.some((r) => eq(r.start, [x, y]) || eq(r.goal, [x, y]))) {
    notice('A robot or goal is on that cell. Remove it first.', 'warn');
    return;
  }
  const row = state.rows[y].split('');
  row[x] = row[x] === '#' ? '.' : '#';
  state.rows[y] = row.join('');
  clearResult();
}

function placeStart(cell) {
  if (isShelf(...cell)) return notice('Robots cannot start on a shelf.', 'warn');
  const i = state.robots.findIndex((r) => eq(r.start, cell));
  if (i >= 0) {
    // Removing a robot also removes its goal; indices of later robots shift, so clear the selection.
    state.robots.splice(i, 1);
    state.selected = null;
  } else if (state.robots.length >= maxRobots()) {
    return notice(`At most ${maxRobots()} robots.`, 'warn');
  } else {
    state.robots.push({ start: cell, goal: null });
  }
  clearResult();
}

function placeGoal(cell) {
  if (isShelf(...cell)) return notice('Goals cannot be on a shelf.', 'warn');
  // Clicking an existing goal removes it.
  const g = state.robots.findIndex((r) => eq(r.goal, cell));
  if (g >= 0) {
    state.robots[g].goal = null;
    clearResult();
    return;
  }
  // Clicking a robot's start selects that robot as the target of the next click.
  const s = state.robots.findIndex((r) => eq(r.start, cell));
  if (s >= 0) {
    state.selected = s;
    notice(`Robot ${letter(s)} selected: click a free cell for its goal.`);
    refreshAll();
    return;
  }
  const target = state.selected ?? state.robots.findIndex((r) => !r.goal);
  if (target === null || target < 0) {
    return notice('Place a robot first (ROBOTS mode), or click a robot start to select it.', 'warn');
  }
  state.robots[target].goal = cell;
  state.selected = null;
  clearResult();
}

function setMode(mode) {
  state.mode = mode;
  state.selected = null;
  for (const b of document.querySelectorAll('.wh-mode')) {
    const on = b.dataset.mode === mode;
    b.classList.toggle('on', on);
    b.setAttribute('aria-pressed', String(on));
  }
  $('map-hint').textContent = MODE_HINTS[mode];
  $('mode-hint').textContent = MODE_HINTS[mode];
}

// ── Canvas drawing ─────────────────────────────────────────────────────

/**
 * Size a canvas for the map at a given CSS width and return a 2D context mapped to CSS pixels.
 * The backing store is multiplied by devicePixelRatio (capped at 2) so lines stay sharp.
 */
function sizeCanvas(canvas, rows, cssWidth, fluid = false) {
  const W = rows[0].length, H = rows.length;
  const cs = Math.max(6, Math.floor(cssWidth / W));   // cell size in CSS pixels
  const w = cs * W, h = cs * H;
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const bw = Math.round(w * dpr), bh = Math.round(h * dpr);
  if (canvas.width !== bw || canvas.height !== bh) {
    canvas.width = bw;
    canvas.height = bh;
  }
  // Fixed size for the main map; thumbnails scale to their column but keep the map's shape.
  canvas.style.width = fluid ? '100%' : `${w}px`;
  canvas.style.height = fluid ? 'auto' : `${h}px`;
  canvas.style.aspectRatio = `${W} / ${H}`;
  const ctx = canvas.getContext('2d');
  ctx.setTransform(bw / w, 0, 0, bh / h, 0, 0);
  return { ctx, cs, w, h };
}

/** Shelf: a neon block with two bars so it reads as a rack rather than a wall. */
function drawShelf(ctx, px, py, cs) {
  const pad = 2;
  ctx.fillStyle = 'rgba(255,0,160,0.2)';
  ctx.shadowColor = PINK;
  ctx.shadowBlur = 10;
  ctx.fillRect(px + pad, py + pad, cs - 2 * pad, cs - 2 * pad);
  ctx.shadowBlur = 0;
  ctx.strokeStyle = 'rgba(255,0,160,0.85)';
  ctx.lineWidth = 1.5;
  ctx.strokeRect(px + pad + 0.5, py + pad + 0.5, cs - 2 * pad - 1, cs - 2 * pad - 1);
  ctx.strokeStyle = 'rgba(255,0,160,0.45)';
  for (const f of [0.35, 0.65]) {
    ctx.beginPath();
    ctx.moveTo(px + pad + 3, py + cs * f);
    ctx.lineTo(px + cs - pad - 3, py + cs * f);
    ctx.stroke();
  }
}

/** Goal: a dashed ring in the robot's colour; solid once that robot has arrived. */
function drawGoal(ctx, cs, goal, i, arrived) {
  const [cx, cy] = ctr(goal, cs);
  ctx.save();
  ctx.strokeStyle = colorOf(i);
  ctx.lineWidth = 2;
  ctx.setLineDash(arrived ? [] : [4, 3]);
  ctx.globalAlpha = arrived ? 0.45 : 0.9;
  ctx.beginPath();
  ctx.arc(cx, cy, cs * 0.3, 0, TAU);
  ctx.stroke();
  ctx.restore();
}

/** Robot: a glowing disc with its letter. pos may be fractional while a step is in progress. */
function drawRobot(ctx, cs, pos, i, dim = false) {
  const cx = (pos[0] + 0.5) * cs, cy = (pos[1] + 0.5) * cs;
  const r = cs * 0.34;
  ctx.save();
  ctx.globalAlpha = dim ? 0.5 : 1;
  ctx.shadowColor = colorOf(i);
  ctx.shadowBlur = 16;
  ctx.fillStyle = colorOf(i);
  ctx.beginPath();
  ctx.arc(cx, cy, r, 0, TAU);
  ctx.fill();
  ctx.shadowBlur = 0;
  ctx.fillStyle = '#02040b';
  ctx.font = `700 ${Math.round(cs * 0.36)}px Orbitron, sans-serif`;
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  ctx.fillText(letter(i), cx, cy + 1);
  ctx.restore();
}

/** Where robot path is at (fractional) time tau: linear interpolation between the two cells around it. */
function interp(path, tau) {
  const L = path.length - 1;
  const j = Math.min(Math.floor(tau), L);
  const f = j >= L ? 0 : tau - j;
  const a = path[j], b = path[Math.min(j + 1, L)];
  return [a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f];
}

/** Glowing trail over the last few steps, fading toward the tail. */
function drawTrail(ctx, cs, path, tau, i) {
  const L = path.length - 1;
  const k = Math.min(Math.floor(tau), L);
  const from = Math.max(0, k - 6);
  const pts = [];
  for (let j = from; j <= k; j++) pts.push(ctr(path[j], cs));
  pts.push(ctr(interp(path, tau), cs));
  ctx.save();
  ctx.lineCap = 'round';
  ctx.strokeStyle = colorOf(i);
  ctx.lineWidth = cs * 0.22;
  ctx.shadowColor = colorOf(i);
  ctx.shadowBlur = 12;
  for (let s = 0; s < pts.length - 1; s++) {
    ctx.globalAlpha = ((s + 1) / pts.length) * 0.85;
    ctx.beginPath();
    ctx.moveTo(pts[s][0], pts[s][1]);
    ctx.lineTo(pts[s + 1][0], pts[s + 1][1]);
    ctx.stroke();
  }
  ctx.restore();
}

/** Red collision markers. Every conflict gets a faint X; the ones happening at this step flash. */
function drawConflicts(ctx, cs, conflicts, tau, hasPlan, now) {
  const step = Math.floor(tau);
  for (const c of conflicts) {
    const active = hasPlan && c.t === step;
    for (const cell of c.cells) {
      const [cx, cy] = ctr(cell, cs);
      if (active) {
        // Flash between full and half strength; under reduced motion the marker just stays bright.
        const pulse = REDUCED ? 1 : 0.55 + 0.45 * Math.sin(now / 70);
        ctx.save();
        ctx.globalAlpha = pulse;
        ctx.fillStyle = RED;
        ctx.shadowColor = RED;
        ctx.shadowBlur = 20;
        ctx.beginPath();
        ctx.arc(cx, cy, cs * 0.42, 0, TAU);
        ctx.fill();
        ctx.restore();
      } else {
        ctx.save();
        ctx.strokeStyle = 'rgba(255,59,59,0.55)';
        ctx.lineWidth = 2;
        const d = cs * 0.2;
        ctx.beginPath();
        ctx.moveTo(cx - d, cy - d); ctx.lineTo(cx + d, cy + d);
        ctx.moveTo(cx + d, cy - d); ctx.lineTo(cx - d, cy + d);
        ctx.stroke();
        ctx.restore();
      }
    }
  }
}

/**
 * Draw the map. With a plan (o.paths) robots are drawn at their position for time o.tau, with
 * trails and arrival flashes; without one they sit on their starts. Used for the main map and,
 * at the end of the plan, for each race thumbnail.
 */
function paintBoard(canvas, o) {
  const { ctx, cs, w, h } = sizeCanvas(canvas, o.rows, o.cssWidth, !!o.fluid);
  const now = performance.now();
  const W = o.rows[0].length, H = o.rows.length;
  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = '#03060f';
  ctx.fillRect(0, 0, w, h);

  // Faint floor grid so aisles read as cells.
  ctx.strokeStyle = 'rgba(0,245,255,0.08)';
  ctx.lineWidth = 1;
  ctx.beginPath();
  for (let x = 0; x <= W; x++) { ctx.moveTo(x * cs + 0.5, 0); ctx.lineTo(x * cs + 0.5, h); }
  for (let y = 0; y <= H; y++) { ctx.moveTo(0, y * cs + 0.5); ctx.lineTo(w, y * cs + 0.5); }
  ctx.stroke();

  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      if (o.rows[y][x] === '#') drawShelf(ctx, x * cs, y * cs, cs);
    }
  }

  const hasPlan = !!o.paths;
  const robots = o.robots ?? [];
  if (o.hover && !hasPlan) {
    ctx.save();
    ctx.strokeStyle = 'rgba(0,245,255,0.8)';
    ctx.setLineDash([3, 3]);
    ctx.strokeRect(o.hover[0] * cs + 2, o.hover[1] * cs + 2, cs - 4, cs - 4);
    ctx.restore();
  }

  robots.forEach((r, i) => {
    if (!r.goal) return;
    const arrived = hasPlan && o.tau >= o.paths[i].length - 1;
    drawGoal(ctx, cs, r.goal, i, arrived);
  });

  if (hasPlan) {
    o.paths.forEach((p, i) => drawTrail(ctx, cs, p, o.tau, i));
    o.paths.forEach((p, i) => {
      const L = p.length - 1;
      const pos = interp(p, o.tau);
      drawRobot(ctx, cs, pos, i);
      // Arrival flash: a bright ring for one time step after the robot reaches its goal, then a
      // steady ring while it waits there.
      if (o.tau >= L && L > 0) {
        const since = o.tau - L;
        const [cx, cy] = ctr(p[L], cs);
        const flash = !REDUCED && since < 1;
        ctx.save();
        ctx.strokeStyle = colorOf(i);
        ctx.lineWidth = flash ? 4 : 2;
        ctx.shadowColor = colorOf(i);
        ctx.shadowBlur = flash ? 24 : 8;
        ctx.globalAlpha = flash ? 1 - since * 0.5 : 0.6;
        ctx.beginPath();
        ctx.arc(cx, cy, cs * (flash ? 0.5 + since * 0.2 : 0.42), 0, TAU);
        ctx.stroke();
        ctx.restore();
      }
    });
  } else {
    robots.forEach((r, i) => {
      if (r.start) drawRobot(ctx, cs, r.start, i, !r.goal);
    });
  }

  drawConflicts(ctx, cs, o.conflicts ?? [], o.tau, hasPlan, now);

  if (o.selected !== null && o.selected !== undefined && robots[o.selected]?.start) {
    const [cx, cy] = ctr(robots[o.selected].start, cs);
    ctx.save();
    ctx.strokeStyle = '#ffffff';
    ctx.lineWidth = 2;
    ctx.setLineDash([3, 3]);
    ctx.beginPath();
    ctx.arc(cx, cy, cs * 0.5, 0, TAU);
    ctx.stroke();
    ctx.restore();
  }
}

function paintMain() {
  if (!state.rows.length) return;   // the map is loaded asynchronously; nothing to draw yet
  const wrap = $('map').parentElement;
  paintBoard($('map'), {
    rows: state.rows,
    cssWidth: Math.min(660, wrap.clientWidth - 14),
    robots: state.robots,
    paths: state.result?.paths ?? null,
    tau: state.tau,
    conflicts: state.result?.conflicts ?? [],
    hover: state.hover,
    selected: state.selected,
  });
}

// ── Playback ───────────────────────────────────────────────────────────

/** One animation frame: advance the playhead while playing, then redraw the map. */
function frame(now) {
  const dt = state.lastFrame ? Math.min(100, now - state.lastFrame) : 16;
  state.lastFrame = now;
  if (state.playing && state.result?.paths) {
    state.tau = Math.min(state.tMax, state.tau + (dt / 1000) * STEPS_PER_SECOND * state.speed);
    if (state.tau >= state.tMax) state.playing = false;   // stop at the end of the plan
    syncPlayback();
  }
  paintMain();
  requestAnimationFrame(frame);
}

function togglePlay() {
  if (!state.result?.paths) return;
  if (!state.playing && state.tau >= state.tMax) state.tau = 0;   // replay from the start
  state.playing = !state.playing;
  syncPlayback();
}

function stepBy(delta) {
  if (!state.result?.paths) return;
  state.playing = false;
  // Stepping back from a fractional time lands on the previous whole step, not one before it.
  state.tau = delta < 0
    ? Math.max(0, Math.ceil(state.tau) - 1)
    : Math.min(state.tMax, Math.floor(state.tau) + 1);
  syncPlayback();
}

/** Update the timeline, time label, play button, and the collision listing for the current step. */
function syncPlayback() {
  const has = !!state.result?.paths;
  const tl = $('timeline');
  tl.max = String(Math.round(state.tMax * 100));
  tl.value = String(Math.round(state.tau * 100));
  tl.disabled = !has;
  for (const id of ['btn-back', 'btn-play', 'btn-fwd']) $(id).disabled = !has;
  $('time-lbl').textContent = has ? `t = ${Math.floor(state.tau)} / ${state.tMax}` : 't = —';
  $('btn-play').querySelector('.btn-txt').textContent = state.playing ? '❚❚ PAUSE' : '▶ PLAY';

  // Show the collision that happens at this step, and highlight its entry in the list.
  const step = Math.floor(state.tau);
  const here = has ? (state.result.conflicts ?? []).filter((c) => c.t === step) : [];
  $('conflict-note').textContent = here.length
    ? here.map(describeConflict).join('. ') + '.'
    : defaultConflictNote();
  for (const li of document.querySelectorAll('#conflicts li[data-t]')) {
    li.classList.toggle('now', has && li.dataset.t === String(step));
  }
}

function defaultConflictNote() {
  const res = state.result;
  if (!res) return 'Run a plan to see where robots would collide.';
  const n = res.conflicts?.length ?? 0;
  if (!n) return 'No collisions: no two robots share a cell or swap places.';
  return `${n} collision${n === 1 ? '' : 's'} in this plan.`;
}

// ── Descriptions ───────────────────────────────────────────────────────

function describeConstraint(c) {
  if (!c) return '';
  const who = `Robot ${letter(c.robot)}`;
  if (c.kind === 'vertex') return `${who} may not be at ${cellText(c.cell)} at t=${c.t}`;
  const src = c.src ?? c.from, dst = c.dst ?? c.to;
  return `${who} may not move ${cellText(src)} → ${cellText(dst)} at t=${c.t}`;
}

function describeConflict(c) {
  const [i, j] = c.robots;
  const who = `${letter(i)} and ${letter(j)}`;
  if (c.kind === 'vertex') return `${who} both reach ${cellText(c.cells[0])} at t=${c.t}`;
  return `${who} swap ${cellText(c.cells[0])} and ${cellText(c.cells[1])} at t=${c.t}`;
}

function describeNode(n) {
  const lines = [];
  if (n.added) lines.push(`Adds: ${describeConstraint(n.added)}.`);
  else lines.push('Root: every robot planned alone, ignoring the others.');
  if (n.dead) lines.push('Dead end: no route satisfies these constraints.');
  else if (n.cost !== null && n.cost !== undefined) lines.push(`Cost ${n.cost} (sum of arrival times).`);
  if (n.conflict) lines.push(`Collision it resolves: ${describeConflict(n.conflict)}.`);
  return lines.join('\n');
}

// ── Constraint tree ────────────────────────────────────────────────────

/**
 * Turn CBS's trace into a laid-out tree. Each trace entry is one expanded node: its id, cost,
 * the collision it resolved, and its two children. A child with no id had no route (a dead end).
 * Leaves get x positions in order and parents sit over the middle of their children.
 */
function layoutTree(trace) {
  const nodes = new Map();
  const root = { key: `n${trace[0].id}`, id: trace[0].id, cost: trace[0].cost, parentIdx: -1, added: null,
    children: [], expanded: false, conflict: null, dead: false };
  nodes.set(root.key, root);
  trace.forEach((entry, idx) => {
    const node = nodes.get(`n${entry.id}`);
    if (!node) return;   // cannot happen: a node is created before the entry that expands it
    node.expanded = true;
    node.conflict = entry.conflict;
    entry.children.forEach((ch, k) => {
      const dead = ch.cost === null || ch.cost === undefined;
      const key = dead ? `d${idx}-${k}` : `n${ch.id}`;
      const child = { key, id: dead ? null : ch.id, cost: dead ? null : ch.cost, parentIdx: idx, parent: node,
        added: ch.constraint, children: [], expanded: false, conflict: null, dead };
      node.children.push(child);
      nodes.set(key, child);
    });
  });

  const list = [];
  let leaves = 0, depth = 0;
  const place = (node, d) => {
    node.depth = d;
    list.push(node);
    depth = Math.max(depth, d);
    if (!node.children.length) node.x = leaves++;
    else {
      node.children.forEach((c) => place(c, d + 1));
      node.x = (node.children[0].x + node.children[node.children.length - 1].x) / 2;
    }
  };
  place(root, 0);
  return { root, list, leaves, depth };
}

/** Draw the tree as it has been revealed so far. The full layout is fixed, so nodes never move. */
function paintTree() {
  const svg = $('tree');
  svg.replaceChildren();
  const t = state.tree;
  if (!t) return;
  const width = Math.max(1, t.leaves) * GAP_X + PAD * 2;
  const height = (t.depth + 1) * GAP_Y + PAD * 2;
  svg.setAttribute('width', width);
  svg.setAttribute('height', height);
  svg.setAttribute('viewBox', `0 0 ${width} ${height}`);

  const px = (n) => PAD + GAP_X / 2 + n.x * GAP_X;
  const py = (n) => PAD + NODE_H / 2 + n.depth * GAP_Y;
  const shown = (n) => n.parentIdx < state.reveal;   // a node appears when its parent is expanded
  const nowId = state.reveal > 0 ? state.result.trace[state.reveal - 1].id : null;
  const solutionCost = state.result?.status === 'solved' ? state.result.stats.sum_of_costs : null;
  let solutionMarked = false;

  // Edges first, so node boxes sit on top of them.
  for (const n of t.list) {
    if (!n.parent || !shown(n)) continue;
    const parent = n.parent;
    const x1 = px(parent), y1 = py(parent) + NODE_H / 2;
    const x2 = px(n), y2 = py(n) - NODE_H / 2;
    const ym = (y1 + y2) / 2;
    const e = el('path', { class: `wh-edge${n.dead ? ' dead' : ''}`, d: `M${x1} ${y1} C${x1} ${ym} ${x2} ${ym} ${x2} ${y2}` });
    svg.appendChild(e);
    if (n.added) {
      const label = el('text', { class: 'wh-edge-lbl', x: (x1 + x2) / 2 + 4, y: ym - 2, 'text-anchor': 'start' });
      label.textContent = shortConstraint(n.added);
      svg.appendChild(label);
    }
  }

  for (const n of t.list) {
    if (!shown(n)) continue;
    // The solution is the cheapest leaf that was never expanded: a node with no collisions left.
    let solution = false;
    if (solutionCost !== null && !n.expanded && !n.dead && n.cost === solutionCost && !solutionMarked) {
      solution = true;
      solutionMarked = true;
    }
    const classes = ['wh-node'];
    if (n.expanded) classes.push('expanded');
    if (n.dead) classes.push('dead');
    if (solution) classes.push('solution');
    if (nowId !== null && n.id === nowId) classes.push('now');
    if (state.treeKey === n.key) classes.push('selected');
    const g = el('g', { class: classes.join(' '), transform: `translate(${px(n) - NODE_W / 2} ${py(n) - NODE_H / 2})` });
    g.setAttribute('tabindex', '0');
    g.appendChild(el('rect', { width: NODE_W, height: NODE_H, rx: 6 }));
    const title = el('text', { class: 'wh-title', x: NODE_W / 2, y: 17, 'text-anchor': 'middle' });
    title.textContent = n.dead ? 'no route' : n.id === null ? '' : n.id === 0 ? `root · ${n.cost}` : `#${n.id} · ${n.cost}`;
    g.appendChild(title);
    const sub = el('text', { x: NODE_W / 2, y: 34, 'text-anchor': 'middle' });
    sub.textContent = n.conflict ? shortConflict(n.conflict) : solution ? 'solution' : n.expanded ? '' : 'open';
    g.appendChild(sub);
    const pick = () => { state.treeKey = n.key; paintTree(); $('tree-detail').textContent = describeNode(n); };
    g.addEventListener('click', pick);
    g.addEventListener('keydown', (ev) => { if (ev.key === 'Enter') pick(); });
    svg.appendChild(g);
    if (state.treeKey === n.key) $('tree-detail').textContent = describeNode(n);
  }
}

/** Create an SVG element with attributes. */
function el(tag, attrs) {
  const node = document.createElementNS('http://www.w3.org/2000/svg', tag);
  for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, String(v));
  return node;
}

function shortConstraint(c) {
  if (c.kind === 'vertex') return `${letter(c.robot)}@${cellText(c.cell)} t${c.t}`;
  return `${letter(c.robot)} ${cellText(c.src ?? c.from)}>${cellText(c.dst ?? c.to)} t${c.t}`;
}

function shortConflict(c) {
  return `t${c.t} ${letter(c.robots[0])}×${letter(c.robots[1])} ${c.kind === 'edge' ? 'swap' : 'hit'}`;
}

/** Reveal the tree one expansion at a time, or all at once when motion is reduced. */
function startReveal() {
  stopReveal();
  const n = state.result?.trace?.length ?? 0;
  state.reveal = !state.tree || REDUCED ? n : 0;
  paintTree();
  if (state.tree && state.reveal < n) state.revealTimer = setTimeout(stepReveal, TREE_STEP_MS);
  updateTreeHeader();
}

function stepReveal() {
  const n = state.result?.trace?.length ?? 0;
  state.reveal = Math.min(n, state.reveal + 1);
  paintTree();
  updateTreeHeader();
  if (state.reveal < n) state.revealTimer = setTimeout(stepReveal, TREE_STEP_MS);
}

function skipReveal() {
  stopReveal();
  state.reveal = state.result?.trace?.length ?? 0;
  paintTree();
  updateTreeHeader();
}

function stopReveal() {
  clearTimeout(state.revealTimer);
  state.revealTimer = 0;
}

function updateTreeHeader() {
  const res = state.result;
  const n = res?.trace?.length ?? 0;
  const skip = $('tree-skip');
  skip.hidden = !state.tree || state.reveal >= n;
  const title = $('tree-title');
  if (!res) {
    title.textContent = 'Solve with Conflict-Based Search to grow its constraint tree.';
    return;
  }
  if (res.planner !== 'cbs') {
    title.textContent = 'Only Conflict-Based Search builds a constraint tree.';
    return;
  }
  if (!n) {
    title.textContent = res.status === 'solved'
      ? 'The root plan already had no collisions, so the tree stops at once.'
      : STATUS_HELP[res.status] ?? 'No constraint tree was built.';
    return;
  }
  const shown = state.reveal;
  title.textContent = `Expanded ${shown} of ${n} shown nodes: each expansion resolves one collision`;
}

function renderTreePanel() {
  const res = state.result;
  const isCbs = res?.planner === 'cbs';
  $('cbs-stats').hidden = !isCbs;
  if (!res) return;
  if (isCbs) {
    const s = res.stats;
    $('cs-high').textContent = s.high_nodes;
    $('cs-exp').textContent = s.expanded_nodes;
    $('cs-res').textContent = s.conflicts_resolved;
    $('cs-low').textContent = s.low_expansions.toLocaleString('en-US');
    $('cs-sum').textContent = s.sum_of_costs ?? '—';
    $('cs-sec').textContent = formatSeconds(s.seconds);
  }
  updateTreeHeader();
}

// ── Panels and chips ───────────────────────────────────────────────────

function renderConflicts() {
  const list = $('conflicts');
  list.replaceChildren();
  const res = state.result;
  if (!res) return;
  const conflicts = res.conflicts ?? [];
  if (!conflicts.length) {
    const li = document.createElement('li');
    li.className = 'none';
    li.textContent = 'No collisions in this plan.';
    list.appendChild(li);
    return;
  }
  for (const c of conflicts) {
    const li = document.createElement('li');
    li.dataset.t = String(c.t);
    li.textContent = describeConflict(c);
    // Clicking a collision jumps the playhead to its time step and pauses there.
    li.addEventListener('click', () => {
      state.playing = false;
      state.tau = c.t;
      syncPlayback();
    });
    list.appendChild(li);
  }
}

function refreshChips() {
  const res = state.result;
  $('chip-robots').textContent = String(state.robots.filter((r) => r.goal).length);
  $('chip-sum').textContent = res?.stats?.sum_of_costs ?? '—';
  $('chip-span').textContent = res?.stats?.makespan ?? '—';
  $('chip-conf').textContent = res ? String(res.stats.conflicts) : '—';
  const status = $('chip-status');
  if (res) {
    status.textContent = statusLabel(res.planner, res.status);
    status.className = `val ${statusClass(res.status)}`;
  } else {
    status.textContent = state.robots.length ? 'PLACE GOALS' : 'PLACE ROBOTS';
    status.className = 'val';
  }
}

function renderRace() {
  const body = $('race-body');
  body.replaceChildren();
  const race = state.race;
  for (const p of PLANNERS) {
    const fig = document.querySelector(`#race canvas[data-planner="${p}"]`).parentElement;
    fig.classList.remove('best');
    if (!race) {
      paintBoard(fig.querySelector('canvas'), { rows: state.rows, cssWidth: 240, fluid: true, robots: state.robots, paths: null, tau: 0, conflicts: [] });
      continue;
    }
    const r = race[p];
    const best = bestRaceSum();
    const isBest = r.status === 'solved' && r.stats.sum_of_costs === best;
    fig.classList.toggle('best', isBest);
    fig.querySelector('figcaption').textContent = `${state.meta.planners.find((x) => x.name === p)?.title ?? p} · ${statusLabel(p, r.status)}`;
    paintBoard(fig.querySelector('canvas'), {
      rows: state.rows, cssWidth: 240, fluid: true, robots: state.robots, paths: r.paths,
      tau: r.paths ? Math.max(0, ...r.paths.map((x) => x.length - 1)) : 0,
      conflicts: r.conflicts ?? [],
    });
    const tr = document.createElement('tr');
    if (isBest) tr.className = 'best';
    else if (r.status !== 'solved') tr.className = 'bad';
    const cells = [
      state.meta.planners.find((x) => x.name === p)?.title ?? p,
      statusLabel(p, r.status),
      r.stats.sum_of_costs ?? '—',
      r.stats.makespan ?? '—',
      r.stats.conflicts,
      formatSeconds(r.stats.seconds),
    ];
    cells.forEach((text, k) => {
      const td = document.createElement('td');
      td.textContent = String(text);
      if (k >= 2) td.className = 'num';
      tr.appendChild(td);
    });
    body.appendChild(tr);
  }
}

/** The cheapest sum of costs among the race's collision-free plans (null if none). */
function bestRaceSum() {
  const sums = PLANNERS.map((p) => state.race[p]).filter((r) => r.status === 'solved').map((r) => r.stats.sum_of_costs);
  return sums.length ? Math.min(...sums) : null;
}

function refreshAll() {
  refreshChips();
  renderConflicts();
  renderTreePanel();
  syncPlayback();
  renderRace();
}

// ── Server calls ───────────────────────────────────────────────────────

function requestBody(planner) {
  return {
    rows: state.rows,
    starts: state.robots.map((r) => r.start),
    goals: state.robots.map((r) => r.goal),
    planner,
    ...SOLVE_LIMITS,
  };
}

/** Check the robots are complete before asking the server. Returns true when ready. */
function readyToSolve() {
  if (!state.robots.length) {
    notice('Place at least one robot, or randomize.', 'warn');
    return false;
  }
  if (state.robots.some((r) => !r.goal)) {
    notice('Every robot needs a goal. Switch to GOALS mode to place the missing ones.', 'warn');
    return false;
  }
  return true;
}

async function randomize() {
  if (state.busy) return;
  const robots = Number($('robots').value);
  const seed = Math.max(0, Math.floor(Number($('seed').value) || 0));
  setBusy(true);
  try {
    // Some draws on crowded maps have no solvable route within the server's budget. The server answers
    // 400 "no solvable instance", so retry with the next seed (up to three tries) before giving up.
    const MAX_TRIES = 3;
    let res = null, used = seed;
    for (let attempt = 0; attempt < MAX_TRIES; attempt++) {
      used = seed + attempt;
      try {
        res = await postJSON('/api/warehouse/random', { rows: state.rows, robots, seed: used });
        break;
      } catch (err) {
        if (!/no solvable instance/.test(err.message) || attempt === MAX_TRIES - 1) throw err;
      }
    }
    // Show the seed that actually worked, so the input matches the instance on screen.
    $('seed').value = used;
    state.robots = res.starts.map((s, i) => ({ start: s, goal: res.goals[i] }));
    state.selected = null;
    clearResult();
    notice(`Seed ${used}: ${robots} robots with solvable routes.`);
  } catch (err) {
    notice(err.message, 'err');
  } finally {
    setBusy(false);
  }
}

async function solve() {
  if (state.busy || !readyToSolve()) return;
  setBusy(true);
  try {
    const res = await postJSON('/api/warehouse/solve', requestBody($('planner').value));
    setResult(res);
    announce(res);
  } catch (err) {
    notice(err.message, 'err');
    shake($('map'));
  } finally {
    setBusy(false);
  }
}

/** Run all three planners on the same instance, one after another, and crown the cheapest valid plan. */
async function race() {
  if (state.busy || !readyToSolve()) return;
  setBusy(true);
  const out = {};
  try {
    for (const p of PLANNERS) out[p] = await postJSON('/api/warehouse/solve', requestBody(p));
    state.race = out;
    setResult(out[$('planner').value]);   // the selected planner's plan is the one on the map (refreshes panels)
    const best = bestRaceSum();
    if (best === null) {
      notice('No planner produced a collision-free plan for these robots.', 'err');
    } else {
      const winner = PLANNERS.find((p) => out[p].status === 'solved' && out[p].stats.sum_of_costs === best);
      const title = state.meta.planners.find((x) => x.name === winner)?.title ?? winner;
      banner(`${title.toUpperCase()} WINS`, `shortest valid plan: sum of costs ${best}`, winner === 'cbs' ? '#00ff88' : '#00f5ff');
      burst($('race'), { count: 90 });
    }
  } catch (err) {
    notice(err.message, 'err');
  } finally {
    setBusy(false);
  }
}

// ── Wiring ─────────────────────────────────────────────────────────────

function populateControls() {
  const layoutSel = $('layout');
  layoutSel.replaceChildren();
  for (const l of state.meta.layouts) {
    const opt = document.createElement('option');
    opt.value = l.name;
    opt.textContent = l.title;
    layoutSel.appendChild(opt);
  }
  const limit = maxRobots();
  $('robots').max = String(limit);
  if (Number($('robots').value) > limit) $('robots').value = String(limit);
  $('robots-lbl').textContent = $('robots').value;
  describePlanner();
}

function describePlanner() {
  const info = state.meta?.planners.find((p) => p.name === $('planner').value);
  $('planner-desc').textContent = info ? info.description : '';
}

function wire() {
  $('map').addEventListener('click', onMapClick);
  $('map').addEventListener('mousemove', (ev) => { state.hover = cellAt(ev); });
  $('map').addEventListener('mouseleave', () => { state.hover = null; });

  for (const b of document.querySelectorAll('.wh-mode')) {
    b.addEventListener('click', () => setMode(b.dataset.mode));
  }
  $('btn-play').addEventListener('click', togglePlay);
  $('btn-back').addEventListener('click', () => stepBy(-1));
  $('btn-fwd').addEventListener('click', () => stepBy(1));
  $('timeline').addEventListener('input', (ev) => {
    state.playing = false;
    state.tau = Number(ev.target.value) / 100;
    syncPlayback();
  });
  $('speed').addEventListener('change', (ev) => { state.speed = Number(ev.target.value); });
  $('tree-skip').addEventListener('click', skipReveal);

  $('layout').addEventListener('change', async (ev) => {
    resetLayout(ev.target.value);
    await randomize();
  });
  $('btn-reset').addEventListener('click', () => resetLayout(state.layoutName));
  $('robots').addEventListener('input', (ev) => { $('robots-lbl').textContent = ev.target.value; });
  $('btn-random').addEventListener('click', randomize);
  $('btn-clear').addEventListener('click', () => {
    if (state.busy) return;
    state.robots = [];
    state.selected = null;
    clearResult();
  });

  $('planner').addEventListener('change', (ev) => {
    describePlanner();
    // If a race has run, switch the map to that planner's plan without re-solving.
    if (state.race) setResult(state.race[ev.target.value]);
    else refreshAll();
  });
  $('btn-solve').addEventListener('click', solve);
  $('btn-race').addEventListener('click', race);
}

async function init() {
  wire();
  setMode('start');
  state.meta = await getJSON('/api/warehouse/meta');
  populateControls();
  resetLayout(state.meta.layouts[0].name);
  $('layout').value = state.layoutName;
  refreshAll();
  requestAnimationFrame(frame);   // start drawing only once there is a map to draw
  await randomize();
}

init().catch((err) => notice(`Could not load the warehouse: ${err.message}`, 'err'));
