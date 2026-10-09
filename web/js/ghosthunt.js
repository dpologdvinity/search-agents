// Ghost Hunt page.
//
// The hidden Markov model runs here, in ghosthunt-core.js: the exact forward filter or the particle filter,
// one per ghost, is updated after every turn. This file draws the board, runs the autopilot timer, and turns
// keys, buttons and clicks into actions. Colours: each ghost has its own fog colour (cyan, pink, yellow); the
// fog brightness is that ghost's belief about each cell. Particles are dots at their cells. A sonar ring grows
// from the player to the reading the ping reported. Busted ghosts stay on the board where they were caught.
// The Viterbi replay animates, after a bust, the most likely path of that ghost (cyan) against its true path
// (pink dashed).

import { Game, choose, NOISE, SIZES } from './ghosthunt-core.js';
import { banner, burst, shake } from './fx.js';

const $ = (id) => document.getElementById(id);
const GHOST_COL = ['#00f5ff', '#ff00a0', '#ffe600'];
const RING_MS = 800;   // how long a sonar ring takes to fade
const REPLAY_MS = 170; // time per turn in the Viterbi replay
const FILTER_NAME = { exact: 'EXACT', particles: 'PARTICLES' };
const MOTION_NAME = { random: 'RANDOM', lurker: 'LURKER', patrol: 'PATROL' };

const cfg = { size: 15, ghosts: 2, motion: 'random', noise: 'med', filter: 'exact', particles: 300, seed: 7 };
const ui = { bust: false, auto: false, autoTimer: 0, cheat: false, showParticles: false, delay: 350 };

const canvas = $('gh-map');
const ctx = canvas.getContext('2d');
const fog = document.createElement('canvas'); // belief fog is painted here and reused until the belief changes
const fctx = fog.getContext('2d');

let game = null;
let rings = [];          // sonar rings: {x, y, rad, t0, color}
let replay = null;       // {gid, t, trace, last, matched}
let busted = [];         // ghost ids in bust order, for the replay button
let replayIdx = 0;
let fogDirty = true;
let baseDirty = true;    // the static layer (background, grid, walls, corridor floor) is cached on base
const base = document.createElement('canvas');
const bctx = base.getContext('2d');
let cell = 20;           // css pixels per maze cell
let dpr = 1;

// ── small helpers ────────────────────────────────────────────────────────

function log(text, cls = 'log-info') {
  const line = document.createElement('div');
  line.className = cls;
  line.textContent = text;
  $('log').appendChild(line);
  $('log').scrollTop = $('log').scrollHeight;
}

function note(text) {
  $('gh-info').textContent = text;
}

// Open-cell index -> (row, col) on the board.
function rc(k) {
  const c = game.maze.cells[k], n = game.maze.n;
  return [Math.floor(c / n), c % n];
}

function label(k) {
  const [r, c] = rc(k);
  return `(${c},${r})`;
}

function argmax(values) {
  let best = -1, bi = 0;
  values.forEach((v, i) => { if (v > best) { best = v; bi = i; } });
  return bi;
}

// Centre of open cell k in css pixels.
function centre(k) {
  const [r, c] = rc(k);
  return [(c + 0.5) * cell, (r + 0.5) * cell];
}

// ── the hunt ─────────────────────────────────────────────────────────────

function newHunt(seed = cfg.seed) {
  stopAuto();
  cfg.seed = seed;
  game = new Game({
    size: cfg.size, ghosts: cfg.ghosts, model: cfg.motion, sigma: NOISE[cfg.noise], seed: cfg.seed,
    filt: cfg.filter, particles: cfg.particles,
  });
  rings = [];
  replay = null;
  busted = [];
  replayIdx = 0;
  fogDirty = true;
  baseDirty = true;
  $('log').textContent = '';
  const pings = game.ghosts.map((gh) => `ghost ${gh.id + 1} ${gh.readings[0]}`).join(', ');
  log(`new hunt: maze ${cfg.size}x${cfg.size}, seed ${cfg.seed}, ${cfg.ghosts} ghost(s), ${cfg.motion} motion, `
    + `sonar sigma ${NOISE[cfg.noise]}, ${FILTER_NAME[cfg.filter].toLowerCase()} belief. First pings: ${pings}.`);
  note('Click a cell next to you, or use the keys, to move. Bust a ghost by firing at its cell.');
  updateAll();
}

// Play one action. Returns false (and says why) when it is illegal or the hunt is over.
function act(action) {
  if (!game || game.done) return false;
  if (!game.legalActions().includes(action)) {
    note(action[0] === 'b' ? 'nothing to bust in that direction (a wall)' : 'that way is a wall');
    return false;
  }
  replay = null; // a new action ends any replay overlay
  let res;
  try {
    res = game.act(action);
  } catch (e) {
    note(e.message);
    return false;
  }
  const now = performance.now();
  const [px, py] = centre(game.p);
  for (const [gid, r] of Object.entries(res.readings)) {
    rings.push({ x: px, y: py, rad: (r + 0.5) * cell, t0: now, color: GHOST_COL[gid] });
  }
  const pingText = Object.entries(res.readings).map(([gid, r]) => `g${Number(gid) + 1}=${r}`).join(' ');
  const moved = action.length === 1 && action !== '.' ? ` -> at ${label(game.p)}` : '';
  log(`turn ${game.t}: ${action}${moved}; pings ${pingText || '(none)'}`);
  for (const gid of res.busted) {
    busted.push(gid);
    const [bx, by] = centre(game.ghosts[gid].true[game.ghosts[gid].bustTurn]);
    const rect = canvas.getBoundingClientRect();
    burst([rect.left + bx, rect.top + by], { count: 90 });
    shake(canvas, 'big');
    banner('GHOST BUSTED', `ghost ${gid + 1} caught on turn ${game.t - 1}`, GHOST_COL[gid]);
    log(`BUST: ghost ${gid + 1} was on ${label(game.ghosts[gid].true[game.ghosts[gid].bustTurn])}`, 'log-info');
  }
  if (game.done) finish();
  fogDirty = true;
  updateAll();
  return true;
}

function finish() {
  stopAuto();
  const won = game.busted === game.ghosts.length;
  banner(won ? 'ALL GHOSTS BUSTED' : 'OUT OF TURNS', won ? `in ${game.t} turns` : `${game.busted} of ${game.ghosts.length} caught`,
    won ? '#00ff88' : '#ff00a0');
  log(won ? `all ${game.ghosts.length} ghosts busted in ${game.t} turns.` : `out of turns after ${game.t}.`);
  for (const gh of game.ghosts) {
    if (gh.bustTurn === null) continue;
    const tr = game.trace(gh.id);
    const same = tr.viterbi.filter((k, t) => k === tr.true[t]).length;
    log(`ghost ${gh.id + 1}: Viterbi path matched ${Math.round((100 * same) / tr.true.length)}% of its turns`);
  }
}

// ── autopilot ────────────────────────────────────────────────────────────

function startAuto() {
  if (!game || game.done) return;
  ui.auto = true;
  $('btn-auto').setAttribute('aria-pressed', 'true');
  $('btn-auto').querySelector('.btn-txt').textContent = '■ PAUSE AUTOPILOT';
  tickAuto();
}

function stopAuto() {
  ui.auto = false;
  clearTimeout(ui.autoTimer);
  const b = $('btn-auto');
  if (b) {
    b.setAttribute('aria-pressed', 'false');
    b.querySelector('.btn-txt').textContent = '▶ AUTOPILOT';
  }
  if (game) updateAll(); // the panels say whether the autopilot is running
}

function tickAuto() {
  if (!ui.auto) return;
  if (!game || game.done) {
    stopAuto();
    return;
  }
  act(choose(game));
  ui.autoTimer = setTimeout(tickAuto, ui.delay);
}

// ── Viterbi replay ───────────────────────────────────────────────────────

function startReplay() {
  if (!busted.length) return;
  const gid = busted[replayIdx % busted.length];
  replayIdx += 1;
  stopAuto();
  replay = { gid, t: 0, trace: game.trace(gid), last: performance.now(), finished: false, matched: 0 };
  log(`replaying ghost ${gid + 1}: Viterbi path (cyan) against its true path (pink).`);
  note(`Replay: ghost ${gid + 1}, turn 0 of ${replay.trace.true.length - 1}`);
  updateButtons();
}

function stopReplay() {
  replay = null;
  note('replay stopped');
}

// Advance the replay one turn each REPLAY_MS. At the end, report how often the Viterbi cell was the true one.
function stepReplay(now) {
  if (!replay || replay.finished) return;
  if (now - replay.last < REPLAY_MS) return;
  replay.last = now;
  const last = replay.trace.true.length - 1;
  if (replay.t < last) {
    replay.t += 1;
    note(`Replay: ghost ${replay.gid + 1}, turn ${replay.t} of ${last}: true ${label(replay.trace.true[replay.t])}, `
      + `Viterbi ${label(replay.trace.viterbi[replay.t])}`);
  } else {
    replay.finished = true;
    const same = replay.trace.viterbi.filter((k, t) => k === replay.trace.true[t]).length;
    note(`Replay done: the Viterbi path matched the ghost on ${same} of ${last + 1} turns.`);
  }
}

// ── drawing ──────────────────────────────────────────────────────────────

function resize() {
  dpr = Math.min(window.devicePixelRatio || 1, 2);
  const w = canvas.clientWidth;
  if (canvas.width !== Math.round(w * dpr)) {
    canvas.width = Math.round(w * dpr);
    canvas.height = Math.round(w * dpr);
    fog.width = canvas.width;
    fog.height = canvas.height;
    base.width = canvas.width;
    base.height = canvas.height;
    fogDirty = true;
    baseDirty = true;
  }
  cell = w / game.maze.n;
}

// The belief fog: for each live ghost, a soft glow at every open cell, brighter where the belief is higher.
// Painted once per belief change and then reused by draw().
function paintFog() {
  fctx.setTransform(1, 0, 0, 1, 0, 0);
  fctx.clearRect(0, 0, fog.width, fog.height);
  fctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  fctx.globalCompositeOperation = 'lighter';
  game.ghosts.forEach((gh) => {
    if (!gh.live) return;
    const marg = game.marginal(gh.id);
    const [r, g, b] = hexRgb(GHOST_COL[gh.id]);
    marg.forEach((v, k) => {
      if (v < 0.002) return;
      const [x, y] = centre(k);
      const rad = cell * 1.25;
      const grad = fctx.createRadialGradient(x, y, 0, x, y, rad);
      const a = Math.min(0.95, Math.sqrt(v) * 1.1); // square root: small but real probabilities stay visible
      grad.addColorStop(0, `rgba(${r},${g},${b},${a})`);
      grad.addColorStop(1, `rgba(${r},${g},${b},0)`);
      fctx.fillStyle = grad;
      fctx.fillRect(x - rad, y - rad, 2 * rad, 2 * rad);
    });
  });
  fctx.globalCompositeOperation = 'source-over';
  fogDirty = false;
}

function hexRgb(hex) {
  return [1, 3, 5].map((i) => parseInt(hex.slice(i, i + 2), 16));
}

function ghostIcon(x, y, r, color, alpha = 1) {
  // A ghost: a rounded head over a wavy skirt.
  ctx.save();
  ctx.globalAlpha = alpha;
  ctx.fillStyle = color;
  ctx.shadowColor = color;
  ctx.shadowBlur = 14;
  ctx.beginPath();
  ctx.arc(x, y - r * 0.15, r * 0.6, Math.PI, 0);
  ctx.lineTo(x + r * 0.6, y + r * 0.55);
  for (let i = 0; i < 3; i++) {
    const x0 = x + r * (0.6 - i * 0.4);
    ctx.quadraticCurveTo(x0 - r * 0.2, y + r * 0.85, x0 - r * 0.4, y + r * 0.55);
  }
  ctx.closePath();
  ctx.fill();
  ctx.restore();
}

// The static layer: background, plotter grid, walls with pink edges and the corridor floor. It only changes
// with the maze and the canvas size, so it is painted once and copied to the board each frame.
function paintBase() {
  const n = game.maze.n;
  const W = canvas.width / dpr;
  bctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  bctx.clearRect(0, 0, W, W);
  bctx.fillStyle = '#02040c';
  bctx.fillRect(0, 0, W, W);
  // The plotter grid, faint under everything.
  bctx.strokeStyle = 'rgba(0,245,255,0.05)';
  bctx.lineWidth = 1;
  for (let i = 0; i <= n; i++) {
    bctx.beginPath(); bctx.moveTo(i * cell, 0); bctx.lineTo(i * cell, W); bctx.stroke();
    bctx.beginPath(); bctx.moveTo(0, i * cell); bctx.lineTo(W, i * cell); bctx.stroke();
  }
  // Walls: dark blocks with a pink edge where they meet a corridor.
  for (let c = 0; c < n * n; c++) {
    if (!game.maze.walls[c]) continue;
    const r = Math.floor(c / n), col = c % n;
    bctx.fillStyle = '#0b1a2e';
    bctx.fillRect(col * cell, r * cell, cell, cell);
  }
  bctx.shadowBlur = 8;
  bctx.shadowColor = '#ff00a0';
  bctx.strokeStyle = 'rgba(255,0,160,0.75)';
  bctx.lineWidth = 1.5;
  const open = (rr, cc) => rr >= 0 && rr < n && cc >= 0 && cc < n && !game.maze.walls[rr * n + cc];
  for (let c = 0; c < n * n; c++) {
    if (!game.maze.walls[c]) continue;
    const r = Math.floor(c / n), col = c % n;
    const x = col * cell, y = r * cell;
    if (open(r - 1, col)) bline(x, y, x + cell, y);
    if (open(r + 1, col)) bline(x, y + cell, x + cell, y + cell);
    if (open(r, col - 1)) bline(x, y, x, y + cell);
    if (open(r, col + 1)) bline(x + cell, y, x + cell, y + cell);
  }
  bctx.shadowBlur = 0;
  // Open cells get a faint cyan floor so the corridors read clearly.
  bctx.fillStyle = 'rgba(0,245,255,0.035)';
  for (let k = 0; k < game.maze.K; k++) {
    const [r, col] = rc(k);
    bctx.fillRect(col * cell + 1, r * cell + 1, cell - 2, cell - 2);
  }
  baseDirty = false;
}

function bline(x1, y1, x2, y2) {
  bctx.beginPath();
  bctx.moveTo(x1, y1);
  bctx.lineTo(x2, y2);
  bctx.stroke();
}

function draw(now) {
  resize();
  const W = canvas.width / dpr;
  if (baseDirty) paintBase();
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.drawImage(base, 0, 0);
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  if (fogDirty) paintFog();
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.drawImage(fog, 0, 0);
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  if (ui.showParticles && game.filt === 'particles') drawParticles();
  // Busted ghosts stay where they were caught.
  for (const gh of game.ghosts) {
    if (gh.live) continue;
    const [x, y] = centre(gh.true[gh.bustTurn]);
    ghostIcon(x, y, cell * 0.9, GHOST_COL[gh.id], 0.35);
  }
  // Cheat: the live ghosts' true cells as dashed rings.
  if (ui.cheat) {
    for (const gh of game.ghosts) {
      if (!gh.live) continue;
      const [x, y] = centre(gh.k);
      ctx.save();
      ctx.strokeStyle = GHOST_COL[gh.id];
      ctx.setLineDash([4, 3]);
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.arc(x, y, cell * 0.42, 0, Math.PI * 2);
      ctx.stroke();
      ctx.restore();
    }
  }
  drawRings(now);
  if (replay) drawReplay();
  // The player: a bright core with a slow pulse.
  const [px, py] = centre(game.p);
  const pulse = 0.5 + 0.5 * Math.sin(now / 260);
  ctx.save();
  ctx.fillStyle = ui.bust ? '#ff00a0' : '#e8fdff';
  ctx.shadowColor = ui.bust ? '#ff00a0' : '#00f5ff';
  ctx.shadowBlur = 16 + 8 * pulse;
  ctx.beginPath();
  ctx.arc(px, py, cell * 0.26, 0, Math.PI * 2);
  ctx.fill();
  ctx.restore();
  // Live ghosts are not drawn: only the fog says where they might be (or the cheat rings above).
}

function drawParticles() {
  game.ghosts.forEach((gh) => {
    if (!gh.live) return;
    const parts = gh.filter.parts;
    const w = gh.filter.w;
    ctx.fillStyle = GHOST_COL[gh.id];
    ctx.shadowColor = GHOST_COL[gh.id];
    ctx.shadowBlur = 6;
    parts.forEach((s, i) => {
      const [x, y] = centre(s >> 2);
      // A deterministic spread within the cell, so particles in the same cell do not stack exactly.
      const jx = (((i * 37) % 7) - 3) / 10, jy = (((i * 53) % 7) - 3) / 10;
      const size = 1.2 + Math.min(2.5, Math.sqrt(w[i] * parts.length) * 0.8);
      ctx.beginPath();
      ctx.arc(x + jx * cell, y + jy * cell, size, 0, Math.PI * 2);
      ctx.fill();
    });
    ctx.shadowBlur = 0;
  });
}

function drawRings(now) {
  rings = rings.filter((rg) => now - rg.t0 < RING_MS);
  for (const rg of rings) {
    const age = (now - rg.t0) / RING_MS;
    const ease = 1 - Math.pow(1 - age, 3);
    ctx.save();
    ctx.strokeStyle = rg.color;
    ctx.globalAlpha = 1 - age;
    ctx.lineWidth = 2;
    ctx.shadowColor = rg.color;
    ctx.shadowBlur = 10;
    ctx.beginPath();
    ctx.arc(rg.x, rg.y, Math.max(2, rg.rad * ease), 0, Math.PI * 2);
    ctx.stroke();
    ctx.restore();
  }
}

// Replay overlay: the true path in pink (dashed) and the Viterbi path in cyan, drawn up to the current turn.
function drawReplay() {
  const { trace, t } = replay;
  const pts = (ks) => ks.slice(0, t + 1).map((k) => centre(k));
  const pathOf = (ks, color, dash) => {
    const p = pts(ks);
    if (!p.length) return;
    ctx.save();
    ctx.strokeStyle = color;
    ctx.lineWidth = 3;
    ctx.shadowColor = color;
    ctx.shadowBlur = 10;
    ctx.setLineDash(dash);
    ctx.beginPath();
    p.forEach(([x, y], i) => (i ? ctx.lineTo(x, y) : ctx.moveTo(x, y)));
    ctx.stroke();
    ctx.restore();
    const [hx, hy] = p[p.length - 1];
    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.arc(hx, hy, cell * 0.2, 0, Math.PI * 2);
    ctx.fill();
  };
  pathOf(trace.true, '#ff00a0', [6, 4]);
  pathOf(trace.viterbi, '#00f5ff', []);
}

// ── panels ───────────────────────────────────────────────────────────────

function updateThink() {
  const lines = [];
  for (const gh of game.ghosts) {
    if (!gh.live) {
      lines.push(`ghost ${gh.id + 1}: busted on turn ${gh.bustTurn}.`);
      continue;
    }
    const marg = game.marginal(gh.id);
    const k = argmax(marg);
    const ping = gh.readings[gh.readings.length - 1];
    let text = `ghost ${gh.id + 1}: the ping says ${ping} cells away. Most likely cell ${label(k)} at `
      + `${Math.round(100 * marg[k])}%.`;
    if (ui.cheat) {
      text += ` True cell ${label(gh.k)}, which holds ${Math.round(100 * marg[gh.k])}% of its belief.`;
    }
    lines.push(text);
  }
  lines.push(`autopilot ${ui.auto ? 'running' : 'off'}; the autopilot busts when a peak is over 50% and next to you.`);
  $('think-body').textContent = lines.join('\n');
  $('think-title').textContent = game.done ? (game.busted === game.ghosts.length ? 'Hunt complete' : 'Out of turns')
    : `Turn ${game.t}`;
}

function updateChips() {
  $('chip-turn').textContent = String(game.t);
  $('chip-busted').textContent = `${game.busted}/${game.ghosts.length}`;
  $('chip-filter').textContent = game.filt === 'exact' ? 'EXACT' : `PARTICLES ${game.particles}`;
  $('chip-motion').textContent = MOTION_NAME[game.model];
  $('chip-sigma').textContent = String(game.sigma.toFixed(1));
  $('chip-status').textContent = game.done ? (game.busted === game.ghosts.length ? 'BUSTED' : 'TIMEOUT') : 'HUNTING';
}

function updateButtons() {
  const bustBtn = $('btn-bust');
  bustBtn.setAttribute('aria-pressed', String(ui.bust));
  bustBtn.querySelector('.btn-txt').textContent = `◉ BUST MODE: ${ui.bust ? 'ON' : 'OFF'}`;
  $('btn-replay').disabled = busted.length === 0;
  $('particles').disabled = cfg.filter !== 'particles';
  $('btn-step').disabled = !game || game.done || ui.auto;
  for (const id of ['mv-N', 'mv-E', 'mv-S', 'mv-W', 'mv-stay']) $(id).disabled = !game || game.done;
  $('speed-val').textContent = String(ui.delay);
}

function updateAll() {
  updateChips();
  updateThink();
  updateButtons();
}

// ── controls ─────────────────────────────────────────────────────────────

// A direction key or button: move that way, or bust that neighbour when bust mode is on.
function go(d) {
  act(ui.bust ? `b${d}` : d);
}

function waitTurn() {
  act(ui.bust ? 'b.' : '.');
}

function readSeed() {
  const v = Math.floor(Number($('seed').value));
  return Number.isFinite(v) && v >= 0 && v <= 4294967295 ? v : 7;
}

$('mv-N').addEventListener('click', () => go('N'));
$('mv-E').addEventListener('click', () => go('E'));
$('mv-S').addEventListener('click', () => go('S'));
$('mv-W').addEventListener('click', () => go('W'));
$('mv-stay').addEventListener('click', waitTurn);
$('btn-bust').addEventListener('click', () => {
  ui.bust = !ui.bust;
  note(ui.bust ? 'bust mode: the next direction fires at that cell instead of moving' : 'move mode');
  updateButtons();
});
$('btn-auto').addEventListener('click', () => (ui.auto ? stopAuto() : startAuto()));
$('btn-step').addEventListener('click', () => {
  stopAuto();
  if (game && !game.done) act(choose(game));
});
$('btn-replay').addEventListener('click', startReplay);
$('btn-new').addEventListener('click', () => newHunt(readSeed()));
$('btn-shuffle').addEventListener('click', () => {
  $('seed').value = String(Math.floor(Math.random() * 999999) + 1);
  newHunt(readSeed());
});
$('seed').addEventListener('change', () => newHunt(readSeed()));
$('size').addEventListener('change', (e) => {
  cfg.size = Number(e.target.value);
  newHunt(readSeed());
});
$('ghosts').addEventListener('change', (e) => {
  cfg.ghosts = Number(e.target.value);
  newHunt(readSeed());
});
$('motion').addEventListener('change', (e) => {
  cfg.motion = e.target.value;
  newHunt(readSeed());
});
$('noise').addEventListener('change', (e) => {
  cfg.noise = e.target.value;
  newHunt(readSeed());
});
$('filter').addEventListener('change', (e) => {
  cfg.filter = e.target.value;
  newHunt(readSeed());
});
$('particles').addEventListener('change', (e) => {
  cfg.particles = Number(e.target.value);
  newHunt(readSeed());
});
$('show-particles').addEventListener('change', (e) => {
  ui.showParticles = e.target.checked;
  fogDirty = true;
});
$('cheat').addEventListener('change', (e) => {
  ui.cheat = e.target.checked;
  updateThink();
});
$('speed').addEventListener('input', (e) => {
  ui.delay = Number(e.target.value);
  $('speed-val').textContent = String(ui.delay);
});

// A click on a cell next to you moves there (or busts it in bust mode). Clicking yourself waits.
canvas.addEventListener('click', (e) => {
  if (!game) return;
  const rect = canvas.getBoundingClientRect();
  const x = e.clientX - rect.left, y = e.clientY - rect.top;
  const n = game.maze.n;
  const col = Math.floor(x / cell), row = Math.floor(y / cell);
  if (col < 0 || row < 0 || col >= n || row >= n) return;
  const k = game.maze.kOf[row * n + col];
  if (k < 0) return;
  if (k === game.p) return waitTurn();
  for (let d = 0; d < 4; d++) {
    if (game.maze.step[game.p][d] === k) return go('NESW'[d]);
  }
  note(`${label(k)} is not next to you; click a neighbouring corridor cell.`);
});

// Keys: arrows and WASD move, . waits, B toggles bust mode, G the autopilot, N a new hunt.
window.addEventListener('keydown', (e) => {
  if (e.target && ['INPUT', 'SELECT', 'TEXTAREA'].includes(e.target.tagName)) return;
  const map = { ArrowUp: 'N', w: 'N', ArrowRight: 'E', d: 'E', ArrowDown: 'S', s: 'S', ArrowLeft: 'W', a: 'W' };
  const key = e.key.length === 1 ? e.key.toLowerCase() : e.key;
  if (map[key]) {
    e.preventDefault();
    go(map[key]);
  } else if (key === '.' || key === ' ') {
    e.preventDefault();
    waitTurn();
  } else if (key === 'b') {
    $('btn-bust').click();
  } else if (key === 'g') {
    $('btn-auto').click();
  } else if (key === 'n') {
    newHunt(readSeed());
  }
});

window.addEventListener('resize', () => { fogDirty = true; });

// ── the animation loop: the board is redrawn while anything animates ──────

function frame(now) {
  stepReplay(now);
  if (game) draw(now);
  requestAnimationFrame(frame);
}

// Start: the static fallback in the HTML is replaced by the game state.
$('chip-status').textContent = 'READY';
newHunt(cfg.seed);
requestAnimationFrame(frame);
