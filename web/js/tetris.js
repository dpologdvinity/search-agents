// Tetris page: play by hand, or watch the agent play.
//
// One animation loop drives everything. Each piece gets a decision from the placement search
// (tetris_engine.js, the same algorithm as tetris/search.py), using the weights from the sliders. The
// decision has two phases: THINK reveals every candidate placement as a heat-coloured ghost, then the
// agent moves: it rotates and slides the piece to its chosen spot and drops it. Human mode uses the same
// decisions as a hint ("show agent"), so the sliders always show what the agent would do right now.
//
// The weights come from /api/tetris/meta (the GA result). Nothing else touches the server.

import { getJSON } from './api.js';
import { banner, burst, pop, shake } from './fx.js';
import {
  FEATURES, H, HAND_WEIGHTS, PIECES, ROTATIONS, W, Game, bestMove, boardFeatures, fits, placementFeatures,
} from './tetris_engine.js';

const $ = (id) => document.getElementById(id);
const COLORS = { I: '#00f5ff', O: '#ffe600', T: '#b36bff', S: '#00ff88', Z: '#ff3b3b', J: '#3d7bff', L: '#ff8a00' };
const HEAT = ['#0b3cff', '#00f5ff', '#00ff88', '#ffe600', '#ff00a0']; // cold to hot
const SPEEDS = { slow: 420, normal: 200, fast: 90, max: 16 }; // ms per AI action
const FEATURE_NAMES = {
  landing_height: 'landing height', eroded_cells: 'eroded cells', row_transitions: 'row transitions',
  column_transitions: 'column transitions', holes: 'holes', wells: 'wells', aggregate_height: 'aggregate height',
  bumpiness: 'bumpiness', lines: 'lines cleared',
};
const FEATURE_HINTS = {
  landing_height: 'where the piece lands, high is bad',
  eroded_cells: 'piece cells removed by clearing lines',
  row_transitions: 'empty/filled changes along rows',
  column_transitions: 'empty/filled changes up columns',
  holes: 'empty cells with a block above them',
  wells: 'deep one-wide gaps, counted 1+2+...+depth',
  aggregate_height: 'sum of column heights',
  bumpiness: 'height steps between neighbours',
  lines: 'rows completed by this placement',
};

const state = {
  tuned: HAND_WEIGHTS.slice(),
  hand: HAND_WEIGHTS.slice(),
  weights: HAND_WEIGHTS.slice(),
  lookahead: false,
  mode: 'human', // 'human' or 'ai'
  speed: 'normal',
  paused: false,
  game: null,
  decision: null, // {move, moves, phase: 'think' | 'move', shown, acc, target}
  needDecide: false,
  decideAt: 0,
  showAgent: false,
  softHold: false,
  fallAcc: 0,
  actAcc: 0,
  bestLines: 0,
  cell: 24,
  dirty: true,
  flash: 0,
  online: false,
  meta: null,
};

// ── Setup ─────────────────────────────────────────────────────────────────

function buildSliders() {
  const box = $('tt-sliders');
  box.innerHTML = '';
  FEATURES.forEach((name, i) => {
    const row = document.createElement('div');
    row.className = 'tt-slider';
    row.innerHTML = `
      <div class="tt-slider-head">
        <label for="tt-w${i}">${FEATURE_NAMES[name]}</label>
        <span class="tt-tuned" id="tt-t${i}" title="the GA-tuned value"></span>
        <output id="tt-v${i}"></output>
      </div>
      <input id="tt-w${i}" type="range" min="-10" max="10" step="0.05">
      <div class="tt-slider-hint">${FEATURE_HINTS[name]}</div>`;
    box.appendChild(row);
    const input = row.querySelector('input');
    input.addEventListener('input', () => {
      state.weights[i] = parseFloat(input.value);
      $(`tt-v${i}`).textContent = state.weights[i].toFixed(2);
      state.needDecide = true;
      state.dirty = true;
      setPreset(null);
    });
  });
  syncSliders();
}

// Each slider shows its live value and, beside it, the GA-tuned value, so the drift is visible while editing.
function syncSliders() {
  state.weights.forEach((w, i) => {
    $(`tt-w${i}`).value = w;
    $(`tt-v${i}`).textContent = w.toFixed(2);
    $(`tt-t${i}`).textContent = `GA ${state.tuned[i].toFixed(2)}`;
  });
}

function setPreset(name) {
  for (const id of ['tt-reset-tuned', 'tt-reset-hand']) {
    $(id).setAttribute('aria-pressed', String(id === `tt-reset-${name}`));
  }
}

function setWeights(w, preset) {
  state.weights = w.slice();
  syncSliders();
  setPreset(preset);
  state.needDecide = true;
  state.dirty = true;
}

// Fitness chart: best and mean lines per generation from the GA log.
function drawFitness() {
  const canvas = $('tt-fitness');
  const hist = state.meta?.history || [];
  const ctx = canvas.getContext('2d');
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const w = canvas.clientWidth || 320, h = 180;
  canvas.width = w * dpr; canvas.height = h * dpr;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = 'rgba(2,6,16,0.9)'; ctx.fillRect(0, 0, w, h);
  const pad = { l: 38, r: 10, t: 12, b: 24 };
  if (hist.length < 2) {
    ctx.fillStyle = '#4a7a9b'; ctx.font = '12px JetBrains Mono, monospace';
    ctx.fillText('no GA history yet', pad.l, h / 2);
    return;
  }
  const maxY = Math.max(...hist.map((r) => r.best)) * 1.1 || 1;
  const x = (g) => pad.l + (g / (hist.length - 1)) * (w - pad.l - pad.r);
  const y = (v) => h - pad.b - (v / maxY) * (h - pad.t - pad.b);
  ctx.strokeStyle = 'rgba(0,245,255,0.12)'; ctx.lineWidth = 1;
  ctx.fillStyle = '#4a7a9b'; ctx.font = '10px JetBrains Mono, monospace';
  for (let k = 0; k <= 4; k++) {
    const v = (maxY * k) / 4;
    ctx.beginPath(); ctx.moveTo(pad.l, y(v)); ctx.lineTo(w - pad.r, y(v)); ctx.stroke();
    ctx.fillText(Math.round(v), 4, y(v) + 3);
  }
  ctx.fillText('gen 0', pad.l, h - 6);
  ctx.textAlign = 'right'; ctx.fillText(`gen ${hist.length - 1}`, w - pad.r, h - 6); ctx.textAlign = 'left';
  const line = (key, color, glow) => {
    ctx.beginPath();
    hist.forEach((r, g) => (g ? ctx.lineTo(x(g), y(r[key])) : ctx.moveTo(x(g), y(r[key]))));
    ctx.strokeStyle = color; ctx.lineWidth = 2; ctx.shadowColor = color; ctx.shadowBlur = glow; ctx.stroke();
    ctx.shadowBlur = 0;
  };
  line('mean', '#ff00a0', 6);
  line('best', '#00f5ff', 10);
}

// ── Decisions ─────────────────────────────────────────────────────────────

// Work out the agent's choice for the falling piece with the current weights. Called when a piece spawns,
// and again when the weights or lookahead change (throttled in the loop).
function decide() {
  const g = state.game;
  if (!g || g.over) { state.decision = null; return; }
  const preview = state.lookahead ? g.bag.peek() : null;
  const { move, moves } = bestMove(g.board, g.piece, state.weights, preview, 6);
  state.decision = { move, moves, phase: 'think', shown: 0, acc: 0, target: move ? { rot: move.rot, x: move.x } : null };
  state.dirty = true;
  // The AI has no legal placement for this piece: the same game over as tetris/game.py.
  if (!move && state.mode === 'ai') { g.over = true; onGameOver(); }
}

function heatColour(t) {
  // Map t in [0, 1] onto the HEAT stops.
  const k = Math.min(0.999, Math.max(0, t)) * (HEAT.length - 1);
  const i = Math.floor(k), f = k - i;
  const a = HEAT[i].match(/\w\w/g).map((h) => parseInt(h, 16));
  const b = HEAT[i + 1].match(/\w\w/g).map((h) => parseInt(h, 16));
  const c = a.map((v, j) => Math.round(v + (b[j] - v) * f));
  return `rgb(${c[0]},${c[1]},${c[2]})`;
}

// ── Game flow ─────────────────────────────────────────────────────────────

function newGame() {
  state.game = new Game(Math.floor(Math.random() * 1e9));
  state.fallAcc = 0; state.actAcc = 0; state.paused = false;
  decide();
  setStatus('PLAYING');
  $('tt-over').hidden = true;
  state.dirty = true;
}

function setStatus(text) { $('chip-status').textContent = text; }

function lockPiece() {
  const g = state.game;
  const before = g.lines;
  const res = g.place(g.rot, g.x, g.y);
  state.decision = null;
  if (res.lines) {
    const cells = res.lines * W;
    shake($('tt-board'), res.lines >= 4 ? 'big' : 'small');
    const r = $('tt-board').getBoundingClientRect();
    burst([r.left + r.width / 2, r.top + r.height / 2], { count: 20 + cells * 2 });
    pop([r.left + r.width / 2, r.top + 40], `+${res.lines} LINE${res.lines > 1 ? 'S' : ''}`, '#ffe600');
    if (res.lines === 4) banner('TETRIS', 'four lines at once', '#ff00a0');
  }
  state.bestLines = Math.max(state.bestLines, g.lines);
  if (g.over) return onGameOver();
  decide();
  state.dirty = true;
  if (g.lines !== before) setStatus(state.mode === 'ai' ? 'AI PLAYING' : 'PLAYING');
}

function onGameOver() {
  setStatus('GAME OVER');
  state.decision = null;
  $('tt-over-text').textContent = `${state.game.lines} lines in ${state.game.pieces} pieces`;
  $('tt-over').hidden = false;
  banner('GAME OVER', `${state.game.lines} lines in ${state.game.pieces} pieces`, '#ff3b3b');
  state.dirty = true;
  if (state.mode === 'ai') setTimeout(() => { if (state.mode === 'ai') newGame(); }, 2600);
}

// One AI action per interval: rotate toward the target, slide to its column, then hard drop.
// Each rotate or slide is one action, so the piece visibly turns and walks to its spot.
function aiAct() {
  const g = state.game, d = state.decision;
  if (!d || !d.target) { hardDrop(); return; }
  if (g.rot !== d.target.rot && g.rotate()) return;
  if (g.x !== d.target.x && g.move(Math.sign(d.target.x - g.x))) return;
  hardDrop();
}

function hardDrop() {
  const g = state.game;
  while (g.softDrop());
  lockPiece();
}

// Human gravity: the piece falls one row every `ms`, faster as the level climbs.
function fallMs() {
  const level = 1 + Math.floor(state.game.lines / 10);
  return Math.max(110, 700 - (level - 1) * 60);
}

function tick(dt) {
  const g = state.game;
  if (!g || g.over || state.paused) return;
  if (state.mode === 'human') {
    state.fallAcc += dt * (state.softHold ? 14 : 1);
    if (state.fallAcc >= fallMs()) {
      state.fallAcc = 0;
      if (!g.softDrop()) lockPiece();
      state.dirty = true;
    }
    return;
  }
  // AI mode: thinking reveals the candidates, then the agent acts.
  const d = state.decision;
  if (!d) return;
  if (d.phase === 'think') {
    // The candidates appear in order over this long; the faster the speed, the quicker the scan.
    const total = d.moves.length;
    const revealMs = Math.max(200, SPEEDS[state.speed] * 4);
    d.acc += dt;
    const n = Math.min(total, Math.floor((d.acc / revealMs) * total) + 1);
    if (n !== d.shown) { d.shown = n; state.dirty = true; }
    if (d.shown >= total) { d.phase = 'move'; d.acc = 0; }
    return;
  }
  state.actAcc += dt;
  const gap = SPEEDS[state.speed];
  if (state.actAcc >= gap) {
    state.actAcc = 0;
    aiAct();
    state.dirty = true;
  }
}

// ── Drawing ───────────────────────────────────────────────────────────────

function sizeBoard() {
  const wrap = $('tt-board').parentElement;
  const avail = Math.min(wrap.clientWidth || 300, 340);
  state.cell = Math.max(14, Math.floor(avail / W));
}

function fillCell(ctx, x, y, cell, colour, alpha = 1, glow = 10) {
  ctx.save();
  ctx.globalAlpha = alpha;
  ctx.shadowColor = colour; ctx.shadowBlur = glow;
  ctx.fillStyle = colour;
  ctx.fillRect(x * cell + 1, y * cell + 1, cell - 2, cell - 2);
  ctx.shadowBlur = 0;
  ctx.fillStyle = 'rgba(255,255,255,0.35)';
  ctx.fillRect(x * cell + 3, y * cell + 3, cell - 8, 3);
  ctx.restore();
}

function outlinePiece(ctx, rot, px, py, cell, colour, width, dash = []) {
  ctx.save();
  ctx.strokeStyle = colour; ctx.lineWidth = width; ctx.setLineDash(dash);
  rot.masks.forEach((m, i) => {
    for (let c = 0; c < rot.width; c++) {
      if ((m >> c) & 1) {
        const row = py + i;
        if (row >= H) continue;
        ctx.strokeRect(((px + c) * cell) + 1.5, ((H - 1 - row) * cell) + 1.5, cell - 3, cell - 3);
      }
    }
  });
  ctx.restore();
}

function drawBoard() {
  const canvas = $('tt-board');
  const cell = state.cell;
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const cw = W * cell, ch = H * cell;
  if (canvas.width !== cw * dpr || canvas.height !== ch * dpr) {
    canvas.width = cw * dpr; canvas.height = ch * dpr;
    canvas.style.width = `${cw}px`; canvas.style.height = `${ch}px`;
  }
  const ctx = canvas.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.fillStyle = 'rgba(2,6,16,0.96)'; ctx.fillRect(0, 0, cw, ch);
  ctx.strokeStyle = 'rgba(0,245,255,0.07)'; ctx.lineWidth = 1;
  for (let c = 1; c < W; c++) { ctx.beginPath(); ctx.moveTo(c * cell, 0); ctx.lineTo(c * cell, ch); ctx.stroke(); }
  for (let r = 1; r < H; r++) { ctx.beginPath(); ctx.moveTo(0, r * cell); ctx.lineTo(cw, r * cell); ctx.stroke(); }

  const g = state.game;
  if (!g) return;
  // Locked blocks: colour shifts with height so the stack reads as a gradient.
  for (let r = 0; r < H; r++) {
    const row = g.board[r];
    if (!row) continue;
    const hue = 185 + (r / H) * 150;
    for (let c = 0; c < W; c++) {
      if ((row >> c) & 1) fillCell(ctx, c, H - 1 - r, cell, `hsl(${hue},100%,60%)`, 0.9, 8);
    }
  }

  const d = state.decision;
  const showCands = d && (state.mode === 'ai' || state.showAgent);
  if (showCands && d.moves.length) {
    const scores = d.moves.map((m) => m.score);
    const lo = Math.min(...scores), hi = Math.max(...scores);
    const span = hi - lo || 1;
    const shown = state.mode === 'ai' && d.phase === 'think' ? d.shown : d.moves.length;
    d.moves.slice(0, shown).forEach((m) => {
      const t = (m.score - lo) / span;
      const rot = ROTATIONS[m.piece][m.rot];
      outlinePiece(ctx, rot, m.x, m.y, cell, heatColour(t), 1.5);
      rot.masks.forEach((mask, i) => {
        for (let c = 0; c < rot.width; c++) {
          if ((mask >> c) & 1) fillCell(ctx, m.x + c, H - 1 - (m.y + i), cell, heatColour(t), 0.1 + 0.25 * t, 0);
        }
      });
    });
    if (d.move && (state.mode === 'human' || d.phase === 'move' || d.shown >= d.moves.length)) {
      const best = d.move;
      outlinePiece(ctx, ROTATIONS[best.piece][best.rot], best.x, best.y, cell, '#ffe600', 3, [6, 4]);
    }
  }

  // Falling piece and its landing shadow.
  const rot = g.shape(g.rot);
  if (!g.over) {
    let land = g.y;
    while (land > 0 && fits(g.board, rot, g.x, land - 1)) land--;
    outlinePiece(ctx, rot, g.x, land, cell, 'rgba(255,255,255,0.45)', 1.5, [3, 3]);
    rot.masks.forEach((mask, i) => {
      for (let c = 0; c < rot.width; c++) {
        if ((mask >> c) & 1 && g.y + i < H) fillCell(ctx, g.x + c, H - 1 - (g.y + i), cell, COLORS[g.piece], 1, 16);
      }
    });
  }
  if (state.paused) {
    ctx.fillStyle = 'rgba(2,6,16,0.6)'; ctx.fillRect(0, 0, cw, ch);
    ctx.fillStyle = '#00f5ff'; ctx.font = `700 ${Math.round(cell * 1.1)}px Orbitron, sans-serif`;
    ctx.textAlign = 'center'; ctx.fillText('PAUSED', cw / 2, ch / 2); ctx.textAlign = 'left';
  }
}

function drawNext() {
  const canvas = $('tt-next');
  const ctx = canvas.getContext('2d');
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  if (!state.game) return;
  const name = state.game.bag.peek();
  const rot = ROTATIONS[name][0];
  const s = 14;
  const ox = (canvas.width - rot.width * s) / 2, oy = (canvas.height - rot.height * s) / 2;
  rot.masks.forEach((m, i) => {
    for (let c = 0; c < rot.width; c++) {
      if ((m >> c) & 1) {
        ctx.save(); ctx.shadowColor = COLORS[name]; ctx.shadowBlur = 10; ctx.fillStyle = COLORS[name];
        ctx.fillRect(ox + c * s + 1, oy + (rot.height - 1 - i) * s + 1, s - 2, s - 2); ctx.restore();
      }
    }
  });
}

function renderPanels() {
  const g = state.game;
  if (!g) return;
  $('chip-lines').textContent = g.lines;
  $('chip-pieces').textContent = g.pieces;
  $('chip-best').textContent = state.bestLines;
  $('chip-mode').textContent = state.mode === 'ai' ? 'AI' : 'HUMAN';
  $('tt-speed-wrap').hidden = state.mode !== 'ai';
  const human = state.mode === 'human';
  document.querySelectorAll('[data-human]').forEach((b) => { b.disabled = !human || g.over; });
  $('tt-mode-human').setAttribute('aria-pressed', String(human));
  $('tt-mode-ai').setAttribute('aria-pressed', String(!human));
  $('tt-look').setAttribute('aria-pressed', String(state.lookahead));
  $('tt-agent').setAttribute('aria-pressed', String(state.showAgent));
  $('tt-pause').setAttribute('aria-pressed', String(state.paused));
  renderWhy();
}

// Explain the current choice: the weighted contribution of each feature for the chosen placement.
function renderWhy() {
  const d = state.decision;
  const box = $('tt-why');
  const line = $('tt-why-line');
  if (!d || !d.move) {
    line.textContent = state.game?.over ? 'No placement is legal. The stack has reached the top.' : 'Thinking...';
    box.innerHTML = '';
    return;
  }
  const m = d.move;
  const turns = (m.rot - state.game.rot + ROTATIONS[m.piece].length) % ROTATIONS[m.piece].length;
  const tail = state.lookahead ? ` With one-piece lookahead it scores ${m.total.toFixed(1)} including the next piece.` : '';
  line.textContent = `Chosen: ${m.piece} piece, ${turns} turn(s), column ${m.x + 1}, score ${m.score.toFixed(1)}, `
    + `clears ${m.lines}.${tail}`;
  const contrib = FEATURES.map((name, i) => ({ name, f: m.features[i], c: state.weights[i] * m.features[i] }));
  const big = Math.max(1e-9, ...contrib.map((r) => Math.abs(r.c)));
  box.innerHTML = contrib.map((r) => {
    const pct = Math.round((Math.abs(r.c) / big) * 100);
    const colour = r.c >= 0 ? '#00ff88' : '#ff00a0';
    return `<div class="tt-bar"><span>${FEATURE_NAMES[r.name]}</span>`
      + `<span class="tt-bar-track"><span style="width:${pct}%;background:${colour}"></span></span>`
      + `<span class="tt-bar-val">${r.f.toFixed(r.f % 1 ? 1 : 0)} x ${state.weights[FEATURES.indexOf(r.name)].toFixed(2)}`
      + ` = ${r.c >= 0 ? '+' : ''}${r.c.toFixed(1)}</span></div>`;
  }).join('');
}

// ── Loop ──────────────────────────────────────────────────────────────────

let last = 0;
function frame(ts) {
  const dt = Math.min(100, ts - (last || ts));
  last = ts;
  if (state.needDecide && ts - state.decideAt > 120) {
    state.needDecide = false;
    state.decideAt = ts;
    if (state.game && !state.game.over) decide();
  }
  tick(dt);
  if (state.dirty) {
    state.dirty = false;
    drawBoard();
    drawNext();
    renderPanels();
  }
  requestAnimationFrame(frame);
}

// ── Controls ──────────────────────────────────────────────────────────────

function humanKey(key) {
  const g = state.game;
  if (!g || g.over || state.mode !== 'human' || state.paused) return false;
  switch (key) {
    case 'ArrowLeft': g.move(-1); break;
    case 'ArrowRight': g.move(1); break;
    case 'ArrowUp': case 'x': case 'X': g.rotate(); break;
    case 'z': case 'Z': g.rotate(); break;
    case 'ArrowDown': if (!g.softDrop()) lockPiece(); break;
    case ' ': hardDrop(); break;
    default: return false;
  }
  state.dirty = true;
  return true;
}

function bindControls() {
  document.addEventListener('keydown', (e) => {
    if (e.target && /INPUT|SELECT|TEXTAREA/.test(e.target.tagName)) return; // let sliders and menus take keys
    const keys = ['ArrowLeft', 'ArrowRight', 'ArrowUp', 'ArrowDown', ' '];
    if (e.key === 'p' || e.key === 'P') { togglePause(); e.preventDefault(); return; }
    if (e.key === 'h' || e.key === 'H') { toggleAgent(); return; }
    if (keys.includes(e.key) || 'xXzZ'.includes(e.key)) {
      if (humanKey(e.key)) e.preventDefault();
      if (e.key === 'ArrowDown') state.softHold = true;
    }
  });
  document.addEventListener('keyup', (e) => { if (e.key === 'ArrowDown') state.softHold = false; });

  $('tt-new').addEventListener('click', newGame);
  $('tt-pause').addEventListener('click', togglePause);
  $('tt-agent').addEventListener('click', toggleAgent);
  $('tt-look').addEventListener('click', () => {
    state.lookahead = !state.lookahead;
    state.needDecide = true; state.dirty = true;
  });
  $('tt-mode-human').addEventListener('click', () => setMode('human'));
  $('tt-mode-ai').addEventListener('click', () => setMode('ai'));
  $('tt-speed').addEventListener('change', (e) => { state.speed = e.target.value; state.dirty = true; });
  $('tt-reset-tuned').addEventListener('click', () => setWeights(state.tuned, 'tuned'));
  $('tt-reset-hand').addEventListener('click', () => setWeights(state.hand, 'hand'));
  $('tt-over-new').addEventListener('click', newGame);

  const press = { left: () => humanKey('ArrowLeft'), right: () => humanKey('ArrowRight'),
    rot: () => humanKey('ArrowUp'), drop: () => humanKey(' ') };
  for (const [id, fn] of Object.entries(press)) {
    $(`tt-m-${id}`).addEventListener('click', fn);
  }
  const down = $('tt-m-down');
  down.addEventListener('pointerdown', () => { state.softHold = true; });
  for (const ev of ['pointerup', 'pointerleave', 'pointercancel']) {
    down.addEventListener(ev, () => { state.softHold = false; });
  }
  down.addEventListener('click', (e) => { if (e.detail === 0) humanKey('ArrowDown'); });

  addEventListener('resize', () => { sizeBoard(); drawFitness(); state.dirty = true; });
}

function togglePause() {
  if (!state.game || state.game.over) return;
  state.paused = !state.paused;
  setStatus(state.paused ? 'PAUSED' : state.mode === 'ai' ? 'AI PLAYING' : 'PLAYING');
  state.dirty = true;
}

function toggleAgent() {
  state.showAgent = !state.showAgent;
  state.dirty = true;
}

function setMode(mode) {
  state.mode = mode;
  state.decision = null;
  newGame();
  setStatus(mode === 'ai' ? 'AI PLAYING' : 'PLAYING');
  $('tt-agent').disabled = mode === 'ai';
}

// ── Start ─────────────────────────────────────────────────────────────────

async function init() {
  buildSliders();
  try {
    const meta = await getJSON('/api/tetris/meta');
    state.meta = meta;
    state.online = true;
    state.tuned = meta.tuned_weights.slice();
    state.hand = meta.hand_weights.slice();
    $('tt-train-fit').textContent = meta.train_fitness == null ? '—' : meta.train_fitness.toFixed(1);
    const cfg = meta.config || {};
    $('tt-cfg').textContent = cfg.population
      ? `${cfg.population} genomes, ${cfg.generations} generations, ${cfg.train_games} games per genome on a `
        + `${cfg.board_height}-row board, capped at ${cfg.train_pieces} pieces, fresh seeds each generation`
      : 'no tuned weights yet';
  } catch (err) {
    state.online = false;
    $('tt-cfg').textContent = 'offline: using the hand-picked weights';
    setStatus('OFFLINE');
  }
  setWeights(state.hand, 'hand'); // the page's AI starts on the hand-picked weights, the stronger set on held-out games
  bindControls();
  sizeBoard();
  drawFitness();
  newGame();
  $('tt-mode-human').setAttribute('aria-pressed', 'true');
  requestAnimationFrame(frame);
}

init();
