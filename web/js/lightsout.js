// Lights Out page.
//
// Pressing a light is simple enough to run here: it and its orthogonal neighbours flip.
// Solving is the server's job (/api/lightsout/solve), which returns the minimum press set,
// the rank, and the elimination steps that the canvas replays.
//
// The solution on screen stays valid as the player presses. Presses combine by XOR and
// the board changes by A e_p when cell p is pressed, so the remaining solution changes
// by XOR with e_p. Toggling p in the plan keeps it a solution, with no new request.

import { getJSON, postJSON } from './api.js';
import { banner, burst, pop, shake } from './fx.js';

const $ = (id) => document.getElementById(id);
const CELL_PX = { 3: 74, 4: 66, 5: 60, 6: 52, 7: 46, 8: 42, 9: 38 };
const PLAY_MS = 420; // pause between presses in AUTO
const FRAME_MS = 650; // pause between elimination steps in PLAY
const NEIGHBOURS = [[-1, 0], [1, 0], [0, -1], [0, 1]];

const state = {
  n: 5,
  start: [],      // the board as dealt; RESET returns here
  board: [],      // the board now (0 dark, 1 lit)
  cells: [],      // DOM elements, one per cell
  moves: 0,       // presses made by the player
  assists: 0,     // presses made by STEP and AUTO
  plan: null,     // cells of the solution shown, or null; always a solution of the current board
  hintCell: null, // the next press HINT suggested
  info: null,     // latest solve response for the current board
  startInfo: null, // solve response for the dealt board
  minStart: null, // minimum presses from the dealt board
  over: false,
  busy: false,
  autoTimer: 0,
  elimTimer: 0,
  elim: null,     // {trace, n, frame} for the canvas
  meta: null,
};

// Cell index to a spreadsheet-style name: column letter, then row number (1-based).
const labelOf = (cell, n) => `${String.fromCharCode(97 + (cell % n))}${Math.floor(cell / n) + 1}`;

const litCount = () => state.board.reduce((a, v) => a + v, 0);

// The board after pressing `cell`: the cell and its orthogonal neighbours flip.
function toggled(n, board, cell) {
  const next = board.slice();
  const r = Math.floor(cell / n), c = cell % n;
  next[cell] ^= 1;
  for (const [dr, dc] of NEIGHBOURS) {
    const rr = r + dr, cc = c + dc;
    if (rr >= 0 && rr < n && cc >= 0 && cc < n) next[rr * n + cc] ^= 1;
  }
  return next;
}

// ── Rendering ────────────────────────────────────────────────────────────

// Build the grid of buttons for the current size. Only classes change after this.
function buildBoard() {
  const board = $('lo');
  board.innerHTML = '';
  board.style.setProperty('--n', state.n);
  board.style.setProperty('--cell', `${CELL_PX[state.n]}px`);
  state.cells = [];
  for (let i = 0; i < state.n * state.n; i++) {
    const b = document.createElement('button');
    b.type = 'button';
    b.className = 'lo-cell';
    b.dataset.cell = String(i);
    b.setAttribute('aria-label', `${labelOf(i, state.n)}, press`);
    b.onclick = () => press(i);
    const num = document.createElement('span');
    num.className = 'lo-num';
    b.appendChild(num);
    board.appendChild(b);
    state.cells.push({ el: b, num });
  }
}

function paint() {
  const planIndex = new Map((state.plan || []).map((cell, k) => [cell, k + 1]));
  state.cells.forEach(({ el, num }, i) => {
    el.classList.toggle('on', !!state.board[i]);
    const k = planIndex.get(i);
    el.classList.toggle('plan', k !== undefined);
    num.textContent = k !== undefined ? String(k) : '';
    el.classList.toggle('next', state.hintCell === i);
    el.setAttribute('aria-label', `${labelOf(i, state.n)}, ${state.board[i] ? 'lit' : 'dark'}`);
    el.disabled = state.busy;
  });
  $('lo').classList.toggle('solved', state.over);
  $('chip-lit').textContent = String(litCount());
  $('chip-presses').textContent = String(state.moves);
  $('chip-min').textContent = state.minStart === null ? '—' : String(state.minStart);
  $('chip-rank').textContent = state.info ? `${state.info.rank} / ${state.n * state.n}` : '—';
  $('chip-status').textContent = statusText();
  $('btn-hint').disabled = state.busy || state.over;
  $('btn-solve').disabled = state.busy || state.over;
  $('btn-step').disabled = state.busy || state.over;
  $('btn-auto').disabled = state.busy || state.over;
  $('btn-auto').setAttribute('aria-pressed', String(!!state.autoTimer));
  $('btn-auto').querySelector('.btn-txt').textContent = state.autoTimer ? '■ STOP' : '▶▶ AUTO';
  renderPlan();
}

function statusText() {
  if (state.busy) return 'THINKING';
  if (state.over) return 'SOLVED';
  if (state.info && !state.info.solvable) return 'UNSOLVABLE';
  return 'PLAYING';
}

// Summary of the solve for the current board: solvable or not, and the solution space.
function describeInfo(info) {
  const cells = info.n * info.n;
  if (!info.solvable) {
    return `Cannot be cleared. Rank ${info.rank} of ${cells}: only 2^${info.rank} of the 2^${cells} boards are reachable, and this one is not among them.`;
  }
  const sizes = [...new Set(info.solution_sizes)].join(', ');
  return `Rank ${info.rank} of ${cells}, nullity ${info.nullity}: ${info.solutions} solutions with ${sizes} presses. Minimum: ${info.min_presses}.`;
}

// The panel text. Solvability and the solution-space summary describe the dealt board, and they stay true
// as the player presses (presses never change the class of a board). Only a shown plan changes with play.
function renderPlan() {
  const title = $('plan-title');
  const list = $('plan-list');
  list.textContent = '';
  if (state.over) {
    title.textContent = 'Every light is dark.';
  } else if (state.plan && state.plan.length) {
    title.textContent = `${state.plan.length} presses left in the solution shown.`;
    list.textContent = state.plan.map((cell, k) => `${k + 1}. ${labelOf(cell, state.n)}`).join('   ');
  } else if (state.info && (!state.info.solvable || state.moves + state.assists === 0)) {
    title.textContent = describeInfo(state.info);
  } else {
    title.textContent = 'Press SOLVE or HINT for the minimum solution from here.';
  }
}

function log(text, cls = 'log-info') {
  const line = document.createElement('div');
  line.className = cls;
  line.textContent = text;
  $('log').appendChild(line);
  $('log').scrollTop = $('log').scrollHeight;
}

// ── Game flow ────────────────────────────────────────────────────────────

// Press a cell. The player's presses count toward the total. STEP and AUTO use the same
// path with `assisted`, so the board and plan stay consistent either way.
function press(cell, { assisted = false } = {}) {
  if (state.over || state.busy) return;
  const before = litCount();
  state.board = toggled(state.n, state.board, cell);
  state.hintCell = null;
  if (assisted) state.assists += 1;
  else state.moves += 1;

  // Pressing cell p flips p in the remaining solution (A e_p is added to the board).
  if (state.plan) {
    const k = state.plan.indexOf(cell);
    if (k >= 0) state.plan.splice(k, 1);
    else state.plan.push(cell);
  }

  const after = litCount();
  const delta = after - before;
  pop(state.cells[cell].el, delta > 0 ? `+${delta}` : `${delta}`, delta > 0 ? '#ff00a0' : '#00ff88');
  log(`${assisted ? 'assist' : 'you'}: ${labelOf(cell, state.n)} (lit ${before} → ${after})`,
      assisted ? 'log-adv' : 'log-move');
  if (after === 0) finish();
  paint();
}

function finish() {
  state.over = true;
  stopAuto();
  state.plan = null;
  state.hintCell = null;
  const assist = state.assists ? `, plus ${state.assists} assisted` : '';
  const presses = state.moves === 1 ? 'press' : 'presses';
  const sub = `${state.moves} ${presses}${assist} · minimum from this board: ${state.minStart}`;
  $('lo-note').textContent = 'Every light is dark.';
  burst($('lo'), { count: 160, colors: ['#00ff88', '#00f5ff', '#ffe600'] });
  shake($('lo'), 'big');
  banner('LIGHTS OUT', sub, '#00ff88');
  log(`★ solved: ${sub}`, 'log-best');
}

// Ask the server to solve the current board and remember the answer.
async function solveCurrent() {
  const info = await postJSON('/api/lightsout/solve', { n: state.n, board: state.board });
  state.info = info;
  return info;
}

// Solve the current board and show the minimum solution, or highlight only its first press (hint).
async function refresh({ hint = false } = {}) {
  if (state.over || state.busy) return;
  state.busy = true;
  paint();
  try {
    const info = await solveCurrent();
    if (!info.solvable) {
      state.plan = null;
      log('✗ this board cannot be cleared: no press sequence reaches dark', 'log-err');
    } else if (info.presses.length === 0) {
      state.plan = null;
      state.hintCell = null;
    } else if (hint) {
      state.plan = info.presses.slice();
      state.hintCell = info.presses[0];
      log(`✦ hint: press ${info.moves[0]} next (${info.min_presses} presses in the minimum solution)`, 'log-best');
    } else {
      state.plan = info.presses.slice();
      state.hintCell = null;
      log(`◉ minimum solution: ${info.min_presses} presses: ${info.moves.join(' ')}`, 'log-best');
    }
  } catch (err) {
    log(`✗ ${err.message}`, 'log-err');
  }
  state.busy = false;
  elimReset();
  paint();
}

// Press the next cell of the shown solution (solving first if none is shown).
async function stepOnce() {
  if (state.over || state.busy) return;
  if (!state.plan || !state.plan.length) await refresh();
  if (state.plan && state.plan.length) press(state.plan[0], { assisted: true });
}

function stopAuto() {
  if (state.autoTimer) clearInterval(state.autoTimer);
  state.autoTimer = 0;
}

async function toggleAuto() {
  if (state.autoTimer) {
    stopAuto();
    paint();
    return;
  }
  if (!state.plan || !state.plan.length) await refresh();
  if (!state.plan || !state.plan.length) return;
  state.autoTimer = setInterval(() => {
    if (!state.plan || !state.plan.length || state.over) {
      stopAuto();
      paint();
      return;
    }
    press(state.plan[0], { assisted: true });
  }, PLAY_MS);
  paint();
}

function reset() {
  stopAuto();
  state.board = state.start.slice();
  state.moves = 0;
  state.assists = 0;
  state.over = false;
  state.plan = null;
  state.hintCell = null;
  state.info = state.startInfo;
  $('lo-note').textContent = 'Back to the dealt board.';
  $('log').innerHTML = '';
  log('↺ reset to the dealt board');
  elimReset();
  paint();
}

async function newBoard() {
  stopAuto();
  state.busy = true;
  paint();
  try {
    const n = Number($('size').value);
    const solvable = $('mode').value === 'solvable';
    const dealt = await getJSON('/api/lightsout/random', { n, solvable });
    Object.assign(state, {
      n, start: dealt.board.slice(), board: dealt.board.slice(), moves: 0, assists: 0,
      plan: null, hintCell: null, over: false, info: null, startInfo: null, minStart: null,
    });
    buildBoard();
    $('log').innerHTML = '';
    const info = await solveCurrent();
    state.startInfo = info;
    state.minStart = info.solvable ? info.min_presses : null;
    $('lo-note').textContent = info.solvable
      ? 'Click a light to press it. Goal: every light dark.'
      : 'This board cannot be cleared. Pressing keeps it in the same class, so try NEW BOARD.';
    log(`new ${n}x${n} board: ${info.solvable ? `minimum ${info.min_presses} presses` : 'unsolvable'}`,
        info.solvable ? 'log-info' : 'log-err');
  } catch (err) {
    log(`✗ ${err.message}`, 'log-err');
  }
  state.busy = false;
  elimReset();
  paint();
}

// ── Elimination viewer ───────────────────────────────────────────────────

// Replay the first `frame` steps on the starting matrix. Each step is a swap, then XOR of the
// pivot row into the cleared rows, which is exactly what the solver did.
function replay(trace, frame) {
  const rows = trace.start.map((s) => Array.from(s, Number));
  for (let k = 0; k < frame; k++) {
    const st = trace.steps[k];
    if (st.swap) {
      const [a, b] = st.swap;
      [rows[a], rows[b]] = [rows[b], rows[a]];
    }
    if (st.pivot !== null) {
      const pivot = rows[st.pivot];
      for (const r of st.cleared) rows[r] = rows[r].map((v, c) => v ^ pivot[c]);
    }
  }
  return rows;
}

function describeStep(st, n) {
  const col = labelOf(st.col, n);
  if (st.pivot === null) return `Column ${col} has no pivot: that press is free, so it is a null-space direction.`;
  const swap = st.swap ? ` (row ${st.swap[1] + 1} swapped up)` : '';
  const cleared = st.cleared.length
    ? `XOR row ${st.pivot + 1} into rows ${st.cleared.map((r) => r + 1).join(', ')}`
    : `no other row has a 1 in column ${col}`;
  return `Column ${col}: pivot in row ${st.pivot + 1}${swap}; ${cleared}.`;
}

function elimReset() {
  stopElim();
  const el = state.info && state.info.elimination;
  state.elim = el ? { trace: el, n: state.info.n, frame: 0 } : null;
  drawElim();
}

function stopElim() {
  if (state.elimTimer) clearInterval(state.elimTimer);
  state.elimTimer = 0;
  $('elim-play').setAttribute('aria-pressed', 'false');
  $('elim-play').querySelector('.btn-txt').textContent = '▶ PLAY';
}

// Canvas redraw. The augmented matrix [A | b] shows as lit or dark cells; the pink column is b.
function drawElim() {
  const canvas = $('elim');
  const ctx = canvas.getContext('2d');
  const E = state.elim;
  if (!E) {
    canvas.width = 420; canvas.height = 40;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    $('elim-count').textContent = '';
    $('elim-title').textContent = state.info ? 'Elimination is drawn for boards up to 7x7.' : 'Each row is one light\'s equation. Pink is the lit state.';
    $('elim-note').textContent = '';
    return;
  }
  const { trace, n } = E;
  const N = n * n;
  const cols = N + 1;
  const steps = trace.steps;
  const rows = replay(trace, E.frame);
  const cell = Math.max(8, Math.min(16, Math.floor(560 / cols)));
  const left = 28, top = 14, gap = 8;
  canvas.width = left + cols * cell + gap;
  canvas.height = top + N * cell + 4;
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  // Column of the step just applied, and pivots placed so far.
  const x = (c) => left + c * cell + (c === N ? gap : 0);
  const y = (r) => top + r * cell;
  if (E.frame > 0) {
    const cur = steps[E.frame - 1];
    ctx.fillStyle = 'rgba(255,230,0,0.12)';
    ctx.fillRect(x(cur.col), top, cell, N * cell);
  }
  for (let r = 0; r < N; r++) {
    for (let c = 0; c < cols; c++) {
      const on = rows[r][c] === 1;
      ctx.fillStyle = c === N ? (on ? '#ff00a0' : 'rgba(255,0,160,0.08)') : (on ? '#00f5ff' : 'rgba(0,245,255,0.05)');
      ctx.fillRect(x(c) + 1, y(r) + 1, cell - 2, cell - 2);
    }
  }
  ctx.lineWidth = 2;
  for (let k = 0; k < E.frame; k++) {
    const st = steps[k];
    if (st.pivot === null) continue;
    ctx.strokeStyle = k === E.frame - 1 ? '#ffe600' : '#00ff88';
    ctx.strokeRect(x(st.col) + 1, y(st.pivot) + 1, cell - 2, cell - 2);
  }
  // Row numbers, when there is room for them.
  if (cell >= 10) {
    ctx.fillStyle = '#4a7a9b';
    ctx.font = '9px monospace';
    ctx.textAlign = 'right';
    for (let r = 0; r < N; r++) ctx.fillText(String(r + 1), left - 4, y(r) + cell * 0.7);
  }

  const placed = steps.slice(0, E.frame).filter((s) => s.pivot !== null).length;
  $('elim-count').textContent = `step ${E.frame} / ${steps.length}`;
  $('elim-title').textContent = `Pivots placed: ${placed} of ${state.info.rank}. Outlined cells are pivots (green), the current one is yellow. Elimination of the last board solved.`;
  let note;
  if (E.frame === 0) {
    note = 'Start: each row is one light\'s equation, the presses that touch it on the left and its lit state (pink) on the right.';
  } else {
    note = describeStep(steps[E.frame - 1], n);
  }
  if (E.frame === steps.length) {
    note += state.info && state.info.solvable
      ? ' Done. Every pivot column is a unit vector, so the presses can be read off, and the free columns span the null space.'
      : ' Done. A row reads 0 = 1 (pink, with no pivot), so the board is outside the reachable set.';
  }
  $('elim-note').textContent = note;
}

function stepElim(delta) {
  if (!state.elim) return;
  stopElim();
  state.elim.frame = Math.max(0, Math.min(state.elim.trace.steps.length, state.elim.frame + delta));
  drawElim();
}

function toggleElimPlay() {
  if (!state.elim) return;
  if (state.elimTimer) {
    stopElim();
    return;
  }
  if (state.elim.frame >= state.elim.trace.steps.length) state.elim.frame = 0;
  $('elim-play').setAttribute('aria-pressed', 'true');
  $('elim-play').querySelector('.btn-txt').textContent = '■ STOP';
  state.elimTimer = setInterval(() => {
    if (state.elim.frame >= state.elim.trace.steps.length) {
      stopElim();
      return;
    }
    state.elim.frame += 1;
    drawElim();
  }, FRAME_MS);
}

// ── Startup ──────────────────────────────────────────────────────────────

async function init() {
  $('btn-new').onclick = newBoard;
  $('btn-hint').onclick = () => refresh({ hint: true });
  $('btn-solve').onclick = () => refresh();
  $('btn-step').onclick = stepOnce;
  $('btn-auto').onclick = toggleAuto;
  $('btn-reset').onclick = reset;
  $('size').onchange = newBoard;
  $('mode').onchange = newBoard;
  $('elim-prev').onclick = () => stepElim(-1);
  $('elim-next').onclick = () => stepElim(1);
  $('elim-play').onclick = toggleElimPlay;
  try {
    state.meta = await getJSON('/api/lightsout/meta');
  } catch (err) {
    log(`✗ Server unreachable (${err.message}).`, 'log-err');
    $('chip-status').textContent = 'OFFLINE';
    return;
  }
  for (const s of state.meta.sizes) {
    const opt = document.createElement('option');
    opt.value = String(s);
    opt.textContent = `${s} × ${s}`;
    if (s === state.meta.default_size) opt.selected = true;
    $('size').appendChild(opt);
  }
  await newBoard();
}

init();
