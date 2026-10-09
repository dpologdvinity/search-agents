// Minesweeper page.
//
// The board, the mines, and the flood fill run here. The mines are placed on the first click, around it,
// and stay in the page. The server only ever sees what a player sees (the numbers and the covered cells).
// It answers with the cells the agent can prove safe or a mine, and the mine probability of every covered
// cell (exact, or an estimate when the board is too wide to count within the server's work budget; the
// reply says which). The page animates those answers: a certain move glows green, and a guess glows pink
// with its odds.
//
// Proofs stay true as the game goes on. A cell proven safe is still safe after other cells are revealed,
// so one analysis can drive a whole round of reveals, one per animation beat.

import { postJSON } from './api.js';
import { banner, burst, pop, shake } from './fx.js';

const $ = (id) => document.getElementById(id);
const COVERED = -1;
const PRESETS = {
  beginner: { rows: 9, cols: 9, mines: 10, cell: 52 },
  intermediate: { rows: 16, cols: 16, mines: 40, cell: 36 },
  expert: { rows: 16, cols: 30, mines: 99, cell: 28 },
};
const CHIP_TEXT = { PLAYING: 'PLAYING', THINKING: 'THINKING', CLEARED: 'CLEARED', BOOM: 'BOOM', READY: 'READY' };

const state = {
  preset: 'beginner',
  rows: 9, cols: 9, mines: 10,
  nb: [],          // neighbour lists, one per cell
  values: [],      // COVERED, or the number shown (0..8)
  mine: null,      // Set of mine cells, placed on the first click
  flags: new Set(), // the player's flags (notes only)
  proven: new Set(), // mines the agent has proven this game
  analysis: null,  // latest server analysis for the current board
  guessCell: -1,   // the cell of the last guess, for its highlight
  flashCell: -1, flashKind: '', // the cell that just moved, and whether it was certain
  boom: -1,        // the mine the player hit
  over: false, won: false,
  busy: false,     // an AI step is running
  auto: false,     // AI PLAY is on
  flagMode: false,
  moves: 0,        // reveals by the player
  guesses: 0,      // reveals the agent made without a proof
  rounds: 0,
  gen: 0,          // bumped by every new board
};

// ── Board ────────────────────────────────────────────────────────────────

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
const labelOf = (cell) => {
  // Spreadsheet-style column name, then row number from 1: a..z, then aa..ad on the expert board.
  let n = (cell % state.cols) + 1, col = '';
  while (n) { const r = (n - 1) % 26; col = String.fromCharCode(97 + r) + col; n = Math.floor((n - 1) / 26); }
  return `${col}${Math.floor(cell / state.cols) + 1}`;
};

function neighbourLists(rows, cols) {
  const out = [];
  for (let i = 0; i < rows * cols; i++) {
    const r = Math.floor(i / cols), c = i % cols, near = [];
    for (let dr = -1; dr <= 1; dr++) for (let dc = -1; dc <= 1; dc++) {
      if ((dr || dc) && r + dr >= 0 && r + dr < rows && c + dc >= 0 && c + dc < cols) near.push((r + dr) * cols + c + dc);
    }
    out.push(near);
  }
  return out;
}

// Mines go anywhere except the first click and its neighbours. A partial Fisher-Yates shuffle over the
// remaining cells picks the mines uniformly.
function placeMines(first) {
  const clear = new Set([first, ...state.nb[first]]);
  const pool = [];
  for (let i = 0; i < state.rows * state.cols; i++) if (!clear.has(i)) pool.push(i);
  for (let k = 0; k < state.mines; k++) {
    const j = k + Math.floor(Math.random() * (pool.length - k));
    [pool[k], pool[j]] = [pool[j], pool[k]];
  }
  state.mine = new Set(pool.slice(0, state.mines));
}

const adjacentMines = (i) => state.nb[i].reduce((a, n) => a + (state.mine.has(n) ? 1 : 0), 0);

// Reveal one cell and return the cells it opened, in order. A zero opens its neighbours, and zeros spread
// the opening breadth first. Flagged cells are left alone, as in the terminal game.
function revealCell(i) {
  if (state.over || state.values[i] !== COVERED || state.flags.has(i)) return [];
  if (!state.mine) placeMines(i);
  if (state.mine.has(i)) {
    state.over = true;
    state.boom = i;
    return [i];
  }
  const opened = [i];
  state.values[i] = adjacentMines(i);
  const queue = [i];
  for (let head = 0; head < queue.length; head++) {
    const here = queue[head];
    if (state.values[here] !== 0) continue; // only zeros spread the opening
    for (const n of state.nb[here]) {
      if (state.values[n] === COVERED && !state.flags.has(n)) {
        state.values[n] = adjacentMines(n);
        opened.push(n);
        queue.push(n);
      }
    }
  }
  if (state.values.filter((v) => v !== COVERED).length === state.rows * state.cols - state.mines) {
    state.over = true;
    state.won = true;
  }
  return opened;
}

function newBoard() {
  const cfg = PRESETS[$('preset').value];
  Object.assign(state, {
    preset: $('preset').value, rows: cfg.rows, cols: cfg.cols, mines: cfg.mines,
    nb: neighbourLists(cfg.rows, cfg.cols), values: new Array(cfg.rows * cfg.cols).fill(COVERED),
    mine: null, flags: new Set(), proven: new Set(), analysis: null, guessCell: -1, flashCell: -1,
    flashKind: '', boom: -1, over: false, won: false, busy: false, auto: false, moves: 0, guesses: 0, rounds: 0,
    gen: (state.gen || 0) + 1, // stale replies and running AI rounds check this and stop
  });
  $('log').innerHTML = '';
  buildBoard();
  log(`new ${state.cols}×${state.rows} board with ${state.mines} mines. The first click is always safe.`, 'log-info');
  paint();
}

// ── Rendering ────────────────────────────────────────────────────────────

// The grid of cells for the current size. Only classes and text change after this.
function buildBoard() {
  const board = $('ms');
  board.innerHTML = '';
  board.style.setProperty('--cols', state.cols);
  board.style.setProperty('--cell', `${PRESETS[state.preset].cell}px`);
  board.style.setProperty('--num-size', `${Math.round(PRESETS[state.preset].cell * 0.46)}px`);
  for (let i = 0; i < state.rows * state.cols; i++) {
    const b = document.createElement('button');
    b.type = 'button';
    b.className = 'ms-cell';
    b.dataset.cell = String(i);
    b.onclick = () => onCellClick(i);
    b.oncontextmenu = (e) => { e.preventDefault(); onCellFlag(i); };
    const num = document.createElement('span');
    num.className = 'ms-num';
    b.appendChild(num);
    board.appendChild(b);
  }
}

const COMP_COUNT = 6;

function paint() {
  const cells = document.querySelectorAll('#ms .ms-cell');
  const comps = new Map(); // cell -> component index
  const analysis = state.analysis;
  const heat = $('heat').checked;
  const showComps = $('comps').checked;
  if (analysis && showComps) analysis.components.forEach((comp, k) => comp.cells.forEach((c) => comps.set(c, k)));

  cells.forEach((el, i) => {
    const v = state.values[i];
    const shown = v !== COVERED;
    const isMine = state.over && state.mine && state.mine.has(i);
    el.classList.toggle('open', shown);
    el.classList.toggle('mine', !!isMine);
    el.classList.toggle('boom', state.boom === i);
    el.classList.toggle('flag', !shown && state.flags.has(i) && !state.over);
    el.classList.toggle('proven', !shown && state.proven.has(i));
    el.classList.toggle('flash-safe', state.flashCell === i && state.flashKind === 'certain');
    el.classList.toggle('flash-guess', state.flashCell === i && state.flashKind === 'guess');
    el.classList.toggle('guess-next', state.guessCell === i && !shown);
    el.classList.toggle('comp', comps.has(i));
    for (let k = 0; k < COMP_COUNT; k++) el.classList.toggle(`comp-${k}`, comps.get(i) === k);
    el.disabled = state.over;
    el.style.backgroundColor = '';
    const num = el.firstChild;
    if (shown) {
      num.textContent = v === 0 ? '' : String(v);
      num.className = `ms-num n${v}`;
      el.setAttribute('aria-label', `${labelOf(i)}, ${v === 0 ? 'empty' : `${v} nearby`}`);
    } else if (state.proven.has(i)) {
      num.textContent = '⚑';
      num.className = 'ms-num ai';
      el.setAttribute('aria-label', `${labelOf(i)}, proven mine`);
    } else if (state.flags.has(i) && !state.over) {
      num.textContent = '⚑';
      num.className = 'ms-num';
      el.setAttribute('aria-label', `${labelOf(i)}, flagged`);
    } else if (isMine) {
      num.textContent = '✸';
      num.className = 'ms-num bomb';
      el.setAttribute('aria-label', `${labelOf(i)}, mine`);
    } else {
      num.textContent = '';
      num.className = 'ms-num';
      el.setAttribute('aria-label', `${labelOf(i)}, covered`);
    }
    // Heat map: green for safe, red for a sure mine. Only covered cells that the analysis priced get a colour.
    if (heat && analysis && !shown && !isMine && analysis.probability[i] !== null && !state.proven.has(i)
        && !state.flags.has(i)) {
      const p = analysis.probability[i];
      el.style.backgroundColor = `hsla(${Math.round(120 * (1 - p))}, 100%, 45%, 0.42)`;
      if (!el.classList.contains('flash-safe') && !el.classList.contains('flash-guess')) {
        num.textContent = `${Math.round(p * 100)}`;
        num.className = 'ms-num heat';
      }
    }
  });

  $('ms').classList.toggle('solved', state.won);
  $('ms').classList.toggle('flagging', state.flagMode);
  $('chip-status').textContent = statusText();
  $('chip-proven').textContent = String(state.proven.size);
  $('chip-revealed').textContent = `${state.values.filter((v) => v !== COVERED).length}/${state.rows * state.cols - state.mines}`;
  $('chip-guesses').textContent = String(state.guesses);
  $('btn-step').disabled = state.over || state.busy || state.auto;
  $('btn-play').disabled = state.over || (state.busy && !state.auto);
  $('btn-play').setAttribute('aria-pressed', String(state.auto));
  $('btn-play').querySelector('.btn-txt').textContent = state.auto ? '■ STOP' : '▶▶ AI PLAY';
  $('btn-flag').setAttribute('aria-pressed', String(state.flagMode));
  $('btn-flag').querySelector('.btn-txt').textContent = state.flagMode ? '⚑ FLAGGING' : '⚑ FLAG MODE';
  renderThinking();
  renderComponents();
}

function statusText() {
  if (state.won) return 'CLEARED';
  if (state.over) return 'BOOM';
  if (state.busy) return 'THINKING';
  if (state.moves + state.guesses + state.rounds === 0 && !state.mine) return 'READY';
  return 'PLAYING';
}

function log(text, cls = 'log-info') {
  const line = document.createElement('div');
  line.className = cls;
  line.textContent = text;
  $('log').appendChild(line);
  $('log').scrollTop = $('log').scrollHeight;
}

// The panel that explains the last analysis in words.
function renderThinking() {
  const a = state.analysis;
  const title = $('think-title');
  if (state.over) {
    title.textContent = state.won
      ? 'Every safe cell is open. Every mine was found by proof or by a guess the odds favoured.'
      : 'The agent does not get a second try. Its last guess was the lowest odds it could find.';
  } else if (!a) {
    title.textContent = 'Press AI STEP. The agent reads the numbers, proves what it can, and only then guesses.';
  } else if (a.safe.length) {
    title.textContent = `${a.safe.length} cell(s) proven safe, ${a.certain_mines.length} proven mine(s). Those are certain, not odds.`;
  } else if (a.guess) {
    const odds = a.exact ? 'at' : 'at an estimated';
    title.textContent = `Nothing is certain. The safest guess is ${a.guess.label} ${odds} ${(a.guess.probability * 100).toFixed(1)}% mine chance.`;
  } else {
    title.textContent = 'Nothing left to decide.';
  }
  const info = $('ms-analysis');
  info.textContent = a
    ? `${a.components.length} component(s); ${a.interior} covered cell(s) touch no number; ${a.remaining} mine(s) left to place.${a.exact ? '' : ' The board was too wide to count within the work budget, so the odds are estimates. The proofs still hold.'}`
    : 'No analysis yet.';
}

function renderComponents() {
  const list = $('comp-list');
  const a = state.analysis;
  list.innerHTML = '';
  if (!a || !$('comps').checked) {
    $('comp-title').textContent = a ? 'Turn on COMPONENTS to outline the groups.' : 'Press AI STEP, or turn on an overlay, to see the groups.';
    return;
  }
  $('comp-title').textContent = a.components.length
    ? 'Each group is counted on its own, then combined with the mine count.'
    : 'No group: every covered cell is either proven or touches no number.';
  a.components.forEach((comp, k) => {
    const row = document.createElement('div');
    row.className = `ms-comp comp-${k % COMP_COUNT}`;
    const bits = Math.round(comp.layouts_log2 * 10) / 10;
    row.textContent = `#${k + 1}: ${comp.cells.length} cells, ${comp.constraints} numbers, about 2^${bits} layouts`;
    list.appendChild(row);
  });
}

// ── Player actions ───────────────────────────────────────────────────────

function onCellClick(i) {
  if (state.busy) return;
  if (state.flagMode) return onCellFlag(i);
  if (state.over || state.values[i] !== COVERED || state.flags.has(i)) return;
  revealPlayer(i);
}

function onCellFlag(i) {
  if (state.over || state.values[i] !== COVERED) return;
  if (state.flags.has(i)) state.flags.delete(i);
  else state.flags.add(i);
  paint();
}

// The player's own reveal. Overlays are refreshed right after it, one request per click.
function revealPlayer(i) {
  const opened = revealCell(i);
  state.moves += 1;
  state.flashCell = -1;
  afterMove(opened, 'you');
  refreshOverlay();
}

// Shared bookkeeping after any reveal: the log line, the endgame announcement, and a repaint.
function afterMove(opened, who) {
  if (state.over && !state.won) {
    log(`✸ ${who === 'you' ? 'you' : 'the agent'} hit a mine at ${labelOf(state.boom)}.`, 'log-err');
    shake($('ms'), 'big');
    banner('BOOM', `${state.moves} reveals, ${state.guesses} guesses, ${state.proven.size} mines proven`, '#ff3b3b');
  } else if (state.won) {
    log(`★ cleared with ${state.moves} player reveals and ${state.guesses} guesses by the agent.`, 'log-best');
    burst($('ms'), { count: 180, colors: ['#00ff88', '#00f5ff', '#ffe600'] });
    shake($('ms'), 'big');
    banner('MINESWEEPER CLEARED', `${state.guesses} guesses, ${state.proven.size} mines proven`, '#00ff88');
  } else if (opened.length > 1) {
    log(`${who === 'you' ? 'you' : 'agent'}: ${labelOf(opened[0])} opened ${opened.length} cells.`,
      who === 'you' ? 'log-move' : 'log-adv');
  }
  paint();
}

// ── Agent ────────────────────────────────────────────────────────────────

// Ask the server what the agent can prove about the board as it stands.
async function analyse() {
  return postJSON('/api/minesweeper/analyze', {
    rows: state.rows, cols: state.cols, mines: state.mines, cells: state.values.slice(), level: 3,
  });
}

// Overlays show the analysis of the current board. Nothing is requested while both are off, and a
// reply that arrives after a new board has started is dropped.
async function refreshOverlay() {
  if (!state.over && ($('heat').checked || $('comps').checked)) {
    const gen = state.gen;
    try {
      const a = await analyse();
      if (gen === state.gen) state.analysis = a;
    } catch (err) {
      log(`✗ ${err.message}`, 'log-err');
    }
  }
  paint(); // always repaint, so a toggle is visible even when no request was needed (e.g. after the game ended)
}

// One AI step is one round of deductions: every cell the analysis proves safe is opened, with a green
// pulse each. If nothing is proven, the agent makes one guess, the lowest exact probability, in pink.
// The mines it proves are marked in green as well, since they are facts for the rest of the game.
async function aiStep() {
  if (state.over || state.busy) return false;
  const gen = state.gen;
  state.busy = true;
  paint();
  let a;
  try {
    a = await analyse();
  } catch (err) {
    state.busy = false;
    log(`✗ ${err.message}`, 'log-err');
    paint();
    return false;
  }
  if (gen !== state.gen) return false; // the board was replaced while the server was thinking
  state.analysis = a;
  for (const m of a.certain_mines) state.proven.add(m);
  if (a.safe.length) {
    state.rounds += 1;
    log(`▸ round ${state.rounds}: ${a.safe.length} proven safe, ${a.certain_mines.length} proven mine(s).`, 'log-best');
    // Pace the beats so a round of many cells still reads as a sequence, and a short round is not slow.
    const beat = Math.max(20, Math.min(110, Math.round(900 / a.safe.length)));
    for (const c of a.safe) {
      if (state.over || gen !== state.gen) break;
      if (state.values[c] !== COVERED) continue; // a flood fill from an earlier beat already opened it
      state.flags.delete(c);
      const opened = revealCell(c);
      state.flashCell = c;
      state.flashKind = 'certain';
      log(`✓ certain: ${labelOf(c)} is safe. ${a.why[c] || 'proven'}.`, 'log-certain');
      pop(document.querySelector(`#ms [data-cell="${c}"]`), 'SAFE', '#00ff88');
      afterMove(opened, 'agent');
      await sleep(beat);
    }
  } else if (a.guess) {
    const c = a.guess.cell;
    const p = a.guess.probability;
    state.guesses += 1;
    state.guessCell = c;
    const kind = a.exact ? 'the lowest odds on the board' : 'the lowest estimated odds (over the work budget)';
    log(`? guess: ${labelOf(c)} at ${(p * 100).toFixed(1)}%, ${kind}.`, 'log-adv');
    state.flags.delete(c); // a player flag on the guess would make revealCell do nothing, and AI PLAY would repeat it
    const opened = revealCell(c);
    state.flashCell = c;
    state.flashKind = 'guess';
    pop(document.querySelector(`#ms [data-cell="${c}"]`), `${(p * 100).toFixed(0)}%`, '#ff00a0');
    afterMove(opened, 'agent');
  } else {
    log('nothing left to decide.', 'log-info');
  }
  // The overlay keeps the analysis from the start of this step. The next step fetches a fresh one,
  // so a round costs one request, which keeps AI PLAY inside the server's per-minute limit.
  state.busy = false;
  paint();
  return true;
}

// AI PLAY repeats AI steps until the board is decided, the player stops it, or the server refuses.
async function togglePlay() {
  if (state.auto) {
    state.auto = false;
    paint();
    return;
  }
  state.auto = true;
  paint();
  while (state.auto && !state.over) {
    let moved = false;
    try {
      moved = await aiStep();
    } catch (err) {
      log(`✗ ${err.message}`, 'log-err');
    }
    if (!moved) break;
    await sleep(120);
  }
  state.auto = false;
  paint();
}

// ── Startup ──────────────────────────────────────────────────────────────

function init() {
  $('btn-step').onclick = () => aiStep();
  $('btn-play').onclick = togglePlay;
  $('btn-new').onclick = () => { state.auto = false; newBoard(); refreshOverlay(); };
  $('btn-flag').onclick = () => { state.flagMode = !state.flagMode; paint(); };
  $('preset').onchange = () => { state.auto = false; newBoard(); refreshOverlay(); };
  $('heat').onchange = refreshOverlay;
  $('comps').onchange = refreshOverlay;
  newBoard();
}

init();
