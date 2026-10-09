// Nonogram page.
//
// The player fills or crosses cells. Clues come from the picker or the random generator. SOLVE, HINT and
// DEDUCE call the server, which runs the line solver and the SAT/hybrid search. This file only draws what
// comes back: the line solver's rounds for DEDUCE, and the recorded search events for the search view.
// Nothing here solves a puzzle. The search view replays the recorded events at a fixed pace, then the exact
// counters from the server replace the replay's running totals.

import { getJSON, postJSON } from './api.js';
import { banner, burst, pop, shake } from './fx.js';

const $ = (id) => document.getElementById(id);
const REDUCED = matchMedia('(prefers-reduced-motion: reduce)').matches;
const EVENT_MS = 40;           // one replay tick: 40 ms, as the search view advertises
const MAX_TICKS = 150;         // long recordings are batched into at most 150 ticks (about 6 s)
const MAX_BACKTRACK_LOGS = 40; // red log lines per replay; the rest are only counted
const C = { green: '#00ff88', yellow: '#ffe600', pink: '#ff00a0', cyan: '#00f5ff', red: '#ff3b3b' };

// Player marks. The hint endpoint uses -1 unknown, 0 empty, 1 filled; these constants are the page's own.
const UNKNOWN = 0, FILLED = 1, CROSSED = 2;

const state = {
  serial: 0,          // bumped on every puzzle change, so late server replies can be ignored
  animToken: 0,       // bumped to cancel a running replay or reveal
  id: null, name: '', rows: 0, cols: 0,
  rowClues: [], colClues: [],
  marks: new Uint8Array(0),   // the player's marks
  ded: null,                  // Int8Array: -1 unknown, 0 empty, 1 filled, from DEDUCE or an undecided solve
  revealed: null,             // Uint8Array: cells of the solution that the reveal has lit so far
  second: null,               // Uint8Array: the second solution, when the picture is not unique
  ghostOn: false,
  hint: null,                 // the hint waiting for APPLY HINT
  hintCell: -1,
  cells: [], base: [],        // DOM buttons, and their static classes
  rowEls: [], colEls: [],     // clue elements
  mode: 'fill',               // 'fill' or 'cross': what a plain tap does (touch users switch with the button)
  drag: null,                 // {value, last} while the pointer is down and painting
  won: false,
  busy: false,
};

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
const idx = (r, c) => r * state.cols + c;
const countOf = (values, v) => values.reduce((a, x) => a + (x === v ? 1 : 0), 0);

// Errors from getJSON carry the response body as JSON text; show only the detail message.
function errText(e) {
  const m = /"detail":"([^"]*)"/.exec(e.message);
  return m ? m[1] : e.message;
}

function h(tag, cls) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  return e;
}

function log(text, cls = 'log-info') {
  const box = $('log');
  const line = h('div', cls);
  line.textContent = text;
  box.appendChild(line);
  while (box.children.length > 200) box.firstChild.remove(); // keep the log short
  box.scrollTop = box.scrollHeight;
}

function setStatus(text) {
  $('chip-status').textContent = text;
}

function setBusy(on) {
  state.busy = on;
  for (const id of ['btn-hint', 'btn-solve', 'btn-deduce', 'btn-random']) $(id).disabled = on;
  if (on) setStatus('WORKING');
}

// ── Runs ─────────────────────────────────────────────────────────────────
// A line's clue is the list of run lengths of filled cells. The API uses [] for an empty line, and the page
// shows that as "0". runsOf turns a line of booleans into the same kind of list, so a clue is met exactly
// when the two lists are equal.

function runsOf(values) {
  const out = [];
  let n = 0;
  for (const v of values) {
    if (v) n++;
    else if (n) { out.push(n); n = 0; }
  }
  if (n) out.push(n);
  return out;
}

function sameRuns(runs, clue) {
  const want = clue.filter((v) => v > 0); // [0] and [] both mean an empty line
  return runs.length === want.length && runs.every((v, k) => v === want[k]);
}

function fillClue(el, clue) {
  const nums = clue.filter((v) => v > 0);
  el.replaceChildren(...(nums.length ? nums : [0]).map((n) => {
    const s = h('span');
    s.textContent = String(n);
    return s;
  }));
}

// ── Building and painting the board ──────────────────────────────────────

// Build the grid for the current puzzle. The board is one CSS grid: the first row holds the corner and the
// column clues, and each later row holds a row clue followed by its cells.
function build() {
  const { rows, cols } = state;
  const board = $('nono');
  board.replaceChildren();
  board.style.setProperty('--rows', String(rows));
  board.style.setProperty('--cols', String(cols));
  board.append(h('div', 'nono-corner'));
  state.colEls = state.colClues.map((clue, c) => {
    const d = h('div', 'nono-ccol');
    d.dataset.col = String(c);
    fillClue(d, clue);
    board.append(d);
    return d;
  });
  state.rowEls = [];
  state.cells = [];
  state.base = [];
  for (let r = 0; r < rows; r++) {
    const rc = h('div', 'nono-rclue');
    rc.dataset.row = String(r);
    fillClue(rc, state.rowClues[r]);
    board.append(rc);
    state.rowEls.push(rc);
    for (let c = 0; c < cols; c++) {
      const i = idx(r, c);
      const b = h('button', 'nono-cell');
      b.type = 'button';
      b.dataset.i = String(i);
      b.tabIndex = i === 0 ? 0 : -1; // one tab stop; the arrow keys move between cells
      // Every fifth line gets a brighter border, so the grid can be counted by eye.
      let base = 'nono-cell';
      if (c % 5 === 4 && c < cols - 1) base += ' e-r';
      if (r % 5 === 4 && r < rows - 1) base += ' e-b';
      state.base.push(base);
      state.cells.push(b);
      board.append(b);
    }
  }
  for (let i = 0; i < rows * cols; i++) paintCell(i);
}

// Set one cell's classes from the state. The transient flashes (lineflash, bad, dec-*) are kept, so
// repainting a cell does not cut a running animation short.
function paintCell(i) {
  const b = state.cells[i];
  if (!b) return;
  const m = state.marks[i];
  let cls = state.base[i];
  if (m === FILLED) cls += ' f';
  else if (m === CROSSED) cls += ' x';
  const d = state.ded ? state.ded[i] : -1;
  if (d === 1) cls += ' dd1';
  else if (d === 0) cls += ' dd0';
  if (state.revealed && state.revealed[i]) cls += ' sol';
  if (state.ghostOn && state.second && state.second[i]) cls += ' ghost';
  if (i === state.hintCell) cls += ' hint';
  for (const t of ['lineflash', 'bad', 'dec-0', 'dec-1']) if (b.classList.contains(t)) cls += ` ${t}`;
  b.className = cls;
  const word = m === FILLED ? 'filled' : m === CROSSED ? 'crossed' : 'unknown';
  b.setAttribute('aria-label', `row ${Math.floor(i / state.cols) + 1} column ${(i % state.cols) + 1}, ${word}`);
}

function paintAll() {
  for (let i = 0; i < state.rows * state.cols; i++) paintCell(i);
}

// Mark each clue as done when the player's filled cells in that line give exactly that clue.
// Returns how many of the rows and columns are met.
function refreshClues() {
  const { rows, cols } = state;
  let met = 0;
  for (let r = 0; r < rows; r++) {
    const vals = [];
    for (let c = 0; c < cols; c++) vals.push(state.marks[idx(r, c)] === FILLED);
    const ok = sameRuns(runsOf(vals), state.rowClues[r]);
    state.rowEls[r].classList.toggle('done', ok);
    if (ok) met++;
  }
  for (let c = 0; c < cols; c++) {
    const vals = [];
    for (let r = 0; r < rows; r++) vals.push(state.marks[idx(r, c)] === FILLED);
    const ok = sameRuns(runsOf(vals), state.colClues[c]);
    state.colEls[c].classList.toggle('done', ok);
    if (ok) met++;
  }
  $('chip-lines').textContent = `${met}/${rows + cols}`;
  return met;
}

// Win check: every row and column clue matches the marks. Any grid that meets every clue is a solution,
// and uniqueness is checked when the puzzle is made, so this is the picture.
function checkWin() {
  if (!state.rows) return;
  const total = state.rows + state.cols;
  const won = refreshClues() === total;
  if (won && !state.won) {
    setStatus('SOLVED BY HAND');
    banner('SOLVED BY HAND', `${state.name}: every clue matches`, C.green);
    burst($('nono'));
    log('solved by hand: every row and column clue matches', 'log-best');
  } else if (!won && state.won) {
    setStatus('PLAYING');
  }
  state.won = won;
}

// Chips and win state after the player's marks change.
function afterMarks() {
  $('chip-marked').textContent = String(countOf(state.marks, FILLED));
  checkWin();
}

// Change one mark. Returns true when the cell changed.
function setMark(i, v) {
  if (state.marks[i] === v) return false;
  state.marks[i] = v;
  if (i === state.hintCell) clearHint();
  paintCell(i);
  return true;
}

// ── Puzzle loading ───────────────────────────────────────────────────────

function setPuzzle(p) {
  state.serial++;
  state.animToken++; // stop any replay or reveal from the old puzzle
  Object.assign(state, {
    id: p.id ?? null, name: p.name, rows: p.rows, cols: p.cols,
    rowClues: p.row_clues, colClues: p.col_clues,
    marks: new Uint8Array(p.rows * p.cols),
    ded: null, revealed: null, second: null, ghostOn: false,
    hint: null, hintCell: -1, drag: null, won: false,
  });
  $('btn-apply').disabled = true;
  $('btn-ghost').disabled = true;
  $('btn-ghost').setAttribute('aria-pressed', 'false');
  $('btn-ded-clear').disabled = true;
  $('chip-size').textContent = `${p.rows}x${p.cols}`;
  $('chip-marked').textContent = '0';
  $('ded-fill').style.width = '0%';
  $('ded-title').textContent = 'DEDUCE runs the line solver one line at a time. Green cells are the ones it can prove.';
  $('ded-count').textContent = 'no deduction yet';
  $('ded-note').textContent = '';
  resetSearchView();
  build();
  refreshClues();
  setStatus('PLAYING');
  $('nono-note').textContent = `${p.name}: ${p.rows} x ${p.cols}. Click or drag to fill. Right-click or shift-click crosses a cell.`;
  log(`loaded ${p.name} (${p.rows} x ${p.cols})`, 'log-best');
}

async function loadPuzzle(id) {
  const serial = state.serial;
  try {
    const p = await getJSON(`/api/nonogram/puzzle/${encodeURIComponent(id)}`);
    if (serial === state.serial) setPuzzle(p);
  } catch (e) {
    log(`✗ puzzle: ${errText(e)}`, 'log-err');
  }
}

async function randomPicture() {
  if (state.busy) return;
  const rows = Number($('rand-rows').value), cols = Number($('rand-cols').value);
  setBusy(true);
  let p;
  try {
    p = await getJSON('/api/nonogram/random', { rows, cols });
  } catch (e) {
    setBusy(false);
    setStatus('PLAYING');
    log(`✗ random ${rows}x${cols}: ${errText(e)}`, 'log-err');
    return;
  }
  setBusy(false);
  // A random picture has no id, so it gets its own picker entry keyed by seed.
  const sel = $('puzzle');
  for (const o of [...sel.options]) if (o.value.startsWith('random:')) o.remove();
  const opt = new Option(`random ${rows}x${cols} (seed ${p.seed})`, `random:${p.seed}`);
  sel.append(opt);
  sel.value = opt.value;
  setPuzzle({ ...p, id: null, name: `Random ${rows}x${cols}` });
  log(`random ${rows}x${cols}, seed ${p.seed}, after ${p.attempts} attempt${p.attempts === 1 ? '' : 's'}`, 'log-info');
}

// ── Painting by hand ─────────────────────────────────────────────────────

function toggleMode() {
  state.mode = state.mode === 'fill' ? 'cross' : 'fill';
  const btn = $('btn-mode');
  const cross = state.mode === 'cross';
  btn.setAttribute('aria-pressed', String(cross));
  btn.querySelector('.btn-txt').textContent = cross ? '✕ CROSS' : '◧ FILL';
}

function toggleCell(i, value) {
  const v = state.marks[i] === value ? UNKNOWN : value; // pressing a cell with the same mark clears it
  if (setMark(i, v)) afterMarks();
}

// Flash one row or column: name is 'r3' or 'c5' (1-based). The cells and the clue light up for a moment.
function flashLine(name, cls = 'lineflash') {
  if (!name || REDUCED || !state.rows) return;
  const axis = name[0], k = Number(name.slice(1)) - 1;
  const cells = [];
  let clue;
  if (axis === 'r') {
    for (let c = 0; c < state.cols; c++) cells.push(idx(k, c));
    clue = state.rowEls[k];
  } else {
    for (let r = 0; r < state.rows; r++) cells.push(idx(r, k));
    clue = state.colEls[k];
  }
  for (const i of cells) state.cells[i]?.classList.add(cls);
  clue?.classList.add('lit');
  setTimeout(() => {
    for (const i of cells) state.cells[i]?.classList.remove(cls);
    clue?.classList.remove('lit');
  }, 480);
}

function clearHint() {
  const old = state.hintCell;
  state.hint = null;
  state.hintCell = -1;
  $('btn-apply').disabled = true;
  if (old >= 0) paintCell(old);
}

// ── HINT ────────────────────────────────────────────────────────────────

async function askHint() {
  if (!state.rows || state.busy) return;
  const serial = state.serial;
  // The hint endpoint wants the marks as rows of cells: -1 unknown, 0 crossed, 1 filled.
  const grid = [];
  for (let r = 0; r < state.rows; r++) {
    const row = [];
    for (let c = 0; c < state.cols; c++) {
      const m = state.marks[idx(r, c)];
      row.push(m === FILLED ? 1 : m === CROSSED ? 0 : -1);
    }
    grid.push(row);
  }
  let res;
  try {
    res = await postJSON('/api/nonogram/hint', { row_clues: state.rowClues, col_clues: state.colClues, grid });
  } catch (e) {
    log(`✗ hint: ${errText(e)}`, 'log-err');
    return;
  }
  if (serial !== state.serial) return;
  if (res.contradiction) {
    // The marks cannot all be right: show which line breaks, and say why.
    flashLine(res.line, 'bad');
    shake($('nono-wrap'));
    log(`✗ ${res.message}`, 'log-err');
    return;
  }
  if (!res.found) {
    log(`hint: ${res.message}`, 'log-info');
    return;
  }
  state.hint = { row: res.row, col: res.col, value: res.value };
  const old = state.hintCell;
  state.hintCell = idx(res.row, res.col);
  if (old >= 0) paintCell(old);
  paintCell(state.hintCell);
  pop(state.cells[state.hintCell], res.value ? 'FILL' : 'EMPTY', C.yellow);
  flashLine(res.line);
  $('btn-apply').disabled = false;
  log(`hint: row ${res.row + 1}, column ${res.col + 1} is ${res.value ? 'filled' : 'empty'}. ${res.reason}`, 'log-best');
}

function applyHint() {
  if (!state.hint) return;
  const { row, col, value } = state.hint;
  clearHint();
  if (setMark(idx(row, col), value ? FILLED : CROSSED)) afterMarks();
  log(`applied: row ${row + 1}, column ${col + 1}`, 'log-info');
}

// ── DEDUCE: replay the line solver's rounds ──────────────────────────────

async function deduce() {
  if (!state.rows || state.busy) return;
  const serial = state.serial;
  const token = ++state.animToken;
  setBusy(true);
  let res;
  try {
    res = await postJSON('/api/nonogram/solve', { row_clues: state.rowClues, col_clues: state.colClues, method: 'line', trace: true });
  } catch (e) {
    setBusy(false);
    log(`✗ deduce: ${errText(e)}`, 'log-err');
    return;
  }
  setBusy(false);
  if (serial !== state.serial) return;

  // Flatten the rounds into one list of lines, in the order the solver visited them.
  const steps = [];
  for (const round of res.rounds || []) {
    for (const ln of round.lines) steps.push({ pass: round.pass, axis: round.axis, line: ln.line, fixed: ln.fixed });
  }
  // Pace: about 1.4 s for the whole run, clamped to 20-120 ms per line.
  const delay = REDUCED ? 0 : Math.min(120, Math.max(20, Math.round(1400 / Math.max(1, steps.length))));
  state.ded = new Int8Array(state.rows * state.cols).fill(-1);
  paintAll();
  $('btn-ded-clear').disabled = false;
  $('ded-fill').style.width = '0%';

  let fixedTotal = 0;
  for (let k = 0; k < steps.length; k++) {
    if (token !== state.animToken) return; // a newer action took over
    const st = steps[k];
    flashLine(st.line);
    const changed = [];
    for (const [r, c, v] of st.fixed) {
      const i = idx(r, c);
      state.ded[i] = v ? 1 : 0;
      changed.push(i);
    }
    fixedTotal += changed.length;
    for (const i of changed) paintCell(i);
    $('ded-title').textContent = `${st.axis === 'rows' ? 'Row' : 'Column'} ${st.line.slice(1)} in pass ${st.pass}`;
    $('ded-count').textContent = `pass ${st.pass}, ${fixedTotal} cells fixed so far (line ${k + 1} of ${steps.length})`;
    $('ded-fill').style.width = `${((k + 1) / steps.length) * 100}%`;
    await sleep(delay);
  }
  if (token !== state.animToken) return;

  // Final verdict: does line solving alone determine the whole grid?
  const unknown = res.partial ? countOf(res.partial.flat(), -1) : null;
  if (res.status === 'contradiction') {
    $('ded-note').textContent = 'The line solver found a contradiction: no picture meets these clues.';
    log('deduce: contradiction in the clues', 'log-err');
  } else if (steps.length === 0) {
    $('ded-note').textContent = 'No line can prove a cell yet. Line solving stops here; the rest needs search.';
    log('deduce: no line can prove a cell', 'log-info');
  } else if (unknown === 0) {
    $('ded-note').textContent = 'Line solving alone finishes this one: every cell is proved.';
    log(`deduce: line solving alone finishes this one (${fixedTotal} cells in ${(res.rounds || []).length} rounds)`, 'log-best');
  } else {
    $('ded-note').textContent = `Line solving stops here with ${unknown} cells unknown. The rest needs search: SOLVE finishes it.`;
    log(`deduce: line solving stops with ${unknown} cells unknown`, 'log-info');
  }
}

function clearDeduction() {
  state.ded = null;
  paintAll();
  $('btn-ded-clear').disabled = true;
  $('ded-fill').style.width = '0%';
  $('ded-count').textContent = 'no deduction yet';
  $('ded-note').textContent = '';
  $('ded-title').textContent = 'DEDUCE runs the line solver one line at a time. Green cells are the ones it can prove.';
}

// ── SOLVE: replay the search, then reveal the picture ────────────────────

// Search view: counters for the replay. The server's event list is decisions, conflicts and backtracks,
// with the cell and value for each decision. Propagations are not recorded per event; they show "..."
// until the exact totals arrive.
function resetSearchView() {
  for (const id of ['ct-dec', 'ct-conf', 'ct-back']) $(id).textContent = '0';
  $('ct-prop').textContent = '0';
  $('ct-vars').textContent = '—';
  $('ct-clauses').textContent = '—';
  $('sat-title').textContent = 'Solve with SAT or HYBRID to see the search. The replay is of a recorded run, not a live stream.';
  $('sat-note').textContent = '';
}

function setCounter(id, v) {
  const el = $(id);
  const text = String(v);
  if (el.textContent === text) return;
  el.textContent = text;
  el.classList.remove('bump');
  void el.offsetWidth; // restart the bump animation
  el.classList.add('bump');
}

// Fill the static parts of the search view from the server reply, before the replay starts.
function showStaticStats(stats, method) {
  const sat = method === 'sat' && stats.variables != null;
  $('ct-vars').textContent = sat ? String(stats.variables) : '—';
  $('ct-clauses').textContent = sat ? String(stats.clauses) : '—';
  $('sat-note').textContent = sat
    ? `Pure literals removed before search: ${stats.pure_eliminated}. Sweeps: ${stats.sweeps}.`
    : 'not a SAT run: hybrid search works on the cells and the line rules directly.';
}

// Play the recorded events. Returns false if a newer action took over.
async function replay(events, truncated, token) {
  const per = REDUCED ? events.length : Math.max(1, Math.ceil(events.length / MAX_TICKS));
  const live = { dec: 0, conf: 0, back: 0 };
  let logged = 0;
  $('sat-title').textContent = 'Replaying the recorded search (not a live stream)...';
  for (let k = 0; k < events.length; k += per) {
    if (token !== state.animToken) return false;
    let conflict = false;
    for (const ev of events.slice(k, k + per)) {
      if (ev[0] === 'd') {
        // A decision: the cell flashes the value the search tried.
        live.dec++;
        const el = state.cells[idx(ev[1], ev[2])];
        if (el && !REDUCED) {
          const v = ev[3] ? 1 : 0;
          el.classList.remove('dec-0', 'dec-1');
          void el.offsetWidth;
          el.classList.add(`dec-${v}`);
        }
      } else if (ev[0] === 'c') {
        live.conf++;
        conflict = true;
      } else if (ev[0] === 'b') {
        // A backtrack: undo the last guess. Logged for the first few only, so the log stays readable.
        live.back++;
        if (logged < MAX_BACKTRACK_LOGS) {
          log('backtrack: undo the last guess', 'log-err');
          logged++;
        }
      }
    }
    setCounter('ct-dec', live.dec);
    setCounter('ct-conf', live.conf);
    setCounter('ct-back', live.back);
    setCounter('ct-prop', '…');
    if (conflict) shake($('sat-panel'));
    await sleep(REDUCED ? 0 : EVENT_MS);
  }
  if (live.back > logged) log(`... ${live.back - logged} more backtracks not shown`, 'log-info');
  if (truncated) log('the event list was capped; the counters below are the full run', 'log-info');
  return token === state.animToken;
}

// Light the solution's filled cells one row at a time, with the flicker. Then announce the result.
async function reveal(sol, token) {
  const { rows, cols } = state;
  state.revealed = new Uint8Array(rows * cols);
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const i = idx(r, c);
      if (sol[i]) state.revealed[i] = 1;
    }
    for (let c = 0; c < cols; c++) paintCell(idx(r, c));
    await sleep(REDUCED ? 0 : 55);
    if (token !== state.animToken) return false;
  }
  return true;
}

async function solve() {
  if (!state.rows || state.busy) return;
  const serial = state.serial;
  const token = ++state.animToken; // cancels an older replay still running
  const method = $('method').value;
  resetSearchView();
  $('chip-method').textContent = method.toUpperCase();
  setBusy(true);
  log(`solve: ${method} on ${state.rows}x${state.cols}`, 'log-info');
  let res;
  try {
    res = await postJSON('/api/nonogram/solve', {
      // trace: the server records the search events (decisions, conflicts, backtracks) for the replay.
      row_clues: state.rowClues, col_clues: state.colClues, method, trace: true,
    });
  } catch (e) {
    setBusy(false);
    setStatus('PLAYING');
    log(`✗ solve: ${errText(e)}`, 'log-err');
    return;
  }
  setBusy(false);
  if (serial !== state.serial) return;
  const s = res.stats;
  showStaticStats(s, method);

  // Replay the search first, so the solution appears as its end.
  if (res.events && res.events.length && (method === 'sat' || method === 'hybrid')) {
    const finished = await replay(res.events, res.events_truncated, token);
    if (!finished) return;
  }
  // Exact totals from the server replace the replay's running numbers.
  setCounter('ct-dec', s.decisions);
  setCounter('ct-conf', s.conflicts);
  setCounter('ct-back', s.backtracks);
  setCounter('ct-prop', s.propagations);
  $('sat-title').textContent = `${method.toUpperCase()} finished in ${s.seconds.toFixed(3)} s`;

  if (res.status === 'unique' || res.status === 'multiple') {
    state.second = res.second ? Uint8Array.from(res.second.flat()) : null;
    $('btn-ghost').disabled = !state.second;
    setStatus(res.status === 'unique' ? 'UNIQUE' : 'MULTIPLE');
    const sol = Uint8Array.from(res.solution.flat());
    const finished = await reveal(sol, token);
    if (!finished) return;
    if (res.status === 'unique') {
      banner('SOLVED', `${method} found the one picture that fits`, C.green);
      log(`solved: unique solution in ${s.seconds.toFixed(3)} s`, 'log-best');
    } else {
      banner('MULTIPLE SOLUTIONS', 'the clues fit more than one picture; SECOND ANSWER shows another', C.yellow);
      log('solved: the clues fit more than one picture', 'log-best');
    }
  } else if (res.status === 'contradiction') {
    setStatus('NO SOLUTION');
    shake($('nono-wrap'), 'big');
    banner('NO SOLUTION', 'these clues cannot all hold', C.red);
    log('no solution: the clues contradict each other', 'log-err');
  } else {
    // Undecided: the server's time limit or a budget stopped the search. Show the cells it did prove, in green.
    setStatus('UNDECIDED');
    if (res.partial) {
      state.ded = Int8Array.from(res.partial.flat());
      paintAll();
      $('btn-ded-clear').disabled = false;
    }
    if (res.timed_out) {
      const took = `${s.seconds.toFixed(1)} s`;
      banner('UNDECIDED: time limit', `the server stopped the search after ${took}; the proved cells show in green`, C.yellow);
      log(`undecided: the time limit ran out after ${took}`, 'log-info');
    } else {
      banner('UNDECIDED: budget', 'the search stopped at its budget; the proved cells show in green', C.yellow);
      log('undecided: the search hit its budget', 'log-info');
    }
  }
}

// ── Second answer, clearing marks ────────────────────────────────────────

function toggleGhost() {
  if (!state.second) return;
  state.ghostOn = !state.ghostOn;
  $('btn-ghost').setAttribute('aria-pressed', String(state.ghostOn));
  paintAll();
}

function clearMarks() {
  state.marks.fill(UNKNOWN);
  clearHint();
  paintAll();
  $('chip-marked').textContent = '0';
  checkWin();
}

// ── Keyboard ────────────────────────────────────────────────────────────

function focusCell(i) {
  const old = state.cells.findIndex((b) => b.tabIndex === 0);
  if (old >= 0) state.cells[old].tabIndex = -1;
  const el = state.cells[i];
  if (!el) return;
  el.tabIndex = 0;
  el.focus();
}

function onKey(e) {
  const cell = e.target.closest?.('.nono-cell');
  if (!cell) return;
  const i = Number(cell.dataset.i);
  const r = Math.floor(i / state.cols), c = i % state.cols;
  const moves = { ArrowUp: [r - 1, c], ArrowDown: [r + 1, c], ArrowLeft: [r, c - 1], ArrowRight: [r, c + 1] };
  if (moves[e.key]) {
    const [nr, nc] = moves[e.key];
    if (nr >= 0 && nr < state.rows && nc >= 0 && nc < state.cols) focusCell(idx(nr, nc));
    e.preventDefault();
    return;
  }
  if (e.key === ' ' || e.key === 'Enter') {
    e.preventDefault();
    toggleCell(i, e.shiftKey ? CROSSED : FILLED);
  } else if (e.key === 'x' || e.key === 'X') {
    e.preventDefault();
    toggleCell(i, CROSSED);
  }
}

// ── Wiring ──────────────────────────────────────────────────────────────

function wire() {
  const board = $('nono');
  // Pointer painting. The first cell of a drag decides the value: a cell already marked with the drag's
  // action is cleared, and every cell the pointer crosses after it takes the same value.
  board.addEventListener('pointerdown', (e) => {
    const cell = e.target.closest('.nono-cell');
    if (!cell || e.button > 2) return; // main button and right button only
    e.preventDefault();
    const i = Number(cell.dataset.i);
    const cross = e.button === 2 || e.shiftKey || state.mode === 'cross';
    const action = cross ? CROSSED : FILLED;
    const value = state.marks[i] === action ? UNKNOWN : action;
    state.drag = { value, last: i };
    if (setMark(i, value)) afterMarks();
  });
  board.addEventListener('pointermove', (e) => {
    if (!state.drag) return;
    // elementFromPoint finds the cell under the pointer. For touch, pointermove stays on the element where
    // the touch began, so the target alone is not enough.
    const cell = document.elementFromPoint(e.clientX, e.clientY)?.closest?.('.nono-cell');
    if (!cell) return;
    const i = Number(cell.dataset.i);
    if (i === state.drag.last) return;
    state.drag.last = i;
    if (setMark(i, state.drag.value)) afterMarks();
  });
  const endDrag = () => { state.drag = null; };
  window.addEventListener('pointerup', endDrag);
  window.addEventListener('pointercancel', endDrag);
  board.addEventListener('contextmenu', (e) => e.preventDefault());
  board.addEventListener('keydown', onKey);
  board.addEventListener('focusin', (e) => {
    const cell = e.target.closest?.('.nono-cell');
    if (!cell) return;
    // Keep the roving tab stop on the cell the user reached.
    for (const b of state.cells) b.tabIndex = -1;
    cell.tabIndex = 0;
  });

  $('btn-mode').addEventListener('click', toggleMode);
  $('btn-hint').addEventListener('click', askHint);
  $('btn-apply').addEventListener('click', applyHint);
  $('btn-solve').addEventListener('click', solve);
  $('btn-deduce').addEventListener('click', deduce);
  $('btn-ghost').addEventListener('click', toggleGhost);
  $('btn-reset').addEventListener('click', clearMarks);
  $('btn-ded-clear').addEventListener('click', clearDeduction);
  $('btn-random').addEventListener('click', randomPicture);
  $('puzzle').addEventListener('change', (e) => {
    const v = e.target.value;
    if (v.startsWith('random:')) return; // a random picture is already loaded
    loadPuzzle(v);
  });
  $('method').addEventListener('change', (e) => {
    $('chip-method').textContent = e.target.value.toUpperCase();
  });
}

async function init() {
  wire();
  try {
    const [meta, list] = await Promise.all([
      getJSON('/api/nonogram/meta'),
      getJSON('/api/nonogram/puzzles'),
    ]);
    for (const id of ['rand-rows', 'rand-cols']) {
      const sel = $(id);
      sel.replaceChildren(...meta.sizes.map((n) => new Option(String(n), String(n))));
      sel.value = String(Math.min(10, meta.sizes[meta.sizes.length - 1]));
    }
    const sel = $('puzzle');
    sel.replaceChildren(...list.puzzles.map((p) => new Option(`${p.name} (${p.rows}x${p.cols})`, p.id)));
    if (list.puzzles.length) await loadPuzzle(list.puzzles[0].id);
  } catch (e) {
    setStatus('OFFLINE');
    log(`✗ could not reach the server: ${errText(e)}`, 'log-err');
  }
}

init();
