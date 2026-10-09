// Sudoku page: solve on the server, then replay every guess, forced fill, and backtrack.

import { getJSON, postJSON } from './api.js';
import { banner, burst, shake } from './fx.js';

const $ = (id) => document.getElementById(id);
const NAMES = { plain: 'Plain backtracking', mrv_fc: 'MRV + forward checking', propagate: 'Constraint propagation' };

const state = { puzzle: '', cells: [], given: [], meta: null, timer: null, trace: [], step: 0, result: null };

function parse(text) {
  const cells = [...text.trim()].filter((ch) => '0123456789.'.includes(ch)).map((ch) => (ch === '.' ? 0 : Number(ch)));
  return cells.length === 81 ? cells : null;
}

function render(highlight = -1, kind = '') {
  const el = $('sboard');
  el.innerHTML = '';
  state.cells.forEach((v, i) => {
    const cell = document.createElement('div');
    const r = Math.floor(i / 9), c = i % 9;
    cell.className = 's-cell' + (state.given[i] ? ' given' : '')
      + (c === 2 || c === 5 ? ' br' : '') + (r === 2 || r === 5 ? ' bb' : '')
      + (i === highlight ? ` ${kind}` : '')
      + (state.marks?.[i] === 'propagate' ? ' propagate' : '');
    cell.textContent = v || '';
    el.appendChild(cell);
  });
}

function load(puzzle, label) {
  stop();
  state.puzzle = puzzle;
  state.cells = parse(puzzle);
  state.given = state.cells.map((v) => v !== 0);
  state.marks = [];
  state.trace = [];
  state.step = 0;
  $('puzzle-name').textContent = label;
  $('cmp-body').innerHTML = '';
  for (const id of ['chip-nodes', 'chip-back']) $(id).textContent = '—';
  $('chip-status').textContent = 'READY';
  progress();
  render();
}

function progress() {
  const total = state.trace.length;
  $('progress-lbl').textContent = `${state.step} / ${total}`;
  $('progress-bar').style.width = total ? `${(100 * state.step) / total}%` : '0%';
}

function stop() {
  clearInterval(state.timer);
  state.timer = null;
  $('btn-skip').disabled = true;
}

function applyStep(k) {
  const [kind, cell, value] = state.trace[k];
  if (kind === 'undo') {
    if (Math.random() < 0.15) shake($('sboard'));
    state.cells[cell] = 0;
    state.marks[cell] = '';
  } else {
    state.cells[cell] = value;
    state.marks[cell] = kind;
  }
  return [cell, kind];
}

function finish() {
  stop();
  const r = state.result;
  if (r.solution) {
    state.cells = parse(r.solution);
    render();
  }
  $('chip-status').textContent = r.status.toUpperCase();
  if (r.status === 'solved') {
    burst($('sboard'), { count: 120 });
    banner('SOLVED', `${NAMES[r.solver]} · ${r.nodes.toLocaleString()} guesses · ${r.backtracks.toLocaleString()} backtracks`, '#00ff88');
  } else {
    shake($('sboard'), 'big');
    banner(r.status === 'limit' ? 'GAVE UP' : 'NO SOLUTION',
           r.status === 'limit' ? `${r.nodes.toLocaleString()} guesses hit the limit` : 'this puzzle has no valid solution', '#ff3b3b');
  }
}

async function solveAndReplay() {
  stop();
  load(state.puzzle, $('puzzle-name').textContent);
  const solver = $('solver').value;
  $('chip-solver').textContent = NAMES[solver];
  $('chip-status').textContent = 'SOLVING';
  try {
    state.result = await postJSON('/api/sudoku/solve', { puzzle: state.puzzle, solver });
  } catch (err) {
    $('chip-status').textContent = 'ERROR';
    $('puzzle-name').textContent = err.message;
    return;
  }
  const r = state.result;
  $('chip-nodes').textContent = r.nodes.toLocaleString();
  $('chip-back').textContent = r.backtracks.toLocaleString();
  state.trace = r.trace;
  state.step = 0;
  $('chip-status').textContent = 'REPLAYING';
  $('btn-skip').disabled = false;
  // Long traces replay several steps per tick so the animation stays short.
  const tick = 25;
  state.timer = setInterval(() => {
    const perTick = Math.max(1, Math.round((Number($('speed').value) * tick) / 1000));
    let last = [-1, ''];
    for (let k = 0; k < perTick && state.step < state.trace.length; k++) last = applyStep(state.step++);
    render(last[0], last[1]);
    progress();
    if (state.step >= state.trace.length) finish();
  }, tick);
}

async function compare() {
  stop();
  const body = $('cmp-body');
  body.innerHTML = '';
  const rows = [];
  for (const solver of Object.keys(NAMES)) {
    try {
      rows.push([solver, await postJSON('/api/sudoku/solve', { puzzle: state.puzzle, solver })]);
    } catch (err) {
      rows.push([solver, { status: 'error', nodes: 0, backtracks: 0, seconds: 0 }]);
    }
    const max = Math.max(10, ...rows.map(([, r]) => r.nodes));
    body.innerHTML = rows.map(([s, r]) => `
      <tr><td>${NAMES[s]}</td><td class="status-${r.status}">${r.status}</td>
      <td class="num">${r.nodes.toLocaleString()}</td><td class="num">${r.backtracks.toLocaleString()}</td>
      <td><div class="cmp-bar" style="width:${((100 * Math.log10(r.nodes + 1)) / Math.log10(max + 1)).toFixed(1)}%"></div></td>
      <td class="num">${(r.seconds * 1000).toFixed(0)} ms</td></tr>`).join('');
  }
}

async function fetchPuzzle(set) {
  try {
    const p = await getJSON('/api/sudoku/puzzle', { set });
    load(p.puzzle, set === 'hard' ? `Hard: ${p.name.replace(/_/g, ' ')}`
      : `Generated puzzle #${p.index + 1}: ${p.clues} clues, ${p.guesses} guesses with propagation`);
  } catch (err) {
    $('puzzle-name').textContent = err.message;
  }
}

async function init() {
  $('btn-solve').onclick = solveAndReplay;
  $('btn-skip').onclick = finish;
  $('btn-reset').onclick = () => load(state.puzzle, $('puzzle-name').textContent);
  $('btn-compare').onclick = compare;
  $('btn-easy').onclick = () => fetchPuzzle('generated');
  $('btn-hard').onclick = () => fetchPuzzle('hard');
  $('speed').oninput = () => { $('speed-lbl').textContent = $('speed').value; };
  $('btn-load').onclick = () => {
    const cells = parse($('custom').value);
    $('custom-err').textContent = cells ? '' : 'enter exactly 81 digits';
    if (cells) load(cells.join(''), 'Custom puzzle');
  };
  try {
    state.meta = await getJSON('/api/sudoku/meta');
  } catch (err) {
    $('puzzle-name').textContent = `Server unreachable (${err.message})`;
    return;
  }
  for (const s of state.meta.solvers) {
    const opt = document.createElement('option');
    opt.value = s.name;
    opt.textContent = NAMES[s.name];
    $('solver').appendChild(opt);
  }
  $('solver').value = 'mrv_fc';
  $('solver').onchange = () => {
    $('solver').title = state.meta.solvers.find((s) => s.name === $('solver').value).description;
  };
  $('solver').onchange();
  await fetchPuzzle('hard');
}

init();
