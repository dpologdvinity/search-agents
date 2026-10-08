// Connect Four page: play against an agent or watch two agents, with each move's analysis.
//
// The position is the string of columns played (1-based), which is all the
// backend needs. Red always moves first.

import { getJSON, postJSON } from './api.js';
import { banner, burst, shake } from './fx.js';

const $ = (id) => document.getElementById(id);
const ROWS = 6, COLS = 7;

const state = {
  moves: '',          // e.g. "4453"
  humanColor: 'red',  // in "You vs AI" mode
  busy: false,
  over: false,
  watching: false,
  meta: null,
};

const NAMES = { alphazero: 'AlphaZero', minimax: 'Minimax', mcts: 'MCTS' };

// ── Rules (for rendering; the backend validates moves too) ──────────────

function grid(moves) {
  const g = Array.from({ length: COLS }, () => []);
  [...moves].forEach((ch, i) => g[Number(ch) - 1].push(i % 2 === 0 ? 'red' : 'yellow'));
  return g;
}

const toMove = (moves) => (moves.length % 2 === 0 ? 'red' : 'yellow');
const canPlay = (g, col) => g[col].length < ROWS;

function winningCells(g) {
  const at = (c, r) => (c >= 0 && c < COLS && r >= 0 && r < ROWS ? g[c][r] : undefined);
  for (let c = 0; c < COLS; c++)
    for (let r = 0; r < ROWS; r++) {
      const v = at(c, r);
      if (!v) continue;
      for (const [dc, dr] of [[1, 0], [0, 1], [1, 1], [1, -1]]) {
        const cells = [0, 1, 2, 3].map((k) => [c + k * dc, r + k * dr]);
        if (cells.every(([cc, rr]) => at(cc, rr) === v)) return cells;
      }
    }
  return null;
}

// ── Rendering ────────────────────────────────────────────────────────────

function render(dropCol = -1, suggest = -1) {
  const g = grid(state.moves);
  const win = winningCells(g);
  const winSet = new Set((win || []).map(([c, r]) => `${c},${r}`));
  const last = state.moves.length ? Number(state.moves[state.moves.length - 1]) - 1 : -1;
  const el = $('c4');
  el.innerHTML = '';
  for (let c = 0; c < COLS; c++) {
    const col = document.createElement('div');
    const clickable = !state.busy && !state.over && !state.watching && humanTurn() && canPlay(g, c);
    col.className = 'c4-col' + (clickable ? '' : ' disabled');
    if (c === suggest) col.style.background = 'rgba(0,255,136,0.12)';
    col.onclick = () => clickable && humanMove(c);
    for (let r = ROWS - 1; r >= 0; r--) {
      const cell = document.createElement('div');
      const v = g[c][r];
      cell.className = 'c4-cell' + (v ? ` ${v}` : '')
        + (winSet.has(`${c},${r}`) ? ' win' : '')
        + (c === last && r === g[c].length - 1 && !win ? ' last' : '')
        + (c === dropCol && r === g[c].length - 1 ? ' drop' : '');
      col.appendChild(cell);
    }
    el.appendChild(col);
  }
  $('chip-move').textContent = String(state.moves.length + 1);
  for (const id of ['btn-undo', 'btn-hint', 'btn-new']) $(id).disabled = state.busy && id !== 'btn-new';
  $('btn-undo').disabled = state.busy || state.watching || !state.moves.length;
  $('btn-hint').disabled = state.busy || state.over || state.watching || !humanTurn();
}

function log(text, cls = 'log-info') {
  const el = $('log');
  const line = document.createElement('div');
  line.className = cls;
  line.textContent = text;
  el.appendChild(line);
  el.scrollTop = el.scrollHeight;
}

function setStatus(text) { $('chip-status').textContent = text; }

// ── Analysis panel ──────────────────────────────────────────────────────

function legend(items) {
  $('an-legend').innerHTML = items.map(([color, text]) =>
    `<span class="field-hint"><span class="legend-dot" style="background:${color}"></span>${text}</span>`).join('');
}

function showAnalysis(result, who) {
  const a = result.analysis;
  const bars = $('an-bars');
  bars.innerHTML = '';
  $('an-value').classList.add('hidden');
  const name = NAMES[result.agent];
  let note = '';

  const column = (c, heights, label) => {
    const col = document.createElement('div');
    col.className = 'c4-bar-col' + (c === result.move ? ' chosen' : '');
    const pair = document.createElement('div');
    pair.className = 'c4-bar-pair';
    for (const [cls, h] of heights) {
      const bar = document.createElement('div');
      bar.className = `c4-bar ${cls}`;
      bar.style.height = `${Math.max(1, h * 110)}px`;
      if (cls === 'score') bar.style.background = h > 0.66 ? 'var(--green)' : h < 0.2 ? 'var(--red)' : 'var(--cyan)';
      pair.appendChild(bar);
    }
    const lbl = document.createElement('div');
    lbl.className = 'c4-bar-lbl';
    lbl.innerHTML = `${c + 1}<br>${label}`;
    col.append(pair, lbl);
    bars.appendChild(col);
  };

  if (a.kind === 'alphazero') {
    const total = a.visits.reduce((x, y) => x + y, 0) || 1;
    const maxShare = Math.max(...a.visits) / total;
    const maxPrior = Math.max(...a.prior);
    const scale = Math.max(maxShare, maxPrior);
    for (let c = 0; c < COLS; c++)
      column(c, [['prior', a.prior[c] / scale], ['visits', a.visits[c] / total / scale]],
        a.visits[c] ? `${Math.round((100 * a.visits[c]) / total)}%` : '');
    legend([['var(--purple)', 'network prior'], ['var(--cyan)', `share of ${a.simulations} search visits`]]);
    $('an-value').classList.remove('hidden');
    $('an-value-lbl').textContent = `${who.toUpperCase()} WIN PROBABILITY (ALPHAZERO'S ESTIMATE): ${Math.round(100 * a.win_probability)}%`;
    $('an-value-bar').style.width = `${100 * a.win_probability}%`;
    note = 'The prior is the network\'s first instinct; search spends visits where the prior and the backed-up values agree.';
  } else if (a.kind === 'minimax') {
    const labelOf = (s) => !s ? '' : s.result === 'win' ? `WIN ${Math.ceil(s.plies / 2)}`
      : s.result === 'loss' ? 'LOSS' : String(s.score);
    const height = (s) => !s ? 0 : s.result === 'win' ? 1 : s.result === 'loss' ? 0.08
      : 0.5 + Math.max(-0.4, Math.min(0.4, s.score / 400));
    for (let c = 0; c < COLS; c++) {
      const s = a.scores ? a.scores[c] : null;
      column(c, [['score', height(s)]], labelOf(s));
    }
    legend([['var(--cyan)', `searched ${a.depth} plies ahead, ${a.nodes.toLocaleString()} positions`]]);
    note = a.best && a.best.result === 'win' ? `${name} has proven a forced win.`
      : a.best && a.best.result === 'loss' ? `${name} has proven it loses against perfect play.`
      : 'Scores count open lines of four at the search horizon. Only the chosen column\'s score is exact; others are upper bounds.';
  } else {
    const total = a.visits.reduce((x, y) => x + y, 0) || 1;
    const max = Math.max(...a.visits) / total;
    for (let c = 0; c < COLS; c++)
      column(c, [['visits', a.visits[c] / total / max]], a.visits[c] ? `${Math.round((100 * a.visits[c]) / total)}%` : '');
    legend([['var(--cyan)', `share of ${a.simulations} random-rollout simulations`]]);
    note = 'Plain MCTS learns nothing: it estimates each column by playing random games to the end.';
  }
  $('an-title').textContent = `${name} → column ${result.move + 1} (${result.seconds.toFixed(2)} s)`;
  $('an-note').textContent = note;
}

// ── Game flow ────────────────────────────────────────────────────────────

function humanTurn() {
  return $('mode').value === 'human' && toMove(state.moves) === state.humanColor;
}

function afterMove(col, who) {
  render(col);
  shake($('c4'));
  const g = grid(state.moves);
  if (winningCells(g)) {
    state.over = true;
    const human = who === 'you';
    const lost = !human && $('mode').value === 'human';
    shake($('c4'), 'big');
    burst($('c4'), { count: 140, colors: lost ? ['#ff3b3b', '#ff00a0', '#ffe600'] : undefined });
    banner(human ? 'YOU WIN' : lost ? 'AI WINS' : `${who.toUpperCase()} WINS`,
           `four in a row after ${state.moves.length} moves`, lost ? '#ff3b3b' : '#00ff88');
    setStatus(`${who.toUpperCase()} WINS`);
    log(`★ ${who} wins in ${state.moves.length} moves.`, 'log-best');
  } else if (state.moves.length === ROWS * COLS) {
    state.over = true;
    setStatus('DRAW');
    log('= Draw: the board is full.', 'log-val');
  }
  render();
}

function humanMove(col) {
  state.moves += String(col + 1);
  log(`You → column ${col + 1}`, 'log-move');
  afterMove(col, 'you');
  if (!state.over) aiMove($('agent-a').value, Number($('level-a').value), 'AI');
}

async function aiMove(agent, level, who) {
  state.busy = true;
  setStatus(`${NAMES[agent].toUpperCase()} THINKING`);
  render();
  try {
    const result = await postJSON('/api/connect4/move', { moves: state.moves, agent, level });
    showAnalysis(result, who === 'AI' ? 'AI' : who);
    state.moves += String(result.move + 1);
    log(`${who} (${NAMES[agent]}) → column ${result.move + 1}`, 'log-adv');
    state.busy = false;
    afterMove(result.move, who);
    if (!state.over) setStatus(state.watching ? 'PLAYING' : 'YOUR TURN');
    return true;
  } catch (err) {
    state.busy = false;
    log(`✗ ${err.message}`, 'log-err');
    setStatus('ERROR');
    render();
    return false;
  }
}

async function watch() {
  newGame();
  state.watching = true;
  render();
  const players = { red: [$('agent-a').value, Number($('level-a').value)],
                    yellow: [$('agent-b').value, Number($('level-b').value)] };
  log(`▶ ${NAMES[players.red[0]]} (red) vs ${NAMES[players.yellow[0]]} (yellow)`, 'log-move');
  while (state.watching && !state.over) {
    const color = toMove(state.moves);
    const [agent, level] = players[color];
    const ok = await aiMove(agent, level, color === 'red' ? 'Red' : 'Yellow');
    if (!ok) break;
    await new Promise((r) => setTimeout(r, 350));
  }
  state.watching = false;
  render();
}

async function hint() {
  state.busy = true;
  render();
  const agent = state.meta.agents.find((a) => a.name === 'alphazero' && a.available) ? 'alphazero' : 'minimax';
  try {
    const result = await postJSON('/api/connect4/move', { moves: state.moves, agent, level: 2 });
    showAnalysis(result, 'your');
    log(`✦ Hint (${NAMES[agent]}): column ${result.move + 1}`, 'log-best');
    state.busy = false;
    render(-1, result.move);
  } catch (err) {
    state.busy = false;
    log(`✗ ${err.message}`, 'log-err');
    render();
  }
}

function undo() {
  if (!state.moves.length) return;
  state.over = false;
  do {
    state.moves = state.moves.slice(0, -1);
  } while (state.moves.length && !humanTurn());
  setStatus('YOUR TURN');
  log('◀ Undo', 'log-info');
  render();
  if (!humanTurn()) aiMove($('agent-a').value, Number($('level-a').value), 'AI');
}

function newGame() {
  state.watching = false;
  state.moves = '';
  state.over = false;
  $('log').innerHTML = '';
  const mode = $('mode').value;
  state.humanColor = $('first').value === 'human' ? 'red' : 'yellow';
  $('chip-you').textContent = mode === 'human' ? state.humanColor.toUpperCase() : '—';
  $('chip-ai').textContent = mode === 'human' ? NAMES[$('agent-a').value] : 'AI vs AI';
  setStatus(mode === 'human' ? 'YOUR TURN' : 'READY');
  render();
  if (mode === 'human' && !humanTurn()) aiMove($('agent-a').value, Number($('level-a').value), 'AI');
}

function updateSetup() {
  const ai = $('mode').value === 'ai';
  $('first-wrap').classList.toggle('hidden', ai);
  $('agent-b-wrap').classList.toggle('hidden', !ai);
  $('agent-a-title').textContent = ai ? 'FIRST AI (RED)' : 'OPPONENT';
  const info = state.meta?.agents.find((a) => a.name === $('agent-a').value);
  $('desc-a').textContent = info ? info.description : '';
}

async function init() {
  $('btn-new').onclick = newGame;
  $('btn-undo').onclick = undo;
  $('btn-hint').onclick = hint;
  $('btn-watch').onclick = watch;
  $('mode').onchange = () => { updateSetup(); newGame(); };
  $('first').onchange = newGame;
  $('agent-a').onchange = () => { updateSetup(); if ($('mode').value === 'human') newGame(); };
  render();
  try {
    state.meta = await getJSON('/api/connect4/meta');
  } catch (err) {
    log(`✗ Server unreachable (${err.message}).`, 'log-err');
    return;
  }
  for (const id of ['agent-a', 'agent-b']) {
    const sel = $(id);
    for (const a of state.meta.agents) {
      const opt = document.createElement('option');
      opt.value = a.name;
      opt.textContent = NAMES[a.name] + (a.available ? '' : ' (not trained yet)');
      opt.disabled = !a.available;
      sel.appendChild(opt);
    }
  }
  const az = state.meta.agents.find((a) => a.name === 'alphazero');
  $('agent-a').value = az && az.available ? 'alphazero' : 'minimax';
  $('agent-b').value = 'minimax';
  updateSetup();
  newGame();
}

init();
