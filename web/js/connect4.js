// Connect Four page: play against an agent or watch two agents, with each move's analysis.
//
// The position is the string of columns played (1-based), which is all the
// backend needs. Red always moves first.

import { getJSON, postJSON } from './api.js';
import { banner, burst, shake } from './fx.js';

const $ = (id) => document.getElementById(id);
const ROWS = 6, COLS = 7;
const RESTART_MS = 1500;  // pause on a finished game before the next one starts in auto-restart

const state = {
  moves: '',          // e.g. "4453"
  humanColor: 'red',  // in "You vs AI" mode
  busy: false,
  over: false,
  watching: false,    // a watch match is running (drives the START/STOP button)
  paused: false,      // watch match paused between moves
  stepRequest: false, // STEP was pressed while paused: let exactly one move through
  run: 0,             // bumped to end the current match loop (stop, new game, new match)
  gen: 0,             // bumped by NEW GAME so a search that was already in flight is dropped
  sideA: 'red',       // the colour AI A plays in the current match
  score: { w: 0, d: 0, l: 0, games: 0 },  // AI A's record across the watch matches
  meta: null,
};

const NAMES = { alphazero: 'AlphaZero', minimax: 'Minimax', mcts: 'MCTS', chance: 'Chance' };
// The algorithm behind each agent, shown beside its name so the page says what each side is doing.
const ALGO = { alphazero: 'PUCT + network', minimax: 'alpha-beta', mcts: 'random rollouts', chance: 'fixed odds' };
const LABEL = (agent) => `${NAMES[agent]} (${ALGO[agent]})`;
// One-line tooltip for each agent menu. The full method is in HOW IT WORKS.
const AGENT_TIPS = {
  alphazero: 'Learned from self-play: a policy-value network proposes columns, and PUCT search refines them.',
  minimax: 'Looks ahead as far as its time budget allows and scores the leaves by counting open lines of four.',
  mcts: 'Estimates each column by playing random games to the end. It learns nothing.',
  chance: 'Draws a column from fixed odds (6% to 25%), with no search.',
};

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

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

// ── Rendering ────────────────────────────────────────────────────────────

function render(dropCol = -1, suggest = -1) {
  const g = grid(state.moves);
  const win = winningCells(g);
  const winSet = new Set((win || []).map(([c, r]) => `${c},${r}`));
  const last = state.moves.length ? Number(state.moves[state.moves.length - 1]) - 1 : -1;
  const el = $('c4');
  // The board is rebuilt on every render, so remember which column had focus and give the focus back.
  const focused = [...el.children].indexOf(document.activeElement);
  el.innerHTML = '';
  for (let c = 0; c < COLS; c++) {
    const col = document.createElement('div');
    const clickable = !state.busy && !state.over && !state.watching && humanTurn() && canPlay(g, c);
    col.className = 'c4-col' + (clickable ? '' : ' disabled');
    if (c === suggest) col.style.background = 'rgba(0,255,136,0.12)';
    col.onclick = () => clickable && humanMove(c);
    // Keyboard: Left and Right move between columns, Enter or Space drops a disc in the focused one.
    col.tabIndex = clickable ? 0 : -1;
    col.setAttribute('role', 'button');
    col.setAttribute('aria-label', `Column ${c + 1}`);
    col.setAttribute('aria-disabled', String(!clickable));
    col.onkeydown = (e) => {
      if (e.key === 'ArrowLeft' || e.key === 'ArrowRight') {
        e.preventDefault();
        el.children[c + (e.key === 'ArrowRight' ? 1 : -1)]?.focus();
      } else if ((e.key === 'Enter' || e.key === ' ') && clickable) {
        e.preventDefault();
        humanMove(c);
      }
    };
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
  if (focused >= 0) el.children[focused].focus();  // also while the AI moves, when the column is not clickable
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
  if (cls === 'log-err') $('log-fold').open = true;  // errors are shown even when the log is folded
  el.appendChild(line);
  el.scrollTop = el.scrollHeight;
}

// The chip holds two spans, the full text and a short one; app.css shows one of them by screen width.
function setStatus(full, short = full) {
  const wide = document.createElement('span');
  wide.className = 'status-wide';
  wide.textContent = full;
  const narrow = document.createElement('span');
  narrow.className = 'status-narrow';
  narrow.textContent = short;
  $('chip-status').replaceChildren(wide, narrow);
}

// Which watch player sits on each colour. AI A takes the colour picked in #side-a; AI B takes the other.
function watchPlayers() {
  const sideA = $('side-a').value;
  const sideB = sideA === 'red' ? 'yellow' : 'red';
  return {
    [sideA]: [$('agent-a').value, Number($('level-a').value)],
    [sideB]: [$('agent-b').value, Number($('level-b').value)],
  };
}

// Who is playing, naming the algorithm on each side, e.g. "You vs Chance (fixed odds)"
// or "Red: AlphaZero (PUCT + network) vs Yellow: Chance (fixed odds)". With short set, only the names,
// e.g. "You vs Chance" or "AlphaZero vs Chance" (Red first), for a phone-width chip.
function matchup(short = false) {
  if ($('mode').value === 'human') {
    const agent = $('agent-a').value;
    return short ? `You vs ${NAMES[agent]}` : `You vs ${LABEL(agent)}`;
  }
  const players = watchPlayers();
  const [red, yellow] = [players.red[0], players.yellow[0]];
  return short ? `${NAMES[red]} vs ${NAMES[yellow]}` : `Red: ${LABEL(red)} vs Yellow: ${LABEL(yellow)}`;
}

// Status chip: the matchup followed by the turn state, e.g. "You vs Chance (fixed odds) · YOUR TURN".
// The short turn text is used with the short matchup on phones.
function showStatus(turn, shortTurn = turn) {
  setStatus(`${matchup()} · ${turn}`, `${matchup(true)} · ${shortTurn}`);
}

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
  } else if (a.kind === 'chance') {
    // Fixed odds: there is no search, so each bar is just the odds that column was drawn with.
    // Full columns have odds 0 and get no label.
    const max = Math.max(...a.odds) || 1;
    for (let c = 0; c < COLS; c++)
      column(c, [['visits', a.odds[c] / max]], a.odds[c] ? `${Math.round(100 * a.odds[c])}%` : '');
    legend([['var(--cyan)', 'fixed odds for this column, renormalised over open columns']]);
    note = 'Chance does no search, evaluation or lookahead: the column was drawn from fixed odds.';
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

// Called after every move. `who` is 'you', 'AI' (human mode) or 'Red'/'Yellow' (watch mode);
// `agent` is the agent that just moved, when there is one.
function afterMove(col, who, agent) {
  render(col);
  shake($('c4'));
  const g = grid(state.moves);
  if (winningCells(g)) {
    state.over = true;
    const human = who === 'you';
    const lost = !human && $('mode').value === 'human';
    const title = human ? 'YOU WIN' : lost ? 'AI WINS' : `${NAMES[agent].toUpperCase()} WINS`;
    shake($('c4'), 'big');
    burst($('c4'), { count: 140, colors: lost ? ['#ff3b3b', '#ff00a0', '#ffe600'] : undefined });
    banner(title, `four in a row after ${state.moves.length} moves`, lost ? '#ff3b3b' : '#00ff88');
    showStatus(title);
    log(`★ ${human ? 'You' : NAMES[agent]} wins in ${state.moves.length} moves.`, 'log-best');
  } else if (state.moves.length === ROWS * COLS) {
    state.over = true;
    showStatus('DRAW');
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
  const gen = state.gen;  // a NEW GAME while this search runs makes its result stale
  state.busy = true;
  showStatus(`${NAMES[agent].toUpperCase()} THINKING`, 'THINKING');
  render();
  try {
    const result = await postJSON('/api/connect4/move', { moves: state.moves, agent, level });
    if (gen !== state.gen) return false;
    showAnalysis(result, who);
    state.moves += String(result.move + 1);
    log(`${who} (${NAMES[agent]}) → column ${result.move + 1}`, 'log-adv');
    state.busy = false;
    afterMove(result.move, who, agent);
    if (!state.over) showStatus(state.watching ? 'PLAYING' : 'YOUR TURN');
    return true;
  } catch (err) {
    if (gen !== state.gen) return false;
    state.busy = false;
    log(`✗ ${err.message}`, 'log-err');
    showStatus('ERROR');
    render();
    return false;
  }
}

// Holds the watch loop while paused. STEP releases exactly one move, then the loop pauses again.
async function pauseGate(run) {
  while (state.paused && state.run === run && !state.stepRequest) await sleep(80);
  state.stepRequest = false;
}

// Play one game of the match from an empty board. Returns 'red', 'yellow' or 'draw' when the game
// ends, or null if the match was stopped or a request failed.
async function playOneGame(players, run) {
  startBoard();
  log(`▶ Game ${state.score.games + 1}`, 'log-move');
  while (state.run === run && !state.over) {
    await pauseGate(run);
    if (state.run !== run) return null;
    const color = toMove(state.moves);
    const [agent, level] = players[color];
    const ok = await aiMove(agent, level, color === 'red' ? 'Red' : 'Yellow');
    if (!ok) return null;
    if (!state.over) await sleep(Number($('speed').value));
  }
  if (!state.over) return null;
  if (!winningCells(grid(state.moves))) return 'draw';
  // Red moves first, so an odd number of moves means red made the last (winning) move.
  return state.moves.length % 2 === 1 ? 'red' : 'yellow';
}

// Add one finished game to AI A's W/D/L record.
function recordResult(winner) {
  const s = state.score;
  s.games++;
  if (winner === 'draw') s.d++;
  else if (winner === state.sideA) s.w++;
  else s.l++;
  renderScore();
}

function renderScore() {
  const s = state.score;
  $('score').textContent = `AI A: W ${s.w} · D ${s.d} · L ${s.l} (${s.games} games)`;
}

function resetScore() {
  state.score = { w: 0, d: 0, l: 0, games: 0 };
  renderScore();
}

// Start a match: play games back to back, restarting after each one when AUTO-RESTART is on.
async function watch() {
  if (state.watching) { stopMatch(); return; }
  const run = ++state.run;
  state.watching = true;
  state.paused = false;
  state.stepRequest = false;
  state.sideA = $('side-a').value;
  const players = watchPlayers();
  setWatchButtons();
  $('log').innerHTML = '';  // a fresh match gets a fresh log; auto-restarts keep appending to it
  log(`▶ Match: Red ${LABEL(players.red[0])} vs Yellow ${LABEL(players.yellow[0])}`, 'log-move');
  while (state.run === run) {
    const winner = await playOneGame(players, run);
    if (state.run !== run || winner === null) break;
    recordResult(winner);
    if (!$('auto').checked) break;
    await sleep(RESTART_MS);  // hold the finished board so the result can be read
  }
  if (state.run === run) state.watching = false;
  setWatchButtons();
  render();
}

// End the match after the current move. The loop notices the new run number and exits.
function stopMatch() {
  state.run++;
  state.watching = false;
  state.paused = false;
  state.stepRequest = false;
  log('■ Match stopped.', 'log-info');
  setWatchButtons();
  render();
}

// Keep the watch controls in step with the match: START/STOP label, pause and step availability,
// and whether the agent pickers can be changed (not while a match runs).
function setWatchButtons() {
  const running = state.watching;
  $('btn-watch').querySelector('.btn-txt').textContent = running ? '■ STOP MATCH' : '▶ START MATCH';
  $('btn-pause').querySelector('.btn-txt').textContent = state.paused ? '▶ RESUME' : '❚❚ PAUSE';
  $('btn-pause').disabled = !running;
  $('btn-step').disabled = !(running && state.paused);
  for (const id of ['side-a', 'agent-a', 'level-a', 'agent-b', 'level-b']) $(id).disabled = running;
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
  showStatus('YOUR TURN');
  log('◀ Undo', 'log-info');
  render();
  if (!humanTurn()) aiMove($('agent-a').value, Number($('level-a').value), 'AI');
}

// Empty the board for a new game. Used by NEW GAME and by each game of a watch match.
function startBoard() {
  state.moves = '';
  state.over = false;
  $('chip-you').textContent = $('mode').value === 'human' ? state.humanColor.toUpperCase() : '—';
  render();
}

function newGame() {
  state.run++;  // ends any running watch match
  state.gen++;  // drops any search still in flight
  state.busy = false;
  state.watching = false;
  state.paused = false;
  state.stepRequest = false;
  $('log').innerHTML = '';
  const mode = $('mode').value;
  state.humanColor = $('first').value === 'human' ? 'red' : 'yellow';
  $('chip-you').textContent = mode === 'human' ? state.humanColor.toUpperCase() : '—';
  startBoard();
  setWatchButtons();
  showStatus(mode === 'human' ? 'YOUR TURN' : 'READY');
  if (mode === 'human' && !humanTurn()) aiMove($('agent-a').value, Number($('level-a').value), 'AI');
}

function updateSetup() {
  const ai = $('mode').value === 'ai';
  $('first-wrap').classList.toggle('hidden', ai);
  $('side-wrap').classList.toggle('hidden', !ai);
  $('pace-wrap').classList.toggle('hidden', !ai);
  $('agent-b-wrap').classList.toggle('hidden', !ai);
  $('agent-a-title').textContent = ai ? 'AI A' : 'OPPONENT';
  $('agent-a').title = AGENT_TIPS[$('agent-a').value] ?? '';
  $('agent-b').title = AGENT_TIPS[$('agent-b').value] ?? '';
}

async function init() {
  $('btn-new').onclick = newGame;
  $('btn-undo').onclick = undo;
  $('btn-hint').onclick = hint;
  $('btn-watch').onclick = watch;
  $('btn-pause').onclick = () => { state.paused = !state.paused; setWatchButtons(); };
  $('btn-step').onclick = () => { state.stepRequest = true; };
  $('btn-reset-score').onclick = resetScore;
  $('mode').onchange = () => { updateSetup(); newGame(); };
  $('first').onchange = newGame;
  // Changing a pick restarts from an empty board so the status line names the new matchup.
  // Watch selectors are disabled while a match runs, so this cannot interrupt one.
  $('agent-a').onchange = () => { updateSetup(); newGame(); };
  $('agent-b').onchange = () => { updateSetup(); newGame(); };
  $('side-a').onchange = newGame;
  renderScore();
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
      opt.textContent = LABEL(a.name) + (a.available ? '' : ' (not trained yet)');
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
