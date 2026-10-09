// Checkers page. The server owns the rules and the agents: it returns the legal moves (with the board after each)
// and an agent's move with its search statistics. This page keeps the game state, draws the n x n board, and runs
// either a game against the player or a watched game between two agents.

import { getJSON, postJSON } from './api.js';
import { banner, burst, pop, shake } from './fx.js';

const $ = (id) => document.getElementById(id);
const RED = 1, WHITE = -1;
const DRAW_PLIES = 80; // 40 moves each without a capture or a man moving; checkers/__main__.py uses the same limit

const AGENT_LABELS = {
  alphabeta: 'Alpha-beta', minimax: 'Minimax', mcts: 'MCTS', greedy: 'Greedy', random: 'Random', chance: 'Chance',
};
// Longer names for the OPPONENT and AI vs AI menus, so each option says what the agent does.
const AGENT_OPTIONS = {
  alphabeta: 'Alpha-beta search (iterative deepening)',
  minimax: 'Minimax (fixed depth, no pruning)',
  mcts: 'Monte Carlo tree search (UCT)',
  greedy: 'Greedy (one move ahead)',
  random: 'Random (uniform)',
  chance: 'Chance (fixed odds, no search)',
};

const state = {
  meta: null,
  n: 8,              // board size: 8, 10 or 12
  forced: true,      // forced captures (standard rules) or optional
  mode: 'human',     // 'human' (you vs AI) or 'watch' (AI vs AI)
  board: [], turn: RED, human: RED, legal: [], selected: null, history: [],
  quiet: 0, ply: 0, busy: false, over: false, hint: null, last: null,
  // Watch mode. gameId changes on every new game, so a move that arrives late is dropped.
  gameId: 0, running: false, paused: false, loopActive: false,
  score: { red: 0, white: 0, draw: 0, games: 0 },
};

// Squares are numbered row by row from the top, half of them on each row (checkers/board.py).
const sqAt = (r, c) => ((r + c) % 2 === 1 ? r * (state.n / 2) + Math.floor(c / 2) : -1);
const count = (side) => state.board.filter((p) => p * side > 0).length;
const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
const sizeInfo = () => state.meta.sizes.find((s) => s.size === state.n);

// ── Keyboard focus ───────────────────────────────────────────────────────
// Rendering rebuilds the board and drops focus. A key press therefore records where focus should land, and
// focusPending() puts it there after the next render. Mouse actions never set it, so mouse play is unchanged.
let focusNext = null; // {sq, role: 'piece' | 'target'}

function focusPending() {
  if (!focusNext) return;
  const { sq, role } = focusNext;
  focusNext = null;
  const cell = document.querySelector(`#ck [data-sq="${sq}"]`);
  (role === 'piece' ? cell?.querySelector('.ck-piece') : cell)?.focus();
}

// ── Rendering ────────────────────────────────────────────────────────────

function render() {
  const el = $('ck');
  el.innerHTML = '';
  el.style.setProperty('--n', state.n);
  const flip = state.mode === 'human' && state.human !== RED; // white sees its own men at the bottom
  const targets = new Set(state.selected === null ? [] :
    state.legal.filter((m) => m.path[0] === state.selected).map((m) => m.path[m.path.length - 1]));
  const movable = new Set(state.legal.map((m) => m.path[0]));
  const yourTurn = state.mode === 'human' && !state.over && !state.busy && state.turn === state.human;
  for (let vr = 0; vr < state.n; vr++) {
    for (let vc = 0; vc < state.n; vc++) {
      const r = flip ? state.n - 1 - vr : vr, c = flip ? state.n - 1 - vc : vc, sq = sqAt(r, c);
      const cell = document.createElement('div');
      cell.className = 'ck-cell' + (sq < 0 ? ' light' : ' dark');
      if (sq >= 0) {
        cell.dataset.sq = String(sq);
        if (state.last && state.last.path.includes(sq)) cell.classList.add('trail');
        if (state.hint && state.hint.path.includes(sq)) cell.classList.add('hint');
        if (targets.has(sq)) {
          cell.classList.add('target');
          cell.onclick = () => moveTo(sq);
          // Keyboard: a target square is a stop in the tab order, named by its board number.
          cell.tabIndex = 0;
          cell.setAttribute('role', 'button');
          cell.setAttribute('aria-label', `move to ${sq + 1}`);
          cell.onkeydown = (e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); moveTo(sq, true); } };
        }
        const p = state.board[sq];
        if (p) {
          const piece = document.createElement('div');
          piece.className = `ck-piece ${p > 0 ? 'red' : 'white'}${Math.abs(p) === 2 ? ' king' : ''}`;
          if (yourTurn && movable.has(sq)) {
            piece.classList.add('movable');
            piece.tabIndex = 0;
            piece.onclick = () => select(sq);
            piece.setAttribute('role', 'button');
            piece.setAttribute('aria-label', `piece on ${sq + 1}`);
            piece.onkeydown = (e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); select(sq, true); } };
          }
          if (sq === state.selected) piece.classList.add('selected');
          cell.appendChild(piece);
        }
      }
      el.appendChild(cell);
    }
  }
  $('chip-pieces').textContent = `${count(RED)} – ${count(WHITE)}`;
  $('chip-move').textContent = String(Math.floor(state.ply / 2) + 1);
  $('chip-you').textContent = state.mode === 'watch' ? 'WATCH' : state.human === RED ? 'RED' : 'WHITE';
  $('ck-hint').textContent = state.mode === 'watch'
    ? 'Two agents are playing. Pause to step through the game one move at a time.'
    : state.forced ? 'Captures are mandatory.' : 'Captures are optional.';
  $('btn-undo').disabled = state.busy || !state.history.length;
  $('btn-hint').disabled = state.busy || state.over || state.turn !== state.human;
  $('btn-pause').textContent = state.paused ? '▶ RESUME' : '❚❚ PAUSE';
  $('btn-step').disabled = !state.paused || state.busy || state.over;
  focusPending();
}

function log(text, cls = 'log-info') {
  const line = document.createElement('div');
  line.className = cls;
  line.textContent = text;
  if (cls === 'log-err') $('log-fold').open = true;  // errors are shown even when the log is folded
  $('log').appendChild(line);
  $('log').scrollTop = $('log').scrollHeight;
}

const setStatus = (t) => { $('chip-status').textContent = t; };
// While paused, the status stays PAUSED, even during a single step, until RESUME.
const watchStatus = (t) => setStatus(state.paused ? 'PAUSED' : t);

// The line that names each side's algorithm and budget. The server builds the text, so the page and the CLI
// always print the same words. A request number guards against answers that arrive out of order.
let matchupSeq = 0;
async function updateMatchup() {
  const seq = ++matchupSeq;
  const watch = state.mode === 'watch';
  const human = Number($('side').value);
  const red = watch ? $('red-agent').value : human === RED ? 'human' : $('agent').value;
  const white = watch ? $('white-agent').value : human === WHITE ? 'human' : $('agent').value;
  const params = {
    red, red_level: watch ? $('red-level').value : $('level').value,
    white, white_level: watch ? $('white-level').value : $('level').value,
    size: state.n, forced: state.forced,
  };
  try {
    const { text } = await getJSON('/api/checkers/describe', params);
    if (seq !== matchupSeq) return;
    $('algo-line').textContent = text;
  } catch (err) {
    log(`✗ ${err.message}`, 'log-err');
  }
}

function renderScore() {
  $('score-red').textContent = state.score.red;
  $('score-white').textContent = state.score.white;
  $('score-draw').textContent = state.score.draw;
  $('score-games').textContent = state.score.games;
}

// The bar height for a root move. Win and loss get fixed heights so they sit above and below any evaluation.
function scoreValue(s) {
  if (!s) return null;
  if (s.result === 'rate') return s.percent;
  if (s.result === 'win') return 400;
  if (s.result === 'loss') return -400;
  return s.score;
}

function scoreText(s) {
  if (!s) return 'not scored';
  if (s.result === 'rate') return `${s.percent}% to win`;
  if (s.result === 'win') return `wins in ${Math.ceil(s.plies / 2)}`;
  if (s.result === 'loss') return `loses in ${Math.ceil(s.plies / 2)}`;
  return (s.score > 0 ? '+' : '') + s.score;
}

function showAnalysis(res, who) {
  const a = res.analysis;
  $('prune').hidden = false;
  $('pm-nodes-lbl').textContent = a.agent === 'mcts' ? 'Tree nodes' : 'Positions searched';
  $('pm-nodes').textContent = a.nodes.toLocaleString();
  $('pm-cuts-lbl').textContent = a.agent === 'mcts' ? 'Simulated games' : 'Branches cut by alpha-beta';
  $('pm-cuts').textContent = a.agent === 'alphabeta' ? a.cutoffs.toLocaleString()
    : a.agent === 'mcts' ? a.rollouts.toLocaleString() : 'none';
  $('pm-depth').textContent = a.depth ? `${a.depth} half-moves` : '–';
  $('an-title').textContent = `${who} played ${res.move.notation} after ${a.seconds.toFixed(2)} s`;
  const vals = a.root.map((r) => scoreValue(r.score));
  const known = vals.filter((v) => v !== null);
  const lo = known.length ? Math.min(...known) : 0, hi = known.length ? Math.max(...known) : 0;
  const list = $('an-root');
  list.innerHTML = '';
  a.root.slice(0, 7).forEach((r, k) => {
    const li = document.createElement('li');
    if (r.notation === res.move.notation) li.className = 'chosen';
    const v = vals[k];
    const w = v === null ? 0 : hi === lo ? 100 : 12 + (88 * (v - lo)) / (hi - lo);
    li.innerHTML = `<span class="mv"></span><span class="bar"><i style="width:${w}%"></i></span><span class="sc"></span>`;
    li.querySelector('.mv').textContent = r.notation;
    li.querySelector('.sc').textContent = scoreText(r.score);
    list.appendChild(li);
  });
  const squares = `Squares are numbered 1–${(state.n * state.n) / 2} from the top.`;
  $('an-note').textContent = a.agent === 'mcts'
    ? `Win rates: the mover's average rollout value after each move. A rollout cut off at 40 plies counts as the evaluation's chance of winning. ${squares}`
    : a.agent === 'random' ? `A random choice, so there are no scores. ${squares}`
    : a.agent === 'greedy' ? `Scores look one move ahead, from the mover's side. ${squares}`
    : `Scores are from the mover's side, assuming the best reply each time. ${squares}`;
}

// ── Game flow ────────────────────────────────────────────────────────────

function select(sq, byKey = false) {
  state.selected = state.selected === sq ? null : sq;
  state.hint = null;
  if (byKey) {
    // Keyboard: focus moves to the first square the piece can reach, so Enter plays it. With no
    // square to reach, or when the piece is deselected, focus stays on the piece.
    const targets = state.selected === null ? [] : state.legal.filter((m) => m.path[0] === sq).map((m) => m.path[m.path.length - 1]);
    focusNext = targets.length ? { sq: Math.min(...targets), role: 'target' } : { sq, role: 'piece' };
  }
  render();
}

// Escape: drop the selection and put focus back on its piece.
function cancelSelection() {
  const sq = state.selected;
  state.selected = null;
  state.hint = null;
  focusNext = { sq, role: 'piece' };
  render();
}

function apply(move, who) {
  const human = state.mode === 'human';
  if (human) state.history.push({ board: state.board, turn: state.turn, quiet: state.quiet, ply: state.ply, legal: state.legal });
  const from = state.board[move.path[0]];
  state.board = move.result;
  state.last = move;
  state.selected = null;
  state.hint = null;
  state.ply++;
  state.quiet = move.captured.length || Math.abs(from) === 1 ? 0 : state.quiet + 1;
  state.turn = -state.turn;
  render();
  const board = $('ck');
  if (move.captured.length) {
    if (human) shake(board, move.captured.length > 1 ? 'big' : 'small');
    pop(board, move.captured.length > 1 ? `${move.captured.length}x JUMP` : 'CAPTURE', '#ff00a0');
  }
  const crowned = Math.abs(from) === 1 && Math.abs(move.result[move.path[move.path.length - 1]]) === 2;
  if (crowned) {
    if (human) burst(board, { count: 70, colors: ['#ffe600', '#ff00a0', '#00f5ff'] });
    pop(board, 'KINGED', '#ffe600');
  }
  log(`${who} ${move.notation}${crowned ? ' (crowned)' : ''}`, who === 'You' ? 'log-move' : 'log-adv');
}

function finish(text, sub, color) {
  state.over = true;
  setStatus(text);
  render();
  burst($('ck'), { count: 150, colors: color === '#ff3b3b' ? ['#ff3b3b', '#ff00a0'] : undefined });
  shake($('ck'), 'big');
  banner(text, sub, color);
  log(`★ ${text}: ${sub}`, 'log-best');
}

function checkDraw() {
  if (state.quiet >= DRAW_PLIES) { finish('DRAW', '40 moves each without a capture or a man moving', '#00f5ff'); return true; }
  return false;
}

async function moveTo(sq, byKey = false) {
  const move = state.legal.find((m) => m.path[0] === state.selected && m.path[m.path.length - 1] === sq);
  if (!move) return;
  apply(move, 'You');
  if (!checkDraw()) await agentTurn(byKey);
}

async function agentTurn(byKey = false) {
  const gid = state.gameId;
  state.busy = true;
  setStatus('THINKING');
  render();
  try {
    const res = await postJSON('/api/checkers/move', {
      board: state.board, turn: state.turn, size: state.n, forced: state.forced,
      agent: $('agent').value, level: Number($('level').value),
    });
    if (gid !== state.gameId) return;
    state.busy = false;
    apply(res.move, 'AI');
    showAnalysis(res, 'The agent');
    state.legal = res.reply;
    if (!state.legal.length) finish('AI WINS', 'you have no legal moves left', '#ff3b3b');
    else if (!checkDraw()) {
      setStatus('YOUR TURN');
      // After a keyboard move, focus returns to a piece you can move, so the next move needs no mouse.
      if (byKey) focusNext = { sq: state.legal[0].path[0], role: 'piece' };
      render();
    }
  } catch (err) {
    if (gid !== state.gameId) return;
    state.busy = false;
    if (/game is over/.test(err.message)) finish('YOU WIN', 'the agent has no legal moves left', '#00ff88');
    else { log(`✗ ${err.message}`, 'log-err'); setStatus('ERROR'); render(); }
  }
}

async function hint() {
  const gid = state.gameId;
  state.busy = true;
  render();
  try {
    const res = await postJSON('/api/checkers/move', {
      board: state.board, turn: state.turn, size: state.n, forced: state.forced, agent: 'alphabeta', level: 2,
    });
    if (gid !== state.gameId) return;
    state.hint = res.move;
    showAnalysis(res, 'Hint: the agent would play');
    log(`✦ Hint: ${res.move.notation}`, 'log-best');
  } catch (err) {
    log(`✗ ${err.message}`, 'log-err');
  }
  state.busy = false;
  render();
}

function undo() {
  // Go back to the player's previous turn.
  while (state.history.length) {
    const prev = state.history.pop();
    Object.assign(state, prev);
    if (state.turn === state.human) break;
  }
  Object.assign(state, { over: false, selected: null, hint: null, last: null });
  setStatus('YOUR TURN');
  log('◀ Undo');
  render();
}

// ── Watch mode: two agents play each other ───────────────────────────────

// Plays one move for the side to move. Returns {kind}: 'go' to continue, 'win' (with winner) or 'draw' when the
// game has ended, 'stale' if a new game started while the move was being chosen, or 'error'.
async function watchStep() {
  const gid = state.gameId;
  const side = state.turn;
  const red = side === RED;
  const agent = $(red ? 'red-agent' : 'white-agent').value;
  const level = Number($(red ? 'red-level' : 'white-level').value);
  const who = `${red ? 'Red' : 'White'} (${AGENT_LABELS[agent]})`;
  state.busy = true;
  watchStatus(`${red ? 'RED' : 'WHITE'} (${AGENT_LABELS[agent]}) THINKING`);
  render();
  let res;
  try {
    res = await postJSON('/api/checkers/move', {
      board: state.board, turn: side, size: state.n, forced: state.forced, agent, level,
    });
  } catch (err) {
    if (gid !== state.gameId) return { kind: 'stale' };
    state.busy = false;
    state.running = false;
    log(`✗ ${err.message}`, 'log-err');
    setStatus('ERROR');
    render();
    return { kind: 'error' };
  }
  if (gid !== state.gameId) return { kind: 'stale' };
  state.busy = false;
  apply(res.move, who);
  showAnalysis(res, who);
  if (!res.reply.length) return { kind: 'win', winner: side };
  if (state.quiet >= DRAW_PLIES) return { kind: 'draw' };
  const nextAgent = $(state.turn === RED ? 'red-agent' : 'white-agent').value;
  watchStatus(`${state.turn === RED ? 'RED' : 'WHITE'} (${AGENT_LABELS[nextAgent]}) TO MOVE`);
  return { kind: 'go' };
}

// Plays moves until the game ends, the watch is paused, or a new game starts. The loop flag keeps two loops from
// running at once; when a loop ends it starts again if a new game began while it was busy.
async function watchLoop() {
  if (state.loopActive) return;
  state.loopActive = true;
  const gid = state.gameId;
  try {
    while (state.running && !state.paused && gid === state.gameId) {
      const outcome = await watchStep();
      if (outcome.kind === 'go') {
        await sleep(Number($('speed').value));
        continue;
      }
      if (outcome.kind === 'win' || outcome.kind === 'draw') endWatchGame(outcome);
      break;
    }
  } finally {
    state.loopActive = false;
  }
  if (state.running && !state.paused && !state.over) watchLoop();
}

function endWatchGame(outcome) {
  state.over = true;
  state.score.games++;
  let text, sub, color;
  if (outcome.kind === 'draw') {
    state.score.draw++;
    text = 'DRAW';
    sub = `${DRAW_PLIES} plies without a capture or a man moving`;
    color = '#00f5ff';
  } else if (outcome.winner === RED) {
    state.score.red++;
    text = 'RED WINS';
    sub = 'white has no legal move left';
    color = '#ff00a0';
  } else {
    state.score.white++;
    text = 'WHITE WINS';
    sub = 'red has no legal move left';
    color = '#00ff88';
  }
  renderScore();
  setStatus(text);
  log(`★ ${text}: ${sub}`, 'log-best');
  render();
  if ($('auto').checked) {
    // Auto-restart keeps the pause state: a paused watch waits for resume or step.
    setTimeout(() => { if (state.mode === 'watch' && state.over && state.running) newGame(); }, 1500);
  } else {
    burst($('ck'), { count: 150, colors: undefined });
    banner(text, sub, color);
  }
}

async function stepOnce() {
  if (state.loopActive || state.busy || state.over) return;
  const outcome = await watchStep();
  if (outcome.kind === 'win' || outcome.kind === 'draw') endWatchGame(outcome);
  render();
}

function togglePause() {
  state.paused = !state.paused;
  if (state.paused) {
    setStatus('PAUSED');
    render();
  } else {
    render();
    watchLoop();
  }
}

// ── New games and mode switches ──────────────────────────────────────────

async function newGame() {
  const gid = ++state.gameId;
  state.human = Number($('side').value);
  Object.assign(state, {
    board: sizeInfo().start.slice(), turn: RED, history: [], quiet: 0, ply: 0, over: false,
    selected: null, hint: null, last: null, legal: [], busy: false,
  });
  $('log').innerHTML = '';
  $('prune').hidden = true;
  $('an-root').innerHTML = '';
  $('an-title').textContent = state.mode === 'watch' ? 'Watch the agents think.' : 'Make a move to see the agent think.';
  if (state.mode === 'watch') {
    state.running = true;
    watchStatus(`RED (${AGENT_LABELS[$('red-agent').value]}) THINKING`);
    render();
    watchLoop();
    return;
  }
  state.running = false;
  state.paused = false;
  if (state.turn === state.human) {
    const moves = (await postJSON('/api/checkers/legal', {
      board: state.board, turn: state.turn, size: state.n, forced: state.forced,
    })).moves;
    if (gid !== state.gameId) return;
    state.legal = moves;
    setStatus('YOUR TURN');
    render();
  } else {
    render();
    await agentTurn();
  }
}

function setMode(mode) {
  state.mode = mode;
  const watch = mode === 'watch';
  $('human-panel').hidden = watch;
  $('watch-panel').hidden = !watch;
  $('watch-pace').hidden = !watch;
  $('btn-undo').hidden = watch;
  $('btn-hint').hidden = watch;
  $('btn-pause').hidden = !watch;
  $('btn-step').hidden = !watch;
  state.paused = false;
  newGame();
  updateMatchup();
}

// One-line tooltip for the OPPONENT menu. The full method is in HOW IT WORKS.
const AGENT_TIPS = {
  alphabeta: 'Searches the game tree as deep as its time allows, skipping lines that cannot change the result.',
  minimax: 'Searches every line to a fixed depth, with no pruning.',
  mcts: 'Scores each move by random games played from it (UCT).',
  greedy: 'Takes the move that looks best one move ahead.',
  random: 'Picks a legal move at random.',
  chance: 'Samples a move from fixed weights, with no search.',
};

function describeAgent() {
  $('agent').title = AGENT_TIPS[$('agent').value] ?? '';
}

function fillAgentSelect(select, selected) {
  select.innerHTML = '';
  for (const a of state.meta.agents) {
    const opt = document.createElement('option');
    opt.value = a.name;
    opt.textContent = AGENT_OPTIONS[a.name] ?? a.name;
    opt.selected = a.name === selected;
    select.appendChild(opt);
  }
}

async function init() {
  $('btn-new').onclick = newGame;
  document.addEventListener('keydown', (e) => { if (e.key === 'Escape' && state.selected !== null) cancelSelection(); });
  $('btn-undo').onclick = undo;
  $('btn-hint').onclick = hint;
  $('btn-pause').onclick = togglePause;
  $('btn-step').onclick = stepOnce;
  $('btn-reset-score').onclick = () => { state.score = { red: 0, white: 0, draw: 0, games: 0 }; renderScore(); };
  $('mode').onchange = () => setMode($('mode').value);
  $('side').onchange = () => { newGame(); updateMatchup(); };
  $('size').onchange = () => { state.n = Number($('size').value); newGame(); updateMatchup(); };
  $('forced').onchange = () => { state.forced = $('forced').checked; newGame(); updateMatchup(); };
  $('agent').onchange = () => { describeAgent(); updateMatchup(); };
  for (const id of ['level', 'red-agent', 'red-level', 'white-agent', 'white-level']) {
    $(id).onchange = updateMatchup;
  }
  $('speed').oninput = () => { $('speed-val').textContent = `${$('speed').value} ms`; };
  $('speed-val').textContent = `${$('speed').value} ms`;
  try {
    state.meta = await getJSON('/api/checkers/meta');
  } catch (err) {
    log(`✗ Server unreachable (${err.message}).`, 'log-err');
    return;
  }
  fillAgentSelect($('agent'), 'alphabeta');
  fillAgentSelect($('red-agent'), 'alphabeta');
  fillAgentSelect($('white-agent'), 'mcts');
  describeAgent();
  setMode($('mode').value);
}

init();
