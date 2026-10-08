// Checkers page. The server owns the rules: it returns the player's legal moves
// (with the board after each) and the agent's move with its search statistics.

import { getJSON, postJSON } from './api.js';
import { banner, burst, pop, shake } from './fx.js';

const $ = (id) => document.getElementById(id);
const RED = 1;
const DRAW_PLIES = 80; // 40 moves each without a capture or a man moving

const state = {
  board: [], turn: RED, human: RED, legal: [], selected: null, history: [],
  quiet: 0, ply: 0, busy: false, over: false, hint: null, last: null, meta: null,
};

const rcOf = (i) => { const r = Math.floor(i / 4); return [r, 2 * (i % 4) + (r % 2 === 0 ? 1 : 0)]; };
const sqAt = (r, c) => ((r + c) % 2 === 1 ? r * 4 + Math.floor(c / 2) : -1);
const count = (side) => state.board.filter((p) => p * side > 0).length;

// ── Rendering ────────────────────────────────────────────────────────────

function render() {
  const el = $('ck');
  el.innerHTML = '';
  const flip = state.human !== RED;
  const targets = new Set(state.selected === null ? [] :
    state.legal.filter((m) => m.path[0] === state.selected).map((m) => m.path[m.path.length - 1]));
  const movable = new Set(state.legal.map((m) => m.path[0]));
  const yourTurn = !state.over && !state.busy && state.turn === state.human;
  for (let vr = 0; vr < 8; vr++) {
    for (let vc = 0; vc < 8; vc++) {
      const r = flip ? 7 - vr : vr, c = flip ? 7 - vc : vc, sq = sqAt(r, c);
      const cell = document.createElement('div');
      cell.className = 'ck-cell' + (sq < 0 ? ' light' : ' dark');
      if (sq >= 0) {
        if (state.last && state.last.path.includes(sq)) cell.classList.add('trail');
        if (state.hint && state.hint.path.includes(sq)) cell.classList.add('hint');
        if (targets.has(sq)) { cell.classList.add('target'); cell.onclick = () => moveTo(sq); }
        const p = state.board[sq];
        if (p) {
          const piece = document.createElement('div');
          piece.className = `ck-piece ${p > 0 ? 'red' : 'white'}${Math.abs(p) === 2 ? ' king' : ''}`;
          if (yourTurn && movable.has(sq)) {
            piece.classList.add('movable');
            piece.tabIndex = 0;
            piece.onclick = () => select(sq);
            piece.onkeydown = (e) => { if (e.key === 'Enter' || e.key === ' ') select(sq); };
          }
          if (sq === state.selected) piece.classList.add('selected');
          cell.appendChild(piece);
        }
      }
      el.appendChild(cell);
    }
  }
  $('chip-pieces').textContent = `${count(RED)} – ${count(-RED)}`;
  $('chip-move').textContent = String(Math.floor(state.ply / 2) + 1);
  $('chip-you').textContent = state.human === RED ? 'RED' : 'WHITE';
  $('btn-undo').disabled = state.busy || !state.history.length;
  $('btn-hint').disabled = state.busy || state.over || state.turn !== state.human;
}

function log(text, cls = 'log-info') {
  const line = document.createElement('div');
  line.className = cls;
  line.textContent = text;
  $('log').appendChild(line);
  $('log').scrollTop = $('log').scrollHeight;
}

const setStatus = (t) => { $('chip-status').textContent = t; };

function scoreText(s) {
  if (!s) return 'forced';
  if (s.result === 'win') return `wins in ${Math.ceil(s.plies / 2)}`;
  if (s.result === 'loss') return `loses in ${Math.ceil(s.plies / 2)}`;
  return (s.score > 0 ? '+' : '') + s.score;
}

function showAnalysis(res, who) {
  const a = res.analysis;
  $('prune').hidden = false;
  $('pm-nodes').textContent = a.nodes.toLocaleString();
  $('pm-cuts').textContent = a.agent === 'alphabeta' ? a.cutoffs.toLocaleString() : 'none (no pruning)';
  $('pm-depth').textContent = `${a.depth} half-moves`;
  $('an-title').textContent = `${who} played ${res.move.notation} after ${a.seconds.toFixed(2)} s`;
  const vals = a.root.map((r) => (r.score && r.score.result === 'eval' ? r.score.score : r.score && r.score.result === 'win' ? 400 : -400));
  const lo = Math.min(...vals), hi = Math.max(...vals);
  const list = $('an-root');
  list.innerHTML = '';
  a.root.slice(0, 7).forEach((r, k) => {
    const li = document.createElement('li');
    if (r.notation === res.move.notation) li.className = 'chosen';
    const w = hi === lo ? 100 : 12 + (88 * (vals[k] - lo)) / (hi - lo);
    li.innerHTML = `<span class="mv"></span><span class="bar"><i style="width:${w}%"></i></span><span class="sc"></span>`;
    li.querySelector('.mv').textContent = r.notation;
    li.querySelector('.sc').textContent = scoreText(r.score);
    list.appendChild(li);
  });
  $('an-note').textContent = 'Scores are from the mover\'s side, assuming the best reply each time. Squares are numbered 1–32 from the top-left.';
}

// ── Game flow ────────────────────────────────────────────────────────────

function select(sq) {
  state.selected = state.selected === sq ? null : sq;
  state.hint = null;
  render();
}

function apply(move, who) {
  state.history.push({ board: state.board, turn: state.turn, quiet: state.quiet, ply: state.ply, legal: state.legal });
  const before = state.board;
  const from = before[move.path[0]];
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
    shake(board, move.captured.length > 1 ? 'big' : 'small');
    pop(board, move.captured.length > 1 ? `${move.captured.length}x JUMP` : 'CAPTURE', '#ff00a0');
  }
  const crowned = Math.abs(from) === 1 && Math.abs(move.result[move.path[move.path.length - 1]]) === 2;
  if (crowned) { burst(board, { count: 70, colors: ['#ffe600', '#ff00a0', '#00f5ff'] }); pop(board, 'KINGED', '#ffe600'); }
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

async function moveTo(sq) {
  const move = state.legal.find((m) => m.path[0] === state.selected && m.path[m.path.length - 1] === sq);
  if (!move) return;
  apply(move, 'You');
  if (!checkDraw()) await agentTurn();
}

async function agentTurn() {
  state.busy = true;
  setStatus('THINKING');
  render();
  try {
    const res = await postJSON('/api/checkers/move', {
      board: state.board, turn: state.turn, agent: $('agent').value, level: Number($('level').value),
    });
    state.busy = false;
    apply(res.move, 'AI');
    showAnalysis(res, 'The agent');
    state.legal = res.reply;
    if (!state.legal.length) finish('AI WINS', 'you have no legal moves left', '#ff3b3b');
    else if (!checkDraw()) { setStatus('YOUR TURN'); render(); }
  } catch (err) {
    state.busy = false;
    if (/game is over/.test(err.message)) finish('YOU WIN', 'the agent has no legal moves left', '#00ff88');
    else { log(`✗ ${err.message}`, 'log-err'); setStatus('ERROR'); render(); }
  }
}

async function hint() {
  state.busy = true;
  render();
  try {
    const res = await postJSON('/api/checkers/move', { board: state.board, turn: state.turn, agent: 'alphabeta', level: 2 });
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

async function newGame() {
  state.human = Number($('side').value);
  Object.assign(state, { board: state.meta.start.slice(), turn: RED, history: [], quiet: 0, ply: 0, over: false,
                         selected: null, hint: null, last: null, legal: [] });
  $('log').innerHTML = '';
  $('prune').hidden = true;
  $('an-root').innerHTML = '';
  $('an-title').textContent = 'Make a move to see the agent think.';
  if (state.turn === state.human) {
    state.legal = (await postJSON('/api/checkers/legal', { board: state.board, turn: state.turn })).moves;
    setStatus('YOUR TURN');
    render();
  } else {
    render();
    await agentTurn();
  }
}

async function init() {
  $('btn-new').onclick = newGame;
  $('btn-undo').onclick = undo;
  $('btn-hint').onclick = hint;
  $('side').onchange = newGame;
  const describe = () => {
    const a = state.meta.agents.find((x) => x.name === $('agent').value);
    $('agent-desc').textContent = a ? a.description : '';
  };
  $('agent').onchange = describe;
  try {
    state.meta = await getJSON('/api/checkers/meta');
  } catch (err) {
    log(`✗ Server unreachable (${err.message}).`, 'log-err');
    return;
  }
  describe();
  await newGame();
}

init();
