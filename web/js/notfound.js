// The 404 page: tic-tac-toe against the alpha-beta agent in tictactoe-core.js, its search tree drawn as the
// agent "thinks", and a grid linking to every game. The rules and the search are pure (tested in node); this
// file only draws them. Each recorded tree node appears in search order, so the drawing is the search itself.

import { GAMES } from './nav.js';
import { EMPTY_BOARD, legalMoves, place, search, winner } from './tictactoe-core.js';
import { expand, play, announce } from './sfx.js';
import { burst, shake } from './fx.js';

const REDUCED = matchMedia('(prefers-reduced-motion: reduce)').matches;
const STEP_MS = 30;          // delay between drawn tree nodes; the whole tree takes a second or two
const SVG = 'http://www.w3.org/2000/svg';
// Tree layout: nine move columns (one per square, in square order), the agent's moves in a row, and the
// replies to each move stacked underneath.
const COL = 58, ROOT_X = 4.5 * COL;
const Y_ROOT = 22, Y_MOVE = 40, MOVE_H = 22, Y_REPLY = 88, REPLY_GAP = 18, REPLY_H = 15;

const $ = (id) => document.getElementById(id);
const boardEl = $('ttt-board');
const statusEl = $('ttt-status');
const verdictEl = $('ttt-verdict');
const treeEl = $('ttt-tree');
const fmt = (n) => Number(n).toLocaleString();

let board = EMPTY_BOARD;
let busy = false;
let over = false;
let run = 0;   // bumps on every new game, so a tree still being drawn stops when the game is reset

// Build the nine squares once; render() only changes their text and state.
for (let i = 0; i < 9; i++) {
  const cell = document.createElement('button');
  cell.type = 'button';
  cell.className = 'ttt-cell';
  cell.dataset.i = String(i);
  cell.addEventListener('click', () => humanMove(i));
  boardEl.appendChild(cell);
}

function render(last = -1) {
  [...boardEl.children].forEach((cell, i) => {
    const mark = board[i];
    cell.textContent = mark === '.' ? '' : mark;
    cell.className = 'ttt-cell' + (mark === 'X' ? ' is-x' : mark === 'O' ? ' is-o' : '') + (i === last ? ' just' : '');
    cell.disabled = mark !== '.' || busy || over;
    cell.setAttribute('aria-label', `square ${i + 1}, ${mark === '.' ? 'empty' : mark === 'X' ? 'you' : 'agent'}`);
  });
}

function humanMove(i) {
  if (busy || over || board[i] !== '.') return;
  board = place(board, i, 'X');
  play('click');
  render(i);
  if (finished()) return;
  agentTurn();
}

// The agent is O. It searches the whole position at once (the search is quick here), then the tree is drawn
// node by node while the readout counts up. The chosen square is played when the drawing ends.
function agentTurn() {
  busy = true;
  statusEl.textContent = 'The agent is thinking...';
  render();
  const id = ++run;
  const result = search(board, 'O');
  drawTree(result, id, () => {
    if (id !== run) return;
    board = place(board, result.move, 'O');
    busy = false;
    render(result.move);
    statusEl.textContent = 'Your move.';
    finished();
  });
}

// Returns true when the game is over, after showing the verdict.
function finished() {
  const w = winner(board);
  if (!w && legalMoves(board).length) return false;
  over = true;
  render();
  if (w === 'X') {
    verdictEl.className = 'nf-verdict impossible';
    verdictEl.innerHTML = 'IMPOSSIBLE. You won. Minimax does not lose, so something is broken, and it is not the agent. '
      + '<a href="#nf-links-title">Try a real game &rarr;</a>';
    statusEl.textContent = 'You won. That should not be possible.';
    burst(boardEl);
    announce('win');
  } else if (w === 'O') {
    verdictEl.className = 'nf-verdict loss';
    verdictEl.innerHTML = 'You can&rsquo;t beat minimax. Nobody can. <a href="#nf-links-title">Try a real game &rarr;</a>';
    statusEl.textContent = 'The agent won.';
    shake(boardEl, 'big');
  } else {
    verdictEl.className = 'nf-verdict draw';
    verdictEl.innerHTML = 'Draw. Perfect play always ends here, and the agent plays perfectly. '
      + '<a href="#nf-links-title">Try a real game &rarr;</a>';
    statusEl.textContent = 'Draw.';
    announce('note');
  }
  return true;
}

function newGame() {
  run++;
  board = EMPTY_BOARD;
  busy = false;
  over = false;
  verdictEl.className = 'nf-verdict';
  verdictEl.textContent = '';
  statusEl.textContent = 'Your move. Pick a square.';
  treeEl.replaceChildren();
  $('ttt-searched').textContent = '0';
  $('ttt-pruned').textContent = '0';
  render();
}

// ── The search tree ─────────────────────────────────────────────────────

const svg = (tag, attrs = {}) => {
  const el = document.createElementNS(SVG, tag);
  for (const [k, v] of Object.entries(attrs)) el.setAttribute(k, v);
  return el;
};

// Values are shown from the agent's side (it is O): a record's v is X's value, so negate it. A bound is
// flipped to match: an X-side "at most" is an agent-side "at least".
function valueText(rec) {
  const a = -rec.v;
  const sign = a > 0 ? '+1' : a < 0 ? '-1' : '0';
  const bound = rec.b === 'hi' ? '≥' : rec.b === 'lo' ? '≤' : '';
  return bound + sign;
}

function classOf(rec) {
  if (rec.b) return 'refuted';
  const a = -rec.v;
  return a > 0 ? 'win' : a < 0 ? 'loss' : 'draw';
}

function drawRoot() {
  treeEl.appendChild(svg('text', { x: ROOT_X, y: Y_ROOT, class: 't-root', 'text-anchor': 'middle' })).textContent = 'AGENT (O) TO MOVE';
}

function drawMove(rec) {
  const x = rec.m * COL + COL / 2;
  treeEl.appendChild(svg('line', { x1: ROOT_X, y1: Y_ROOT + 4, x2: x, y2: Y_MOVE, class: 't-edge' }));
  const g = svg('g', { class: `t-node move ${classOf(rec)}`, 'data-move': rec.m });
  g.appendChild(svg('rect', { x: x - 23, y: Y_MOVE, width: 46, height: MOVE_H, rx: 3 }));
  const t = svg('text', { x, y: Y_MOVE + 14.5, 'text-anchor': 'middle' });
  t.textContent = `${rec.m + 1}  ${valueText(rec)}`;
  g.appendChild(t);
  treeEl.appendChild(g);
}

// Replies sit under their move, in the order the search tried them. Pruned replies are struck through.
function drawReply(move, rec, k) {
  const x = move * COL + COL / 2;
  const y = Y_REPLY + k * REPLY_GAP;
  treeEl.appendChild(svg('path', { d: `M${x} ${Y_MOVE + MOVE_H} V${y + REPLY_H / 2}`, class: 't-edge' }));
  const g = svg('g', { class: `t-node reply ${rec.pruned ? 'pruned' : classOf(rec)}` });
  g.appendChild(svg('rect', { x: x - 23, y, width: 46, height: REPLY_H, rx: 2 }));
  const t = svg('text', { x, y: y + 11, 'text-anchor': 'middle' });
  t.textContent = rec.pruned ? `${rec.m + 1}` : `${rec.m + 1}  ${valueText(rec)}`;
  g.appendChild(t);
  if (rec.pruned) g.appendChild(svg('line', { x1: x - 20, y1: y + REPLY_H / 2, x2: x + 20, y2: y + REPLY_H / 2, class: 't-strike' }));
  treeEl.appendChild(g);
}

// Draws the recorded tree in order, one node per step. Each step updates the counters from the record.
function drawTree(result, id, done) {
  treeEl.replaceChildren();
  drawRoot();
  const steps = [];
  for (const rec of result.tree) {
    steps.push({ rec, move: null, k: 0 });
    rec.kids.forEach((kid, k) => steps.push({ rec: kid, move: rec.m, k }));
  }
  const total = steps.length || 1;
  const show = (step, n) => {
    if (step.move === null) drawMove(step.rec);
    else drawReply(step.move, step.rec, step.k);
    $('ttt-searched').textContent = fmt(step.rec.s);
    $('ttt-pruned').textContent = fmt(step.rec.p);
    if (!REDUCED) expand(n / total);
  };
  if (REDUCED) {
    steps.forEach((s, n) => show(s, n));
    finishTree(result, done);
    return;
  }
  let n = 0;
  const next = () => {
    if (id !== run) return;
    if (n >= steps.length) { finishTree(result, done, id); return; }
    show(steps[n], n);
    n++;
    setTimeout(next, STEP_MS);
  };
  next();
}

// The readout lands on the exact totals (records only cover the first two plies), the chosen move is marked,
// and the game continues after a short pause.
function finishTree(result, done, id) {
  $('ttt-searched').textContent = fmt(result.searched);
  $('ttt-pruned').textContent = fmt(result.pruned);
  treeEl.querySelectorAll('.t-node.move').forEach((g) => {
    if (Number(g.dataset.move) === result.move) g.classList.add('chosen');
  });
  setTimeout(() => { if (!id || id === run) done(); }, REDUCED ? 0 : 260);
}

// ── Quick links to every game ───────────────────────────────────────────

$('nf-links').innerHTML = GAMES.map((g) => `<a class="nf-link" href="${g.href}">${g.label}</a>`).join('');

newGame();
$('ttt-new').addEventListener('click', newGame);
