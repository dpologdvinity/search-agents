// The 404 page: tic-tac-toe against the alpha-beta agent in tictactoe-core.js, its search tree drawn as the
// agent "thinks", and a grid linking to every game. The rules and the search are pure (tested in node), and
// the board and tree drawing is shared with the tic-tac-toe page (tictactoe-view.js).

import { GAMES } from './nav.js';
import { EMPTY_BOARD, legalMoves, place, search, winner } from './tictactoe-core.js';
import { createBoard, createTree } from './tictactoe-view.js';
import { play, announce } from './sfx.js';
import { burst, shake } from './fx.js';

const $ = (id) => document.getElementById(id);
const boardEl = $('ttt-board');
const statusEl = $('ttt-status');
const verdictEl = $('ttt-verdict');
const tree = createTree($('ttt-tree'), $('ttt-searched'), $('ttt-pruned'));

let board = EMPTY_BOARD;
let busy = false;
let over = false;

const drawBoard = createBoard(boardEl, humanMove);
const render = (last = -1) => drawBoard(board, { last, locked: busy || over });

function humanMove(i) {
  if (busy || over || board[i] !== '.') return;
  board = place(board, i, 'X');
  play('click');
  render(i);
  if (finished(i)) return;
  agentTurn();
}

// The agent is O. It searches the whole position at once (the search is quick here), then the tree is drawn
// node by node while the readout counts up. The chosen square is played when the drawing ends.
function agentTurn() {
  busy = true;
  statusEl.textContent = 'The agent is thinking...';
  render();
  const result = search(board, 'O');
  tree.draw(result, 'O', () => {
    board = place(board, result.move, 'O');
    busy = false;
    render(result.move);
    statusEl.textContent = 'Your move.';
    finished(result.move);
  });
}

// Returns true when the game is over, after showing the verdict. last is the square just played, so it
// keeps its pop when the board locks.
function finished(last) {
  const w = winner(board);
  if (!w && legalMoves(board).length) return false;
  over = true;
  render(last);
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
  tree.clear();
  board = EMPTY_BOARD;
  busy = false;
  over = false;
  verdictEl.className = 'nf-verdict';
  verdictEl.textContent = '';
  statusEl.textContent = 'Your move. Pick a square.';
  render();
}

// ── Quick links to every game ───────────────────────────────────────────

$('nf-links').innerHTML = GAMES.map((g) => `<a class="nf-link" href="${g.href}">${g.label}</a>`).join('');

newGame();
$('ttt-new').addEventListener('click', newGame);
