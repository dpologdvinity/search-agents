// The tic-tac-toe page: you against the alpha-beta agent in tictactoe-core.js at a chosen level, with its
// search tree drawn as it thinks (tictactoe-view.js). Everything runs in the browser; there is no server call.
// A level is a search depth: EASY and MEDIUM stop a few plies ahead and score what they cannot see as a draw,
// so they walk into forks past their horizon. PERFECT searches to the end of the game and never loses.

import { EMPTY_BOARD, LEVELS, legalMoves, other, place, search, winner } from './tictactoe-core.js';
import { createBoard, createTree } from './tictactoe-view.js';
import { play } from './sfx.js';
import { banner, burst, shake } from './fx.js';

const $ = (id) => document.getElementById(id);
const boardEl = $('ttt-board');
const noteEl = $('ttt-note');
const tree = createTree($('ttt-tree'), $('ttt-searched'), $('ttt-pruned'));
const LEVEL_NAMES = { easy: 'EASY', medium: 'MEDIUM', perfect: 'PERFECT' };

let board = EMPTY_BOARD;
let human = 'X';
let level = 'medium';
let busy = false;
let over = false;
const score = { w: 0, d: 0, l: 0 };   // this visit only, reset with the page

const drawBoard = createBoard(boardEl, humanMove);
const render = (last = -1) => drawBoard(board, { last, locked: busy || over, human });

function setStatus(text) {
  $('chip-status').textContent = text;
}

function showScore() {
  $('chip-score').textContent = `${score.w}-${score.d}-${score.l}`;
}

// The horizon in words, for the notes: a depth is the number of moves (plies) the agent looks ahead.
function horizonText() {
  const d = LEVELS[level];
  return d === null ? 'searches to the end of the game' : `looks ${d} moves ahead`;
}

function humanMove(i) {
  if (busy || over || board[i] !== '.') return;
  board = place(board, i, human);
  play('click');
  render(i);
  if (!finished(i)) agentTurn();
}

// The agent searches the whole position at once (it takes milliseconds), then the tree is drawn node by
// node while the counters climb. The chosen square is played when the drawing ends.
function agentTurn() {
  const agent = other(human);
  busy = true;
  setStatus('THINKING');
  noteEl.textContent = `The agent (${LEVEL_NAMES[level]}) ${horizonText()}...`;
  render();
  const result = search(board, agent, LEVELS[level]);
  tree.draw(result, agent, () => {
    board = place(board, result.move, agent);
    busy = false;
    render(result.move);
    if (!finished(result.move)) {
      setStatus('YOUR MOVE');
      noteEl.textContent = `The agent played square ${result.move + 1}. Your move.`;
    }
  });
}

// Returns true when the game is over, after scoring it and showing the verdict. last is the square just
// played, so it keeps its pop when the board locks.
function finished(last) {
  const w = winner(board);
  if (!w && legalMoves(board).length) return false;
  over = true;
  render(last);
  if (w === human) {
    score.w++;
    setStatus('YOU WIN');
    noteEl.textContent = `You won. The agent ${horizonText()}, and your winning line only showed up past that.`;
    banner('YOU WIN', `against ${LEVEL_NAMES[level]}`, '#00ff88');
    burst(boardEl);
  } else if (w) {
    score.l++;
    setStatus('AGENT WINS');
    noteEl.textContent = 'The agent won. Press NEW GAME to try again.';
    banner('AGENT WINS', `${LEVEL_NAMES[level]} level`, '#ff3b3b');
    shake(boardEl, 'big');
  } else {
    score.d++;
    setStatus('DRAW');
    noteEl.textContent = level === 'perfect'
      ? 'Draw. With perfect play on both sides every game ends here, so this is the best result against PERFECT.'
      : 'Draw. Try a different opening: the agent cannot see far enough to stop every fork.';
    banner('DRAW', `${LEVEL_NAMES[level]} level`, '#ffe600');
  }
  showScore();
  return true;
}

// A new game with the current settings. X moves first, so the agent opens when you play O.
function newGame() {
  tree.clear();
  human = $('you').value;
  level = $('level').value;
  board = EMPTY_BOARD;
  busy = false;
  over = false;
  $('chip-level').textContent = LEVEL_NAMES[level];
  render();
  if (human === 'O') {
    agentTurn();
  } else {
    setStatus('YOUR MOVE');
    noteEl.textContent = `You are X and move first. The agent ${horizonText()}.`;
  }
}

$('btn-new').addEventListener('click', newGame);
$('you').addEventListener('change', newGame);
$('level').addEventListener('change', newGame);
$('btn-score-reset').addEventListener('click', () => {
  score.w = score.d = score.l = 0;
  showScore();
});

showScore();
newGame();
