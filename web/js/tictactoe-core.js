// Tic-tac-toe rules and an alpha-beta minimax search, for the 404 page's game.
//
// This is a pure module (no DOM), so node can test it. It mirrors tictactoe/core.py in the repo: the same
// move order, the same cutoffs and the same records, so tests/web/tictactoe.test.mjs can check the two
// agree on every reachable position. A board is a 9-character string, row by row: '.' is empty, 'X' and
// 'O' are the marks. X moves first. Values are from X's point of view: 1 X wins, 0 draw, -1 O wins.

export const LINES = [[0, 1, 2], [3, 4, 5], [6, 7, 8], [0, 3, 6], [1, 4, 7], [2, 5, 8], [0, 4, 8], [2, 4, 6]];
export const EMPTY_BOARD = '.........';
// Outside the value range -1..1, so the first child always improves on it.
const INF = 2;
// Records are kept for plies 0 and 1 only: the agent's moves and the replies to them.
const MAX_PLY = 2;

/** 'X' or 'O' if that side has three in a row, else ''. */
export function winner(board) {
  for (const [a, b, c] of LINES) {
    if (board[a] !== '.' && board[a] === board[b] && board[b] === board[c]) return board[a];
  }
  return '';
}

/** Empty squares in index order, which is the order the search tries them. */
export function legalMoves(board) {
  const out = [];
  for (let i = 0; i < 9; i++) if (board[i] === '.') out.push(i);
  return out;
}

export function other(turn) {
  return turn === 'X' ? 'O' : 'X';
}

export function place(board, square, mark) {
  return board.slice(0, square) + mark + board.slice(square + 1);
}

/** The side to move from a board: X moves when the counts are equal. */
export function turnOf(board) {
  let x = 0, o = 0;
  for (const ch of board) { if (ch === 'X') x++; else if (ch === 'O') o++; }
  return x === o ? 'X' : 'O';
}

// Alpha-beta over one node. Returns [value, exact, bestMove]. kids receives this node's child records while
// ply < MAX_PLY. stats is [searched, pruned], shared by the whole search.
function node(board, turn, alpha, beta, ply, kids, stats) {
  stats[0]++;
  const w = winner(board);
  if (w) return [w === 'X' ? 1 : -1, true, null];
  const moves = legalMoves(board);
  if (!moves.length) return [0, true, null];
  const keep = ply < MAX_PLY;
  const maximize = turn === 'X';
  const a0 = alpha, b0 = beta;
  let v = maximize ? -INF : INF;
  let best = null;
  for (let k = 0; k < moves.length; k++) {
    const m = moves[k];
    const rec = { m, v: null, b: '', pruned: false, kids: [], s: 0, p: 0 };
    const [cv, cexact] = node(place(board, m, turn), other(turn), alpha, beta, ply + 1, rec.kids, stats);
    rec.v = cv;
    // The child was searched with the window (alpha, beta) as it stood at the call: a value at or below
    // alpha, or at or above beta, is only a bound.
    rec.b = cexact ? '' : (cv <= alpha ? 'hi' : 'lo');
    rec.s = stats[0];
    rec.p = stats[1];
    if (keep) kids.push(rec);
    if ((maximize && cv > v) || (!maximize && cv < v)) { v = cv; best = m; }
    if (maximize) alpha = Math.max(alpha, v);
    else beta = Math.min(beta, v);
    if (alpha >= beta) {
      // Cutoff: the remaining siblings can not change this node's value, so they are never searched.
      const rest = moves.slice(k + 1);
      stats[1] += rest.length;
      if (keep) {
        for (const r of rest) kids.push({ m: r, v: null, b: '', pruned: true, kids: [], s: stats[0], p: stats[1] });
      }
      break;
    }
  }
  return [v, a0 < v && v < b0, best];
}

/**
 * Alpha-beta minimax for the side to move. Returns {move, value, searched, pruned, tree}: move is null when
 * the game is already over, and tree holds the first two plies as described in tictactoe/core.py.
 */
export function search(board, turn) {
  if (winner(board) || !legalMoves(board).length) {
    const stats = [0, 0];
    const [value] = node(board, turn, -INF, INF, 0, [], stats);
    return { move: null, value, searched: 1, pruned: 0, tree: [] };
  }
  const stats = [0, 0];
  const tree = [];
  const [value, , move] = node(board, turn, -INF, INF, 0, tree, stats);
  return { move, value, searched: stats[0], pruned: stats[1], tree };
}

/** Every board reachable from the empty board. Boards that end the game are included only on request. */
export function reachablePositions(includeFinished = false) {
  const seen = new Set([EMPTY_BOARD]);
  const queue = [EMPTY_BOARD];
  for (let q = 0; q < queue.length; q++) {
    const board = queue[q];
    if (winner(board)) continue;
    const turn = turnOf(board);
    for (const m of legalMoves(board)) {
      const next = place(board, m, turn);
      if (!seen.has(next)) { seen.add(next); queue.push(next); }
    }
  }
  return [...seen]
    .filter((b) => includeFinished || (!winner(b) && b.includes('.')))
    .sort();
}
