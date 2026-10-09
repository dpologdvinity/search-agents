// Tests for the Tetris page's engine (web/js/tetris_engine.js): the two game-over rules, and the AI's
// placement, which must lock at the chosen placement even when the piece cannot reach it by moving.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import {
  H, W, Game, bestMove, dropRow, legalMoves, reachesByMoves, stepToward,
} from '../../web/js/tetris_engine.js';

// A board with the given filled cells, each [x, y] with y counting up from the bottom row.
function boardOf(cells) {
  const rows = new Array(H).fill(0);
  for (const [x, y] of cells) rows[y] |= 1 << x;
  return rows;
}

// A game whose next piece is `piece`, on `board`, with the spawn already taken.
function gameWith(piece, board, opts) {
  const g = new Game(1, opts);
  g.bag.take = () => piece;
  g.board = board;
  g.spawn();
  return g;
}

test('a blocked centre spawn ends a game played by hand', () => {
  const board = boardOf([[3, 18], [4, 18], [5, 18], [6, 18], [3, 19], [4, 19], [5, 19], [6, 19]]);
  // Every piece spawns over cols 3-6 of the top rows, so the centre spawn is blocked for each of them.
  for (const piece of 'IOTSZJL') {
    assert.equal(gameWith(piece, board, { endOnBlockedSpawn: true }).over, true, piece);
  }
});

test('the AI keeps playing when the centre spawn is blocked but another column is open', () => {
  const board = boardOf([[3, 18], [4, 18], [5, 18], [6, 18], [3, 19], [4, 19], [5, 19], [6, 19]]);
  const g = gameWith('T', board, { endOnBlockedSpawn: false });
  assert.equal(g.over, false);
  // The game goes on because the search still finds a placement: the Python rule in tetris/game.py.
  assert.notEqual(bestMove(g.board, g.piece, [0, 0, 0, 0, 0, 0, 0, 0, 0]).move, null);
});

test('the AI ends a game only when no placement is legal', () => {
  // Rows 18 and 19 full: no rotation or column of any piece can spawn, so there is no placement.
  const board = boardOf(Array.from({ length: W * 2 }, (_, i) => [i % W, 18 + Math.floor(i / W)]));
  const g = gameWith('O', board, { endOnBlockedSpawn: false });
  assert.equal(g.over, false);
  assert.equal(bestMove(g.board, g.piece, [0, 0, 0, 0, 0, 0, 0, 0, 0]).move, null);
});

test('every placement on an empty board is reachable by moving', () => {
  const empty = new Array(H).fill(0);
  for (const piece of 'IOTSZJL') {
    const g = gameWith(piece, empty);
    for (const t of legalMoves(empty, piece, [0, 0, 0, 0, 0, 0, 0, 0, 0])) {
      assert.equal(reachesByMoves(g, t), true, `${piece} rot ${t.rot} x ${t.x}`);
    }
  }
});

// A T on the spawn row with a block at column 6. Column 6 is filled in the two top rows, so the slide from
// the spawn column (3) to column 7 is blocked at column 6. The old page turned, tried to slide, failed, and
// dropped the piece at column 3. The new page must lock it at the chosen placement, column 7.
test('a slide blocked near the top locks the piece at the chosen placement, not where it stopped', () => {
  const g = gameWith('T', boardOf([[6, 18], [6, 19]]), { endOnBlockedSpawn: true });
  assert.equal(g.x, 3);
  const t = { rot: 0, x: 7, y: dropRow(g.board, g.shape(0), 7) };
  assert.equal(t.y, 0);
  assert.equal(reachesByMoves(g, t), false);
  assert.equal(stepToward(g, t), false, 'no step toward the target is possible');
  assert.equal(g.x, 3, 'the piece did not move');

  g.place(t.rot, t.x, t.y);
  // Row 0 gets the T's three bottom cells at columns 7, 8 and 9, and nothing in columns 3 to 5.
  assert.equal(g.board[0], (1 << 7) | (1 << 8) | (1 << 9));
});

test('a reachable placement animates one turn or slide at a time and then aligns', () => {
  const empty = new Array(H).fill(0);
  const g = gameWith('L', empty);
  const t = { rot: 2, x: 6, y: dropRow(empty, g.shape(2), 6) };
  assert.equal(reachesByMoves(g, t), true);
  let steps = 0;
  while (stepToward(g, t)) steps++;
  assert.equal(steps, 2 + 3, 'two quarter turns and three slides');
  assert.equal(g.rot, t.rot);
  assert.equal(g.x, t.x);
});

test('a turn blocked by a block beside the spawn is not reachable, and the piece stays where it spawned', () => {
  // The T spawns with its top row at columns 3-5 of row 18. Turned a quarter, it would need column 3 of row 19,
  // which is filled, so the turn fails and the piece stays in its spawn orientation.
  const g = gameWith('T', boardOf([[3, 19]]), { endOnBlockedSpawn: true });
  assert.equal(g.over, false);
  const t = { rot: 1, x: 3, y: 0 };
  assert.equal(reachesByMoves(g, t), false);
  assert.equal(stepToward(g, t), false);
  assert.equal(g.rot, 0);
});

test('the falling piece cannot be asked to reach a row above where it is', () => {
  const g = gameWith('O', new Array(H).fill(0));
  assert.equal(reachesByMoves(g, { rot: 0, x: g.x, y: g.y + 1 }), false);
});

test('the lock matches the placement the search chose on a board with a stack', () => {
  const board = boardOf([[0, 0], [1, 0], [2, 0], [3, 1], [4, 2], [5, 3]]);
  const g = gameWith('S', board, { endOnBlockedSpawn: false });
  const { move } = bestMove(g.board, g.piece, [-1, 1, -1, -1, -4, -1, 0, 0, 0]);
  const t = { rot: move.rot, x: move.x, y: move.y };
  const res = g.place(t.rot, t.x, t.y);
  assert.equal(res.lines, move.lines);
  assert.deepEqual(res.board, move.board);
});
