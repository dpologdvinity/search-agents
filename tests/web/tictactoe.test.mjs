// Tests for the 404 page's tic-tac-toe core. The parity test runs the Python reference (tictactoe/) on every
// reachable position and checks the JS core returns the same move, value, counts and recorded tree.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import path from 'node:path';
import {
  winner, legalMoves, search, reachablePositions, turnOf, EMPTY_BOARD,
} from '../../web/js/tictactoe-core.js';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..', '..');
// The interpreter for the reference: TTT_PYTHON overrides it (e.g. the repo venv), else python3.
const PYTHON = process.env.TTT_PYTHON || 'python3';

test('winner finds each line and ignores open boards', () => {
  assert.equal(winner('XXX......'), 'X');
  assert.equal(winner('O..O..O..'), 'O');
  assert.equal(winner('X...X...X'), 'X');
  assert.equal(winner('..O.O.O..'), 'O');
  assert.equal(winner('XO.XO.XO.'), 'X');
  assert.equal(winner('XOXXOOOXX'), '');
});

test('legal moves are the empty squares in index order', () => {
  assert.deepEqual(legalMoves('XO.......'), [2, 3, 4, 5, 6, 7, 8]);
  assert.deepEqual(legalMoves('XOXXOOOXX'), []);
});

test('the empty board is a draw and the agent takes the forced block', () => {
  assert.equal(search(EMPTY_BOARD, 'X').value, 0);
  const r = search('XX.O.....', 'O');
  assert.equal(r.move, 2);
});

test('finished games have no move and an empty tree', () => {
  const r = search('XXXOO....', 'O');
  assert.deepEqual([r.move, r.value, r.tree], [null, 1, []]);
});

test('the search records the first two plies with running counts', () => {
  const r = search(EMPTY_BOARD, 'X');
  assert.ok(r.pruned > 0);
  assert.ok(r.searched < 549946);
  assert.deepEqual(r.tree.map((rec) => rec.m), [0, 1, 2, 3, 4, 5, 6, 7, 8]);
  assert.ok(r.tree.every((rec) => rec.kids.length > 0));
  const seq = r.tree.map((rec) => rec.s);
  assert.deepEqual(seq, [...seq].sort((a, b) => a - b));
});

test('reachable positions: 5,478 boards, 4,520 still in play', () => {
  assert.equal(reachablePositions(true).length, 5478);
  assert.equal(reachablePositions().length, 4520);
});

test('turnOf follows the mark counts', () => {
  assert.equal(turnOf(EMPTY_BOARD), 'X');
  assert.equal(turnOf('X........'), 'O');
});

// Python reference: positions for every reachable in-play board, with the search result for its side to move.
function pythonPositions() {
  const raw = execFileSync(PYTHON, ['-m', 'tictactoe', 'positions'], {
    cwd: ROOT,
    env: { ...process.env, PYTHONPATH: ROOT },
    maxBuffer: 1 << 28,
    encoding: 'utf8',
  });
  return JSON.parse(raw);
}

let reference = null;
try {
  reference = pythonPositions();
} catch (err) {
  // Without a Python that can import tictactoe, the parity test is skipped rather than failed.
  reference = null;
  console.log(`# parity skipped: ${err.code || err.message.split('\n')[0]}`);
}

test('JS core agrees with the Python reference on every reachable position', { skip: !reference }, () => {
  assert.equal(reference.length, 4520);
  assert.deepEqual(reference.map((p) => p.board), reachablePositions());
  for (const p of reference) {
    const r = search(p.board, p.turn);
    assert.equal(turnOf(p.board), p.turn, p.board);
    assert.deepEqual(
      { move: r.move, value: r.value, searched: r.searched, pruned: r.pruned, tree: r.tree },
      { move: p.move, value: p.value, searched: p.searched, pruned: p.pruned, tree: p.tree },
      `position ${p.board}`,
    );
  }
});
