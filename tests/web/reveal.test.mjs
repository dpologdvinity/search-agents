// The search-wave reveal's grid and wave functions (web/js/reveal.js). The DOM part runs only in a browser.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { CELL, WAVE_MS, bfsDistances, gridSize, startIndex, waveTimes } from '../../web/js/reveal.js';

test('the grid covers the viewport with whole cells', () => {
  assert.deepEqual(gridSize(1280, 800), { cols: Math.ceil(1280 / CELL), rows: Math.ceil(800 / CELL) });
  assert.deepEqual(gridSize(0, 0), { cols: 1, rows: 1 });
});

test('a click maps to its cell and clamps at the edges', () => {
  const cols = 10, rows = 5;
  assert.equal(startIndex(0, 0, cols, rows), 0);
  assert.equal(startIndex(CELL * 2 + 5, CELL + 5, cols, rows), 1 * cols + 2);
  assert.equal(startIndex(10000, 10000, cols, rows), rows * cols - 1);
  assert.equal(startIndex(-20, -20, cols, rows), 0);
});

test('BFS distances are Manhattan distances on an open grid', () => {
  const cols = 6, rows = 4, start = 0;
  const dist = bfsDistances(cols, rows, start);
  for (let i = 0; i < cols * rows; i++) {
    const c = i % cols, r = (i / cols) | 0;
    assert.equal(dist[i], c + r, `cell ${i}`);
  }
});

test('the wave starts at the click and reaches the far corner last', () => {
  const cols = 8, rows = 6;
  const start = startIndex(3 * CELL + 1, 2 * CELL + 1, cols, rows);
  const dist = bfsDistances(cols, rows, start);
  const times = waveTimes(dist);
  assert.equal(times[start], 0);
  assert.ok(Math.abs(Math.max(...times) - WAVE_MS) < 1e-9);
  // Cells arrive in BFS order: a farther cell never arrives before a nearer one.
  for (let i = 0; i < times.length; i++) {
    for (let j = 0; j < times.length; j++) {
      if (dist[i] < dist[j]) assert.ok(times[i] <= times[j]);
    }
  }
});

test('the wave lasts 400 to 600 ms', () => {
  assert.ok(WAVE_MS >= 400 && WAVE_MS <= 600);
});

test('a one-cell grid has no wave to draw', () => {
  const times = waveTimes(bfsDistances(1, 1, 0));
  assert.deepEqual([...times], [0]);
});
