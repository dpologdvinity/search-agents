// The Konami matcher behind AGENT OVERDRIVE: the full code completes, mistakes restart it, and letters ignore case.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { KONAMI, konamiStep, normalizeKey } from '../../web/js/konami.js';

// Feed keys through the matcher and return the indexes at which the code completed.
function feed(keys) {
  let progress = 0;
  const completed = [];
  keys.forEach((key, i) => {
    const step = konamiStep(progress, normalizeKey(key));
    progress = step.progress;
    if (step.done) completed.push(i);
  });
  return completed;
}

test('the full code completes once, on its last key', () => {
  assert.deepEqual(feed(KONAMI), [KONAMI.length - 1]);
});

test('the code completes again after a second full entry', () => {
  assert.deepEqual(feed([...KONAMI, ...KONAMI]), [9, 19]);
});

test('a wrong key restarts the code', () => {
  assert.deepEqual(feed(['ArrowUp', 'ArrowUp', 'ArrowDown', 'ArrowLeft', ...KONAMI]), [13]);
});

test('an up arrow that is not the second key still starts a new attempt', () => {
  // The stray ArrowUp after the first one is taken as the start of a fresh attempt, so the code still completes.
  assert.deepEqual(feed(['ArrowUp', 'ArrowUp', 'ArrowUp', ...KONAMI.slice(1)]), [KONAMI.length + 1]);
});

test('letters ignore case', () => {
  const upper = KONAMI.map((k) => (k.length === 1 ? k.toUpperCase() : k));
  assert.deepEqual(feed(upper), [KONAMI.length - 1]);
});

test('named keys are not lowercased into letters', () => {
  assert.equal(normalizeKey('ArrowUp'), 'ArrowUp');
  assert.equal(normalizeKey('Escape'), 'Escape');
  assert.equal(normalizeKey('B'), 'b');
});
