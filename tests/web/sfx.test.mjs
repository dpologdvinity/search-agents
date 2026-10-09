// Node tests for web/js/sfx.js. The module must load with no browser globals (no window, document, or
// AudioContext), stay silent while off, and treat banners by colour and text. Run: node --test tests/web/
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { expand, isEnabled, outcome, play, setEnabled, waitFor } from '../../web/js/sfx.js';

test('loads without browser globals and is off by default', () => {
  assert.equal(isEnabled(), false);
  assert.equal(play('click'), false);
  assert.equal(expand(0.5), false);
});

test('turning sound on without Web Audio is a safe no-op', () => {
  setEnabled(true);
  assert.equal(isEnabled(), true);
  assert.equal(play('win'), false); // no AudioContext in Node, so nothing is scheduled
  setEnabled(false);
  assert.equal(isEnabled(), false);
});

test('banners are classified by colour first, then by text', () => {
  assert.equal(outcome('SOLVED', '#00ff88'), 'win');
  assert.equal(outcome('GAME OVER', '#ff3b3b'), 'fail');
  assert.equal(outcome('AI WINS', '#ff00a0'), 'fail');
  assert.equal(outcome('OUT OF BUDGET', '#ff3b3b'), 'fail');
  assert.equal(outcome('YOU WIN', '#ffe600'), 'win');
  assert.equal(outcome('ROUTE FOUND', '#00f5ff'), 'win');
  assert.equal(outcome('ADD CITIES', '#ffe600'), 'note');
});

test('waitFor passes results and errors through', async () => {
  assert.equal(await waitFor(Promise.resolve(7)), 7);
  await assert.rejects(waitFor(Promise.reject(new Error('boom'))), /boom/);
});
