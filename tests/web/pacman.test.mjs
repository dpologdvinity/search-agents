import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { runInNewContext } from 'node:vm';

// Exercise the actual page controller with a fake clock and server, without a browser dependency.
const source = readFileSync(new URL('../../web/js/pacman.js', import.meta.url), 'utf8')
  .replace(/^import .*;$/gm, '').replace(/\ninit\(\);\s*$/, '');

function page() {
  const elements = new Map();
  const timers = new Map();
  const requests = [];
  const el = (id) => {
    if (!elements.has(id)) elements.set(id, {
      value: ({ mode: 'play', movement: 'continuous', speed: '240', maze: 'lanes', seed: '7' })[id] ?? '',
      textContent: '', getContext: () => ({}), querySelector() { return this; },
    });
    return elements.get(id);
  };
  let nextId = 0;
  const context = {
    document: { getElementById: el, hidden: false },
    performance: { now: () => 1000 },
    setTimeout(fn, ms) { const id = ++nextId; timers.set(id, { fn, ms }); return id; },
    clearTimeout(id) { timers.delete(id); },
    postJSON: async (path, body) => {
      requests.push(body);
      return { state: { pac: [3, 7 + body.actions.filter(a => a === 'E').length],
        ghosts: [], pellets: [], points: 0, turn: body.actions.length, status: 'playing' },
        legal: ['E', 'W', 'WAIT'], last: null };
    },
  };
  runInNewContext(source + `
    updateAll = () => {};
    globalThis.pm = { S, inputMove, tickHuman, pauseHuman, humanMove };
  `, context);
  Object.assign(context.pm.S, { pac: [3, 7], status: 'playing', legal: ['E', 'W', 'WAIT'] });
  return { ...context.pm, el, timers, requests, context };
}
const settle = async () => { for (let i = 0; i < 6; i++) await Promise.resolve(); };

async function nextTick(p) {
  const [id, timer] = p.timers.entries().next().value;
  p.timers.delete(id);
  await timer.fn();
}

test('continuous play repeats moves, buffers turns, and stops Pac-Man while ghosts keep ticking', async () => {
  const p = page();
  p.inputMove('E');
  await settle();
  assert.equal(p.S.turn, 1);
  assert.equal(p.timers.size, 1);
  p.inputMove('N'); // blocked turn: keep moving east until the turn opens
  await nextTick(p);
  assert.equal(p.requests.at(-1).actions.at(-1), 'E');
  p.S.legal.push('N');
  await nextTick(p);
  assert.equal(p.requests.at(-1).actions.at(-1), 'N');
  p.inputMove('WAIT');
  const pac = [...p.S.pac];
  await nextTick(p);
  assert.equal(p.requests.at(-1).actions.at(-1), 'WAIT');
  assert.deepEqual([...p.S.pac], pac);
  assert.equal(p.timers.size, 1);
  p.pauseHuman();
  assert.equal(p.timers.size, 0);
  assert.equal(p.S.humanRunning, false);
});

test('step mode advances only once and schedules no timer', async () => {
  const p = page();
  p.el('movement').value = 'step';
  p.inputMove('WAIT');
  await settle();
  assert.equal(p.S.turn, 1);
  assert.equal(p.timers.size, 0);
});

test('continuous requests are serialized and speed stays within the API rate limit', async () => {
  const p = page();
  p.el('speed').value = '60';
  p.inputMove('E');
  p.inputMove('W');
  await settle();
  assert.equal(p.requests.length, 1);
  assert.equal([...p.timers.values()][0].ms, 200);
  await nextTick(p);
  assert.equal(p.requests.length, 2);
  assert.equal(p.requests.at(-1).actions.at(-1), 'W');
});

test('stale responses cannot overwrite a replacement board', async () => {
  const p = page();
  let resolve;
  p.context.postJSON = () => new Promise(r => { resolve = r; });
  const move = p.humanMove('E');
  p.S.generation++;
  resolve({ state: { pac: [9, 9] } });
  assert.equal(await move, false);
  assert.deepEqual([...p.S.pac], [3, 7]);
});

test('a failed request pauses continuous play instead of retrying in a loop', async () => {
  const p = page();
  p.context.postJSON = async () => { throw new Error('server busy'); };
  runInNewContext('log = () => {};', p.context);
  p.inputMove('E');
  await settle();
  assert.equal(p.S.humanRunning, false);
  assert.equal(p.S.busy, false);
  assert.equal(p.timers.size, 0);
});
