import { test } from 'node:test';
import assert from 'node:assert/strict';
import { play, setEnabled } from '../../web/js/sfx.js';

test('Pac-Man voices schedule audio when enabled and stay silent when disabled', () => {
  const notes = [];
  const param = () => ({ value: 0, setValueAtTime() {}, exponentialRampToValueAtTime() {} });
  const node = () => ({ connect() { return this; }, gain: param(), frequency: param(),
    Q: param(), delayTime: param(), detune: param() });
  globalThis.AudioContext = class {
    state = 'running'; currentTime = 0; destination = {};
    createGain = node; createDynamicsCompressor = node; createDelay = node; createBiquadFilter = node;
    createOscillator() {
      const osc = { ...node(), start() { notes.push(this); }, stop() {} };
      osc.frequency.setValueAtTime = f => { osc.from = f; };
      osc.frequency.exponentialRampToValueAtTime = f => { osc.to = f; };
      return osc;
    }
    suspend() { this.state = 'suspended'; }
  };
  try {
    setEnabled(true);
    notes.length = 0;
    assert.equal(play('pacChomp', 0), true);
    assert.equal(notes[0].type, 'triangle');
    assert.equal(notes[0].from, 900);
    assert.equal(notes[0].to, 450);
    for (const voice of ['pacPower', 'pacGhost', 'pacDeath', 'pacWin']) {
      const before = notes.length;
      assert.equal(play(voice), true);
      assert.ok(notes.length > before, `${voice} schedules notes`);
    }
    setEnabled(false);
    notes.length = 0;
    assert.equal(play('pacChomp'), false);
    assert.equal(notes.length, 0);
  } finally {
    delete globalThis.AudioContext;
  }
});
