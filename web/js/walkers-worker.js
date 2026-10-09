// Web Worker for EVOLVING WALKERS: runs the genetic algorithm off the main thread so the arena stays at 60 fps.
//
// Messages in:  {cmd:'init'|'reset', seed, popSize, mutationRate, world}   start a new run
//               {cmd:'step', topK}                                          evaluate and breed one generation
//               {cmd:'set', mutationRate, world}                            change settings from the next generation
//               {cmd:'inject', code}                                        queue a user creature for the next generation
// Messages out: {type:'ready', popSize}
//               {type:'gen', summary, top, records}   summary of the scored generation; top: frames of the best four
//                                                       (transferred); records: lineage entries new since the last message
//               {type:'error', message}
import { Evolver, decode, DEFAULT_WORLD, GA_DEFAULTS } from './walkers-core.js';

let ev = null;
let sent = 0; // index into ev.records of the first record not yet posted

function start(m) {
  ev = new Evolver({
    seed: m.seed,
    popSize: m.popSize || GA_DEFAULTS.popSize,
    mutationRate: m.mutationRate ?? GA_DEFAULTS.mutationRate,
    world: { ...DEFAULT_WORLD, ...(m.world || {}) },
  });
  sent = 0;
  self.postMessage({ type: 'ready', popSize: ev.popSize });
}

self.onmessage = (e) => {
  const m = e.data;
  try {
    if (m.cmd === 'init' || m.cmd === 'reset') {
      start(m);
    } else if (m.cmd === 'set') {
      if (m.mutationRate !== undefined) ev.mutationRate = m.mutationRate;
      if (m.world) ev.world = { ...ev.world, ...m.world };
    } else if (m.cmd === 'inject') {
      ev.inject(decode(m.code));
    } else if (m.cmd === 'step') {
      const summary = ev.step(m.topK || 4);
      const top = summary.top.map((t) => ({ id: t.id, fitness: t.fitness, N: t.N, frameCount: t.frameCount, sampleEvery: t.sampleEvery, frames: t.frames }));
      delete summary.top;
      const records = ev.records.slice(sent);
      sent = ev.records.length;
      const transfer = top.map((t) => t.frames.buffer);
      self.postMessage({ type: 'gen', summary, top, records }, transfer);
    }
  } catch (err) {
    self.postMessage({ type: 'error', message: String((err && err.message) || err) });
  }
};
