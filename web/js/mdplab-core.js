// MDP lab core: the grid as a Markov decision process, the two planners, and the two learners.
//
// This is a port of the Python package `mdplab` (mdplab/grid.py, mdp.py, solve.py, learn.py). It has no DOM
// code, so the same numbers come out of Python and JavaScript for the same layout, parameters and seed. The
// ports keep the same operations in the same order (sums over the same transition lists, the same tie rule,
// the same mulberry32 draws), so values agree to the last bit and trajectories are identical.
//
// Cell kinds: '.' empty, '#' wall, 'S' start, '+' reward (terminal), '-' pit (terminal), 'C' cliff (falling pays
// the cliff value and sends the agent back to S). Actions are 0 up, 1 right, 2 down, 3 left.

export const ACTIONS = ['up', 'right', 'down', 'left'];
export const DELTAS = [[0, -1], [1, 0], [0, 1], [-1, 0]];
export const ARROWS = ['↑', '→', '↓', '←'];
// Action values within TIE of the best count as tied. Same constant as mdplab/grid.py.
export const TIE = 1e-9;

export const DEFAULT_PARAMS = {
  gamma: 0.95, slip: 0.1, living: -0.04, reward: 1.0, pit: -1.0, cliff: -100.0,
  alpha: 0.1, epsilon: 0.1, decay: 0.999, eps_min: 0.01, max_steps: 500,
};

// Presets, mirroring mdplab/grid.py PRESETS. server/mdplab_api.py serves the Python copy; a parity test checks
// the two lists are equal.
export const PRESETS = [
  {
    key: 'cliff', label: 'Cliff walk',
    blurb: "Sutton and Barto's cliff. Walk along the top edge or along the cliff? Q-learning hugs the edge; SARSA keeps away from it.",
    rows: ['............', '............', '............', 'SCCCCCCCCCC+'],
    params: { gamma: 0.99, slip: 0.0, living: -1.0, reward: 0.0, pit: -1.0, cliff: -100.0,
              alpha: 0.5, epsilon: 0.1, decay: 1.0, eps_min: 0.1, max_steps: 500 },
  },
  {
    key: 'rooms', label: 'Four rooms',
    blurb: 'Four rooms joined by doorways, a goal in the far corner and a pit by the door. Wind on.',
    rows: ['###########', '#....#...+#', '#....#....#', '#.........#', '#....#....#', '##.#####.##',
           '#....#....#', '#.........#', '#....#..-.#', '#S...#....#', '###########'],
    params: { gamma: 0.95, slip: 0.1, living: -0.04, reward: 1.0, pit: -1.0, cliff: -100.0,
              alpha: 0.1, epsilon: 0.1, decay: 0.999, eps_min: 0.01, max_steps: 500 },
  },
  {
    key: 'maze', label: 'Maze with a shortcut',
    blurb: 'A maze whose shortest route runs through a pit. The lab finds the long safe way round.',
    rows: ['#########', '#S..#...#', '#.#.#.#.#', '#.#...#.#', '#.#####.#', '#....-..#', '###.#.#.#',
           '#...#.#+#', '#########'],
    params: { gamma: 0.95, slip: 0.1, living: -0.04, reward: 1.0, pit: -1.0, cliff: -100.0,
              alpha: 0.1, epsilon: 0.1, decay: 0.999, eps_min: 0.01, max_steps: 500 },
  },
  {
    key: 'windy', label: 'Windy corridor',
    blurb: 'A corridor with pits in its middle row. The wind (slip 0.3) pushes you sideways into them.',
    rows: ['##############', '#S...........#', '#.-.-.-.-.-.+#', '#............#', '##############'],
    params: { gamma: 0.95, slip: 0.3, living: -0.04, reward: 1.0, pit: -1.0, cliff: -100.0,
              alpha: 0.1, epsilon: 0.1, decay: 0.999, eps_min: 0.01, max_steps: 500 },
  },
];

export function presetByKey(key) {
  const p = PRESETS.find((x) => x.key === key);
  if (!p) throw new Error(`unknown preset ${key}`);
  return p;
}

// ── rng.py: mulberry32, the same stream as bandits/rng.py ────────────────

const TWO32 = 4294967296;

export class Rng {
  constructor(seed) {
    this.state = seed >>> 0; // seed modulo 2**32, as in Python
  }

  // A double in [0, 1): one mulberry32 step. Math.imul keeps the low 32 bits, like the Python masks.
  uniform() {
    this.state = (this.state + 0x6D2B79F5) >>> 0;
    let t = this.state;
    t = Math.imul(t ^ (t >>> 15), 1 | t) >>> 0;
    t = ((t + (Math.imul(t ^ (t >>> 7), 61 | t) >>> 0)) >>> 0) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / TWO32;
  }

  // A uniform integer in [0, k).
  index(k) {
    return Math.min(k - 1, Math.trunc(this.uniform() * k));
  }
}

// ── mdp.py: the transition table ─────────────────────────────────────────

export function validateParams(p) {
  if (!(p.gamma >= 0 && p.gamma <= 1)) throw new Error('gamma must be in [0, 1]');
  if (!(p.slip >= 0 && p.slip <= 0.5)) throw new Error('slip must be in [0, 0.5]');
  if (!(p.alpha > 0 && p.alpha <= 1)) throw new Error('alpha must be in (0, 1]');
  if (!(p.epsilon >= 0 && p.epsilon <= 1)) throw new Error('epsilon must be in [0, 1]');
  if (!(p.decay > 0 && p.decay <= 1)) throw new Error('decay must be in (0, 1]');
  if (!(p.eps_min >= 0 && p.eps_min <= 1)) throw new Error('eps_min must be in [0, 1]');
  if (!(p.max_steps >= 1)) throw new Error('max_steps must be at least 1');
}

export class GridModel {
  constructor(rows, overrides = {}) {
    if (!rows.length) throw new Error('a grid needs at least one row');
    const W = rows[0].length;
    if (W === 0 || rows.some((r) => r.length !== W)) throw new Error('every row must have the same non-zero length');
    const kinds = rows.join('');
    if (/[^.#SC+\-]/.test(kinds)) throw new Error('unknown cell characters');
    if (kinds.split('S').length !== 2) throw new Error('a grid needs exactly one start cell (S)');

    this.params = { ...DEFAULT_PARAMS, ...overrides };
    validateParams(this.params);
    const p = this.params;
    this.rows = rows.slice();
    this.W = W;
    this.H = rows.length;
    this.n = W * this.H;
    this.kinds = kinds;
    this.start = kinds.indexOf('S');
    this.terminal = [];
    this.isState = [];
    this.states = [];
    this.cellPay = [];
    for (let i = 0; i < this.n; i++) {
      const k = kinds[i];
      this.terminal.push(k === '+' || k === '-');
      this.isState.push(k !== '#' && k !== '+' && k !== '-' && k !== 'C');
      if (this.isState[i]) this.states.push(i);
      // Payout for entering a cell. Empty cells and the start pay nothing.
      this.cellPay.push(k === '+' ? p.reward : k === '-' ? p.pit : k === 'C' ? p.cliff : 0.0);
    }
    this.res = new Array(this.n).fill(null);
    this.P = new Array(this.n).fill(null);
    for (const s of this.states) {
      this.res[s] = [0, 1, 2, 3].map((d) => this.resolve(s, d));
    }
    for (const s of this.states) {
      this.P[s] = [0, 1, 2, 3].map((a) => this.transitions(s, a));
    }
  }

  // Where a deterministic move from s in direction d ends up, and the payout for the cell it enters.
  resolve(s, d) {
    const W = this.W;
    const x = s % W;
    const y = Math.floor(s / W);
    const nx = x + DELTAS[d][0];
    const ny = y + DELTAS[d][1];
    // Off the edge or into a wall: the agent stays where it is, and pays nothing extra.
    if (!(nx >= 0 && nx < W && ny >= 0 && ny < this.H) || this.kinds[ny * W + nx] === '#') return [s, 0.0];
    const t = ny * W + nx;
    if (this.kinds[t] === 'C') return [this.start, this.cellPay[t]];
    return [t, this.cellPay[t]];
  }

  // The [probability, next state, reward] triples for action a from s. Intended direction first, then the
  // left slip, then the right slip. Zero-probability entries are dropped.
  transitions(s, a) {
    const p = this.params.slip;
    const outcomes = [[1.0 - p, a], [p / 2.0, (a + 3) % 4], [p / 2.0, (a + 1) % 4]];
    const out = [];
    for (const [prob, d] of outcomes) {
      if (prob <= 0.0) continue;
      const [dest, pay] = this.res[s][d];
      out.push([prob, dest, this.params.living + pay]);
    }
    return out;
  }

  // The direction actually moved for intended action a and a uniform draw u in [0, 1).
  directionFor(a, u) {
    const p = this.params.slip;
    if (u < 1.0 - p) return a;
    if (u < 1.0 - p / 2.0) return (a + 3) % 4;
    return (a + 1) % 4;
  }
}

// ── solve.py: exact planning ─────────────────────────────────────────────

// Q(s, a) for all four actions from values V. Terminal next states have V = 0 because they are never updated.
export function backup(m, V, s) {
  const g = m.params.gamma;
  const row = [];
  for (let a = 0; a < 4; a++) {
    let q = 0.0;
    for (const [prob, dest, r] of m.P[s][a]) q += prob * (r + g * V[dest]);
    row.push(q);
  }
  return row;
}

// The lowest-index action within TIE of the best.
export function bestAction(qs) {
  const top = Math.max(...qs);
  for (let a = 0; a < qs.length; a++) if (qs[a] >= top - TIE) return a;
  return 0;
}

export function greedyPolicy(m, V) {
  const pi = new Array(m.n).fill(-1);
  for (const s of m.states) pi[s] = bestAction(backup(m, V, s));
  return pi;
}

export function qTable(m, V) {
  const table = [];
  for (let i = 0; i < m.n; i++) table.push([0, 0, 0, 0]);
  for (const s of m.states) table[s] = backup(m, V, s);
  return table;
}

// States the agent reaches from the start by following pi (any slip with positive probability counts).
export function reachable(m, pi) {
  const seen = new Set([m.start]);
  const stack = [m.start];
  while (stack.length) {
    const s = stack.pop();
    for (const [, dest] of m.P[s][pi[s]]) {
      if (!seen.has(dest) && !m.terminal[dest]) {
        seen.add(dest);
        stack.push(dest);
      }
    }
  }
  return [...seen].sort((x, y) => x - y);
}

// Value iteration, one synchronous sweep at a time, from V = 0.
export class ValueIterator {
  constructor(model) {
    this.m = model;
    this.V = new Array(model.n).fill(0.0);
    this.sweeps = 0;
    this.residual = Infinity;
  }

  sweep() {
    const old = this.V;
    const next = old.slice();
    let res = 0.0;
    for (const s of this.m.states) {
      const best = Math.max(...backup(this.m, old, s));
      res = Math.max(res, Math.abs(best - old[s]));
      next[s] = best;
    }
    this.V = next;
    this.sweeps += 1;
    this.residual = res;
    return res;
  }

  solve(eps, maxSweeps = 100000) {
    while (this.residual >= eps && this.sweeps < maxSweeps) this.sweep();
    return this.sweeps;
  }

  policy() {
    return greedyPolicy(this.m, this.V);
  }
}

// Policy iteration: evaluate the current policy, then improve it greedily. Starts from a seeded random policy.
export class PolicyIterator {
  constructor(model, seed = 1) {
    this.m = model;
    this.V = new Array(model.n).fill(0.0);
    this.pi = new Array(model.n).fill(-1);
    const rng = new Rng(seed);
    for (const s of model.states) this.pi[s] = rng.index(4);
    this.evalSweeps = 0;
    this.rounds = 0;
    this.residual = Infinity;
    this.stable = false;
    this.lastChanged = -1;
  }

  evalSweep() {
    const g = this.m.params.gamma;
    const old = this.V;
    const next = old.slice();
    let res = 0.0;
    for (const s of this.m.states) {
      let q = 0.0;
      for (const [prob, dest, r] of this.m.P[s][this.pi[s]]) q += prob * (r + g * old[dest]);
      res = Math.max(res, Math.abs(q - old[s]));
      next[s] = q;
    }
    this.V = next;
    this.evalSweeps += 1;
    this.residual = res;
    return res;
  }

  // Evaluate until the residual is below eps. Returns the sweeps this phase used.
  evaluate(eps, maxSweeps = 5000) {
    let used = 0;
    this.residual = Infinity;
    while (this.residual >= eps && used < maxSweeps) {
      this.evalSweep();
      used += 1;
    }
    return used;
  }

  // Greedy improvement. A state changes action only when the new one beats the current by more than TIE.
  improve() {
    let changed = 0;
    for (const s of this.m.states) {
      const qs = backup(this.m, this.V, s);
      const top = Math.max(...qs);
      if (qs[this.pi[s]] >= top - TIE) continue;
      this.pi[s] = bestAction(qs);
      changed += 1;
    }
    this.rounds += 1;
    this.lastChanged = changed;
    this.stable = changed === 0;
    return changed;
  }

  // Run evaluate and improve rounds until the policy is stable. Returns a summary.
  solve(eps, maxRounds = 500, maxSweeps = 5000) {
    let capped = false;
    while (!this.stable && this.rounds < maxRounds) {
      const used = this.evaluate(eps, maxSweeps);
      capped = capped || used >= maxSweeps;
      this.improve();
    }
    return { rounds: this.rounds, evalSweeps: this.evalSweeps, stable: this.stable, capped };
  }
}

// ── learn.py: learning from episodes ─────────────────────────────────────

// Tabular Q-learning ('q') or SARSA ('sarsa'), epsilon-greedy, one step at a time. Random draws follow the
// Python order exactly: exploration test (and index), slip draw, then tie-break index on the greedy branch.
export class QLearner {
  constructor(model, opts = {}) {
    const p = model.params;
    this.m = model;
    this.algo = opts.algo || 'q';
    if (this.algo !== 'q' && this.algo !== 'sarsa') throw new Error("algo must be 'q' or 'sarsa'");
    this.alpha = opts.alpha ?? p.alpha;
    this.epsilon0 = opts.epsilon ?? p.epsilon;
    this.decay = opts.decay ?? p.decay;
    this.epsMin = opts.eps_min ?? p.eps_min;
    this.maxSteps = opts.max_steps ?? p.max_steps;
    if (!(this.alpha > 0 && this.alpha <= 1)) throw new Error('alpha must be in (0, 1]');
    this.rng = new Rng(opts.seed ?? 1);
    this.Q = [];
    for (let i = 0; i < model.n; i++) this.Q.push([0, 0, 0, 0]);
    this.eps = this.epsilon0;
    this.episodes = 0;
    this.returns = [];
    this.last = null;
  }

  greedy(s) {
    const qs = this.Q[s];
    const top = Math.max(...qs);
    for (let a = 0; a < 4; a++) if (qs[a] >= top - TIE) return a;
    return 0;
  }

  choose(s) {
    if (this.rng.uniform() < this.eps) return this.rng.index(4);
    const qs = this.Q[s];
    const top = Math.max(...qs);
    const ties = [];
    for (let a = 0; a < 4; a++) if (qs[a] >= top - TIE) ties.push(a);
    if (ties.length === 1) return ties[0];
    return ties[this.rng.index(ties.length)];
  }

  // One episode from the start until a terminal cell or maxSteps. Learns on every step.
  episode() {
    const m = this.m;
    const p = m.params;
    const g = p.gamma;
    const Q = this.Q;
    let s = m.start;
    let a = this.choose(s);
    let ret = 0.0;
    let steps = 0;
    const path = [s];
    let ended = false;
    for (;;) {
      const d = m.directionFor(a, this.rng.uniform());
      const [dest, pay] = m.res[s][d];
      const r = p.living + pay;
      ret += r;
      steps += 1;
      path.push(dest);
      const done = m.terminal[dest];
      let aNext = -1;
      let target;
      if (done) {
        target = r;
      } else if (this.algo === 'sarsa') {
        aNext = this.choose(dest);
        target = r + g * Q[dest][aNext];
      } else {
        target = r + g * Math.max(...Q[dest]);
      }
      Q[s][a] += this.alpha * (target - Q[s][a]);
      if (done || steps >= this.maxSteps) {
        ended = done;
        break;
      }
      s = dest;
      a = this.algo === 'sarsa' ? aNext : this.choose(s);
    }
    this.eps = Math.max(this.epsMin, this.eps * this.decay);
    this.episodes += 1;
    this.returns.push(ret);
    const ep = { ret, steps, path, reachedTerminal: ended, fell: ended && m.kinds[path[path.length - 1]] === '-' };
    this.last = ep;
    return ep;
  }

  run(count) {
    const out = [];
    for (let i = 0; i < count; i++) out.push(this.episode().ret);
    return out;
  }

  policy() {
    const pi = new Array(this.m.n).fill(-1);
    for (const s of this.m.states) pi[s] = this.greedy(s);
    return pi;
  }

  // Share of states the optimal policy reaches where the learned greedy action is optimal (ties count).
  agreement(optimalPi, optimalQ, slack = 1e-6) {
    const scope = reachable(this.m, optimalPi);
    const flags = new Array(this.m.n).fill(false);
    let ok = 0;
    for (const s of scope) {
      const g = this.greedy(s);
      const best = Math.max(...optimalQ[s]);
      const same = g === optimalPi[s] || optimalQ[s][g] >= best - slack;
      flags[s] = same;
      if (same) ok += 1;
    }
    return { frac: scope.length ? ok / scope.length : 1.0, flags, scope };
  }
}
