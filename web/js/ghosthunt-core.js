// Ghost Hunt core: a line-for-line port of ghosthunt/maze.py, motion.py, sonar.py, hmm.py, particles.py,
// game.py and agent.py.
//
// Pure JavaScript with no DOM access, so the page and node (tests/test_ghosthunt_parity.py) can both load it.
// Every choice matches the Python reference: the same mulberry32 stream, the same neighbour order (N, E, S, W),
// the same tie-breaks (lowest index wins), and the same order of floating-point sums. The hidden state of a
// ghost is s = 4*k + h: open cell k and heading h. The sonar reading is a whole number from 0 to the maze
// diameter, drawn from a discrete Gaussian centred on the true distance.
//
// Why the filters live in the browser: the whole board has a few hundred hidden states, so the exact
// forward recursion costs well under a millisecond per turn, and the particle filter with a few hundred
// samples is also cheap. The server only serves the seeded maze and the metadata.

export const MODELS = ['random', 'lurker', 'patrol'];
export const ACTIONS = ['N', 'E', 'S', 'W', '.', 'bN', 'bE', 'bS', 'bW', 'b.'];
export const SIZES = [11, 15, 21];
export const NOISE = { low: 0.5, med: 1.0, high: 2.0 };
export const BUST_THRESHOLD = 0.5;
export const MIN_START_DISTANCE = 4;
export const TURN_CAP = 300;

const DR = [-1, 0, 1, 0];
const DC = [0, 1, 0, -1];
const DIR_OF = { N: 0, E: 1, S: 2, W: 3, '.': 4 };
const DIR_CHARS = 'NESW';
const STAY_P = 0.1;
const BETA = 1.0;
const STRAIGHT_P = 0.8;
const FLOOR = 1e-300;
const NEG = -Infinity;

// ── random numbers ───────────────────────────────────────────────────────

// mulberry32, as in bandits/rng.py. Math.imul keeps the low 32 bits of a product, like the Python mask.
export class Rng {
  constructor(seed) {
    this.a = seed >>> 0;
  }
  uniform() {
    this.a = (this.a + 0x6d2b79f5) >>> 0;
    let t = this.a;
    t = Math.imul(t ^ (t >>> 15), 1 | t) >>> 0;
    t = ((((t + (Math.imul(t ^ (t >>> 7), 61 | t) >>> 0)) >>> 0) ^ t) >>> 0);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  }
  index(k) {
    return Math.min(k - 1, Math.floor(this.uniform() * k));
  }
}

// ── maze ─────────────────────────────────────────────────────────────────

// A perfect maze on an odd grid by recursive backtracking (see maze.py for why this is a tree).
export function generate(n, seed) {
  if (n < 5 || n % 2 === 0) throw new Error('maze side must be an odd number, at least 5');
  const rng = new Rng(seed);
  const walls = new Array(n * n).fill(1);
  walls[n + 1] = 0; // room (1, 1)
  const stack = [[1, 1]];
  while (stack.length) {
    const [r, c] = stack[stack.length - 1];
    const options = [];
    for (let d = 0; d < 4; d++) {
      const r2 = r + 2 * DR[d], c2 = c + 2 * DC[d];
      if (r2 > 0 && r2 < n - 1 && c2 > 0 && c2 < n - 1 && walls[r2 * n + c2]) options.push([d, r2, c2]);
    }
    if (!options.length) {
      stack.pop();
      continue;
    }
    const [d, r2, c2] = options[rng.index(options.length)];
    walls[(r + DR[d]) * n + (c + DC[d])] = 0;
    walls[r2 * n + c2] = 0;
    stack.push([r2, c2]);
  }
  return walls;
}

// Open cells numbered 0..K-1 in row-major order, neighbour steps per direction, all-pairs BFS distances.
export class Maze {
  constructor(n, walls) {
    this.n = n;
    this.walls = Array.from(walls);
    this.cells = [];
    for (let c = 0; c < n * n; c++) if (!walls[c]) this.cells.push(c);
    this.K = this.cells.length;
    this.kOf = new Array(n * n).fill(-1);
    this.cells.forEach((c, k) => { this.kOf[c] = k; });
    this.step = this.cells.map((c) => {
      const r = Math.floor(c / n), col = c % n;
      const row = [];
      for (let d = 0; d < 4; d++) {
        const r2 = r + DR[d], c2 = col + DC[d];
        const inside = r2 >= 0 && r2 < n && c2 >= 0 && c2 < n;
        row.push(inside ? this.kOf[r2 * n + c2] : -1);
      }
      return row;
    });
    this.dist = this.cells.map((_, k) => this._bfs(k));
    this.diameter = Math.max(...this.dist.map((row) => Math.max(...row)));
    this.start = this.kOf[n + 1];
    this.cache = {};
  }
  _bfs(src) {
    const dist = new Array(this.K).fill(-1);
    dist[src] = 0;
    const queue = [src];
    for (let qi = 0; qi < queue.length; qi++) {
      const u = queue[qi];
      for (const v of this.step[u]) {
        if (v >= 0 && dist[v] < 0) {
          dist[v] = dist[u] + 1;
          queue.push(v);
        }
      }
    }
    return dist;
  }
}

// ── motion models (motion.py) ───────────────────────────────────────────

function randomMove(maze, k, h) {
  const opts = [];
  for (let d = 0; d < 4; d++) if (maze.step[k][d] >= 0) opts.push([d, maze.step[k][d]]);
  if (!opts.length) return [[4 * k + h, 1.0]];
  const out = [[4 * k + h, STAY_P]];
  const share = (1.0 - STAY_P) / opts.length;
  for (const [d, k2] of opts) out.push([4 * k2 + d, share]);
  return out;
}

function lurkerMove(maze, k, h, player) {
  const opts = [[4 * k + h, maze.dist[k][player]]];
  for (let d = 0; d < 4; d++) {
    const k2 = maze.step[k][d];
    if (k2 >= 0) opts.push([4 * k2 + d, maze.dist[k2][player]]);
  }
  const weights = opts.map(([, dist]) => Math.exp(-BETA * dist));
  let total = 0;
  for (const w of weights) total += w;
  return opts.map(([s2], i) => [s2, weights[i] / total]);
}

function patrolMove(maze, k, h) {
  const open = [false, false, false, false];
  for (let d = 0; d < 4; d++) open[d] = maze.step[k][d] >= 0;
  const back = (h + 2) % 4;
  const fwdOk = open[h];
  const turns = [];
  for (let d = 0; d < 4; d++) if (open[d] && d !== h && d !== back) turns.push(d);
  const backOk = open[back];
  const out = [];
  if (fwdOk) {
    if (turns.length) {
      out.push([4 * maze.step[k][h] + h, STRAIGHT_P]);
      for (const d of turns) out.push([4 * maze.step[k][d] + d, (1.0 - STRAIGHT_P) / turns.length]);
    } else if (backOk) {
      out.push([4 * maze.step[k][h] + h, STRAIGHT_P]);
      out.push([4 * maze.step[k][back] + back, 1.0 - STRAIGHT_P]);
    } else {
      out.push([4 * maze.step[k][h] + h, 1.0]);
    }
  } else if (turns.length) {
    for (const d of turns) out.push([4 * maze.step[k][d] + d, 1.0 / turns.length]);
  } else if (backOk) {
    out.push([4 * maze.step[k][back] + back, 1.0]);
  } else {
    out.push([4 * k + h, 1.0]);
  }
  return out;
}

export function transitions(maze, model, s, player) {
  const k = s >> 2, h = s & 3;
  if (model === 'random') return randomMove(maze, k, h);
  if (model === 'lurker') return lurkerMove(maze, k, h, player);
  if (model === 'patrol') return patrolMove(maze, k, h);
  throw new Error(`unknown motion model ${model}`);
}

// transitions() for every state; random and patrol are cached per maze, lurker is rebuilt per player cell.
export function motionTable(maze, model, player) {
  const S = 4 * maze.K;
  if (model !== 'lurker') {
    const key = `motion:${model}`;
    if (!maze.cache[key]) {
      maze.cache[key] = [];
      for (let s = 0; s < S; s++) maze.cache[key].push(transitions(maze, model, s, player));
    }
    return maze.cache[key];
  }
  const out = [];
  for (let s = 0; s < S; s++) out.push(transitions(maze, model, s, player));
  return out;
}

// ── sonar (sonar.py) ─────────────────────────────────────────────────────

export function emissionTable(sigma, R) {
  const rows = [];
  for (let d = 0; d <= R; d++) {
    const w = [];
    for (let r = 0; r <= R; r++) w.push(Math.exp(-((r - d) ** 2) / (2.0 * sigma * sigma)));
    let z = 0;
    for (const x of w) z += x;
    rows.push(w.map((x) => Math.max(x / z, FLOOR)));
  }
  return rows;
}

export function sampleReading(row, u) {
  let acc = 0;
  for (let r = 0; r < row.length; r++) {
    acc += row[r];
    if (u < acc) return r;
  }
  return row.length - 1;
}

// ── exact filter and Viterbi (hmm.py) ────────────────────────────────────

export class ExactFilter {
  constructor(maze, model, prior, emis) {
    this.maze = maze;
    this.model = model;
    this.emis = emis;
    this.belief = prior.slice();
  }
  predict(player) {
    const trans = motionTable(this.maze, this.model, player);
    const next = new Array(this.belief.length).fill(0);
    for (let s = 0; s < this.belief.length; s++) {
      const p = this.belief[s];
      if (p === 0) continue;
      for (const [s2, q] of trans[s]) next[s2] += p * q;
    }
    this.belief = next;
  }
  update(reading, player) {
    const dist = this.maze.dist, emis = this.emis;
    const next = this.belief.map((p, s) => p * emis[dist[s >> 2][player]][reading]);
    let total = 0;
    for (const x of next) total += x;
    if (total > 0) {
      this.belief = next.map((x) => x / total);
    } else { // numerical guard, as in hmm.py: restart from the sonar alone
      const like = this.belief.map((_, s) => emis[dist[s >> 2][player]][reading]);
      let z = 0;
      for (const x of like) z += x;
      this.belief = like.map((x) => x / z);
    }
  }
  marginal() {
    const out = new Array(this.maze.K).fill(0);
    for (let s = 0; s < this.belief.length; s++) out[s >> 2] += this.belief[s];
    return out;
  }
}

// The most likely hidden state at each time given all readings (Viterbi in log space, as in hmm.py).
export function viterbi(maze, model, prior, emis, players, readings) {
  const S = prior.length;
  const dist = maze.dist;
  const logEmis = (s, t) => Math.log(emis[dist[s >> 2][players[t]]][readings[t]]);
  let delta = prior.map((p, s) => (p > 0 ? Math.log(p) + logEmis(s, 0) : NEG));
  const back = [];
  for (let t = 1; t < readings.length; t++) {
    const trans = motionTable(maze, model, players[t]);
    const next = new Array(S).fill(NEG);
    const ptr = new Array(S).fill(-1);
    for (let s = 0; s < S; s++) {
      const d = delta[s];
      if (d === NEG) continue;
      for (const [s2, q] of trans[s]) {
        const cand = d + Math.log(q);
        if (cand > next[s2]) { // strict '>' keeps the lowest-index predecessor on ties
          next[s2] = cand;
          ptr[s2] = s;
        }
      }
    }
    for (let s2 = 0; s2 < S; s2++) if (next[s2] > NEG) next[s2] += logEmis(s2, t);
    delta = next;
    back.push(ptr);
  }
  let last = 0;
  for (let s = 1; s < S; s++) if (delta[s] > delta[last]) last = s;
  const path = [last];
  for (let i = back.length - 1; i >= 0; i--) path.push(back[i][path[path.length - 1]]);
  return path.reverse();
}

// ── particle filter (particles.py) ───────────────────────────────────────

function search(cum, u) {
  let lo = 0, hi = cum.length - 1;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (u < cum[mid]) hi = mid;
    else lo = mid + 1;
  }
  return lo;
}

export class ParticleFilter {
  constructor(maze, model, prior, emis, n, rng) {
    this.maze = maze;
    this.model = model;
    this.emis = emis;
    this.n = n;
    this.rng = rng;
    const states = [];
    for (let s = 0; s < prior.length; s++) if (prior[s] > 0) states.push(s);
    const cum = [];
    let acc = 0;
    for (const s of states) {
      acc += prior[s];
      cum.push(acc);
    }
    this.parts = [];
    for (let i = 0; i < n; i++) this.parts.push(states[search(cum, rng.uniform() * acc)]);
    this.w = new Array(n).fill(1.0 / n);
    this.resamples = 0;
  }
  predict(player) {
    const trans = motionTable(this.maze, this.model, player);
    for (let i = 0; i < this.n; i++) {
      const options = trans[this.parts[i]];
      if (options.length === 1) { // deterministic move: no draw, the stream stays aligned with Python
        this.parts[i] = options[0][0];
        continue;
      }
      const u = this.rng.uniform();
      let acc = 0, chosen = options[options.length - 1][0];
      for (const [s2, q] of options) {
        acc += q;
        if (u < acc) { chosen = s2; break; }
      }
      this.parts[i] = chosen;
    }
  }
  update(reading, player) {
    const dist = this.maze.dist, emis = this.emis;
    let w = this.w.map((wi, i) => wi * emis[dist[this.parts[i] >> 2][player]][reading]);
    let total = 0;
    for (const x of w) total += x;
    if (total > 0) {
      w = w.map((x) => x / total);
    } else {
      w = new Array(this.n).fill(1.0 / this.n);
    }
    let sq = 0;
    for (const x of w) sq += x * x;
    const ess = 1.0 / sq;
    if (ess < this.n / 2.0) this._resample(w);
    else this.w = w;
  }
  _resample(w) {
    const n = this.n;
    const cum = [];
    let acc = 0;
    for (const x of w) {
      acc += x;
      cum.push(acc);
    }
    const u0 = this.rng.uniform() / n;
    const next = [];
    for (let i = 0; i < n; i++) next.push(this.parts[search(cum, u0 + i / n)]);
    this.parts = next;
    this.w = new Array(n).fill(1.0 / n);
    this.resamples += 1;
  }
  marginal() {
    const out = new Array(this.maze.K).fill(0);
    for (let i = 0; i < this.n; i++) out[this.parts[i] >> 2] += this.w[i];
    return out;
  }
}

// ── the board (game.py) and the autopilot (agent.py) ─────────────────────

function pick(options, u) {
  let acc = 0;
  for (const [s2, q] of options) {
    acc += q;
    if (u < acc) return s2;
  }
  return options[options.length - 1][0];
}

export class Game {
  constructor({ size = 15, ghosts = 2, model = 'random', sigma = 1.0, seed = 1, filt = 'exact',
                particles = 200, maxTurns = TURN_CAP } = {}) {
    this.size = size;
    this.model = model;
    this.sigma = sigma;
    this.seed = seed;
    this.filt = filt;
    this.particles = particles;
    this.maxTurns = maxTurns;
    this.maze = new Maze(size, generate(size, seed));
    this.emis = emissionTable(sigma, this.maze.diameter);
    const eligible = [];
    for (let k = 0; k < this.maze.K; k++) {
      if (this.maze.dist[k][this.maze.start] >= MIN_START_DISTANCE) for (let h = 0; h < 4; h++) eligible.push(4 * k + h);
    }
    this.eligible = eligible;
    this.prior = new Array(4 * this.maze.K).fill(0);
    for (const s of eligible) this.prior[s] = 1.0 / eligible.length;
    this.p = this.maze.start;
    this.players = [this.p];
    this.t = 0;
    this.ghosts = [];
    for (let g = 0; g < ghosts; g++) {
      const rng = new Rng(seed + 7919 * (g + 1));
      const x = eligible[rng.index(eligible.length)];
      const f = filt === 'exact'
        ? new ExactFilter(this.maze, model, this.prior, this.emis)
        : new ParticleFilter(this.maze, model, this.prior, this.emis, particles,
                             new Rng(seed + 7919 * (g + 1) + 0x10000));
      this.ghosts.push({ id: g, x, rng, filter: f, live: true, bustTurn: null, true: [x >> 2], readings: [] });
    }
    for (const gh of this.ghosts) this._ping(gh);
  }

  get done() {
    return this.ghosts.every((gh) => !gh.live) || this.t >= this.maxTurns;
  }
  get busted() {
    return this.ghosts.filter((gh) => !gh.live).length;
  }

  legalActions() {
    const out = ['.', 'b.'];
    for (let d = 0; d < 4; d++) {
      if (this.maze.step[this.p][d] >= 0) out.push(DIR_CHARS[d], 'b' + DIR_CHARS[d]);
    }
    return out;
  }

  act(action) {
    if (this.done) throw new Error('the game is over');
    if (!ACTIONS.includes(action)) throw new Error(`unknown action ${action}`);
    let p = this.p;
    const busted = [];
    if (action[0] === 'b') {
      const d = DIR_OF[action.slice(1)];
      const target = d === 4 ? p : this.maze.step[p][d];
      if (target < 0) throw new Error('no open cell in that direction');
      for (const gh of this.ghosts) {
        if (gh.live && (gh.x >> 2) === target) {
          gh.live = false;
          gh.bustTurn = this.t;
          busted.push(gh.id);
        }
      }
    } else {
      const d = DIR_OF[action];
      if (d !== 4) {
        const q = this.maze.step[p][d];
        if (q < 0) throw new Error('that way is a wall');
        p = q;
      }
    }
    this.p = p;
    this.t += 1;
    this.players.push(p);
    const readings = {};
    for (const gh of this.ghosts) {
      if (!gh.live) continue;
      gh.x = pick(transitions(this.maze, this.model, gh.x, p), gh.rng.uniform());
      gh.true.push(gh.x >> 2);
      readings[gh.id] = this._ping(gh);
    }
    return { t: this.t, action, busted, readings };
  }

  _ping(gh) {
    const d = this.maze.dist[gh.x >> 2][this.p];
    const r = sampleReading(this.emis[d], gh.rng.uniform());
    gh.readings.push(r);
    if (this.t > 0) gh.filter.predict(this.p);
    gh.filter.update(r, this.p);
    return r;
  }

  marginal(gid) {
    return this.ghosts[gid].filter.marginal();
  }

  viterbiPath(gid) {
    const gh = this.ghosts[gid];
    const n = gh.readings.length;
    return viterbi(this.maze, this.model, this.prior, this.emis, this.players.slice(0, n), gh.readings)
      .map((s) => s >> 2);
  }

  trace(gid) {
    const gh = this.ghosts[gid];
    return { true: gh.true.slice(), viterbi: this.viterbiPath(gid), readings: gh.readings.slice(), bustTurn: gh.bustTurn };
  }
}

function argmax(values, skip = -1) {
  let best = -1, bestI = -1;
  for (let i = 0; i < values.length; i++) {
    if (i !== skip && values[i] > best) {
      best = values[i];
      bestI = i;
    }
  }
  return bestI;
}

// Direction of the first step of a shortest corridor path from p to target (-1 if none).
export function firstStep(maze, p, target) {
  const here = maze.dist[p][target];
  for (let d = 0; d < 4; d++) {
    const q = maze.step[p][d];
    if (q >= 0 && maze.dist[q][target] === here - 1) return d;
  }
  return -1;
}

// The autopilot: bust the most confident peak when it is next to us, otherwise walk toward it.
export function choose(game, threshold = BUST_THRESHOLD) {
  const maze = game.maze, p = game.p;
  let best = null;
  for (const gh of game.ghosts) {
    if (!gh.live) continue;
    const marg = game.marginal(gh.id);
    const k = argmax(marg);
    if (best === null || marg[k] > best.conf) best = { conf: marg[k], k, marg };
  }
  if (best === null) return '.';
  const { conf, marg } = best;
  let k = best.k;
  if (conf > threshold) {
    if (k === p) return 'b.';
    const d = firstStep(maze, p, k);
    if (d >= 0 && maze.step[p][d] === k) return 'b' + DIR_CHARS[d];
  }
  if (k === p) k = argmax(marg, p); // on the peak but unsure: look from the second-best cell
  const d = firstStep(maze, p, k);
  if (d < 0) return '.';
  return DIR_CHARS[d];
}

// Exact forward beliefs for a known list of players and readings (the same as hmm.py forward()).
export function forward(maze, model, prior, emis, players, readings) {
  const f = new ExactFilter(maze, model, prior, emis);
  f.update(readings[0], players[0]);
  const out = [f.belief.slice()];
  for (let t = 1; t < readings.length; t++) {
    f.predict(players[t]);
    f.update(readings[t], players[t]);
    out.push(f.belief.slice());
  }
  return out;
}
