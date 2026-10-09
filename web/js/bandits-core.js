// Bandits core: the casino and the agents, ported from the Python package `bandits`.
//
// This file has no DOM code, so the same numbers come out of Python and JavaScript for a seed.
// The ports keep the same operations in the same order: mulberry32 with Math.imul, the Marsaglia
// polar normal, Marsaglia-Tsang gammas for Beta draws, and the outcome table rolled pull by pull.
// Machine means are not drawn here; the server (GET /api/bandits/machines) is the source of truth
// for them, and the constants come from GET /api/bandits/meta.

// Defaults; the page overrides them with meta.constants so the two sides cannot drift apart.
export const CONSTANTS = {
  sigma: 0.2, drift_period: 500, gap: 0.05, low: 0.1, high: 0.9,
  reward_offset: 1000003, agent_offset: 2000029, window: 200, window_xi: 0.6, last_pulls: 1000,
};

const TWO32 = 4294967296;

// ── rng.py ───────────────────────────────────────────────────────────────

export class Rng {
  constructor(seed) {
    this.state = seed >>> 0; // seed modulo 2**32, as in Python
  }

  // A double in [0, 1): one mulberry32 step. Math.imul keeps only the low 32 bits, like the Python masks.
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

  // A standard normal by the polar method.
  normal() {
    let u, v, s;
    for (;;) {
      u = 2.0 * this.uniform() - 1.0;
      v = 2.0 * this.uniform() - 1.0;
      s = u * u + v * v;
      if (s > 0.0 && s < 1.0) break;
    }
    return u * Math.sqrt(-2.0 * Math.log(s) / s);
  }

  // Gamma(shape, 1) for shape >= 1, by Marsaglia and Tsang.
  gamma(shape) {
    if (shape < 1.0) throw new Error('gamma is implemented for shape >= 1 only');
    const d = shape - 1.0 / 3.0;
    const c = 1.0 / Math.sqrt(9.0 * d);
    for (;;) {
      const x = this.normal();
      let v = 1.0 + c * x;
      if (v <= 0.0) continue;
      v = v * v * v;
      const u = this.uniform();
      if (u <= 0.0) continue;
      if (u < 1.0 - 0.0331 * x * x * x * x) return d * v;
      if (Math.log(u) < 0.5 * x * x + d * (1.0 - v + Math.log(v))) return d * v;
    }
  }

  // Beta(a, b) for a, b >= 1.
  beta(a, b) {
    const x = this.gamma(a);
    const y = this.gamma(b);
    return x / (x + y);
  }
}

// ── env.py ───────────────────────────────────────────────────────────────

// Casino: the machine schedule from the server plus the outcome table rolled from the seed.
// `machines` is the JSON from GET /api/bandits/machines: {kind, k, seed, pulls, period, schedule}.
export class Casino {
  constructor(machines, constants = CONSTANTS) {
    this.kind = machines.kind;
    this.k = machines.k;
    this.seed = machines.seed;
    this.horizon = machines.pulls;
    this.period = machines.period;
    this.schedule = machines.schedule;
    this.sigma = this.kind === 'gaussian' ? constants.sigma : 1.0;
    // One luck value per pull and machine, rolled pull-major, exactly as env.py does.
    const rng = new Rng(this.seed + constants.reward_offset);
    const n = this.horizon * this.k;
    this.luck = new Float64Array(n);
    const gaussian = this.kind === 'gaussian';
    for (let i = 0; i < n; i++) this.luck[i] = gaussian ? rng.normal() : rng.uniform();
  }

  mean(t) {
    return this.schedule[Math.min(Math.floor(t / this.period), this.schedule.length - 1)];
  }

  best(t) {
    return Math.max(...this.mean(t));
  }

  reward(t, arm) {
    const luck = this.luck[t * this.k + arm];
    const p = this.mean(t)[arm];
    if (this.kind === 'gaussian') return p + this.sigma * luck;
    return luck < p ? 1.0 : 0.0;
  }
}

// Lai-Robbins constant for fixed machines: sum over suboptimal arms of gap / KL(p_i || p*).
export function laiRobbinsRate(means, kind, sigma = CONSTANTS.sigma) {
  const best = Math.max(...means);
  let rate = 0.0;
  for (const p of means) {
    const gap = best - p;
    if (gap <= 0) continue;
    const kl = kind === 'bernoulli'
      ? p * Math.log(p / best) + (1.0 - p) * Math.log((1.0 - p) / (1.0 - best))
      : gap * gap / (2.0 * sigma * sigma);
    rate += gap / kl;
  }
  return rate;
}

// ── agents.py ────────────────────────────────────────────────────────────

const letter = (i) => String.fromCharCode(65 + i);

// Python's format(x, '.Nf') rounds an exact tie to the even digit (0.625 -> 0.62), where toFixed rounds it up
// (0.63). The reasons use this so the page prints the same text as watch mode. x is never negative here.
function fixed(x, digits) {
  // toFixed(100) writes the double's exact decimal value, so a tie shows up as a 5 followed only by zeros.
  const exact = x.toFixed(100);
  const cut = digits ? exact.indexOf('.') + 1 + digits : exact.indexOf('.');
  const kept = exact.slice(0, cut);
  if (/^50*$/.test(exact.slice(cut).replace('.', '')) && Number(kept.slice(-1)) % 2 === 0) return kept;
  return x.toFixed(digits);
}

function argmax(xs) {
  let best = 0;
  for (let i = 1; i < xs.length; i++) if (xs[i] > xs[best]) best = i;
  return best;
}

class Agent {
  constructor(k, horizon, rng, sigma = 1.0, bernoulli = true) {
    this.k = k;
    this.horizon = horizon;
    this.rng = rng;
    this.sigma = sigma;
    this.bernoulli = bernoulli;
    this.counts = new Array(k).fill(0);
    this.sums = new Array(k).fill(0.0);
    this.t = 0;
    this.scores = new Array(k).fill(0.0);
    this.reason = '';
  }

  means() {
    return this.sums.map((s, i) => (this.counts[i] ? s / this.counts[i] : 0.0));
  }

  update(arm, reward) {
    this.counts[arm] += 1;
    this.sums[arm] += reward;
    this.t += 1;
  }

  forced() {
    this.scores = this.means();
    this.reason = `first round: try machine ${letter(this.t)} once, since nothing is known yet`;
    return this.t;
  }
}

class Greedy extends Agent {
  choose() {
    if (this.t < this.k) return this.forced();
    const m = this.means();
    this.scores = m;
    const arm = argmax(m);
    this.reason = `exploit: machine ${letter(arm)} has the best average so far (${fixed(m[arm], 2)})`;
    return arm;
  }
}

class EpsilonGreedy extends Agent {
  constructor(k, horizon, rng, sigma = 1.0, bernoulli = true) {
    super(k, horizon, rng, sigma, bernoulli);
    this.decay = false;
    this.epsilon = 0.1;
  }

  epsilonNow() {
    return this.decay ? Math.min(1.0, this.k / this.t) : this.epsilon;
  }

  choose() {
    if (this.t < this.k) return this.forced();
    const m = this.means();
    this.scores = m;
    const eps = this.epsilonNow();
    // Two draws, in this order: explore-or-exploit first, then the random machine if exploring.
    let arm;
    if (this.rng.uniform() < eps) {
      arm = this.rng.index(this.k);
      this.reason = `explore: a random machine (${letter(arm)}), with epsilon ${fixed(eps, 3)}`;
    } else {
      arm = argmax(m);
      this.reason = `exploit: machine ${letter(arm)} has the best average (${fixed(m[arm], 2)}); epsilon ${fixed(eps, 3)}`;
    }
    return arm;
  }
}

class EpsilonDecay extends EpsilonGreedy {
  constructor(k, horizon, rng, sigma = 1.0, bernoulli = true) {
    super(k, horizon, rng, sigma, bernoulli);
    this.decay = true;
  }
}

class UCB1 extends Agent {
  choose() {
    if (this.t < this.k) return this.forced();
    const t = this.t;
    const m = this.means();
    const idx = m.map((mi, i) => mi + this.sigma * Math.sqrt(2.0 * Math.log(t) / this.counts[i]));
    this.scores = idx;
    const arm = argmax(idx);
    const bonus = idx[arm] - m[arm];
    this.reason = `highest optimistic bound: machine ${letter(arm)} has mean ${fixed(m[arm], 2)} `
      + `plus a bonus of ${fixed(bonus, 2)} for its ${this.counts[arm]} pulls`;
    return arm;
  }
}

class Thompson extends Agent {
  posterior() {
    return this.counts.map((n, i) => {
      const s = this.sums[i];
      if (this.bernoulli) return [1.0 + s, 1.0 + n - s];
      const prec = 1.0 + n / (this.sigma * this.sigma);
      return [(s / (this.sigma * this.sigma)) / prec, 1.0 / Math.sqrt(prec)];
    });
  }

  choose() {
    const post = this.posterior();
    const samples = post.map(([a, b]) => (this.bernoulli
      ? this.rng.beta(a, b)
      : a + b * this.rng.normal()));
    this.scores = samples;
    const arm = argmax(samples);
    const [a, b] = post[arm];
    this.reason = this.bernoulli
      ? `sampled ${fixed(samples[arm], 2)} from machine ${letter(arm)}'s Beta(${fixed(a, 0)}, ${fixed(b, 0)}) posterior, the best sample this round`
      : `sampled ${fixed(samples[arm], 2)} from machine ${letter(arm)}'s posterior (mean ${fixed(a, 2)}, sd ${fixed(b, 2)}), the best sample this round`;
    return arm;
  }
}

class EXP3 extends Agent {
  constructor(k, horizon, rng, sigma = 1.0, bernoulli = true) {
    super(k, horizon, rng, sigma, bernoulli);
    this.gamma = Math.min(1.0, Math.sqrt(k * Math.log(k) / ((Math.E - 1) * horizon)));
    this.weights = new Array(k).fill(1.0);
    this._probs = new Array(k).fill(1.0 / k);
  }

  probabilities() {
    let total = 0;
    for (const w of this.weights) total += w;
    return this.weights.map((w) => (1.0 - this.gamma) * w / total + this.gamma / this.k);
  }

  choose() {
    const probs = this.probabilities();
    this._probs = probs;
    this.scores = probs;
    const u = this.rng.uniform();
    let cum = 0.0;
    let arm = this.k - 1;
    for (let i = 0; i < probs.length; i++) {
      cum += probs[i];
      if (u < cum) {
        arm = i;
        break;
      }
    }
    this.reason = `sample machine ${letter(arm)} with probability ${fixed(probs[arm], 2)}`;
    return arm;
  }

  update(arm, reward) {
    super.update(arm, reward);
    const estimate = reward / this._probs[arm];
    this.weights[arm] *= Math.exp(this.gamma * estimate / this.k);
    const top = Math.max(...this.weights);
    this.weights = this.weights.map((w) => w / top);
  }
}

class SlidingWindowUCB extends Agent {
  constructor(k, horizon, rng, sigma = 1.0, bernoulli = true, window = CONSTANTS.window, xi = CONSTANTS.window_xi) {
    super(k, horizon, rng, sigma, bernoulli);
    this.window = window;
    this.xi = xi;
    this.recent = [];
    this.wcounts = new Array(k).fill(0);
    this.wsums = new Array(k).fill(0.0);
  }

  update(arm, reward) {
    super.update(arm, reward);
    this.recent.push([arm, reward]);
    this.wcounts[arm] += 1;
    this.wsums[arm] += reward;
    if (this.recent.length > this.window) {
      const [oldArm, oldReward] = this.recent.shift();
      this.wcounts[oldArm] -= 1;
      this.wsums[oldArm] -= oldReward;
    }
  }

  choose() {
    for (let i = 0; i < this.k; i++) {
      if (this.wcounts[i] === 0) {
        this.scores = new Array(this.k).fill(0.0);
        this.reason = `machine ${letter(i)} has not paid out in the last ${this.window} pulls: look again`;
        return i;
      }
    }
    const logT = Math.log(Math.min(this.t, this.window));
    const idx = this.wcounts.map((n, i) => this.wsums[i] / n + this.sigma * Math.sqrt(this.xi * logT / n));
    this.scores = idx;
    const arm = argmax(idx);
    this.reason = `highest window index: machine ${letter(arm)} averages `
      + `${fixed(this.wsums[arm] / this.wcounts[arm], 2)} over its last ${this.wcounts[arm]} pulls in the window`;
    return arm;
  }
}

export const AGENT_CLASSES = {
  greedy: Greedy,
  eps: EpsilonGreedy,
  eps_decay: EpsilonDecay,
  ucb1: UCB1,
  thompson: Thompson,
  exp3: EXP3,
  sw_ucb: SlidingWindowUCB,
};

// The agent named `key`, set up for a casino of kind `kind`, with its own stream derived from the seed.
export function makeAgent(key, { kind, k, horizon, seed }, constants = CONSTANTS) {
  if (!(key in AGENT_CLASSES)) throw new Error(`unknown agent ${key}`);
  const rng = new Rng(seed + constants.agent_offset);
  const gaussian = kind === 'gaussian';
  const sigma = gaussian ? constants.sigma : 1.0;
  return new AGENT_CLASSES[key](k, horizon, rng, sigma, !gaussian);
}

// ── sim.py ───────────────────────────────────────────────────────────────

// One agent playing a casino, one pull per call to step(). The arrays are the same record as sim.Run.
export class Racer {
  constructor(key, casino, constants = CONSTANTS) {
    this.key = key;
    this.casino = casino;
    this.agent = makeAgent(key, casino, constants);
    this.arms = [];
    this.rewards = [];
    this.regret = [];
    this.optimal = [];
    this.cum = 0.0;
  }

  get t() {
    return this.arms.length;
  }

  get done() {
    return this.arms.length >= this.casino.horizon;
  }

  // One pull: the agent chooses, the casino pays, the agent learns. Returns the pull's record.
  step() {
    const t = this.arms.length;
    const arm = this.agent.choose();
    const reward = this.casino.reward(t, arm);
    const m = this.casino.mean(t);
    const best = Math.max(...m);
    this.cum += best - m[arm];
    this.arms.push(arm);
    this.rewards.push(reward);
    this.regret.push(this.cum);
    this.optimal.push(m[arm] === best);
    this.agent.update(arm, reward);
    return { t, arm, reward, regret: this.cum, reason: this.agent.reason };
  }

  // Fraction of pulls, from `start` on, that went to a machine tied for the best mean.
  shareOptimal(start = 0) {
    const tail = this.optimal.slice(start);
    return tail.length ? tail.filter(Boolean).length / tail.length : 0;
  }
}

// Score a list of arms (the player's pulls) on a casino. Mirrors sim.score_pulls.
export function scorePulls(casino, arms) {
  let cum = 0.0;
  const regret = [];
  const optimal = [];
  const rewards = [];
  arms.forEach((arm, t) => {
    const m = casino.mean(t);
    const best = Math.max(...m);
    cum += best - m[arm];
    regret.push(cum);
    optimal.push(m[arm] === best);
    rewards.push(casino.reward(t, arm));
  });
  return { regret, optimal, rewards, total: cum };
}
