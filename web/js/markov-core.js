// Markov chain core: tokens, n-gram counts, temperature, seeded sampling, copy statistics.
//
// This file has no DOM code, so the same token sequence comes out of Python and JavaScript for a corpus, order,
// temperature and seed. It is a port of markov/model.py, and the operations run in the same order. The seeded
// generator is mulberry32 (the same one as bandits/rng.py and bandits-core.js), and each generated token draws
// exactly one uniform, so the streams stay aligned.

const TWO32 = 4294967296;
export const MAX_RUN = 20;
const WORD_RE = /[A-Za-z]+(?:['’][A-Za-z]+)*|[^\sA-Za-z]/g;
const NO_SPACE_BEFORE = new Set(['.', ',', ';', ':', '!', '?', ')']);
const NO_SPACE_AFTER = new Set(['(']);
// Keys join tokens with a control character that cannot appear in the text.
const SEP = '\u0001';

// mulberry32 with Math.imul, so both sides keep the low 32 bits of each product.
export class Rng {
  constructor(seed) {
    this.state = seed >>> 0;
  }

  // A double in [0, 1), one generator step.
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

// Whitespace runs collapse to one space, then words or single punctuation marks (word level) or characters.
export function tokenize(text, level = 'word') {
  const body = text.replace(/\s+/g, ' ').trim();
  if (level === 'char') return Array.from(body);
  if (level === 'word') return body.match(WORD_RE) || [];
  throw new Error(`unknown level ${level}: use 'word' or 'char'`);
}

// Whether a space goes between prev and tok when the text is rebuilt: no space before closing punctuation or
// after an opening bracket, and none at all at character level.
export function spaceBefore(prev, tok, level = 'word') {
  return level !== 'char' && prev !== null && !NO_SPACE_BEFORE.has(tok) && !NO_SPACE_AFTER.has(prev);
}

// Rebuild readable text from tokens (the same rule as markov/model.py detokenize).
export function detokenize(tokens, level = 'word') {
  let out = '';
  let prev = null;
  for (const tok of tokens) {
    if (spaceBefore(prev, tok, level)) out += ' ';
    out += tok;
    prev = tok;
  }
  return out;
}

// Counts of every token after each (order-1)-token context. Map keeps insertion order, as the Python dict does.
export function buildModel(tokens, order) {
  if (order < 1) throw new Error('order must be at least 1');
  if (tokens.length < order) throw new Error(`need at least ${order} tokens`);
  const successors = new Map();
  const k = order - 1;
  for (let i = 0; i < tokens.length - k; i++) {
    const key = tokens.slice(i, i + k).join(SEP);
    let succ = successors.get(key);
    if (!succ) {
      succ = new Map();
      successors.set(key, succ);
    }
    const w = tokens[i + k];
    succ.set(w, (succ.get(w) || 0) + 1);
  }
  return {
    order,
    tokens,
    successors,
    vocabSize: new Set(tokens).size,
    contextCount: successors.size,
    ngramCount: [...successors.values()].reduce((n, d) => n + d.size, 0),
  };
}

// Next-token probabilities after ctx (an array of tokens) at this temperature, or null if ctx never precedes anything.
export function distribution(model, ctx, temperature = 1) {
  const succ = model.successors.get(ctx.join(SEP));
  if (!succ || succ.size === 0) return null;
  const items = [...succ.entries()];
  const weights = items.map(([, c]) => Math.pow(c, 1 / temperature));
  const total = weights.reduce((a, b) => a + b, 0);
  return items.map(([tok], i) => ({ token: tok, prob: weights[i] / total }));
}

// Inverse CDF on one uniform: the same walk as _pick in markov/model.py.
function pick(weights, rng) {
  const total = weights.reduce((a, b) => a + b, 0);
  const u = rng.uniform() * total;
  let cum = 0;
  for (let idx = 0; idx < weights.length; idx++) {
    cum += weights[idx];
    if (u < cum) return idx;
  }
  return weights.length - 1;
}

// The chain, one token at a time. The constructor draws the seeded starting window (the prompt); next() emits one
// token. A context with no successor (only at the corpus end) makes the chain jump to another seeded window: the
// jump emits no token, is counted in `restarts`, and is reported in the `jumps` of the step where it happened.
// The page calls next() on a timer; generate() below runs the same loop to the end for the parity test.
export class Walker {
  constructor(model, rng, temperature = 1) {
    this.model = model;
    this.rng = rng;
    this.temperature = temperature;
    this.k = model.order - 1;
    const i = rng.index(model.tokens.length - this.k);
    this.out = model.tokens.slice(i, i + this.k);
    this.promptLen = this.out.length;
    this.restarts = 0;
  }

  next() {
    const { model, rng, k, out } = this;
    const n = model.tokens.length;
    let jumps = 0;
    for (;;) {
      const ctx = out.slice(out.length - k); // the last k tokens; [] when k == 0
      const succ = model.successors.get(ctx.join(SEP));
      if (!succ || succ.size === 0) {
        const j = rng.index(n - k);
        for (const t of model.tokens.slice(j, j + k)) out.push(t);
        jumps += 1;
        this.restarts += 1;
        continue;
      }
      const items = [...succ.entries()];
      const weights = items.map(([, c]) => Math.pow(c, 1 / this.temperature));
      const total = weights.reduce((a, b) => a + b, 0);
      const idx = pick(weights, rng);
      const token = items[idx][0];
      const index = out.length;
      out.push(token);
      return {
        index, token, prob: weights[idx] / total, ctx, jumps,
        dist: items.map(([t], i) => ({ token: t, prob: weights[i] / total })),
      };
    }
  }
}

// Sample `count` tokens from a seeded corpus window. Same loop as markov/model.py generate().
export function generate(model, rng, count, temperature = 1) {
  const w = new Walker(model, rng, temperature);
  const picks = [];
  while (picks.length < count) {
    const r = w.next();
    picks.push({ index: r.index, token: r.token, prob: r.prob });
  }
  return { tokens: w.out, promptLen: w.promptLen, picks, restarts: w.restarts };
}

// Index l-1 holds every window of length l in the corpus, for l = 1 .. maxLen, as Sets of joined keys.
export function ngramSets(tokens, maxLen = MAX_RUN) {
  const sets = [];
  for (let length = 1; length <= maxLen; length++) {
    const s = new Set();
    for (let i = 0; i + length <= tokens.length; i++) s.add(tokens.slice(i, i + length).join(SEP));
    sets.push(s);
  }
  return sets;
}

// Copy check for the token at position p of out, as in markov/model.py copy_report: is the order-n window ending
// there in the corpus, and how long is the longest suffix ending there that is (capped at MAX_RUN)?
export function windowStats(out, p, order, sets) {
  const copied = sets[order - 1].has(out.slice(p - order + 1, p + 1).join(SEP));
  let length = 1;
  while (length <= Math.min(MAX_RUN, p + 1) && sets[length - 1].has(out.slice(p - length + 1, p + 1).join(SEP))) {
    length += 1;
  }
  return { copied, run: length - 1 };
}

// Copy statistics over a whole generation.
export function copyReport(gen, sets, order) {
  const copied = [];
  let longest = 0;
  for (const pick of gen.picks) {
    const s = windowStats(gen.tokens, pick.index, order, sets);
    copied.push(s.copied);
    longest = Math.max(longest, s.run);
  }
  const pct = copied.length ? (100 * copied.filter(Boolean).length) / copied.length : 0;
  return { copied, copiedPct: pct, longestRun: longest };
}
