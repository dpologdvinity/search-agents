// Markov text page: a chain that writes one token at a time from counts in a public-domain book.
//
// markov-core.js does the counting and sampling (a port of the Python package `markov`, which produces the same
// tokens for a corpus, order, temperature and seed). This file draws it: the output with its context window and
// copy marks, the next-token panel with the sampled candidate flashing, the copy meter, and a transcript.

import { pop, shake } from './fx.js';
import {
  MAX_RUN, Rng, Walker, buildModel, distribution, ngramSets, spaceBefore, tokenize, windowStats,
} from './markov-core.js';

const $ = (id) => document.getElementById(id);
const REDUCED = matchMedia('(prefers-reduced-motion: reduce)').matches;
const MAX_SEED = 2147483647;
const RARE = 0.05;       // a pick below this probability glitches
const TOP = 8;           // candidates shown in the panel
const LOG_LINES = 200;

const CORPORA = [
  { id: 'alice', label: 'ALICE IN WONDERLAND (1865)' },
  { id: 'pride', label: 'PRIDE AND PREJUDICE (1813)' },
  { id: 'sonnets', label: 'SHAKESPEARE SONNETS (1609)' },
  { id: 'constitution', label: 'US CONSTITUTION: PREAMBLE + ARTICLE I (1787)' },
];

const ORDER_NOTES = [
  'Order 1 reads nothing: each token is drawn by its frequency in the book alone.',
  'Order 2 reads the one token before it.',
  'Order 3 reads the two tokens before it.',
  'Order 4 reads the three tokens before it.',
  'Order 5 reads the four tokens before it. Nearly every phrase it writes is copied from the book.',
];

const S = {
  corpus: 'alice',
  level: 'word',
  order: 3,
  temp: 1,
  seed: 1,
  speed: 6,
  max: 120,
  model: null,
  sets: null,
  setsKey: '',
  walker: null,
  spans: [],       // DOM span per entry of walker.out
  kinds: [],       // 'prompt' | 'gen' | 'jump' per entry, for the copy meter and the log
  highlight: [],   // indices currently marked as context or last
  last: null,      // the last step's result, to redraw the panel when the temperature moves
  picks: 0,
  copied: 0,
  longest: 0,
  timer: 0,
  running: false,
  sources: new Map(),
};

// ── Corpus ───────────────────────────────────────────────────────────────

// A corpus file is a header (its Source line), a line with ---, then the text. The model reads only the text.
async function loadCorpus(id) {
  if (!S.sources.has(id)) {
    S.sources.set(id, fetch(`data/markov/${id}.txt`).then((r) => {
      if (!r.ok) throw new Error(`could not load ${id}.txt (${r.status})`);
      return r.text();
    }).then((raw) => {
      const cut = raw.indexOf('\n---\n');
      const head = cut < 0 ? '' : raw.slice(0, cut);
      const body = cut < 0 ? raw : raw.slice(cut + 5);
      const line = head.split('\n').find((l) => l.startsWith('Source:')) || '';
      return { id, source: line.slice('Source:'.length).trim(), body };
    }));
  }
  return S.sources.get(id);
}

// Counts and copy sets for the current corpus and level. The copy sets depend only on those, not on the order.
async function rebuild() {
  const corpus = await loadCorpus(S.corpus);
  const key = `${S.corpus}|${S.level}`;
  const tokens = tokenize(corpus.body, S.level);
  if (S.setsKey !== key) {
    S.sets = ngramSets(tokens, MAX_RUN);
    S.setsKey = key;
  }
  S.tokens = tokens;
  S.model = buildModel(tokens, S.order);
  $('source-note').textContent = `SOURCE: ${corpus.source}. Public domain.`;
  showChips();
}

// ── Display helpers ──────────────────────────────────────────────────────

const show = (tok) => (tok === ' ' ? '␣' : tok);
const pctText = (p) => {
  const pct = 100 * p;
  return `${pct < 1 ? pct.toFixed(2) : pct.toFixed(1)}%`;
};

function showChips() {
  const m = S.model;
  $('chip-vocab').textContent = String(m.vocabSize);
  $('chip-contexts').textContent = String(m.contextCount);
  $('chip-ngrams').textContent = String(m.ngramCount);
  const pct = S.picks ? (100 * S.copied) / S.picks : 0;
  $('chip-copied').textContent = S.picks ? `${pct.toFixed(0)}%` : '—';
  $('chip-run').textContent = S.picks ? String(S.longest) : '—';
  $('meter-fill').style.width = `${pct}%`;
  $('meter-note').textContent = S.picks
    ? `${S.copied} of ${S.picks} order-${S.order} n-grams are copied verbatim from the book. The longest copied run is ${S.longest} tokens.`
    : 'Order 5 copies: a 5-word phrase the chain writes almost always already exists in the book. The longer the order, the more of the book it repeats.';
}

function setStatus(text) {
  $('status').textContent = text;
}

// One output token, as a span. The leading space is part of the span, so the typewriter reveal covers it.
function makeSpan(tok, kind, prev, animate) {
  const span = document.createElement('span');
  span.className = `mk-tok mk-${kind}`;
  const sp = spaceBefore(prev, tok, S.level) ? ' ' : '';
  span.textContent = sp + show(tok);
  if (animate && !REDUCED) {
    const n = (sp + show(tok)).length;
    span.classList.add('mk-new');
    span.style.setProperty('--n', String(Math.max(1, n)));
    span.style.setProperty('--dur', `${Math.min(260, Math.round(900 / S.speed))}ms`);
    span.addEventListener('animationend', () => span.classList.remove('mk-new'), { once: true });
  }
  return span;
}

function appendToken(tok, kind, animate) {
  const out = $('out');
  const idx = S.spans.length;
  const prev = idx > 0 ? S.walker.out[idx - 1] : null;
  const span = makeSpan(tok, kind, prev, animate);
  out.appendChild(span);
  S.spans.push(span);
  S.kinds.push(kind);
  out.scrollTop = out.scrollHeight;
}

// Context window: the last `order` tokens. The first order-1 are the context the chain reads; the last is the
// token it just picked (or the newest one, when the chain is idle).
function highlightWindow(last) {
  for (const i of S.highlight) {
    S.spans[i]?.classList.remove('mk-ctx', 'mk-last');
  }
  S.highlight = [];
  if (last < 0) return;
  for (let i = Math.max(0, last - S.order + 1); i <= last; i++) {
    const span = S.spans[i];
    if (!span) continue;
    span.classList.add(i === last ? 'mk-last' : 'mk-ctx');
    S.highlight.push(i);
  }
}

function renderPanel(r) {
  const k = S.order - 1;
  const line = $('ctx-line');
  line.innerHTML = '';
  if (r === null) {
    line.textContent = k === 0 ? 'Order 1 reads no words.' : 'Press GENERATE or STEP to pick the first token.';
  } else if (k === 0) {
    line.textContent = 'Order 1 reads no words: every candidate is weighted by its count in the book alone.';
  } else {
    for (const tok of r.ctx) {
      const chip = document.createElement('span');
      chip.className = 'mk-chip';
      chip.textContent = show(tok);
      line.appendChild(chip);
    }
    const label = document.createElement('span');
    label.className = 'mk-chip-note';
    const n = S.order - 1;
    label.textContent = `→ ${n} ${S.level === 'char' ? 'character' : 'word'}${n === 1 ? '' : 's'} read, ${r.dist.length} candidate${r.dist.length === 1 ? '' : 's'}`;
    line.appendChild(label);
  }
  $('ctx-title').textContent = `Context window: order ${S.order}, temperature ${S.temp.toFixed(2)}`;

  const bars = $('bars');
  bars.innerHTML = '';
  if (r === null) return;
  const ranked = [...r.dist].sort((a, b) => b.prob - a.prob);
  const rows = ranked.slice(0, TOP);
  if (!rows.some((c) => c.token === r.token && c.prob === r.prob)) {
    const picked = ranked.find((c) => c.token === r.token);
    if (picked) rows[TOP - 1] = picked;
  }
  const top = ranked[0].prob;
  for (const c of rows) {
    const isPick = c.token === r.token;
    const row = document.createElement('div');
    row.className = `mk-bar-row${isPick ? ' picked' : ''}`;
    row.setAttribute('role', 'listitem');
    row.innerHTML = `<span class="mk-bar-lbl"></span><span class="mk-bar-track"><span class="mk-bar-fill"></span></span><span class="mk-bar-pct"></span>`;
    row.querySelector('.mk-bar-lbl').textContent = show(c.token);
    row.querySelector('.mk-bar-fill').style.width = `${(100 * c.prob) / top}%`;
    row.querySelector('.mk-bar-pct').textContent = pctText(c.prob);
    bars.appendChild(row);
    if (isPick && !REDUCED) {
      row.classList.add('flash');
      row.addEventListener('animationend', () => row.classList.remove('flash'), { once: true });
    }
  }
  const rank = ranked.findIndex((c) => c.token === r.token) + 1;
  $('pick-line').textContent = `Sampled ${JSON.stringify(show(r.token))} with p = ${pctText(r.prob)} (rank ${rank} of ${ranked.length}).`;
}

function logPick(n, tok, prob, copied, jumps) {
  const log = $('log');
  const row = document.createElement('div');
  row.className = 'mk-row';
  const tag = copied ? '<span class="log-best">COPY</span>' : '<span class="log-val">NEW </span>';
  const jump = jumps ? ' <span class="log-adv">JUMP</span>' : '';
  row.innerHTML = `<span class="log-info">${String(n).padStart(3, '0')}</span> ${tag} <span class="log-move"></span> <span class="log-info">p=${pctText(prob)}</span>${jump}`;
  row.querySelector('.log-move').textContent = JSON.stringify(show(tok));
  log.appendChild(row);
  while (log.childElementCount > LOG_LINES) log.firstElementChild.remove();
  log.scrollTop = log.scrollHeight;
}

// ── The chain ────────────────────────────────────────────────────────────

function reset() {
  stop();
  S.walker = new Walker(S.model, new Rng(S.seed), S.temp);
  S.spans = [];
  S.kinds = [];
  S.highlight = [];
  S.last = null;
  S.picks = 0;
  S.copied = 0;
  S.longest = 0;
  $('out').innerHTML = '';
  $('log').innerHTML = '';
  S.walker.out.forEach((tok, i) => {
    const prev = i > 0 ? S.walker.out[i - 1] : null;
    const span = makeSpan(tok, 'prompt', prev, false);
    $('out').appendChild(span);
    S.spans.push(span);
    S.kinds.push('prompt');
  });
  highlightWindow(S.spans.length - 1);
  renderPanel(null);
  showChips();
  setStatus(`Starting window from seed ${S.seed}. Press GENERATE, or STEP for one token.`);
  syncRunButton();
}

// One token: draw from the candidates, show it, mark the context window, update the copy meter and the panel.
function step() {
  if (!S.walker) reset();
  if (S.picks >= S.max) {
    stop();
    setStatus(`END: ${S.max} tokens written. RESET starts again with the same seed; RANDOM picks a new one.`);
    return;
  }
  const r = S.walker.next();
  // Jump tokens (a restart at the end of the book) were added to the walker's output without a pick.
  for (let i = S.spans.length; i < r.index; i++) appendToken(S.walker.out[i], 'jump', true);
  appendToken(r.token, 'gen', true);

  const stats = windowStats(S.walker.out, r.index, S.order, S.sets);
  S.picks += 1;
  if (stats.copied) S.copied += 1;
  S.longest = Math.max(S.longest, stats.run);
  const span = S.spans[r.index];
  if (!stats.copied) span.classList.add('mk-novel');
  else span.classList.add('mk-copied');

  highlightWindow(r.index);
  const rare = r.prob < RARE;
  if (rare && !REDUCED) {
    span.classList.add('mk-glitch');
    setTimeout(() => span.classList.remove('mk-glitch'), 520);
    shake($('out'));
    pop(span, 'RARE', '#ff00a0');
  }
  if (r.jumps) pop($('out'), 'JUMP', '#ffe600');

  S.last = r;
  renderPanel(r);
  logPick(S.picks, r.token, r.prob, stats.copied, r.jumps);
  showChips();
  const parts = [`pick ${S.picks}/${S.max}`, `p ${pctText(r.prob)}`, stats.copied ? 'copied from the book' : 'new phrase'];
  if (rare) parts.push('low-probability pick');
  if (r.jumps) parts.push('jumped to a new window at the end of the book');
  setStatus(parts.join(' · '));
  if (S.picks >= S.max) {
    stop();
    setStatus(`END: ${S.max} tokens written. RESET starts again with the same seed; RANDOM picks a new one.`);
  }
}

function start() {
  if (!S.walker || S.picks >= S.max) reset(); // a finished run starts again with the same seed
  S.running = true;
  schedule();
  syncRunButton();
}

function schedule() {
  clearInterval(S.timer);
  S.timer = setInterval(step, Math.round(1000 / S.speed));
}

function stop() {
  S.running = false;
  clearInterval(S.timer);
  S.timer = 0;
  syncRunButton();
}

function syncRunButton() {
  const btn = $('btn-run');
  btn.setAttribute('aria-pressed', String(S.running));
  btn.querySelector('.btn-txt').textContent = S.running ? '❚❚ PAUSE' : '▶ GENERATE';
}

// ── Controls ─────────────────────────────────────────────────────────────

function readSeed() {
  const v = Math.floor(Number($('seed').value));
  S.seed = Number.isFinite(v) && v >= 0 && v <= MAX_SEED ? v : 1 + Math.floor(Math.random() * (MAX_SEED - 1));
  $('seed').value = String(S.seed);
}

function setPressed(ids, on) {
  for (const id of ids) $(id).setAttribute('aria-pressed', String(id === on));
}

function setupControls() {
  const sel = $('corpus');
  for (const c of CORPORA) {
    const o = document.createElement('option');
    o.value = c.id;
    o.textContent = c.label;
    sel.appendChild(o);
  }
  sel.value = S.corpus;
  sel.addEventListener('change', async () => {
    S.corpus = sel.value;
    await rebuild();
    reset();
  });

  for (const level of ['word', 'char']) {
    $(level === 'word' ? 'lvl-word' : 'lvl-char').addEventListener('click', async () => {
      if (S.level === level) return;
      S.level = level;
      setPressed(['lvl-word', 'lvl-char'], level === 'word' ? 'lvl-word' : 'lvl-char');
      await rebuild();
      reset();
    });
  }

  // The order only takes effect on release: the model is rebuilt then, so the chain never runs on a stale order.
  const order = $('order');
  const setOrderNote = (n) => {
    $('order-val').textContent = String(n);
    $('order-note').textContent = ORDER_NOTES[n - 1];
  };
  order.addEventListener('input', () => setOrderNote(Number(order.value)));
  order.addEventListener('change', async () => {
    S.order = Number(order.value);
    setOrderNote(S.order);
    await rebuild();
    reset();
  });
  setOrderNote(S.order);

  // Temperature is live: the chain uses the new weights from the next token, and the panel is redrawn for the
  // context of the last pick.
  const temp = $('temp');
  temp.addEventListener('input', () => {
    S.temp = Number(temp.value);
    $('temp-val').textContent = S.temp.toFixed(2);
    if (S.walker) S.walker.temperature = S.temp;
    if (S.last) {
      const dist = distribution(S.model, S.last.ctx, S.temp);
      if (dist) {
        const mine = dist.find((c) => c.token === S.last.token);
        S.last = { ...S.last, dist, prob: mine ? mine.prob : S.last.prob };
        renderPanel(S.last);
      }
    } else {
      renderPanel(null);
    }
  });

  $('seed').value = String(S.seed);
  $('seed').addEventListener('change', () => {
    readSeed();
    reset();
  });
  $('btn-seed').addEventListener('click', () => {
    S.seed = 1 + Math.floor(Math.random() * (MAX_SEED - 1));
    $('seed').value = String(S.seed);
    reset();
    start();
  });

  const speed = $('speed');
  speed.addEventListener('input', () => {
    S.speed = Number(speed.value);
    $('speed-val').textContent = String(S.speed);
    if (S.running) schedule();
  });
  $('speed-val').textContent = String(S.speed);

  $('length').addEventListener('change', () => {
    S.max = Number($('length').value);
    if (S.picks >= S.max) stop();
  });

  $('btn-run').addEventListener('click', () => (S.running ? stop() : start()));
  $('btn-step').addEventListener('click', () => {
    stop();
    step();
  });
  $('btn-reset').addEventListener('click', reset);

  // Space starts and pauses the chain when nothing else has focus; n takes one step.
  document.addEventListener('keydown', (e) => {
    if (e.target !== document.body || e.ctrlKey || e.metaKey || e.altKey) return;
    if (e.key === ' ') {
      e.preventDefault();
      S.running ? stop() : start();
    } else if (e.key === 'n') {
      stop();
      step();
    }
  });
}

async function main() {
  setupControls();
  setPressed(['lvl-word', 'lvl-char'], 'lvl-word');
  try {
    await rebuild();
    reset();
  } catch (err) {
    setStatus(`Could not load the corpus: ${err.message}. Serve the page from the site root.`);
  }
}

main();
