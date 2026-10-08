// Poker page: heads-up Leduc hold'em against the CFR+ bot.
//
// The server holds each hand (so the bot's card stays on the server until showdown) and runs the bot's
// move as a lookup in the committed average strategy. This page renders what comes back: the table,
// the bot's probability bars for each decision it made, the explainer for its bets, and the training
// chart. The strategy table is fetched once, so the explainer can compare the bot's mix across cards.

import { getJSON, postJSON } from './api.js';
import { banner, burst, pop, shake } from './fx.js';

const $ = (id) => document.getElementById(id);
const REDUCED = matchMedia('(prefers-reduced-motion: reduce)').matches;
const RANKS = ['J', 'Q', 'K'];
const NAME = { f: 'fold', k: 'check', c: 'call', b: 'bet', r: 'raise' };
const LABEL = { f: 'FOLD', k: 'CHECK', c: 'CALL', b: 'BET', r: 'RAISE' };
const ORDER = ['f', 'k', 'c', 'b', 'r'];
const BIG_BLIND = 2;

const S = {
  hand: null,      // id of the hand on the server
  view: null,      // latest view from the server
  meta: null,
  table: null,     // the committed strategy: {information set: {action: probability}}
  training: null,  // exploitability curves
  net: 0,          // your winnings over the session, in chips
  hands: 0,        // hands settled
  busy: false,
  handMoves: [],   // the bot's decisions in this hand, with their mixes
  lastBot: null,   // the bot's most recent decision
  hintMode: false, // the mix panel shows your equilibrium mix instead of the bot's
};

const pct = (p) => `${Math.round(p * 100)}%`;
const mbb = (chips) => (chips * 1000 / BIG_BLIND).toFixed(3);

// ── Cards and table ─────────────────────────────────────────────────────

/** A card face (rank letter) or the back. The server sends ranks as letters J, Q, K. */
function cardEl(rank, { back = false, deal = false, empty = false } = {}) {
  const el = document.createElement('div');
  if (empty) {
    el.className = 'pk-card empty';
    return el;
  }
  if (back || !rank) {
    el.className = 'pk-card back';
    el.setAttribute('aria-label', 'face-down card');
  } else {
    el.className = `pk-card ${rank.toLowerCase()}`;
    el.textContent = rank;
    el.setAttribute('aria-label', `${rank} card`);
  }
  if (deal && !REDUCED) el.classList.add('deal');
  return el;
}

function renderTable(v) {
  const hidden = !v;
  $('your-cards').replaceChildren(hidden ? cardEl(null, { empty: true }) : cardEl(v.card, { deal: true }));
  if (hidden) {
    $('bot-cards').replaceChildren(cardEl(null, { empty: true }));
    $('public-cards').replaceChildren(cardEl(null, { empty: true }));
    $('pot').textContent = 'POT 0';
    $('bot-put').textContent = '';
    $('you-put').textContent = '';
    return;
  }
  $('bot-cards').replaceChildren(v.terminal ? cardEl(v.bot_card, { deal: true }) : cardEl(null, { back: true }));
  $('public-cards').replaceChildren(v.public ? cardEl(v.public, { deal: true }) : cardEl(null, { back: true }));
  $('pot').textContent = `POT ${v.pot}`;
  $('you-put').textContent = `in ${v.you_put}`;
  $('bot-put').textContent = `in ${v.bot_put}`;
}

function setButtons(v) {
  const busy = S.busy;
  const legal = new Set(v?.legal || []);
  for (const a of ['k', 'b', 'c', 'r', 'f']) $(`btn-${a}`).disabled = busy || !legal.has(a);
  $('btn-deal').disabled = busy || (v !== null && v !== undefined && !v.terminal);
  $('btn-hint').disabled = busy || !(v && v.hint);
  $('btn-hint').setAttribute('aria-pressed', String(S.hintMode));
}

function say(text, cls = '') {
  const el = $('msg');
  el.textContent = text;
  el.className = `pk-msg ${cls}`;
}

function log(text, cls = '') {
  const box = $('log');
  const line = document.createElement('span');
  line.className = `pk-ln ${cls}`;
  line.textContent = text;
  box.appendChild(line);
  box.scrollTop = box.scrollHeight;
}

// ── The bot's mix and the hint ─────────────────────────────────────────

/** Probability bars for a set of actions; the chosen one is highlighted. */
function renderBars(probs, chosen = null) {
  const box = $('mix-bars');
  box.replaceChildren();
  for (const a of ORDER) {
    if (!(a in probs)) continue;
    const row = document.createElement('div');
    row.className = `pk-bar-row${a === chosen ? ' chosen' : ''}`;
    const name = document.createElement('span');
    name.className = 'pk-bar-name';
    name.textContent = LABEL[a];
    const track = document.createElement('div');
    track.className = 'pk-bar-track';
    const fill = document.createElement('div');
    fill.className = 'pk-bar-fill';
    fill.style.width = `${(probs[a] * 100).toFixed(2)}%`;
    track.appendChild(fill);
    const val = document.createElement('span');
    val.className = 'pk-bar-pct';
    val.textContent = pct(probs[a]);
    row.append(name, track, val);
    box.appendChild(row);
  }
}

function renderMix() {
  const v = S.view;
  if (S.hintMode && v && v.hint) {
    $('mix-title').textContent = 'YOUR EQUILIBRIUM MIX: the probability of each action for your card here';
    renderBars(v.hint.probs);
    $('mix-note').textContent = 'Equilibrium says play this mix. Any single action is a best response only in spots where the bot is indifferent, so mixing is part of the equilibrium, not a hedge.';
    return;
  }
  if (S.lastBot) {
    const m = S.lastBot;
    const round = m.round === 1 ? 'round 1' : 'round 2';
    $('mix-title').textContent = `THE BOT'S ${round.toUpperCase()} DECISION: it chose ${LABEL[m.action]}, ${pct(m.probs[m.action])} of the time`;
    renderBars(m.probs, m.action);
    $('mix-note').textContent = 'Its card is not shown during the hand. The mix only depends on what the bot can see: its own card, the public card once it is turned up, and the betting so far.';
    return;
  }
  $('mix-title').textContent = 'Waiting for the bot\'s first decision.';
  $('mix-bars').replaceChildren();
}

// ── Explainer ───────────────────────────────────────────────────────────

/** Share of the time the bot bets (or raises) with each card in the same spot. Uses the committed table. */
function aggression(move) {
  return RANKS.map((r) => {
    const key = move.infoset.replace(/^[JQK]/, r);
    const row = S.table[key];
    return { rank: r, p: row ? (row.b || 0) + (row.r || 0) : null };
  });
}

function renderExplain() {
  const el = $('explain');
  const v = S.view;
  const bets = S.handMoves.filter((m) => m.action === 'b' || m.action === 'r');
  const move = bets[0] || S.handMoves[0];
  if (!move) {
    el.innerHTML = 'The bot has not decided anything this hand yet. When it bets, this panel shows how often it bets with each card it could hold, then explains the bet once its card is shown.';
    return;
  }
  const rows = aggression(move);
  const table = `<table><tr><th>card</th><th>bets here</th></tr>${rows.map((x) =>
    `<tr${v && v.terminal && x.rank === v.bot_card ? ' style="color:var(--yellow)"' : ''}><td>${x.rank}</td><td>${x.p === null ? '—' : pct(x.p)}</td></tr>`).join('')}</table>`;
  const why = `
    <p>The bot's mix is a random draw: it bets with some cards and checks with others in this exact spot, on purpose. Here is how often it bets with each card it could hold:</p>
    ${table}`;
  if (!v || !v.terminal) {
    el.innerHTML = `${why}<p>Its card is hidden until the showdown. The explanation follows once it is shown.</p>`;
    return;
  }
  const held = v.bot_card;
  const round1 = move.round === 1;
  let verdict;
  if (held === 'J') {
    verdict = `<p><span class="bluff">A bluff.</span> It held a <b>J</b>, the weakest card, which loses to any Q or K at showdown. The bet only wins if you fold.</p>
      ${round1 ? '<p>It bet before the public card was shown, so a pair was possible (about 1 time in 5), but it bets without that knowledge.</p>' : ''}`;
  } else if (held === 'K') {
    verdict = '<p><span class="value">A value bet.</span> A <b>K</b> beats any Q or J at showdown, so the bot wants you to call. Its bets with weaker cards keep this one from being readable.</p>';
  } else {
    verdict = '<p>A <b>Q</b> sits in the middle: it beats a J and loses to a K. Its bet mixes value with pressure, at the frequency the table shows.</p>';
  }
  el.innerHTML = `${verdict}${why}
    <p><b>Why mixing is the point.</b> If the bot never bluffed, you could fold every bet and lose nothing. If it always bluffed, you could call every bet and win with every Q and K. At equilibrium its bluff frequency leaves you indifferent between calling and folding, so neither choice can be exploited. That is the sense in which the mix is optimal.</p>`;
}

// ── Settling a hand and the session ────────────────────────────────────

function settle(v) {
  const p = v.payoff;
  S.net += p;
  S.hands += 1;
  $('chip-net').textContent = `${S.net >= 0 ? '+' : ''}${S.net}`;
  $('chip-hands').textContent = String(S.hands);
  const verdict = v.result === 'win' ? `You win ${p} chips.` : v.result === 'lose' ? `You lose ${-p} chips.` : 'Split pot.';
  // A fold ends the hand before the public card is turned up, so the showdown details only apply when it was.
  const shown = v.public ? `, and the public card was ${v.public}` : ' (the hand ended before the public card)';
  const cls = v.result === 'win' ? 'win' : v.result === 'lose' ? 'lose' : '';
  say(`${verdict} The bot held ${v.bot_card}${shown}.`, cls);
  log(`${verdict} The bot held ${v.bot_card}${v.public ? `; public ${v.public}` : ''}.`, cls);
  if (v.result === 'win') {
    const botFolded = S.handMoves.some((m) => m.action === 'f');
    banner(`+${p} CHIPS`, botFolded ? 'the bot folded' : 'showdown', '#00ff88');
    burst($('table'));
  } else if (v.result === 'lose') {
    shake($('table'), 'small');
  }
  renderExplain();
  S.lastBot = S.handMoves[S.handMoves.length - 1] || S.lastBot;
}

function render(v) {
  S.view = v;
  renderTable(v);
  const status = v.terminal ? 'HAND OVER' : v.to_act === 0 ? 'YOUR TURN' : 'BOT THINKING';
  $('chip-status').textContent = status;
  if (!v.terminal && v.to_act === 0) {
    say(v.round === 2
      ? `The public card is ${v.public}. Your turn in round 2.`
      : 'Your turn. Check, bet, call, raise or fold.');
  }
  setButtons(v);
  renderMix();
}

// Each action runs one round trip; the bot's answers arrive with the reply.
async function deal() {
  if (S.busy || S.hand) return; // a hand in progress must finish first
  S.busy = true;
  $('chip-status').textContent = 'DEALING';
  setButtons(S.view);
  try {
    const v = await postJSON('/api/poker/deal', {});
    S.hand = v.hand;
    S.handMoves = [];
    S.lastBot = null;
    S.hintMode = false;
    log(`HAND ${S.hands + 1}: you act first`, 'hand');
    render(v);
    renderExplain();
  } catch (e) {
    say(`Could not deal: ${e.message}`, 'lose');
  } finally {
    S.busy = false;
    setButtons(S.view);
  }
}

async function act(action) {
  if (S.busy || !S.hand) return;
  S.busy = true;
  $('chip-status').textContent = 'BOT THINKING'; // shown until the reply says whose move it is
  setButtons(S.view);
  try {
    const v = await postJSON('/api/poker/act', { hand: S.hand, action });
    log(`you ${NAME[action]}`, 'you');
    for (const m of v.bot_moves) {
      S.handMoves.push(m);
      S.lastBot = m;
      log(`the bot ${NAME[m.action]}${m.action === 'b' || m.action === 'r' ? ` ${m.round === 1 ? 2 : 4}` : ''}  (${pct(m.probs[m.action])})`, 'bot');
      if (m.action === 'b' || m.action === 'r') pop($('bot-cards'), m.action === 'b' ? 'BET' : 'RAISE', '#ff00a0');
    }
    S.hintMode = false;
    render(v);
    renderExplain();
    if (v.terminal) {
      S.hand = null;
      settle(v);
    }
  } catch (e) {
    say(e.message, 'lose');
    if (S.view && !S.view.terminal) $('chip-status').textContent = 'YOUR TURN'; // the request failed: the hand is still live
  } finally {
    S.busy = false;
    setButtons(S.view);
  }
}

// ── Training chart ──────────────────────────────────────────────────────

/** Exploitability against iterations on log-log axes: one line per algorithm, dots at the measured points. */
function drawChart() {
  const canvas = $('chart');
  const ctx = canvas.getContext('2d');
  const W = canvas.width, H = canvas.height;
  const pad = { l: 58, r: 16, t: 14, b: 36 };
  ctx.clearRect(0, 0, W, H);
  const runs = S.training?.runs || [];
  const pts = runs.flatMap((r) => r.iterations.map((t, i) => [t, r.exploitability[i]]).filter(([, e]) => e > 0));
  if (!pts.length) return;
  const xmax = Math.max(...pts.map(([t]) => t));
  const ymax = Math.pow(10, Math.ceil(Math.log10(Math.max(...pts.map(([, e]) => e)))));
  const ymin = Math.pow(10, Math.floor(Math.log10(Math.min(...pts.map(([, e]) => e)))));
  const X = (t) => pad.l + (Math.log10(t) / Math.log10(xmax)) * (W - pad.l - pad.r);
  const Y = (e) => pad.t + ((Math.log10(ymax) - Math.log10(e)) / (Math.log10(ymax) - Math.log10(ymin))) * (H - pad.t - pad.b);

  // Grid: one line per decade on the y axis, and powers of ten on the x axis.
  ctx.font = '10px "JetBrains Mono", monospace';
  ctx.lineWidth = 1;
  for (let e = ymin; e <= ymax * 1.0001; e *= 10) {
    ctx.strokeStyle = 'rgba(0,245,255,0.12)';
    ctx.beginPath(); ctx.moveTo(pad.l, Y(e)); ctx.lineTo(W - pad.r, Y(e)); ctx.stroke();
    ctx.fillStyle = '#7a8aa0';
    ctx.fillText(e >= 0.001 ? e.toString() : e.toExponential(0), 4, Y(e) + 3);
  }
  for (let t = 1; t <= xmax * 1.0001; t *= 10) {
    ctx.strokeStyle = 'rgba(0,245,255,0.12)';
    ctx.beginPath(); ctx.moveTo(X(t), pad.t); ctx.lineTo(X(t), H - pad.b); ctx.stroke();
    ctx.fillStyle = '#7a8aa0';
    ctx.fillText(String(t), X(t) - 8, H - pad.b + 14);
  }
  ctx.fillStyle = '#7a8aa0';
  ctx.fillText('iterations', W - pad.r - 60, H - 6);
  ctx.fillText('chips / hand', pad.l + 4, pad.t + 10);

  const colors = { 'CFR': '#ff00a0', 'CFR+': '#00f5ff' };
  for (const r of runs) {
    const color = colors[r.algorithm] || '#ffe600';
    ctx.strokeStyle = color;
    ctx.shadowColor = color;
    ctx.shadowBlur = 8;
    ctx.lineWidth = 2;
    ctx.beginPath();
    r.iterations.forEach((t, i) => {
      const e = r.exploitability[i];
      if (e <= 0) return;
      const x = X(t), y = Y(e);
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    });
    ctx.stroke();
    ctx.shadowBlur = 0;
    const last = r.iterations.length - 1;
    ctx.fillStyle = color;
    ctx.beginPath(); ctx.arc(X(r.iterations[last]), Y(r.exploitability[last]), 4, 0, Math.PI * 2); ctx.fill();
  }
}

// ── Start-up ────────────────────────────────────────────────────────────

async function init() {
  try {
    const [meta, training, table] = await Promise.all([
      getJSON('/api/poker/meta'),
      getJSON('/api/poker/training'),
      getJSON('/api/poker/strategy'),
    ]);
    S.meta = meta;
    S.training = training;
    S.table = table;
    $('chip-expl').textContent = `${mbb(meta.exploitability)} mbb/hand`;
    $('chip-value').textContent = `${meta.value_seat0.toFixed(4)} chips`;
    const runs = training.runs.map((r) => `${r.algorithm} ${r.iterations.at(-1)}`).join(' · ');
    const kuhn = training.kuhn_check?.['cfr+'];
    $('chart-note').textContent = `${runs} iterations. ${kuhn ? `Kuhn check: CFR+ lands within ${kuhn.value_error.toExponential(1)} of the exact -1/18.` : ''}`;
    drawChart();
    $('chip-status').textContent = 'READY';
    say('Deal a hand to start. You act first in both rounds.');
    setButtons(null);
    $('btn-deal').disabled = false;
  } catch (e) {
    $('chip-status').textContent = 'OFFLINE';
    say(`Could not reach the poker server: ${e.message}`, 'lose');
  }
}

$('btn-deal').addEventListener('click', deal);
for (const a of ['k', 'b', 'c', 'r', 'f']) $(`btn-${a}`).addEventListener('click', () => act(a));
$('btn-hint').addEventListener('click', () => {
  if (!S.view || !S.view.hint) return;
  S.hintMode = !S.hintMode;
  renderMix();
  setButtons(S.view);
});

// Keys: k b c r f act when legal; h toggles the hint; n deals the next hand.
addEventListener('keydown', (e) => {
  if (e.target.matches('input, textarea, select') || e.ctrlKey || e.metaKey || e.altKey) return;
  const key = e.key.toLowerCase();
  if (key === 'n') deal();
  else if (key === 'h') $('btn-hint').click();
  else if ('kbcrf'.includes(key) && key.length === 1) {
    const btn = $(`btn-${key}`);
    if (!btn.disabled) act(key);
  }
});

init();
