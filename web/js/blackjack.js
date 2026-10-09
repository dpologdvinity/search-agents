// Blackjack page. The table runs here, with the same rules as blackjack/game.py: a 6-deck shoe, S17, the
// dealer's peek, 3:2 blackjack, double on any two cards, one split with doubling allowed after it. The server
// answers three questions: the exact basic-strategy table, the expected value of each move for the hand in
// front of the player, and a bounded Monte Carlo learning run. The server never sees the dealer's hole card,
// and it keeps no game state, so the client deals and settles every hand.

import { getJSON, postJSON } from './api.js';
import { banner, burst, pop, shake } from './fx.js';

const $ = (id) => document.getElementById(id);
const REDUCED = matchMedia('(prefers-reduced-motion: reduce)').matches;
const DECKS = 6;
const PENETRATION = 0.75; // reshuffle between rounds once this share of the shoe has been dealt
const START_CHIPS = 500;
const MIN_BET = 5;
const ORDER = ['stand', 'hit', 'double', 'split'];
const LETTER = { stand: 'S', hit: 'H', double: 'D', split: 'P' };
const LABEL = { stand: 'STAND', hit: 'HIT', double: 'DOUBLE', split: 'SPLIT' };
const COLOR = { S: '#00f5ff', H: '#00ff88', D: '#ffe600', P: '#ff00a0' };

const state = {
  chips: START_CHIPS, // the bankroll; a round's wagers are reserved against it until it settles
  bet: 10,
  shoe: [],
  pos: 0,
  round: null, // { hands, dealer, dealerNatural, seen }
  busy: false,
  adviceSeq: 0, // lets a late advice response be ignored once the hand has moved on
  strategy: null,
  stats: { hands: 0, won: 0, lost: 0, pushed: 0, net: 0 },
  learning: false,
  learnData: null,
};

const rankValue = (r) => Math.min(r, 10);
const rankLabel = (r) => (r === 1 ? 'A' : r === 11 ? 'J' : r === 12 ? 'Q' : r === 13 ? 'K' : String(r));
const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, REDUCED ? 0 : ms));

/** Hard total, best total, soft flag and bust flag of a list of ranks (an ace counts 1, or 11 when it fits). */
function score(ranks) {
  let hard = 0;
  let ace = false;
  for (const r of ranks) {
    hard += rankValue(r);
    if (r === 1) ace = true;
  }
  const best = ace && hard + 10 <= 21 ? hard + 10 : hard;
  return { hard, best, soft: ace && best !== hard, bust: hard > 21 };
}

const isNatural = (hand) => hand.cards.length === 2 && score(hand.cards).best === 21 && !hand.fromSplit;
const current = (round) => round.hands.find((h) => !h.done) || null;
const wagered = (round) => round.hands.reduce((sum, h) => sum + h.bet, 0);

// ── Shoe ─────────────────────────────────────────────────────────────────

function shuffleShoe() {
  const cards = [];
  for (let d = 0; d < DECKS; d++) for (let r = 1; r <= 13; r++) for (let s = 0; s < 4; s++) cards.push(r);
  for (let i = cards.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [cards[i], cards[j]] = [cards[j], cards[i]];
  }
  state.shoe = cards;
  state.pos = 0;
}

function draw() {
  if (state.pos >= state.shoe.length) shuffleShoe();
  return state.shoe[state.pos++];
}

// ── Rules: deal, legal actions, actions, settlement ─────────────────────

/** Deal in the same order as a table: player, dealer upcard, player, dealer hole card. */
function startRound(bet) {
  if (state.pos >= PENETRATION * state.shoe.length) shuffleShoe();
  const p1 = draw();
  const up = draw();
  const p2 = draw();
  const hole = draw();
  const hand = { cards: [p1, p2], bet, fromSplit: false, splitAces: false, done: false };
  const dealer = [up, hole];
  const dealerNatural = score(dealer).best === 21;
  hand.done = isNatural(hand) || dealerNatural; // a natural, or a dealer natural, ends the round at once
  return { hands: [hand], dealer, dealerNatural, seen: new Set() };
}

/** Actions the rules allow on the hand in play. Double and split also need chips for one more bet. */
function legalActions(round) {
  const hand = current(round);
  if (!hand) return [];
  const actions = ['stand', 'hit'];
  if (hand.cards.length === 2) {
    const canAfford = state.chips - wagered(round) >= hand.bet;
    if (canAfford) actions.push('double');
    const pair = rankValue(hand.cards[0]) === rankValue(hand.cards[1]);
    if (pair && !hand.fromSplit && round.hands.length === 1 && canAfford) actions.push('split');
  }
  return actions;
}

function applyAction(round, action) {
  const hand = current(round);
  if (action === 'stand') {
    hand.done = true;
  } else if (action === 'hit') {
    hand.cards.push(draw());
    const s = score(hand.cards);
    hand.done = s.bust || s.best === 21; // a hand at 21 stands automatically
  } else if (action === 'double') {
    hand.bet *= 2;
    hand.cards.push(draw());
    hand.done = true;
  } else {
    // Split: each hand keeps one card of the pair and takes a second card at once. Split aces stop there.
    const rank = hand.cards[0];
    round.hands = [0, 1].map(() => {
      const h = { cards: [rank, draw()], bet: hand.bet, fromSplit: true, splitAces: rank === 1, done: false };
      h.done = h.splitAces || score(h.cards).best === 21;
      return h;
    });
  }
}

/** Net chips for one hand once the dealer's final hand is known. */
function netOf(round, hand) {
  if (round.dealerNatural) return isNatural(hand) ? 0 : -hand.bet;
  if (isNatural(hand)) return 1.5 * hand.bet;
  const h = score(hand.cards);
  if (h.bust) return -hand.bet;
  const d = score(round.dealer);
  if (d.bust) return hand.bet;
  if (h.best > d.best) return hand.bet;
  if (h.best < d.best) return -hand.bet;
  return 0;
}

// ── Rendering ────────────────────────────────────────────────────────────

function cardEl(rank, { hidden = false, animate = false, delay = 0 } = {}) {
  const el = document.createElement('div');
  el.className = 'bj-card';
  if (hidden) {
    el.classList.add('back');
  } else {
    if (rank === 1) el.classList.add('ace');
    if (rank > 10) el.classList.add('face');
    el.innerHTML = `<span class="rk">${rankLabel(rank)}</span><span class="pip">${rankValue(rank)}</span>`;
  }
  if (animate) {
    el.classList.add('deal');
    el.style.animationDelay = `${delay}s`;
  }
  return el;
}

/**
 * Draw the table. A card animates only the first time it appears: `round.seen` remembers the cards already on
 * screen, so re-rendering after a player action does not replay the whole deal.
 */
function renderTable(round, { reveal = false } = {}) {
  const dealerEl = $('dealer-cards');
  dealerEl.innerHTML = '';
  round.dealer.forEach((rank, i) => {
    const hidden = i === 1 && !reveal;
    const key = `dealer-${i}-${hidden ? 'back' : 'face'}`;
    const fresh = !round.seen.has(key);
    round.seen.add(key);
    dealerEl.appendChild(cardEl(rank, { hidden, animate: fresh, delay: i * 0.12 }));
  });
  const d = score(round.dealer);
  if (!reveal) $('dealer-total').textContent = `· showing ${rankValue(round.dealer[0])}`;
  else $('dealer-total').textContent = d.bust ? `· BUST ${d.hard}` : `· ${d.best}`;

  const handsEl = $('hands');
  handsEl.innerHTML = '';
  const active = current(round);
  round.hands.forEach((hand, hi) => {
    const box = document.createElement('div');
    box.className = `bj-hand${hand === active ? ' active' : ''}${hand.done ? ' done' : ''}`;
    hand.cards.forEach((rank, ci) => {
      const key = `hand-${hi}-${ci}-${rank}`;
      const fresh = !round.seen.has(key);
      round.seen.add(key);
      box.appendChild(cardEl(rank, { animate: fresh, delay: ci * 0.12 }));
    });
    const s = score(hand.cards);
    let text = s.soft ? `soft ${s.best}` : `${s.best}`;
    if (s.bust) text = `BUST ${s.hard}`;
    else if (isNatural(hand)) text = 'BLACKJACK';
    const label = document.createElement('div');
    label.className = 'bj-hand-label';
    label.textContent = `${round.hands.length > 1 ? `HAND ${hi + 1}` : 'YOU'} · ${text} · bet ${hand.bet}`;
    box.appendChild(label);
    handsEl.appendChild(box);
  });
}

function updateControls() {
  const round = state.round;
  const hand = round ? current(round) : null;
  const idle = !state.busy && (round === null || hand === null);
  const broke = idle && state.chips < MIN_BET;
  const dealBtn = $('btn-deal');
  dealBtn.disabled = !(idle && (state.chips >= state.bet || broke));
  dealBtn.querySelector('.btn-txt').textContent = broke ? '↺ NEW BANKROLL' : '▶ DEAL';
  const legal = round && hand ? legalActions(round) : [];
  for (const a of ORDER) $(`btn-${a}`).disabled = state.busy || !legal.includes(a);
  $('btn-hint').disabled = state.busy || !hand;
  document.querySelectorAll('.bj-bet').forEach((b) => {
    const value = Number(b.dataset.bet);
    b.disabled = !idle || value > state.chips;
    b.classList.toggle('sel', value === state.bet);
  });
  $('shoe-left').textContent = String(Math.max(0, state.shoe.length - state.pos));
  $('chip-bankroll').textContent = Math.round(state.chips).toLocaleString('en-US');
  const s = state.stats;
  $('chip-record').textContent = `${s.won} · ${s.lost} · ${s.pushed}`;
  $('chip-net').textContent = `${s.net >= 0 ? '+' : ''}${Math.round(s.net).toLocaleString('en-US')}`;
  $('chip-net').style.color = s.net >= 0 ? 'var(--green)' : 'var(--red)';
}

function log(text, cls = 'log-info') {
  const line = document.createElement('div');
  line.className = cls;
  line.textContent = text;
  $('log').appendChild(line);
  $('log').scrollTop = $('log').scrollHeight;
}

// ── Advisor ──────────────────────────────────────────────────────────────

function renderAdvice(res, hand) {
  const list = $('advice-list');
  list.innerHTML = '';
  if (!res || !hand) {
    $('advice-best').textContent = 'Deal to see the expected value of each move.';
    return;
  }
  const evs = res.actions;
  const scale = Math.max(0.05, ...Object.values(evs).map((v) => Math.abs(v)));
  for (const a of ORDER) {
    if (!(a in evs)) continue;
    const ev = evs[a];
    const chips = ev * hand.bet;
    const row = document.createElement('div');
    row.className = `bj-advice-row${a === res.best ? ' best' : ''}`;
    row.innerHTML = `<span class="nm">${LABEL[a]}</span>`
      + `<span class="bar"><i class="${ev >= 0 ? 'pos' : 'neg'}" style="width:${(100 * Math.abs(ev)) / scale}%"></i></span>`
      + `<span class="sc">${chips >= 0 ? '+' : ''}${chips.toFixed(2)}</span>`;
    list.appendChild(row);
  }
  $('advice-best').textContent = res.blackjack ? 'Blackjack: take the 3:2 payout.'
    : `Basic strategy: ${LABEL[res.best]}.`;
  $('advice-note').textContent = `Expected chips for this hand (${res.total}${res.soft ? ', soft' : ''}).`;
}

/** Ask the server for the expected value of each legal move for the hand in front of the player. */
async function refreshAdvice() {
  const round = state.round;
  const hand = round ? current(round) : null;
  const seq = ++state.adviceSeq;
  if (!hand) {
    renderAdvice(null, null);
    highlightCurrent(null, null);
    return;
  }
  try {
    const res = await postJSON('/api/blackjack/advice', {
      cards: hand.cards,
      upcard: round.dealer[0],
      from_split: hand.fromSplit,
      split_aces: hand.splitAces,
    });
    if (seq !== state.adviceSeq) return;
    renderAdvice(res, hand);
    highlightCurrent(hand, round);
  } catch (err) {
    $('advice-note').textContent = `Advisor offline (${err.message}).`;
  }
}

// ── Strategy table and learner grid ──────────────────────────────────────

function hexToRgba(hex, alpha) {
  const n = parseInt(hex.slice(1), 16);
  return `rgba(${(n >> 16) & 255},${(n >> 8) & 255},${n & 255},${alpha})`;
}

/** The table row that matches the hand in front of the player, or null when the table has no such row. */
function rowFor(hand) {
  const s = score(hand.cards);
  if (hand.cards.length === 2 && !hand.fromSplit && rankValue(hand.cards[0]) === rankValue(hand.cards[1])) {
    return { kind: 'pair', total: rankValue(hand.cards[0]) };
  }
  if (s.bust) return null;
  if (s.soft && s.best >= 13 && s.best <= 20) return { kind: 'soft', total: s.best };
  if (!s.soft && s.best >= 5 && s.best <= 19) return { kind: 'hard', total: s.best };
  return null;
}

/** Outline the cell for the player's hand. Two-card hands get the solid outline; longer hands get a dashed one. */
function highlightCurrent(hand, round) {
  document.querySelectorAll('#strategy-grid .bj-cell').forEach((c) => c.classList.remove('current', 'ref'));
  const row = hand ? rowFor(hand) : null;
  if (!row) return;
  const upcard = rankValue(round.dealer[0]);
  const cell = document.querySelector(
    `#strategy-grid .bj-cell[data-kind="${row.kind}"][data-total="${row.total}"][data-upcard="${upcard}"]`);
  if (cell) cell.classList.add(hand.cards.length === 2 ? 'current' : 'ref');
}

/** Build a table of rows by upcard. `variant` is 'strategy' (exact, with glow) or 'learn' (the learner's guess). */
function buildGrid(el, rows, upcards, variant) {
  el.innerHTML = '';
  const head = document.createElement('div');
  head.className = 'bj-row head';
  head.innerHTML = '<span class="bj-label"></span>'
    + upcards.map((u) => `<span class="bj-up">${u === 1 ? 'A' : u === 10 ? 'T' : u}</span>`).join('');
  el.appendChild(head);
  for (const row of rows) {
    const line = document.createElement('div');
    line.className = 'bj-row';
    const label = document.createElement('span');
    label.className = 'bj-label';
    label.textContent = row.label;
    line.appendChild(label);
    for (const cell of row.cells) {
      const b = document.createElement('span');
      b.className = `bj-cell ${variant} ${row.kind}`;
      b.dataset.kind = row.kind;
      b.dataset.total = String(row.total);
      b.dataset.upcard = String(cell.upcard);
      const letter = LETTER[cell.action];
      b.textContent = letter;
      if (variant === 'strategy') {
        // Brighter when the best move is worth more per bet (EV runs from -1 to +1).
        const alpha = 0.18 + 0.6 * Math.min(1, Math.max(0, (cell.ev + 1) / 2));
        b.style.setProperty('--glow', hexToRgba(COLOR[letter], alpha));
        b.style.color = COLOR[letter];
        const up = cell.upcard === 1 ? 'ace' : cell.upcard;
        b.title = `${row.label} vs ${up}: ${LABEL[cell.action]} (EV ${cell.ev.toFixed(3)} per bet)`;
      }
      line.appendChild(b);
    }
    el.appendChild(line);
  }
}

function buildTables() {
  const data = state.strategy;
  buildGrid($('strategy-grid'), data.rows, data.upcards, 'strategy');
  buildGrid($('learn-grid'), data.rows, data.upcards, 'learn');
  $('strategy-note').textContent = 'Exact solution for an infinite deck. Hover a cell for its expected value.';
  $('chip-edge').textContent = `${(100 * data.house_edge).toFixed(2)}%`;
}

/** Paint the learner's greedy table: green where it matches the exact move, dim where it does not. */
function showSnapshot(i) {
  const data = state.learnData;
  const snap = data.snapshots[i];
  document.querySelectorAll('#learn-grid .bj-cell').forEach((c, idx) => {
    const letter = snap.actions[idx];
    c.textContent = letter === '?' ? '·' : letter;
    c.className = `bj-cell learn ${c.dataset.kind}`;
    if (letter === '?') c.classList.add('unk');
    else if (letter === data.exact[idx]) c.classList.add('ok');
    else c.classList.add('miss');
    c.style.color = COLOR[letter] || 'var(--dim)';
  });
  $('learn-episodes').textContent = snap.episodes.toLocaleString('en-US');
  $('learn-agree').textContent = `${Math.round(100 * snap.agreement)}%`;
  $('learn-loss').textContent = snap.loss.toFixed(3);
  drawChart(i);
}

/** Agreement with the exact table over hands played, drawn on a canvas. Points up to index `upto` are shown. */
function drawChart(upto) {
  const canvas = $('learn-chart');
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const w = canvas.clientWidth;
  const h = canvas.clientHeight;
  canvas.width = w * dpr;
  canvas.height = h * dpr;
  const ctx = canvas.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);
  const pad = 26;
  const snaps = state.learnData.snapshots;
  const x = (k) => pad + (k / Math.max(1, snaps.length - 1)) * (w - pad - 10);
  const y = (v) => h - pad - v * (h - 2 * pad);
  ctx.font = '11px JetBrains Mono, monospace';
  for (const v of [0, 0.5, 1]) {
    ctx.strokeStyle = 'rgba(0,245,255,0.18)';
    ctx.beginPath();
    ctx.moveTo(pad, y(v));
    ctx.lineTo(w - 8, y(v));
    ctx.stroke();
    ctx.fillStyle = '#4a7a9b';
    ctx.fillText(`${Math.round(100 * v)}%`, 2, y(v) + 4);
  }
  ctx.beginPath();
  snaps.slice(0, upto + 1).forEach((s, k) => (k ? ctx.lineTo(x(k), y(s.agreement)) : ctx.moveTo(x(k), y(s.agreement))));
  ctx.strokeStyle = '#00f5ff';
  ctx.shadowColor = '#00f5ff';
  ctx.shadowBlur = 10;
  ctx.lineWidth = 2;
  ctx.stroke();
  ctx.shadowBlur = 0;
  ctx.fillStyle = '#ff00a0';
  for (let k = 0; k <= upto; k++) {
    ctx.beginPath();
    ctx.arc(x(k), y(snaps[k].agreement), 3, 0, Math.PI * 2);
    ctx.fill();
  }
}

/** Run a bounded learning job on the server, then step through its snapshots so the table visibly converges. */
async function runLearn() {
  if (state.learning) return;
  state.learning = true;
  $('btn-learn').disabled = true;
  $('learn-status').textContent = 'Playing simulated hands on the server…';
  try {
    const episodes = Number($('learn-hands').value);
    const seed = 1 + Math.floor(Math.random() * 1_000_000_000);
    state.learnData = await postJSON('/api/blackjack/learn', { episodes, seed, snapshots: 16 });
    const n = state.learnData.snapshots.length;
    for (let i = 0; i < n; i++) {
      showSnapshot(i);
      const hands = state.learnData.snapshots[i].episodes.toLocaleString('en-US');
      $('learn-status').textContent = `Snapshot ${i + 1} of ${n}: the learner's moves after ${hands} hands.`;
      await sleep(260);
    }
    const last = state.learnData.snapshots[n - 1];
    $('learn-status').textContent = `After ${last.episodes.toLocaleString('en-US')} hands the learner agrees with the `
      + `exact table on ${Math.round(100 * last.agreement)}% of the clear cells. More hands sharpen the close calls.`;
  } catch (err) {
    $('learn-status').textContent = `✗ ${err.message}`;
  } finally {
    state.learning = false;
    $('btn-learn').disabled = false;
  }
}

// ── Game flow ────────────────────────────────────────────────────────────

async function deal() {
  if (state.busy || state.chips < state.bet) return;
  state.busy = true;
  const round = startRound(state.bet);
  state.round = round;
  $('msg').textContent = 'Dealing…';
  renderTable(round, { reveal: false });
  updateControls();
  await sleep(450);
  state.busy = false;
  if (round.dealerNatural || isNatural(round.hands[0])) {
    await finish(round);
    return;
  }
  $('msg').textContent = 'Your move.';
  updateControls();
  refreshAdvice();
}

async function act(action) {
  const round = state.round;
  if (state.busy || !round || !legalActions(round).includes(action)) return;
  state.busy = true;
  state.adviceSeq++;
  applyAction(round, action);
  log(LABEL[action], 'log-move');
  renderTable(round, { reveal: false });
  state.busy = false;
  if (current(round)) {
    $('msg').textContent = 'Your move.';
    updateControls();
    refreshAdvice();
  } else {
    await finish(round);
  }
}

/** Reveal the hole card, let the dealer draw, pay every hand, and show what happened. */
async function finish(round) {
  state.busy = true;
  state.adviceSeq++;
  renderAdvice(null, null);
  highlightCurrent(null, null);
  renderTable(round, { reveal: true });
  const live = round.hands.some((h) => !score(h.cards).bust && !isNatural(h));
  if (!round.dealerNatural && live) {
    while (score(round.dealer).best < 17) {
      await sleep(550);
      round.dealer.push(draw());
      renderTable(round, { reveal: true });
    }
    await sleep(300);
  }
  const nets = round.hands.map((h) => netOf(round, h));
  const total = nets.reduce((a, b) => a + b, 0);
  state.chips += total;
  state.stats.net += total;
  round.hands.forEach((h, i) => {
    state.stats.hands++;
    const net = nets[i];
    if (net > 0) state.stats.won++;
    else if (net < 0) state.stats.lost++;
    else state.stats.pushed++;
    const who = round.hands.length > 1 ? `Hand ${i + 1}: ` : '';
    const outcome = net > 0 ? `wins ${net}` : net < 0 ? `loses ${-net}` : 'pushes';
    log(`${who}${outcome}`, net > 0 ? 'log-best' : net < 0 ? 'log-err' : 'log-info');
  });
  showOutcome(round, nets, total);
  state.busy = false;
  $('msg').textContent = describeResult(round, nets, total);
  updateControls();
}

function describeResult(round, nets, total) {
  if (round.dealerNatural && nets.every((n) => n <= 0)) return 'Dealer blackjack. Better luck next hand.';
  if (round.hands.length === 1 && isNatural(round.hands[0])) return `Blackjack pays ${nets[0]}.`;
  const verb = total > 0 ? 'You win' : total < 0 ? 'You lose' : 'Push';
  const amount = total ? ` ${Math.abs(total)} chips` : '';
  const broke = state.chips < MIN_BET ? ' You are out of chips: start a new bankroll.' : ' Deal again when ready.';
  return `${verb}${amount}.${broke}`;
}

function showOutcome(round, nets, total) {
  if (round.dealerNatural && nets.every((n) => n <= 0)) {
    shake($('table'), 'big');
    banner('DEALER BLACKJACK', 'the peek ends the round', '#ff3b3b');
    return;
  }
  if (round.hands.length === 1 && isNatural(round.hands[0])) {
    burst($('hands'), { count: 90, colors: ['#ffe600', '#00f5ff', '#ff00a0'] });
    banner('BLACKJACK', `pays ${nets[0]} chips`, '#ffe600');
    return;
  }
  if (round.hands.some((h) => score(h.cards).bust)) shake($('table'), 'small');
  if (total > 0) {
    burst($('hands'), { count: 60, colors: ['#00ff88', '#00f5ff', '#ffe600'] });
    pop($('hands'), `+${total}`, '#00ff88');
  } else if (total < 0) {
    pop($('hands'), `${total}`, '#ff3b3b');
  } else {
    pop($('hands'), 'PUSH', '#00f5ff');
  }
}

function resetBankroll() {
  state.chips = START_CHIPS;
  state.round = null;
  state.bet = 10;
  $('hands').innerHTML = '';
  $('dealer-cards').innerHTML = '';
  $('dealer-total').textContent = '';
  $('msg').textContent = 'New bankroll. Choose a bet and deal.';
  renderAdvice(null, null);
  log(`↺ New bankroll: ${START_CHIPS} chips`, 'log-info');
  updateControls();
}

function dealOrReset() {
  if (state.chips < MIN_BET && !state.busy) resetBankroll();
  else deal();
}

function onKey(e) {
  if (['INPUT', 'SELECT', 'TEXTAREA'].includes(e.target.tagName)) return;
  const k = e.key.toLowerCase();
  if (k === 't') act('hit');
  else if (k === 's') act('stand');
  else if (k === 'd') act('double');
  else if (k === 'p') act('split');
  else if (k === 'h' && state.round && current(state.round) && !state.busy) refreshAdvice();
  else if (k === 'enter' || k === 'n') {
    // Enter on a focused button only presses that button, so a focused toggle or bet chip does not also deal.
    if (k === 'enter' && ['BUTTON', 'A'].includes(e.target.tagName)) return;
    if (!$('btn-deal').disabled) dealOrReset();
  }
}

async function init() {
  $('btn-deal').onclick = dealOrReset;
  $('btn-hit').onclick = () => act('hit');
  $('btn-stand').onclick = () => act('stand');
  $('btn-double').onclick = () => act('double');
  $('btn-split').onclick = () => act('split');
  $('btn-hint').onclick = () => {
    const hand = state.round ? current(state.round) : null;
    if (hand) log(`✦ Hint: ${hand.cards.map(rankLabel).join(' ')}`, 'log-best');
    refreshAdvice();
  };
  $('btn-learn').onclick = runLearn;
  document.querySelectorAll('.bj-bet').forEach((b) => {
    b.onclick = () => {
      state.bet = Number(b.dataset.bet);
      updateControls();
    };
  });
  addEventListener('keydown', onKey);
  shuffleShoe();
  renderAdvice(null, null);
  updateControls();
  try {
    state.strategy = await getJSON('/api/blackjack/strategy');
  } catch (err) {
    log(`✗ Server unreachable (${err.message}).`, 'log-err');
    $('strategy-note').textContent = 'The strategy table needs the server.';
    return;
  }
  buildTables();
  log('Shoe shuffled: six decks, the dealer stands on all 17s.', 'log-info');
}

init();
