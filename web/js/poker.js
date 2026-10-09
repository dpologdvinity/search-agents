// Poker page: heads-up Leduc hold'em against the CFR+ bot or against chance, plus a watch mode where the
// CFR+ AI plays chance.
//
// The server holds each hand (so the bot's card and mix stay on the server until showdown). A CFR+ bot's
// move is a lookup in the committed average strategy; a chance bot's move is drawn by the server from fixed
// odds that ignore the card. This page renders what comes back: the table, the bot's actions, its
// probability bars once the showdown reveals them, the explainer for its bets, and the training chart.
// The strategy table is fetched once, so the explainer can compare the bot's mix across cards in the same
// betting spot without knowing which card the bot holds.
//
// Watch mode has no human. The page plays the CFR+ seat itself: each move is sampled from the table row
// the server already sends for seat 0 (view.hint), and the chance seat is answered by the server.

import { getJSON, postJSON } from './api.js';
import { banner, burst, pop, shake } from './fx.js';

const $ = (id) => document.getElementById(id);
const REDUCED = matchMedia('(prefers-reduced-motion: reduce)').matches;
const RANKS = ['J', 'Q', 'K'];
const NAME = { f: 'fold', k: 'check', c: 'call', b: 'bet', r: 'raise' };
const LABEL = { f: 'FOLD', k: 'CHECK', c: 'CALL', b: 'BET', r: 'RAISE' };
const ORDER = ['f', 'k', 'c', 'b', 'r'];
const BIG_BLIND = 2;
const BOT_VERB = { f: 'folds', k: 'checks', c: 'calls', b: 'bets', r: 'raises' };
// The chance table groups actions into three kinds (see poker/chance.py): fold, call or check, bet or raise.
const KIND = { f: 'fold', k: 'call', c: 'call', b: 'bet', r: 'bet' };
const KIND_LABEL = { fold: 'FOLD', call: 'CALL / CHECK', bet: 'BET / RAISE' };
const KIND_ORDER = ['fold', 'call', 'bet'];
// Pause after each move in watch mode. The fastest setting sends about 100 moves a minute, under the server's
// limit of 120; a hand takes at least one move and the gap below, so deals stay under the limit of 30 a minute.
const WATCH_MS = { slow: 1500, normal: 1000, fast: 600 };
const HAND_GAP_MS = 2200;

const S = {
  hand: null,      // id of the hand on the server
  view: null,      // latest view from the server
  meta: null,
  table: null,     // the committed strategy: {information set: {action: probability}}
  training: null,  // exploitability curves
  mode: 'play',    // 'play' (you act) or 'watch' (the CFR+ AI acts, against chance)
  opponent: 'cfr', // who you play in play mode: 'cfr' or 'chance'
  // Session totals per mode. Net is in chips for seat 0 of the server: you in play mode, the CFR+ AI in watch.
  play: { net: 0, hands: 0 },
  watch: { net: 0, hands: 0, wins: 0, losses: 0, splits: 0 },
  busy: false,
  handMoves: [],   // the bot's decisions in this hand: actions and spots, plus the full mixes after the showdown
  lastBot: null,   // the bot's most recent decision
  hintMode: false, // the mix panel shows your equilibrium mix instead of the bot's
};

const pct = (p) => `${Math.round(p * 100)}%`;
const mbb = (chips) => (chips * 1000 / BIG_BLIND).toFixed(3);
const stats = () => (S.mode === 'watch' ? S.watch : S.play);
const opponent = () => (S.mode === 'watch' ? 'chance' : S.opponent); // watch mode always pits chance against the AI
const chanceOdds = () => S.meta?.opponents.find((o) => o.name === 'chance')?.action_odds;

/** Draw one action from a {action: probability} row, the same way the server's sample() does. */
function sampleRow(row) {
  const r = Math.random();
  let acc = 0;
  let last = null;
  for (const [a, p] of Object.entries(row)) {
    acc += p;
    last = a;
    if (r < acc) return a;
  }
  return last; // float round-off can leave acc just under one
}

/** A mix as text for the log, most likely action first, e.g. "bet 40%, check 60%". Actions it never plays are left out. */
const mixText = (row) => Object.entries(row)
  .filter(([, p]) => p > 0.0005)
  .sort((x, y) => y[1] - x[1])
  .map(([a, p]) => `${NAME[a]} ${pct(p)}`)
  .join(', ');

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
  const watch = S.mode === 'watch'; // in watch mode the page plays the AI, so the human controls stay off
  const legal = new Set(v?.legal || []);
  for (const a of ['k', 'b', 'c', 'r', 'f']) $(`btn-${a}`).disabled = busy || watch || !legal.has(a);
  $('btn-deal').disabled = busy || watch || (v !== null && v !== undefined && !v.terminal);
  $('btn-hint').disabled = busy || watch || !(v && v.hint);
  $('btn-hint').setAttribute('aria-pressed', String(S.hintMode));
  // The opponent cannot change in the middle of a hand, or while watching.
  for (const b of document.querySelectorAll('[data-opp]')) b.disabled = busy || watch || Boolean(S.hand);
  updateWatchControls();
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

/** Probability bars for a set of actions; the chosen one is highlighted. Labels and order default to actions. */
function renderBars(probs, chosen = null, labels = LABEL, order = ORDER) {
  const box = $('mix-bars');
  box.replaceChildren();
  for (const a of order) {
    if (!(a in probs)) continue;
    const row = document.createElement('div');
    row.className = `pk-bar-row${a === chosen ? ' chosen' : ''}`;
    const name = document.createElement('span');
    name.className = 'pk-bar-name';
    name.textContent = labels[a];
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

/** Chance's mix: the same three kinds at every card. Shown before the showdown, with the kind it chose highlighted. */
function renderChanceMix() {
  const odds = chanceOdds();
  if (!odds) return;
  const m = S.lastBot;
  const probs = { fold: odds.fold, call: odds.call_or_check, bet: odds.bet_or_raise };
  $('mix-title').textContent = m
    ? `CHANCE'S ROUND ${m.round} DECISION: it chose ${LABEL[m.action]}`
    : "CHANCE'S FIXED ODDS: the same at every card";
  renderBars(probs, m ? KIND[m.action] : null, KIND_LABEL, KIND_ORDER);
  $('mix-note').textContent = 'Chance reads no card: each action is drawn from these odds, over the legal actions only.';
}

function renderMix() {
  const v = S.view;
  if (S.hintMode && v && v.hint) {
    $('mix-title').textContent = 'YOUR EQUILIBRIUM MIX: the probability of each action for your card here';
    renderBars(v.hint.probs);
    $('mix-note').textContent = 'The equilibrium mix for your card in this spot.';
    return;
  }
  if (opponent() === 'chance') return renderChanceMix();
  if (S.lastBot) {
    const m = S.lastBot;
    const round = m.round === 1 ? 'round 1' : 'round 2';
    const title = `THE BOT'S ${round.toUpperCase()} DECISION: it chose ${LABEL[m.action]}`;
    if (!m.probs) {
      // Mid-hand the mix depends on the bot's hidden card, so the page shows the action and nothing else.
      $('mix-title').textContent = `${title}. Its mix is shown at the showdown.`;
      $('mix-note').textContent = 'The mix depends on the bot\'s hidden card. The explainer shows its bet rate for each card.';
      $('mix-bars').replaceChildren();
      return;
    }
    $('mix-title').textContent = `${title}, ${pct(m.probs[m.action])} of the time`;
    renderBars(m.probs, m.action);
    $('mix-note').textContent = 'Its card is now shown. The mix depends only on what the bot could see.';
    return;
  }
  $('mix-title').textContent = 'Waiting for the bot\'s first decision.';
  $('mix-bars').replaceChildren();
}

// ── Explainer ───────────────────────────────────────────────────────────

/** Share of the time the bot bets (or raises) with each card in the same spot. Uses the committed table.
 *  The spot has its card replaced by "?", so this works before the showdown too. */
function aggression(move) {
  return RANKS.map((r) => {
    const key = move.spot.replace('?', r);
    const row = S.table[key];
    return { rank: r, p: row ? (row.b || 0) + (row.r || 0) : null };
  });
}

/** Chance's bets come from fixed odds, so there is no bluff to explain: the same odds apply to every card. */
function renderChanceExplain() {
  const el = $('explain');
  const v = S.view;
  const odds = chanceOdds();
  let html = `<p><span class="value">Chance is a dice roll.</span> Its odds are one table, ${odds ? `fold ${pct(odds.fold)}, call or check ${pct(odds.call_or_check)}, bet or raise ${pct(odds.bet_or_raise)}` : 'a fixed table'}, whatever its card.</p>`;
  if (v && v.terminal) {
    html += `<p>It held <b>${v.bot_card}</b>. Its card did not change any decision in this hand.</p>`;
  }
  el.innerHTML = html;
}

function renderExplain() {
  if (opponent() === 'chance') return renderChanceExplain();
  const el = $('explain');
  const v = S.view;
  const bets = S.handMoves.filter((m) => m.action === 'b' || m.action === 'r');
  const move = bets[0] || S.handMoves[0];
  if (!move) {
    el.innerHTML = 'The bot has not acted yet. Its bet rate for each card appears here once it bets.';
    return;
  }
  const rows = aggression(move);
  const table = `<table><tr><th>card</th><th>bets here</th></tr>${rows.map((x) =>
    `<tr${v && v.terminal && x.rank === v.bot_card ? ' style="color:var(--yellow)"' : ''}><td>${x.rank}</td><td>${x.p === null ? '—' : pct(x.p)}</td></tr>`).join('')}</table>`;
  const why = `
    <p>How often the bot bets with each card in this spot:</p>
    ${table}`;
  if (!v || !v.terminal) {
    el.innerHTML = `${why}<p>Its card is hidden until the showdown.</p>`;
    return;
  }
  const held = v.bot_card;
  const round1 = move.round === 1;
  let verdict;
  if (held === 'J') {
    verdict = `<p><span class="bluff">A bluff.</span> It held a <b>J</b>, the weakest rank. It loses to an unpaired Q or K at showdown, and it wins only if the public card is the other J, which makes a pair. The bet only wins if you fold.</p>
      ${round1 ? '<p>It bet before the public card was shown, so a pair was possible (about 1 time in 5).</p>' : ''}`;
  } else if (held === 'K') {
    verdict = '<p><span class="value">A value bet.</span> A <b>K</b> beats any unpaired Q or J at showdown, so the bot wants you to call. Its bets with weaker cards keep this one from being readable.</p>';
  } else {
    verdict = '<p>A <b>Q</b> sits in the middle: it beats an unpaired J and loses to an unpaired K. Its bet mixes value with pressure, at the frequency the table shows.</p>';
  }
  el.innerHTML = `${verdict}${why}`;
}

// ── Settling a hand and the session ────────────────────────────────────

function renderChips() {
  const st = stats();
  $('chip-net').textContent = `${st.net >= 0 ? '+' : ''}${st.net}`;
  $('chip-hands').textContent = String(st.hands);
}

/** Watch mode's running score: wins, losses and splits of the CFR+ AI, and its chip total. */
function renderScore() {
  const el = $('score');
  if (S.mode !== 'watch') {
    el.textContent = '';
    return;
  }
  const w = S.watch;
  const net = `${w.net >= 0 ? '+' : ''}${w.net}`;
  el.textContent = `CFR+ vs Chance over ${w.hands} hands: ${w.wins} won, ${w.losses} lost, ${w.splits} split, net ${net} chips.`;
}

function settle(v) {
  // Seat 0 is the human in play mode and the CFR+ AI in watch mode, so p is that seat's chips in both.
  const p = v.payoff;
  const watch = S.mode === 'watch';
  const st = stats();
  st.net += p;
  st.hands += 1;
  if (watch) {
    if (p > 0) st.wins += 1;
    else if (p < 0) st.losses += 1;
    else st.splits += 1;
  }
  renderChips();
  renderScore();
  const win = watch ? 'CFR+ wins' : 'You win';
  const lose = watch ? 'CFR+ loses' : 'You lose';
  const verdict = v.result === 'win' ? `${win} ${p} chips.` : v.result === 'lose' ? `${lose} ${-p} chips.` : 'Split pot.';
  const who = opponent() === 'chance' ? 'Chance' : 'The bot';
  // A fold ends the hand before the public card is turned up, so the showdown details only apply when it was.
  const shown = v.public ? `, and the public card was ${v.public}` : ' (the hand ended before the public card)';
  const cls = v.result === 'win' ? 'win' : v.result === 'lose' ? 'lose' : '';
  say(`${verdict} ${who} held ${v.bot_card}${shown}.`, cls);
  log(`${verdict} ${who} held ${v.bot_card}${v.public ? `; public ${v.public}` : ''}.`, cls);
  if (!watch && v.result === 'win') {
    const botFolded = S.handMoves.some((m) => m.action === 'f');
    banner(`+${p} CHIPS`, botFolded ? 'the bot folded' : 'showdown', '#00ff88');
    burst($('table'));
  } else if (!watch && v.result === 'lose') {
    shake($('table'), 'small');
  }
  renderExplain();
  S.lastBot = S.handMoves[S.handMoves.length - 1] || S.lastBot;
}

function render(v) {
  S.view = v;
  renderTable(v);
  const watch = S.mode === 'watch';
  const status = v.terminal ? 'HAND OVER' : v.to_act === 0 ? (watch ? 'CFR+ TO MOVE' : 'YOUR TURN') : 'BOT THINKING';
  $('chip-status').textContent = status;
  if (!v.terminal && v.to_act === 0 && !watch) {
    say(v.round === 2
      ? `The public card is ${v.public}. Your turn in round 2.`
      : 'Your turn. Check, bet, call, raise or fold.');
  }
  setButtons(v);
  renderMix();
}

// Each action runs one round trip; the bot's answers arrive with the reply.
async function deal() {
  if (S.mode === 'watch') return; // watch mode deals its own hands
  await dealHand();
}

/** Deal a hand against the current opponent. Returns false if it could not be dealt. */
async function dealHand() {
  if (S.busy || S.hand) return false; // a hand in progress must finish first
  S.busy = true;
  $('chip-status').textContent = 'DEALING';
  setButtons(S.view);
  try {
    const v = await postJSON('/api/poker/deal', { opponent: opponent() });
    S.hand = v.hand;
    S.handMoves = [];
    S.lastBot = null;
    S.hintMode = false;
    log(`HAND ${stats().hands + 1}: ${S.mode === 'watch' ? 'the CFR+ AI acts first' : 'you act first'}`, 'hand');
    render(v);
    renderExplain();
    return true;
  } catch (e) {
    say(`Could not deal: ${e.message}`, 'lose');
    return false;
  } finally {
    S.busy = false;
    setButtons(S.view);
  }
}

/** One move. `actor` names the CFR+ seat in watch mode; otherwise the action is yours. Returns success. */
async function act(action, actor = null) {
  if (S.busy || !S.hand) return false;
  S.busy = true;
  const round = S.view ? S.view.round : 1;
  const size = action === 'b' || action === 'r' ? ` ${round === 1 ? 2 : 4}` : '';
  // The CFR+ seat's mix comes from the table row the server sent with this turn (view.hint).
  const mix = actor && S.view?.hint ? S.view.hint.probs : null;
  $('chip-status').textContent = actor ? 'CHANCE THINKING' : 'BOT THINKING'; // shown until the reply says whose move it is
  setButtons(S.view);
  try {
    const v = await postJSON('/api/poker/act', { hand: S.hand, action });
    if (actor) log(`${actor} ${BOT_VERB[action]}${size}${mix ? ` (its mix: ${mixText(mix)})` : ''}`, 'you');
    else log(`you ${NAME[action]}`, 'you');
    const who = opponent() === 'chance' ? 'chance' : 'the bot';
    for (const m of v.bot_moves) {
      S.handMoves.push(m);
      S.lastBot = m;
      log(`${who} ${NAME[m.action]}${m.action === 'b' || m.action === 'r' ? ` ${m.round === 1 ? 2 : 4}` : ''}`, 'bot');
      if (m.action === 'b' || m.action === 'r') pop($('bot-cards'), m.action === 'b' ? 'BET' : 'RAISE', '#ff00a0');
    }
    if (v.terminal) {
      // The showdown reveals every decision of the hand, with the bot's card and the mix behind each one.
      S.handMoves = v.bot_reveal;
      S.lastBot = v.bot_reveal.at(-1) || null;
    }
    S.hintMode = false;
    render(v);
    renderExplain();
    if (v.terminal) {
      S.hand = null;
      settle(v);
    }
    return true;
  } catch (e) {
    say(e.message, 'lose');
    if (S.view && !S.view.terminal) $('chip-status').textContent = S.mode === 'watch' ? 'CFR+ TO MOVE' : 'YOUR TURN'; // the request failed: the hand is still live
    return false;
  } finally {
    S.busy = false;
    setButtons(S.view);
  }
}

// ── Watch mode: the CFR+ AI plays chance ────────────────────────────────

let watchTimer = null;     // the next watch tick, when one is scheduled
let watchRunning = false;  // START was pressed and PAUSE has not been
let watchOnce = false;     // STEP: run one tick while paused

function scheduleWatch(ms) {
  clearTimeout(watchTimer);
  watchTimer = setTimeout(watchTick, ms);
}

/** One tick: deal a hand if none is live, otherwise make the CFR+ seat's move. Then schedule the next tick. */
async function watchTick() {
  watchTimer = null;
  if (!watchRunning && !watchOnce) return;
  watchOnce = false;
  // The CFR+ seat draws its move from the row the server sent for this turn; chance answers inside the reply.
  const ok = S.hand ? await act(sampleRow(S.view.hint.probs), 'CFR+') : await dealHand();
  if (S.mode !== 'watch' || !watchRunning) return; // switched away, or paused while the request was in flight
  if (!ok) return scheduleWatch(3000); // e.g. rate limited: wait, then try the same step again
  if (S.hand) return scheduleWatch(WATCH_MS[$('watch-speed').value]);
  // The hand just settled: deal the next one after a short gap, or stop here when auto-restart is off.
  if ($('watch-auto').checked) return scheduleWatch(HAND_GAP_MS);
  stopWatch();
}

function startWatch() {
  watchRunning = true;
  updateWatchControls();
  scheduleWatch(0);
}

function stopWatch() {
  watchRunning = false;
  watchOnce = false;
  clearTimeout(watchTimer);
  watchTimer = null;
  updateWatchControls();
}

function stepWatch() {
  if (watchRunning) return;
  watchOnce = true;
  scheduleWatch(0);
}

function updateWatchControls() {
  const toggle = $('watch-toggle');
  toggle.querySelector('.btn-txt').textContent = watchRunning ? '❚❚ PAUSE' : '▶ START';
  toggle.classList.toggle('green', !watchRunning);
  toggle.classList.toggle('pink', watchRunning);
  toggle.setAttribute('aria-pressed', String(watchRunning));
  $('watch-step').disabled = watchRunning || S.busy;
}

// ── Modes, opponents, and the live line that names each side ────────────

/** Switch between playing a hand yourself and watching the CFR+ AI play chance. Ignored mid-request. */
function setMode(mode) {
  if (mode === S.mode || S.busy) return;
  stopWatch();
  S.mode = mode;
  S.hand = null; // a live hand is left on the server to expire; the table starts clean
  S.view = null;
  S.handMoves = [];
  S.lastBot = null;
  S.hintMode = false;
  $('log').replaceChildren();
  renderTable(null);
  $('chip-status').textContent = 'READY';
  $('watch-bar').hidden = mode !== 'watch';
  $('opp-group').hidden = mode === 'watch';
  for (const [id, on] of [['mode-play', mode === 'play'], ['mode-watch', mode === 'watch']]) {
    $(id).classList.toggle('active', on);
    $(id).setAttribute('aria-pressed', String(on));
  }
  say(mode === 'watch'
    ? 'Press START. The CFR+ AI plays its average strategy against chance, and the score builds below the log.'
    : 'Deal a hand to start. You act first in both rounds.');
  renderChips();
  renderScore();
  renderSides();
  renderMix();
  renderExplain();
  setButtons(null);
}

/** Choose the opponent for play mode. Only between hands. */
function setOpponent(opp) {
  if (opp === S.opponent || S.busy || S.hand) return;
  S.opponent = opp;
  for (const b of document.querySelectorAll('[data-opp]')) {
    const on = b.dataset.opp === opp;
    b.classList.toggle('active', on);
    b.setAttribute('aria-pressed', String(on));
  }
  S.hintMode = false;
  renderSides();
  renderMix();
  renderExplain();
}

/** Name the algorithm on each side: in the description under the title and in the live line above the table. */
function renderSides() {
  const meta = S.meta;
  if (!meta) return;
  // The exploitability chip measures the committed CFR+ table, so it only applies when that bot is the opponent.
  $('expl-chip').style.display = S.mode === 'play' && S.opponent === 'cfr' ? '' : 'none';
  const cfr = meta.opponents.find((o) => o.name === 'cfr');
  const chance = meta.opponents.find((o) => o.name === 'chance');
  if (S.mode === 'watch') {
      $('sides').textContent = `CFR+ AI: ${cfr.label}  ·  OPPONENT: ${chance.label}`;
    $('you-name').textContent = 'CFR+ AI';
    $('bot-name').textContent = 'CHANCE';
    return;
  }
  const chanceMode = S.opponent === 'chance';
  const opp = chanceMode ? chance : cfr;
  $('you-name').textContent = 'YOU';
  $('bot-name').textContent = chanceMode ? 'CHANCE' : 'BOT';
  $('sides').textContent = `YOU  ·  OPPONENT: ${opp.label}`;
}

/** The chance table under HOW IT WORKS, filled from the same numbers the server uses. */
function renderChanceOdds() {
  const odds = chanceOdds();
  if (!odds) return;
  const items = [
    `Fold ${pct(odds.fold)}`,
    `Call or check ${pct(odds.call_or_check)}`,
    `Bet or raise ${pct(odds.bet_or_raise)}`,
    'Renormalised over the legal actions at each decision, so the chosen shares always sum to one.',
  ];
  $('chance-odds').replaceChildren(...items.map((text) => {
    const li = document.createElement('li');
    li.textContent = text;
    return li;
  }));
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
    renderChanceOdds();
    renderSides();
    $('chip-expl').textContent = `${mbb(meta.exploitability)} mbb/hand`;
    const runs = training.runs.map((r) => `${r.algorithm} ${r.iterations.at(-1)}`).join(' · ');
    const kuhn = training.kuhn_check?.['cfr+'];
    $('chart-note').textContent = `Game value for seat 0: ${meta.value_seat0.toFixed(4)} chips. ${runs} iterations. ${kuhn ? `Kuhn check: CFR+ lands within ${kuhn.value_error.toExponential(1)} of the exact -1/18.` : ''}`;
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
$('mode-play').addEventListener('click', () => setMode('play'));
$('mode-watch').addEventListener('click', () => setMode('watch'));
for (const b of document.querySelectorAll('[data-opp]')) b.addEventListener('click', () => setOpponent(b.dataset.opp));
$('watch-toggle').addEventListener('click', () => (watchRunning ? stopWatch() : startWatch()));
$('watch-step').addEventListener('click', stepWatch);
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
