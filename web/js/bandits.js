// Bandits page: a casino floor of slot machines with hidden payouts.
//
// You pull the levers. Every agent in the lineup plays the same casino in parallel, pull by pull,
// on the same outcome table (bandits-core.js is a port of the Python package, and the two agree
// exactly for a seed). The race chart shows cumulative regret, the Thompson panel shows the
// posteriors the agent is sampling from, and the UCB panel shows the bounds it is comparing.
// STOP AND REVEAL shows the true payouts and scores you and the agents on the same pulls. The
// score comes from the server (POST /api/bandits/score, the Python agents), and the page's own
// numbers are the fallback if that call fails.

import { getJSON, postJSON } from './api.js';
import { banner, burst, pop, shake } from './fx.js';
import { CONSTANTS, Casino, Racer, laiRobbinsRate, scorePulls } from './bandits-core.js';

const $ = (id) => document.getElementById(id);
const LETTER = (i) => String.fromCharCode(65 + i);
const MACHINE_COLORS = ['#00f5ff', '#ff00a0', '#ffe600', '#00ff88', '#9b00ff', '#ff9d00', '#ffffff', '#ff3b3b', '#7dd3fc', '#bef264'];
const AGENT_COLORS = {
  greedy: '#ff3b3b', eps: '#ffe600', eps_decay: '#ff9d00', ucb1: '#00f5ff',
  thompson: '#ff00a0', exp3: '#9b00ff', sw_ucb: '#00ff88',
};
const PLAYER_COLOR = '#ffffff';
const JACKPOT_STREAK = 5;

const state = {
  meta: null,
  constants: CONSTANTS,
  labels: {},             // agent key -> label from meta
  kind: 'bernoulli',
  k: 5,
  pulls: 300,
  seed: 1,
  casino: null,           // Casino for the current game
  racers: {},             // agent key -> Racer, every lineup agent, always running
  visible: new Set(),     // agent keys shown in the charts
  player: { arms: [], rewards: [], regret: 0, optimal: 0, total: 0, streak: 0 },
  speed: 8,               // agent pulls per second
  running: true,
  revealed: false,
  scored: null,           // server score response for this game, if any
  bench: null,            // benchmark JSON, or null when it is not on the server
  acc: 0,
  last: 0,
  dirty: true,
  legendKey: '',
  postCache: null,        // {t, curves} for the Thompson panel
  busy: false,
};

// ── Setup ────────────────────────────────────────────────────────────────

function randomSeed() {
  return 1 + Math.floor(Math.random() * 2147483646);
}

function readControls() {
  state.kind = $('kind').value;
  state.k = Number($('machines').value);
  state.pulls = Number($('pulls').value);
  const seed = Math.floor(Number($('seed').value));
  state.seed = Number.isFinite(seed) && seed >= 0 && seed <= 2147483647 ? seed : randomSeed();
  $('seed').value = String(state.seed);
}

function lineup() {
  return state.meta.lineup[state.kind] || [];
}

function defaultVisible() {
  const keys = lineup();
  const on = ['ucb1', 'thompson'];
  if (state.kind === 'drifting') on.push('sw_ucb');
  return new Set(on.filter((k) => keys.includes(k)));
}

function renderAgentToggles() {
  const box = $('agents');
  box.innerHTML = '';
  for (const key of lineup()) {
    const label = document.createElement('label');
    label.className = 'bd-agent';
    label.style.setProperty('--col', AGENT_COLORS[key]);
    const cb = document.createElement('input');
    cb.type = 'checkbox';
    cb.checked = state.visible.has(key);
    cb.dataset.key = key;
    cb.addEventListener('change', () => {
      if (cb.checked) state.visible.add(key); else state.visible.delete(key);
      state.dirty = true;
    });
    const sw = document.createElement('span');
    sw.className = 'bd-swatch';
    sw.style.background = AGENT_COLORS[key];
    const text = document.createElement('span');
    text.textContent = state.labels[key] || key;
    label.append(cb, sw, text);
    box.append(label);
  }
}

function fillMachineOptions() {
  const sel = $('machines');
  if (sel.options.length) return;
  for (const n of [2, 3, 4, 5, 6, 7, 8, 9, 10]) {
    const o = document.createElement('option');
    o.value = String(n);
    o.textContent = `${n} machines`;
    sel.append(o);
  }
  sel.value = '5';
}

// ── Game lifecycle ───────────────────────────────────────────────────────

async function newGame() {
  if (state.busy) return;
  readControls();
  state.busy = true;
  setStatus('LOADING');
  try {
    const machines = await getJSON('/api/bandits/machines', {
      kind: state.kind, k: state.k, pulls: state.pulls, seed: state.seed,
    });
    state.casino = new Casino(machines, state.constants);
    state.game = (state.game || 0) + 1;
    state.racers = {};
    for (const key of lineup()) state.racers[key] = new Racer(key, state.casino, state.constants);
    state.player = { arms: [], rewards: [], regret: 0, optimal: 0, total: 0, streak: 0 };
    state.revealed = false;
    state.scored = null;
    state.acc = 0;
    state.visible = defaultVisible();
    state.postCache = null;
    $('reveal').hidden = true;
    $('log').innerHTML = '';
    renderAgentToggles();
    renderFloor();
    renderThinkingIdle();
    setStatus('LIVE');
    state.dirty = true;
  } catch (err) {
    setStatus('ERROR');
    $('floor-note').textContent = `Could not load the casino: ${err.message}`;
  } finally {
    state.busy = false;
  }
}

function setStatus(text) {
  $('chip-status').textContent = text;
}

// ── Floor: cabinets, levers, reels ───────────────────────────────────────

function renderFloor() {
  const floor = $('floor');
  floor.innerHTML = '';
  for (let i = 0; i < state.k; i++) {
    const b = document.createElement('button');
    b.type = 'button';
    b.className = 'bd-machine';
    b.dataset.arm = String(i);
    b.style.setProperty('--col', MACHINE_COLORS[i % MACHINE_COLORS.length]);
    b.setAttribute('aria-label', `Pull machine ${LETTER(i)}`);
    b.innerHTML = '<span class="bd-name"></span><span class="bd-key"></span>'
      + '<span class="bd-screen"><span class="bd-reel"><span>?</span></span>'
      + '<span class="bd-reel"><span>?</span></span><span class="bd-reel"><span>?</span></span></span>'
      + '<span class="bd-result"></span><span class="bd-lever" aria-hidden="true"></span>';
    b.querySelector('.bd-name').textContent = `MACHINE ${LETTER(i)}`;
    b.querySelector('.bd-key').textContent = i < 9 ? `key ${i + 1}` : (i === 9 ? 'key 0' : '');
    b.addEventListener('click', () => pull(i, b));
    floor.append(b);
  }
  $('floor-note').textContent = 'Click a lever (or press 1 to 9 and 0) to pull. Payouts stay hidden until the reveal.';
}

function machineButton(i) {
  return document.querySelector(`.bd-machine[data-arm="${i}"]`);
}

function setReels(button, symbols, cls) {
  const reels = button.querySelectorAll('.bd-reel span');
  reels.forEach((r, j) => { r.textContent = symbols[j]; });
  button.classList.remove('paid', 'missed');
  if (cls) button.classList.add(cls);
}

// Lever pull: the lever swings, the reels spin once, then the result lands.
function animatePull(button, reward, paid) {
  button.classList.remove('pulling', 'spin');
  void button.offsetWidth; // restart the CSS animations
  button.classList.add('pulling', 'spin');
  setTimeout(() => button.classList.remove('spin'), 520);
  const result = button.querySelector('.bd-result');
  if (state.kind === 'gaussian') {
    result.textContent = `paid ${reward >= 0 ? '+' : ''}${reward.toFixed(2)}`;
  } else {
    result.textContent = paid ? 'paid +1' : 'nothing';
  }
  const syms = paid || state.kind === 'gaussian' ? ['7', '7', '7'] : ['-', '-', '-'];
  setReels(button, syms, paid ? 'paid' : 'missed');
  button.addEventListener('animationend', () => button.classList.remove('pulling'), { once: true });
}

// ── Player pulls ─────────────────────────────────────────────────────────

function pull(arm, button) {
  if (state.revealed || !state.casino || arm < 0 || arm >= state.k) return;
  const p = state.player;
  const t = p.arms.length;
  if (t >= state.pulls) return;
  const casino = state.casino;
  const reward = casino.reward(t, arm);
  const means = casino.mean(t);
  const best = Math.max(...means);
  const optimal = means[arm] === best;
  const paid = state.kind === 'gaussian' ? reward > 0 : reward === 1;

  p.arms.push(arm);
  p.rewards.push(reward);
  p.total += reward;
  p.regret += best - means[arm];
  if (optimal) p.optimal += 1;
  p.streak = optimal ? p.streak + 1 : 0;

  const btn = button || machineButton(arm);
  animatePull(btn, reward, paid);
  if (paid) {
    pop(btn, state.kind === 'gaussian' ? `+${reward.toFixed(2)}` : '+1', '#ffe600');
    burst(btn, { count: 22, speed: 4 });
  } else {
    shake(btn, 'small');
  }
  addLog(`pull ${t + 1}: machine ${LETTER(arm)} -> ${state.kind === 'gaussian' ? reward.toFixed(2) : (paid ? 'paid' : 'nothing')}  (total ${p.total.toFixed(2)})`);

  if (p.streak === JACKPOT_STREAK) {
    banner('JACKPOT', `${JACKPOT_STREAK} best-machine pulls in a row`, '#ffe600');
    btn.classList.add('jackpot');
    setTimeout(() => btn.classList.remove('jackpot'), 2000);
  }
  if (p.arms.length >= state.pulls) {
    const game = state.game;
    setTimeout(() => { if (state.game === game) reveal(); }, 900); // a new casino may have started meanwhile
  }
  state.dirty = true;
}

function addLog(text) {
  const log = $('log');
  const row = document.createElement('div');
  row.className = 'bd-row';
  row.textContent = text;
  log.append(row);
  while (log.children.length > 200) log.firstChild.remove();
  log.scrollTop = log.scrollHeight;
}

// ── Agents: one pull each, in step with the others ──────────────────────

function stepAgents(n) {
  for (let i = 0; i < n; i++) {
    let moved = false;
    for (const racer of Object.values(state.racers)) {
      if (!racer.done) {
        racer.step();
        moved = true;
      }
    }
    if (!moved) break;
  }
  state.dirty = true;
}

function agentsDone() {
  return Object.values(state.racers).every((r) => r.done);
}

// The ticker: agents run on their own clock at the chosen speed, whatever the player is doing.
function frame(ts) {
  const dt = state.last ? Math.min(0.25, (ts - state.last) / 1000) : 0;
  state.last = ts;
  if (state.running && state.casino && !agentsDone()) {
    state.acc += state.speed * dt;
    const n = Math.min(60, Math.floor(state.acc));
    state.acc -= n;
    if (n > 0) stepAgents(n);
  }
  if (state.dirty) {
    state.dirty = false;
    draw();
  }
  requestAnimationFrame(frame);
}

// ── Reveal ───────────────────────────────────────────────────────────────

function reveal() {
  if (state.revealed || !state.casino) return;
  const count = state.player.arms.length;
  if (count === 0) {
    $('floor-note').textContent = 'Pull at least one machine before you reveal.';
    return;
  }
  state.revealed = true;
  setStatus('REVEALED');
  // Run every agent to the end of the game now, so the chart and the table show full runs.
  for (const racer of Object.values(state.racers)) while (!racer.done) racer.step();

  const casino = state.casino;
  const final = casino.schedule.length - 1;
  for (let i = 0; i < state.k; i++) {
    const b = machineButton(i);
    b.classList.add('revealed');
    const p = casino.schedule[final][i];
    b.querySelector('.bd-result').textContent = `${casino.kind === 'drifting' ? 'final p' : 'p'} ${p.toFixed(3)}`;
    b.classList.toggle('is-best', i === casino.schedule[final].indexOf(Math.max(...casino.schedule[final])));
  }
  renderOdds();

  const arms = state.player.arms.slice();
  const mine = scorePulls(casino, arms);
  const rows = [{ label: 'You', regret: mine.total, share: shareOf(mine.optimal), reward: sum(mine.rewards), player: true }];
  for (const key of lineup()) {
    const r = state.racers[key];
    // Each agent is scored on the first `count` pulls, the same pulls the player made.
    rows.push({
      label: state.labels[key] || key,
      regret: r.regret[count - 1],
      share: shareOf(r.optimal.slice(0, count)),
      reward: sum(r.rewards.slice(0, count)),
    });
  }
  renderRevealTable(rows, 'Page numbers: the JavaScript port, computed on this casino.');
  const best = rows.slice(1).reduce((a, b) => (b.regret < a.regret ? b : a));
  banner('REVEAL', `you ${mine.total.toFixed(2)} regret, best agent ${best.label} ${best.regret.toFixed(2)}`, '#00ff88');
  $('reveal').hidden = false;
  $('reveal').scrollIntoView({ behavior: 'smooth', block: 'start' });
  state.dirty = true;
  verifyWithServer(arms, mine.total);
}

// The server runs the Python agents on the same casino; its table replaces the page's numbers.
async function verifyWithServer(arms, pageRegret) {
  const note = $('reveal-note');
  note.textContent = 'Checking the score on the server...';
  try {
    const body = await postJSON('/api/bandits/score', {
      kind: state.kind, k: state.k, seed: state.seed, arms,
    });
    state.scored = body;
    const rows = [{ label: 'You', regret: body.player.regret, share: body.player.share_best, reward: body.player.reward, player: true }];
    for (const key of lineup()) {
      const a = body.agents[key];
      if (a) rows.push({ label: a.label || state.labels[key], regret: a.regret, share: a.share_best, reward: a.reward });
    }
    // The server rounds to 4 decimals, so compare at that precision.
    const agree = Math.abs(body.player.regret - pageRegret) < 5e-5;
    renderRevealTable(rows, agree
      ? 'Server-verified: the Python agents on this casino. The page and the server agree on your regret.'
      : 'Server-verified: the Python agents on this casino. The page and the server disagree on your regret; the server is the reference.');
  } catch (err) {
    note.textContent = `Server check unavailable (${err.message}). The table shows the page's own numbers.`;
  }
}

// Fraction of pulls that went to a machine tied for the best mean.
function shareOf(optimalFlags) {
  return optimalFlags.length ? optimalFlags.filter(Boolean).length / optimalFlags.length : 0;
}

function sum(xs) {
  let s = 0;
  for (const x of xs) s += x;
  return s;
}

function renderOdds() {
  const casino = state.casino;
  const box = $('reveal-odds');
  box.innerHTML = '';
  $('reveal-title').textContent = casino.schedule.length > 1
    ? 'The true payouts, redrawn every 500 pulls'
    : 'The true payouts';
  casino.schedule.forEach((seg, s) => {
    const start = s * casino.period + 1;
    const end = Math.min(casino.horizon, (s + 1) * casino.period);
    const best = seg.indexOf(Math.max(...seg));
    const row = document.createElement('div');
    row.className = 'bd-odds';
    const label = document.createElement('span');
    label.textContent = casino.schedule.length > 1 ? `pulls ${start}-${end}` : 'all pulls';
    label.style.borderStyle = 'dashed';
    row.append(label);
    seg.forEach((p, i) => {
      const c = document.createElement('span');
      c.textContent = `${LETTER(i)} ${p.toFixed(3)}`;
      if (i === best) c.className = 'best';
      row.append(c);
    });
    box.append(row);
  });
}

function renderRevealTable(rows, note) {
  const tbody = $('reveal-table').querySelector('tbody');
  tbody.innerHTML = '';
  for (const r of rows) {
    const tr = document.createElement('tr');
    if (r.player) tr.className = 'player';
    const cells = [
      r.label,
      r.regret.toFixed(2),
      `${(100 * r.share).toFixed(1)}%`,
      r.reward.toFixed(2),
    ];
    cells.forEach((text, i) => {
      const td = document.createElement('td');
      if (i > 0) td.className = 'num';
      td.textContent = text;
      tr.append(td);
    });
    tbody.append(tr);
  }
  $('reveal-note').textContent = note;
}

// ── Thinking panel and chips ─────────────────────────────────────────────

function renderThinkingIdle() {
  const box = $('think');
  box.innerHTML = '';
  const row = document.createElement('div');
  row.className = 'bd-row';
  row.textContent = 'Pull a lever or press Run. Each agent shows its reason for its latest pull here.';
  box.append(row);
}

function renderThinking() {
  const box = $('think');
  box.innerHTML = '';
  const playerLine = state.player.arms.length
    ? `machine ${LETTER(state.player.arms.at(-1))} -> ${state.kind === 'gaussian'
      ? state.player.rewards.at(-1).toFixed(2)
      : (state.player.rewards.at(-1) ? 'paid' : 'nothing')}`
    : 'no pull yet';
  addThinkRow('YOU', PLAYER_COLOR, playerLine);
  for (const key of lineup()) {
    if (!state.visible.has(key)) continue;
    const r = state.racers[key];
    if (!r || r.t === 0) continue;
    addThinkRow(state.labels[key] || key, AGENT_COLORS[key], `pull ${r.t}: ${r.agent.reason}`);
  }
}

function addThinkRow(name, color, text) {
  const box = $('think');
  const row = document.createElement('div');
  row.className = 'bd-row';
  const b = document.createElement('b');
  b.style.color = color;
  b.textContent = `${name}  `;
  row.append(b, document.createTextNode(text));
  box.append(row);
}

function renderChips() {
  const p = state.player;
  $('chip-pulls').textContent = `${p.arms.length}/${state.pulls}`;
  $('chip-regret').textContent = state.casino ? p.regret.toFixed(2) : '—';
  $('chip-best').textContent = p.arms.length ? `${(100 * p.optimal / p.arms.length).toFixed(0)}%` : '—';
  $('chip-leader').textContent = leaderText();
}

// The agent with the lowest regret after the same number of pulls the player has made.
function leaderText() {
  const count = state.player.arms.length;
  if (!count) return '—';
  let best = null;
  for (const key of lineup()) {
    const r = state.racers[key];
    if (!r || r.t < count) continue;
    if (!best || r.regret[count - 1] < best.regret) best = { key, regret: r.regret[count - 1] };
  }
  return best ? `${state.labels[best.key] || best.key} ${best.regret.toFixed(1)}` : '—';
}

// ── Drawing ──────────────────────────────────────────────────────────────

// Match the canvas backing store to its displayed size, so text stays sharp, and return a context
// whose coordinates are CSS pixels. The aspect ratio is the one set in the HTML.
function fit(canvas) {
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  canvas.dataset.ratio ??= String(canvas.height / canvas.width);
  const ratio = Number(canvas.dataset.ratio);
  const w = Math.max(200, canvas.clientWidth || canvas.width);
  const h = Math.round(w * ratio);
  if (canvas.width !== Math.round(w * dpr) || canvas.height !== Math.round(h * dpr)) {
    canvas.width = Math.round(w * dpr);
    canvas.height = Math.round(h * dpr);
  }
  const ctx = canvas.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return { ctx, w, h };
}

function niceMax(v) {
  if (!(v > 0)) return 1;
  const pow = Math.pow(10, Math.floor(Math.log10(v)));
  const f = v / pow;
  const nice = f <= 1 ? 1 : f <= 2 ? 2 : f <= 2.5 ? 2.5 : f <= 5 ? 5 : 10;
  return nice * pow;
}

function draw() {
  renderChips();
  renderThinking();
  drawRace();
  drawThompson();
  drawUCB();
  drawLegend();
}

function drawRace() {
  const canvas = $('race');
  const { ctx, w, h } = fit(canvas);
  ctx.clearRect(0, 0, w, h);
  const casino = state.casino;
  if (!casino) return;
  const N = casino.horizon;
  const x0 = 46, y0 = 12, pw = w - x0 - 14, ph = h - y0 - 30;
  const xs = (t) => x0 + (t / N) * pw;

  // Scale: the largest regret on screen (visible agents and the player), with the floor line if drawn.
  let top = state.player.regret;
  for (const key of state.visible) {
    const r = state.racers[key];
    if (r && r.regret.length) top = Math.max(top, r.regret[r.regret.length - 1]);
  }
  const floorRate = casino.kind === 'drifting' ? null : laiRobbinsRate(casino.schedule[0], casino.kind, casino.sigma);
  if (floorRate !== null) top = Math.max(top, floorRate * Math.log(N));
  const yMax = niceMax(top * 1.05);
  const ys = (v) => y0 + ph - (v / yMax) * ph;

  ctx.strokeStyle = 'rgba(0,245,255,0.12)';
  ctx.fillStyle = '#4a7a9b';
  ctx.font = '12px JetBrains Mono, monospace';
  ctx.lineWidth = 1;
  for (let g = 0; g <= 4; g++) {
    const v = (yMax * g) / 4;
    ctx.beginPath(); ctx.moveTo(x0, ys(v)); ctx.lineTo(x0 + pw, ys(v)); ctx.stroke();
    ctx.fillText(v >= 100 ? v.toFixed(0) : v.toFixed(1), 4, ys(v) + 4);
  }
  ctx.fillText('0', x0 - 12, h - 8);
  ctx.fillText(String(N), x0 + pw - 24, h - 8);
  ctx.fillText('pulls', x0 + pw / 2 - 16, h - 8);

  if (floorRate !== null) {
    ctx.setLineDash([6, 5]);
    ctx.strokeStyle = 'rgba(255,230,0,0.75)';
    ctx.beginPath();
    for (let t = 1; t <= N; t += Math.max(1, Math.floor(N / 200))) {
      const px = xs(t), py = ys(floorRate * Math.log(t));
      if (t === 1) ctx.moveTo(px, py); else ctx.lineTo(px, py);
    }
    ctx.stroke();
    ctx.setLineDash([]);
  }

  const series = (regret, color, width) => {
    ctx.strokeStyle = color;
    ctx.lineWidth = width;
    ctx.beginPath();
    for (let i = 0; i < regret.length; i++) {
      const px = xs(i + 1), py = ys(regret[i]);
      if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
    }
    ctx.stroke();
  };
  for (const key of lineup()) {
    if (!state.visible.has(key)) continue;
    const r = state.racers[key];
    if (r && r.regret.length) series(r.regret, AGENT_COLORS[key], 2);
  }
  if (state.player.arms.length) {
    const mine = scorePulls(casino, state.player.arms).regret;
    series(mine, PLAYER_COLOR, 3);
    const last = mine.length - 1;
    ctx.fillStyle = PLAYER_COLOR;
    ctx.beginPath();
    ctx.arc(xs(last + 1), ys(mine[last]), 4, 0, Math.PI * 2);
    ctx.fill();
  }
  $('race-title').textContent = floorRate !== null
    ? `Regret vs pulls. Dashed: the Lai-Robbins floor for these machines, ${floorRate.toFixed(1)} ln(t).`
    : 'Regret vs pulls. The payouts drift, so there is no fixed floor to compare against.';
}

function drawLegend() {
  const box = $('race-legend');
  const key = [...state.visible].sort().join(',') + '|' + state.player.arms.length + '|' + (state.revealed ? 1 : 0);
  if (key === state.legendKey) return;
  state.legendKey = key;
  box.innerHTML = '';
  const add = (color, text) => {
    const s = document.createElement('span');
    const sw = document.createElement('span');
    sw.className = 'bd-swatch';
    sw.style.background = color;
    s.append(sw, document.createTextNode(text));
    box.append(s);
  };
  add(PLAYER_COLOR, 'you');
  for (const k of lineup()) {
    if (state.visible.has(k)) {
      const r = state.racers[k];
      const count = state.player.arms.length;
      const at = r && r.t >= count && count > 0 ? ` ${r.regret[count - 1].toFixed(1)}` : '';
      add(AGENT_COLORS[k], `${state.labels[k] || k}${at}`);
    }
  }
}

// Thompson posteriors: one density per machine, plus the sample the agent drew last.
function betaLogPdf(x, a, b, lnB) {
  return (a - 1) * Math.log(x) + (b - 1) * Math.log(1 - x) - lnB;
}

function lgamma(x) {
  const g = [0.99999999999980993, 676.5203681218851, -1259.1392167224028, 771.32342877765313,
    -176.61503916999385, 12.507343278686905, -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7];
  const z = x - 1;
  let a = g[0];
  const t = z + 7.5;
  for (let i = 1; i < 9; i++) a += g[i] / (z + i);
  return 0.5 * Math.log(2 * Math.PI) + (z + 0.5) * Math.log(t) - t + Math.log(a);
}

function thompsonCurves(racer) {
  if (state.postCache && state.postCache.t === racer.t && state.postCache.key === state.kind) return state.postCache.curves;
  const gaussian = state.kind === 'gaussian';
  const post = racer.agent.posterior();
  const curves = post.map(([a, b], i) => {
    const pts = [];
    if (gaussian) {
      const mu = a, sd = b;
      for (let j = 0; j <= 160; j++) {
        const x = -0.1 + (1.2 * j) / 160;
        pts.push([x, Math.exp(-0.5 * ((x - mu) / sd) ** 2) / (sd * Math.sqrt(2 * Math.PI))]);
      }
    } else {
      const lnB = lgamma(a) + lgamma(b) - lgamma(a + b);
      for (let j = 1; j < 160; j++) {
        const x = j / 160;
        pts.push([x, Math.exp(betaLogPdf(x, a, b, lnB))]);
      }
    }
    return { i, pts, a, b };
  });
  state.postCache = { t: racer.t, key: state.kind, curves };
  return curves;
}

function drawThompson() {
  const canvas = $('post');
  const { ctx, w, h } = fit(canvas);
  ctx.clearRect(0, 0, w, h);
  const racer = state.racers.thompson;
  const note = $('post-note');
  if (!racer || racer.t === 0) {
    $('post-title').textContent = 'Thompson posteriors';
    note.textContent = state.visible.has('thompson')
      ? 'Waiting for the first pull of Thompson sampling.'
      : 'Turn on Thompson sampling in the agents list to see its posteriors.';
    return;
  }
  const curves = thompsonCurves(racer);
  const gaussian = state.kind === 'gaussian';
  const x0 = 34, y0 = 10, pw = w - x0 - 10, ph = h - y0 - 24;
  let ymax = 0;
  for (const c of curves) for (const [, y] of c.pts) if (y > ymax) ymax = y;
  ymax = Math.min(ymax, 12) || 1; // cap the spike of very confident posteriors so the rest stays readable
  const xs = (x) => x0 + ((x - (gaussian ? -0.1 : 0)) / (gaussian ? 1.2 : 1)) * pw;
  const ys = (y) => y0 + ph - Math.min(y, ymax) / ymax * ph;
  ctx.font = '12px JetBrains Mono, monospace';
  ctx.fillStyle = '#4a7a9b';
  ctx.strokeStyle = 'rgba(0,245,255,0.12)';
  ctx.beginPath(); ctx.moveTo(x0, y0 + ph); ctx.lineTo(x0 + pw, y0 + ph); ctx.stroke();
  ctx.fillText('0', x0 - 4, h - 6);
  ctx.fillText('1', x0 + pw - 4, h - 6);
  curves.forEach((c) => {
    const color = MACHINE_COLORS[c.i % MACHINE_COLORS.length];
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.beginPath();
    c.pts.forEach(([x, y], j) => {
      const px = xs(x), py = ys(y);
      if (j === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
    });
    ctx.stroke();
    // The sample drawn at the last pull, as a dashed vertical line.
    const sample = racer.agent.scores[c.i];
    if (sample >= -0.1 && sample <= 1.1) {
      ctx.setLineDash([4, 4]);
      ctx.beginPath(); ctx.moveTo(xs(sample), y0); ctx.lineTo(xs(sample), y0 + ph); ctx.stroke();
      ctx.setLineDash([]);
    }
  });
  $('post-title').textContent = `Thompson posteriors after ${racer.t} pulls`;
  note.textContent = gaussian
    ? 'Each curve is the agent\'s belief about a machine\'s mean payout. The dashed line is the sample it drew last.'
    : 'Each curve is the agent\'s belief about a machine\'s payout rate: Beta(1 + wins, 1 + losses). The dashed line is the sample it drew last.';
}

// UCB confidence bars: the average so far (solid) and the bonus up to the index (pale).
function drawUCB() {
  const canvas = $('ucb');
  const { ctx, w, h } = fit(canvas);
  ctx.clearRect(0, 0, w, h);
  const key = ['ucb1', 'sw_ucb'].find((k) => state.visible.has(k) && state.racers[k] && state.racers[k].t > 0);
  const note = $('ucb-note');
  if (!key) {
    $('ucb-title').textContent = 'UCB confidence bars';
    note.textContent = 'Turn on UCB1 or sliding-window UCB in the agents list to see its bounds.';
    return;
  }
  const racer = state.racers[key];
  const agent = racer.agent;
  const means = key === 'ucb1'
    ? agent.means()
    : agent.wcounts.map((n, i) => (n ? agent.wsums[i] / n : 0));
  const counts = key === 'ucb1' ? agent.counts : agent.wcounts;
  const idx = agent.scores;
  const finite = idx.filter((v) => Number.isFinite(v));
  const top = niceMax(Math.max(1, ...finite, ...means) * 1.05);
  const x0 = 30, y0 = 10, pw = w - x0 - 10, ph = h - y0 - 34;
  const bw = pw / means.length;
  ctx.font = '12px JetBrains Mono, monospace';
  means.forEach((m, i) => {
    const color = MACHINE_COLORS[i % MACHINE_COLORS.length];
    const bx = x0 + i * bw + bw * 0.18, wd = bw * 0.64;
    const yOf = (v) => y0 + ph - (Math.min(v, top) / top) * ph;
    // The bonus: from the mean up to the index.
    const index = idx[i];
    if (counts[i] > 0 && Number.isFinite(index) && index > m) {
      ctx.fillStyle = color + '44';
      ctx.fillRect(bx, yOf(index), wd, yOf(m) - yOf(index));
    }
    ctx.fillStyle = color;
    ctx.fillRect(bx, yOf(m), wd, y0 + ph - yOf(m));
    ctx.fillStyle = '#c0e8ff';
    ctx.fillText(LETTER(i), bx + wd / 2 - 4, h - 18);
    ctx.fillStyle = '#4a7a9b';
    ctx.fillText(counts[i] ? `n=${counts[i]}` : 'new', bx, h - 4);
  });
  ctx.strokeStyle = 'rgba(0,245,255,0.12)';
  ctx.beginPath(); ctx.moveTo(x0, y0 + ph); ctx.lineTo(x0 + pw, y0 + ph); ctx.stroke();
  ctx.fillStyle = '#4a7a9b';
  ctx.fillText(top.toFixed(1), 2, y0 + 10);
  $('ucb-title').textContent = `${state.labels[key]} after ${racer.t} pulls`;
  note.textContent = 'Solid bar: the average so far. Pale bar: the bonus that lifts an under-tried machine to its bound.';
}

// Benchmark panel: committed results, drawn from results/bandits_benchmark.json.
function renderBenchmark() {
  if (!state.bench) return;
  const setting = state.bench.settings[state.kind];
  const body = $('bench-table').querySelector('tbody');
  body.innerHTML = '';
  if (!setting) {
    $('bench-note').textContent = `No benchmark for ${state.kind} casinos.`;
    return;
  }
  const at = state.bench.table_at;
  const pct = (x) => `${(100 * x.mean).toFixed(1)}% ± ${(50 * (x.hi - x.lo)).toFixed(1)}`;
  const reg = (x) => `${x.mean.toFixed(1)} ± ${((x.hi - x.lo) / 2).toFixed(1)}`;
  for (const key of Object.keys(setting.agents)) {
    const a = setting.agents[key];
    const tr = document.createElement('tr');
    const cells = [
      state.bench.agents[key]?.label || key,
      reg(a.regret_at[String(at[0])]),
      reg(a.regret_at[String(at[1])]),
      pct(a.share_best_last),
    ];
    cells.forEach((text, i) => {
      const td = document.createElement('td');
      if (i > 0) td.className = 'num';
      td.textContent = text;
      tr.append(td);
    });
    body.append(tr);
  }
  const floor = setting.lai_robbins ? `Lai-Robbins floor: ${setting.lai_robbins.rate} ln(t) on average over the machine sets.` : '';
  $('bench-note').textContent = `${state.bench.seeds} machine sets, ${state.bench.machines} machines, horizon ${state.bench.horizon.toLocaleString('en-US')}. `
    + `Best % is over the last ${state.bench.last_pulls} pulls. ${floor}`;
  drawBenchCurves(setting);
}

function drawBenchCurves(setting) {
  const canvas = $('bench-curves');
  const { ctx, w, h } = fit(canvas);
  ctx.clearRect(0, 0, w, h);
  const cps = state.bench.checkpoints;
  const N = cps[cps.length - 1];
  const x0 = 44, y0 = 10, pw = w - x0 - 12, ph = h - y0 - 28;
  let top = 1;
  for (const a of Object.values(setting.agents)) top = Math.max(top, ...a.curve.hi);
  if (setting.lai_robbins) top = Math.max(top, ...setting.lai_robbins.curve);
  const yMax = niceMax(top);
  const xs = (t) => x0 + (t / N) * pw;
  const ys = (v) => y0 + ph - (v / yMax) * ph;
  ctx.font = '12px JetBrains Mono, monospace';
  ctx.fillStyle = '#4a7a9b';
  ctx.strokeStyle = 'rgba(0,245,255,0.12)';
  for (let g = 0; g <= 4; g++) {
    const v = (yMax * g) / 4;
    ctx.beginPath(); ctx.moveTo(x0, ys(v)); ctx.lineTo(x0 + pw, ys(v)); ctx.stroke();
    ctx.fillText(v.toFixed(v >= 100 ? 0 : 1), 4, ys(v) + 4);
  }
  ctx.fillText('pulls', x0 + pw / 2 - 16, h - 6);
  ctx.fillText(String(N), x0 + pw - 40, h - 6);
  for (const [key, a] of Object.entries(setting.agents)) {
    const color = AGENT_COLORS[key] || '#ffffff';
    ctx.fillStyle = color + '22';
    ctx.beginPath();
    cps.forEach((t, j) => { const px = xs(t), py = ys(a.curve.hi[j]); if (j === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py); });
    for (let j = cps.length - 1; j >= 0; j--) ctx.lineTo(xs(cps[j]), ys(a.curve.lo[j]));
    ctx.closePath();
    ctx.fill();
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.beginPath();
    cps.forEach((t, j) => { const px = xs(t), py = ys(a.curve.mean[j]); if (j === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py); });
    ctx.stroke();
  }
  if (setting.lai_robbins) {
    ctx.setLineDash([6, 5]);
    ctx.strokeStyle = 'rgba(255,230,0,0.8)';
    ctx.beginPath();
    cps.forEach((t, j) => { const px = xs(t), py = ys(setting.lai_robbins.curve[j]); if (j === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py); });
    ctx.stroke();
    ctx.setLineDash([]);
  }
}

// ── Boot ─────────────────────────────────────────────────────────────────

async function init() {
  fillMachineOptions();
  try {
    state.meta = await getJSON('/api/bandits/meta');
    state.labels = Object.fromEntries(state.meta.agents.map((a) => [a.key, a.label]));
    state.constants = { ...CONSTANTS, ...state.meta.constants };
  } catch (err) {
    setStatus('ERROR');
    $('floor-note').textContent = `Could not reach the casino API: ${err.message}`;
    return;
  }
  try {
    state.bench = await getJSON('/api/bandits/benchmark');
  } catch (err) {
    state.bench = null;
    if (!String(err.message).startsWith('404')) console.warn('benchmark unavailable', err);
  }
  $('bench').hidden = !state.bench;
  if (state.bench) renderBenchmark();

  $('seed').value = String(randomSeed());
  $('seed').addEventListener('change', newGame);
  $('kind').addEventListener('change', () => {
    state.kind = $('kind').value;
    renderBenchmark();
    newGame();
  });
  $('machines').addEventListener('change', newGame);
  $('pulls').addEventListener('change', newGame);
  $('btn-new').addEventListener('click', newGame);
  $('btn-seed').addEventListener('click', () => { $('seed').value = String(randomSeed()); newGame(); });
  $('btn-reveal').addEventListener('click', reveal);
  $('speed').addEventListener('input', () => {
    state.speed = Number($('speed').value);
    $('speed-val').textContent = String(state.speed);
  });
  $('btn-run').addEventListener('click', () => {
    state.running = !state.running;
    $('btn-run').setAttribute('aria-pressed', String(state.running));
    $('btn-run').querySelector('.btn-txt').textContent = state.running ? '❚❚ PAUSE AGENTS' : '▶ RUN AGENTS';
  });
  document.addEventListener('keydown', (e) => {
    if (e.target.matches('input, select, textarea') || e.metaKey || e.ctrlKey || e.altKey) return;
    if (/^[1-9]$/.test(e.key)) pull(Number(e.key) - 1);
    else if (e.key === '0') pull(9);
  });
  window.addEventListener('resize', () => { state.dirty = true; });

  await newGame();
  requestAnimationFrame(frame);
}

init();
