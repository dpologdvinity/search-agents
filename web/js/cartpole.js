// CartPole page.
//
// The physics, the policies and the rollout loop run here in JavaScript, ported line for line from
// cartpole/env.py, cartpole/nets.py and cartpole/agents.py. The server supplies the trained weights
// (exact float32 values, so both sides make the same decisions), the learning curves and the benchmark.
// VERIFY AGAINST PYTHON asks the server to replay an episode from its own start state and checks every step, which tests the port.
//
// Browser training is REINFORCE with a batch baseline, with the same equations as cartpole/train.py.
// The network is stored as one flat vector, so Adam and the gradient are plain loops over it.

import { getJSON, postJSON } from './api.js';
import { banner, shake } from './fx.js';

const $ = (id) => document.getElementById(id);
const C = {
  cyan: '#00f5ff', pink: '#ff00a0', purple: '#9b00ff', green: '#00ff88', yellow: '#ffe600', dim: '#4a7a9b',
};

// ── Physics: the constants and equations of cartpole/env.py ──────────────
const GRAVITY = 9.8, MASS_CART = 1.0, MASS_POLE = 0.1, TOTAL_MASS = MASS_CART + MASS_POLE;
const HALF_LENGTH = 0.5, POLE_MASS_LENGTH = MASS_POLE * HALF_LENGTH, FORCE = 10.0, TAU = 0.02;
const THETA_LIMIT = 12 * (Math.PI / 180), X_LIMIT = 2.4, MAX_STEPS = 500, INIT_RANGE = 0.05;
const OBS_SCALE = [1 / X_LIMIT, 1 / 2.0, 1 / THETA_LIMIT, 1 / 2.0];
const PD_GAINS = [30.0, 6.0, 0.1, 0.5];
const HIDDEN = 16;

// A small seeded generator (mulberry32). Used for start states, random actions and browser training.
function rng(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// Start state: each variable uniform in [-0.05, 0.05], as in CartPole.reset. The values are not Python's for the same
// seed: this generator (mulberry32) differs from cartpole/env.py's random.Random. Verify therefore starts from the server's states[0].
function newState(seed) {
  const r = rng(seed);
  const u = () => (r() * 2 - 1) * INIT_RANGE;
  return { x: u(), xd: u(), th: u(), thd: u(), steps: 0, term: false, trunc: false };
}

// One Euler step. The comments in cartpole/env.py explain each line; the arithmetic order matches it.
function physStep(s, action) {
  const force = action === 1 ? FORCE : -FORCE;
  const c = Math.cos(s.th), sn = Math.sin(s.th);
  const temp = (force + POLE_MASS_LENGTH * s.thd * s.thd * sn) / TOTAL_MASS;
  const thacc = (GRAVITY * sn - c * temp) / (HALF_LENGTH * (4 / 3 - MASS_POLE * c * c / TOTAL_MASS));
  const xacc = temp - POLE_MASS_LENGTH * thacc * c / TOTAL_MASS;
  s.x += TAU * s.xd; s.xd += TAU * xacc; s.th += TAU * s.thd; s.thd += TAU * thacc;
  s.steps += 1;
  s.term = Math.abs(s.x) > X_LIMIT || Math.abs(s.th) > THETA_LIMIT;
  s.trunc = !s.term && s.steps >= MAX_STEPS;
}

const scaled = (s) => [s.x * OBS_SCALE[0], s.xd * OBS_SCALE[1], s.th * OBS_SCALE[2], s.thd * OBS_SCALE[3]];

// ── Networks: the forward pass of cartpole/nets.py ───────────────────────
// W = {W1 (H x 4), b1, W2 (K x H), b2}. tanh hidden layer, linear output.
function mlp(W, x) {
  const h = new Float64Array(W.b1.length);
  for (let i = 0; i < h.length; i++) {
    let a = 0;
    const row = W.W1[i];
    for (let j = 0; j < x.length; j++) a += row[j] * x[j];
    h[i] = Math.tanh(a + W.b1[i]);
  }
  const z = W.b2.map((b, k) => {
    let a = 0;
    for (let i = 0; i < h.length; i++) a += W.W2[k][i] * h[i];
    return a + b;
  });
  return { h, z };
}

// What an agent does in state s: an action, and whatever the page shows about its reasoning.
function decide(agent, s) {
  if (agent.name === 'random') return { action: agent.rng() < 0.5 ? 0 : 1 };
  if (agent.name === 'pd') {
    const u = PD_GAINS[0] * s.th + PD_GAINS[1] * s.thd + PD_GAINS[2] * s.x + PD_GAINS[3] * s.xd;
    return { action: u > 0 ? 1 : 0, u };
  }
  const x = scaled(s);
  if (agent.name === 'cem') {
    const w = agent.theta;
    let sc = 0;
    for (let j = 0; j < 4; j++) sc += w[j] * x[j];
    sc += w[4];
    return { action: sc > 0 ? 1 : 0, u: sc };
  }
  const { z } = mlp(agent.net, x);
  const pl = 1 / (1 + Math.exp(z[1] - z[0]));   // softmax over (left, right), written as a logistic
  const out = { action: z[1] > z[0] ? 1 : 0, pl, pr: 1 - pl, value: null };
  if (agent.vnet) out.value = agent.vscale * mlp(agent.vnet, x).z[0];
  return out;
}

// ── Page state ──────────────────────────────────────────────────────────
const state = {
  meta: null,
  agents: {},          // name -> loaded agent
  agentName: 'reinforce',
  mode: 'watch',       // 'watch' (the agent pushes) or 'play' (you push)
  sim: null,           // {s, seed, episode, info, action, waiting, resumeAt, fallAt, pushDir, hist}
  running: true,
  speed: 1,
  best: 0,
  episodes: 0,
  stepAcc: 0,
  lastTs: 0,
  frame: 0,
  curves: null,
  visible: { reinforce: true, actor_critic: true, cem: true, browser: true },
  browser: null,       // in-browser training run: {returns, ...}
  pushDir: 1,          // play mode: the latest key sets the push, and it repeats until the other key
};

const LABEL = {
  random: 'RANDOM', pd: 'PD CONTROLLER', reinforce: 'REINFORCE + BASELINE',
  actor_critic: 'ACTOR-CRITIC', cem: 'CROSS-ENTROPY SEARCH',
};
const CURVE_COLOR = { reinforce: C.cyan, actor_critic: C.pink, cem: C.yellow, browser: C.green };
const CURVE_LABEL = { reinforce: 'REINFORCE', actor_critic: 'ACTOR-CRITIC', cem: 'CEM', browser: 'YOUR BROWSER RUN' };

function logEvent(msg, color = C.cyan) {
  const log = $('log');
  const row = document.createElement('div');
  row.style.color = color;
  row.textContent = `> ${msg}`;
  log.prepend(row);
  while (log.children.length > 60) log.lastChild.remove();
}

// ── Loading the trained agents ──────────────────────────────────────────
async function loadAgent(name) {
  if (state.agents[name]) return state.agents[name];
  let agent;
  if (name === 'random') agent = { name, rng: rng(7) };
  else if (name === 'pd') agent = { name };
  else {
    const { weights } = await getJSON(`/api/cartpole/policy/${name}`);
    if (name === 'cem') agent = { name, theta: weights.theta };
    else {
      agent = { name, net: { W1: weights.W1, b1: weights.b1, W2: weights.W2, b2: weights.b2 } };
      if (weights.v_W1) {
        agent.vnet = { W1: weights.v_W1, b1: weights.v_b1, W2: weights.v_W2, b2: weights.v_b2 };
        agent.vscale = weights.value_scale[0];
      }
    }
  }
  state.agents[name] = agent;
  return agent;
}

// ── Episodes ────────────────────────────────────────────────────────────
function startEpisode(seed) {
  state.sim = {
    s: newState(seed), seed, agent: state.agentName, info: null, action: null,
    waiting: false, resumeAt: 0, hist: [],
  };
  state.episodes += 1;
  $('chip-episode').textContent = String(state.episodes);
  $('chip-steps').textContent = '0';
  $('seed').value = String(seed);
}

function finishEpisode(now) {
  const sim = state.sim;
  const steps = sim.s.steps;
  state.best = Math.max(state.best, steps);
  $('chip-best').textContent = String(state.best);
  const cap = sim.s.trunc;
  sim.waiting = true;
  sim.resumeAt = now + (state.mode === 'play' ? 1500 : 1200);
  if (cap) {
    logEvent(`${LABEL[sim.agent]} balanced the full ${MAX_STEPS} steps (seed ${sim.seed})`, C.green);
    banner('BALANCED', `${MAX_STEPS} steps`, C.green);
  } else {
    logEvent(`${LABEL[sim.agent]} fell after ${steps} steps (seed ${sim.seed})`, C.pink);
    banner('FALLEN', `after ${steps} steps`, C.pink);
    shake($('stage'), 'big');
  }
  $('status-line').textContent = cap
    ? `balanced ${MAX_STEPS} steps · next episode in a moment`
    : `fell after ${steps} steps · next episode in a moment`;
}

// Advance the simulation by one control step with the current agent (watch) or the player's push (play).
function advance(now) {
  const sim = state.sim;
  if (sim.s.term || sim.s.trunc) return;
  let action;
  if (state.mode === 'play') {
    action = state.pushDir > 0 ? 1 : 0;
    sim.info = null;
  } else {
    const agent = state.agents[state.agentName];
    const d = decide(agent, sim.s);
    action = d.action;
    sim.info = d;
  }
  sim.action = action;
  physStep(sim.s, action);
  // The sparkline shows the critic's value when there is one, otherwise P(right).
  if (sim.info && (sim.info.value !== null && sim.info.value !== undefined || sim.info.pr !== undefined)) {
    sim.hist.push(sim.info.value ?? sim.info.pr);
    if (sim.hist.length > 240) sim.hist.shift();
  }
  if (sim.s.term || sim.s.trunc) finishEpisode(now);
}

function nudge() {
  const sim = state.sim;
  if (!sim || sim.waiting) return;
  const dir = Math.random() < 0.5 ? -1 : 1;
  sim.s.thd += dir * 0.9; // a sharp kick to the pole's angular velocity
  logEvent(`nudge: the pole was knocked ${dir > 0 ? 'right' : 'left'}. Watch the policy recover.`, C.yellow);
  shake($('stage'));
}

// ── Stage: the neon cart and pole ─────────────────────────────────────────
const stage = $('stage');
const sctx = stage.getContext('2d');
let sizeW = 0, sizeH = 0, dpr = 1;

function sizeStage() {
  dpr = Math.min(window.devicePixelRatio || 1, 2);
  const w = stage.clientWidth, h = stage.clientHeight;
  if (w === sizeW && h === sizeH) return;
  sizeW = w; sizeH = h;
  stage.width = Math.round(w * dpr);
  stage.height = Math.round(h * dpr);
}

function drawStage(now) {
  sizeStage();
  const ctx = sctx;
  const W = sizeW, H = sizeH;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, W, H);
  const bg = ctx.createRadialGradient(W / 2, H * 1.1, 10, W / 2, H * 1.1, W * 0.8);
  bg.addColorStop(0, 'rgba(155,0,255,0.28)');
  bg.addColorStop(1, 'rgba(2,6,16,1)');
  ctx.fillStyle = bg;
  ctx.fillRect(0, 0, W, H);

  const pxPerM = W / 6.6;
  const ox = W / 2;
  const trackY = H * 0.74;
  const toX = (m) => ox + m * pxPerM;

  // Floor grid: vertical lines every 0.5 m and a horizon of horizontal lines below the track.
  ctx.lineWidth = 1;
  ctx.strokeStyle = 'rgba(0,245,255,0.08)';
  for (let m = -3; m <= 3; m += 0.5) {
    ctx.beginPath(); ctx.moveTo(toX(m), trackY); ctx.lineTo(toX(m) + (m) * pxPerM * 0.6, H); ctx.stroke();
  }
  for (let k = 1; k <= 5; k++) {
    const y = trackY + ((H - trackY) * k) / 5;
    ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(W, y); ctx.stroke();
  }

  // Track and the limits at |x| = 2.4 m, shown as pink posts; past them the episode is over.
  ctx.save();
  ctx.shadowColor = C.cyan; ctx.shadowBlur = 14;
  ctx.strokeStyle = C.cyan; ctx.lineWidth = 3;
  ctx.beginPath(); ctx.moveTo(0, trackY); ctx.lineTo(W, trackY); ctx.stroke();
  ctx.restore();
  for (const m of [-X_LIMIT, X_LIMIT]) {
    ctx.save();
    ctx.shadowColor = C.pink; ctx.shadowBlur = 16;
    ctx.strokeStyle = C.pink; ctx.lineWidth = 4;
    ctx.beginPath(); ctx.moveTo(toX(m), trackY - H * 0.36); ctx.lineTo(toX(m), trackY + 6); ctx.stroke();
    ctx.restore();
  }
  ctx.fillStyle = C.dim;
  ctx.font = `${Math.max(10, Math.round(H * 0.045))}px JetBrains Mono, monospace`;
  ctx.textAlign = 'center';
  ctx.fillText('-2.4 m', toX(-X_LIMIT), trackY + H * 0.085);
  ctx.fillText('+2.4 m', toX(X_LIMIT), trackY + H * 0.085);
  ctx.fillText('0', ox, trackY + H * 0.085);

  const sim = state.sim;
  if (!sim) return;
  const s = sim.s;
  const cartW = 0.5 * pxPerM, cartH = 0.2 * pxPerM, wheel = 0.06 * pxPerM;
  const cx = toX(s.x);
  const cartTop = trackY - wheel * 2 - cartH;
  const fallen = s.term;

  // Guide line: where the pole would be if it were perfectly upright.
  ctx.setLineDash([4, 6]);
  ctx.strokeStyle = 'rgba(255,230,0,0.25)';
  ctx.beginPath(); ctx.moveTo(cx, cartTop); ctx.lineTo(cx, cartTop - pxPerM); ctx.stroke();
  ctx.setLineDash([]);

  // Pole: from the hinge at the top centre of the cart, 1 m long.
  const tipX = cx + pxPerM * 1.0 * Math.sin(s.th) * 2 * HALF_LENGTH;
  const tipY = cartTop - pxPerM * 1.0 * Math.cos(s.th);
  const poleCol = fallen ? '#ff3b3b' : C.pink;
  ctx.save();
  ctx.lineCap = 'round';
  ctx.shadowColor = poleCol; ctx.shadowBlur = 18;
  ctx.strokeStyle = poleCol; ctx.lineWidth = Math.max(4, pxPerM * 0.05);
  ctx.beginPath(); ctx.moveTo(cx, cartTop); ctx.lineTo(tipX, tipY); ctx.stroke();
  ctx.shadowBlur = 0;
  ctx.strokeStyle = 'rgba(255,255,255,0.7)'; ctx.lineWidth = Math.max(1.5, pxPerM * 0.012);
  ctx.beginPath(); ctx.moveTo(cx, cartTop); ctx.lineTo(tipX, tipY); ctx.stroke();
  ctx.restore();
  // Bob at the tip.
  ctx.save();
  ctx.shadowColor = C.yellow; ctx.shadowBlur = 20;
  ctx.fillStyle = C.yellow;
  ctx.beginPath(); ctx.arc(tipX, tipY, Math.max(6, pxPerM * 0.07), 0, Math.PI * 2); ctx.fill();
  ctx.restore();

  // Cart: a rounded body with glowing edges and two wheels.
  ctx.save();
  ctx.shadowColor = C.cyan; ctx.shadowBlur = 14;
  const g = ctx.createLinearGradient(0, cartTop, 0, cartTop + cartH);
  g.addColorStop(0, '#9feeff'); g.addColorStop(0.5, C.cyan); g.addColorStop(1, '#0a5d68');
  ctx.fillStyle = g;
  roundRect(ctx, cx - cartW / 2, cartTop, cartW, cartH, cartH * 0.25);
  ctx.fill();
  ctx.restore();
  for (const dx of [-0.32, 0.32]) {
    ctx.beginPath();
    ctx.fillStyle = '#050d1a';
    ctx.arc(cx + dx * cartW, trackY - wheel, wheel, 0, Math.PI * 2); ctx.fill();
    ctx.strokeStyle = C.cyan; ctx.lineWidth = 2; ctx.stroke();
  }

  // Push arrow from the cart's side while the agent (or you) is pushing.
  if (sim.action !== null && !sim.waiting) {
    const dir = sim.action === 1 ? 1 : -1;
    const y = cartTop + cartH * 0.5;
    const x0 = cx + dir * (cartW / 2 + 8);
    const x1 = x0 + dir * pxPerM * 0.35;
    ctx.strokeStyle = dir > 0 ? C.green : C.yellow;
    ctx.shadowColor = ctx.strokeStyle; ctx.shadowBlur = 12; ctx.lineWidth = 4;
    ctx.beginPath(); ctx.moveTo(x0, y); ctx.lineTo(x1, y); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(x1, y); ctx.lineTo(x1 - dir * 10, y - 7); ctx.lineTo(x1 - dir * 10, y + 7); ctx.closePath();
    ctx.fillStyle = ctx.strokeStyle; ctx.fill();
    ctx.shadowBlur = 0;
  }

  // Heads-up text in the top corners.
  ctx.textAlign = 'left';
  ctx.fillStyle = C.cyan;
  const fs = Math.max(11, Math.round(H * 0.058));
  ctx.font = `${fs}px JetBrains Mono, monospace`;
  ctx.fillText(`STEP ${s.steps}`, 12, fs + 8);
  ctx.fillText(`x ${s.x >= 0 ? '+' : ''}${s.x.toFixed(2)} m`, 12, fs * 2 + 12);
  ctx.fillText(`θ ${(s.th * 180 / Math.PI >= 0 ? '+' : '')}${(s.th * 180 / Math.PI).toFixed(1)}°`, 12, fs * 3 + 16);
  ctx.textAlign = 'right';
  ctx.fillStyle = C.pink;
  const who = state.mode === 'play' ? 'YOU' : LABEL[sim.agent];
  ctx.fillText(who, W - 12, fs + 8);
  ctx.fillStyle = C.dim;
  ctx.fillText(`seed ${sim.seed}`, W - 12, fs * 2 + 12);
  if (fallen) {
    ctx.fillStyle = 'rgba(255,59,59,0.18)';
    ctx.fillRect(0, 0, W, H);
  }
}

function roundRect(ctx, x, y, w, h, r) {
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.arcTo(x + w, y, x + w, y + h, r);
  ctx.arcTo(x + w, y + h, x, y + h, r);
  ctx.arcTo(x, y + h, x, y, r);
  ctx.arcTo(x, y, x + w, y, r);
  ctx.closePath();
}

// ── Mind panel: the policy's probabilities, the critic's value, and a plain-language reading ────────────
const spark = $('vspark');
const sparkCtx = spark.getContext('2d');

function renderMind() {
  const sim = state.sim;
  const bars = $('mind-bars');
  const title = $('mind-title');
  const narr = $('narration');
  if (!sim) return;
  const agent = state.agents[sim.agent];
  const info = sim.info;
  if (state.mode === 'play' || !info) {
    bars.innerHTML = '';
    title.textContent = state.mode === 'play'
      ? 'You are pushing. Switch to WATCH to see an agent reason about the same state.'
      : 'Waiting for the first step.';
    narr.textContent = state.mode === 'play' ? playNarration(sim.s) : '';
    return;
  }
  if (info.pl !== undefined) {
    title.textContent = `P(right) ${info.pr.toFixed(2)}`
      + (info.value !== null && info.value !== undefined ? `   ·   critic V(s) ${info.value.toFixed(1)}` : '   ·   no critic (REINFORCE)');
    bars.innerHTML = [
      bar('LEFT', info.pl, C.yellow, 'pl'),
      bar('RIGHT', info.pr, C.cyan, 'pr'),
    ].join('');
  } else if (info.u !== undefined) {
    title.textContent = agent.name === 'pd'
      ? `PD control u = ${info.u.toFixed(2)}: ${info.u > 0 ? 'push right' : 'push left'} when u > 0`
      : `Linear score ${info.u.toFixed(2)}: ${info.u > 0 ? 'push right' : 'push left'} when the score is positive`;
    bars.innerHTML = signedBar(info.u, agent.name === 'pd' ? 12 : 1);
  } else {
    title.textContent = 'A coin flip. There is no model and no memory.';
    bars.innerHTML = bar('LEFT', 0.5, C.yellow, 'pl') + bar('RIGHT', 0.5, C.cyan, 'pr');
  }
  narr.textContent = narrate(sim.s, sim.action, info, agent);
  drawSpark(info);
}

function bar(label, p, color) {
  return `<div class="cp-bar"><span>${label}</span><div class="cp-track"><i style="width:${(p * 100).toFixed(1)}%;background:${color};box-shadow:0 0 12px ${color}"></i></div><b>${p.toFixed(2)}</b></div>`;
}

// A bar that grows left or right from the centre: positive scores push right.
function signedBar(u, scale) {
  const t = Math.tanh(u / scale);
  const w = Math.abs(t) * 50;
  const left = t >= 0 ? 50 : 50 - w;
  const col = t >= 0 ? C.cyan : C.yellow;
  return `<div class="cp-bar"><span>LEFT</span><div class="cp-track cp-centre"><i style="left:${left.toFixed(1)}%;width:${w.toFixed(1)}%;background:${col};box-shadow:0 0 12px ${col}"></i></div><span>RIGHT</span></div>`;
}

function narrate(s, action, info, agent) {
  const deg = s.th * 180 / Math.PI;
  const side = s.th >= 0 ? 'right' : 'left';
  const falling = s.thd * (s.th >= 0 ? 1 : -1) > 0;
  let t = `The pole leans ${Math.abs(deg).toFixed(1)}° ${side}, ${falling ? 'and is still falling that way' : 'and is swinging back'}. `;
  if (info.pl !== undefined) {
    const p = action ? info.pr : info.pl;
    t += `The policy puts ${(p * 100).toFixed(0)}% on ${action ? 'right' : 'left'}, so it pushes ${action ? 'right' : 'left'}. `;
  } else {
    const who = agent.name === 'pd' ? 'The PD rule' : agent.name === 'random' ? 'The coin flip' : 'The search';
    t += `${who} says push ${action ? 'right' : 'left'}. `;
  }
  if (Math.abs(s.x) > 1.8) t += 'The cart is near the edge of the track, so the pole has to lean back toward the middle.';
  return t;
}

function playNarration(s) {
  const deg = s.th * 180 / Math.PI;
  return `The pole leans ${Math.abs(deg).toFixed(1)}° ${s.th >= 0 ? 'right' : 'left'}. Push it back under the cart: `
    + 'a line of pushes that keeps it upright is the same thing the trained policies have learned.';
}

function drawSpark(info) {
  const ctx = sparkCtx;
  const w = spark.clientWidth || 300, h = spark.clientHeight || 60;
  if (spark.width !== Math.round(w * dpr)) { spark.width = Math.round(w * dpr); spark.height = Math.round(h * dpr); }
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);
  const hist = state.sim.hist;
  if (hist.length < 2) return;
  const isValue = info.value !== null && info.value !== undefined;
  const values = hist;
  const lo = isValue ? Math.min(...values) : 0, hi = isValue ? Math.max(...values, lo + 1) : 1;
  ctx.strokeStyle = isValue ? C.pink : C.cyan;
  ctx.shadowColor = ctx.strokeStyle; ctx.shadowBlur = 8; ctx.lineWidth = 2;
  ctx.beginPath();
  values.forEach((v, i) => {
    const x = (i / (values.length - 1)) * w;
    const y = h - 4 - ((v - lo) / (hi - lo)) * (h - 8);
    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  });
  ctx.stroke();
  ctx.shadowBlur = 0;
  ctx.fillStyle = C.dim;
  ctx.font = '10px JetBrains Mono, monospace';
  ctx.textAlign = 'left';
  ctx.fillText(isValue ? `critic V(s), last ${values.length} steps` : `P(right), last ${values.length} steps`, 6, 12);
}

// ── Learning curves ─────────────────────────────────────────────────────
function movingAverage(a, k) {
  const out = new Array(a.length);
  let sum = 0;
  for (let i = 0; i < a.length; i++) {
    sum += a[i];
    if (i >= k) sum -= a[i - k];
    out[i] = sum / Math.min(i + 1, k);
  }
  return out;
}

// For each algorithm: the mean across seeds, and the min and max band, all smoothed over 25 episodes.
function curveSeries(name) {
  const entry = state.curves && state.curves.algorithms[name];
  if (!entry) return null;
  const runs = Object.values(entry.seeds).map((r) => movingAverage(r, 25));
  const n = Math.min(...runs.map((r) => r.length));
  const mean = [], lo = [], hi = [];
  for (let i = 0; i < n; i++) {
    const v = runs.map((r) => r[i]);
    mean.push(v.reduce((a, b) => a + b, 0) / v.length);
    lo.push(Math.min(...v));
    hi.push(Math.max(...v));
  }
  return { mean, lo, hi };
}

function drawCurves() {
  const cv = $('curves');
  const dprCv = dpr;
  const w = cv.clientWidth || 720, h = cv.clientHeight || 300;
  if (cv.width !== Math.round(w * dprCv)) { cv.width = Math.round(w * dprCv); cv.height = Math.round(h * dprCv); }
  const ctx = cv.getContext('2d');
  ctx.setTransform(dprCv, 0, 0, dprCv, 0, 0);
  ctx.clearRect(0, 0, w, h);
  const pad = { l: 44, r: 12, t: 12, b: 28 };
  const pw = w - pad.l - pad.r, ph = h - pad.t - pad.b;
  const maxEp = 1500;
  const X = (e) => pad.l + (e / maxEp) * pw;
  const Y = (v) => pad.t + ph - (v / MAX_STEPS) * ph;
  ctx.font = '11px JetBrains Mono, monospace';
  ctx.strokeStyle = 'rgba(0,245,255,0.12)';
  ctx.fillStyle = C.dim;
  for (const v of [0, 100, 200, 300, 400, 500]) {
    ctx.beginPath(); ctx.moveTo(pad.l, Y(v)); ctx.lineTo(w - pad.r, Y(v)); ctx.stroke();
    ctx.textAlign = 'right'; ctx.fillText(String(v), pad.l - 6, Y(v) + 4);
  }
  for (const e of [0, 500, 1000, 1500]) {
    ctx.textAlign = 'center'; ctx.fillText(String(e), X(e), h - 8);
  }
  ctx.textAlign = 'right';
  ctx.fillText('training episode', w - pad.r - 4, pad.t + 14);

  const series = [];
  for (const name of ['reinforce', 'actor_critic', 'cem']) {
    if (!state.visible[name]) continue;
    const s = curveSeries(name);
    if (s) series.push([name, s]);
  }
  if (state.visible.browser && state.browser && state.browser.returns.length > 25) {
    series.push(['browser', { mean: movingAverage(state.browser.returns, 25), lo: null, hi: null }]);
  }
  for (const [name, s] of series) {
    const col = CURVE_COLOR[name];
    if (s.lo) {
      ctx.fillStyle = hexA(col, 0.14);
      ctx.beginPath();
      s.hi.forEach((v, i) => (i ? ctx.lineTo(X(i), Y(v)) : ctx.moveTo(X(i), Y(v))));
      for (let i = s.lo.length - 1; i >= 0; i--) ctx.lineTo(X(i), Y(s.lo[i]));
      ctx.closePath(); ctx.fill();
    }
    ctx.save();
    ctx.shadowColor = col; ctx.shadowBlur = 8; ctx.strokeStyle = col; ctx.lineWidth = 2.2;
    ctx.beginPath();
    s.mean.forEach((v, i) => (i ? ctx.lineTo(X(i), Y(v)) : ctx.moveTo(X(i), Y(v))));
    ctx.stroke();
    ctx.restore();
  }
}

function hexA(hex, a) {
  const n = parseInt(hex.slice(1), 16);
  return `rgba(${(n >> 16) & 255},${(n >> 8) & 255},${n & 255},${a})`;
}

function renderLegend() {
  const names = ['reinforce', 'actor_critic', 'cem', 'browser'];
  $('curve-legend').innerHTML = names.map((n) => {
    const on = state.visible[n];
    return `<button type="button" class="cp-chip" data-curve="${n}" aria-pressed="${on}" style="--c:${CURVE_COLOR[n]}">${CURVE_LABEL[n]}</button>`;
  }).join('');
}

// ── Browser training: REINFORCE from random weights, as in cartpole/train.py ───────────────
// Flat layout: W1 (H*4), b1 (H), W2 (2*H), b2 (2).
const OFF = { W1: 0, b1: HIDDEN * 4, W2: HIDDEN * 4 + HIDDEN, b2: HIDDEN * 4 + HIDDEN + 2 * HIDDEN };
const NPARAM = OFF.b2 + 2;

function trainRun(seed, episodes) {
  const r = rng(seed);
  const gauss = () => Math.sqrt(-2 * Math.log(r() + 1e-12)) * Math.cos(2 * Math.PI * r());
  const p = new Float64Array(NPARAM);
  for (let i = 0; i < HIDDEN * 4; i++) p[OFF.W1 + i] = gauss() / Math.sqrt(4);
  for (let i = 0; i < 2 * HIDDEN; i++) p[OFF.W2 + i] = 0.1 * gauss() / Math.sqrt(HIDDEN);
  return { r, p, m: new Float64Array(NPARAM), v: new Float64Array(NPARAM), t: 0, returns: [], episodes, seed };
}

// Forward pass on the flat vector, returning the activations that backprop needs.
function flatForward(p, x) {
  const h = new Float64Array(HIDDEN);
  for (let i = 0; i < HIDDEN; i++) {
    let a = 0;
    for (let j = 0; j < 4; j++) a += p[OFF.W1 + i * 4 + j] * x[j];
    h[i] = Math.tanh(a + p[OFF.b1 + i]);
  }
  const z0 = dot2(p, 0, h) + p[OFF.b2];
  const z1 = dot2(p, 1, h) + p[OFF.b2 + 1];
  return { h, z0, z1 };
}

function dot2(p, k, h) {
  let a = 0;
  for (let i = 0; i < HIDDEN; i++) a += p[OFF.W2 + k * HIDDEN + i] * h[i];
  return a;
}

// Run one batch of episodes and take one Adam step on the REINFORCE loss, as in train.py.
function trainBatch(run) {
  const BATCH = 8, GAMMA = 0.99, LR = 1e-2;
  const recs = []; // per step: x, h, action, return
  for (let b = 0; b < BATCH; b++) {
    const s = newState(Math.floor(run.r() * 2 ** 31));
    const steps = [];
    while (!(s.term || s.trunc)) {
      const x = scaled(s);
      const f = flatForward(run.p, x);
      const pl = 1 / (1 + Math.exp(f.z1 - f.z0));
      const a = run.r() < pl ? 0 : 1;
      physStep(s, a);
      steps.push({ x, h: f.h, a });
    }
    // discounted returns, backwards
    let g = 0;
    for (let t = steps.length - 1; t >= 0; t--) { g = 1 + GAMMA * g; steps[t].G = g; }
    recs.push(...steps);
    run.returns.push(s.steps);
  }
  // Batch-normalised advantages, then the gradient of L = -(1/T) sum adv log pi(a|x).
  const T = recs.length;
  const mean = recs.reduce((acc, q) => acc + q.G, 0) / T;
  const sd = Math.sqrt(recs.reduce((acc, q) => acc + (q.G - mean) ** 2, 0) / T);
  const grad = new Float64Array(NPARAM);
  for (const q of recs) {
    const adv = (q.G - mean) / (sd + 1e-8);
    const f = flatForward(run.p, q.x);
    const pl = 1 / (1 + Math.exp(f.z1 - f.z0));
    const onehot0 = q.a === 0 ? 1 : 0;
    // dL/dz_k = -(adv / T) * (onehot_k - p_k)
    const dz = [-(adv / T) * (onehot0 - pl), -(adv / T) * ((1 - onehot0) - (1 - pl))];
    for (let k = 0; k < 2; k++) {
      grad[OFF.b2 + k] += dz[k];
      for (let i = 0; i < HIDDEN; i++) grad[OFF.W2 + k * HIDDEN + i] += dz[k] * q.h[i];
    }
    for (let i = 0; i < HIDDEN; i++) {
      const dh = dz[0] * run.p[OFF.W2 + i] + dz[1] * run.p[OFF.W2 + HIDDEN + i];
      const da = dh * (1 - q.h[i] * q.h[i]); // through the tanh
      grad[OFF.b1 + i] += da;
      for (let j = 0; j < 4; j++) grad[OFF.W1 + i * 4 + j] += da * q.x[j];
    }
  }
  // Adam
  run.t += 1;
  const b1 = 0.9, b2 = 0.999, eps = 1e-8;
  const c1 = 1 - b1 ** run.t, c2 = 1 - b2 ** run.t;
  for (let i = 0; i < NPARAM; i++) {
    run.m[i] = b1 * run.m[i] + (1 - b1) * grad[i];
    run.v[i] = b2 * run.v[i] + (1 - b2) * grad[i] * grad[i];
    run.p[i] -= LR * (run.m[i] / c1) / (Math.sqrt(run.v[i] / c2) + eps);
  }
}

function startBrowserTraining() {
  if (state.browser && state.browser.busy) return;
  const seed = 1 + Math.floor(Math.random() * 999);
  state.browser = trainRun(seed, 600);
  state.browser.busy = true;
  state.browser.t0 = performance.now();
  $('btn-train').disabled = true;
  logEvent(`training REINFORCE in your browser from random weights (seed ${seed})`, C.green);
  const step = () => {
    const run = state.browser;
    for (let k = 0; k < 5 && run.returns.length < run.episodes; k++) trainBatch(run);
    $('train-note').textContent = `episode ${Math.min(run.returns.length, run.episodes)} of ${run.episodes}, last 25 mean ${
      (run.returns.slice(-25).reduce((a, b) => a + b, 0) / Math.max(1, Math.min(25, run.returns.length))).toFixed(0)} steps`;
    drawCurves();
    if (run.returns.length < run.episodes) setTimeout(step, 0);
    else {
      run.busy = false;
      $('btn-train').disabled = false;
      $('train-note').textContent = `done in ${((performance.now() - run.t0) / 1000).toFixed(1)} s: the last 50 mean is ${
        (run.returns.slice(-50).reduce((a, b) => a + b, 0) / 50).toFixed(0)} steps. Pick a seed and watch it fly.`;
      logEvent('browser REINFORCE finished. Its weights live only in this tab.', C.green);
    }
  };
  setTimeout(step, 0);
}

// ── Benchmark table ─────────────────────────────────────────────────────
async function loadBenchmark() {
  try {
    const doc = await getJSON('/api/cartpole/benchmark');
    const tbody = document.querySelector('#bench-table tbody');
    tbody.innerHTML = Object.values(doc.agents).map((r) =>
      `<tr><td>${r.label}</td><td>${r.mean.toFixed(1)}</td><td>±${r.ci95.toFixed(1)}</td><td>${r.pct_reached_max.toFixed(1)}%</td></tr>`
    ).join('');
  } catch (e) {
    document.querySelector('#bench-table tbody').innerHTML = `<tr><td colspan="4">${e.message}</td></tr>`;
  }
}

// ── Verify the port: a Python episode replayed step by step in JavaScript ───────────────
async function verifyPort() {
  const out = $('verify-out');
  const name = state.agentName;
  if (name === 'random') { out.textContent = 'Pick a learned agent or PD: a random agent has no shared generator.'; return; }
  const seed = Math.max(0, Math.floor(Number($('seed').value) || 0));
  out.textContent = 'asking the server to replay the episode from its own start state...';
  try {
    const py = await postJSON('/api/cartpole/rollout', { agent: name, seed, steps: MAX_STEPS });
    const agent = await loadAgent(name);
    // Each step is checked from the Python state, not from the JavaScript run's own state. The controllers
    // chatter around the switching surface, so one flipped decision would make a whole-episode run diverge.
    // Checking step by step isolates each step's arithmetic and decision. The first check starts from the server's
    // states[0] (the Python start), not from newState(seed), which would give different values for the same seed.
    let worst = 0, mismatches = 0;
    for (let t = 0; t < py.actions.length; t++) {
      const [x, xd, th, thd] = py.states[t];
      const s = { x, xd, th, thd, steps: 0, term: false, trunc: false };
      if (decide(agent, s).action !== py.actions[t]) mismatches++;
      physStep(s, py.actions[t]);
      const ref = py.states[t + 1];
      worst = Math.max(worst, Math.abs(s.x - ref[0]), Math.abs(s.xd - ref[1]), Math.abs(s.th - ref[2]), Math.abs(s.thd - ref[3]));
    }
    const ok = mismatches === 0 && worst < 1e-9;
    out.style.color = ok ? C.green : C.pink;
    out.textContent = `${ok ? 'MATCH' : 'DIFFERS'}: ${py.steps} steps checked one by one, ${mismatches} different actions, largest one-step state error ${worst.toExponential(1)}`;
    logEvent(`port check ${ok ? 'matched' : 'differs'}: ${py.steps} steps, max one-step error ${worst.toExponential(1)}`, ok ? C.green : C.pink);
  } catch (e) {
    out.style.color = C.pink;
    out.textContent = `could not verify: ${e.message}`;
  }
}

// ── Controls ───────────────────────────────────────────────────────────
function setMode(mode) {
  state.mode = mode;
  $('mode-watch').classList.toggle('act', mode === 'watch');
  $('mode-play').classList.toggle('act', mode === 'play');
  $('mode-watch').setAttribute('aria-selected', String(mode === 'watch'));
  $('mode-play').setAttribute('aria-selected', String(mode === 'play'));
  if (mode === 'play' && state.speed > 0.5) setSpeed(0.5);
  if (state.sim) startEpisode(state.sim.seed);
  $('keys-hint').textContent = mode === 'play'
    ? 'Keys: ← / A pushes left, → / D pushes right; the last key keeps pushing. Space pauses, N nudges.'
    : 'Keys: space pauses, N nudges the pole. Pick PLAY to push it yourself.';
  logEvent(mode === 'play' ? 'you have the cart: push it with the arrow keys or the buttons' : 'the agent has the cart');
}

function setSpeed(v) {
  state.speed = v;
  $('speed').value = String(v);
}

function pressPush(dir) {
  if (state.mode !== 'play') setMode('play');
  state.pushDir = dir;
}

function init() {
  $('mode-watch').onclick = () => setMode('watch');
  $('mode-play').onclick = () => setMode('play');
  $('btn-run').onclick = () => {
    state.running = !state.running;
    $('btn-run').querySelector('.btn-txt').textContent = state.running ? '❚❚ PAUSE' : '▶ RUN';
  };
  $('btn-nudge').onclick = nudge;
  $('btn-reset').onclick = () => startEpisode(Math.floor(Math.random() * 1e6));
  $('push-left').onclick = () => pressPush(-1);
  $('push-right').onclick = () => pressPush(1);
  $('agent').onchange = async (e) => {
    state.agentName = e.target.value;
    await loadAgent(state.agentName);
    if (state.sim) startEpisode(state.sim.seed);
    logEvent(`now balancing with ${LABEL[state.agentName]}`);
  };
  $('seed').onchange = (e) => startEpisode(Math.max(0, Math.floor(Number(e.target.value) || 0)));
  $('speed').onchange = (e) => setSpeed(Number(e.target.value));
  $('btn-verify').onclick = verifyPort;
  $('btn-train').onclick = startBrowserTraining;
  $('curve-legend').onclick = (e) => {
    const b = e.target.closest('[data-curve]');
    if (!b) return;
    state.visible[b.dataset.curve] = !state.visible[b.dataset.curve];
    renderLegend();
    drawCurves();
  };
  addEventListener('keydown', (e) => {
    if (e.target && /INPUT|SELECT|TEXTAREA/.test(e.target.tagName)) return;
    if (e.key === 'ArrowLeft' || e.key === 'a' || e.key === 'A') { e.preventDefault(); pressPush(-1); }
    else if (e.key === 'ArrowRight' || e.key === 'd' || e.key === 'D') { e.preventDefault(); pressPush(1); }
    else if (e.key === ' ') { e.preventDefault(); $('btn-run').click(); }
    else if (e.key === 'n' || e.key === 'N') nudge();
  });
  new ResizeObserver(() => { sizeW = 0; }).observe(stage);
}

// ── Main loop ──────────────────────────────────────────────────────────
function frameLoop(ts) {
  const now = performance.now();
  const dt = state.lastTs ? Math.min(0.1, (ts - state.lastTs) / 1000) : 0;
  state.lastTs = ts;
  const sim = state.sim;
  if (sim) {
    if (sim.waiting) {
      if (now >= sim.resumeAt) {
        sim.waiting = false;
        startEpisode(Math.floor(sim.seed) + 1);
      }
    } else if (state.running) {
      // The physics runs at 50 steps per second of real time, times the speed setting.
      state.stepAcc += dt * 50 * state.speed;
      while (state.stepAcc >= 1 && sim && !sim.waiting) {
        advance(now);
        state.stepAcc -= 1;
      }
    }
    $('chip-steps').textContent = String(sim.s.steps);
  }
  state.frame += 1;
  drawStage(now);
  if (state.frame % 6 === 0) renderMind();
  $('chip-time').textContent = `${(now / 1000).toFixed(1)} s`;
  requestAnimationFrame(frameLoop);
}

async function main() {
  init();
  state.meta = await getJSON('/api/cartpole/meta');
  const sel = $('agent');
  sel.innerHTML = state.meta.agents.map((a) => `<option value="${a.name}">${a.label}</option>`).join('');
  sel.value = state.agentName;
  try {
    state.curves = await getJSON('/api/cartpole/curves');
  } catch {
    state.curves = null;
  }
  renderLegend();
  await Promise.all(['reinforce', 'actor_critic', 'cem'].map((n) => loadAgent(n).catch(() => null)));
  await loadAgent('pd');
  await loadAgent('random');
  startEpisode(1);
  $('chip-status').textContent = 'RUNNING';
  logEvent('weights loaded. The policies run here in JavaScript.');
  drawCurves();
  loadBenchmark();
  requestAnimationFrame(frameLoop);
}

main().catch((e) => {
  $('chip-status').textContent = 'ERROR';
  logEvent(`could not start: ${e.message}`, C.pink);
});
