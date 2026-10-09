// Snake page.
//
// Two learned agents run in this browser, and the page switches between them:
//   * the evaluation function (the default): eight weights over hand-built features of each move,
//     from /api/snake/evaluator, scored by snake-core.js moveTable and chooseEval;
//   * the neural net: 339 weights evolved by a genetic algorithm, from /api/snake/champion, run by
//     snake-core.js decide.
// Both do the arithmetic the benchmark measured (the weights are committed at four decimals). The
// server is only asked for the planner's hint, which is a BFS and a tail-chasing check.
//
// Three views of one decision: the board (where the snake goes), the explanation (which senses or
// features pushed the chosen turn, and by how much), and, for the net, the network (which units fire).

import { getJSON, postJSON } from './api.js';
import { banner, burst, pop, shake } from './fx.js';
import {
  ACTION_NAMES, EVAL_FEATURES, INPUT_NAMES, LEFT, RIGHT, STRAIGHT, alive, chooseEval, decide, moveTable,
  newGame, step,
} from './snake-core.js';

const $ = (id) => document.getElementById(id);
const REDUCED = matchMedia('(prefers-reduced-motion: reduce)').matches;
const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
const C = {
  cyan: css('--cyan') || '#00f5ff', pink: css('--pink') || '#ff00a0', green: css('--green') || '#00ff88',
  yellow: css('--yellow') || '#ffe600', dim: css('--dim') || '#4a7a9b', purple: css('--purple') || '#9b00ff',
};
// Short labels for the 17 senses, to fit the network diagram's margins.
const SHORT = ['danger ahead', 'danger left', 'danger right', 'free ahead', 'free left', 'free right',
  'food forward', 'food right', 'tail forward', 'tail right', 'heading N', 'heading E', 'heading S', 'heading W',
  'room ahead', 'room left', 'room right'];
// Key -> absolute heading (w north, d east, s south, a west). Arrow keys are the same directions.
const KEYS = { w: 0, d: 1, s: 2, a: 3, arrowup: 0, arrowright: 1, arrowdown: 2, arrowleft: 3 };
// The three-step learning loop, as the page describes each learned agent.
const LOOP = {
  eval: [
    ['1 · SAMPLE', 'Each candidate is a set of eight weights, drawn from a Gaussian around the elite mean. Every candidate plays the same seeded boards.'],
    ['2 · KEEP THE ELITE', 'The best 12 of 48 candidates are kept, and the Gaussian is refit to them. Its spread is floored, so the search keeps exploring.'],
    ['3 · REPEAT ON NEW BOARDS', 'Each generation draws fresh boards, so no candidate can memorise one fixed set. Starving counts as a death.'],
  ],
  net: [
    ['1 · PLAY', 'Each genome plays the same seeded boards. Fitness is apples eaten, plus a small bonus for surviving, which only breaks ties.'],
    ['2 · KEEP THE BEST', 'The top genomes are copied unchanged, so the best result never gets lost. Tournament selection picks parents from random groups of three.'],
    ['3 · CROSS AND MUTATE', 'Uniform crossover takes each weight from one parent or the other. Then about one weight in ten moves by a small Gaussian nudge.'],
  ],
};

const state = {
  net: null,            // { w1, b1, w2, b2 } from the champion JSON
  netMeta: null,        // the net's champion metadata: generation, training numbers, settings
  netHistory: [],       // the net's one record per generation
  evalW: null,          // the evaluation function's eight weights, in EVAL_FEATURES order
  evalMeta: null,       // its metadata: settings, best generation, validation numbers
  evalHistory: [],      // its one record per generation
  agent: 'eval',        // which learned agent steers: 'eval' (the default) or 'net'
  game: null,
  control: 'agent',     // 'agent' (the learned agent steers) or 'you' (keyboard or touch steers)
  running: false,       // the timer is ticking
  timer: 0,
  tickMs: 120,
  desired: 1,           // play mode: the heading the snake is asked to take
  hintPath: null,       // planner route to draw, as cells starting at the head
  decision: null,       // the last decision, for the explanation and the network view
  games: 0,
  bestApples: 0,
  lastCause: null,
};

// The learned agent in use, its metadata and its history.
const agentReady = () => (state.agent === 'eval' ? state.evalW !== null : state.net !== null);
const activeMeta = () => (state.agent === 'eval' ? state.evalMeta : state.netMeta);
const activeHistory = () => (state.agent === 'eval' ? state.evalHistory : state.netHistory);
const agentName = () => (state.agent === 'eval' ? 'The evaluation function' : 'The evolved net');

// ── Chips, log, notes ───────────────────────────────────────────────────

function setStatus(text, color = 'var(--green)') {
  $('chip-status').textContent = text;
  $('chip-status').style.color = color;
}

function setChips() {
  const g = state.game;
  $('chip-apples').textContent = g.apples;
  $('chip-steps').textContent = g.steps;
  $('chip-length').textContent = g.body.length;
}

function log(text, cls = 'log-info') {
  const line = document.createElement('div');
  line.className = cls;
  line.textContent = text;
  $('log').appendChild(line);
  $('log').scrollTop = $('log').scrollHeight;
}

function setButtons() {
  const watching = state.running && state.control === 'agent';
  const playing = state.running && state.control === 'you';
  $('btn-watch').setAttribute('aria-pressed', String(watching));
  $('btn-watch').querySelector('.btn-txt').textContent = watching ? '❚❚ PAUSE' : '▶ WATCH THE AGENT';
  $('btn-play').setAttribute('aria-pressed', String(playing));
  $('btn-play').querySelector('.btn-txt').textContent = playing ? '❚❚ PAUSE' : '✎ PLAY YOURSELF';
}

// ── Board ───────────────────────────────────────────────────────────────

function sizeCanvas(canvas, width, height) {
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  canvas.width = Math.round(width * dpr);
  canvas.height = Math.round(height * dpr);
  const ctx = canvas.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return ctx;
}

let boardCtx = null, boardPx = 0;

function resizeBoard() {
  const canvas = $('sn-board');
  const px = Math.round(Math.min(canvas.parentElement.clientWidth, 480));
  boardPx = px;
  boardCtx = sizeCanvas(canvas, px, px);
  canvas.style.width = px + 'px';
  canvas.style.height = px + 'px';
}

function drawBoard(now) {
  if (!boardCtx || !state.game) return;
  const ctx = boardCtx, g = state.game, n = g.size, W = boardPx, cell = W / n;
  ctx.clearRect(0, 0, W, W);
  // Floor: a dark field with a faint grid.
  const bg = ctx.createRadialGradient(W / 2, W / 2, W * 0.1, W / 2, W / 2, W * 0.75);
  bg.addColorStop(0, '#071a2e'); bg.addColorStop(1, '#020610');
  ctx.fillStyle = bg; ctx.fillRect(0, 0, W, W);
  ctx.strokeStyle = 'rgba(0,245,255,0.07)'; ctx.lineWidth = 1;
  for (let i = 1; i < n; i++) {
    ctx.beginPath(); ctx.moveTo(i * cell, 0); ctx.lineTo(i * cell, W); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(0, i * cell); ctx.lineTo(W, i * cell); ctx.stroke();
  }
  // Planner route, dashed pink, from the head along the path.
  if (state.hintPath && state.hintPath.length) {
    ctx.save();
    ctx.setLineDash([6, 6]); ctx.strokeStyle = C.pink; ctx.lineWidth = Math.max(2, cell * 0.12);
    ctx.shadowColor = C.pink; ctx.shadowBlur = 10;
    ctx.beginPath();
    const [hx, hy] = g.body[0];
    ctx.moveTo((hx + 0.5) * cell, (hy + 0.5) * cell);
    for (const [x, y] of state.hintPath) ctx.lineTo((x + 0.5) * cell, (y + 0.5) * cell);
    ctx.stroke();
    ctx.restore();
  }
  // Food: a pink orb that breathes.
  if (g.food) {
    const [fx, fy] = g.food;
    const pulse = REDUCED ? 0 : Math.sin(now / 180) * 0.08;
    const r = cell * (0.34 + pulse);
    const cx = (fx + 0.5) * cell, cy = (fy + 0.5) * cell;
    const grd = ctx.createRadialGradient(cx, cy, 1, cx, cy, r * 1.8);
    grd.addColorStop(0, 'rgba(255,0,160,0.9)'); grd.addColorStop(0.45, 'rgba(255,0,160,0.35)');
    grd.addColorStop(1, 'rgba(255,0,160,0)');
    ctx.fillStyle = grd; ctx.beginPath(); ctx.arc(cx, cy, r * 1.8, 0, Math.PI * 2); ctx.fill();
    ctx.fillStyle = '#ffd1ec'; ctx.beginPath(); ctx.arc(cx, cy, r * 0.55, 0, Math.PI * 2); ctx.fill();
  }
  // Snake: the tail is purple and fades into cyan toward the head. The head is green.
  const len = g.body.length;
  for (let i = len - 1; i >= 0; i--) {
    const [x, y] = g.body[i];
    const t = len > 1 ? 1 - i / (len - 1) : 1; // 0 at the tail, 1 at the head
    const pad = cell * 0.08;
    const size = cell - pad * 2;
    ctx.fillStyle = i === 0 ? C.green : mix(C.purple, C.cyan, t);
    ctx.globalAlpha = i === 0 ? 1 : 0.55 + 0.45 * t;
    if (i === 0) { ctx.shadowColor = C.green; ctx.shadowBlur = 16; }
    roundRect(ctx, x * cell + pad, y * cell + pad, size, size, cell * 0.22);
    ctx.fill();
    ctx.shadowBlur = 0;
  }
  ctx.globalAlpha = 1;
  // Eyes on the head, facing the heading.
  const [hx, hy] = g.body[0];
  const [ex, ey] = [[0, -1], [1, 0], [0, 1], [-1, 0]][g.heading];
  const cx = (hx + 0.5) * cell, cy = (hy + 0.5) * cell;
  ctx.fillStyle = '#021a10';
  for (const side of [-1, 1]) {
    const px = cx + ex * cell * 0.18 - ey * side * cell * 0.18;
    const py = cy + ey * cell * 0.18 + ex * side * cell * 0.18;
    ctx.beginPath(); ctx.arc(px, py, cell * 0.07, 0, Math.PI * 2); ctx.fill();
  }
  // Frame, brighter when the game has just ended.
  ctx.strokeStyle = alive(g) ? C.cyan : C.pink;
  ctx.lineWidth = 3; ctx.shadowColor = ctx.strokeStyle; ctx.shadowBlur = 18;
  ctx.strokeRect(1.5, 1.5, W - 3, W - 3);
  ctx.shadowBlur = 0;
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

// Linear blend of two hex colours, t in [0, 1].
function mix(a, b, t) {
  const pa = [1, 3, 5].map((i) => parseInt(a.slice(i, i + 2), 16));
  const pb = [1, 3, 5].map((i) => parseInt(b.slice(i, i + 2), 16));
  const c = pa.map((v, i) => Math.round(v + (pb[i] - v) * t));
  return `rgb(${c[0]},${c[1]},${c[2]})`;
}

// ── Network view ────────────────────────────────────────────────────────

let netCtx = null, netW = 0, netH = 0;

function resizeNet() {
  const canvas = $('sn-net');
  netW = canvas.parentElement.clientWidth;
  netH = Math.round(Math.max(260, Math.min(380, netW * 0.72)));
  netCtx = sizeCanvas(canvas, netW, netH);
  canvas.style.height = netH + 'px';
}

// The network view only exists for the neural net: the evaluation function has no hidden units.
function drawNet() {
  if (!netCtx || !state.net || state.agent !== 'net') return;
  const ctx = netCtx, W = netW, H = netH, net = state.net;
  const d = state.decision && state.decision.kind === 'net' ? state.decision : null;
  ctx.clearRect(0, 0, W, H);
  const narrow = W < 420;
  const left = narrow ? 92 : 118, right = narrow ? 64 : 84;
  const xIn = left, xHid = W / 2, xOut = W - right;
  const yAt = (i, n) => 14 + ((H - 28) * (i + 0.5)) / n;
  const nIn = net.w1.length, nHid = net.b1.length, nOut = 3;
  // Positions: inputs in a column, hidden in the middle, outputs on the right.
  const inY = Array.from({ length: nIn }, (_, i) => yAt(i, nIn));
  const hidY = Array.from({ length: nHid }, (_, j) => yAt(j, nHid));
  const outY = Array.from({ length: nOut }, (_, k) => H * (0.25 + k * 0.25));
  // Links into the hidden layer: x_i * w1[i][j]. Cyan adds to the hidden unit, pink subtracts.
  for (let i = 0; i < nIn; i++) {
    for (let j = 0; j < nHid; j++) {
      const c = d ? d.x[i] * net.w1[i][j] : net.w1[i][j] * 0.15;
      strokeLink(ctx, xIn, inY[i], xHid, hidY[j], c, d ? 0.6 : 0.12);
    }
  }
  // Links into the outputs: h_j * w2[j][k].
  for (let j = 0; j < nHid; j++) {
    for (let k = 0; k < nOut; k++) {
      const c = d ? d.h[j] * net.w2[j][k] : net.w2[j][k] * 0.15;
      strokeLink(ctx, xHid, hidY[j], xOut, outY[k], c, d ? 0.6 : 0.12);
    }
  }
  // Nodes: brightness is the unit's activity.
  ctx.font = `${narrow ? 9 : 10}px var(--mono, monospace)`;
  ctx.textBaseline = 'middle';
  for (let i = 0; i < nIn; i++) {
    const v = d ? Math.abs(d.x[i]) : 0;
    glowNode(ctx, xIn, inY[i], 4.5, v, d && d.x[i] < 0 ? C.pink : C.cyan);
    ctx.fillStyle = v > 0.05 ? C.cyan : C.dim;
    ctx.textAlign = 'right';
    ctx.fillText(SHORT[i], xIn - 9, inY[i]);
  }
  for (let j = 0; j < nHid; j++) {
    const v = d ? Math.abs(d.h[j]) : 0;
    glowNode(ctx, xHid, hidY[j], 3.2, v, d && d.h[j] < 0 ? C.pink : C.cyan);
  }
  const names = ['left', 'straight', 'right'];
  for (let k = 0; k < nOut; k++) {
    const p = d ? d.probs[k] : 0;
    const chosen = d && d.action === k;
    glowNode(ctx, xOut, outY[k], 9, p, chosen ? C.yellow : C.green);
    if (chosen) {
      ctx.strokeStyle = C.yellow; ctx.lineWidth = 2; ctx.shadowColor = C.yellow; ctx.shadowBlur = 14;
      ctx.beginPath(); ctx.arc(xOut, outY[k], 14, 0, Math.PI * 2); ctx.stroke(); ctx.shadowBlur = 0;
    }
    ctx.fillStyle = chosen ? C.yellow : C.dim;
    ctx.textAlign = 'left';
    ctx.fillText(d ? `${names[k]} ${Math.round(p * 100)}%` : names[k], xOut + 20, outY[k]);
  }
  ctx.textAlign = 'center';
  ctx.fillStyle = C.dim;
  ctx.fillText(`${nHid} hidden, tanh`, xHid, 9);
}

// A link from (x1,y1) to (x2,y2) whose colour and strength show the signed contribution c.
function strokeLink(ctx, x1, y1, x2, y2, c, scale) {
  const a = Math.min(1, Math.abs(c) * scale);
  ctx.strokeStyle = c >= 0 ? `rgba(0,245,255,${0.04 + 0.5 * a})` : `rgba(255,0,160,${0.04 + 0.5 * a})`;
  ctx.lineWidth = 0.6 + 1.6 * a;
  ctx.beginPath(); ctx.moveTo(x1, y1); ctx.lineTo(x2, y2); ctx.stroke();
}

function glowNode(ctx, x, y, r, v, color) {
  ctx.save();
  ctx.fillStyle = 'rgba(2,6,16,0.9)';
  ctx.beginPath(); ctx.arc(x, y, r + 1.5, 0, Math.PI * 2); ctx.fill();
  if (v > 0.02) {
    ctx.shadowColor = color; ctx.shadowBlur = 6 + 18 * Math.min(1, v);
    ctx.fillStyle = color; ctx.globalAlpha = Math.min(1, 0.25 + v);
    ctx.beginPath(); ctx.arc(x, y, r * (0.7 + 0.3 * Math.min(1, v)), 0, Math.PI * 2); ctx.fill();
  }
  ctx.restore();
}

// ── Decisions ───────────────────────────────────────────────────────────

// The decision for the current position under the chosen agent. For the evaluation function, `rows`
// holds every move with its features and score, `safe` the moves that do not end the game, and `x`
// the features of the chosen move.
function decideNow(g) {
  if (state.agent === 'eval') {
    const rows = moveTable(g, state.evalW);
    const action = chooseEval(state.evalW, g);
    const chosen = rows.find((r) => r.action === action);
    return { kind: 'eval', action, rows, safe: rows.filter((r) => !r.dies), x: chosen.features };
  }
  return { kind: 'net', ...decide(state.net, g) };
}

// ── Explanation panel ───────────────────────────────────────────────────

// Reads the net's decision in words: what the snake senses, what the net weighs, and what it does.
function explain(d, action, who) {
  if (d.kind === 'eval') return explainEval(d, action, who);
  const g = state.game;
  const chosen = ACTION_NAMES[action];
  const probs = d.probs;
  const order = [...d.attr.keys()].sort((a, b) => Math.abs(d.attr[b]) - Math.abs(d.attr[a])).slice(0, 3);
  const pieces = order.filter((i) => Math.abs(d.attr[i]) > 0.02).map((i) => {
    const push = d.attr[i] >= 0 ? 'pushes toward' : 'pushes against';
    return `${SHORT[i]} (${push} this turn)`;
  });
  const title = who === 'you'
    ? `You turn ${chosen}. The net would turn ${ACTION_NAMES[d.action]} (${Math.round(probs[d.action] * 100)}%).`
    : `The net turns ${chosen} (${Math.round(probs[action] * 100)}%).`;
  let body = pieces.length ? `Strongest influences: ${pieces.join('; ')}.` : 'No single sense dominates.';
  if (d.x[0] === 1 && d.x[1] === 0 && d.x[2] === 0) body += ' Only the way ahead is blocked, so it turns.';
  if (d.x[0] === 1 && d.x[1] === 1 && d.x[2] === 1) body += ' Blocked on all three sides: it picks the least bad turn.';
  if (g.food && d.x[6] > 0.3) body += ' The food is ahead of it.';
  $('think-title').textContent = title;
  $('think-text').textContent = body;
  renderBars(d);
}

// Reads the evaluation function's choice: the move's weighted terms, and the runner-up it beat.
function explainEval(d, action, who) {
  const chosen = d.rows.find((r) => r.action === action);
  const title = who === 'you'
    ? `You turn ${ACTION_NAMES[action]}. The evaluator would turn ${ACTION_NAMES[d.action]}.`
    : `The evaluator turns ${ACTION_NAMES[action]}.`;
  if (d.safe.length === 0) {
    $('think-title').textContent = title;
    $('think-text').textContent = 'Every turn ends the game, so it goes straight.';
    renderBars(d);
    return;
  }
  // You steered into a deadly cell: that move has no score, so explain it and stop.
  if (chosen.dies) {
    $('think-title').textContent = title;
    $('think-text').textContent = `Turning ${ACTION_NAMES[action]} ends the game, so the evaluator gives it no score.`;
    renderBars(d);
    return;
  }
  // Each term is the weight times the feature: the largest terms are the reasons for the score.
  const terms = chosen.features.map((v, i) => state.evalW[i] * v);
  const order = [...terms.keys()].sort((a, b) => Math.abs(terms[b]) - Math.abs(terms[a])).slice(0, 3);
  const pieces = order.filter((i) => Math.abs(terms[i]) > 0.02).map((i) => {
    const verb = terms[i] >= 0 ? 'adds' : 'takes away';
    return `${EVAL_FEATURES[i]} (${verb} ${Math.abs(terms[i]).toFixed(2)})`;
  });
  let body = pieces.length ? `Biggest terms in its score: ${pieces.join('; ')}.` : 'No single term dominates.';
  const runnerUp = d.safe.filter((r) => r.action !== action).sort((a, b) => b.score - a.score)[0];
  if (runnerUp) {
    body += ` Next best is turn ${ACTION_NAMES[runnerUp.action]} at ${runnerUp.score.toFixed(2)}, against ${chosen.score.toFixed(2)}.`;
  } else {
    body += ' It is the only safe turn.';
  }
  if (chosen.features[0] === 1) body += ' It eats.';
  $('think-title').textContent = title;
  $('think-text').textContent = body;
  renderBars(d);
}

function renderBars(d) {
  const inputs = $('think-inputs');
  inputs.innerHTML = '';
  const names = d.kind === 'eval' ? EVAL_FEATURES : INPUT_NAMES;
  names.forEach((name, i) => {
    const v = d.x[i];
    const row = document.createElement('div');
    row.className = 'sn-bar';
    const bar = document.createElement('div');
    bar.className = 'sn-bar-fill' + (v < 0 ? ' neg' : '');
    bar.style.width = `${Math.round(Math.abs(v) * 100)}%`;
    row.innerHTML = `<span class="sn-bar-name"></span><span class="sn-bar-track"></span><span class="sn-bar-val"></span>`;
    row.querySelector('.sn-bar-name').textContent = name;
    row.querySelector('.sn-bar-track').appendChild(bar);
    row.querySelector('.sn-bar-val').textContent = v.toFixed(2);
    inputs.appendChild(row);
  });
  const outputs = $('think-outputs');
  outputs.innerHTML = '';
  if (d.kind === 'eval') {
    // Each move's score, scaled between the lowest and highest safe score so the gap is visible.
    const scores = d.safe.map((r) => r.score);
    const lo = scores.length ? Math.min(...scores) : 0, hi = scores.length ? Math.max(...scores) : 0;
    [LEFT, STRAIGHT, RIGHT].forEach((k) => {
      const r = d.rows.find((x) => x.action === k);
      const row = document.createElement('div');
      row.className = 'sn-bar sn-bar-out' + (d.action === k ? ' chosen' : '');
      row.innerHTML = `<span class="sn-bar-name"></span><span class="sn-bar-track"><span class="sn-bar-fill"></span></span><span class="sn-bar-val"></span>`;
      row.querySelector('.sn-bar-name').textContent = `turn ${ACTION_NAMES[k]}`;
      const fill = row.querySelector('.sn-bar-fill');
      const val = row.querySelector('.sn-bar-val');
      if (r.dies) {
        fill.style.width = '0%';
        val.textContent = 'ends game';
      } else {
        fill.style.width = `${Math.round(hi > lo ? 10 + (90 * (r.score - lo)) / (hi - lo) : 100)}%`;
        val.textContent = r.score.toFixed(2);
      }
      outputs.appendChild(row);
    });
    return;
  }
  ['left', 'straight', 'right'].forEach((name, k) => {
    const row = document.createElement('div');
    row.className = 'sn-bar sn-bar-out' + (d.action === k ? ' chosen' : '');
    row.innerHTML = `<span class="sn-bar-name"></span><span class="sn-bar-track"><span class="sn-bar-fill"></span></span><span class="sn-bar-val"></span>`;
    row.querySelector('.sn-bar-name').textContent = `turn ${name}`;
    row.querySelector('.sn-bar-fill').style.width = `${Math.round(d.probs[k] * 100)}%`;
    row.querySelector('.sn-bar-val').textContent = `${Math.round(d.probs[k] * 100)}%`;
    outputs.appendChild(row);
  });
}

// ── Game flow ───────────────────────────────────────────────────────────

function newRound() {
  state.game = newGame(state.size, Date.now() & 0xffffffff);
  state.hintPath = null;
  state.desired = state.game.heading;
  state.decision = null;
  state.lastCause = null;
  setChips();
  drawBoard(performance.now());
  drawNet();
}

// The heading the keyboard or touch asked for, as the relative action this step.
function actionFromDesired() {
  const diff = (state.desired - state.game.heading + 4) % 4;
  if (diff === 1) return RIGHT;
  if (diff === 3) return LEFT;
  return STRAIGHT; // straight, or a reversal the rules forbid
}

// One step of the game: decide, move, redraw, and handle the end of a game.
function tick() {
  const g = state.game;
  if (!alive(g)) return;
  const d = decideNow(g);
  state.decision = d;
  const action = state.control === 'you' ? actionFromDesired() : d.action;
  const apples = g.apples;
  explain(d, action, state.control === 'you' ? 'you' : 'agent');
  state.hintPath = null;
  step(g, action);
  if (g.apples > apples) {
    pop($('sn-board'), '+1', C.green);
    burst($('sn-board'), { count: 40, colors: [C.green, C.pink, C.cyan] });
  }
  setChips();
  drawBoard(performance.now());
  drawNet();
  if (!alive(g)) endRound();
}

function endRound() {
  const g = state.game;
  state.games += 1;
  state.bestApples = Math.max(state.bestApples, g.apples);
  const cause = { wall: 'hit a wall', body: 'bit its own body', starved: 'starved (no apple for two laps)', full: 'filled the board' }[g.cause];
  log(`game ${state.games}: ${g.apples} apples in ${g.steps} steps, ${cause}`, g.cause === 'full' ? 'log-best' : 'log-val');
  shake($('sn-board'), 'small');
  if (g.cause === 'full') banner('BOARD FILLED', `${g.apples} apples`, C.green);
  setStatus(g.cause === 'full' ? 'FULL BOARD' : 'GAME OVER', C.pink);
  if (state.running && state.control === 'agent') {
    // Watching the agent: start the next game after a short pause, so the end is visible.
    clearTimeout(state.timer);
    state.timer = setTimeout(() => { newRound(); schedule(); }, REDUCED ? 300 : 1200);
  } else {
    state.running = false;
    setButtons();
  }
}

function schedule() {
  clearTimeout(state.timer);
  if (!state.running) return;
  state.timer = setTimeout(() => {
    tick();
    if (!alive(state.game)) return; // endRound() has already set what happens next
    setStatus(state.control === 'you' ? 'YOUR TURN' : 'WATCHING', state.control === 'you' ? C.pink : C.green);
    schedule();
  }, state.tickMs);
}

function start(control) {
  if (!agentReady()) return;
  if (state.running && state.control === control) {
    state.running = false;
    clearTimeout(state.timer);
    setStatus('PAUSED', C.yellow);
  } else {
    state.control = control;
    state.running = true;
    if (!alive(state.game)) newRound();
    setStatus(control === 'you' ? 'YOUR TURN' : 'WATCHING', control === 'you' ? C.pink : C.green);
    $('sn-note').textContent = control === 'you'
      ? 'W A S D or the arrow keys steer. The agent shows what it would have done.'
      : `${agentName()} is steering. Pause, step, or take the wheel at any time.`;
    schedule();
  }
  setButtons();
}

async function hint() {
  if (!state.game || !alive(state.game)) return;
  const g = state.game;
  try {
    const res = await postJSON('/api/snake/hint', { body: g.body, heading: g.heading, food: g.food });
    state.hintPath = [g.body[0], ...res.path];
    const where = { food: 'the food', tail: 'its tail (no safe route to the food yet)', none: 'nothing: no route at all' }[res.route];
    $('sn-note').textContent = `Planner: turn ${res.action_name}, heading for ${where}.`;
    drawBoard(performance.now());
  } catch (err) {
    $('sn-note').textContent = `Hint unavailable: ${err.message}`;
  }
}

function steer(dir) {
  if (!state.game || !agentReady()) return;
  if (!alive(state.game)) newRound(); // a key after a finished game starts a new one
  const g = state.game;
  if ((dir + 2) % 4 === g.heading) return; // a reversal is not allowed
  state.desired = dir;
  if (state.control !== 'you' || !state.running) {
    state.control = 'you';
    if (!state.running) state.running = true;
    setStatus('YOUR TURN', C.pink);
    schedule();
    setButtons();
  }
}

// ── Agent switch, chart and champion card ───────────────────────────────

// Switches the learned agent that steers. The game in progress carries on with the new agent.
function setAgent(name) {
  state.agent = name;
  state.decision = null;
  setAgentUI();
  drawNet();
}

function setAgentUI() {
  const isEval = state.agent === 'eval';
  $('btn-agent-eval').setAttribute('aria-pressed', String(isEval));
  $('btn-agent-net').setAttribute('aria-pressed', String(!isEval));
  $('net-panel').hidden = isEval;
  $('think-section-title').textContent = isEval ? 'WHAT THE AGENT IS THINKING' : 'WHAT THE NET IS THINKING';
  $('loop-section-title').textContent = isEval ? 'THE LEARNING LOOP: CROSS-ENTROPY' : 'THE LEARNING LOOP: GENETIC ALGORITHM';
  LOOP[state.agent].forEach(([title, text], i) => {
    $(`loop-t-${i + 1}`).textContent = title;
    $(`loop-x-${i + 1}`).textContent = text;
  });
  renderChampion();
  drawChart();
  setButtons();
}

function drawChart() {
  const canvas = $('sn-chart');
  const w = canvas.parentElement.clientWidth;
  const h = Math.round(Math.max(200, Math.min(260, w * 0.6)));
  const ctx = sizeCanvas(canvas, w, h);
  canvas.style.height = h + 'px';
  ctx.clearRect(0, 0, w, h);
  const hist = activeHistory();
  if (!hist.length) return;
  const pad = { l: 36, r: 12, t: 12, b: 24 };
  const xs = (g) => pad.l + ((g - 1) / Math.max(1, hist.length - 1)) * (w - pad.l - pad.r);
  const top = Math.max(1, Math.ceil(Math.max(...hist.map((r) => r.best_apples)) + 0.5));
  const ys = (v) => h - pad.b - (v / top) * (h - pad.t - pad.b);
  ctx.strokeStyle = 'rgba(0,245,255,0.12)'; ctx.fillStyle = C.dim; ctx.font = '10px monospace';
  for (let t = 0; t <= top; t += Math.max(1, Math.round(top / 4))) {
    ctx.beginPath(); ctx.moveTo(pad.l, ys(t)); ctx.lineTo(w - pad.r, ys(t)); ctx.stroke();
    ctx.textAlign = 'right'; ctx.fillText(String(t), pad.l - 6, ys(t) + 3);
  }
  ctx.textAlign = 'center';
  ctx.fillText('generation', w / 2, h - 6);
  ctx.fillText(String(hist.length), xs(hist.length), h - 6);
  ctx.fillText('1', xs(1), h - 6);
  const line = (key, color, width) => {
    ctx.strokeStyle = color; ctx.lineWidth = width; ctx.shadowColor = color; ctx.shadowBlur = 8;
    ctx.beginPath();
    hist.forEach((r, i) => (i ? ctx.lineTo(xs(r.generation), ys(r[key])) : ctx.moveTo(xs(r.generation), ys(r[key]))));
    ctx.stroke(); ctx.shadowBlur = 0;
  };
  line('mean_apples', C.pink, 1.5);
  line('best_apples', C.green, 2.2);
  const gen = activeMeta() && activeMeta().best_generation;
  if (gen) {
    ctx.strokeStyle = C.yellow; ctx.setLineDash([4, 4]); ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(xs(gen), pad.t); ctx.lineTo(xs(gen), h - pad.b); ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = C.yellow; ctx.textAlign = 'left';
    ctx.fillText(`champion gen ${gen}`, Math.min(xs(gen) + 4, w - 110), pad.t + 10);
  }
  const last = hist[hist.length - 1];
  $('chart-title').textContent = `Gen ${last.generation}: best ${last.best_apples.toFixed(2)} apples per game, population mean ${last.mean_apples.toFixed(2)}.`;
}

// The champion card for the active agent: its generation, its numbers, and for the evaluator its weights.
function renderChampion() {
  const meta = activeMeta() || {};
  const s = meta.settings || {};
  $('chip-gen').textContent = meta.best_generation ?? '—';
  if (state.agent === 'eval') {
    const weights = state.evalW
      ? EVAL_FEATURES.map((name, i) => `<div class="sn-champ-row"><span>${name}</span><b>${state.evalW[i].toFixed(3)}</b></div>`).join('')
      : '';
    $('champ').innerHTML = `
    <div class="sn-champ-row"><span>generation</span><b>${meta.best_generation ?? '—'} of ${s.generations ?? '—'}</b></div>
    <div class="sn-champ-row"><span>validation apples per game</span><b>${Number(meta.validation_apples ?? 0).toFixed(2)}</b></div>
    <div class="sn-champ-row"><span>candidates x games</span><b>${s.pop ?? '—'} x ${s.games ?? '—'}</b></div>
    <div class="sn-champ-row"><span>elites kept</span><b>${s.elites ?? '—'}</b></div>
    <div class="field-hint" style="margin-top:0.6rem">The eight weights. Each term is its weight times its feature.</div>
    ${weights}
    <div class="field-hint">Trained on seeds 1000 to 39,999. Chosen on seeds 40,000 to 40,031. The benchmark uses 50,000 and up.</div>`;
    return;
  }
  $('champ').innerHTML = `
    <div class="sn-champ-row"><span>generation</span><b>${meta.best_generation ?? '—'} of ${s.generations ?? '—'}</b></div>
    <div class="sn-champ-row"><span>training apples per game</span><b>${Number(meta.train_apples ?? 0).toFixed(2)}</b></div>
    <div class="sn-champ-row"><span>population x games</span><b>${s.pop ?? '—'} x ${s.games ?? '—'}</b></div>
    <div class="sn-champ-row"><span>elites, mutation</span><b>${s.elites ?? '—'}, ${s.mutation_rate ?? '—'} at ${s.mutation_sigma ?? '—'}</b></div>
    <div class="field-hint">Trained on seeds 1000 and up. The benchmark uses seeds 50000 and up, so these boards are new to it.</div>`;
}

function renderPlayers(agents) {
  const box = $('players');
  box.innerHTML = '';
  for (const [name, desc] of Object.entries(agents)) {
    const card = document.createElement('div');
    card.className = 'heuristic-card';
    card.style.padding = '0.7rem';
    card.innerHTML = `<div style="font-family:var(--orb);font-size:0.65rem;color:var(--cyan);margin-bottom:0.3rem"></div><div class="field-hint"></div>`;
    card.firstChild.textContent = name.toUpperCase();
    card.lastChild.textContent = desc;
    box.appendChild(card);
  }
}

// ── Wiring ──────────────────────────────────────────────────────────────

function bind() {
  $('btn-watch').addEventListener('click', () => start('agent'));
  $('btn-play').addEventListener('click', () => start('you'));
  $('btn-agent-eval').addEventListener('click', () => setAgent('eval'));
  $('btn-agent-net').addEventListener('click', () => setAgent('net'));
  $('btn-step').addEventListener('click', () => {
    if (!agentReady()) return;
    if (!alive(state.game)) newRound();
    state.running = false; clearTimeout(state.timer); setButtons();
    tick();
    setStatus('STEPPED', C.yellow);
  });
  $('btn-hint').addEventListener('click', hint);
  $('btn-new').addEventListener('click', () => {
    state.running = false; clearTimeout(state.timer);
    newRound(); setStatus('READY', C.green); setButtons();
  });
  $('speed').addEventListener('input', (e) => {
    state.tickMs = Number(e.target.value);
    $('speed-val').textContent = `${state.tickMs} ms per step`;
  });
  document.querySelectorAll('.sn-key').forEach((b) => b.addEventListener('click', () => steer(Number(b.dataset.dir))));
  window.addEventListener('keydown', (e) => {
    const dir = KEYS[e.key.toLowerCase()];
    if (dir === undefined) return;
    if (e.target && ['INPUT', 'SELECT', 'TEXTAREA'].includes(e.target.tagName)) return;
    e.preventDefault();
    steer(dir);
  });
  // Swipe on the board steers on touch screens.
  let touch = null;
  const board = $('sn-board');
  board.addEventListener('touchstart', (e) => { const t = e.touches[0]; touch = [t.clientX, t.clientY]; }, { passive: true });
  board.addEventListener('touchend', (e) => {
    if (!touch) return;
    const t = e.changedTouches[0];
    const dx = t.clientX - touch[0], dy = t.clientY - touch[1];
    touch = null;
    if (Math.max(Math.abs(dx), Math.abs(dy)) < 18) return;
    steer(Math.abs(dx) > Math.abs(dy) ? (dx > 0 ? 1 : 3) : (dy > 0 ? 2 : 0));
  }, { passive: true });
  let resizeTimer = 0;
  window.addEventListener('resize', () => {
    clearTimeout(resizeTimer);
    resizeTimer = setTimeout(() => { resizeBoard(); resizeNet(); drawChart(); drawBoard(performance.now()); drawNet(); }, 120);
  });
  // The food keeps breathing while the page is visible.
  // About 30 frames a second is plenty for the pulse, and it keeps phones cool.
  let last = 0;
  const frame = (now) => {
    if (!document.hidden && now - last > 33) { last = now; drawBoard(now); }
    requestAnimationFrame(frame);
  };
  requestAnimationFrame(frame);
}

async function load() {
  bind();
  resizeBoard();
  resizeNet();
  newRound();
  try {
    const [meta, champ, hist, ev] = await Promise.all([
      getJSON('/api/snake/meta'),
      getJSON('/api/snake/champion'),
      getJSON('/api/snake/history'),
      getJSON('/api/snake/evaluator'),
    ]);
    state.net = { w1: champ.w1, b1: champ.b1, w2: champ.w2, b2: champ.b2 };
    state.netMeta = champ;
    state.netHistory = hist.history;
    state.evalW = ev.weights;
    state.evalMeta = ev;
    state.evalHistory = ev.history;
    state.size = meta.board;
    renderPlayers(meta.agents);
    setAgentUI();
    newRound();
    setStatus('READY', C.green);
    start('agent');
  } catch (err) {
    setStatus('OFFLINE', '#ff3b3b');
    $('sn-note').textContent = `The learned agents could not be loaded: ${err.message}`;
  }
}

// The board size is fixed by the rules; newRound() reads it, so it must be set before the first game.
state.size = 12;
load();
