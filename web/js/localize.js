// Lost Robot page.
//
// The filters and the robot run here, in localize-core.js, so the page needs no server round trips. This file
// drives the simulation at a fixed tick, draws the floor, the particle cloud, the beams, the estimate and its
// uncertainty ellipse, plots the errors and the effective sample size, and writes the "what the filters believe"
// text from the same numbers the filters report.
//
// The robot is hidden from both filters. They see only the commanded drive and turn (the odometry, which they know)
// and the beams. The autopilot steers by the true pose, which is what a robot's own controller would know.

import {
  CELL_M, HEADS, MAX_RANGE, SUB, TAU, Autopilot, GridFilter, ParticleFilter, Sim, countModes, makeFloor, LAYOUTS,
} from './localize-core.js';
import { banner, burst } from './fx.js';

const $ = (id) => document.getElementById(id);
const TICK_MS = 240;     // slow enough to watch the belief split and collapse (about four steps a second)
const LOCK_M = 0.5;      // the estimate counts as locked when it is within this distance ...
const LOCK_STEPS = 10;   // ... for this many steps in a row
const HISTORY = 240;     // points kept for the charts
const MAP_W = 960, MAP_H = 640;

// Seed 4 on Twin Halls starts the robot in a place that looks like two others: the belief splits into three
// places, narrows to one within six steps, and stays on the robot from then on (the LOCKED banner waits for ten
// steps in a row). The seed is a choice, not a tuned result; other seeds split for longer or never settle, and
// the benchmark counts those.
const S = {
  layout: 'halls', seed: 4, particles: 2000, rays: 8, sigma: 0.3, scale: 1.0,
  auto: true, running: true, heat: false, showParticles: true, showRays: true,
};
const keys = new Set();          // w/a/s/d held (keyboard or touch)
let kidnapPending = false;
let floor = null, sim = null, ap = null, gf = null, pf = null;
let step = 0, lockRun = 0, locked = false, lastModes = 0, lastInject = false;
let history = [];                // {pf, grid, ess, inj} per step
let staticCanvas = null;         // walls and the neon grid, drawn once per floor
let heatCanvas = null;           // the grid belief, drawn when the heat map is on
const canvas = $('lr-map'), ctx = canvas.getContext('2d');
canvas.width = MAP_W;   // the drawing buffer; CSS scales it to the column width
canvas.height = MAP_H;

function log(text, cls = '') {
  const el = $('log');
  const div = document.createElement('div');
  div.textContent = `[${String(step).padStart(4, '0')}] ${text}`;
  if (cls) div.className = cls;
  el.prepend(div);
  while (el.childElementCount > 60) el.lastElementChild.remove();
}

// Build everything for the current floor, seed and sensor settings. Changing the beam count rebuilds the grid's
// sensor table; changing only the noise levels is applied live to the existing objects.
function newWorld(message) {
  floor = makeFloor(S.layout, S.seed);
  sim = new Sim(floor, S.seed * 7 + 1, S.rays, S.sigma, 0.05, S.scale);
  ap = new Autopilot(floor, S.seed * 13 + 5);
  gf = new GridFilter(floor, S.rays, S.sigma, 0.05, S.scale);
  pf = new ParticleFilter(floor, S.particles, S.rays, S.sigma, 0.05, S.scale, S.seed * 17 + 9);
  // The first scan, before anything moves: the same starting step the CLI and the benchmark take.
  pf.step(0, 0, sim.z);
  gf.update(sim.z);
  step = 0; lockRun = 0; locked = false; lastModes = countModes(pf.positionHistogram(), 0.02);
  history = [];
  kidnapPending = false;
  buildStatic();
  heatCanvas = null;
  $('lr-info').textContent = `${floor.title} · seed ${S.seed} · ${floor.pillars.length} pillar${floor.pillars.length === 1 ? '' : 's'} · the robot starts at a random place`;
  $('chip-status').textContent = 'LOST';
  log(message || `new floor: ${floor.title}, seed ${S.seed}`, 'log-win');
  paintText();
}

// The static layer: the floor's walls with a neon edge, and a faint grid of cell lines. Drawn once per floor.
function buildStatic() {
  staticCanvas = document.createElement('canvas');
  staticCanvas.width = MAP_W;
  staticCanvas.height = MAP_H;
  const c = staticCanvas.getContext('2d');
  const cs = MAP_W / floor.w;
  c.fillStyle = '#02060f';
  c.fillRect(0, 0, MAP_W, MAP_H);
  c.strokeStyle = 'rgba(0,245,255,0.07)';
  c.lineWidth = 1;
  for (let x = 0; x <= floor.w; x++) { c.beginPath(); c.moveTo(x * cs, 0); c.lineTo(x * cs, MAP_H); c.stroke(); }
  for (let y = 0; y <= floor.h; y++) { c.beginPath(); c.moveTo(0, y * cs); c.lineTo(MAP_W, y * cs); c.stroke(); }
  // Each wall cell is a dark block with a neon rim: the rims of neighbouring walls merge into lines.
  for (let y = 0; y < floor.h; y++) {
    for (let x = 0; x < floor.w; x++) {
      if (!floor.walls[y * floor.w + x]) continue;
      c.fillStyle = 'rgba(255,0,160,0.18)';
      c.fillRect(x * cs, y * cs, cs, cs);
      c.strokeStyle = 'rgba(255,0,160,0.9)';
      c.lineWidth = 2;
      c.strokeRect(x * cs + 2, y * cs + 2, cs - 4, cs - 4);
    }
  }
}

// The grid belief as a heat map, one pixel per position bin, scaled up with smoothing off so the bins show.
function buildHeat() {
  const map = gf.positionMap();
  let max = 0;
  for (const v of map) if (v > max) max = v;
  heatCanvas = document.createElement('canvas');
  heatCanvas.width = gf.w2;
  heatCanvas.height = gf.h2;
  const c = heatCanvas.getContext('2d');
  const img = c.createImageData(gf.w2, gf.h2);
  for (let i = 0; i < map.length; i++) {
    // The square root spreads out the low values, so faint belief is still visible next to the peak.
    const t = max > 0 ? Math.sqrt(map[i] / max) : 0;
    img.data[i * 4] = 0;
    img.data[i * 4 + 1] = Math.round(245 * t);
    img.data[i * 4 + 2] = Math.round(255 * t);
    img.data[i * 4 + 3] = Math.round(230 * t);
  }
  c.putImageData(img, 0, 0);
}

// One tick: choose the drive, move the robot, then both filters predict and update on the same odometry and beams.
function tick() {
  if (!S.running || !floor) return;
  let rot = 0, fwd = 0, z;
  if (kidnapPending) {
    // A kidnap is a teleport with no motion: the filters see zero odometry and a new set of beams.
    kidnapPending = false;
    z = sim.kidnap();
    ap.route = [];
    log('kidnapped: the robot was carried to a new place', 'log-warn');
  } else {
    if (keys.has('a')) rot -= 0.2;
    if (keys.has('d')) rot += 0.2;
    if (keys.has('w')) fwd += 0.3;
    if (keys.has('s')) fwd -= 0.2;
    if (!rot && !fwd && S.auto) [rot, fwd] = ap.command(sim.x, sim.y, sim.th);
    z = sim.drive(rot, fwd);
  }
  step++;
  gf.predict(rot, fwd);
  gf.update(z);
  if (S.heat) buildHeat();
  pf.step(rot, fwd, z);
  const e = pf.estimate();
  const g = gf.estimate();
  const pfErr = Math.hypot(e.x - sim.x, e.y - sim.y) * CELL_M;
  const gErr = Math.hypot(g.x - sim.x, g.y - sim.y) * CELL_M;
  const modes = countModes(pf.positionHistogram(), 0.02);
  history.push({ pf: pfErr, grid: gErr, ess: pf.ess / pf.n, inj: pf.injected / pf.n });
  if (history.length > HISTORY) history.shift();
  // Lock: the particle estimate has stayed within LOCK_M for LOCK_STEPS steps.
  lockRun = pfErr <= LOCK_M ? lockRun + 1 : 0;
  if (!locked && lockRun >= LOCK_STEPS) {
    locked = true;
    log(`locked on after ${step} steps: estimate ${pfErr.toFixed(2)} m from the robot`, 'log-win');
    banner('LOCKED', `particles agree on one place after ${step} steps`, '#00ff88');
    burst(canvas);
  } else if (locked && pfErr > 1.0) {
    locked = false;
    log(`lost it again: estimate ${pfErr.toFixed(2)} m away`, 'log-warn');
  }
  if (modes !== lastModes) {
    log(`the belief now sits in ${modes} place${modes === 1 ? '' : 's'}`, modes > 1 ? 'log-warn' : 'log-win');
    lastModes = modes;
  }
  const injecting = pf.injected > 0;
  if (injecting && !lastInject) {
    log(`kidnap suspected: injecting ${pf.injected} random particles`, 'log-warn');
    banner('KIDNAP?', 'the beams stopped matching the belief', '#ff00a0');
  }
  lastInject = injecting;
  paintText({ pfErr, gErr, modes, e, g, rot, fwd });
}

// The text panel: the same numbers the filters report, in words.
function paintText(v) {
  const body = $('think-body');
  if (!v) {
    body.innerHTML = 'The particles start spread over the whole floor. Drive the robot or start the autopilot to see the belief change.';
    $('think-title').textContent = 'Ready';
    return;
  }
  const { pfErr, gErr, modes, e, g, rot, fwd } = v;
  const pct = (x) => `${Math.round(100 * x)}%`;
  const split = modes > 1
    ? `<span class="lr-warn">The particles sit in ${modes} separate places</span>: the beams from these places look alike, so the filter cannot yet tell them apart. Moving on will show which one is right.`
    : `The particles agree on one place.`;
  const injecting = pf.injected > 0
    ? ` <span class="lr-warn">The beams have stopped matching the belief, so ${pf.injected} particles were re-scattered over the floor.</span>`
    : '';
  const cmd = rot || fwd ? '' : (S.auto ? '' : ' The robot is standing still.');
  $('think-title').textContent = locked ? 'LOCKED' : (step === 0 ? 'Ready' : 'Searching');
  body.innerHTML =
    `<p style="margin:0 0 0.5rem">${split}${injecting}${cmd}</p>` +
    `<p style="margin:0 0 0.5rem">Particle estimate: <span class="lr-num">${pfErr.toFixed(2)} m</span> from the robot, ` +
    `holding <span class="lr-num">${pct(e.mass)}</span> of the weight in its cluster. ESS ` +
    `<span class="lr-num">${pf.ess.toFixed(0)}</span> of <span class="lr-num">${pf.n}</span>.</p>` +
    `<p style="margin:0">Grid filter (exact bins): best bin <span class="lr-num">${gErr.toFixed(2)} m</span> away, ` +
    `holding <span class="lr-num">${pct(g.mass)}</span> of the probability.</p>`;
  $('chip-status').textContent = locked ? 'LOCKED' : (injecting ? 'KIDNAPPED?' : 'SEARCHING');
  $('chip-pf').textContent = `${pfErr.toFixed(2)} m`;
  $('chip-grid').textContent = `${gErr.toFixed(2)} m`;
  $('chip-step').textContent = String(step);
  $('lr-info').textContent = locked
    ? `Locked: the particles and the robot agree to ${pfErr.toFixed(2)} m.`
    : 'Drive the robot with the arrow keys or WASD, or let the autopilot run. The filters never see the robot\'s true position.';
}

// --- drawing ---------------------------------------------------------------------------------------------------

function ellipse(cov, x, y, cs) {
  // 2-sigma ellipse of a 2x2 covariance: eigenvalues give the radii, the eigenvector angle gives the rotation.
  const [[a, b], [, d]] = cov;
  const tr = a + d;
  const disc = Math.sqrt(((a - d) / 2) ** 2 + b * b);
  const l1 = tr / 2 + disc, l2 = Math.max(0, tr / 2 - disc);
  const ang = 0.5 * Math.atan2(2 * b, a - d);
  ctx.save();
  ctx.translate(x * cs, y * cs);
  ctx.rotate(ang);
  ctx.beginPath();
  ctx.ellipse(0, 0, Math.max(0.15, 2 * Math.sqrt(l1)) * cs, Math.max(0.15, 2 * Math.sqrt(l2)) * cs, 0, 0, TAU);
  ctx.setLineDash([6, 5]);
  ctx.strokeStyle = 'rgba(0,245,255,0.85)';
  ctx.lineWidth = 2;
  ctx.stroke();
  ctx.restore();
}

function draw() {
  if (!floor) return;
  const cs = MAP_W / floor.w;
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.drawImage(staticCanvas, 0, 0);
  if (S.heat && heatCanvas) {
    ctx.imageSmoothingEnabled = false;
    ctx.globalAlpha = 0.85;
    ctx.drawImage(heatCanvas, 0, 0, MAP_W, MAP_H);
    ctx.globalAlpha = 1;
  }
  // Particles: each is a small dot, brighter and larger the heavier its weight.
  if (S.showParticles) {
    let max = 0;
    for (let i = 0; i < pf.n; i++) if (pf.w[i] > max) max = pf.w[i];
    for (let i = 0; i < pf.n; i++) {
      const rel = max > 0 ? pf.w[i] / max : 0;
      ctx.globalAlpha = 0.18 + 0.82 * rel;
      ctx.fillStyle = rel > 0.5 ? '#e8ffff' : '#00f5ff';
      const r = 1.6 + 2.8 * rel;
      ctx.fillRect(pf.x[i] * cs - r, pf.y[i] * cs - r, 2 * r, 2 * r);
    }
    ctx.globalAlpha = 1;
  }
  // Beams: from the true robot to each reading's end point (a dropout or a max-range reading ends at the limit).
  if (S.showRays && sim.z) {
    ctx.strokeStyle = 'rgba(0,255,136,0.45)';
    ctx.lineWidth = 1.2;
    for (let j = 0; j < sim.rays; j++) {
      const a = sim.th + (TAU * j) / sim.rays;
      const d = sim.z[j];
      ctx.beginPath();
      ctx.moveTo(sim.x * cs, sim.y * cs);
      ctx.lineTo((sim.x + d * Math.cos(a)) * cs, (sim.y + d * Math.sin(a)) * cs);
      ctx.stroke();
    }
  }
  // Estimate: the uncertainty ellipse of the densest cluster, and a dashed ghost at its centre with a heading tick.
  const e = pf.estimate();
  ellipse(e.cov, e.x, e.y, cs);
  ctx.strokeStyle = 'rgba(0,245,255,0.9)';
  ctx.lineWidth = 2;
  ctx.setLineDash([3, 4]);
  ctx.beginPath();
  ctx.arc(e.x * cs, e.y * cs, 0.45 * cs, 0, TAU);
  ctx.stroke();
  ctx.setLineDash([]);
  ctx.beginPath();
  ctx.moveTo(e.x * cs, e.y * cs);
  ctx.lineTo((e.x + 0.9 * Math.cos(e.th)) * cs, (e.y + 0.9 * Math.sin(e.th)) * cs);
  ctx.stroke();
  // Grid estimate: a green cross.
  const g = gf.estimate();
  ctx.strokeStyle = '#00ff88';
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(g.x * cs - 8, g.y * cs); ctx.lineTo(g.x * cs + 8, g.y * cs);
  ctx.moveTo(g.x * cs, g.y * cs - 8); ctx.lineTo(g.x * cs, g.y * cs + 8);
  ctx.stroke();
  // The true robot: a yellow disc with a heading line. Drawn last so it is always visible.
  ctx.fillStyle = '#ffe600';
  ctx.shadowColor = '#ffe600';
  ctx.shadowBlur = 12;
  ctx.beginPath();
  ctx.arc(sim.x * cs, sim.y * cs, 0.22 * cs, 0, TAU);
  ctx.fill();
  ctx.shadowBlur = 0;
  ctx.strokeStyle = '#ffe600';
  ctx.lineWidth = 3;
  ctx.beginPath();
  ctx.moveTo(sim.x * cs, sim.y * cs);
  ctx.lineTo((sim.x + 0.5 * Math.cos(sim.th)) * cs, (sim.y + 0.5 * Math.sin(sim.th)) * cs);
  ctx.stroke();
}

function drawChart(id, series, yMax, threshold, colors) {
  const c = $(id);
  const x = c.getContext('2d');
  const W = c.width, H = c.height, pad = 30;
  x.clearRect(0, 0, W, H);
  x.strokeStyle = 'rgba(0,245,255,0.12)';
  x.lineWidth = 1;
  for (let i = 0; i <= 4; i++) {
    const y = pad + (H - 2 * pad) * (i / 4);
    x.beginPath(); x.moveTo(pad, y); x.lineTo(W - 8, y); x.stroke();
  }
  x.fillStyle = 'rgba(255,255,255,0.45)';
  x.font = '18px JetBrains Mono, monospace';
  for (let i = 0; i <= 4; i++) {
    const v = yMax * (1 - i / 4);
    x.fillText(v.toFixed(yMax <= 1 ? 2 : 1), 2, pad + (H - 2 * pad) * (i / 4) + 6);
  }
  const yOf = (v) => H - pad - (H - 2 * pad) * Math.min(1, v / yMax);
  if (threshold !== null) {
    x.setLineDash([8, 6]);
    x.strokeStyle = 'rgba(255,0,160,0.8)';
    x.beginPath(); x.moveTo(pad, yOf(threshold)); x.lineTo(W - 8, yOf(threshold)); x.stroke();
    x.setLineDash([]);
  }
  series.forEach((key, si) => {
    x.strokeStyle = colors[si];
    x.lineWidth = 3;
    x.shadowColor = colors[si];
    x.shadowBlur = 8;
    x.beginPath();
    history.forEach((h, i) => {
      const px = pad + (W - pad - 8) * (i / (HISTORY - 1));
      const py = yOf(h[key]);
      if (i === 0) x.moveTo(px, py); else x.lineTo(px, py);
    });
    x.stroke();
    x.shadowBlur = 0;
  });
}

function drawCharts() {
  drawChart('lr-err', ['pf', 'grid'], 3, LOCK_M, ['#00f5ff', '#00ff88']);
  drawChart('lr-ess', ['ess'], 1, 0.5, ['#ffe600']);
}

function frame() {
  draw();
  drawCharts();
  requestAnimationFrame(frame);
}

// --- controls --------------------------------------------------------------------------------------------------

function syncToggle(el, on) {
  el.setAttribute('aria-pressed', String(on));
  el.classList.toggle('green', on);
}

function bindControls() {
  $('layout').addEventListener('change', (e) => { S.layout = e.target.value; newWorld(); });
  $('seed').addEventListener('change', (e) => {
    S.seed = Math.max(0, Math.min(4294967295, Math.floor(Number(e.target.value) || 0)));
    e.target.value = String(S.seed);
    newWorld();
  });
  $('btn-shuffle').addEventListener('click', () => {
    S.seed = 1 + Math.floor(Math.random() * 9999);
    $('seed').value = String(S.seed);
    newWorld(`new seed ${S.seed}`);
  });
  $('particles').addEventListener('change', (e) => {
    S.particles = Number(e.target.value);
    pf = new ParticleFilter(floor, S.particles, S.rays, S.sigma, 0.05, S.scale, S.seed * 17 + 9);
    log(`particle filter restarted with ${S.particles} particles`);
    paintText();
  });
  $('rays').addEventListener('input', (e) => { $('rays-val').textContent = e.target.value; });
  $('rays').addEventListener('change', (e) => { S.rays = Number(e.target.value); newWorld('beams changed: new run'); });
  $('sigma').addEventListener('input', (e) => {
    S.sigma = Number(e.target.value);
    $('sigma-val').textContent = `${S.sigma.toFixed(2)} cells (${(S.sigma * CELL_M).toFixed(2)} m)`;
    if (sim) { sim.sigma = S.sigma; pf.sigma = S.sigma; gf.sigma = S.sigma; }
  });
  $('scale').addEventListener('input', (e) => {
    S.scale = Number(e.target.value);
    $('scale-val').textContent = `${S.scale.toFixed(1)}×`;
    if (sim) { sim.scale = S.scale; pf.scale = S.scale; gf.scale = S.scale; }
  });
  $('btn-auto').addEventListener('click', () => {
    S.auto = !S.auto;
    syncToggle($('btn-auto'), S.auto);
    log(S.auto ? 'autopilot on' : 'autopilot off: drive with the keys');
  });
  $('btn-kidnap').addEventListener('click', () => { kidnapPending = true; });
  $('btn-run').addEventListener('click', () => {
    S.running = !S.running;
    $('btn-run').querySelector('.btn-txt').textContent = S.running ? '❚❚ PAUSE' : '▶ RUN';
    syncToggle($('btn-run'), S.running);
  });
  $('btn-new').addEventListener('click', () => {
    S.seed = 1 + Math.floor(Math.random() * 9999);
    $('seed').value = String(S.seed);
    newWorld(`new floor, seed ${S.seed}`);
  });
  $('btn-reset').addEventListener('click', () => newWorld(`replay seed ${S.seed}`));
  $('tog-particles').addEventListener('click', () => {
    S.showParticles = !S.showParticles;
    $('tog-particles').querySelector('.btn-txt').textContent = S.showParticles ? 'PARTICLES ON' : 'PARTICLES OFF';
    syncToggle($('tog-particles'), S.showParticles);
  });
  $('tog-rays').addEventListener('click', () => {
    S.showRays = !S.showRays;
    $('tog-rays').querySelector('.btn-txt').textContent = S.showRays ? 'BEAMS ON' : 'BEAMS OFF';
    syncToggle($('tog-rays'), S.showRays);
  });
  $('tog-heat').addEventListener('click', () => {
    S.heat = !S.heat;
    $('tog-heat').querySelector('.btn-txt').textContent = S.heat ? 'HEAT MAP ON' : 'GRID HEAT MAP';
    syncToggle($('tog-heat'), S.heat);
    if (S.heat) buildHeat();
  });

  // Drive keys: WASD and the arrows, held while pressed. Touch buttons set the same flags with pointer events.
  const keyMap = { ArrowUp: 'w', ArrowDown: 's', ArrowLeft: 'a', ArrowRight: 'd', w: 'w', a: 'a', s: 's', d: 'd',
    W: 'w', A: 'a', S: 's', D: 'd' };
  const typing = (t) => t && (t.tagName === 'INPUT' || t.tagName === 'SELECT' || t.tagName === 'TEXTAREA');
  addEventListener('keydown', (e) => {
    const k = keyMap[e.key];
    if (!k || typing(e.target)) return;
    e.preventDefault();
    keys.add(k);
    document.querySelector(`.lr-key[data-key="${k}"]`)?.classList.add('lr-held');
  });
  addEventListener('keyup', (e) => {
    const k = keyMap[e.key];
    if (!k) return;
    keys.delete(k);
    document.querySelector(`.lr-key[data-key="${k}"]`)?.classList.remove('lr-held');
  });
  addEventListener('blur', () => {
    keys.clear();
    document.querySelectorAll('.lr-key.lr-held').forEach((b) => b.classList.remove('lr-held'));
  });
  for (const btn of document.querySelectorAll('.lr-key')) {
    const k = btn.dataset.key;
    const down = (e) => { e.preventDefault(); keys.add(k); btn.classList.add('lr-held'); };
    const up = () => { keys.delete(k); btn.classList.remove('lr-held'); };
    btn.addEventListener('pointerdown', down);
    btn.addEventListener('pointerup', up);
    btn.addEventListener('pointerleave', up);
    btn.addEventListener('pointercancel', up);
  }
}

function init() {
  $('layout').value = S.layout;
  $('seed').value = String(S.seed);
  $('particles').value = String(S.particles);
  $('rays').value = String(S.rays);
  $('rays-val').textContent = String(S.rays);
  $('sigma').value = String(S.sigma);
  $('sigma-val').textContent = `${S.sigma.toFixed(2)} cells (${(S.sigma * CELL_M).toFixed(2)} m)`;
  $('scale').value = String(S.scale);
  $('scale-val').textContent = `${S.scale.toFixed(1)}×`;
  syncToggle($('btn-auto'), S.auto);
  syncToggle($('btn-run'), S.running);
  syncToggle($('tog-particles'), S.showParticles);
  syncToggle($('tog-rays'), S.showRays);
  syncToggle($('tog-heat'), S.heat);
  bindControls();
  newWorld();
  setInterval(tick, TICK_MS);
  requestAnimationFrame(frame);
}

init();
