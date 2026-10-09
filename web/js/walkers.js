// EVOLVING WALKERS page: the arena, the charts, the lineage and hall of fame, the creature lab, and the wiring to
// the genetic algorithm in a Web Worker (walkers-worker.js). The physics and the GA live in walkers-core.js; this
// file only draws, reads the controls, and sends the worker messages.
//
// Flow: the worker runs one generation at a time and posts its summary, the best four creatures as frames (60 Hz
// positions, transferred without copying), and the new lineage records. The arena plays the newest champion and the
// runners-up in a loop, with translucent ghosts of the last few champions racing alongside. Anything the user loads
// (a hall-of-fame entry, a lineage ancestor, a preset, the creature lab's test run) replaces the live view until
// BACK TO LIVE.
import { DEFAULT_WORLD, DT, GA_DEFAULTS, HOLE, NODE_MAX, NODE_MIN, REST_MAX, REST_MIN, SPRING_MAX, TERRAINS, TWO_PI, clamp, decode, encode, groundAt, q4, repair, simulate, wsin } from './walkers-core.js';
import { banner, pop, shake } from './fx.js';

const $ = (id) => document.getElementById(id);
const REDUCED = matchMedia('(prefers-reduced-motion: reduce)').matches;
const SPEEDS = [0.25, 0.5, 1, 2, 4, 8, Infinity]; // generations per second; the last is "as fast as possible"
// Node glow colours (RGB triplets for the sprite gradients) and the matching CSS colours for banners and handles.
const PALETTE = { champ: '230,253,255', live: '159,247,255', ghost: '179,136,255', user: '255,230,0' };
const HOF_MAX = 12;
const GHOSTS = 3;

// ---------------------------------------------------------------- state
const S = {
  world: { ...DEFAULT_WORLD },
  popSize: GA_DEFAULTS.popSize,
  mutationRate: GA_DEFAULTS.mutationRate,
  seed: 7,
  running: true,
  genRate: 1,
  worker: null,
  ready: false,
  busy: false,
  lastGenDone: 0,
  pumpTimer: 0,
  gen: 0,
  best: -Infinity,
  recordBest: -Infinity,
  lastRecordBanner: 0,
  lastSpeciesFlash: 0,
  history: [], // {gen, best, mean, median, species, sizes: [{id,size}]}
  records: new Map(), // id -> lineage record {id, gen, parents, fitness, distance, code, species, user}
  champ: null, // the newest champion: {id, gen, fitness, distance, code}
  ghosts: [], // older champions' runs
  hof: [], // milestone champions {gen, fitness, code, frames, sampleEvery, N, world}
  hofIndex: 0,
  pendingLive: null,
  lastLiveSwap: 0,
  user: { pending: [], best: null },
  view: { mode: 'live', runs: [], t: 0, speed: 1, camX: undefined, label: '', world: { ...DEFAULT_WORLD }, playing: true },
  presets: [],
  editor: { nodes: [], springs: [], tool: 'node', type: 'muscle', drag: null, sel: -1, mouse: null, tested: null },
  ed: null,
  log: [],
};

// ---------------------------------------------------------------- small helpers
// Sizes a canvas to its box at the device pixel ratio (capped at maxDpr) and resets its transform for the frame.
function fit(c, maxDpr = 2) {
  const r = c.getBoundingClientRect();
  const dpr = Math.min(maxDpr, window.devicePixelRatio || 1);
  const w = Math.max(10, Math.round(r.width));
  const h = Math.max(10, Math.round(r.height));
  if (c.width !== Math.round(w * dpr) || c.height !== Math.round(h * dpr)) {
    c.width = Math.round(w * dpr);
    c.height = Math.round(h * dpr);
  }
  const ctx = c.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return { ctx, w, h };
}

function hueOf(id) {
  return `hsl(${(id * 47) % 360} 90% 62%)`;
}

// Muscle colour from tension in [-1, 1]: stretched past its rest length (pulling) fades from a neutral blue to pink,
// squeezed short of it (pushing) fades to cyan.
function tensionColor(t, a) {
  const k = Math.min(1, Math.abs(t));
  const base = [120, 200, 255];
  const target = t >= 0 ? [255, 0, 160] : [0, 245, 255];
  const c = base.map((v, i) => Math.round(v + (target[i] - v) * k));
  return `rgba(${c[0]},${c[1]},${c[2]},${a})`;
}

function say(text, cls = '') {
  S.log.unshift({ text, cls });
  S.log = S.log.slice(0, 60);
  const box = $('ew-log');
  if (!box) return;
  box.innerHTML = S.log.map((l) => `<div class="${l.cls}">${l.text}</div>`).join('');
}

// Metres to two decimals; values this close to zero print as 0 (never "-0.00 m").
const fmtM = (v) => (Number.isFinite(v) ? `${(Math.abs(v) < 0.005 ? 0 : v).toFixed(2)} m` : '—');

function genomeOf(code) {
  return decode(code);
}

function bodyCounts(g) {
  const muscles = g.springs.reduce((a, s) => a + (s.muscle ? 1 : 0), 0);
  return `${g.nodes.length} nodes · ${g.springs.length} links · ${muscles} muscles`;
}

// ---------------------------------------------------------------- sprites and skyline (cached)
const sprites = {};
// A soft radial glow drawn once per kind. drawImage on a cached sprite is far cheaper than shadowBlur per node.
function glowSprite(key, rgb) {
  if (sprites[key]) return sprites[key];
  const c = document.createElement('canvas');
  c.width = c.height = 64;
  const g = c.getContext('2d');
  const grad = g.createRadialGradient(32, 32, 0, 32, 32, 32);
  grad.addColorStop(0, `rgba(${rgb},1)`);
  grad.addColorStop(0.3, `rgba(${rgb},0.85)`);
  grad.addColorStop(0.55, `rgba(${rgb},0.25)`);
  grad.addColorStop(1, `rgba(${rgb},0)`);
  g.fillStyle = grad;
  g.fillRect(0, 0, 64, 64);
  sprites[key] = c;
  return c;
}

// The backdrop (sky, sun and a skyline twice as wide as the arena) is painted once per size into an offscreen canvas.
// Each frame then copies one arena-wide slice of it, offset for parallax, which is a single drawImage call.
let backdrop = null;
let backdropKey = '';
function paintBackdrop(w, h) {
  backdrop = document.createElement('canvas');
  backdrop.width = Math.round(w * 2);
  backdrop.height = h;
  const g = backdrop.getContext('2d');
  const grad = g.createLinearGradient(0, 0, 0, h);
  grad.addColorStop(0, '#02030c');
  grad.addColorStop(0.55, '#0b0424');
  grad.addColorStop(1, '#020610');
  g.fillStyle = grad;
  g.fillRect(0, 0, backdrop.width, h);
  g.fillStyle = 'rgba(255,0,160,0.16)';
  g.beginPath();
  g.arc(w * 0.78, h * 0.3, Math.min(w, h) * 0.2, 0, TWO_PI);
  g.fill();
  // Building silhouettes with neon windows. A fixed seed keeps the city the same on every paint.
  let seed = 99;
  const rnd = () => {
    seed = (seed * 16807) % 2147483647;
    return seed / 2147483647;
  };
  let x = 0;
  while (x < backdrop.width) {
    const bw = 24 + rnd() * 46;
    const bh = h * (0.18 + rnd() * 0.3);
    g.fillStyle = '#080a1f';
    g.fillRect(x, h * 0.62 - bh, bw, bh + h);
    for (let wy = h * 0.62 - bh + 6; wy < h * 0.62 - 4; wy += 9) {
      for (let wx = x + 4; wx < x + bw - 4; wx += 8) {
        if (rnd() < 0.18) {
          g.fillStyle = rnd() < 0.5 ? 'rgba(0,245,255,0.45)' : 'rgba(255,0,160,0.4)';
          g.fillRect(wx, wy, 3, 4);
        }
      }
    }
    x += bw + 2;
  }
}

function drawSky(ctx, w, h, camX, ppm) {
  const key = `${w}x${h}`;
  if (key !== backdropKey) {
    backdropKey = key;
    paintBackdrop(w, h);
  }
  // Slide the skyline at about a fifth of the floor's speed; the modulo keeps the offset within one arena width.
  const off = ((camX * ppm * 0.22) % w + w) % w;
  ctx.drawImage(backdrop, off, 0, w, h, 0, 0, w, h);
}

// ---------------------------------------------------------------- run playback
// A run is {frames, N, sampleEvery, frameCount, genome, kind, label}. Frame k is the body at time k * sampleEvery * DT,
// so at 60 Hz (sampleEvery 4) playback time t maps to frame t * 60.
function runPositions(r, t, out) {
  const N = r.N;
  const rate = r.sampleEvery * DT;
  let fi = t / rate;
  const last = r.frameCount - 1;
  if (fi < 0) fi = 0;
  if (fi > last) fi = last;
  const i0 = Math.floor(fi);
  const i1 = Math.min(i0 + 1, last);
  const fr = fi - i0;
  const f = r.frames;
  const b0 = i0 * N * 2;
  const b1 = i1 * N * 2;
  for (let k = 0; k < N * 2; k++) out[k] = f[b0 + k] * (1 - fr) + f[b1 + k] * fr;
  return out;
}

function comOf(r, t) {
  const P = runPositions(r, t, new Float64Array(r.N * 2));
  let x = 0;
  for (let i = 0; i < r.N; i++) x += P[2 * i];
  return x / r.N;
}

const POS = new Map();
function scratch(n) {
  if (!POS.has(n)) POS.set(n, new Float64Array(n));
  return POS.get(n);
}

function drawCreature(ctx, r, t, sx, sy, ppm, alpha) {
  const P = runPositions(r, t, scratch(r.N * 2));
  const g = r.genome;
  const kind = r.kind;
  // springs first, then nodes on top
  for (const s of g.springs) {
    const ax = sx(P[2 * s.a]);
    const ay = sy(P[2 * s.a + 1]);
    const bx = sx(P[2 * s.b]);
    const by = sy(P[2 * s.b + 1]);
    if (s.muscle) {
      const dx = P[2 * s.b] - P[2 * s.a];
      const dy = P[2 * s.b + 1] - P[2 * s.a + 1];
      const L = Math.sqrt(dx * dx + dy * dy);
      const restNow = s.rest * (1 + s.amp * wsin(TWO_PI * s.freq * t + s.phase));
      // tension in [-1, 1]: how far the muscle is stretched past (or squeezed short of) its current rest length
      const tension = clamp((L - restNow) / (0.12 * s.rest), -1, 1);
      const lw = 1.2 + 3 * Math.abs(tension);
      if (kind === 'champ') {
        // The champion's muscles get a wide, faint glow pass; the others are drawn plain to keep frames cheap.
        ctx.strokeStyle = tensionColor(tension, 0.22 * alpha);
        ctx.lineWidth = lw + 4;
        ctx.beginPath();
        ctx.moveTo(ax, ay);
        ctx.lineTo(bx, by);
        ctx.stroke();
      }
      ctx.strokeStyle = tensionColor(tension, 0.95 * alpha);
      ctx.lineWidth = lw;
      ctx.beginPath();
      ctx.moveTo(ax, ay);
      ctx.lineTo(bx, by);
      ctx.stroke();
    } else {
      ctx.strokeStyle = kind === 'ghost' ? `rgba(179,136,255,${0.5 * alpha})` : `rgba(0,245,255,${0.42 * alpha})`;
      ctx.lineWidth = 1.2;
      ctx.beginPath();
      ctx.moveTo(ax, ay);
      ctx.lineTo(bx, by);
      ctx.stroke();
    }
  }
  const sprite = glowSprite(kind, PALETTE[kind] || PALETTE.live);
  const size = Math.max(10, 0.2 * ppm);
  ctx.globalAlpha = alpha;
  for (let i = 0; i < r.N; i++) {
    ctx.drawImage(sprite, sx(P[2 * i]) - size / 2, sy(P[2 * i + 1]) - size / 2, size, size);
  }
  ctx.globalAlpha = 1;
  return P;
}

// ---------------------------------------------------------------- particles (sparks on footfalls)
const sparks = [];
function spawnSparks(x, y, n) {
  for (let k = 0; k < n && sparks.length < 260; k++) {
    sparks.push({ x, y, vx: (Math.random() - 0.5) * 2.4, vy: Math.random() * 2.2 + 0.4, life: 1 });
  }
}

function stepSparks(ctx, dt) {
  ctx.save();
  ctx.globalCompositeOperation = 'lighter';
  for (let i = sparks.length - 1; i >= 0; i--) {
    const p = sparks[i];
    p.life -= dt * 2.4;
    if (p.life <= 0) {
      sparks.splice(i, 1);
      continue;
    }
    p.x += p.vx * 60 * dt;
    p.y += p.vy * 60 * dt;
    p.vy += 6 * dt * 60;
    ctx.fillStyle = `rgba(255,230,0,${p.life})`;
    ctx.fillRect(p.x, p.y, 2, 2);
  }
  ctx.restore();
}

// ---------------------------------------------------------------- arena
const A = { canvas: null, last: 0, ctxInfo: null };

function drawArena(now) {
  const dt = Math.min(0.05, Math.max(0, (now - A.last) / 1000));
  A.last = now;
  const { ctx, w, h } = fit(A.canvas, 1.5);
  const v = S.view;
  if (v.playing) v.t += dt * v.speed;
  const duration = v.world.duration;
  if (v.t >= duration) v.t -= duration;
  if (v.t < 0) v.t = 0;

  // The camera follows the lead run (the champion or the first run) at a lag, so the body stays in the left third.
  const lead = v.runs.find((r) => r.kind === 'champ' || r.kind === 'user') || v.runs[0];
  const target = lead ? comOf(lead, v.t) : 2;
  if (v.camX === undefined) v.camX = target;
  v.camX += (target - v.camX) * (1 - Math.exp(-dt * 5));
  const ppm = h / 4.4; // the arena shows about 4.4 m of height, so a 1 m creature reads at a glance
  const originX = w * 0.36;
  const groundY = h * 0.8;
  const sx = (x) => originX + (x - v.camX) * ppm;
  const sy = (y) => groundY - y * ppm;

  drawSky(ctx, w, h, v.camX, ppm);
  drawFloor(ctx, w, h, v.world.terrain, v.camX, ppm, sx, sy, originX);

  // Ghosts behind, then the rest of the live population, then the champion, then the user's creature.
  for (const r of v.runs) if (r.kind === 'ghost') drawCreature(ctx, r, v.t, sx, sy, ppm, r.alpha);
  for (const r of v.runs) if (r.kind === 'live') drawCreature(ctx, r, v.t, sx, sy, ppm, r.alpha);
  for (const r of v.runs) {
    if (r.kind === 'champ' || r.kind === 'user') {
      const P = drawCreature(ctx, r, v.t, sx, sy, ppm, 1);
      if (r.kind === 'champ' && !REDUCED && v.playing) footfalls(r, P, sx, sy, groundY, ppm);
    }
  }
  if (!REDUCED) stepSparks(ctx, dt);
  drawMarkers(ctx, w, h, v.camX, ppm, sx, sy);
  if (lead) updateHud(lead, v.t);
  requestAnimationFrame(drawArena);
}

function drawFloor(ctx, w, h, terrain, camX, ppm, sx, sy, originX) {
  const xMin = camX - originX / ppm - 0.2;
  const xMax = camX + (w - originX) / ppm + 0.2;
  const step = 2 / ppm;
  ctx.beginPath();
  let pen = false;
  let first = null;
  let last = null;
  for (let x = xMin; x <= xMax; x += step) {
    const hgt = groundAt(terrain, x).h;
    if (hgt <= HOLE + 1) {
      pen = false;
      continue;
    }
    const px = sx(x);
    const py = sy(hgt);
    if (!pen) {
      ctx.moveTo(px, py);
      pen = true;
      if (first === null) first = px;
    } else ctx.lineTo(px, py);
    last = px;
  }
  // Fill: the ground down to the bottom of the arena, in a soft purple gradient.
  ctx.save();
  ctx.lineTo(last ?? w, h);
  ctx.lineTo(first ?? 0, h);
  ctx.closePath();
  const grad = ctx.createLinearGradient(0, sy(0), 0, h);
  grad.addColorStop(0, 'rgba(155,0,255,0.35)');
  grad.addColorStop(1, 'rgba(2,6,16,0.95)');
  ctx.fillStyle = grad;
  ctx.fill();
  ctx.restore();
  // Neon floor line: a wide faint pass for glow, then a sharp one.
  ctx.beginPath();
  pen = false;
  for (let x = xMin; x <= xMax; x += step) {
    const hgt = groundAt(terrain, x).h;
    if (hgt <= HOLE + 1) {
      pen = false;
      continue;
    }
    if (!pen) {
      ctx.moveTo(sx(x), sy(hgt));
      pen = true;
    } else ctx.lineTo(sx(x), sy(hgt));
  }
  ctx.save();
  ctx.lineJoin = 'round';
  ctx.strokeStyle = 'rgba(0,245,255,0.25)';
  ctx.lineWidth = 8;
  ctx.stroke();
  ctx.strokeStyle = '#00f5ff';
  ctx.lineWidth = 2;
  ctx.stroke();
  ctx.restore();
  // Perspective-free grid under the floor: vertical lines each metre, scrolling with the camera.
  ctx.save();
  ctx.strokeStyle = 'rgba(0,245,255,0.09)';
  ctx.lineWidth = 1;
  ctx.beginPath();
  const g0 = Math.floor(xMin);
  for (let m = g0; m <= xMax; m++) {
    const px = sx(m);
    ctx.moveTo(px, sy(0));
    ctx.lineTo(px, h);
  }
  for (let d = 1; d < 6; d++) {
    const py = sy(0) + (h - sy(0)) * (d / 6);
    ctx.moveTo(0, py);
    ctx.lineTo(w, py);
  }
  ctx.stroke();
  ctx.restore();
}

function drawMarkers(ctx, w, h, camX, ppm, sx, sy) {
  ctx.save();
  ctx.font = '11px "JetBrains Mono", monospace';
  ctx.fillStyle = 'rgba(74,122,155,0.95)';
  ctx.strokeStyle = 'rgba(255,230,0,0.6)';
  const first = Math.ceil((camX - (w * 0.36) / ppm) / 5) * 5;
  for (let m = first; m <= camX + (w * 0.64) / ppm; m += 5) {
    if (m < 0) continue;
    const px = sx(m);
    ctx.beginPath();
    ctx.moveTo(px, sy(0));
    ctx.lineTo(px, sy(0) + 10);
    ctx.stroke();
    ctx.fillText(`${m} m`, px + 3, sy(0) + 22);
  }
  ctx.restore();
}

function footfalls(r, P, sx, sy, groundY, ppm) {
  // A node that was above the floor last frame and is on it now makes a spark burst. Last heights are kept per run.
  if (!r.lastY || r.lastY.length !== r.N) r.lastY = new Float64Array(r.N).fill(10);
  for (let i = 0; i < r.N; i++) {
    const x = P[2 * i];
    const y = P[2 * i + 1];
    const hgt = groundAt(S.view.world.terrain, x).h;
    if (r.lastY[i] - hgt > 0.05 && y - hgt <= 0.02 && y - hgt >= -0.02 && Math.random() < 0.7) spawnSparks(sx(x), sy(hgt), 4);
    r.lastY[i] = y;
  }
}

// The HUD text changes a few times a second at most, so it is written only when it differs (no layout churn).
const hudLast = { mode: '', label: '', dist: '', clock: '' };
function updateHud(lead, t) {
  let d = comOf(lead, t) - 2; // distance travelled from the spawn point (x = 2 m)
  if (Math.abs(d) < 0.005) d = 0; // no "-0.00 m" when the body has not moved
  const mode = S.view.mode === 'live' ? 'LIVE' : 'REPLAY';
  const vals = { mode, label: S.view.label, dist: `${d.toFixed(2)} m`, clock: `${t.toFixed(1)} / ${S.view.world.duration} s` };
  const ids = { mode: 'ew-mode', label: 'ew-run-label', dist: 'ew-run-dist', clock: 'ew-clock' };
  for (const k of Object.keys(vals)) {
    if (hudLast[k] !== vals[k]) {
      hudLast[k] = vals[k];
      $(ids[k]).textContent = vals[k];
    }
  }
  $('btn-live').hidden = S.view.mode === 'live';
}

// ---------------------------------------------------------------- live runs and loads
function setView(mode, runs, label, world) {
  S.view.mode = mode;
  S.view.runs = runs;
  S.view.t = 0;
  S.view.camX = undefined;
  S.view.label = label;
  S.view.world = world;
  S.view.speed = mode === 'replay' && S.view.slowmo ? 0.25 : S.view.speed;
  $('ew-speed-val').textContent = `${S.view.speed}×`;
  $('replay-speed').value = String(Math.log2(S.view.speed * 4)); // 0.25×→0, 1×→2, 2×→3
}

function runFrom(rec, frames, kind, alpha = 1) {
  return { frames, N: frames.N, sampleEvery: frames.sampleEvery, frameCount: frames.frameCount, genome: genomeOf(rec.code), kind, alpha, label: '' };
}

function liveRunsFor(top) {
  const runs = [];
  for (const g of S.ghosts) runs.push({ ...g, kind: 'ghost', alpha: 0.2 });
  top.forEach((t, i) => {
    const rec = S.records.get(t.id);
    if (!rec) return;
    runs.push({
      frames: t.frames, N: t.N, sampleEvery: t.sampleEvery, frameCount: t.frameCount,
      genome: genomeOf(rec.code), kind: i === 0 ? 'champ' : 'live', alpha: i === 0 ? 1 : 0.42, label: '',
    });
  });
  return runs;
}

function applyLive(msg) {
  const top = msg.top;
  if (!top.length) return;
  const champ = top[0];
  const rec = S.records.get(champ.id);
  if (S.view.mode === 'live') {
    setView('live', liveRunsFor(top), `GEN ${msg.summary.gen} CHAMPION`, S.world);
  }
  S.champ = { id: champ.id, gen: msg.summary.gen, fitness: champ.fitness, code: rec ? rec.code : null };
  // Ghosts: the champion we just left behind joins the racing ghosts (oldest falls off).
  if (rec) {
    S.ghosts.push({ frames: champ.frames, N: champ.N, sampleEvery: champ.sampleEvery, frameCount: champ.frameCount, genome: genomeOf(rec.code), label: '' });
    S.ghosts = S.ghosts.slice(-GHOSTS);
  }
  if (S.view.mode === 'live' && rec) renderLineage();
}

// ---------------------------------------------------------------- the worker
function startRun(seed) {
  S.seed = seed;
  S.history = [];
  S.records = new Map();
  S.champ = null;
  S.ghosts = [];
  S.hof = [];
  S.hofIndex = 0;
  S.gen = 0;
  S.best = -Infinity;
  S.recordBest = -Infinity;
  S.user = { awaiting: 0, best: null };
  S.pendingLive = null;
  S.lastTop = null;
  S.busy = false;
  S.ready = false;
  if (S.worker) S.worker.terminate();
  S.worker = new Worker(new URL('./walkers-worker.js', import.meta.url), { type: 'module' });
  S.worker.onmessage = onWorker;
  S.worker.onerror = (e) => {
    // The worker script stopped, so its population is gone and the run cannot resume: NEW RUN starts over.
    S.ready = false;
    stopRun(`worker error: ${e.message || 'unknown'}. Press NEW RUN to start again.`);
  };
  S.worker.postMessage({ cmd: 'init', seed, popSize: S.popSize, mutationRate: S.mutationRate, world: S.world });
  $('ew-seed').value = String(seed);
  renderStats();
  drawCharts();
  renderHof();
  renderLineage();
  say(`new run: seed ${seed}, population ${S.popSize}, terrain ${S.world.terrain}`);
}

function onWorker(e) {
  const m = e.data;
  if (m.type === 'ready') {
    S.ready = true;
    pump();
    return;
  }
  if (m.type === 'error') {
    stopRun(`worker: ${m.message}. Press PLAY to try again.`);
    return;
  }
  if (m.type === 'gen') {
    S.busy = false;
    S.lastGenDone = performance.now();
    for (const r of m.records) S.records.set(r.id, r);
    handleGen(m);
    pump();
  }
}

// Pause after a failed generation, so the run does not spin on the same error. PLAY or STEP asks for another try.
function stopRun(text) {
  S.running = false;
  S.busy = false;
  $('btn-toggle').querySelector('.btn-txt').textContent = '▶ PLAY';
  say(text, 'err');
}

function pump() {
  if (!S.worker || !S.ready || S.busy || !S.running) return;
  const gap = S.genRate === Infinity ? 0 : 1000 / S.genRate;
  const wait = S.lastGenDone + gap - performance.now();
  if (wait > 0) {
    clearTimeout(S.pumpTimer);
    S.pumpTimer = setTimeout(pump, wait);
    return;
  }
  S.busy = true;
  S.worker.postMessage({ cmd: 'step', topK: 4 });
}

function handleGen(m) {
  const sm = m.summary;
  S.gen = sm.gen;
  S.best = sm.best;
  const rec = {
    gen: sm.gen,
    best: sm.best,
    mean: sm.mean,
    median: sm.median,
    species: sm.species,
    sizes: sm.speciesSizes,
    newSpecies: sm.newSpecies,
    exploded: sm.exploded,
  };
  S.history.push(rec);
  S.history = S.history.slice(-400);

  // NEW RECORD: a milestone when the best distance improves by 2 cm or more (the first generation always counts).
  const isRecord = sm.best > S.recordBest + 0.02;
  if (isRecord) {
    const first = S.recordBest === -Infinity;
    S.recordBest = sm.best;
    const code = S.records.get(sm.champion.id)?.code;
    if (code) {
      S.hof.push({ gen: sm.gen, fitness: sm.best, code, frames: m.top[0].frames, N: m.top[0].N, sampleEvery: m.top[0].sampleEvery, frameCount: m.top[0].frameCount, world: { ...S.world } });
      S.hof = S.hof.slice(-HOF_MAX);
      S.hofIndex = S.hof.length - 1;
    }
    if (!first) {
      const now = performance.now();
      if (now - S.lastRecordBanner > 2500) {
        S.lastRecordBanner = now;
        banner('NEW RECORD', `${sm.best.toFixed(2)} m · generation ${sm.gen}`, '#00ff88');
      }
      say(`gen ${sm.gen}: NEW RECORD ${fmtM(sm.best)}`, 'rec');
    } else say(`gen ${sm.gen}: first champion ${fmtM(sm.best)}`);
  } else {
    say(`gen ${sm.gen}: best ${fmtM(sm.best)} · mean ${sm.mean.toFixed(2)} m · ${sm.species} species`);
  }

  // A new species: the arena flashes, and it is logged.
  if (sm.newSpecies && sm.newSpecies.length && performance.now() - S.lastSpeciesFlash > 4000) {
    S.lastSpeciesFlash = performance.now();
    glitchFlash();
    say(`gen ${sm.gen}: new species #${sm.newSpecies.join(', #')}`, 'spc');
  }

  // The user's creatures that were dropped in: report how each did against this generation's champion. Each drop
  // is one entry in sm.user, and `awaiting` counts the drops not yet scored.
  if (sm.user && sm.user.length && S.user.awaiting > 0) {
    for (const u of sm.user.slice(0, S.user.awaiting)) {
      const beat = u.fitness > sm.best; // a tie (for example a copy of the champion) is not a win
      S.user.best = { gen: sm.gen, fitness: u.fitness, champion: sm.best, beat };
      if (beat) {
        banner('YOUR DESIGN WINS', `${u.fitness.toFixed(2)} m beats evolution's ${sm.best.toFixed(2)} m`, '#ffe600');
        pop($('ew-frame'), 'CHALLENGE WON', '#ffe600');
      }
      say(`gen ${sm.gen}: your creature ${fmtM(u.fitness)} (champion ${fmtM(sm.best)})`, beat ? 'rec' : '');
    }
    S.user.awaiting = 0;
  }

  // The arena swaps to a new champion at most every 450 ms, so fast-forward speeds do not strobe.
  S.pendingLive = m;
  S.lastTop = m.top;
  renderStats();
  drawCharts();
  renderHof();
  renderChallenger();
  renderLineage();
}

function glitchFlash() {
  const f = $('ew-frame');
  f.classList.remove('ew-glitch');
  void f.offsetWidth;
  f.classList.add('ew-glitch');
  shake(f, 'small');
}

function renderStats() {
  $('st-gen').textContent = String(S.gen);
  $('st-best').textContent = Number.isFinite(S.best) ? S.best.toFixed(2) : '—';
  const last = S.history[S.history.length - 1];
  $('st-mean').textContent = last ? last.mean.toFixed(2) : '—';
  $('st-species').textContent = last ? String(last.species) : '—';
  $('st-record').textContent = Number.isFinite(S.recordBest) ? S.recordBest.toFixed(2) : '—';
}

function swapPendingLive(now) {
  if (!S.pendingLive) return;
  // The preset intro plays until the first generation arrives, then the live view takes over.
  if (S.presetIntro) {
    S.presetIntro = false;
    S.view.mode = 'live';
  }
  if (S.view.mode !== 'live') return;
  if (now - S.lastLiveSwap < 450) return;
  const m = S.pendingLive;
  S.pendingLive = null;
  S.lastLiveSwap = now;
  applyLive(m);
}

// ---------------------------------------------------------------- charts
function drawCharts() {
  drawFitness();
  drawSpecies();
  const legend = $('chart-legend');
  if (legend) {
    const last = S.history[S.history.length - 1];
    legend.textContent = last ? `GEN ${last.gen} · BEST ${last.best.toFixed(2)} · MEAN ${last.mean.toFixed(2)} · MEDIAN ${last.median.toFixed(2)} m` : 'waiting for generation 1';
  }
}

function drawFitness() {
  const c = $('ch-fitness');
  if (!c) return;
  const { ctx, w, h } = fit(c);
  ctx.clearRect(0, 0, w, h);
  const hist = S.history.slice(-200);
  const pad = { l: 40, r: 10, t: 10, b: 20 };
  if (!hist.length) {
    ctx.fillStyle = '#4a7a9b';
    ctx.font = '12px "JetBrains Mono", monospace';
    ctx.fillText('waiting for generation 1', pad.l, h / 2);
    return;
  }
  let lo = 0;
  let hi = 0.5;
  for (const r of hist) {
    hi = Math.max(hi, r.best);
    lo = Math.min(lo, r.median);
  }
  hi *= 1.1;
  const X = (i) => pad.l + (hist.length <= 1 ? 0 : (i / (hist.length - 1)) * (w - pad.l - pad.r));
  const Y = (v) => pad.t + (1 - (v - lo) / (hi - lo || 1)) * (h - pad.t - pad.b);
  ctx.strokeStyle = 'rgba(0,245,255,0.12)';
  ctx.fillStyle = '#4a7a9b';
  ctx.font = '10px "JetBrains Mono", monospace';
  for (let k = 0; k <= 4; k++) {
    const v = lo + ((hi - lo) * k) / 4;
    const y = Y(v);
    ctx.beginPath();
    ctx.moveTo(pad.l, y);
    ctx.lineTo(w - pad.r, y);
    ctx.stroke();
    ctx.fillText(`${v.toFixed(1)}`, 4, y + 3);
  }
  const line = (key, color, width) => {
    ctx.beginPath();
    hist.forEach((r, i) => (i ? ctx.lineTo(X(i), Y(r[key])) : ctx.moveTo(X(i), Y(r[key]))));
    ctx.strokeStyle = color;
    ctx.lineWidth = width;
    ctx.shadowColor = color;
    ctx.shadowBlur = 8;
    ctx.stroke();
    ctx.shadowBlur = 0;
  };
  line('median', '#ff00a0', 1.5);
  line('mean', '#00f5ff', 1.5);
  line('best', '#00ff88', 2.2);
}

function drawSpecies() {
  const c = $('ch-species');
  if (!c) return;
  const { ctx, w, h } = fit(c);
  ctx.clearRect(0, 0, w, h);
  const hist = S.history.slice(-120);
  const pad = { l: 40, r: 10, t: 10, b: 20 };
  if (!hist.length) return;
  const cw = (w - pad.l - pad.r) / Math.max(hist.length, 40);
  let maxPop = 1;
  for (const r of hist) maxPop = Math.max(maxPop, r.sizes.reduce((a, s) => a + s.size, 0));
  const Y = (v) => h - pad.b - (v / maxPop) * (h - pad.t - pad.b);
  hist.forEach((r, col) => {
    let acc = 0;
    const x = pad.l + col * cw;
    const parts = r.sizes.slice().sort((a, b) => a.id - b.id);
    for (const s of parts) {
      const y0 = Y(acc);
      acc += s.size;
      const y1 = Y(acc);
      ctx.fillStyle = hueOf(s.id);
      ctx.globalAlpha = 0.85;
      ctx.fillRect(x, y1, Math.max(1, cw - 0.5), y0 - y1);
    }
    ctx.globalAlpha = 1;
  });
  ctx.fillStyle = '#4a7a9b';
  ctx.font = '10px "JetBrains Mono", monospace';
  ctx.fillText(`${maxPop}`, 4, pad.t + 8);
  ctx.fillText('0', 4, h - pad.b);
}

// ---------------------------------------------------------------- hall of fame and lineage
function renderHof() {
  const box = $('hof-title');
  if (!box) return;
  const n = S.hof.length;
  if (!n) {
    box.textContent = 'records appear here';
    $('hof-stats').textContent = '';
    $('hof-count').textContent = '0 / 0';
    return;
  }
  S.hofIndex = clamp(S.hofIndex, 0, n - 1);
  const e = S.hof[S.hofIndex];
  box.textContent = `GEN ${e.gen} · ${fmtM(e.fitness)}`;
  $('hof-stats').textContent = bodyCounts(genomeOf(e.code));
  $('hof-count').textContent = `${S.hofIndex + 1} / ${n}`;
}

function renderLineage() {
  const box = $('lineage');
  if (!box) return;
  if (!S.champ) {
    box.innerHTML = '<div class="field-hint">the champion and its ancestors appear after generation 1</div>';
    return;
  }
  const rows = [];
  let rec = S.records.get(S.champ.id);
  for (let depth = 0; rec && depth < 6; depth++) {
    const g = genomeOf(rec.code);
    const kids = depth === 0 ? 'CHAMPION' : depth === 1 ? 'PARENT' : `ANCESTOR ${depth}`;
    const parents = rec.parents && rec.parents.length ? (rec.parents.length === 2 ? 'crossover' : 'clone') : 'seed';
    rows.push(
      `<button class="lin-row" type="button" data-id="${rec.id}" style="--indent:${depth * 14}px">` +
      `<span class="lin-tag">${kids}</span>` +
      `<span class="lin-main">GEN ${rec.gen} · ${fmtM(rec.fitness)}</span>` +
      `<span class="lin-sub">${bodyCounts(g)} · ${parents}${rec.user ? ' · USER' : ''}</span></button>`
    );
    rec = rec.parents && rec.parents[0] ? S.records.get(rec.parents[0]) : null;
  }
  box.innerHTML = rows.join('');
}

function renderChallenger() {
  const box = $('challenger');
  if (!box) return;
  const b = S.user.best;
  if (!b) {
    box.textContent = 'drop a creature from the lab into the population to challenge the champion.';
    return;
  }
  box.innerHTML =
    `<div class="chal-line ${b.beat ? 'win' : ''}">${b.beat ? 'YOUR DESIGN WINS' : 'EVOLUTION WINS'}</div>` +
    `<div class="field-hint">gen ${b.gen}: yours ${fmtM(b.fitness)} · champion ${fmtM(b.champion)}</div>`;
}

// ---------------------------------------------------------------- replays from the panels
function replayRecord(rec, label, opts = {}) {
  const g = genomeOf(rec.code);
  const world = opts.world || S.world;
  const r = simulate(g, world, { sampleEvery: 4 });
  const run = { frames: r.frames, N: r.N, sampleEvery: 4, frameCount: r.frameCount, genome: g, kind: opts.kind || 'champ', alpha: 1, label };
  S.view.slowmo = !!opts.slowmo;
  setView('replay', [run], label, world);
  if (opts.slowmo) S.view.speed = 0.25;
  $('ew-speed-val').textContent = `${S.view.speed}×`;
  $('btn-live').hidden = false;
  return run;
}

function replayHof() {
  const e = S.hof[S.hofIndex];
  if (!e) return;
  const g = genomeOf(e.code);
  const run = { frames: e.frames, N: e.N, sampleEvery: e.sampleEvery, frameCount: e.frameCount, genome: g, kind: 'champ', alpha: 1, label: '' };
  S.view.slowmo = false;
  setView('replay', [run], `HALL OF FAME · GEN ${e.gen}`, e.world);
  $('btn-live').hidden = false;
}

function backToLive() {
  S.view.slowmo = false;
  S.presetIntro = false;
  // Show the newest generation again, without pushing its champion into the ghosts a second time.
  setView('live', S.lastTop ? liveRunsFor(S.lastTop) : [], S.lastTop ? `GEN ${S.gen} CHAMPION` : '', S.world);
  S.pendingLive = null;
  $('btn-live').hidden = true;
  renderLineage();
}

// ---------------------------------------------------------------- creature lab (editor)
const ED = { canvas: null };

function edGenome() {
  const e = S.editor;
  return {
    nodes: e.nodes.map((n) => ({ x: n.x, y: n.y })),
    springs: e.springs.map((s) => ({ ...s })),
  };
}

function edWorldToScreen(x, y, w, h) {
  const ppm = Math.min(w / 3.6, h / 2.6);
  return { px: w / 2 + x * ppm, py: h * 0.8 - y * ppm, ppm };
}

function edScreenToWorld(px, py, w, h) {
  const ppm = Math.min(w / 3.6, h / 2.6);
  return { x: q4(clamp((px - w / 2) / ppm, -1.5, 1.5)), y: q4(clamp((h * 0.8 - py) / ppm, 0, 1.5)) };
}

function edDraw() {
  const { ctx, w, h } = fit(ED.canvas);
  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = 'rgba(0,0,0,0.35)';
  ctx.fillRect(0, 0, w, h);
  // grid dots and baseline
  ctx.fillStyle = 'rgba(0,245,255,0.12)';
  for (let gx = 0; gx < w; gx += 20) for (let gy = 0; gy < h; gy += 20) ctx.fillRect(gx, gy, 1, 1);
  const base = edWorldToScreen(0, 0, w, h).py;
  ctx.strokeStyle = 'rgba(0,255,136,0.5)';
  ctx.beginPath();
  ctx.moveTo(0, base);
  ctx.lineTo(w, base);
  ctx.stroke();
  const e = S.editor;
  // springs: muscles pink and thick, plain springs cyan and thin
  e.springs.forEach((s, i) => {
    const A1 = edWorldToScreen(e.nodes[s.a].x, e.nodes[s.a].y, w, h);
    const B1 = edWorldToScreen(e.nodes[s.b].x, e.nodes[s.b].y, w, h);
    ctx.strokeStyle = s.muscle ? 'rgba(255,0,160,0.95)' : 'rgba(0,245,255,0.6)';
    ctx.lineWidth = s.muscle ? 4 : 1.6;
    ctx.beginPath();
    ctx.moveTo(A1.px, A1.py);
    ctx.lineTo(B1.px, B1.py);
    ctx.stroke();
  });
  if (e.drag && e.drag.kind === 'link' && e.drag.to) {
    const A1 = edWorldToScreen(e.nodes[e.drag.from].x, e.nodes[e.drag.from].y, w, h);
    ctx.setLineDash([4, 4]);
    ctx.strokeStyle = e.type === 'muscle' ? 'rgba(255,0,160,0.9)' : 'rgba(0,245,255,0.9)';
    ctx.beginPath();
    ctx.moveTo(A1.px, A1.py);
    ctx.lineTo(e.drag.to.px, e.drag.to.py);
    ctx.stroke();
    ctx.setLineDash([]);
  }
  e.nodes.forEach((n, i) => {
    const p = edWorldToScreen(n.x, n.y, w, h);
    ctx.beginPath();
    ctx.arc(p.px, p.py, i === e.sel ? 7 : 5, 0, TWO_PI);
    ctx.fillStyle = i === e.sel ? '#ffe600' : '#e6fdff';
    ctx.shadowColor = '#00f5ff';
    ctx.shadowBlur = 10;
    ctx.fill();
    ctx.shadowBlur = 0;
    ctx.fillStyle = '#4a7a9b';
    ctx.font = '10px "JetBrains Mono", monospace';
    ctx.fillText(String(i), p.px + 7, p.py - 6);
  });
  if (!e.nodes.length) {
    ctx.fillStyle = '#4a7a9b';
    ctx.font = '12px "JetBrains Mono", monospace';
    ctx.fillText('click to place nodes, then LINK two nodes', 12, 22);
  }
  $('ed-count').textContent = `${e.nodes.length} nodes · ${e.springs.length} links · ${e.springs.filter((s) => s.muscle).length} muscles`;
}

function edHit(px, py, w, h) {
  const e = S.editor;
  let best = -1;
  let bestD = 14;
  e.nodes.forEach((n, i) => {
    const p = edWorldToScreen(n.x, n.y, w, h);
    const d = Math.hypot(p.px - px, p.py - py);
    if (d < bestD) {
      bestD = d;
      best = i;
    }
  });
  return { node: best };
}

function edSpringHit(px, py, w, h) {
  const e = S.editor;
  let best = -1;
  let bestD = 8;
  e.springs.forEach((s, i) => {
    const A1 = edWorldToScreen(e.nodes[s.a].x, e.nodes[s.a].y, w, h);
    const B1 = edWorldToScreen(e.nodes[s.b].x, e.nodes[s.b].y, w, h);
    const vx = B1.px - A1.px;
    const vy = B1.py - A1.py;
    const L2 = vx * vx + vy * vy || 1;
    const t = clamp(((px - A1.px) * vx + (py - A1.py) * vy) / L2, 0, 1);
    const d = Math.hypot(A1.px + vx * t - px, A1.py + vy * t - py);
    if (d < bestD) {
      bestD = d;
      best = i;
    }
  });
  return best;
}

function edAddLink(a, b) {
  const e = S.editor;
  if (a === b) return false;
  const lo = Math.min(a, b);
  const hi = Math.max(a, b);
  if (e.springs.some((s) => s.a === lo && s.b === hi)) return false;
  if (e.springs.length >= SPRING_MAX) return false;
  const na = e.nodes[lo];
  const nb = e.nodes[hi];
  const rest = q4(clamp(Math.hypot(nb.x - na.x, nb.y - na.y), REST_MIN, REST_MAX));
  e.springs.push({ a: lo, b: hi, rest, k: 150, c: 2, muscle: e.type === 'muscle' ? 1 : 0, amp: 0.25, freq: 1.5, phase: 0 });
  return true;
}

function edEvent(kind, ev) {
  const e = S.editor;
  const rect = ED.canvas.getBoundingClientRect();
  const { w, h } = { w: rect.width, h: rect.height };
  const px = ev.clientX - rect.left;
  const py = ev.clientY - rect.top;
  const hit = edHit(px, py, w, h).node;
  if (kind === 'down') {
    ED.canvas.setPointerCapture(ev.pointerId);
    if (e.tool === 'node') {
      if (hit >= 0) e.sel = hit;
      else if (e.nodes.length < NODE_MAX) {
        const p = edScreenToWorld(px, py, w, h);
        e.nodes.push({ x: p.x, y: p.y });
        e.sel = e.nodes.length - 1;
      }
    } else if (e.tool === 'link') {
      if (hit >= 0) e.drag = { kind: 'link', from: hit, to: null };
    } else if (e.tool === 'move') {
      if (hit >= 0) e.drag = { kind: 'move', node: hit };
    } else if (e.tool === 'erase') {
      if (hit >= 0) edRemoveNode(hit);
      else {
        const si = edSpringHit(px, py, w, h);
        if (si >= 0) e.springs.splice(si, 1);
      }
    } else if (e.tool === 'toggle') {
      const si = edSpringHit(px, py, w, h);
      if (si >= 0) e.springs[si].muscle = e.springs[si].muscle ? 0 : 1;
    }
  } else if (kind === 'move') {
    if (e.drag && e.drag.kind === 'move') {
      const p = edScreenToWorld(px, py, w, h);
      e.nodes[e.drag.node].x = p.x;
      e.nodes[e.drag.node].y = p.y;
    } else if (e.drag && e.drag.kind === 'link') {
      const target = hit >= 0 && hit !== e.drag.from ? hit : -1;
      e.drag.to = { px, py };
      e.drag.over = target;
    }
  } else if (kind === 'up') {
    if (e.drag && e.drag.kind === 'link') {
      if (e.drag.over !== undefined && e.drag.over >= 0) {
        if (!edAddLink(e.drag.from, e.drag.over)) say('that link already exists or the body is full', 'warn');
      }
    }
    e.drag = null;
  }
  edDraw();
  edSyncHint();
}

function edRemoveNode(i) {
  const e = S.editor;
  e.nodes.splice(i, 1);
  e.springs = e.springs
    .filter((s) => s.a !== i && s.b !== i)
    .map((s) => ({ ...s, a: s.a > i ? s.a - 1 : s.a, b: s.b > i ? s.b - 1 : s.b }));
  e.sel = -1;
}

function edSyncHint() {
  const e = S.editor;
  const hint = $('ed-hint');
  const tips = {
    node: 'NODE: click empty space to place a node (up to 12). Click a node to select it.',
    link: 'LINK: press on one node and drag to another to add a link. The TYPE button sets muscle or spring.',
    move: 'MOVE: drag a node to reposition it.',
    erase: 'ERASE: click a node or a link to delete it.',
    toggle: 'TOGGLE: click a link to switch it between muscle and plain spring.',
  };
  hint.textContent = tips[e.tool];
}

function edTest() {
  const e = S.editor;
  if (e.nodes.length < NODE_MIN || !e.springs.length) {
    say('add at least two nodes and one link first', 'warn');
    return;
  }
  const g = repair(edGenome());
  const r = simulate(g, S.world, { sampleEvery: 4 });
  const code = encode(g);
  e.tested = { code, fitness: r.fitness, distance: r.distance, exploded: r.exploded };
  const note = r.exploded ? 'exploded (fitness 0)' : `${fmtM(r.distance)} in ${S.world.duration} s`;
  $('ed-result').textContent = `TEST: ${note}`;
  // Show the test in the arena as a user run.
  const run = { frames: r.frames, N: r.N, sampleEvery: 4, frameCount: r.frameCount, genome: g, kind: 'user', alpha: 1, label: '' };
  S.view.slowmo = false;
  setView('replay', [run], `YOUR CREATURE · ${note}`, S.world);
  $('btn-live').hidden = false;
  // Load the repaired body back so the lab shows what was simulated.
  edLoad(g, false);
}

function edDrop() {
  const e = S.editor;
  if (e.nodes.length < NODE_MIN || !e.springs.length) {
    say('add at least two nodes and one link first', 'warn');
    return;
  }
  const g = repair(edGenome());
  const code = encode(g);
  if (!S.worker || !S.ready) {
    say('the population is not ready yet', 'warn');
    return;
  }
  S.worker.postMessage({ cmd: 'inject', code });
  S.user.awaiting += 1;
  renderChallenger();
  say('creature dropped into the population: it is scored in the next generation', 'spc');
  $('ed-result').textContent = 'dropped: it is scored in the next generation';
  edLoad(g, false);
}

function edLoad(g, announce = true) {
  const e = S.editor;
  e.nodes = g.nodes.map((n) => ({ x: n.x, y: n.y }));
  e.springs = g.springs.map((s) => ({ ...s }));
  e.sel = -1;
  edDraw();
  $('ed-code').value = encode(repair(edGenome()));
  if (announce) say('loaded a creature into the lab');
}

function edCode() {
  const g = repair(edGenome());
  $('ed-code').value = encode(g);
}

// ---------------------------------------------------------------- presets and sharing
// The preset file is optional: if it cannot be read, the lab still runs and says so in the presets panel.
async function loadPresets() {
  let world = null;
  try {
    const res = await fetch('data/walkers-champions.json');
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    S.presets = data.presets || [];
    world = data.world || null;
  } catch (err) {
    S.presets = [];
  }
  const list = $('presets');
  if (!list) return;
  list.innerHTML = '';
  for (const p of S.presets) {
    const b = document.createElement('button');
    b.className = 'cyber-btn green';
    b.type = 'button';
    b.innerHTML = `<span class="btn-txt">${p.label}</span>`;
    b.title = p.note || '';
    b.addEventListener('click', () => showPreset(p, world));
    list.appendChild(b);
  }
  if (!S.presets.length) {
    list.innerHTML = '<div class="field-hint">presets could not be loaded</div>';
    return;
  }
  // Until the first generation arrives, the first preset plays in the arena so the page is never empty.
  S.presetIntro = true;
  showPreset(S.presets[0], world, { intro: true });
}

// Replays a preset in its own world (its terrain), loads it into the lab, and reports its distance.
function showPreset(p, worldExtra, opts = {}) {
  const world = { ...DEFAULT_WORLD, ...(worldExtra || {}), terrain: p.terrain || 'flat' };
  const g = decode(p.code);
  const r = simulate(g, world, { sampleEvery: 4 });
  const run = { frames: r.frames, N: r.N, sampleEvery: 4, frameCount: r.frameCount, genome: g, kind: 'champ', alpha: 1, label: '' };
  if (!opts.intro) S.presetIntro = false;
  S.view.slowmo = false;
  setView('replay', [run], `PRESET · ${p.label} · ${r.distance.toFixed(2)} m`, world);
  $('btn-live').hidden = false;
  edLoad(g, false);
  $('ed-result').textContent = `${p.label}: ${r.distance.toFixed(2)} m (preset, ${world.terrain})`;
  if (!opts.intro) say(`preset loaded: ${p.label} (${r.distance.toFixed(2)} m)`);
}

// ---------------------------------------------------------------- controls
function bindControls() {
  $('btn-toggle').addEventListener('click', () => {
    S.running = !S.running;
    $('btn-toggle').querySelector('.btn-txt').textContent = S.running ? '❚❚ PAUSE' : '▶ PLAY';
    if (S.running) pump();
  });
  $('btn-step').addEventListener('click', () => {
    S.running = false;
    $('btn-toggle').querySelector('.btn-txt').textContent = '▶ PLAY';
    if (S.ready && !S.busy) {
      S.busy = true;
      S.worker.postMessage({ cmd: 'step', topK: 4 });
    }
  });
  $('gen-speed').addEventListener('input', (e) => {
    S.genRate = SPEEDS[Number(e.target.value)];
    $('gen-speed-val').textContent = S.genRate === Infinity ? 'MAX' : `${S.genRate} gen/s`;
    pump();
  });
  $('replay-speed').addEventListener('input', (e) => {
    S.view.speed = [0.25, 0.5, 1, 2][Number(e.target.value)] ?? 1;
    $('ew-speed-val').textContent = `${S.view.speed}×`;
  });
  $('pop-size').addEventListener('change', (e) => {
    S.popSize = Number(e.target.value);
    startRun(S.seed);
  });
  $('mutation').addEventListener('input', (e) => {
    S.mutationRate = Number(e.target.value);
    $('mutation-val').textContent = S.mutationRate.toFixed(2);
    if (S.worker) S.worker.postMessage({ cmd: 'set', mutationRate: S.mutationRate });
  });
  $('terrain').addEventListener('change', (e) => {
    S.world = { ...S.world, terrain: e.target.value };
    startRun(S.seed);
  });
  $('gravity').addEventListener('input', (e) => {
    S.world = { ...S.world, gravity: Number(e.target.value) };
    $('gravity-val').textContent = `${S.world.gravity.toFixed(1)} m/s²`;
    if (S.worker) S.worker.postMessage({ cmd: 'set', world: { gravity: S.world.gravity } });
  });
  $('friction').addEventListener('input', (e) => {
    S.world = { ...S.world, friction: Number(e.target.value) };
    $('friction-val').textContent = S.world.friction.toFixed(2);
    if (S.worker) S.worker.postMessage({ cmd: 'set', world: { friction: S.world.friction } });
  });
  $('btn-reset').addEventListener('click', () => startRun(S.seed));
  $('ew-seed').addEventListener('change', (e) => {
    const v = Math.max(0, Math.min(2147483647, Math.floor(Number(e.target.value) || 0)));
    startRun(v);
  });
  $('btn-seed').addEventListener('click', () => startRun(Math.floor(Math.random() * 2147483646) + 1));
  $('btn-live').addEventListener('click', backToLive);

  $('hof-prev').addEventListener('click', () => {
    S.hofIndex = Math.max(0, S.hofIndex - 1);
    renderHof();
  });
  $('hof-next').addEventListener('click', () => {
    S.hofIndex = Math.min(S.hof.length - 1, S.hofIndex + 1);
    renderHof();
  });
  $('hof-replay').addEventListener('click', replayHof);
  $('hof-slow').addEventListener('click', () => {
    replayHof();
    S.view.speed = 0.25;
    S.view.slowmo = true;
    $('replay-speed').value = '0';
    $('ew-speed-val').textContent = '0.25×';
  });
  $('hof-edit').addEventListener('click', () => {
    const e = S.hof[S.hofIndex];
    if (e) edLoad(genomeOf(e.code));
  });
  $('lineage').addEventListener('click', (ev) => {
    const row = ev.target.closest('.lin-row');
    if (!row) return;
    const rec = S.records.get(Number(row.dataset.id));
    if (rec) replayRecord(rec, `ANCESTOR · GEN ${rec.gen} · ${fmtM(rec.fitness)}`, { kind: 'champ' });
  });

  // Creature lab.
  const tools = document.querySelectorAll('[data-tool]');
  tools.forEach((b) =>
    b.addEventListener('click', () => {
      S.editor.tool = b.dataset.tool;
      tools.forEach((x) => x.classList.toggle('on', x === b));
      edSyncHint();
    })
  );
  document.querySelectorAll('[data-type]').forEach((b) =>
    b.addEventListener('click', () => {
      S.editor.type = b.dataset.type;
      document.querySelectorAll('[data-type]').forEach((x) => x.classList.toggle('on', x === b));
    })
  );
  $('ed-clear').addEventListener('click', () => edLoad({ nodes: [], springs: [] }, false));
  $('ed-test').addEventListener('click', edTest);
  $('ed-drop').addEventListener('click', edDrop);
  $('ed-champ').addEventListener('click', () => {
    if (S.champ && S.champ.code) edLoad(genomeOf(S.champ.code));
  });
  $('ed-export').addEventListener('click', edCode);
  $('ed-import').addEventListener('click', () => {
    try {
      const g = decode($('ed-code').value);
      edLoad(g);
      $('ed-result').textContent = `loaded ${g.nodes.length} nodes`;
    } catch (err) {
      $('ed-result').textContent = `not a creature: ${err.message}`;
    }
  });
  $('ed-copy').addEventListener('click', async () => {
    edCode();
    const box = $('ed-code');
    try {
      await navigator.clipboard.writeText(box.value);
      $('ed-result').textContent = 'copied';
    } catch (err) {
      box.select();
      $('ed-result').textContent = 'select and copy (clipboard blocked)';
    }
  });

  ED.canvas.addEventListener('pointerdown', (ev) => edEvent('down', ev));
  ED.canvas.addEventListener('pointermove', (ev) => {
    if (S.editor.drag) edEvent('move', ev);
  });
  ED.canvas.addEventListener('pointerup', (ev) => edEvent('up', ev));
  ED.canvas.addEventListener('pointercancel', (ev) => edEvent('up', ev));
  window.addEventListener('resize', () => {
    backdropKey = '';
    edDraw();
    drawCharts();
  });
}

// ---------------------------------------------------------------- start
function init() {
  A.canvas = $('ew-arena');
  ED.canvas = $('ed-canvas');
  const world = S.world;
  $('ed-hint').textContent = '';
  $('terrain').innerHTML = TERRAINS.map((t) => `<option value="${t}"${t === world.terrain ? ' selected' : ''}>${t.toUpperCase()}</option>`).join('');
  $('gravity').value = String(world.gravity);
  $('gravity-val').textContent = `${world.gravity.toFixed(1)} m/s²`;
  $('friction').value = String(world.friction);
  $('friction-val').textContent = world.friction.toFixed(2);
  $('mutation').value = String(S.mutationRate);
  $('mutation-val').textContent = S.mutationRate.toFixed(2);
  $('pop-size').value = String(S.popSize);
  $('gen-speed').value = '2';
  $('gen-speed-val').textContent = `${S.genRate} gen/s`;
  $('replay-speed').value = '2';
  $('ew-speed-val').textContent = '1×';
  bindControls();
  edSyncHint();
  edDraw();
  renderChallenger();
  requestAnimationFrame((t) => {
    A.last = t;
    drawArena(t);
  });
  // The evolution starts at once; the presets load alongside and never block it.
  startRun(S.seed);
  loadPresets().catch((err) => say(`presets: ${err.message || err}`, 'warn'));
  // Keep the arena's live swap throttled on every frame.
  const tick = (now) => {
    swapPendingLive(now);
    requestAnimationFrame(tick);
  };
  requestAnimationFrame(tick);
}

init();
