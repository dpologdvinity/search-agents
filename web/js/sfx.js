// Synth sound effects: short voices built from Web Audio oscillators, with no audio files.
//
// Off by default. The toggle in the nav turns it on and the choice is kept in localStorage. The AudioContext
// is created only inside a user gesture (a click, key press or touch) while sound is on, so nothing plays on
// page load and nothing autoplays. Sound stops while the tab is hidden. Every sound passes a global rate
// limit and a per-sound minimum gap, so watch modes and fast searches stay quiet. Games do not call this
// module directly for most moments: fx.js (burst, shake, pop, banner) and the document-level listeners below
// cover the shared effects, so most pages get sound with no per-page edits.

const KEY = 'search-agents.sound';
const MASTER = 0.5;            // overall level; each voice is already quiet
const MAX_PER_SECOND = 30;     // global cap across all voices
const SOFT_DELAY_MS = 80;      // soft voices wait this long so a banner can silence them

let enabled = readStored();
let ctx = null, master = null, send = null, noiseBuf = null;
const listeners = new Set();
const recent = [];             // start times of recent sounds, for the global cap
const lastAt = new Map();      // per-sound start times, for the per-sound gap
let lastImportant = -Infinity; // when a banner-level sound last played

function readStored() {
  try { return globalThis.localStorage?.getItem(KEY) === '1'; } catch { return false; }
}

// ── Audio graph ─────────────────────────────────────────────────────────

// Created only from a user gesture. Returns null when the browser has no Web Audio.
function ensureContext() {
  if (ctx) return ctx;
  const AC = globalThis.AudioContext || globalThis.webkitAudioContext;
  if (!AC) return null;
  ctx = new AC();
  master = ctx.createGain();
  master.gain.value = MASTER;
  const limiter = ctx.createDynamicsCompressor(); // stacked voices must not clip
  master.connect(limiter).connect(ctx.destination);
  // A synthwave echo: a quarter-second delay whose repeats are darkened by a lowpass each pass.
  send = ctx.createGain();
  send.gain.value = 0.25;
  const delay = ctx.createDelay(1);
  delay.delayTime.value = 0.23;
  const feedback = ctx.createGain();
  feedback.gain.value = 0.3;
  const tone = ctx.createBiquadFilter();
  tone.type = 'lowpass';
  tone.frequency.value = 2400;
  send.connect(delay);
  delay.connect(tone);
  tone.connect(feedback).connect(delay);
  tone.connect(master);
  return ctx;
}

// One oscillator through a lowpass and a short envelope, optionally sent to the echo.
function note(c, when, { wave = 'square', f0, f1 = f0, dur = 0.12, vol = 0.1, cut = 3000, cutEnd = cut, detune = 0, echo = 0 }) {
  const osc = c.createOscillator();
  const lp = c.createBiquadFilter();
  const env = c.createGain();
  osc.type = wave;
  osc.frequency.setValueAtTime(f0, when);
  if (f1 !== f0) osc.frequency.exponentialRampToValueAtTime(f1, when + dur);
  osc.detune.value = detune;
  lp.type = 'lowpass';
  lp.frequency.setValueAtTime(cut, when);
  if (cutEnd !== cut) lp.frequency.exponentialRampToValueAtTime(cutEnd, when + dur);
  env.gain.setValueAtTime(0.0001, when);
  env.gain.exponentialRampToValueAtTime(vol, when + Math.min(0.006, dur / 4));
  env.gain.exponentialRampToValueAtTime(0.0001, when + dur);
  osc.connect(lp).connect(env).connect(master);
  if (echo) {
    const s = c.createGain();
    s.gain.value = echo;
    env.connect(s).connect(send);
  }
  osc.start(when);
  osc.stop(when + dur + 0.03);
}

// A detuned pair of saws (or squares): the wide, slightly out-of-tune synthwave lead.
function pair(c, when, opts) {
  note(c, when, { ...opts, wave: opts.wave ?? 'sawtooth', detune: -7 });
  note(c, when, { ...opts, wave: opts.wave ?? 'sawtooth', detune: 7 });
}

function noise(c, when, dur, vol) {
  if (!noiseBuf) {
    noiseBuf = c.createBuffer(1, Math.round(c.sampleRate * 0.4), c.sampleRate);
    const data = noiseBuf.getChannelData(0);
    for (let i = 0; i < data.length; i++) data[i] = Math.random() * 2 - 1;
  }
  const src = c.createBufferSource();
  src.buffer = noiseBuf;
  const bp = c.createBiquadFilter();
  bp.type = 'bandpass';
  bp.Q.value = 3;
  bp.frequency.setValueAtTime(3200, when);
  bp.frequency.exponentialRampToValueAtTime(400, when + dur);
  const env = c.createGain();
  env.gain.setValueAtTime(vol, when);
  env.gain.exponentialRampToValueAtTime(0.0001, when + dur);
  src.connect(bp).connect(env).connect(master);
  src.start(when);
  src.stop(when + dur + 0.02);
}

// ── Voices ──────────────────────────────────────────────────────────────

const C5 = 523.25, E5 = 659.25, G5 = 783.99, B5 = 987.77, C6 = 1046.5, E6 = 1318.51, G6 = 1567.98;

// gap: minimum ms between two plays of this voice. soft: skipped when a banner sounds nearby, and deferred.
const VOICES = {
  click: { gap: 35, play: (c, t) => note(c, t, { wave: 'square', f0: 1400, f1: 800, dur: 0.045, vol: 0.05, cut: 4200 }) },
  hover: { gap: 90, play: (c, t) => note(c, t, { wave: 'triangle', f0: 2200, dur: 0.02, vol: 0.012 }) },
  menu: { gap: 150, play: (c, t) => {
    note(c, t, { wave: 'triangle', f0: 330, f1: 495, dur: 0.08, vol: 0.06 });
    note(c, t + 0.07, { wave: 'triangle', f0: 495, f1: 660, dur: 0.1, vol: 0.05 });
  } },
  // Expansion micro-blip: pitch climbs with p in [0, 1].
  expand: { gap: 40, play: (c, t, p = 0) => note(c, t, { wave: 'triangle', f0: 330 * 2 ** (1.6 * p), dur: 0.035, vol: 0.03 }) },
  burst: { gap: 150, soft: true, play: (c, t) => [660, 880, 1100].forEach((f, i) =>
    note(c, t + i * 0.045, { wave: 'triangle', f0: f, dur: 0.07, vol: 0.04, echo: 0.3 })) },
  thud: { gap: 120, soft: true, play: (c, t) => {
    note(c, t, { wave: 'sine', f0: 110, f1: 42, dur: 0.2, vol: 0.3 });
  } },
  thudBig: { gap: 160, soft: true, play: (c, t) => {
    note(c, t, { wave: 'sine', f0: 95, f1: 30, dur: 0.36, vol: 0.34 });
    note(c, t, { wave: 'square', f0: 70, f1: 36, dur: 0.2, vol: 0.05, cut: 500 });
  } },
  tick: { gap: 60, soft: true, play: (c, t) => note(c, t, { wave: 'sine', f0: 1700, f1: 2600, dur: 0.05, vol: 0.035 }) },
  // Win: a synthwave arpeggio resolving into a chord with echo.
  win: { gap: 400, important: true, play: (c, t) => {
    [C5, E5, G5, C6].forEach((f, i) => pair(c, t + i * 0.085, { f0: f, dur: 0.12, vol: 0.05, cut: 1800, cutEnd: 4200, echo: 0.3 }));
    [C6, E6, G6].forEach((f) => pair(c, t + 0.36, { f0: f, dur: 0.6, vol: 0.035, cut: 2600, cutEnd: 1200, echo: 0.6 }));
  } },
  // Fail: a falling saw buzz with a sub under it.
  fail: { gap: 400, important: true, play: (c, t) => {
    pair(c, t, { f0: 185, f1: 70, dur: 0.42, vol: 0.05, cut: 1400, cutEnd: 180 });
    note(c, t, { wave: 'square', f0: 92, f1: 46, dur: 0.42, vol: 0.05, cut: 900, cutEnd: 120 });
  } },
  // Neutral banner: a rising two-note chime.
  chime: { gap: 250, important: true, play: (c, t) => {
    note(c, t, { wave: 'triangle', f0: C5, dur: 0.18, vol: 0.07 });
    note(c, t + 0.09, { wave: 'triangle', f0: G5, dur: 0.3, vol: 0.07, echo: 0.5 });
  } },
  // Glitch: a noise sweep with a falling saw.
  zap: { gap: 400, important: true, play: (c, t) => {
    noise(c, t, 0.22, 0.12);
    pair(c, t, { f0: 1800, f1: 150, dur: 0.22, vol: 0.03, cut: 5000, cutEnd: 400, echo: 0.4 });
  } },
  // Insert coin: the two-note arcade pickup.
  coin: { gap: 150, play: (c, t) => {
    note(c, t, { wave: 'square', f0: B5, dur: 0.07, vol: 0.07, cut: 5000 });
    note(c, t + 0.07, { wave: 'square', f0: E6, dur: 0.3, vol: 0.07, cut: 5000, echo: 0.4 });
  } },
  on: { gap: 0, play: (c, t) => note(c, t, { wave: 'sawtooth', f0: 330, f1: 880, dur: 0.14, vol: 0.05, cut: 1200, cutEnd: 5000 }) },
  off: { gap: 0, play: (c, t) => note(c, t, { wave: 'sawtooth', f0: 880, f1: 220, dur: 0.16, vol: 0.05, cut: 5000, cutEnd: 600 }) },
  // Agent thinking: one soft sine pulse. thinking() paces these.
  think: { gap: 300, play: (c, t) => note(c, t, { wave: 'sine', f0: 220, dur: 0.3, vol: 0.02 }) },
};

const IMPORTANT = new Set(Object.keys(VOICES).filter((k) => VOICES[k].important));

function schedule(name, p) {
  const v = VOICES[name];
  const t = ctx.currentTime + 0.005;
  v.play(ctx, t, p);
}

// ── Public API ──────────────────────────────────────────────────────────

/** Play a named voice if sound is on, the context is running, and the rate limits allow it. */
export function play(name, p) {
  if (!enabled || !ctx || ctx.state !== 'running' || !VOICES[name]) return false;
  if (typeof document !== 'undefined' && document.hidden) return false;
  const now = performance.now();
  const v = VOICES[name];
  if (now - (lastAt.get(name) ?? -Infinity) < v.gap) return false;
  while (recent.length && now - recent[0] > 1000) recent.shift();
  if (recent.length >= MAX_PER_SECOND) return false;
  lastAt.set(name, now);
  recent.push(now);
  if (IMPORTANT.has(name)) lastImportant = now;
  if (v.soft) {
    // A soft voice is silent when a banner sounds within SOFT_DELAY_MS either side of it. It waits that long
    // first, so a banner fired right after it (burst, then banner) still silences it.
    if (now - lastImportant < SOFT_DELAY_MS) return false;
    setTimeout(() => {
      if (lastImportant >= now - SOFT_DELAY_MS || !enabled || ctx?.state !== 'running') return;
      schedule(name, p);
    }, SOFT_DELAY_MS);
  } else {
    schedule(name, p);
  }
  return true;
}

/** Banner outcome from its colour and text: win, fail, or a neutral chime. */
export function outcome(text = '', color = '') {
  const t = String(text).toUpperCase();
  if (/GAME OVER|OUT OF|BOOM|CAUGHT|FALLEN|DEADLOCK|UNSOLVABLE|NO SOLUTION|GAVE UP|AI WINS|DEALER|ERROR|UNREACHABLE|LONGER|TIME UP|BUST/.test(t)
      || color.toLowerCase() === '#ff3b3b') return 'fail';
  if (/SOLVED|WIN|CLEARED|REACHED|BALANCED|JACKPOT|TETRIS|CHECKMATE|FOUND|OPTIMAL|LIGHTS OUT|BEAT|CHIPS|BLACKJACK/.test(t)
      || color.toLowerCase() === '#00ff88') return 'win';
  return 'note';
}

/** Play the sound for a banner: win, fail, or neutral. */
export function announce(kind) {
  return play(kind === 'win' ? 'win' : kind === 'fail' ? 'fail' : 'chime');
}

/** Expansion micro-blip for search progress: p in [0, 1] raises the pitch. Rate-limited to ~25 per second. */
export function expand(p) {
  return play('expand', Math.max(0, Math.min(1, p)));
}

/** Pulse softly while a request the user started is pending. Starts after 350 ms, at most 8 pulses. */
export function waitFor(promise) {
  let pulses = 0, timer = 0, done = false;
  const pulse = () => {
    if (done || pulses >= 8) return;
    pulses++;
    play('think');
    timer = setTimeout(pulse, 650);
  };
  timer = setTimeout(pulse, 350);
  const end = () => { done = true; clearTimeout(timer); };
  return promise.then((v) => { end(); return v; }, (err) => { end(); throw err; });
}

export function isEnabled() {
  return enabled;
}

/** Turn sound on or off, remember the choice, and notify subscribers. Turning it on is a user gesture. */
export function setEnabled(on) {
  try { globalThis.localStorage?.setItem(KEY, on ? '1' : '0'); } catch { /* storage may be blocked */ }
  if (on) {
    enabled = true;
    ensureContext();
    const chirp = () => play('on');
    if (ctx?.state === 'running') chirp();
    else if (ctx) ctx.resume().then(chirp, () => {});
  } else {
    if (enabled) play('off'); // still enabled here, so the goodbye chirp is allowed to play
    enabled = false;
    // Suspend after the chirp has had time to finish, so the audio thread goes idle.
    if (ctx) setTimeout(() => { if (!enabled && ctx.state === 'running') ctx.suspend(); }, 200);
  }
  for (const fn of listeners) fn(enabled);
}

/** Call fn(enabled) when the setting changes, including from another tab. */
export function subscribe(fn) {
  listeners.add(fn);
  return () => listeners.delete(fn);
}

// ── Browser wiring ──────────────────────────────────────────────────────

const CLICKABLE = 'button, summary, .cyber-btn, [role="button"], input[type="checkbox"], input[type="radio"], .nav-links a, .games-panel a';
const HOVERABLE = '.cabinet, .nav-links a, .games-panel a, button:not([disabled]), .cyber-btn, summary';

if (typeof window !== 'undefined' && typeof document !== 'undefined') {
  // The first gesture may create the context, but only when sound is already on.
  const onGesture = () => {
    if (!enabled) return;
    ensureContext();
    if (ctx?.state === 'suspended' && !document.hidden) ctx.resume();
  };
  for (const type of ['pointerdown', 'keydown', 'touchend', 'click']) window.addEventListener(type, onGesture, { passive: true });

  document.addEventListener('visibilitychange', () => {
    if (!ctx) return;
    if (document.hidden) ctx.suspend();
    else if (enabled) ctx.resume();
  });

  // Clicks: cabinets get the coin, the games menu its own chirp, other controls a click. The toggle plays its own.
  document.addEventListener('click', (e) => {
    const t = e.target;
    if (!(t instanceof Element) || t.closest('.sound-toggle')) return;
    if (t.closest('.cabinet')) play('coin');
    else if (t.closest('summary')) play('menu');
    else if (t.closest(CLICKABLE)) play('click');
  });

  // Hover ticks, mouse only and only when entering the element (not moving between its children).
  document.addEventListener('pointerover', (e) => {
    if (e.pointerType !== 'mouse' || !(e.target instanceof Element)) return;
    const el = e.target.closest(HOVERABLE);
    if (!el || (e.relatedTarget instanceof Node && el.contains(e.relatedTarget))) return;
    play('hover');
  });

  // Another tab turning sound on or off applies here too, without a reload.
  window.addEventListener('storage', (e) => {
    if (e.key !== KEY) return;
    enabled = e.newValue === '1';
    if (!enabled && ctx?.state === 'running') ctx.suspend();
    if (enabled && ctx?.state === 'suspended' && !document.hidden) ctx.resume();
    for (const fn of listeners) fn(enabled);
  });
}
