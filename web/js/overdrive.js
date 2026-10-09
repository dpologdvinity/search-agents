// AGENT OVERDRIVE: a temporary mode, opened by the Konami code or by seven quick taps on the nav logo.
// It runs the landing page's cabinet sims at ten times speed and turns the hero search into an A*-vs-BFS race
// (landing.js reads speedFactor() and subscribe()). On every page it adds a banner, a scanline and pulse layer,
// a countdown badge, and fx bursts. Escape, the code again, the badge, or 30 seconds all end it.
//
// Under prefers-reduced-motion the banner still shows, but nothing animates, strobes or bursts. The module
// is imported by nav.js, so every page has the listeners.

import { burst } from './fx.js';
import { konamiStep, normalizeKey } from './konami.js';
import { play } from './sfx.js';

export const SPEEDUP = 10;     // cabinet sims tick this many times faster while overdrive is on
const DURATION_MS = 30000;     // overdrive ends itself after this long
// Seven taps no more than 0.8 s apart. The navigation wait is longer than that gap, so a run of taps is
// never cut short by a link that follows the first tap.
const TAPS = 7, TAP_GAP_MS = 800, NAV_DELAY_MS = 850;
const REDUCED = matchMedia('(prefers-reduced-motion: reduce)').matches;

let active = false;
let endAt = 0;
let timers = { end: 0, badge: 0, burst: 0, banner: 0 };
let root = null;
const listeners = new Set();

export function isActive() {
  return active;
}

/** The speed multiplier for animations that tick on their own (1 normally, SPEEDUP in overdrive). */
export function speedFactor() {
  return active ? SPEEDUP : 1;
}

/** Call fn(on) whenever overdrive turns on or off. Returns the unsubscribe function. */
export function subscribe(fn) {
  listeners.add(fn);
  return () => listeners.delete(fn);
}

/** Turn overdrive on or off. Idempotent. */
export function setOverdrive(on) {
  if (on === active) return;
  if (on) enter();
  else leave();
  for (const fn of listeners) fn(active);
}

function enter() {
  active = true;
  document.documentElement.classList.add('overdrive');
  root = document.createElement('div');
  root.className = 'od-root';
  root.innerHTML = '<div class="od-scan" aria-hidden="true"></div>'
    + '<div class="od-pulse" aria-hidden="true"></div>'
    + '<div class="od-banner" role="status"><span class="od-banner-title glitch" data-text="AGENT OVERDRIVE">AGENT OVERDRIVE</span>'
    + '<span class="od-banner-sub">Esc or the code again exits. Auto-exit in 30 s.</span></div>'
    + '<button class="od-badge" type="button" title="Exit overdrive"></button>';
  document.body.appendChild(root);
  root.querySelector('.od-badge').addEventListener('click', () => setOverdrive(false));
  endAt = Date.now() + DURATION_MS;
  updateBadge();
  timers.badge = setInterval(updateBadge, 250);
  timers.end = setTimeout(() => setOverdrive(false), DURATION_MS);
  timers.banner = setTimeout(() => root?.querySelector('.od-banner')?.classList.add('out'), 2600);
  if (!REDUCED) timers.burst = setInterval(randomBurst, 1400);
  play('zap');
}

function leave() {
  active = false;
  clearTimeout(timers.end);
  clearTimeout(timers.banner);
  clearInterval(timers.badge);
  clearInterval(timers.burst);
  timers = { end: 0, badge: 0, burst: 0, banner: 0 };
  document.documentElement.classList.remove('overdrive');
  root?.remove();
  root = null;
  play('off');
}

function updateBadge() {
  const left = Math.max(0, Math.ceil((endAt - Date.now()) / 1000));
  const badge = root?.querySelector('.od-badge');
  if (badge) badge.textContent = `OVERDRIVE ${left}s // Esc exits`;
}

// Bursts land at random points in the viewport, so every page gets the same treatment.
function randomBurst() {
  burst([innerWidth * (0.1 + Math.random() * 0.8), innerHeight * (0.15 + Math.random() * 0.7)], { count: 36 });
}

// ── Keyboard: the Konami code toggles, Escape exits ─────────────────────

let progress = 0;   // keys of the Konami code matched so far (konami.js)
document.addEventListener('keydown', (e) => {
  if (e.repeat || e.ctrlKey || e.metaKey || e.altKey) return;
  if (active && e.key === 'Escape') { setOverdrive(false); return; }
  // Typing in a field should never trigger the code.
  if (e.target instanceof Element && e.target.closest('input, textarea, select, [contenteditable="true"]')) return;
  const step = konamiStep(progress, normalizeKey(e.key));
  progress = step.progress;
  if (step.done) setOverdrive(!active);
});

// ── Touch: seven quick taps on the nav logo ────────────────────────────
// A touch tap on the logo waits briefly before following the link, so a run of taps is not cut short by
// the navigation. Mouse and keyboard clicks are left alone.

const logo = document.querySelector('.nav-logo');
if (logo) {
  let touching = false, taps = 0, last = 0, navTimer = 0;
  logo.addEventListener('pointerdown', (e) => { touching = e.pointerType === 'touch'; }, true);
  logo.addEventListener('click', (e) => {
    const isTouch = touching;
    touching = false;
    if (!isTouch) return;
    e.preventDefault();
    const now = performance.now();
    taps = now - last < TAP_GAP_MS ? taps + 1 : 1;
    last = now;
    clearTimeout(navTimer);
    if (taps >= TAPS) { taps = 0; setOverdrive(!active); return; }
    const href = logo.querySelector('a')?.href;
    if (href) navTimer = setTimeout(() => { location.href = href; }, NAV_DELAY_MS);
  });
}
