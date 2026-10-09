// Search-wave reveal: a page arrives under a coarse grid of cover cells, and a breadth-first wave peels them
// away from the point the visitor clicked to get here, in about half a second. The point is kept in
// sessionStorage by a click on a same-site link (the click handler below); with no click, the wave starts at
// the centre of the screen. The overlay is a fixed canvas with pointer-events off, so it never blocks anything
// and never moves the layout. It is skipped under prefers-reduced-motion, on back/forward navigation, and when
// a reveal has just run (within a second), so quick repeat navigation does not stack waves.
//
// The grid and wave functions are pure, so tests/web/reveal.test.mjs can check them under node. The DOM code
// only runs in a browser. nav.js imports this module, so every page has it.

export const CELL = 48;          // cover cell size in CSS pixels: coarse enough to read as a grid
export const WAVE_MS = 480;      // time for the wave to reach the far corner
const FADE_MS = 110;             // each cell fades over this long once the wave reaches it
const STALE_MS = 5000;           // a click older than this is not used for the wave origin
const REPEAT_MS = 1000;          // a reveal this soon after the previous one is skipped
const KEY = 'search-agents.reveal';        // the last click point, written before navigating
const LAST_KEY = 'search-agents.reveal.last'; // when the last reveal ran

// ── Pure grid and wave functions ────────────────────────────────────────

/** Number of cover columns and rows for a viewport. */
export function gridSize(width, height, cell = CELL) {
  return { cols: Math.max(1, Math.ceil(width / cell)), rows: Math.max(1, Math.ceil(height / cell)) };
}

/** Index of the cell under (x, y), clamped to the grid, so a click at the very edge still starts a wave. */
export function startIndex(x, y, cols, rows, cell = CELL) {
  const c = Math.min(cols - 1, Math.max(0, Math.floor(x / cell)));
  const r = Math.min(rows - 1, Math.max(0, Math.floor(y / cell)));
  return r * cols + c;
}

/** Breadth-first distance (in cells, 4-neighbour steps) from the start cell to every cell. */
export function bfsDistances(cols, rows, start) {
  const dist = new Int32Array(cols * rows).fill(-1);
  const queue = new Int32Array(cols * rows);
  let head = 0, tail = 0;
  dist[start] = 0;
  queue[tail++] = start;
  while (head < tail) {
    const u = queue[head++];
    const c = u % cols, r = (u / cols) | 0;
    const next = [];
    if (c > 0) next.push(u - 1);
    if (c < cols - 1) next.push(u + 1);
    if (r > 0) next.push(u - cols);
    if (r < rows - 1) next.push(u + cols);
    for (const v of next) {
      if (dist[v] < 0) { dist[v] = dist[u] + 1; queue[tail++] = v; }
    }
  }
  return dist;
}

/** When each cell is reached by the wave, in milliseconds: the start at 0, the farthest cell at waveMs. */
export function waveTimes(dist, waveMs = WAVE_MS) {
  let max = 0;
  for (const d of dist) if (d > max) max = d;
  const times = new Float64Array(dist.length);
  if (max === 0) return times;
  for (let i = 0; i < dist.length; i++) times[i] = (dist[i] / max) * waveMs;
  return times;
}

// ── Browser: the click point and the overlay ────────────────────────────

const browser = typeof window !== 'undefined' && typeof document !== 'undefined';
const REDUCED = browser && matchMedia('(prefers-reduced-motion: reduce)').matches;

/** A click on a same-site link stores its point, so the next page's wave starts there. */
function rememberClick(e) {
  if (e.defaultPrevented || e.button !== 0 || e.metaKey || e.ctrlKey || e.shiftKey || e.altKey) return;
  const a = e.target instanceof Element ? e.target.closest('a[href]') : null;
  if (!a || (a.target && a.target !== '_self') || a.hasAttribute('download')) return;
  const url = new URL(a.href, location.href);
  if (url.origin !== location.origin) return;
  // A same-page link (only a hash changes) does not load a new page, so there is nothing to reveal.
  if (url.pathname === location.pathname && url.search === location.search) return;
  try {
    sessionStorage.setItem(KEY, JSON.stringify({ x: e.clientX, y: e.clientY, t: Date.now() }));
  } catch { /* storage may be blocked: the wave then starts at the centre */ }
}

function playWave(x, y) {
  const w = innerWidth, h = innerHeight;
  const { cols, rows } = gridSize(w, h);
  const dist = bfsDistances(cols, rows, startIndex(x, y, cols, rows));
  const times = waveTimes(dist);
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const canvas = document.createElement('canvas');
  // Inline style keeps this self-contained: fixed over the page, never receiving pointer events.
  canvas.style.cssText = 'position:fixed;left:0;top:0;width:100vw;height:100vh;pointer-events:none;z-index:65;';
  canvas.setAttribute('aria-hidden', 'true');
  canvas.width = Math.round(w * dpr);
  canvas.height = Math.round(h * dpr);
  document.body.appendChild(canvas);
  const ctx = canvas.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  const t0 = performance.now();
  const end = WAVE_MS + FADE_MS;

  const frame = (now) => {
    const t = now - t0;
    ctx.clearRect(0, 0, w, h);
    for (let i = 0; i < times.length; i++) {
      // Covered until the wave arrives, then fading out over FADE_MS.
      const left = Math.min(1, Math.max(0, (t - times[i]) / FADE_MS));
      const alpha = 1 - left;
      if (alpha <= 0) continue;
      const c = i % cols, r = (i / cols) | 0;
      ctx.fillStyle = `rgba(2,6,16,${alpha})`;
      ctx.fillRect(c * CELL, r * CELL, CELL, CELL);
      // The grid lines of the cover, so the peel reads as a grid being cleared.
      ctx.strokeStyle = `rgba(0,245,255,${0.28 * alpha})`;
      ctx.lineWidth = 1;
      ctx.strokeRect(c * CELL + 0.5, r * CELL + 0.5, CELL - 1, CELL - 1);
    }
    if (t < end) requestAnimationFrame(frame);
    else canvas.remove();
  };
  requestAnimationFrame(frame);
}

function revealOnArrival() {
  let point = null;
  try {
    const raw = sessionStorage.getItem(KEY);
    sessionStorage.removeItem(KEY);   // used once: a reload or a back visit must not reuse it
    if (raw) point = JSON.parse(raw);
  } catch { point = null; }
  const nav = performance.getEntriesByType?.('navigation')?.[0];
  if (REDUCED || document.hidden || nav?.type === 'back_forward') return;
  const now = Date.now();
  let last = 0;
  try { last = Number(sessionStorage.getItem(LAST_KEY)) || 0; } catch { /* ignore */ }
  if (now - last < REPEAT_MS) return;
  try { sessionStorage.setItem(LAST_KEY, String(now)); } catch { /* ignore */ }
  const fresh = point && now - point.t <= STALE_MS;
  const x = fresh ? point.x : innerWidth / 2;
  const y = fresh ? point.y : innerHeight / 2;
  playWave(x, y);
}

if (browser) {
  document.addEventListener('click', rememberClick);
  revealOnArrival();
}
