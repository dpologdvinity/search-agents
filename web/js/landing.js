// Landing page: A* across a neon city grid that chases the pointer, a one-time
// boot sequence, and four arcade cabinets each animating its agent's mechanism.
// Everything pauses while off-screen and honors prefers-reduced-motion.

import { play } from './sfx.js';
import { makeMaze, run } from './pathfind-core.js';
// Overdrive (overdrive.js): sims tick faster, and the hero search gets a BFS race beside it while it is on.
import { speedFactor, subscribe as onOverdrive } from './overdrive.js';

const REDUCED = matchMedia('(prefers-reduced-motion: reduce)').matches;
const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
const C = {
  cyan: css('--cyan') || '#00f5ff', pink: css('--pink') || '#ff00a0', purple: css('--purple') || '#9b00ff',
  green: css('--green') || '#00ff88', yellow: css('--yellow') || '#ffe600', dim: css('--dim') || '#4a7a9b',
  bg: '#020610',
};

function onVisible(el, cb) {
  // cb(true/false) whenever el enters or leaves the viewport.
  new IntersectionObserver((entries) => entries.forEach((e) => cb(e.isIntersecting))).observe(el);
}

function sizeCanvas(canvas, w, h) {
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  canvas.width = Math.round(w * dpr);
  canvas.height = Math.round(h * dpr);
  const ctx = canvas.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return ctx;
}

// ── Boot sequence (runs once) ───────────────────────────────────────────

const BOOT = [
  '> init search_agents --target=browser',
  '> load heuristics: manhattan, linear_conflict, pdb_555, neural ... ok',
  '> load policy_value_net, ntuple_2048 ... ok',
  '> open frontier ... A* online',
];

function boot() {
  const el = document.getElementById('boot');
  if (REDUCED) { el.textContent = BOOT.join('\n'); return; }
  let line = 0, ch = 0;
  const tick = () => {
    if (line >= BOOT.length) return;
    ch += 3;
    el.textContent = BOOT.slice(0, line).join('\n') + (line ? '\n' : '') + BOOT[line].slice(0, ch);
    if (ch >= BOOT[line].length) { line++; ch = 0; }
    setTimeout(tick, 18);
  };
  tick();
}

// ── Hero: A* across the city ────────────────────────────────────────────

class MinHeap {
  constructor() { this.a = []; }
  get size() { return this.a.length; }
  push(x) { const a = this.a; a.push(x); let i = a.length - 1;
    while (i > 0) { const p = (i - 1) >> 1; if (a[p][0] <= a[i][0]) break; [a[p], a[i]] = [a[i], a[p]]; i = p; } }
  pop() { const a = this.a; const top = a[0]; const last = a.pop();
    if (a.length) { a[0] = last; let i = 0;
      for (;;) { const l = 2 * i + 1, r = l + 1; let m = i;
        if (l < a.length && a[l][0] < a[m][0]) m = l; if (r < a.length && a[r][0] < a[m][0]) m = r;
        if (m === i) break; [a[m], a[i]] = [a[i], a[m]]; i = m; } }
    return top; }
}

function city() {
  const canvas = document.getElementById('city');
  const hero = canvas.parentElement;
  const CELL = 26;
  let ctx, W, H, cols, rows, walls, start, goal, search, visible = true, raf = 0, t = 0;
  let race = null, racePanel = null;   // overdrive only: the BFS race state and its counters panel
  const hud = { expanded: document.getElementById('hud-expanded'),
                frontier: document.getElementById('hud-frontier'), path: document.getElementById('hud-path') };

  function layout() {
    W = hero.clientWidth; H = hero.clientHeight;
    ctx = sizeCanvas(canvas, W, H);
    cols = Math.ceil(W / CELL); rows = Math.ceil(H / CELL);
    walls = new Uint8Array(cols * rows);
    // A deterministic skyline: blocks of buildings with streets between them.
    let seed = 7;
    const rnd = () => ((seed = (seed * 16807) % 2147483647) / 2147483647);
    for (let r = 0; r < rows; r++)
      for (let c = 0; c < cols; c++)
        if (r % 4 !== 0 && c % 5 !== 0 && rnd() < 0.62) walls[r * cols + c] = 1 + Math.floor(rnd() * 3);
    start = idx(Math.floor(cols * 0.82), Math.floor(rows * 0.78));
    walls[start] = 0;
    goal = idx(Math.floor(cols * 0.15), Math.floor(rows * 0.2));
    walls[goal] = 0;
    restart();
  }

  const idx = (c, r) => r * cols + c;
  const h = (a, b) => Math.abs((a % cols) - (b % cols)) + Math.abs(((a / cols) | 0) - ((b / cols) | 0));

  function restart() {
    const g = new Float64Array(cols * rows).fill(Infinity);
    g[start] = 0;
    const heap = new MinHeap();
    heap.push([h(start, goal), start]);
    search = { g, parent: new Int32Array(cols * rows).fill(-1), closed: new Uint8Array(cols * rows),
               open: new Uint8Array(cols * rows), heap, expanded: 0, done: false, path: [] };
    search.open[start] = 1;
    if (race) race = raceState();
  }

  function step(budget) {
    const s = search;
    while (budget-- > 0 && !s.done) {
      if (!s.heap.size) { s.done = true; break; }
      const [, u] = s.heap.pop();
      if (s.closed[u]) continue;
      s.closed[u] = 1; s.open[u] = 0; s.expanded++;
      if (u === goal) {
        s.done = true;
        for (let v = u; v !== -1; v = s.parent[v]) s.path.push(v);
        break;
      }
      const c = u % cols, r = (u / cols) | 0;
      for (const [dc, dr] of [[1, 0], [-1, 0], [0, 1], [0, -1]]) {
        const nc = c + dc, nr = r + dr;
        if (nc < 0 || nr < 0 || nc >= cols || nr >= rows) continue;
        const v = idx(nc, nr);
        if (walls[v] || s.closed[v]) continue;
        const ng = s.g[u] + 1;
        if (ng < s.g[v]) { s.g[v] = ng; s.parent[v] = u; s.open[v] = 1; s.heap.push([ng + h(v, goal), v]); }
      }
    }
  }

  function draw() {
    ctx.fillStyle = C.bg;
    ctx.fillRect(0, 0, W, H);
    const s = search;
    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        const i = idx(c, r), x = c * CELL, y = r * CELL;
        const wall = walls[i];
        if (wall) {
          // Buildings: taller ones glow more, with lit windows.
          const a = 0.1 + wall * 0.08;
          ctx.fillStyle = `rgba(155,0,255,${a})`;
          ctx.fillRect(x + 2, y + 2, CELL - 4, CELL - 4);
          ctx.strokeStyle = `rgba(155,0,255,${a + 0.25})`;
          ctx.strokeRect(x + 2.5, y + 2.5, CELL - 5, CELL - 5);
          if ((i * 7 + wall) % 5 === 0) { ctx.fillStyle = 'rgba(255,230,0,0.55)'; ctx.fillRect(x + 8, y + 8, 3, 3); }
        } else if (s.closed[i]) {
          ctx.fillStyle = 'rgba(0,245,255,0.10)';
          ctx.fillRect(x + 1, y + 1, CELL - 2, CELL - 2);
        } else if (s.open[i]) {
          // Frontier: hollow cyan squares (shape, not only colour, marks the state).
          ctx.strokeStyle = C.cyan;
          ctx.lineWidth = 1.5;
          ctx.strokeRect(x + 5, y + 5, CELL - 10, CELL - 10);
          ctx.lineWidth = 1;
        } else {
          ctx.fillStyle = 'rgba(0,245,255,0.05)';
          ctx.fillRect(x + CELL / 2 - 1, y + CELL / 2 - 1, 2, 2);
        }
      }
    }
    // Path: a glowing magenta polyline.
    if (s.path.length > 1) {
      ctx.save();
      ctx.shadowColor = C.pink; ctx.shadowBlur = 16;
      ctx.strokeStyle = C.pink; ctx.lineWidth = 3.5; ctx.lineJoin = 'round';
      ctx.beginPath();
      s.path.forEach((v, k) => {
        const x = (v % cols) * CELL + CELL / 2, y = ((v / cols) | 0) * CELL + CELL / 2;
        if (k) ctx.lineTo(x, y); else ctx.moveTo(x, y);
      });
      ctx.stroke();
      ctx.restore();
    }
    // Start (green diamond) and goal (yellow crosshair).
    const sx = (start % cols) * CELL + CELL / 2, sy = ((start / cols) | 0) * CELL + CELL / 2;
    ctx.save(); ctx.translate(sx, sy); ctx.rotate(Math.PI / 4);
    ctx.fillStyle = C.green; ctx.shadowColor = C.green; ctx.shadowBlur = 14;
    ctx.fillRect(-6, -6, 12, 12); ctx.restore();
    const gx = (goal % cols) * CELL + CELL / 2, gy = ((goal / cols) | 0) * CELL + CELL / 2;
    const pulse = 9 + Math.sin(t / 8) * 2;
    ctx.save(); ctx.strokeStyle = C.yellow; ctx.shadowColor = C.yellow; ctx.shadowBlur = 12; ctx.lineWidth = 2;
    ctx.beginPath(); ctx.arc(gx, gy, pulse, 0, Math.PI * 2); ctx.moveTo(gx - 15, gy); ctx.lineTo(gx + 15, gy);
    ctx.moveTo(gx, gy - 15); ctx.lineTo(gx, gy + 15); ctx.stroke(); ctx.restore();

    hud.expanded.textContent = s.expanded.toLocaleString();
    hud.frontier.textContent = s.open.reduce((a, b) => a + b, 0).toLocaleString();
    hud.path.textContent = s.path.length ? String(s.path.length - 1) : (s.done ? 'no route' : '…');
    if (race) drawRace();
  }

  // ── Overdrive race (overdrive.js) ──
  // A breadth-first search from the same start to the same cursor, drawn in yellow beside the A* search.
  // BFS has no heuristic: it floods outward in rings, so on this open map it expands many more cells than A*,
  // which heads for the goal. Both share the walls, the start and the budget of 30 expansions per frame.
  function raceState() {
    const n = cols * rows;
    const r = { closed: new Uint8Array(n), seen: new Uint8Array(n), parent: new Int32Array(n).fill(-1),
                queue: [start], head: 0, expanded: 0, done: false, path: [] };
    r.seen[start] = 1;
    return r;
  }

  function raceStep(budget) {
    const r = race;
    while (budget-- > 0 && !r.done) {
      if (r.head >= r.queue.length) { r.done = true; break; }
      const u = r.queue[r.head++];
      r.closed[u] = 1; r.expanded++;
      if (u === goal) {
        r.done = true;
        for (let v = u; v !== -1; v = r.parent[v]) r.path.push(v);
        break;
      }
      const c = u % cols, row = (u / cols) | 0;
      for (const [dc, dr] of [[1, 0], [-1, 0], [0, 1], [0, -1]]) {
        const nc = c + dc, nr = row + dr;
        if (nc < 0 || nr < 0 || nc >= cols || nr >= rows) continue;
        const v = idx(nc, nr);
        if (walls[v] || r.seen[v]) continue;
        r.seen[v] = 1; r.parent[v] = u; r.queue.push(v);
      }
    }
  }

  function drawRace() {
    const r = race;
    for (let row = 0; row < rows; row++) {
      for (let c = 0; c < cols; c++) {
        const i = idx(c, row);
        if (walls[i]) continue;
        const x = c * CELL, y = row * CELL;
        if (r.closed[i]) {
          ctx.fillStyle = 'rgba(255,230,0,0.13)';
          ctx.fillRect(x + 3, y + 3, CELL - 6, CELL - 6);
        } else if (r.seen[i]) {
          // Frontier: yellow dots, so the BFS frontier reads differently from A*'s hollow cyan squares.
          ctx.fillStyle = C.yellow;
          ctx.beginPath(); ctx.arc(x + CELL / 2, y + CELL / 2, 2.5, 0, Math.PI * 2); ctx.fill();
        }
      }
    }
    if (r.path.length > 1) {
      ctx.save();
      ctx.strokeStyle = C.yellow; ctx.lineWidth = 2; ctx.setLineDash([6, 4]);
      ctx.beginPath();
      r.path.forEach((v, k) => {
        const x = (v % cols) * CELL + CELL / 2, y = ((v / cols) | 0) * CELL + CELL / 2;
        if (k) ctx.lineTo(x, y); else ctx.moveTo(x, y);
      });
      ctx.stroke();
      ctx.restore();
    }
    raceReadout();
  }

  function raceReadout() {
    const s = search, r = race;
    const set = (k, v) => { racePanel.querySelector(`[data-k="${k}"]`).textContent = v; };
    const steps = (st) => (st.path.length ? String(st.path.length - 1) : (st.done ? 'no route' : '…'));
    set('ae', s.expanded.toLocaleString()); set('ap', steps(s));
    set('be', r.expanded.toLocaleString()); set('bp', steps(r));
    let note = 'Both start at the green diamond and race to the yellow cross.';
    if (s.done && r.done && s.path.length && r.path.length) {
      note = r.expanded > s.expanded
        ? `A* expanded ${(r.expanded / s.expanded).toFixed(1)}x fewer cells than BFS`
        : 'BFS expanded fewer cells this time';
    }
    set('note', note);
  }

  function raceStart() {
    racePanel = document.createElement('div');
    racePanel.className = 'od-race';
    racePanel.setAttribute('aria-hidden', 'true');
    racePanel.innerHTML = '<div class="od-race-col a"><span class="od-race-name">A* // Manhattan</span>'
      + '<span>expanded <b data-k="ae">0</b></span><span>steps <b data-k="ap">0</b></span></div>'
      + '<div class="od-race-col b"><span class="od-race-name">BFS // no heuristic</span>'
      + '<span>expanded <b data-k="be">0</b></span><span>steps <b data-k="bp">0</b></span></div>'
      + '<div class="od-race-note" data-k="note"></div>';
    hero.appendChild(racePanel);
    race = raceState();
    if (REDUCED) frame();
  }

  function raceStop() {
    race = null;
    racePanel?.remove();
    racePanel = null;
    if (REDUCED) frame();
  }

  function frame() {
    t++;
    step(REDUCED ? Infinity : 30);
    if (race) raceStep(REDUCED ? Infinity : 30);
    draw();
    if (visible && !REDUCED) raf = requestAnimationFrame(frame);
  }

  function cellAt(e) {
    const rect = canvas.getBoundingClientRect();
    const c = Math.floor((e.clientX - rect.left) / CELL), r = Math.floor((e.clientY - rect.top) / CELL);
    return c >= 0 && r >= 0 && c < cols && r < rows ? idx(c, r) : -1;
  }

  hero.addEventListener('pointermove', (e) => {
    const i = cellAt(e);
    if (i < 0 || i === goal || walls[i] || i === start) return;
    goal = i; restart();
    if (REDUCED) frame();
  });
  canvas.addEventListener('click', (e) => {
    const i = cellAt(e);
    if (i < 0 || i === start || i === goal) return;
    walls[i] = walls[i] ? 0 : 3;
    restart();
    if (REDUCED) frame();
  });
  document.getElementById('city-reset').addEventListener('click', () => { walls.fill(0); restart(); if (REDUCED) frame(); });

  onOverdrive((on) => (on ? raceStart() : raceStop()));

  let resizeTimer;
  window.addEventListener('resize', () => { clearTimeout(resizeTimer); resizeTimer = setTimeout(() => { layout(); frame(); }, 150); });
  onVisible(hero, (v) => { visible = v; cancelAnimationFrame(raf); if (v) frame(); });
  layout();
  frame();
}

// ── Cabinet simulations ─────────────────────────────────────────────────

const rand = (n) => Math.floor(Math.random() * n);

function npuzzleSim(ctx, S) {
  // 3x3 sliding puzzle: scramble, solve with A* (Manhattan), animate the moves.
  const goal = [0, 1, 2, 3, 4, 5, 6, 7, 8];
  const man = (b) => b.reduce((d, t, i) => d + (t ? Math.abs((i / 3 | 0) - (t / 3 | 0)) + Math.abs(i % 3 - t % 3) : 0), 0);
  const nbrs = (b) => { const z = b.indexOf(0), out = [];
    for (const [dr, dc] of [[-1, 0], [1, 0], [0, -1], [0, 1]]) {
      const r = (z / 3 | 0) + dr, c = z % 3 + dc;
      if (r >= 0 && r < 3 && c >= 0 && c < 3) { const nb = b.slice(); nb[z] = nb[r * 3 + c]; nb[r * 3 + c] = 0; out.push(nb); } }
    return out; };
  function solve(start) {
    const key = (b) => b.join(''), heap = new MinHeap(), seen = new Map([[key(start), null]]);
    heap.push([man(start), start, 0]);
    while (heap.size) {
      const [, b, g] = heap.pop();
      if (key(b) === key(goal)) { const path = [b]; let k = seen.get(key(b));
        while (k) { path.push(k); k = seen.get(key(k)); } return path.reverse(); }
      for (const nb of nbrs(b)) if (!seen.has(key(nb))) { seen.set(key(nb), b); heap.push([g + 1 + man(nb), nb, g + 1]); }
    }
    return [start];
  }
  let path = [], k = 0, hold = 0;
  function reset() { let b = goal.slice(); for (let i = 0; i < 24; i++) { const n = nbrs(b); b = n[rand(n.length)]; } path = solve(b); k = 0; hold = 0; }
  reset();
  return (speed) => {
    const b = path[Math.min(k, path.length - 1)], cell = S / 3;
    ctx.fillStyle = C.bg; ctx.fillRect(0, 0, S, S);
    b.forEach((t, i) => {
      if (!t) return;
      const x = (i % 3) * cell, y = (i / 3 | 0) * cell, home = t === i;
      ctx.fillStyle = home ? 'rgba(0,255,136,0.18)' : 'rgba(255,0,160,0.18)';
      ctx.fillRect(x + 6, y + 6, cell - 12, cell - 12);
      ctx.strokeStyle = home ? C.green : C.pink; ctx.lineWidth = 2;
      ctx.strokeRect(x + 6, y + 6, cell - 12, cell - 12);
      ctx.fillStyle = home ? C.green : C.pink; ctx.font = `900 ${cell * 0.42}px Orbitron, sans-serif`;
      ctx.textAlign = 'center'; ctx.textBaseline = 'middle'; ctx.fillText(t, x + cell / 2, y + cell / 2 + 2);
    });
    if (k < path.length - 1) { if ((hold += speed) >= 12) { hold = 0; k++; } }
    else if ((hold += speed) > 90) reset();
  };
}

function connect4Sim(ctx, S) {
  // Pieces drop with gravity; each side takes a win, blocks a win, or prefers the centre.
  const R = 6, Cn = 7;
  let grid, turn, falling, winner, hold;
  const lines = () => { const out = [];
    for (let r = 0; r < R; r++) for (let c = 0; c < Cn; c++)
      for (const [dr, dc] of [[0, 1], [1, 0], [1, 1], [1, -1]]) {
        const cells = [0, 1, 2, 3].map((k) => [r + dr * k, c + dc * k]);
        if (cells.every(([rr, cc]) => rr >= 0 && rr < R && cc >= 0 && cc < Cn)) out.push(cells); }
    return out; };
  const LINES = lines();
  const findWin = (p) => LINES.find((l) => l.every(([r, c]) => grid[r][c] === p));
  const dropRow = (c) => { for (let r = R - 1; r >= 0; r--) if (!grid[r][c]) return r; return -1; };
  function choose(p) {
    const legal = [...Array(Cn).keys()].filter((c) => dropRow(c) >= 0);
    for (const who of [p, 3 - p]) for (const c of legal) {
      const r = dropRow(c); grid[r][c] = who; const w = findWin(who); grid[r][c] = 0; if (w) return c; }
    return legal.sort((a, b) => Math.abs(a - 3) - Math.abs(b - 3) + (Math.random() - 0.5) * 3)[0];
  }
  function reset() { grid = [...Array(R)].map(() => Array(Cn).fill(0)); turn = 1; falling = null; winner = null; hold = 0; }
  reset();
  return (speed) => {
    const cell = S / Cn, top = (S - cell * R) / 2;
    ctx.fillStyle = C.bg; ctx.fillRect(0, 0, S, S);
    for (let r = 0; r < R; r++) for (let c = 0; c < Cn; c++) {
      const v = grid[r][c], x = c * cell + cell / 2, y = top + r * cell + cell / 2;
      ctx.beginPath(); ctx.arc(x, y, cell * 0.38, 0, Math.PI * 2);
      ctx.fillStyle = v === 1 ? C.pink : v === 2 ? C.yellow : 'rgba(0,245,255,0.07)'; ctx.fill();
      if (winner && winner.some(([rr, cc]) => rr === r && cc === c)) { ctx.strokeStyle = C.green; ctx.lineWidth = 3; ctx.stroke(); }
    }
    if (falling) {
      falling.y += 0.9 * speed;
      const x = falling.c * cell + cell / 2, y = top + falling.y * cell + cell / 2;
      ctx.beginPath(); ctx.arc(x, y, cell * 0.38, 0, Math.PI * 2); ctx.fillStyle = turn === 1 ? C.pink : C.yellow; ctx.fill();
      if (falling.y >= falling.r) { grid[falling.r][falling.c] = turn; winner = findWin(turn) || null; falling = null; turn = 3 - turn; }
    } else if (winner || grid[0].every((v) => v)) { if ((hold += speed) > 70) reset(); }
    else if ((hold += speed) > 8) { hold = 0; const c = choose(turn); falling = { c, r: dropRow(c), y: -1 }; }
  };
}

function g2048Sim(ctx, S) {
  // 2048 played by a one-move lookahead on empty cells and a corner gradient.
  let b, hold;
  const slide = (row) => { const t = row.filter(Boolean), out = [];
    for (let i = 0; i < t.length; i++) { if (t[i] === t[i + 1]) { out.push(t[i] * 2); i++; } else out.push(t[i]); }
    while (out.length < 4) out.push(0); return out; };
  const move = (g, d) => { const n = g.map((r) => r.slice());
    for (let k = 0; k < 4; k++) {
      let line = d === 0 ? n.map((r) => r[k]) : d === 1 ? n.map((r) => r[k]).reverse() : d === 2 ? n[k] : n[k].slice().reverse();
      line = slide(line); if (d === 1 || d === 3) line.reverse();
      for (let i = 0; i < 4; i++) { if (d < 2) n[i][k] = line[i]; else n[k][i] = line[i]; } }
    return n; };
  const same = (a, c) => a.every((r, i) => r.every((v, j) => v === c[i][j]));
  const spawn = (g) => { const e = []; g.forEach((r, i) => r.forEach((v, j) => { if (!v) e.push([i, j]); }));
    if (e.length) { const [i, j] = e[rand(e.length)]; g[i][j] = Math.random() < 0.9 ? 2 : 4; } };
  const score = (g) => { let s = 0; g.forEach((r, i) => r.forEach((v, j) => { if (!v) s += 40; else s += v * (i + 1) * (4 - j); })); return s; };
  function reset() { b = [...Array(4)].map(() => Array(4).fill(0)); spawn(b); spawn(b); hold = 0; }
  reset();
  return (speed) => {
    const cell = S / 4;
    ctx.fillStyle = C.bg; ctx.fillRect(0, 0, S, S);
    b.forEach((r, i) => r.forEach((v, j) => {
      const x = j * cell, y = i * cell;
      if (!v) { ctx.fillStyle = 'rgba(155,0,255,0.08)'; ctx.fillRect(x + 4, y + 4, cell - 8, cell - 8); return; }
      const k = Math.log2(v), hue = 280 - Math.min(k, 13) * 18;
      ctx.fillStyle = `hsla(${hue},100%,55%,${0.18 + k * 0.05})`; ctx.fillRect(x + 4, y + 4, cell - 8, cell - 8);
      ctx.fillStyle = `hsl(${hue},100%,75%)`; ctx.font = `900 ${v < 100 ? cell * 0.36 : v < 1000 ? cell * 0.28 : cell * 0.22}px Orbitron, sans-serif`;
      ctx.textAlign = 'center'; ctx.textBaseline = 'middle'; ctx.fillText(v, x + cell / 2, y + cell / 2 + 2);
    }));
    if ((hold += speed) < 7) return;
    hold = 0;
    const options = [0, 1, 2, 3].map((d) => move(b, d)).filter((n) => !same(n, b));
    if (!options.length) { reset(); return; }
    b = options.sort((p, q) => score(q) - score(p))[0];
    spawn(b);
  };
}

function sudokuSim(ctx, S) {
  // Fill cells in minimum-remaining-values order; forced cells flash green, choices flash magenta.
  const PUZ = '009000000160004023000009061007000000000100085006857004650000000000008000703920000';
  const SOL = '329615478165784923874239561587342619432196785916857234658473192291568347743921856';
  let g, flash, hold;
  const peers = (i) => { const r = i / 9 | 0, c = i % 9, br = r - r % 3, bc = c - c % 3, out = new Set();
    for (let k = 0; k < 9; k++) { out.add(r * 9 + k); out.add(k * 9 + c); out.add((br + (k / 3 | 0)) * 9 + bc + k % 3); }
    out.delete(i); return out; };
  const PEERS = [...Array(81)].map((_, i) => peers(i));
  const options = (i) => { const used = new Set([...PEERS[i]].map((j) => g[j])); return [1, 2, 3, 4, 5, 6, 7, 8, 9].filter((d) => !used.has(d)).length; };
  function reset() { g = [...PUZ].map(Number); flash = new Map(); hold = 0; }
  reset();
  return (speed) => {
    const cell = S / 9;
    ctx.fillStyle = C.bg; ctx.fillRect(0, 0, S, S);
    for (let i = 0; i < 81; i++) {
      const x = (i % 9) * cell, y = (i / 9 | 0) * cell, f = flash.get(i);
      if (f) { ctx.fillStyle = f.forced ? `rgba(0,255,136,${f.a})` : `rgba(255,0,160,${f.a})`; ctx.fillRect(x, y, cell, cell); f.a *= 0.93; }
      if (g[i]) { ctx.fillStyle = PUZ[i] !== '0' ? '#c0e8ff' : C.cyan; ctx.font = `700 ${cell * 0.6}px Orbitron, sans-serif`;
        ctx.textAlign = 'center'; ctx.textBaseline = 'middle'; ctx.fillText(g[i], x + cell / 2, y + cell / 2 + 1); }
    }
    ctx.strokeStyle = 'rgba(0,255,136,0.5)';
    for (let k = 0; k <= 9; k++) { ctx.lineWidth = k % 3 ? 0.5 : 2;
      ctx.beginPath(); ctx.moveTo(k * cell, 0); ctx.lineTo(k * cell, S); ctx.moveTo(0, k * cell); ctx.lineTo(S, k * cell); ctx.stroke(); }
    if ((hold += speed) < 5) return;
    hold = 0;
    const empty = [...Array(81).keys()].filter((i) => !g[i]);
    if (!empty.length) { if (flash.size > 1) flash.clear(); else reset(); hold = -80; return; }
    const pick = empty.reduce((best, i) => (options(i) < options(best) ? i : best), empty[0]);
    flash.set(pick, { a: 0.75, forced: options(pick) === 1 });
    g[pick] = Number(SOL[pick]);
  };
}

function checkersSim(ctx, S) {
  // Two sides play random legal moves (forced single captures, crowning) on an 8x8 board.
  let b, side, hold, lastTo;
  const reset = () => { b = [...Array(64)].map((_, i) => { const r = i >> 3, c = i & 7;
    return (r + c) % 2 ? (r < 3 ? -1 : r > 4 ? 1 : 0) : 0; }); side = 1; hold = 0; lastTo = -1; };
  const moves = () => { const caps = [], steps = [];
    for (let i = 0; i < 64; i++) { const p = b[i]; if (p * side <= 0) continue;
      const r = i >> 3, c = i & 7, dirs = Math.abs(p) === 2 ? [-1, 1] : [-side];
      for (const dr of dirs) for (const dc of [-1, 1]) {
        const r1 = r + dr, c1 = c + dc, r2 = r + 2 * dr, c2 = c + 2 * dc;
        if (r2 >= 0 && r2 < 8 && c2 >= 0 && c2 < 8 && b[r1 * 8 + c1] * side < 0 && !b[r2 * 8 + c2]) caps.push([i, r2 * 8 + c2, r1 * 8 + c1]);
        if (r1 >= 0 && r1 < 8 && c1 >= 0 && c1 < 8 && !b[r1 * 8 + c1]) steps.push([i, r1 * 8 + c1, -1]); } }
    return caps.length ? caps : steps; };
  reset();
  return (speed) => {
    const cell = S / 8;
    ctx.fillStyle = C.bg; ctx.fillRect(0, 0, S, S);
    for (let i = 0; i < 64; i++) {
      const r = i >> 3, c = i & 7, x = c * cell, y = r * cell;
      ctx.fillStyle = (r + c) % 2 ? (i === lastTo ? 'rgba(0,245,255,0.25)' : 'rgba(255,0,160,0.10)') : 'rgba(155,0,255,0.05)';
      ctx.fillRect(x, y, cell, cell);
      const p = b[i]; if (!p) continue;
      ctx.beginPath(); ctx.arc(x + cell / 2, y + cell / 2, cell * 0.36, 0, Math.PI * 2);
      ctx.fillStyle = p > 0 ? C.pink : '#9feeff'; ctx.shadowColor = ctx.fillStyle; ctx.shadowBlur = 8; ctx.fill(); ctx.shadowBlur = 0;
      if (Math.abs(p) === 2) { ctx.fillStyle = C.yellow; ctx.fillRect(x + cell / 2 - 3, y + cell / 2 - 3, 6, 6); }
    }
    if ((hold += speed) < 10) return;
    hold = 0;
    const ms = moves();
    if (!ms.length || b.filter((p) => p).length < 4) { reset(); return; }
    const [from, to, cap] = ms[rand(ms.length)];
    b[to] = b[from]; b[from] = 0; if (cap >= 0) b[cap] = 0;
    if ((to >> 3) === (side > 0 ? 0 : 7)) b[to] = 2 * side;
    lastTo = to; side = -side;
  };
}

function routesSim(ctx, S) {
  // Simulated annealing on 18 random cities: random segment reversals,
  // accepting worse routes with probability exp(-delta / T) as T cools.
  let pts, tour, T, hold;
  const d = (a, b) => Math.hypot(pts[a][0] - pts[b][0], pts[a][1] - pts[b][1]);
  const reset = () => {
    pts = Array.from({ length: 18 }, () => [0.08 + Math.random() * 0.84, 0.08 + Math.random() * 0.84]);
    tour = [...pts.keys()].sort(() => Math.random() - 0.5); T = 0.3; hold = 0;
  };
  reset();
  return (speed) => {
    for (let k = 0; k < 40 * speed; k++) {
      const n = tour.length, i = 1 + rand(n - 2), j = i + 1 + rand(n - 1 - i);
      const a = tour[i - 1], b = tour[i], c = tour[j], e = tour[(j + 1) % n];
      const delta = d(a, c) + d(b, e) - d(a, b) - d(c, e);
      if (delta < 0 || Math.random() < Math.exp(-delta / T)) tour.splice(i, j - i + 1, ...tour.slice(i, j + 1).reverse());
      T *= 0.9995;
    }
    ctx.fillStyle = C.bg; ctx.fillRect(0, 0, S, S);
    ctx.save(); ctx.strokeStyle = C.green; ctx.shadowColor = C.green; ctx.shadowBlur = 10; ctx.lineWidth = 2;
    ctx.beginPath(); tour.forEach((i, k) => (k ? ctx.lineTo(pts[i][0] * S, pts[i][1] * S) : ctx.moveTo(pts[i][0] * S, pts[i][1] * S)));
    ctx.closePath(); ctx.stroke(); ctx.restore();
    ctx.fillStyle = C.yellow; pts.forEach(([x, y]) => ctx.fillRect(x * S - 3, y * S - 3, 6, 6));
    if (T < 0.002 && (hold += speed) > 120) reset();
  };
}

function lightsoutSim(ctx, S) {
  // 5x5 Lights Out. Deal a random solvable board by pressing random cells on the dark board, solve it with
  // Gauss-Jordan elimination over GF(2) (the same method as lightsout/solver.py), then replay the presses.
  const n = 5, N = 25, cell = S / n;
  // Bit mask of what pressing cell i toggles: the cell and its orthogonal neighbours.
  const toggles = [...Array(N)].map((_, i) => {
    const r = (i / n) | 0, c = i % n;
    let m = 1 << i;
    for (const [dr, dc] of [[-1, 0], [1, 0], [0, -1], [0, 1]]) {
      const rr = r + dr, cc = c + dc;
      if (rr >= 0 && rr < n && cc >= 0 && cc < n) m |= 1 << (rr * n + cc);
    }
    return m;
  });
  // Cells to press that clear `board`. Row i holds toggles[i] plus the lit bit of cell i at bit N.
  function presses(board) {
    const rows = toggles.map((t, i) => t | (((board >> i) & 1) << N));
    const pivots = [];
    let rank = 0;
    for (let c = 0; c < N; c++) {
      let f = rank;
      while (f < N && !((rows[f] >> c) & 1)) f++;
      if (f === N) continue; // no pivot here: this press is free, so leave it unpressed
      [rows[rank], rows[f]] = [rows[f], rows[rank]];
      for (let r = 0; r < N; r++) if (r !== rank && ((rows[r] >> c) & 1)) rows[r] ^= rows[rank];
      pivots.push(c);
      rank++;
    }
    let x = 0; // free presses stay 0, so each pivot press equals its row's right-hand side
    for (let r = 0; r < rank; r++) if ((rows[r] >> N) & 1) x |= 1 << pivots[r];
    return [...Array(N).keys()].filter((i) => (x >> i) & 1);
  }
  let board = 0, plan = [], hold = 0;
  function deal() {
    do {
      let x = 0;
      for (let i = 0; i < N; i++) if (Math.random() < 0.4) x ^= 1 << i;
      board = 0;
      for (let i = 0; i < N; i++) if ((x >> i) & 1) board ^= toggles[i];
    } while (!board);
    plan = presses(board);
    hold = 0;
  }
  deal();
  return (speed) => {
    ctx.fillStyle = C.bg;
    ctx.fillRect(0, 0, S, S);
    const next = plan[0];
    for (let i = 0; i < N; i++) {
      const x = (i % n) * cell, y = ((i / n) | 0) * cell, on = (board >> i) & 1;
      ctx.fillStyle = on ? C.cyan : 'rgba(0,245,255,0.06)';
      ctx.shadowColor = C.cyan;
      ctx.shadowBlur = on ? 14 : 0;
      ctx.fillRect(x + 5, y + 5, cell - 10, cell - 10);
      ctx.shadowBlur = 0;
      if (i === next) { ctx.strokeStyle = C.yellow; ctx.lineWidth = 2; ctx.strokeRect(x + 3, y + 3, cell - 6, cell - 6); }
    }
    if (!plan.length) { if ((hold += speed) > 140) deal(); return; }
    if ((hold += speed) < 16) return;
    hold = 0;
    board ^= toggles[plan.shift()]; // one press: its cell and neighbours flip
  };
}

function blackjackSim(ctx, S) {
  // Hard totals 8-17 against upcards 2-A, from the exact table. A cursor sweeps one upcard column at a time.
  const TABLE = ['HHHHHHHHHH', 'HDDDDHHHHH', 'DDDDDDDDHH', 'DDDDDDDDDH', 'HHSSSHHHHH',
                 'SSSSSHHHHH', 'SSSSSHHHHH', 'SSSSSHHHHH', 'SSSSSHHHHH', 'SSSSSSSSSS'];
  const COL = { S: C.cyan, H: C.green, D: C.yellow, P: C.pink };
  const pad = 22, cw = (S - pad - 6) / 10, ch = (S - pad - 6) / TABLE.length;
  let t = 0;
  return (speed) => {
    t += 0.02 * speed;
    const col = Math.floor(t) % 10;
    ctx.clearRect(0, 0, S, S);
    ctx.font = '600 10px JetBrains Mono, monospace';
    TABLE.forEach((row, r) => {
      for (let c = 0; c < 10; c++) {
        const letter = row[c], x = pad + c * cw, y = 6 + r * ch;
        ctx.globalAlpha = c === col ? 1 : 0.35;
        ctx.fillStyle = COL[letter];
        ctx.shadowColor = COL[letter]; ctx.shadowBlur = c === col ? 12 : 0;
        ctx.fillRect(x + 1, y + 1, cw - 2, ch - 2);
        ctx.fillStyle = C.bg;
        ctx.fillText(letter, x + cw / 2 - 3, y + ch / 2 + 3);
      }
    });
    ctx.globalAlpha = 1; ctx.shadowBlur = 0;
  };
}

function battleshipSim(ctx, S) {
  // Cabinet screen: a random fleet being sunk. The shooter here is the hunt/target baseline (checkerboard
  // until a hit, then the neighbours of its wounded ships), which is cheap enough for the landing page; the
  // live Battleship page runs the Bayesian model. The yellow box marks its latest shot.
  const N = 10, LENS = [5, 4, 3, 3, 2];
  let ship, shot, next, hold;
  const adj = (i) => {
    const r = Math.floor(i / N), c = i % N;
    return [[r - 1, c], [r + 1, c], [r, c - 1], [r, c + 1]]
      .filter(([a, b]) => a >= 0 && a < N && b >= 0 && b < N).map(([a, b]) => a * N + b);
  };
  const place = () => {
    ship = Array(N * N).fill(-1); // -1 water, else the index of the ship on that cell
    shot = Array(N * N).fill(0); // 0 unknown, 1 miss, 2 hit
    LENS.forEach((len, k) => {
      for (let tries = 0; tries < 500; tries++) {
        const vert = Math.random() < 0.5;
        const r = rand(vert ? N - len + 1 : N), c = rand(vert ? N : N - len + 1);
        const cells = [...Array(len)].map((_, j) => (vert ? (r + j) * N + c : r * N + c + j));
        if (cells.every((i) => ship[i] < 0)) { cells.forEach((i) => (ship[i] = k)); return; }
      }
    });
    next = -1; hold = 0;
  };
  const pick = () => {
    const unknown = [...shot.keys()].filter((i) => !shot[i]);
    const wounded = unknown.filter((i) => adj(i).some((j) => shot[j] === 2));
    const parity = unknown.filter((i) => (Math.floor(i / N) + (i % N)) % 2 === 0);
    const pool = wounded.length ? wounded : parity.length ? parity : unknown;
    return pool[rand(pool.length)];
  };
  place();
  return (speed) => {
    const cell = S / N;
    ctx.fillStyle = C.bg; ctx.fillRect(0, 0, S, S);
    for (let i = 0; i < N * N; i++) {
      const x = (i % N) * cell, y = Math.floor(i / N) * cell;
      ctx.strokeStyle = 'rgba(0,245,255,0.12)'; ctx.strokeRect(x, y, cell, cell);
      if (shot[i] === 2) {
        ctx.strokeStyle = C.pink; ctx.shadowColor = C.pink; ctx.shadowBlur = 8; ctx.lineWidth = 2;
        ctx.beginPath(); ctx.moveTo(x + 6, y + 6); ctx.lineTo(x + cell - 6, y + cell - 6);
        ctx.moveTo(x + cell - 6, y + 6); ctx.lineTo(x + 6, y + cell - 6); ctx.stroke(); ctx.shadowBlur = 0;
      } else if (shot[i] === 1) {
        ctx.fillStyle = C.cyan; ctx.beginPath(); ctx.arc(x + cell / 2, y + cell / 2, 2.5, 0, Math.PI * 2); ctx.fill();
      }
    }
    if (next >= 0) {
      const x = (next % N) * cell, y = Math.floor(next / N) * cell;
      ctx.strokeStyle = C.yellow; ctx.shadowColor = C.yellow; ctx.shadowBlur = 10; ctx.lineWidth = 2;
      ctx.strokeRect(x + 3, y + 3, cell - 6, cell - 6); ctx.shadowBlur = 0;
    }
    if ((hold += speed) < 12) return;
    hold = 0;
    const afloat = ship.some((k, i) => k >= 0 && !shot[i]);
    if (!afloat) { place(); return; }
    next = pick();
    shot[next] = ship[next] >= 0 ? 2 : 1;
  };
}

function pacmanSim(ctx, S) {
  // A small maze: Pac-Man eats his way toward the nearest pellet, and each ghost follows a
  // shortest route (breadth-first search) toward him. The dashed lines are those routes.
  const MAZE = ['#########', '#...#...#', '#.#.#.#.#', '#.......#', '###.#.###', '#.......#', '#.#.#.#.#', '#...#...#', '#########'];
  const open = (r, c) => MAZE[r] && MAZE[r][c] !== undefined && MAZE[r][c] !== '#';
  const N = [[-1, 0], [0, 1], [1, 0], [0, -1]];
  let pellets, pac, ghosts, hold;
  const reset = () => {
    pellets = new Set();
    MAZE.forEach((row, r) => [...row].forEach((ch, c) => { if (ch === '.') pellets.add(r * 9 + c); }));
    pac = [3, 4]; ghosts = [[1, 1], [7, 7]]; hold = 0;
  };
  // Distance from `src` to every cell, then the neighbour of `at` that is one step closer to `src`.
  const distFrom = (src) => {
    const d = new Map([[src[0] * 9 + src[1], 0]]), q = [src];
    while (q.length) { const [r, c] = q.shift(); for (const [dr, dc] of N) {
      const k = (r + dr) * 9 + c + dc; if (open(r + dr, c + dc) && !d.has(k)) { d.set(k, d.get(r * 9 + c) + 1); q.push([r + dr, c + dc]); } } }
    return d;
  };
  const stepToward = (at, d) => {
    let best = at, bestD = d.get(at[0] * 9 + at[1]) ?? Infinity;
    for (const [dr, dc] of N) { const v = d.get((at[0] + dr) * 9 + at[1] + dc); if (open(at[0] + dr, at[1] + dc) && v < bestD) { best = [at[0] + dr, at[1] + dc]; bestD = v; } }
    return best;
  };
  reset();
  return (speed) => {
    const cell = S / 9;
    ctx.fillStyle = C.bg; ctx.fillRect(0, 0, S, S);
    ctx.strokeStyle = C.cyan; ctx.shadowColor = C.cyan; ctx.shadowBlur = 6; ctx.lineWidth = 2;
    MAZE.forEach((row, r) => [...row].forEach((ch, c) => { if (ch === '#') ctx.strokeRect(c * cell + 2, r * cell + 2, cell - 4, cell - 4); }));
    ctx.shadowBlur = 0; ctx.fillStyle = C.yellow;
    for (const k of pellets) { ctx.beginPath(); ctx.arc((k % 9 + 0.5) * cell, (Math.floor(k / 9) + 0.5) * cell, 2.5, 0, Math.PI * 2); ctx.fill(); }
    ghosts.forEach((g, i) => {
      ctx.strokeStyle = [C.pink, C.cyan][i]; ctx.setLineDash([4, 4]); ctx.shadowColor = ctx.strokeStyle; ctx.shadowBlur = 8;
      const d = distFrom(pac); ctx.beginPath(); let at = g; ctx.moveTo((at[1] + 0.5) * cell, (at[0] + 0.5) * cell);
      for (let k = 0; k < 20 && !(at[0] === pac[0] && at[1] === pac[1]); k++) { at = stepToward(at, d); ctx.lineTo((at[1] + 0.5) * cell, (at[0] + 0.5) * cell); }
      ctx.stroke(); ctx.setLineDash([]);
      ctx.fillStyle = [C.pink, C.cyan][i]; ctx.beginPath(); ctx.arc((g[1] + 0.5) * cell, (g[0] + 0.5) * cell, cell * 0.36, 0, Math.PI * 2); ctx.fill();
    });
    ctx.fillStyle = C.yellow; ctx.beginPath(); ctx.arc((pac[1] + 0.5) * cell, (pac[0] + 0.5) * cell, cell * 0.4, 0, Math.PI * 2); ctx.fill();
    if ((hold += speed) < 14) return;
    hold = 0;
    const key = pac[0] * 9 + pac[1];
    pellets.delete(key);
    const dp = distFrom(pac);
    if (pellets.size) { // head for the nearest pellet
      let target = null, best = Infinity;
      for (const k of pellets) { const v = dp.get(k); if (v < best) { best = v; target = k; } }
      pac = stepToward(pac, distFrom([Math.floor(target / 9), target % 9]));
    }
    ghosts = ghosts.map((g) => stepToward(g, distFrom(pac)));
    if (!pellets.size || ghosts.some((g) => g[0] === pac[0] && g[1] === pac[1])) reset();
  };
}

function warehouseSim(ctx, S) {
  // Replays precomputed Conflict-Based Search plans on small warehouses. Each robot glides along its
  // path; the plans were found by splitting on collisions, so no two robots share a cell or swap places.
  const PLANS = [
    { rows: ["...........", ".###...###.", ".###...###.", "...........", ".....#.....", "...........", ".###...###.", ".###...###.", "..........."], paths: [[82,71,60,59,58,57],[46,35,36,37,26,27],[92,81,70,71,60,61]] },
    { rows: [".............", ".###.###.###.", ".............", "#####.#######", ".............", ".###.###.###.", "............."], paths: [[8,21,34,33,32,31,30,30,31,44,57,56,55,54,53,52,65,78],[57,44,31,30,17,4],[84,83,82,69,56,57,44,31,32,33,34,35,36,37,38,25],[81,82,69,56,57,44,31,32,33,34,21,8,7]] },
  ];
  let k = 0, t = 0, prog = 0, hold = 0;
  const COLORS = ['#00f5ff', '#ff00a0', '#ffe600', '#00ff88'];
  return (speed) => {
    const plan = PLANS[k], W = plan.rows[0].length, H = plan.rows.length;
    const span = Math.max(...plan.paths.map((p) => p.length - 1));  // the plan's makespan
    const cell = Math.min(S / W, S / H), ox = (S - W * cell) / 2, oy = (S - H * cell) / 2;
    const at = (path, step) => path[Math.min(step, path.length - 1)];
    ctx.fillStyle = C.bg; ctx.fillRect(0, 0, S, S);
    // Shelves are neon blocks; floor cells stay dark.
    ctx.save(); ctx.fillStyle = 'rgba(155,0,255,0.45)'; ctx.shadowColor = '#9b00ff'; ctx.shadowBlur = 6;
    for (let y = 0; y < H; y++) for (let x = 0; x < W; x++)
      if (plan.rows[y][x] === '#') ctx.fillRect(ox + x * cell + 1, oy + y * cell + 1, cell - 2, cell - 2);
    ctx.restore();
    // Goals are hollow squares in each robot's colour.
    plan.paths.forEach((path, i) => {
      const g = path[path.length - 1], x = g % W, y = (g / W) | 0;
      ctx.strokeStyle = COLORS[i % COLORS.length]; ctx.lineWidth = 1.5;
      ctx.strokeRect(ox + x * cell + 3, oy + y * cell + 3, cell - 6, cell - 6);
    });
    // Robots sit between their cell at step t and step t+1, so motion looks continuous.
    plan.paths.forEach((path, i) => {
      const a = at(path, t), b = at(path, t + 1), col = COLORS[i % COLORS.length];
      const ax = a % W, ay = (a / W) | 0, bx = b % W, by = (b / W) | 0;
      const x = (ax + (bx - ax) * prog + 0.5) * cell + ox;
      const y = (ay + (by - ay) * prog + 0.5) * cell + oy;
      ctx.beginPath(); ctx.arc(x, y, cell * 0.33, 0, Math.PI * 2);
      ctx.fillStyle = col; ctx.shadowColor = col; ctx.shadowBlur = 10; ctx.fill(); ctx.shadowBlur = 0;
    });
    // About 8 frames per time step at speed 1. The final positions are held, then the next plan starts.
    if ((prog += speed / 8) < 1) return;
    prog = 0;
    if (t < span) { t++; return; }
    if ((hold += speed) < 90) return;
    hold = 0; t = 0; k = (k + 1) % PLANS.length;
  };
}

function endgameSim(ctx, S) {
  // King and queen against a lone king: the lone king's moves to mate on each square, Black to move, with the white
  // king on a1 and the queen on b2, read from the solved table (-1 = not a position). A cursor sweeps the board.
  const ROW = [-1,-1,4,4,5,7,6,7,-1,-1,6,8,8,8,8,8,4,6,6,8,8,8,8,8,4,8,8,8,10,10,10,9,5,8,8,10,10,10,10,9,7,8,8,10,10,10,10,9,6,8,8,10,10,10,9,9,7,8,8,9,9,9,9,7];
  const cell = S / 8;
  let k = 0, hold = 0;
  // Near mates are green, long defences pink.
  const colour = (v) => {
    const t = Math.min(1, Math.max(0, (v - 4) / 6));
    return `rgb(${Math.round(255 * t)},${Math.round(255 * (1 - t))},${Math.round(136 + 24 * t)})`;
  };
  return (speed) => {
    ctx.fillStyle = C.bg;
    ctx.fillRect(0, 0, S, S);
    ctx.font = `${Math.round(cell * 0.36)}px Orbitron, sans-serif`;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    for (let sq = 0; sq < 64; sq++) {
      const x = (sq & 7) * cell, y = (7 - (sq >> 3)) * cell;
      const v = ROW[sq];
      if (v > 0) {
        ctx.globalAlpha = sq === k ? 1 : 0.55;
        ctx.fillStyle = colour(v);
        ctx.fillRect(x + 2, y + 2, cell - 4, cell - 4);
        ctx.globalAlpha = 1;
        ctx.fillStyle = C.bg;
        ctx.fillText(String(v), x + cell / 2, y + cell / 2);
      } else {
        ctx.fillStyle = 'rgba(0,245,255,0.05)';
        ctx.fillRect(x + 2, y + 2, cell - 4, cell - 4);
      }
    }
    // The cursor: a yellow frame on the square being looked up.
    const cx = (k & 7) * cell, cy = (7 - (k >> 3)) * cell;
    ctx.strokeStyle = C.yellow;
    ctx.lineWidth = 2;
    ctx.strokeRect(cx + 3, cy + 3, cell - 6, cell - 6);
    // The fixed pieces: white king on a1, queen on b2.
    ctx.fillStyle = C.cyan;
    ctx.fillText('K', cell * 0.5, S - cell * 0.5);
    ctx.fillText('Q', cell * 1.5, S - cell * 1.5);
    if ((hold += speed) > 26) {
      hold = 0;
      k = (k + 1) % 64;
    }
  };
}

function sokobanSim(ctx, S) {
  // Pillar level from the Sokoban set, with its 3-push solution fixed in advance: the cabinet replays it on a
  // small grid with no search in the browser. The loop holds for a moment at the end, then restarts.
  const TEXT = ['########', '#  .   #', '# #$#  #', '#  $ . #', '#  @   #', '#      #', '########'];
  const H = TEXT.length, W = TEXT[0].length, cell = S / W;
  const oy = (S - H * cell) / 2;
  const MOVES = 'awDWsD';
  const DIR = { w: [-1, 0], s: [1, 0], a: [0, -1], d: [0, 1] };
  const at = (r, c) => TEXT[r][c];
  const goalAt = (r, c) => at(r, c) === '.';
  let pr = 0, pc = 0, boxes = [], step = 0, hold = 0;
  function restart() {
    boxes = [];
    TEXT.forEach((row, r) => [...row].forEach((ch, c) => {
      if (ch === '@') { pr = r; pc = c; }
      if (ch === '$') boxes.push([r, c]);
    }));
    step = 0;
    hold = 0;
  }
  restart();
  const isBox = (r, c) => boxes.some(([br, bc]) => br === r && bc === c);
  const isWall = (r, c) => at(r, c) === '#';
  function apply(letter) {
    const [dr, dc] = DIR[letter.toLowerCase()];
    const tr = pr + dr, tc = pc + dc;
    if (isBox(tr, tc)) {
      const br = tr + dr, bc = tc + dc;
      const i = boxes.findIndex(([x, y]) => x === tr && y === tc);
      boxes[i] = [br, bc];
    }
    pr = tr; pc = tc;
  }
  return (speed) => {
    ctx.fillStyle = C.bg;
    ctx.fillRect(0, 0, S, S);
    for (let r = 0; r < H; r++) for (let c = 0; c < W; c++) {
      const x = c * cell, y = oy + r * cell;
      if (isWall(r, c)) {
        ctx.fillStyle = 'rgba(0,245,255,0.16)';
        ctx.fillRect(x + 1, y + 1, cell - 2, cell - 2);
        continue;
      }
      if (goalAt(r, c)) {
        ctx.strokeStyle = C.pink; ctx.lineWidth = 2;
        ctx.strokeRect(x + cell * 0.28, y + cell * 0.28, cell * 0.44, cell * 0.44);
      }
    }
    for (const [r, c] of boxes) {
      const x = c * cell, y = oy + r * cell, onGoal = goalAt(r, c);
      ctx.fillStyle = onGoal ? C.green : C.yellow;
      ctx.shadowColor = onGoal ? C.green : C.yellow;
      ctx.shadowBlur = 12;
      ctx.fillRect(x + 4, y + 4, cell - 8, cell - 8);
      ctx.shadowBlur = 0;
    }
    ctx.fillStyle = C.cyan;
    ctx.shadowColor = C.cyan;
    ctx.shadowBlur = 14;
    ctx.beginPath();
    ctx.arc(pc * cell + cell / 2, oy + pr * cell + cell / 2, cell * 0.22, 0, Math.PI * 2);
    ctx.fill();
    ctx.shadowBlur = 0;
    if (step >= MOVES.length) { if ((hold += speed) > 160) restart(); return; }
    if ((hold += speed) < 40) return;
    hold = 0;
    apply(MOVES[step++]);
  };
}

function wordleSim(ctx, S) {
  // The solver on one answer (ABIDE). First the bars: the expected bits of the best openers, as the solver ranks
  // them (python -m wordle benchmark). Then its three guesses, tile by tile, with the feedback each one got
  // (feedback.score). The numbers are fixed, so this sim does no scoring and makes no server calls.
  const OPENERS = [['TARES', 6.23], ['LARES', 6.17], ['RALES', 6.15], ['TALES', 6.14], ['RATES', 6.12]];
  const GUESSES = [['TARES', '.y.y.'], ['ABODE', 'gg.gg'], ['ABIDE', 'ggggg']];
  const TILE = 30, GAP = 6, X0 = (S - (5 * TILE + 4 * GAP)) / 2, Y0 = 124;
  const FILL = { g: C.green, y: C.yellow, '.': '#1b2638' };
  const BARS_T = 70, TILE_T = 14, HOLD_T = 160; // ticks for the bars to grow, between tiles, and to hold the result
  let t = 0, phase = 0;

  function drawTile(x, y, letter, fill, lit) {
    ctx.fillStyle = fill;
    ctx.shadowColor = fill;
    ctx.shadowBlur = lit ? 12 : 0;
    ctx.fillRect(x, y, TILE, TILE);
    ctx.shadowBlur = 0;
    ctx.strokeStyle = 'rgba(0,245,255,0.25)';
    ctx.strokeRect(x + 0.5, y + 0.5, TILE - 1, TILE - 1);
    if (letter) {
      ctx.fillStyle = '#fff';
      ctx.font = 'bold 16px monospace';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText(letter, x + TILE / 2, y + TILE / 2 + 1);
    }
  }

  return (speed) => {
    t += speed;
    ctx.fillStyle = C.bg;
    ctx.fillRect(0, 0, S, S);
    ctx.font = '10px monospace';
    ctx.textAlign = 'left';
    ctx.textBaseline = 'top';
    if (phase === 0) {
      // Bar length is expected bits above 5 bits, so small differences between openers still show.
      const grow = Math.min(1, t / BARS_T);
      ctx.fillStyle = C.yellow;
      ctx.fillText('expected bits', 8, 8);
      OPENERS.forEach(([word, bits], i) => {
        const h = ((bits - 5) / 1.5) * 84 * grow, x = 22 + i * 42, base = 102;
        ctx.fillStyle = i === 0 ? C.green : C.cyan;
        ctx.shadowColor = ctx.fillStyle;
        ctx.shadowBlur = i === 0 ? 14 : 6;
        ctx.fillRect(x, base - h, 26, h);
        ctx.shadowBlur = 0;
        ctx.fillStyle = C.dim;
        ctx.textAlign = 'center';
        ctx.fillText(word, x + 13, base + 6);
        ctx.textAlign = 'left';
      });
      if (t >= BARS_T + 60) { phase = 1; t = 0; }
    } else {
      // Tiles reveal one at a time; each guess is scored before the next is typed.
      const revealed = Math.floor(t / TILE_T);
      ctx.fillStyle = phase === 2 ? C.green : C.pink;
      ctx.fillText(phase === 2 ? 'solved: ABIDE' : 'feedback', 8, 8);
      for (let r = 0; r < GUESSES.length; r++) {
        const [word, fb] = GUESSES[r];
        for (let c = 0; c < 5; c++) {
          const k = r * 5 + c, x = X0 + c * (TILE + GAP), y = Y0 + r * (TILE + GAP);
          if (k < revealed) drawTile(x, y, word[c].toUpperCase(), FILL[fb[c]], true);
          else drawTile(x, y, '', '#0a1322', false);
        }
      }
      if (phase === 1 && revealed >= 15) { phase = 2; t = 0; }
      if (phase === 2 && t > HOLD_T) { phase = 0; t = 0; }
    }
  };
}

// Poker: the bot's opening bet rate for each card, from the committed Leduc strategy (poker/data/leduc_strategy.json).
// Deals a card, samples the bot's move from its mix (as the server does), and stamps the result.
function pokerSim(ctx, S) {
  const MIX = { J: 0.0724, Q: 0.7419, K: 0.752 }; // round 1, first to act: P(bet)
  const RANKS = ['J', 'Q', 'K'];
  const bars = { J: 0, Q: 0, K: 0 }; // animated bar lengths, 0..1
  let card = 'K', hold = 0, phase = 0, move = '', moveAt = 0;
  const pickCard = () => RANKS[rand(3)];
  function drawCard(x, y, w, h, label, face) {
    ctx.fillStyle = face ? '#ffe8f6' : 'rgba(155,0,255,0.25)';
    ctx.strokeStyle = face ? C.pink : C.purple;
    ctx.lineWidth = 2;
    ctx.shadowColor = face ? C.pink : C.purple;
    ctx.shadowBlur = 10;
    ctx.fillRect(x, y, w, h);
    ctx.strokeRect(x, y, w, h);
    ctx.shadowBlur = 0;
    if (face) {
      ctx.fillStyle = '#14021f';
      ctx.font = '900 26px Orbitron, sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText(label, x + w / 2, y + h / 2 + 9);
    }
  }
  return (speed) => {
    ctx.fillStyle = C.bg;
    ctx.fillRect(0, 0, S, S);
    ctx.textAlign = 'left';
    ctx.font = '10px "JetBrains Mono", monospace';
    ctx.fillStyle = C.dim;
    ctx.fillText('BOT OPENING BET RATE', 10, 16);
    // Three bars, one per card: the bet frequency that the bot's strategy assigns to that card.
    RANKS.forEach((r, i) => {
      const y = 32 + i * 34;
      bars[r] += (MIX[r] - bars[r]) * 0.06;
      ctx.fillStyle = C.dim;
      ctx.fillText(r, 10, y + 12);
      ctx.fillStyle = 'rgba(0,245,255,0.08)';
      ctx.fillRect(28, y, 150, 14);
      ctx.fillStyle = r === card ? C.yellow : C.pink;
      ctx.shadowColor = ctx.fillStyle;
      ctx.shadowBlur = 8;
      ctx.fillRect(28, y, 150 * bars[r], 14);
      ctx.shadowBlur = 0;
      ctx.fillStyle = C.cyan;
      ctx.fillText(`${Math.round(MIX[r] * 100)}%`, 186, y + 12);
    });
    // Each cycle: show a card, sample the bot's move from its mix, and stamp the result.
    hold += speed;
    if (phase === 0 && hold > 120) { card = pickCard(); hold = 0; phase = 1; }
    if (phase === 1 && hold > 60) {
      move = Math.random() < MIX[card] ? 'BET' : 'CHECK';
      moveAt = hold;
      hold = 0;
      phase = 2;
    }
    if (phase === 2 && hold > 150) { phase = 0; hold = 0; }
    drawCard(30, 150, 60, 84, card, phase > 0);
    ctx.fillStyle = C.cyan;
    ctx.font = '11px "JetBrains Mono", monospace';
    ctx.fillText('the bot holds', 100, 170);
    ctx.fillStyle = phase === 2 ? C.green : C.dim;
    ctx.font = '900 18px Orbitron, sans-serif';
    ctx.fillText(phase === 2 ? move : '...', 100, 200);
    ctx.font = '10px "JetBrains Mono", monospace';
    ctx.fillStyle = C.dim;
    ctx.fillText(moveAt ? 'a sample from its mix' : 'dealing', 100, 218);
  };
}

function minesweeperSim(ctx, S) {
  // 9x9 Minesweeper, the first click cleared around it, 10 mines. Each beat the agent reveals a cell that a
  // number proves safe, or guesses a covered cell at random when nothing is proven. The cabinet runs the
  // single-cell rules only, with no server calls. The full counting agent is on the minesweeper page.
  const n = 9, N = 81, cell = S / n, MINES = 10;
  const NUM = ['', C.cyan, C.green, C.pink, C.purple, C.yellow, '#7ff7ff', '#ffffff', '#ff3b3b'];
  // Neighbour list per cell: up to eight cells around it, inside the grid.
  const nb = [...Array(N)].map((_, i) => {
    const r = (i / n) | 0, c = i % n, out = [];
    for (let dr = -1; dr <= 1; dr++) {
      for (let dc = -1; dc <= 1; dc++) {
        if ((dr || dc) && r + dr >= 0 && r + dr < n && c + dc >= 0 && c + dc < n) out.push((r + dr) * n + c + dc);
      }
    }
    return out;
  });
  let mine, val, known, last, kind, hold, done;
  const count = (i) => nb[i].filter((j) => mine.has(j)).length;
  // Reveal i. A zero opens its neighbours, and zeros spread the opening.
  function open(i) {
    const q = [i];
    val[i] = count(i);
    for (let h = 0; h < q.length; h++) {
      const x = q[h];
      if (val[x] !== 0) continue;
      for (const j of nb[x]) {
        if (val[j] < 0) { val[j] = count(j); q.push(j); }
      }
    }
  }
  function reset() {
    mine = new Set(); val = Array(N).fill(-1); known = new Set(); kind = ''; hold = 0; done = false;
    const first = rand(N), clear = new Set([first, ...nb[first]]);
    const pool = [...Array(N).keys()].filter((i) => !clear.has(i));
    for (let k = 0; k < MINES; k++) {
      const j = k + rand(pool.length - k);
      [pool[k], pool[j]] = [pool[j], pool[k]];
      mine.add(pool[k]);
    }
    open(first);
    last = first;
  }
  // The single-cell rules over the numbers on screen. Returns the cells proven safe, and marks proven mines.
  function proven() {
    const safe = [];
    for (let i = 0; i < N; i++) {
      if (val[i] <= 0) continue;
      const cov = nb[i].filter((j) => val[j] < 0);
      const open_ = cov.filter((j) => !known.has(j));
      if (!open_.length) continue;
      const need = val[i] - cov.filter((j) => known.has(j)).length;
      if (need === 0) safe.push(...open_);
      else if (need === open_.length) open_.forEach((j) => known.add(j));
    }
    return safe;
  }
  reset();
  return (speed) => {
    ctx.fillStyle = C.bg;
    ctx.fillRect(0, 0, S, S);
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    for (let i = 0; i < N; i++) {
      const x = (i % n) * cell, y = ((i / n) | 0) * cell, v = val[i];
      ctx.fillStyle = v >= 0 ? 'rgba(3,8,18,0.9)' : 'rgba(0,245,255,0.07)';
      ctx.fillRect(x + 1, y + 1, cell - 2, cell - 2);
      if (v > 0) {
        ctx.fillStyle = NUM[v];
        ctx.font = `bold ${Math.round(cell * 0.55)}px monospace`;
        ctx.fillText(String(v), x + cell / 2, y + cell / 2 + 1);
      } else if (v < 0 && known.has(i)) {
        ctx.fillStyle = C.pink;
        ctx.fillRect(x + cell * 0.35, y + cell * 0.35, cell * 0.3, cell * 0.3);
      } else if (done && mine.has(i)) {
        ctx.fillStyle = '#ff3b3b';
        ctx.fillRect(x + cell * 0.3, y + cell * 0.3, cell * 0.4, cell * 0.4);
      }
      if (i === last) {
        ctx.strokeStyle = kind === 'certain' ? C.green : C.pink;
        ctx.lineWidth = 2;
        ctx.strokeRect(x + 3, y + 3, cell - 6, cell - 6);
      }
    }
    if (done) {
      if ((hold += speed) > 140) reset();
      return;
    }
    if ((hold += speed) < 18) return;
    hold = 0;
    const safe = proven();
    let pick;
    if (safe.length) {
      pick = safe[rand(safe.length)];
      kind = 'certain';
    } else {
      const covered = [...Array(N).keys()].filter((i) => val[i] < 0 && !known.has(i));
      pick = covered[rand(covered.length)];
      kind = 'guess';
    }
    last = pick;
    if (mine.has(pick)) {
      done = true; // the agent hit a mine: show them, then deal a new board
      hold = 0;
      return;
    }
    open(pick);
    if (val.filter((v) => v >= 0).length === N - MINES) {
      done = true; // cleared
      hold = 0;
    }
  };
}

function hexgameSim(ctx, S) {
  // 5x5 Hex on a canvas. Each game fills the board in a random order and the colours alternate, which is
  // what the agent's playouts do. Hex has no draws, so the full board has exactly one winner: find its
  // chain by breadth-first search, then light the chain. No server calls.
  const n = 5, N = 25, R = S / 12.5, w = Math.sqrt(3) * R;
  const ox = S * 0.12, oy = S * 0.2;
  const X = (r, c) => ox + w / 2 + (c + r / 2) * w;
  const Y = (r) => oy + R + r * 1.5 * R;
  // Neighbours of each cell: left, right, up, down, up-right, down-left.
  const NB = [...Array(N)].map((_, i) => {
    const r = (i / n) | 0, c = i % n;
    return [[0, -1], [0, 1], [-1, 0], [1, 0], [-1, 1], [1, -1]]
      .map(([dr, dc]) => [r + dr, c + dc])
      .filter(([rr, cc]) => rr >= 0 && rr < n && cc >= 0 && cc < n)
      .map(([rr, cc]) => rr * n + cc);
  });
  // Shortest chain of player p (1 = top to bottom, 2 = left to right), as a list of cells, or null.
  function chain(cells, p) {
    const starts = p === 1 ? [0, 1, 2, 3, 4] : [0, 5, 10, 15, 20];
    const goal = p === 1 ? (i) => i >= 20 : (i) => i % 5 === 4;
    const from = new Array(N).fill(-2);
    const queue = [];
    for (const s of starts) if (cells[s] === p) { from[s] = -1; queue.push(s); }
    for (let h = 0; h < queue.length; h++) {
      const u = queue[h];
      if (goal(u)) {
        const path = [];
        for (let x = u; x !== -1; x = from[x]) path.push(x);
        return path.reverse();
      }
      for (const v of NB[u]) if (from[v] === -2 && cells[v] === p) { from[v] = u; queue.push(v); }
    }
    return null;
  }
  let order, cells, k, hold, path;
  function deal() {
    order = [...Array(N).keys()];
    for (let i = N - 1; i > 0; i--) { const j = rand(i + 1); [order[i], order[j]] = [order[j], order[i]]; }
    cells = new Array(N).fill(0);
    k = 0; hold = 0; path = null;
  }
  deal();
  function hexPath(x, y) {
    ctx.beginPath();
    for (let j = 0; j < 6; j++) {
      const a = ((-90 + 60 * j) * Math.PI) / 180;
      const px = x + (R - 2) * Math.cos(a), py = y + (R - 2) * Math.sin(a);
      if (j) ctx.lineTo(px, py); else ctx.moveTo(px, py);
    }
    ctx.closePath();
  }
  return (speed) => {
    ctx.fillStyle = C.bg;
    ctx.fillRect(0, 0, S, S);
    for (let r = 0; r < n; r++) {
      for (let c = 0; c < n; c++) {
        const i = r * n + c, v = cells[i];
        hexPath(X(r, c), Y(r));
        ctx.fillStyle = v === 1 ? C.pink : v === 2 ? C.cyan : 'rgba(0,245,255,0.06)';
        ctx.shadowColor = v === 1 ? C.pink : C.cyan;
        ctx.shadowBlur = v ? 12 : 0;
        ctx.fill();
        ctx.shadowBlur = 0;
        if (path && path.includes(i)) { ctx.strokeStyle = C.yellow; ctx.lineWidth = 2; ctx.stroke(); }
      }
    }
    if (k < N) {
      if ((hold += speed) < 18) return;
      hold = 0;
      cells[order[k]] = k % 2 ? 2 : 1; // DOWN (1) moves first
      k++;
      if (k === N) path = chain(cells, chain(cells, 1) ? 1 : 2);
      return;
    }
    if ((hold += speed) > 200) deal();
  };
}

function banditsSim(ctx, S) {
  // Five slot machines with hidden payout rates. A Thompson agent keeps a Beta posterior per machine,
  // samples each one and pulls the best sample. Bars show the posterior means, the pulled machine
  // flashes, and the regret line grows wherever the agent pays for exploring. Cheap: no fetch, and
  // the bandit loop is a few lines.
  const K = 5, MAX_PULLS = 300;
  let p, wins, losses, t, cum, hist, flash, hold;
  const gauss = () => Math.sqrt(-2 * Math.log(Math.random() + 1e-12)) * Math.cos(2 * Math.PI * Math.random());
  function reset() {
    p = Array.from({ length: K }, () => 0.1 + 0.8 * Math.random());
    wins = Array(K).fill(0);
    losses = Array(K).fill(0);
    t = 0; cum = 0; hist = [0]; flash = -1; hold = 0;
  }
  // Thompson choice: a normal approximation to each Beta(1 + wins, 1 + losses), then the largest sample.
  function choose() {
    let best = 0, bestSample = -Infinity;
    for (let i = 0; i < K; i++) {
      const a = wins[i] + 1, b = losses[i] + 1, n = a + b;
      const mean = a / n, sd = Math.sqrt(a * b / (n * n * (n + 1)));
      const s = mean + sd * gauss();
      if (s > bestSample) { bestSample = s; best = i; }
    }
    return best;
  }
  function pullOnce() {
    const arm = choose();
    const won = rand(1000) / 1000 < p[arm];
    if (won) wins[arm]++; else losses[arm]++;
    cum += Math.max(...p) - p[arm];
    hist.push(cum);
    flash = arm;
    t++;
  }
  reset();
  return (speed) => {
    ctx.fillStyle = C.bg;
    ctx.fillRect(0, 0, S, S);
    hold += speed;
    if (t >= MAX_PULLS) {
      if (hold > 90) reset();
    } else if (hold >= 9) {
      hold = 0;
      pullOnce();
    }
    // Posterior means as bars across the top half, the true rates as ticks.
    const colW = (S - 20) / K;
    for (let i = 0; i < K; i++) {
      const a = wins[i] + 1, b = losses[i] + 1;
      const mean = a / (a + b);
      const x = 10 + i * colW + colW * 0.2, w = colW * 0.6;
      const barH = mean * 110;
      ctx.fillStyle = i === flash ? C.yellow : C.cyan;
      ctx.shadowColor = ctx.fillStyle;
      ctx.shadowBlur = i === flash ? 12 : 0;
      ctx.fillRect(x, 120 - barH, w, barH);
      ctx.shadowBlur = 0;
      ctx.fillStyle = C.pink;
      ctx.fillRect(x - 2, 120 - p[i] * 110, w + 4, 2);
    }
    // Regret over pulls, in the bottom half: where the line bends flat, the agent has settled.
    const top = Math.max(10, cum);
    ctx.strokeStyle = C.green;
    ctx.lineWidth = 2;
    ctx.beginPath();
    hist.forEach((v, i) => {
      const x = 10 + (i / MAX_PULLS) * (S - 20), y = 228 - (v / top) * 96;
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    });
    ctx.stroke();
    ctx.fillStyle = C.dim;
    ctx.font = '10px monospace';
    ctx.fillText('THOMPSON  REGRET ' + cum.toFixed(1), 10, 237);
  };
}

function cartpoleSim(ctx, S) {
  // Cart and pole held upright by the hand-tuned PD rule from cartpole/agents.py, with the same physics
  // as cartpole/env.py. When the pole falls, the screen holds for a moment and then deals a new start.
  const G = 9.8, MP = 0.1, TM = 1.1, PML = 0.05, HL = 0.5, F = 10, TAU = 0.02, THL = 12 * (Math.PI / 180);
  let s, hold, down;
  const reset = () => { s = [0, 0, (Math.random() - 0.5) * 0.1, 0]; hold = 0; down = 0; };
  const push = (a) => {
    const [x, xd, th, thd] = s, f = a ? F : -F, c = Math.cos(th), sn = Math.sin(th);
    const temp = (f + PML * thd * thd * sn) / TM;
    const tha = (G * sn - c * temp) / (HL * (4 / 3 - MP * c * c / TM));
    const xa = temp - PML * tha * c / TM;
    s = [x + TAU * xd, xd + TAU * xa, th + TAU * thd, thd + TAU * tha];
  };
  reset();
  const ppm = S / 6, track = S * 0.72;
  return (speed) => {
    ctx.fillStyle = C.bg;
    ctx.fillRect(0, 0, S, S);
    const [x, , th] = s;
    const cx = S / 2 + x * ppm, cw = S * 0.14, ch = S * 0.05;
    ctx.strokeStyle = C.cyan; ctx.lineWidth = 2; ctx.shadowColor = C.cyan; ctx.shadowBlur = 10;
    ctx.beginPath(); ctx.moveTo(0, track); ctx.lineTo(S, track); ctx.stroke();
    ctx.fillStyle = C.cyan; ctx.fillRect(cx - cw / 2, track - ch * 2, cw, ch);
    const tipX = cx + ppm * Math.sin(th), tipY = track - ch * 2 - ppm * Math.cos(th);
    ctx.strokeStyle = down ? C.dim : C.pink; ctx.shadowColor = C.pink; ctx.lineWidth = 3;
    ctx.beginPath(); ctx.moveTo(cx, track - ch * 2); ctx.lineTo(tipX, tipY); ctx.stroke();
    ctx.shadowBlur = 0;
    ctx.fillStyle = C.yellow; ctx.beginPath(); ctx.arc(tipX, tipY, S * 0.03, 0, Math.PI * 2); ctx.fill();
    if (down) { if ((down += speed) > 90) reset(); return; }
    if (Math.abs(th) > THL || Math.abs(x) > 2.4) { down = 1; return; }
    if ((hold += speed) < 2) return;
    hold = 0;
    const [, xd, , thd] = s;
    push(30 * th + 6 * thd + 0.1 * x + 0.5 * xd > 0 ? 1 : 0);
  };
}

function queensSim(ctx, S) {
  // 8 queens. Deal a random board, then repeatedly move a conflicted queen to its least attacked column
  // (min-conflicts, the same rule as queens/agents.py). A solved board holds for a moment, then a new one is dealt.
  const n = 8, cell = S / n;
  const L = { col: new Int32Array(n), d1: new Int32Array(2 * n - 1), d2: new Int32Array(2 * n - 1) };
  let cols = [], hold = 0;
  const add = (r, c, s) => { L.col[c] += s; L.d1[r + c] += s; L.d2[r - c + n - 1] += s; };
  // Queens attacking square (r, c), not counting the row-r queen (its own three counts are removed).
  const att = (r, c) => L.col[c] + L.d1[r + c] + L.d2[r - c + n - 1] - (cols[r] === c ? 3 : 0);
  function deal() {
    L.col.fill(0); L.d1.fill(0); L.d2.fill(0);
    cols = Array.from({ length: n }, () => Math.floor(Math.random() * n));
    cols.forEach((c, r) => add(r, c, 1));
    hold = 0;
  }
  // One repair; returns false when no queen is in conflict.
  function repair() {
    const bad = [];
    for (let r = 0; r < n; r++) if (att(r, cols[r]) > 0) bad.push(r);
    if (!bad.length) return false;
    const r = bad[Math.floor(Math.random() * bad.length)];
    add(r, cols[r], -1);
    let best = Infinity;
    const ties = [];
    for (let c = 0; c < n; c++) {
      const v = L.col[c] + L.d1[r + c] + L.d2[r - c + n - 1];
      if (v < best) { best = v; ties.length = 0; }
      if (v === best) ties.push(c);
    }
    const to = ties[Math.floor(Math.random() * ties.length)];
    cols[r] = to;
    add(r, to, 1);
    return true;
  }
  deal();
  return (speed) => {
    ctx.fillStyle = C.bg;
    ctx.fillRect(0, 0, S, S);
    let solved = true;
    for (let r = 0; r < n; r++) {
      for (let c = 0; c < n; c++) {
        const x = c * cell, y = r * cell;
        const q = cols[r] === c;
        ctx.fillStyle = (r + c) & 1 ? 'rgba(155,0,255,0.09)' : 'rgba(0,245,255,0.05)';
        ctx.fillRect(x, y, cell, cell);
        if (q) {
          const bad = att(r, c) > 0;
          if (bad) solved = false;
          ctx.fillStyle = bad ? C.pink : C.cyan;
          ctx.shadowColor = ctx.fillStyle;
          ctx.shadowBlur = 12;
          ctx.beginPath();
          ctx.arc(x + cell / 2, y + cell / 2, cell * 0.28, 0, Math.PI * 2);
          ctx.fill();
          ctx.shadowBlur = 0;
        } else {
          const h = att(r, c);
          if (h > 0) {
            ctx.fillStyle = `rgba(255,0,160,${Math.min(0.3, 0.04 + 0.07 * h).toFixed(3)})`;
            ctx.fillRect(x, y, cell, cell);
          }
        }
      }
    }
    if (solved) { if ((hold += speed) > 120) deal(); return; }
    if ((hold += speed) < 14) return;
    hold = 0;
    repair();
  };
}

function snakeSim(ctx, S) {
  // 10x10 snake steered by the greedy rule (the safe move that ends closest to the food), restarted
  // when it dies or fills the board. No server and no network: it only shows the snake moving on the
  // cabinet. The evolved net and the planner are on the snake page.
  const n = 10, cell = S / n;
  const DIRS = [[0, -1], [1, 0], [0, 1], [-1, 0]];
  let body, dir, food, hold, dead;
  const at = (x, y, segs) => segs.some(([bx, by]) => bx === x && by === y);
  const placeFood = () => {
    let x, y;
    do { x = rand(n); y = rand(n); } while (at(x, y, body));
    food = [x, y];
  };
  const reset = () => { body = [[4, 5], [3, 5], [2, 5]]; dir = 1; hold = 0; dead = 0; placeFood(); };
  // Straight first, so ties keep the snake going straight.
  const advance = () => {
    let best = null, bestD = Infinity;
    for (const turn of [0, 3, 1]) { // 0 straight, 3 turn left, 1 turn right (offsets from dir)
      const d = (dir + turn) % 4;
      const [dx, dy] = DIRS[d];
      const x = body[0][0] + dx, y = body[0][1] + dy;
      if (x < 0 || x >= n || y < 0 || y >= n) continue;
      const eats = x === food[0] && y === food[1];
      if (at(x, y, eats ? body : body.slice(0, -1))) continue;
      const dist = Math.abs(x - food[0]) + Math.abs(y - food[1]);
      if (dist < bestD) { best = { d, x, y, eats }; bestD = dist; }
    }
    if (!best) { dead = 40; return; }
    dir = best.d;
    body.unshift([best.x, best.y]);
    if (best.eats) { if (body.length === n * n) { dead = 60; return; } placeFood(); } else body.pop();
  };
  const draw = () => {
    ctx.fillStyle = C.bg; ctx.fillRect(0, 0, S, S);
    ctx.strokeStyle = 'rgba(0,245,255,0.08)'; ctx.lineWidth = 1;
    for (let i = 1; i < n; i++) {
      ctx.beginPath(); ctx.moveTo(i * cell, 0); ctx.lineTo(i * cell, S); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(0, i * cell); ctx.lineTo(S, i * cell); ctx.stroke();
    }
    ctx.fillStyle = C.pink; ctx.shadowColor = C.pink; ctx.shadowBlur = 12;
    ctx.beginPath(); ctx.arc((food[0] + 0.5) * cell, (food[1] + 0.5) * cell, cell * 0.36, 0, Math.PI * 2); ctx.fill();
    ctx.shadowBlur = 0;
    body.forEach(([x, y], i) => {
      ctx.fillStyle = i === 0 ? C.green : (i % 2 ? C.cyan : C.purple);
      ctx.globalAlpha = dead > 0 ? 0.35 : 1;
      ctx.fillRect(x * cell + cell * 0.1, y * cell + cell * 0.1, cell * 0.8, cell * 0.8);
    });
    ctx.globalAlpha = 1;
  };
  reset();
  return (speed) => {
    if (dead > 0) {
      dead -= speed;
      if (dead <= 0) reset();
    } else if ((hold += speed) > 5) {
      hold = 0;
      advance();
    }
    draw();
  };
}

function roverSim(ctx, S) {
  // 9x9 maze with a 3x3 sensor. The rover learns walls as it goes and re-plans with breadth-first search over
  // what it knows (unseen cells count as free). A cheap stand-in for the page's D* Lite: the cabinet only
  // needs to show the discovery and the replan, not the algorithm.
  const n = 9, N = n * n, cs = S / n;
  let walls, seen, belief, pos, route, hold, steps;
  const nb = (c) => {
    const r = (c / n) | 0, k = c % n, out = [];
    if (r > 0) out.push(c - n);
    if (k < n - 1) out.push(c + 1);
    if (r < n - 1) out.push(c + n);
    if (k > 0) out.push(c - 1);
    return out;
  };
  // Shortest route from `from` to the goal over the believed map, as a list of cells (empty if none).
  function plan(from) {
    const prev = new Int16Array(N).fill(-2);
    prev[from] = -1;
    const q = [from];
    for (let qi = 0; qi < q.length; qi++) {
      const u = q[qi];
      if (u === N - 1) break;
      for (const v of nb(u)) if (!belief[v] && prev[v] === -2) { prev[v] = u; q.push(v); }
    }
    if (prev[N - 1] === -2) return [];
    const out = [];
    for (let c = N - 1; c !== -1; c = prev[c]) out.push(c);
    return out.reverse();
  }
  function sense() {
    const r = (pos / n) | 0, k = pos % n;
    let changed = false;
    for (let dr = -1; dr <= 1; dr++) for (let dk = -1; dk <= 1; dk++) {
      const rr = r + dr, kk = k + dk;
      if (rr < 0 || rr >= n || kk < 0 || kk >= n) continue;
      const c = rr * n + kk;
      seen[c] = 1;
      if (belief[c] !== walls[c]) { belief[c] = walls[c]; changed = true; }
    }
    return changed;
  }
  function deal() {
    walls = new Uint8Array(N);
    for (let i = 1; i < N - 1; i++) walls[i] = Math.random() < 0.28 ? 1 : 0;
    belief = new Uint8Array(N); seen = new Uint8Array(N);
    pos = 0; hold = 0; steps = 0;
    sense();
    route = plan(pos);
    // Keep only mazes with a route, so the cabinet never sits waiting on an impossible one.
    if (!route.length) return deal();
  }
  deal();
  return (speed) => {
    if (hold > 0) { hold -= speed; if (hold <= 0) deal(); }
    else {
      if (sense()) route = plan(pos);
      if (pos === N - 1) hold = 90;
      else if (route.length > 1) { pos = route[1]; route = route.slice(1); steps++; }
    }
    ctx.fillStyle = C.bg;
    ctx.fillRect(0, 0, S, S);
    for (let i = 0; i < N; i++) {
      const r = (i / n) | 0, k = i % n;
      if (belief[i]) { ctx.fillStyle = C.pink; ctx.shadowColor = C.pink; ctx.shadowBlur = 6; }
      else if (seen[i]) { ctx.fillStyle = 'rgba(0,245,255,0.12)'; ctx.shadowBlur = 0; }
      else { ctx.fillStyle = 'rgba(0,245,255,0.04)'; ctx.shadowBlur = 0; }
      ctx.fillRect(k * cs + 1, r * cs + 1, cs - 2, cs - 2);
    }
    ctx.shadowBlur = 0;
    ctx.fillStyle = C.cyan;
    for (const c of route.slice(1)) {
      ctx.beginPath(); ctx.arc(((c % n) + 0.5) * cs, (((c / n) | 0) + 0.5) * cs, cs * 0.12, 0, 6.283); ctx.fill();
    }
    ctx.fillStyle = C.green;
    ctx.fillRect((n - 1) * cs + cs * 0.25, (n - 1) * cs + cs * 0.25, cs * 0.5, cs * 0.5);
    ctx.shadowColor = C.yellow; ctx.shadowBlur = 10;
    ctx.fillStyle = C.yellow;
    ctx.beginPath(); ctx.arc(((pos % n) + 0.5) * cs, (((pos / n) | 0) + 0.5) * cs, cs * 0.3, 0, 6.283); ctx.fill();
    ctx.shadowBlur = 0;
  };
}

function tetrisSim(ctx, S) {
  // Tetris on a 10x20 board. A tiny version of the placement search: each rotation and column of the falling
  // piece is dropped and scored (aggregate height, holes, bumpiness, lines), the best one is drawn as a ghost,
  // then locked. Board rows are bitmasks, as in tetris/board.py. No server calls.
  const W = 10, H = 20, FULL = (1 << W) - 1, cell = Math.floor(S / H), ox = (S - W * cell) / 2;
  const SHAPES = { I: [[0, 0], [1, 0], [2, 0], [3, 0]], O: [[0, 0], [1, 0], [0, 1], [1, 1]], T: [[0, 0], [1, 0], [2, 0], [1, 1]],
    S: [[1, 0], [2, 0], [0, 1], [1, 1]], Z: [[0, 0], [1, 0], [1, 1], [2, 1]], J: [[0, 0], [0, 1], [1, 1], [2, 1]],
    L: [[2, 0], [0, 1], [1, 1], [2, 1]] };
  const COLOURS = { I: C.cyan, O: C.yellow, T: C.purple, S: C.green, Z: C.pink, J: '#3d7bff', L: '#ff8a00' };
  // Every distinct orientation of each piece as row masks (bottom row first) and its size.
  const states = {};
  for (const name of Object.keys(SHAPES)) {
    const seen = new Set(), list = [];
    let cells = SHAPES[name];
    for (let k = 0; k < 4; k++) {
      const mx = Math.min(...cells.map((c) => c[0])), my = Math.min(...cells.map((c) => c[1]));
      cells = cells.map(([x, y]) => [x - mx, y - my]);
      const key = JSON.stringify(cells.slice().sort());
      if (!seen.has(key)) {
        seen.add(key);
        const h = Math.max(...cells.map((c) => c[1])) + 1, w = Math.max(...cells.map((c) => c[0])) + 1;
        const masks = [...Array(h)].map((_, r) => cells.reduce((m, [x, y]) => (y === r ? m | (1 << x) : m), 0));
        list.push({ masks, w, h });
      }
      cells = cells.map(([x, y]) => [y, -x]);
    }
    states[name] = list;
  }
  const names = Object.keys(SHAPES);
  const pop = (n) => { let c = 0; for (; n; n &= n - 1) c++; return c; };
  // Hand-picked weights for this demo: stack height and holes hurt, bumpiness hurts a little, lines help.
  const score = (rows, lines) => {
    let top = 0, holes = 0, agg = 0, bump = 0, seen = 0;
    const heights = Array(W).fill(0);
    for (let r = H - 1; r >= 0; r--) if (rows[r]) { top = r + 1; break; }
    for (let r = top - 1; r >= 0; r--) {
      holes += pop(seen & ~rows[r] & FULL);
      for (let c = 0; c < W; c++) if (((rows[r] & ~seen) >> c) & 1) heights[c] = r + 1;
      seen |= rows[r];
      agg += pop(seen);
    }
    for (let c = 0; c < W - 1; c++) bump += Math.abs(heights[c] - heights[c + 1]);
    return -0.5 * agg - 0.4 * holes - 0.2 * bump + 0.8 * lines;
  };
  // Best placement of `name` on `rows`, or null if the piece cannot spawn.
  function best(rows, name) {
    let pick = null;
    for (const st of states[name]) {
      for (let px = 0; px + st.w <= W; px++) {
        let py = H - st.h;
        const fits = (y) => st.masks.every((m, i) => y + i >= H || !(rows[y + i] & (m << px)));
        if (!fits(py)) continue;
        while (py > 0 && fits(py - 1)) py--;
        const next = rows.slice();
        st.masks.forEach((m, i) => { if (py + i < H) next[py + i] |= m << px; });
        let lines = 0;
        const kept = next.filter((r) => (r === FULL ? (lines++, false) : true));
        while (kept.length < H) kept.push(0);
        const s = score(kept, lines);
        if (!pick || s > pick.s) pick = { s, st, px, py, rows: kept, lines };
      }
    }
    return pick;
  }
  let rows = Array(H).fill(0), piece = names[rand(names.length)], plan = null, lines = 0, hold = 0, pieces = 0;
  const draw = () => {
    ctx.clearRect(0, 0, S, S);
    ctx.fillStyle = C.bg; ctx.fillRect(0, 0, S, S);
    ctx.strokeStyle = 'rgba(0,245,255,0.08)';
    for (let r = 0; r < H; r++) for (let c = 0; c < W; c++) ctx.strokeRect(ox + c * cell, (H - 1 - r) * cell, cell, cell);
    rows.forEach((row, r) => {
      for (let c = 0; c < W; c++) if ((row >> c) & 1) {
        ctx.fillStyle = `hsl(${185 + (r / H) * 150},100%,58%)`;
        ctx.fillRect(ox + c * cell + 1, (H - 1 - r) * cell + 1, cell - 2, cell - 2);
      }
    });
    if (plan) {
      ctx.strokeStyle = COLOURS[piece]; ctx.lineWidth = 2;
      plan.st.masks.forEach((m, i) => { for (let c = 0; c < plan.st.w; c++) if ((m >> c) & 1) {
        ctx.strokeRect(ox + (plan.px + c) * cell + 1, (H - 1 - (plan.py + i)) * cell + 1, cell - 2, cell - 2);
      } });
    }
    ctx.fillStyle = C.cyan; ctx.font = '10px monospace';
    ctx.fillText(`LINES ${lines}`, 6, 12);
    ctx.fillText(`PIECES ${pieces}`, 6, 24);
  };
  draw();
  return (speed) => {
    hold += speed;
    if (hold < 16) return;
    hold = 0;
    if (!plan) {
      plan = best(rows, piece);
      if (!plan) { rows = Array(H).fill(0); lines = 0; pieces = 0; }
      draw();
      return;
    }
    // The agent drops the piece it chose: lock it and bring in the next piece.
    lines += plan.lines;
    rows = plan.rows;
    pieces++;
    piece = names[rand(names.length)];
    plan = null;
    draw();
  };
}

function nonogramSim(ctx, S) {
  // A 10x10 hand-drawn picture. Its clues are computed from the picture, then the rows are revealed one at a
  // time, as the line solver fixes whole lines. A yellow scan bar marks the row being solved, and the picture
  // mirrors on each loop. Cheap: no server calls, and the clues are only recomputed when the picture changes.
  const n = 10, M = S * 0.2, cell = (S - M) / n;
  const ART = ['..#....#..', '...#..#...', '..######..', '.##.##.##.', '##########',
               '#.######.#', '#.#....#.#', '...##.##..', '..........', '..........'];
  const runs = (vals) => {
    const out = []; let k = 0;
    for (const v of vals) { if (v) k++; else if (k) { out.push(k); k = 0; } }
    if (k) out.push(k);
    return out.length ? out : [0];
  };
  let pic, rowClues, colClues, flip = false;
  const setPic = () => {
    pic = [];
    for (let r = 0; r < n; r++) for (let c = 0; c < n; c++) {
      const ch = ART[r][flip ? n - 1 - c : c];
      pic.push(ch === '#' ? 1 : 0);
    }
    rowClues = [...Array(n)].map((_, r) => runs(pic.slice(r * n, r * n + n)));
    colClues = [...Array(n)].map((_, c) => runs([...Array(n)].map((_, r) => pic[r * n + c])));
  };
  setPic();
  const shown = new Uint8Array(n * n);
  let row = -1, hold = 0, phase = 'reveal';
  ctx.font = '9px JetBrains Mono, monospace';
  return (speed) => {
    ctx.fillStyle = C.bg;
    ctx.fillRect(0, 0, S, S);
    for (let r = 0; r < n; r++) for (let c = 0; c < n; c++) {
      const i = r * n + c, on = shown[i];
      const x = M + c * cell, y = M + r * cell;
      ctx.fillStyle = on ? C.cyan : 'rgba(0,245,255,0.06)';
      ctx.shadowColor = C.cyan;
      ctx.shadowBlur = on ? 10 : 0;
      ctx.fillRect(x + 2, y + 2, cell - 4, cell - 4);
    }
    ctx.shadowBlur = 0;
    // Clues: a row that is revealed turns green; the others stay dim.
    ctx.textAlign = 'right';
    ctx.textBaseline = 'middle';
    for (let r = 0; r < n; r++) {
      ctx.fillStyle = r <= row ? C.green : C.dim;
      ctx.fillText(rowClues[r].join(' '), M - 4, M + r * cell + cell / 2);
    }
    ctx.textAlign = 'center';
    ctx.textBaseline = 'bottom';
    for (let c = 0; c < n; c++) {
      ctx.fillStyle = C.dim;
      colClues[c].forEach((v, k) => {
        const off = colClues[c].length - k;
        ctx.fillText(String(v), M + c * cell + cell / 2, M - 3 - (off - 1) * 9);
      });
    }
    if (row >= 0 && row < n) {
      ctx.fillStyle = C.yellow;
      ctx.fillRect(M, M + row * cell, S - M, 2);
    }
    // Advance: a row every 18 ticks, a hold after the last row, then restart on the mirrored picture.
    if (phase === 'reveal') {
      if ((hold += speed) < 18) return;
      hold = 0;
      row++;
      if (row < n) for (let c = 0; c < n; c++) shown[row * n + c] = pic[row * n + c];
      else phase = 'hold';
    } else if ((hold += speed) > 140) {
      shown.fill(0);
      row = -1; hold = 0; phase = 'reveal';
      flip = !flip;
      setPic();
    }
  };
}

function markovSim(ctx, S) {
  const TEXT = "Shall I compare thee to a summer's day? Thou art more lovely and more temperate: "
    + "Rough winds do shake the darling buds of May, And summer's lease hath all too short a date; "
    + "Sometime too hot the eye of heaven shines, And often is his gold complexion dimm'd; "
    + "And every fair from fair sometime declines, By chance or nature's changing course untrimm'd; "
    + "But thy eternal summer shall not fade, Nor lose possession of that fair thou ow'st; "
    + "Nor shall Death brag thou wander'st in his shade, When in eternal lines to time thou grow'st.";
  const words = TEXT.match(/[A-Za-z]+(?:'[A-Za-z]+)*|[^\sA-Za-z]/g);
  const KEY = '\u0001';
  // Counts: for every pair of words, how often each word followed it.
  const succ = new Map();
  for (let i = 0; i + 2 < words.length; i++) {
    const key = words[i] + KEY + words[i + 1];
    if (!succ.has(key)) succ.set(key, new Map());
    const d = succ.get(key);
    d.set(words[i + 2], (d.get(words[i + 2]) || 0) + 1);
  }
  let out, hold, lines;
  function reset() {
    out = words.slice(0, 2);
    hold = 0;
    lines = [''];
  }
  // One draw: pick the next word in proportion to its count after the last two words.
  function nextWord() {
    const d = succ.get(out.slice(-2).join(KEY));
    if (!d) { out.push(...words.slice(0, 2)); return; } // dead end at the end of the passage: jump back
    let total = 0;
    for (const c of d.values()) total += c;
    let u = Math.random() * total;
    for (const [w, c] of d) {
      u -= c;
      if (u < 0) { out.push(w); return; }
    }
  }
  reset();
  return function tick(speed) {
    hold += speed;
    if (hold > 7) {
      hold = 0;
      nextWord();
      if (out.length > 400) out = out.slice(-40);
      // Word-wrap the whole output into lines that fit the canvas.
      lines = [''];
      ctx.font = '12px monospace';
      for (const w of out) {
        const cand = lines[lines.length - 1] ? lines[lines.length - 1] + ' ' + w : w;
        if (ctx.measureText(cand).width > S - 20) lines.push(w);
        else lines[lines.length - 1] = cand;
      }
    }
    ctx.fillStyle = '#05070d';
    ctx.fillRect(0, 0, S, S);
    ctx.font = '12px monospace';
    ctx.fillStyle = '#c0e8ff';
    ctx.shadowColor = '#00f5ff';
    ctx.shadowBlur = 6;
    const top = Math.max(0, lines.length - 9);
    for (let i = top; i < lines.length; i++) ctx.fillText(lines[i], 10, 22 + (i - top) * 18);
    ctx.shadowBlur = 0;
    ctx.fillStyle = '#ff00a0';
    ctx.fillRect(10, S - 22, 6 + 10 * ((Date.now() / 300) % 2 < 1 ? 1 : 0), 3);
  };
}

function regressionSim(ctx, S) {
  // Scatter of noisy points around a parabola, with a live least-squares quadratic through them.
  // Each squared residual is drawn as a square (its area is the squared error). Every so often the
  // points jitter and the fit snaps again. Cheap: 3x3 normal equations solved by Cramer's rule, no fetch.
  const N = 14, PAD = 12, SPAN = S - 2 * PAD;
  const gauss = () => Math.sqrt(-2 * Math.log(Math.random() + 1e-12)) * Math.cos(2 * Math.PI * Math.random());
  const X = (x) => PAD + ((x + 1) / 2) * SPAN;
  const Y = (y) => S - PAD - ((y + 1) / 2) * SPAN;
  let xs, ys, t;
  function reset() {
    xs = Array.from({ length: N }, () => -1 + 2 * Math.random());
    ys = xs.map((x) => 0.7 * x * x - 0.3 + 0.12 * gauss());
    t = 0;
  }
  // Quadratic least squares: the normal matrix A[r][c] = sum x^(r+c), right-hand side sum x^r y.
  function det3(a) {
    return a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1])
      - a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0])
      + a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0]);
  }
  function fit() {
    const s = [0, 0, 0, 0, 0], b = [0, 0, 0];
    for (let i = 0; i < N; i++) {
      let p = 1;
      for (let k = 0; k < 5; k++) { s[k] += p; p *= xs[i]; }
      p = 1;
      for (let k = 0; k < 3; k++) { b[k] += p * ys[i]; p *= xs[i]; }
    }
    const A = [[s[0], s[1], s[2]], [s[1], s[2], s[3]], [s[2], s[3], s[4]]];
    const D = det3(A);
    if (Math.abs(D) < 1e-12) return [0, 0, 0];
    // Cramer's rule: replace column c with the right-hand side and take the determinant ratio.
    return [0, 1, 2].map((c) => det3(A.map((row, r) => row.map((v, cc) => (cc === c ? b[r] : v)))) / D);
  }
  reset();
  return function tick() {
    if (++t % 260 === 0) {
      // Jitter the points a little, and start over now and then so the cloud does not drift away.
      if (Math.random() < 0.3) reset();
      else ys = ys.map((y) => y + 0.06 * gauss());
    }
    ctx.clearRect(0, 0, S, S);
    ctx.fillStyle = 'rgba(2,6,16,0.95)';
    ctx.fillRect(0, 0, S, S);
    const [w0, w1, w2] = fit();
    // Squares: side = |residual| in pixels, so the drawn area equals the squared error.
    ctx.fillStyle = 'rgba(255,0,160,0.18)';
    ctx.strokeStyle = '#ff00a0';
    ctx.lineWidth = 1;
    for (let i = 0; i < N; i++) {
      const yhat = w0 + w1 * xs[i] + w2 * xs[i] * xs[i];
      const r = ys[i] - yhat;
      const side = Math.abs(r) * (SPAN / 2);
      if (side < 0.5) continue;
      const x0 = X(xs[i]), y0 = Math.min(Y(ys[i]), Y(yhat));
      ctx.fillRect(x0, y0, side, side);
      ctx.strokeRect(x0, y0, side, side);
    }
    // The fitted curve, then the points on top.
    ctx.strokeStyle = '#00f5ff';
    ctx.lineWidth = 2;
    ctx.shadowColor = '#00f5ff';
    ctx.shadowBlur = 8;
    ctx.beginPath();
    for (let k = 0; k <= 60; k++) {
      const x = -1 + (2 * k) / 60;
      const y = w0 + w1 * x + w2 * x * x;
      if (k === 0) ctx.moveTo(X(x), Y(y)); else ctx.lineTo(X(x), Y(y));
    }
    ctx.stroke();
    ctx.shadowBlur = 0;
    ctx.fillStyle = '#ffe600';
    for (let i = 0; i < N; i++) {
      ctx.beginPath();
      ctx.arc(X(xs[i]), Y(ys[i]), 3, 0, Math.PI * 2);
      ctx.fill();
    }
  };
}

function localizeSim(ctx, S) {
  // Cabinet stand-in for the Lost Robot page: a 24x16 Twin Halls floor (same walls as localize-core.js), a robot
  // that wanders the corridor, and a 150-particle cloud that collapses onto it. The page runs the full filters;
  // here the cloud only has to show the collapse, so it is a short-range version with Math.random (no parity needed).
  const W = 24, H = 16, cs = S / W;
  const walls = new Uint8Array(W * H);
  const rects = [[0, 0, 23, 0], [0, 15, 23, 15], [0, 0, 0, 15], [23, 0, 23, 15],
    [1, 6, 2, 6], [4, 6, 8, 6], [10, 6, 14, 6], [16, 6, 19, 6], [21, 6, 22, 6],
    [1, 9, 2, 9], [4, 9, 8, 9], [10, 9, 14, 9], [16, 9, 19, 9], [21, 9, 22, 9],
    [6, 1, 6, 5], [12, 1, 12, 5], [18, 1, 18, 5], [6, 10, 6, 14], [12, 10, 12, 14], [18, 10, 18, 14]];
  for (const [x0, y0, x1, y1] of rects) for (let y = y0; y <= y1; y++) for (let x = x0; x <= x1; x++) walls[y * W + x] = 1;
  const blocked = (x, y) => {
    const i = Math.floor(x), j = Math.floor(y);
    return i < 0 || i >= W || j < 0 || j >= H || walls[j * W + i] === 1;
  };
  const cast = (x, y, a) => {
    const dx = Math.cos(a) * 0.25, dy = Math.sin(a) * 0.25;
    let px = x, py = y;
    for (let k = 1; k <= 32; k++) { px += dx; py += dy; if (blocked(px, py)) return k * 0.25; }
    return 8;
  };
  const N = 150, NB = 8, SIG = 0.4;
  const rx = new Float64Array(N), ry = new Float64Array(N), rt = new Float64Array(N), rw = new Float64Array(N);
  const cells = [];
  for (let c = 0; c < W * H; c++) if (!walls[c]) cells.push(c);
  const spawn = (i) => {
    const c = cells[(Math.random() * cells.length) | 0];
    rx[i] = (c % W) + Math.random(); ry[i] = ((c / W) | 0) + Math.random(); rt[i] = Math.random() * 6.283 - 3.1416;
  };
  let bx, by, bt, z;
  const beams = (x, y, t) => {
    const out = new Float64Array(NB);
    for (let j = 0; j < NB; j++) out[j] = cast(x, y, t + (6.283 * j) / NB);
    return out;
  };
  const deal = () => {
    for (let i = 0; i < N; i++) { spawn(i); rw[i] = 1 / N; }
    const cell = cells[(Math.random() * cells.length) | 0];
    bx = (cell % W) + 0.5; by = ((cell / W) | 0) + 0.5; bt = Math.random() * 6.283;
    z = beams(bx, by, bt);
  };
  deal();
  let hold = 0, steps = 0;
  return (speed) => {
    if (hold > 0) { hold -= speed; if (hold <= 0) deal(); }
    else {
      // The robot turns when its next step would hit a wall, and drives otherwise.
      if (blocked(bx + 0.3 * Math.cos(bt), by + 0.3 * Math.sin(bt))) bt += 0.5 + Math.random();
      else { bx += 0.3 * Math.cos(bt); by += 0.3 * Math.sin(bt); }
      z = beams(bx, by, bt);
      steps++;
      // Predict with the odometry noise, weigh by the beams, resample when the weights bunch up.
      let total = 0;
      for (let i = 0; i < N; i++) {
        rt[i] += 0.05 * (Math.random() - 0.5) + 0.1 * (Math.random() - 0.5);
        const nx = rx[i] + 0.3 * Math.cos(rt[i]), ny = ry[i] + 0.3 * Math.sin(rt[i]);
        if (!blocked(nx, ny)) { rx[i] = nx; ry[i] = ny; }
        const e = beams(rx[i], ry[i], rt[i]);
        let ll = 0;
        for (let j = 0; j < NB; j++) ll -= ((z[j] - e[j]) / SIG) ** 2 / 2;
        rw[i] *= Math.exp(ll / 4);
        total += rw[i];
      }
      for (let i = 0; i < N; i++) rw[i] /= total || 1;
      let sq = 0;
      for (let i = 0; i < N; i++) sq += rw[i] * rw[i];
      if (1 / sq < N / 2) {
        // Multinomial resampling is enough at this size: copy particles in proportion to their weight.
        const nx = new Float64Array(N), ny = new Float64Array(N), nt = new Float64Array(N);
        for (let i = 0; i < N; i++) {
          let u = Math.random(), k = 0;
          for (let acc = rw[0]; acc < u && k < N - 1; ) acc += rw[++k];
          nx[i] = rx[k]; ny[i] = ry[k]; nt[i] = rt[k];
        }
        rx.set(nx); ry.set(ny); rt.set(nt);
        rw.fill(1 / N);
      }
      if (steps > 90) hold = 60;
    }
    ctx.fillStyle = C.bg;
    ctx.fillRect(0, 0, S, S);
    for (let c = 0; c < W * H; c++) {
      if (!walls[c]) continue;
      ctx.fillStyle = 'rgba(255,0,160,0.35)';
      ctx.fillRect((c % W) * cs + 0.5, ((c / W) | 0) * cs + 0.5, cs - 1, cs - 1);
    }
    ctx.fillStyle = C.cyan;
    for (let i = 0; i < N; i++) {
      ctx.globalAlpha = 0.25 + 0.75 * Math.min(1, rw[i] * N / 2);
      ctx.fillRect(rx[i] * cs - 1, ry[i] * cs - 1, 2.5, 2.5);
    }
    ctx.globalAlpha = 1;
    ctx.shadowColor = C.yellow; ctx.shadowBlur = 10;
    ctx.fillStyle = C.yellow;
    ctx.beginPath(); ctx.arc(bx * cs, by * cs, cs * 0.3, 0, 6.283); ctx.fill();
    ctx.shadowBlur = 0;
  };
}

function walkersSim(ctx, S) {
  // A stylised walker for the cabinet: six nodes in a row, each link a muscle whose length swings with a wave that
  // travels along the body, so it inches to the right. Pure drawing, no physics (the real run is on walkers.html).
  const N = 6, seg = S * 0.105, groundY = S * 0.72;
  let t = 0;
  return (speed) => {
    ctx.fillStyle = C.bg;
    ctx.fillRect(0, 0, S, S);
    ctx.strokeStyle = C.cyan; ctx.lineWidth = 2; ctx.shadowColor = C.cyan; ctx.shadowBlur = 8;
    ctx.beginPath(); ctx.moveTo(0, groundY); ctx.lineTo(S, groundY); ctx.stroke();
    ctx.shadowBlur = 0;
    t += 0.05 * speed;
    const px = [], py = [];
    for (let i = 0; i < N; i++) {
      const phase = t - i * 0.7;
      const stretch = 0.5 + 0.5 * Math.sin(phase);
      px.push(S * 0.12 + i * seg * (0.9 + 0.2 * stretch));
      py.push(groundY - S * 0.05 - S * 0.05 * Math.max(0, Math.sin(phase + 1.2)));
    }
    for (let i = 0; i < N - 1; i++) {
      const pull = Math.sin(t - i * 0.7);
      ctx.strokeStyle = pull > 0 ? C.pink : C.cyan;
      ctx.lineWidth = 2 + 2 * Math.abs(pull);
      ctx.shadowColor = ctx.strokeStyle; ctx.shadowBlur = 8;
      ctx.beginPath(); ctx.moveTo(px[i], py[i]); ctx.lineTo(px[i + 1], py[i + 1]); ctx.stroke();
    }
    ctx.shadowBlur = 0;
    ctx.fillStyle = C.yellow;
    for (let i = 0; i < N; i++) { ctx.beginPath(); ctx.arc(px[i], py[i], S * 0.025, 0, Math.PI * 2); ctx.fill(); }
  };
}

// Landing cabinet sim for the cluster lab. Paste into web/js/landing.js next to the other *Sim functions and add
// `clusters: clustersSim` to the sims map in cabinets(). Same contract as the others: draw into ctx (S x S units)
// and return tick(speed). No server calls and no imports.
function clustersSim(ctx, S) {
  // k-means on 90 points in three blobs, one assign or update step per tick. The start is deliberately poor
  // (all three centroids on one side), so the centroids visibly walk into the blobs, then the run pauses and
  // restarts from a new random start. The points come from a fixed LCG, so every visit looks the same.
  const N = 90, K = 3;
  const COLS = ['#00f5ff', '#ff00a0', '#00ff88'];
  const pad = 14, span = S - 2 * pad;
  const X = (x) => pad + x * span, Y = (y) => pad + (1 - y) * span;
  let seed = 12345;
  const rnd = () => ((seed = (seed * 1664525 + 1013904223) >>> 0) / 4294967296);
  const centres0 = [[0.25, 0.3], [0.75, 0.3], [0.5, 0.75]];
  const P = Array.from({ length: N }, (_, i) => {
    const c = centres0[i % K];
    return [c[0] + (rnd() - 0.5) * 0.22, c[1] + (rnd() - 0.5) * 0.22];
  });
  let cent = [];
  let lab = new Array(N).fill(0);
  let trail = [];
  let phase = 0;     // 0 = assign next, 1 = update next
  let hold = 0;      // frames to pause on the converged picture
  let done = false;
  const restart = () => {
    cent = [0, 1, 2].map(() => [0.15 + rnd() * 0.7, 0.15 + rnd() * 0.7]);
    lab = new Array(N).fill(0);
    trail = [];
    phase = 0;
    done = false;
    hold = 0;
  };
  restart();
  const step = () => {
    if (done) {
      if (++hold > 90) restart();
      return;
    }
    if (phase === 0) {
      let changed = false;
      for (let i = 0; i < N; i++) {
        let best = 0;
        let bd = Infinity;
        for (let j = 0; j < K; j++) {
          const d = (P[i][0] - cent[j][0]) ** 2 + (P[i][1] - cent[j][1]) ** 2;
          if (d < bd) { bd = d; best = j; }
        }
        if (lab[i] !== best) changed = true;
        lab[i] = best;
      }
      if (!changed && trail.length) done = true;
      phase = 1;
    } else {
      const sum = Array.from({ length: K }, () => [0, 0, 0]);
      for (let i = 0; i < N; i++) {
        sum[lab[i]][0] += P[i][0];
        sum[lab[i]][1] += P[i][1];
        sum[lab[i]][2]++;
      }
      trail.push(cent.map((c) => c.slice()));
      if (trail.length > 8) trail.shift();
      cent = cent.map((c, j) => (sum[j][2] ? [sum[j][0] / sum[j][2], sum[j][1] / sum[j][2]] : c));
      phase = 0;
    }
  };
  const draw = () => {
    ctx.fillStyle = '#03060f';
    ctx.fillRect(0, 0, S, S);
    // Voronoi shading of the current centroids, on a coarse grid
    const g = 20;
    for (let gy = 0; gy < g; gy++) {
      for (let gx = 0; gx < g; gx++) {
        const x = (gx + 0.5) / g, y = 1 - (gy + 0.5) / g;
        let best = 0, bd = Infinity;
        for (let j = 0; j < K; j++) {
          const d = (x - cent[j][0]) ** 2 + (y - cent[j][1]) ** 2;
          if (d < bd) { bd = d; best = j; }
        }
        ctx.fillStyle = COLS[best] + '14';
        ctx.fillRect(pad + (gx / g) * span, pad + (gy / g) * span, span / g + 1, span / g + 1);
      }
    }
    for (let i = 0; i < N; i++) {
      ctx.fillStyle = COLS[lab[i]];
      ctx.beginPath();
      ctx.arc(X(P[i][0]), Y(P[i][1]), 2.2, 0, Math.PI * 2);
      ctx.fill();
    }
    trail.forEach((snap, t) => {
      snap.forEach((c, j) => {
        ctx.strokeStyle = COLS[j] + '66';
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.arc(X(c[0]), Y(c[1]), 3 + t * 0.3, 0, Math.PI * 2);
        ctx.stroke();
      });
    });
    cent.forEach((c, j) => {
      ctx.save();
      ctx.shadowColor = COLS[j];
      ctx.shadowBlur = 14;
      ctx.fillStyle = COLS[j];
      ctx.beginPath();
      ctx.arc(X(c[0]), Y(c[1]), 5, 0, Math.PI * 2);
      ctx.fill();
      ctx.restore();
    });
  };
  let acc = 0;
  return (speed) => {
    acc += speed * 0.25;
    while (acc >= 1) {
      acc -= 1;
      step();
    }
    draw();
  };
}

function nnlabSim(ctx, S) {
  // A three-neuron tanh network learns a ring of 40 points by plain full-batch gradient descent, two steps
  // per frame. The plane behind is a coarse neon heat map of its answer, so the boundary bends as it learns.
  // Cheap: no fetch, 40 points, 16 x 16 cells, and the network restarts every 600 steps.
  const N = 40, H = 3, G = 16, RESTART = 600;
  const pts = Array.from({ length: N }, (_, i) => {
    const inner = i % 2 === 0;
    const r = inner ? 0.35 : 0.8, a = Math.random() * Math.PI * 2;
    return { x: r * Math.cos(a), y: r * Math.sin(a), c: inner ? 0 : 1 };
  });
  let W1, b1, W2, b2, steps = 0;
  const rnd = () => (Math.random() * 2 - 1) * 0.6;
  function reset() {
    W1 = Array.from({ length: H }, () => [rnd(), rnd()]);
    b1 = Array(H).fill(0);
    W2 = Array.from({ length: H }, rnd);
    b2 = 0;
    steps = 0;
  }
  // Forward pass: hidden tanh units h, then a sigmoid output q = P(pink).
  function fwd(x, y) {
    const h = W1.map((w, j) => Math.tanh(w[0] * x + w[1] * y + b1[j]));
    let z = b2;
    for (let j = 0; j < H; j++) z += W2[j] * h[j];
    return [h, 1 / (1 + Math.exp(-z))];
  }
  // One full-batch step. Output: dz = (q - c) / N. Hidden: dh = dz * W2 * (1 - h^2), the chain rule through tanh.
  function step() {
    const gW1 = W1.map(() => [0, 0]), gb1 = Array(H).fill(0), gW2 = Array(H).fill(0);
    let gb2 = 0;
    for (const p of pts) {
      const [h, q] = fwd(p.x, p.y);
      const dz = (q - p.c) / N;
      gb2 += dz;
      for (let j = 0; j < H; j++) {
        gW2[j] += dz * h[j];
        const dh = dz * W2[j] * (1 - h[j] * h[j]);
        gW1[j][0] += dh * p.x;
        gW1[j][1] += dh * p.y;
        gb1[j] += dh;
      }
    }
    const lr = 2.5;
    for (let j = 0; j < H; j++) {
      W1[j][0] -= lr * gW1[j][0];
      W1[j][1] -= lr * gW1[j][1];
      b1[j] -= lr * gb1[j];
      W2[j] -= lr * gW2[j];
    }
    b2 -= lr * gb2;
  }
  reset();
  return () => {
    ctx.fillStyle = C.bg;
    ctx.fillRect(0, 0, S, S);
    for (let k = 0; k < 2; k++) step();
    if ((steps += 2) >= RESTART) reset();
    const cell = S / G;
    for (let gy = 0; gy < G; gy++) {
      for (let gx = 0; gx < G; gx++) {
        const x = ((gx + 0.5) / G) * 2 - 1, y = 1 - ((gy + 0.5) / G) * 2;
        const q = fwd(x, y)[1];
        const near = Math.min(1, Math.abs(q - 0.5) * 2);
        ctx.fillStyle = q >= 0.5
          ? `rgba(255,0,160,${(0.12 + 0.4 * (1 - near)).toFixed(3)})`
          : `rgba(0,245,255,${(0.12 + 0.4 * (1 - near)).toFixed(3)})`;
        ctx.fillRect(gx * cell, gy * cell, cell + 0.5, cell + 0.5);
      }
    }
    for (const p of pts) {
      ctx.fillStyle = p.c ? C.pink : C.cyan;
      ctx.beginPath();
      ctx.arc((p.x + 1) / 2 * S, (1 - p.y) / 2 * S, S * 0.025, 0, Math.PI * 2);
      ctx.fill();
    }
  };
}

function pathfindSim(ctx, S) {
  // A 21x21 perfect maze, searched by A* from corner to corner, one expansion per tick. The frontier
  // (cyan) and expanded cells (dim purple) spread over the maze, then the path lights up pink. A new
  // maze is dealt after a short hold.
  const n = 21, cell = S / n;
  let race = null, hold = 0;
  const deal = () => {
    const seed = 1 + Math.floor(Math.random() * 99999);
    const m = makeMaze('prim', n, n, seed);
    const res = run('astar', m.grid, m.start, m.goal, { heur: 'octile' });
    race = { grid: m.grid, res, state: new Uint8Array(n * n), i: 0, done: false };
    hold = 0;
  };
  deal();
  return (speed) => {
    ctx.fillStyle = C.bg;
    ctx.fillRect(0, 0, S, S);
    const { grid, res, state } = race;
    for (let i = 0; i < n * n; i++) {
      const x = (i % n) * cell, y = Math.floor(i / n) * cell;
      if (grid.cells[i] === 0) { // 0 is a wall in pathfind-core
        ctx.fillStyle = 'rgba(0,245,255,0.35)';
        ctx.fillRect(x + 1, y + 1, cell - 2, cell - 2);
      } else if (state[i] === 2) {
        ctx.fillStyle = 'rgba(155,0,255,0.35)';
        ctx.fillRect(x, y, cell, cell);
      } else if (state[i] === 1) {
        ctx.fillStyle = 'rgba(0,245,255,0.75)';
        ctx.fillRect(x, y, cell, cell);
      }
    }
    if (!race.done) {
      const to = Math.min(res.steps.length, race.i + Math.max(1, Math.round(speed * 0.6)));
      for (; race.i < to; race.i++) {
        const [c, , fresh] = res.steps[race.i];
        state[c] = 2;
        for (const [v] of fresh) if (!state[v]) state[v] = 1;
      }
      if (race.i >= res.steps.length) race.done = true;
      return;
    }
    hold += speed;
    const k = Math.min(res.path.length, Math.floor(hold / 5));
    if (k > 1) {
      ctx.strokeStyle = C.pink;
      ctx.lineWidth = 3;
      ctx.shadowColor = C.pink;
      ctx.shadowBlur = 8;
      ctx.beginPath();
      for (let j = 0; j < k; j++) {
        const p = res.path[j];
        const x = ((p % n) + 0.5) * cell, y = (Math.floor(p / n) + 0.5) * cell;
        if (j) ctx.lineTo(x, y); else ctx.moveTo(x, y);
      }
      ctx.stroke();
      ctx.shadowBlur = 0;
    }
    if (hold > 220) deal();
  };
}

function mdplabSim(ctx, S) {
  // A 6-by-5 grid with a reward, a pit and a wall. Value iteration runs a few sweeps per tick, so the heat
  // spreads out from the reward and the arrows settle; when the values converge the sim restarts. Cheap:
  // 26 states, and only a couple of sweeps a frame. Colours come from the landing page's palette C.
  const rows = ['......', '.#..-.', '..#...', '.#...+', 'S.....'];
  const W = 6, H = 5, gamma = 0.95, living = -0.04, slip = 0.1;
  const kind = rows.join('');
  const dirs = [[0, -1], [1, 0], [0, 1], [-1, 0]];
  const pay = (k) => (k === '+' ? 1 : k === '-' ? -1 : 0);
  const term = (i) => kind[i] === '+' || kind[i] === '-';
  const move = (s, d) => {
    const x = (s % W) + dirs[d][0], y = ((s / W) | 0) + dirs[d][1];
    if (x < 0 || y < 0 || x >= W || y >= H || kind[y * W + x] === '#') return s;
    return y * W + x;
  };
  const states = [];
  for (let i = 0; i < W * H; i++) if (kind[i] !== '#' && !term(i)) states.push(i);
  const P = {};
  for (const s of states) {
    P[s] = [0, 1, 2, 3].map((a) => [[1 - slip, a], [slip / 2, (a + 3) % 4], [slip / 2, (a + 1) % 4]]
      .map(([p, d]) => { const t = move(s, d); return [p, t, living + pay(kind[t])]; }));
  }
  let V, sweeps, done, hold, greedy;
  const reset = () => { V = Array(W * H).fill(0); sweeps = 0; done = false; hold = 0; greedy = Array(W * H).fill(-1); };
  function sweep() {
    const next = V.slice();
    let res = 0;
    for (const s of states) {
      let best = -Infinity, arg = 0;
      for (let a = 0; a < 4; a++) {
        let q = 0;
        for (const [p, t, r] of P[s][a]) q += p * (r + gamma * V[t]); // terminal V stays 0
        if (q > best) { best = q; arg = a; }
      }
      res = Math.max(res, Math.abs(best - V[s]));
      next[s] = best;
      greedy[s] = arg;
    }
    V = next;
    sweeps++;
    done = res < 1e-4;
  }
  reset();
  return (speed) => {
    ctx.fillStyle = C.bg;
    ctx.fillRect(0, 0, S, S);
    hold += speed;
    if (!done) {
      if (hold >= 4) { hold = 0; sweep(); }
    } else if (hold > 120) {
      reset();
    }
    const vmax = Math.max(0.05, ...states.map((s) => Math.abs(V[s])));
    const cell = S / W, oy = (S - H * cell) / 2;
    for (let i = 0; i < W * H; i++) {
      const x = (i % W) * cell, y = oy + ((i / W) | 0) * cell, k = kind[i];
      if (k === '#') {
        ctx.fillStyle = C.dim;
        ctx.globalAlpha = 0.5;
        ctx.fillRect(x + 2, y + 2, cell - 4, cell - 4);
        ctx.globalAlpha = 1;
        continue;
      }
      if (k === '+') ctx.fillStyle = C.green;
      else if (k === '-') ctx.fillStyle = C.pink;
      else {
        const t = Math.min(1, Math.abs(V[i]) / vmax), a = 0.1 + 0.7 * t;
        ctx.fillStyle = V[i] >= 0 ? `rgba(0,245,255,${a})` : `rgba(255,0,160,${a})`;
      }
      ctx.fillRect(x + 1, y + 1, cell - 2, cell - 2);
    }
    ctx.strokeStyle = C.yellow;
    ctx.lineWidth = 2;
    for (const s of states) {
      if (greedy[s] < 0) continue;
      const cx = (s % W) * cell + cell / 2, cy = oy + ((s / W) | 0) * cell + cell / 2;
      const [dx, dy] = dirs[greedy[s]], L = cell * 0.3;
      ctx.beginPath();
      ctx.moveTo(cx - dx * L, cy - dy * L);
      ctx.lineTo(cx + dx * L, cy + dy * L);
      ctx.stroke();
    }
    ctx.fillStyle = C.dim;
    ctx.font = '10px monospace';
    ctx.fillText(`sweep ${sweeps}`, 6, S - 6);
  };
}

function optlabSim(ctx, S) {
  // Cabinet screen for the optimizer race: three optimizers race down Rosenbrock's banana from one point, with
  // their step sizes from the lab's defaults scaled for this view. Paths are computed once at load (about 400
  // steps each), then revealed along their length, and the heat map is cached. No server calls.
  const pad = 14, span = S - pad * 2;
  const px = (x) => pad + ((x + 2) / 4) * span; // view box: x in [-2, 2], y in [-1, 3]
  const py = (y) => pad + ((3 - y) / 4) * span;
  const gradOf = (x, y) => {
    const r = y - x * x;
    return [-2 * (1 - x) - 400 * x * r, 200 * r];
  };
  const lossOf = (x, y) => (1 - x) * (1 - x) + 100 * ((y - x * x) * (y - x * x));
  const runs = [
    { c: '#00f5ff', x: -1.5, y: 2, opt: 'sgd', lr: 0.002 },
    { c: '#00ff88', x: -1.5, y: 2, opt: 'momentum', lr: 0.001, b: 0.9 },
    { c: '#b56bff', x: -1.5, y: 2, opt: 'adam', lr: 0.05 },
  ].map((r) => {
    const pts = [[r.x, r.y]];
    let x = r.x, y = r.y, vx = 0, vy = 0, m = [0, 0], v = [0, 0], t = 0;
    for (let k = 0; k < 400; k++) {
      const [gx, gy] = gradOf(x, y);
      if (r.opt === 'sgd') { x -= r.lr * gx; y -= r.lr * gy; }
      else if (r.opt === 'momentum') {
        vx = r.b * vx - r.lr * gx; vy = r.b * vy - r.lr * gy; x += vx; y += vy;
      } else {
        t += 1;
        const g = [gx, gy];
        const nx = [x, y];
        for (let i = 0; i < 2; i++) {
          m[i] = 0.9 * m[i] + 0.1 * g[i];
          v[i] = 0.999 * v[i] + 0.001 * (g[i] * g[i]);
          nx[i] -= r.lr * (m[i] / (1 - 0.9 ** t)) / (Math.sqrt(v[i] / (1 - 0.999 ** t)) + 1e-8);
        }
        [x, y] = nx;
      }
      if (!Number.isFinite(x) || !Number.isFinite(y) || Math.abs(x) > 1e6 || Math.abs(y) > 1e6) break;
      pts.push([x, y]);
    }
    return { ...r, pts };
  });
  // Cached heat map: log loss on a 48 by 48 grid, scaled up with smoothing.
  const N = 48;
  const heat = document.createElement('canvas');
  heat.width = N; heat.height = N;
  const hctx = heat.getContext('2d');
  const img = hctx.createImageData(N, N);
  for (let r = 0; r < N; r++) for (let c = 0; c < N; c++) {
    const x = -2 + ((c + 0.5) / N) * 4, y = 3 - ((r + 0.5) / N) * 4;
    const t = Math.min(1, Math.max(0, Math.log10(lossOf(x, y) + 0.01) / 4 + 0.5));
    const i = 4 * (r * N + c);
    img.data[i] = 20 + 90 * t; img.data[i + 1] = 10 + 40 * (1 - t); img.data[i + 2] = 60 + 150 * (1 - t);
    img.data[i + 3] = 255;
  }
  hctx.putImageData(img, 0, 0);
  let k = 0, hold = 0;
  return (speed) => {
    ctx.fillStyle = '#020610';
    ctx.fillRect(0, 0, S, S);
    ctx.imageSmoothingEnabled = true;
    ctx.drawImage(heat, pad, pad, span, span);
    const reach = Math.floor(k);
    for (const r of runs) {
      const n = Math.min(reach, r.pts.length - 1);
      if (n < 1) continue;
      ctx.strokeStyle = r.c; ctx.lineWidth = 2; ctx.shadowColor = r.c; ctx.shadowBlur = 8;
      ctx.beginPath();
      ctx.moveTo(px(r.pts[0][0]), py(r.pts[0][1]));
      for (let i = 1; i <= n; i++) ctx.lineTo(px(r.pts[i][0]), py(r.pts[i][1]));
      ctx.stroke();
      const [hx, hy] = r.pts[n];
      ctx.fillStyle = r.c; ctx.beginPath(); ctx.arc(px(hx), py(hy), 3.5, 0, Math.PI * 2); ctx.fill();
    }
    ctx.shadowBlur = 0;
    ctx.fillStyle = '#ffffff';
    ctx.beginPath(); ctx.arc(px(1), py(1), 3, 0, Math.PI * 2); ctx.fill();
    const longest = Math.max(...runs.map((r) => r.pts.length - 1));
    k += 2.2 * speed;
    if (k >= longest) { hold += 1; if (hold > 90) { k = 0; hold = 0; } }
  };
}

function treelabSim(ctx, S) {
  // A depth-2 tree on the XOR pattern: a laser cuts x = 0.5, then y = 0.5, and the four quadrants shade in by
  // class. The Tree lab grows the same kind of tree with the full exhaustive split search on its own page.
  let seed = 11;
  const rnd = () => (seed = (seed * 16807) % 2147483647) / 2147483647;
  let pts, step, phase, hold;
  const reset = () => {
    pts = Array.from({ length: 48 }, () => {
      const x = rnd(), y = rnd();
      return [x, y, (x > 0.5) !== (y > 0.5) ? 1 : 0];
    });
    step = 0; phase = 0; hold = 0;
  };
  reset();
  return (speed) => {
    ctx.fillStyle = C.bg;
    ctx.fillRect(0, 0, S, S);
    const shade = Math.min(1, step / 2);
    for (let i = 0; i < 2; i++) {
      for (let j = 0; j < 2; j++) {
        ctx.globalAlpha = 0.28 * shade;
        ctx.fillStyle = (i ^ j) ? C.pink : C.cyan;
        ctx.fillRect(i * S / 2, j * S / 2, S / 2, S / 2);
      }
    }
    ctx.globalAlpha = 1;
    for (const [x, y, c] of pts) {
      ctx.fillStyle = c ? C.pink : C.cyan;
      ctx.beginPath();
      ctx.arc(x * S, (1 - y) * S, 2.6, 0, Math.PI * 2);
      ctx.fill();
    }
    if (step < 2) {
      phase = Math.min(1, phase + speed / 40);
      ctx.strokeStyle = C.yellow; ctx.shadowColor = C.yellow; ctx.shadowBlur = 10; ctx.lineWidth = 3;
      ctx.beginPath();
      if (step === 0) { ctx.moveTo(S / 2, 0); ctx.lineTo(S / 2, S * phase); }
      else { ctx.moveTo(0, S / 2); ctx.lineTo(S * phase, S / 2); }
      ctx.stroke();
      ctx.shadowBlur = 0;
      if (phase >= 1 && (hold += speed) > 50) { step++; phase = 0; hold = 0; }
    } else if ((hold += speed) > 90) reset();
  };
}

function ghosthuntSim(ctx, S) {
  // Cabinet preview of Ghost Hunt: an 11x11 maze, one ghost drifting at random, and a cheap belief over its cell.
  // The belief is a one-pass forward step (random-walk predict, discrete-Gaussian sonar update) on cells only, so
  // the cabinet shows the fog and the bust without the full heading-aware filter the page runs. Ping noise uses
  // Math.random: this is a picture, not a parity-checked result.
  const n = 11, N = n * n, cs = S / n, SIGMA = 1.0;
  let walls, pos, ghost, belief, ring, wait, hunt = 0;
  const nb = (c) => {
    const r = (c / n) | 0, k = c % n, out = [];
    if (r > 0 && !walls[c - n]) out.push(c - n);
    if (k < n - 1 && !walls[c + 1]) out.push(c + 1);
    if (r < n - 1 && !walls[c + n]) out.push(c + n);
    if (k > 0 && !walls[c - 1]) out.push(c - 1);
    return out;
  };
  // Distances from one cell to all others (BFS over the corridors).
  const dists = (src) => {
    const d = new Int16Array(N).fill(-1);
    d[src] = 0;
    const q = [src];
    for (let i = 0; i < q.length; i++) for (const v of nb(q[i])) if (d[v] < 0) { d[v] = d[q[i]] + 1; q.push(v); }
    return d;
  };
  // A small perfect maze by depth-first carving on the odd cells.
  const carve = () => {
    const w = new Uint8Array(N).fill(1), seen = new Uint8Array(N);
    const start = n + 1;
    w[start] = 0; seen[start] = 1;
    const stack = [start];
    while (stack.length) {
      const c = stack[stack.length - 1], r = (c / n) | 0, k = c % n;
      const opts = [[-2 * n, -n], [2, 1], [2 * n, n], [-2, -1]].filter(([dd]) => {
        const t = c + dd;
        const tr = (t / n) | 0, tk = t % n;
        return tr > 0 && tr < n - 1 && tk > 0 && tk < n - 1 && !seen[t] && r >= 0 && k >= 0;
      });
      if (!opts.length) { stack.pop(); continue; }
      const [dd, mid] = opts[(Math.random() * opts.length) | 0];
      w[c + mid] = 0; w[c + dd] = 0; seen[c + dd] = 1;
      stack.push(c + dd);
    }
    return w;
  };
  const newHunt = () => {
    walls = carve();
    const open = [];
    for (let c = 0; c < N; c++) if (!walls[c]) open.push(c);
    pos = n + 1;
    const far = dists(pos);
    const cand = open.filter((c) => far[c] >= 4);
    ghost = cand[(Math.random() * cand.length) | 0];
    belief = new Float64Array(N);
    for (const c of cand) belief[c] = 1 / cand.length;
    ring = null;
    wait = 20;
    hunt += 1;
  };
  // One turn: the ghost drifts, the belief is predicted, a ping arrives and the belief is weighed by it.
  const turn = () => {
    const moves = nb(ghost).concat([ghost]);
    ghost = moves[(Math.random() * moves.length) | 0];
    const pred = new Float64Array(N);
    for (let c = 0; c < N; c++) {
      if (!belief[c]) continue;
      const opts = nb(c).concat([c]);
      for (const o of opts) pred[o] += belief[c] / opts.length;
    }
    const d = dists(pos);
    const reading = Math.max(0, Math.round(d[ghost] + SIGMA * Math.sqrt(-2 * Math.log(Math.random() + 1e-12)) * Math.cos(2 * Math.PI * Math.random())));
    let z = 0;
    for (let c = 0; c < N; c++) {
      if (d[c] < 0) { pred[c] = 0; continue; }
      const e = Math.exp(-((reading - d[c]) ** 2) / (2 * SIGMA * SIGMA));
      pred[c] *= e;
      z += pred[c];
    }
    if (z > 0) for (let c = 0; c < N; c++) pred[c] /= z;
    belief = pred;
    ring = { t: 0, r: reading + 0.5 };
  };
  // The player walks toward the most likely cell and fires when it is next door and the belief is sure.
  const step = () => {
    let peak = 0;
    for (let c = 1; c < N; c++) if (belief[c] > belief[peak]) peak = c;
    if (belief[peak] > 0.5 && (peak === pos || nb(pos).includes(peak))) {
      if (peak === ghost) { wait = 30; ring = { t: 0, r: 0, hit: true }; } else { ring = { t: 0, r: 0 }; }
      newHunt();
      return;
    }
    const d = dists(peak);
    const next = nb(pos).filter((c) => d[c] === d[pos] - 1);
    if (next.length) pos = next[0];
    turn();
  };
  newHunt();
  return (speed) => {
    if (wait > 0) { wait -= 1; }
    else step();
    if (ring) ring.t += 0.05 * speed;
    if (ring && ring.t > 1) ring = null;
    ctx.fillStyle = '#05060f';
    ctx.fillRect(0, 0, S, S);
    for (let c = 0; c < N; c++) {
      const r = (c / n) | 0, k = c % n;
      if (walls[c]) { ctx.fillStyle = '#1a0f2e'; ctx.fillRect(k * cs, r * cs, cs, cs); continue; }
      const p = belief[c];
      if (p > 0.004) {
        ctx.fillStyle = `rgba(255,0,160,${Math.min(0.9, Math.sqrt(p) * 1.1)})`;
        ctx.fillRect(k * cs + 1, r * cs + 1, cs - 2, cs - 2);
      } else {
        ctx.fillStyle = 'rgba(0,245,255,0.05)';
        ctx.fillRect(k * cs + 1, r * cs + 1, cs - 2, cs - 2);
      }
    }
    if (ring) {
      const px = ((pos % n) + 0.5) * cs, py = (((pos / n) | 0) + 0.5) * cs;
      ctx.strokeStyle = `rgba(0,245,255,${1 - ring.t})`;
      ctx.lineWidth = 2;
      ctx.beginPath(); ctx.arc(px, py, Math.max(2, ring.r * cs * ring.t * 2), 0, 6.283); ctx.stroke();
    }
    ctx.shadowColor = '#00f5ff'; ctx.shadowBlur = 10;
    ctx.fillStyle = '#e8fdff';
    ctx.beginPath(); ctx.arc(((pos % n) + 0.5) * cs, (((pos / n) | 0) + 0.5) * cs, cs * 0.3, 0, 6.283); ctx.fill();
    ctx.shadowBlur = 0;
  };
}

function cabinets() {
  const sims = { npuzzle: npuzzleSim, connect4: connect4Sim, checkers: checkersSim, routes: routesSim, g2048: g2048Sim, sudoku: sudokuSim, lightsout: lightsoutSim, blackjack: blackjackSim, battleship: battleshipSim, pacman: pacmanSim, warehouse: warehouseSim, endgame: endgameSim, sokoban: sokobanSim, wordle: wordleSim, poker: pokerSim, minesweeper: minesweeperSim, hexgame: hexgameSim, bandits: banditsSim, cartpole: cartpoleSim, queens: queensSim, snake: snakeSim, rover: roverSim, tetris: tetrisSim, nonogram: nonogramSim, clusters: clustersSim, nnlab: nnlabSim, pathfind: pathfindSim, mdplab: mdplabSim, optlab: optlabSim, treelab: treelabSim, ghosthunt: ghosthuntSim, markov: markovSim, regression: regressionSim, localize: localizeSim, walkers: walkersSim };
  document.querySelectorAll('.cab-screen').forEach((canvas) => {
    const S = 240, ctx = sizeCanvas(canvas, S, S), tick = sims[canvas.dataset.sim](ctx, S);
    const cab = canvas.closest('.cabinet');
    let speed = 1, visible = false, raf = 0;
    cab.addEventListener('pointerenter', () => { speed = 2.5; });
    cab.addEventListener('pointerleave', () => { speed = 1; });
    // speedFactor() is 10 during overdrive (overdrive.js), 1 otherwise.
    const loop = () => { tick(speed * speedFactor()); if (visible && !REDUCED) raf = requestAnimationFrame(loop); };
    if (REDUCED) { for (let i = 0; i < 400; i++) tick(1); return; }
    onVisible(cab, (v) => { visible = v; cancelAnimationFrame(raf); if (v) loop(); });
  });
}

boot();
city();
cabinets();

// A mouse over the glitch title zaps (sfx.js). The CSS glitch itself is unchanged.
document.getElementById('hero-title')?.addEventListener('pointerenter', (e) => { if (e.pointerType === 'mouse') play('zap'); });
