// Landing page: A* across a neon city grid that chases the pointer, a one-time
// boot sequence, and four arcade cabinets each animating its agent's mechanism.
// Everything pauses while off-screen and honors prefers-reduced-motion.

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
  }

  function frame() {
    t++;
    step(REDUCED ? Infinity : 30);
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

function cabinets() {
  const sims = { npuzzle: npuzzleSim, connect4: connect4Sim, checkers: checkersSim, routes: routesSim, g2048: g2048Sim, sudoku: sudokuSim, lightsout: lightsoutSim, blackjack: blackjackSim, battleship: battleshipSim, pacman: pacmanSim, warehouse: warehouseSim, endgame: endgameSim, sokoban: sokobanSim, wordle: wordleSim, poker: pokerSim, minesweeper: minesweeperSim, hexgame: hexgameSim, bandits: banditsSim, cartpole: cartpoleSim, queens: queensSim, snake: snakeSim, rover: roverSim, tetris: tetrisSim };
  document.querySelectorAll('.cab-screen').forEach((canvas) => {
    const S = 240, ctx = sizeCanvas(canvas, S, S), tick = sims[canvas.dataset.sim](ctx, S);
    const cab = canvas.closest('.cabinet');
    let speed = 1, visible = false, raf = 0;
    cab.addEventListener('pointerenter', () => { speed = 2.5; });
    cab.addEventListener('pointerleave', () => { speed = 1; });
    const loop = () => { tick(speed); if (visible && !REDUCED) raf = requestAnimationFrame(loop); };
    if (REDUCED) { for (let i = 0; i < 400; i++) tick(1); return; }
    onVisible(cab, (v) => { visible = v; cancelAnimationFrame(raf); if (v) loop(); });
  });
}

boot();
city();
cabinets();
