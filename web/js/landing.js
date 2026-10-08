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

function cabinets() {
  const sims = { npuzzle: npuzzleSim, connect4: connect4Sim, checkers: checkersSim, g2048: g2048Sim, sudoku: sudokuSim };
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
