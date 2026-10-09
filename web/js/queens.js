// N-Queens page.
//
// One queen per row. Two queens attack each other when they share a column or a diagonal. The board
// keeps three count arrays (column, down-right diagonal, down-left diagonal), exactly as queens/board.py
// does, so the attacks on any square are three array reads. That is what lets the heat map, the hint and
// the live min-conflicts loop all run in the browser without the server.
//
// Modes:
//   PLAY   place the queens yourself on an 8x8 board, with heat and hints
//   WATCH  run one agent on the server and step through its frames
//   RACE   run all four agents on the same seeded start and compare them
//   BIG N  min-conflicts on 500 to 2,000 queens, one repair per tick, drawn one pixel per square

import { getJSON, postJSON } from './api.js';
import { banner, burst, pop, shake } from './fx.js';

const $ = (id) => document.getElementById(id);

const AGENTS = {
  backtrack: {
    label: 'BACKTRACKING',
    blurb: 'Systematic. It places one queen per row in the lowest free column and backs up when a row has none. Three bitmasks say which columns and diagonals are taken. It can prove that no solution exists, but the tree grows explosively.',
    tip: 'It takes the lowest open column; a row with none sends it back up.',
    steps: 'a queen placed or taken back',
    work: 'placements examined',
  },
  hill: {
    label: 'HILL CLIMBING',
    blurb: 'Steepest ascent on the conflict count. Each step scores every move (N squares per row, N rows) and plays the best one. When no move lowers the count it is on a plateau or a local minimum, so it restarts from a random board.',
    tip: 'Steepest ascent: it scored every move and played the best.',
    steps: 'moves played (restarts count)',
    work: 'squares scored',
  },
  anneal: {
    label: 'SIMULATED ANNEALING',
    blurb: 'Proposes one random queen move at a time. Moves that do not add conflicts are always taken; a move that adds d conflicts is taken with probability exp(-d/T). T cools over each run, so early on the search wanders and late on it settles.',
    tip: 'Annealing: an uphill move is taken with probability exp(-d/T).',
    steps: 'proposed moves',
    work: 'squares scored',
  },
  minconf: {
    label: 'MIN-CONFLICTS',
    blurb: 'Pick a queen in conflict and move it to the column with the fewest attacks. The start is a greedy placement that leaves very few conflicts, so the repairs are few, and that is why it scales.',
    tip: 'Min-conflicts: the queen moved to a least-attacked column.',
    steps: 'repairs',
    work: 'squares scored',
  },
};
const AGENT_ORDER = ['backtrack', 'hill', 'anneal', 'minconf'];

let maxN = { backtrack: 64, hill: 100, anneal: 2000, minconf: 10000 };

// ── Board arithmetic (the same as queens/board.py) ───────────────────────

// Line counts for a placement; cols[r] = column of the row-r queen, or -1 for an empty row.
function lineCounts(n, cols) {
  const col = new Int32Array(n), d1 = new Int32Array(2 * n - 1), d2 = new Int32Array(2 * n - 1);
  for (let r = 0; r < n; r++) {
    const c = cols[r];
    if (c >= 0) { col[c]++; d1[r + c]++; d2[r - c + n - 1]++; }
  }
  return { col, d1, d2 };
}

// Queens attacking square (r, c), not counting the row-r queen. Same rule as Board.attacks_if.
function attacksOn(n, L, cols, r, c) {
  let a = L.col[c] + L.d1[r + c] + L.d2[r - c + n - 1];
  if (cols[r] === c) a -= 3;
  return a;
}

// Attacking pairs: each line with k queens holds k(k-1)/2 of them.
function conflictsOf(n, cols) {
  const L = lineCounts(n, cols);
  let t = 0;
  for (const arr of [L.col, L.d1, L.d2]) for (const k of arr) t += k * (k - 1) / 2;
  return t;
}

const sqName = (r, c) => `${String.fromCharCode(97 + c)}${r + 1}`;

// The min-conflicts move in words: the most attacked queen (first row on a tie) and its best column.
function hintMove(n, cols) {
  const L = lineCounts(n, cols);
  let pick = -1, pa = 0;
  for (let r = 0; r < n; r++) {
    if (cols[r] < 0) continue;
    const a = attacksOn(n, L, cols, r, cols[r]);
    if (a > pa) { pa = a; pick = r; }
  }
  const bestColumn = (r) => {
    let bc = 0, ba = Infinity;
    for (let c = 0; c < n; c++) {
      const a = attacksOn(n, L, cols, r, c);
      if (a < ba) { ba = a; bc = c; }
    }
    return { c: bc, a: ba };
  };
  if (pick >= 0) {
    const b = bestColumn(pick);
    return { kind: 'move', r: pick, from: cols[pick], to: b.c, before: pa, after: b.a };
  }
  const empty = cols.findIndex((c) => c < 0);
  if (empty >= 0) {
    const b = bestColumn(empty);
    return { kind: 'place', r: empty, to: b.c, after: b.a };
  }
  return { kind: 'done' };
}

// ── Board rendering ──────────────────────────────────────────────────────

// Build n*n squares inside el. Returns the square elements, indexed r * n + c.
function buildGrid(el, n, onClick) {
  el.innerHTML = '';
  el.style.setProperty('--n', n);
  const squares = [];
  for (let r = 0; r < n; r++) {
    for (let c = 0; c < n; c++) {
      const b = document.createElement('button');
      b.type = 'button';
      b.className = 'qn-sq' + ((r + c) & 1 ? ' dark-sq' : '');
      b.dataset.r = r;
      b.dataset.c = c;
      b.setAttribute('aria-label', `${sqName(r, c)}`);
      if (onClick) b.addEventListener('click', () => onClick(r, c));
      el.appendChild(b);
      squares.push(b);
    }
  }
  return squares;
}

// Paint placement cols onto the squares: queens (red when attacked), heat on empty squares, and
// optional highlights: {hint: {r, c}, moved: row, solved: bool}.
function paint(el, squares, n, cols, marks = {}) {
  const L = lineCounts(n, cols);
  for (let r = 0; r < n; r++) {
    for (let c = 0; c < n; c++) {
      const sq = squares[r * n + c];
      const isQueen = cols[r] === c;
      let cls = 'qn-sq' + ((r + c) & 1 ? ' dark-sq' : '');
      sq.textContent = '';
      sq.style.removeProperty('--heat');
      if (isQueen) {
        const bad = attacksOn(n, L, cols, r, c) > 0;
        cls += ' queen' + (bad ? ' bad' : ' good-fit');
      } else {
        const h = attacksOn(n, L, cols, r, c);
        if (h > 0) {
          cls += ' attacked';
          cls += ' hot';
          sq.style.setProperty('--heat', `rgba(255,0,160,${Math.min(0.3, 0.04 + 0.07 * h).toFixed(3)})`);
        }
      }
      if (marks.hint && marks.hint.r === r && marks.hint.c === c) cls += ' hint';
      if (marks.moved === r) cls += ' moved';
      sq.className = cls;
      sq.setAttribute('aria-label', `${sqName(r, c)}${isQueen ? (cls.includes('bad') ? ', queen under attack' : ', queen') : ''}`);
    }
  }
  el.classList.toggle('solved', !!marks.solved);
}

// ── Charts ───────────────────────────────────────────────────────────────

// A polyline of (xs, ys) with a cursor at index `cur`. Axes are labelled with their maxima.
function drawChart(canvas, xs, ys, cur = -1, color = '#00f5ff', label = '') {
  const ctx = canvas.getContext('2d');
  const W = canvas.width, H = canvas.height, pad = 26;
  ctx.clearRect(0, 0, W, H);
  ctx.strokeStyle = 'rgba(0,245,255,0.12)';
  ctx.lineWidth = 1;
  for (let i = 1; i < 4; i++) {
    const y = pad + ((H - 2 * pad) * i) / 4;
    ctx.beginPath(); ctx.moveTo(pad, y); ctx.lineTo(W - pad, y); ctx.stroke();
  }
  if (!xs.length) return;
  const xmax = Math.max(1, xs[xs.length - 1]);
  const ymax = Math.max(1, ...ys);
  const X = (x) => pad + (x / xmax) * (W - 2 * pad);
  const Y = (y) => H - pad - (y / ymax) * (H - 2 * pad);
  ctx.strokeStyle = color;
  ctx.lineWidth = 2;
  ctx.shadowColor = color; ctx.shadowBlur = 8;
  ctx.beginPath();
  xs.forEach((x, i) => (i ? ctx.lineTo(X(x), Y(ys[i])) : ctx.moveTo(X(x), Y(ys[i]))));
  ctx.stroke();
  ctx.shadowBlur = 0;
  if (cur >= 0 && cur < xs.length) {
    const cx = X(xs[cur]), cy = Y(ys[cur]);
    ctx.strokeStyle = '#ff00a0';
    ctx.beginPath(); ctx.moveTo(cx, pad); ctx.lineTo(cx, H - pad); ctx.stroke();
    ctx.fillStyle = '#ff00a0';
    ctx.beginPath(); ctx.arc(cx, cy, 4, 0, Math.PI * 2); ctx.fill();
  }
  ctx.fillStyle = '#4a7a9b';
  ctx.font = '12px JetBrains Mono, monospace';
  ctx.fillText(`${ymax}`, 4, pad + 4);
  ctx.fillText('0', 4, H - pad);
  ctx.fillText(`${label || 'step'} ${xmax}`, W - pad - 90, H - 6);
}

// ── Mode switching ───────────────────────────────────────────────────────

const modes = ['play', 'watch', 'race', 'big'];
const modeNames = { play: 'PLAY', watch: 'WATCH', race: 'RACE', big: 'BIG N' };

function setMode(mode) {
  stopWatch();
  stopBig();
  for (const m of modes) {
    $(`mode-${m}`).classList.toggle('hidden', m !== mode);
    const tab = $(`tab-${m}`);
    tab.setAttribute('aria-selected', String(m === mode));
    tab.classList.toggle('act', m === mode);
  }
  $('chip-mode').textContent = modeNames[mode];
  if (mode === 'big' && !big.cols) bigNew();
}

// ── PLAY ─────────────────────────────────────────────────────────────────

const play = { n: 8, cols: new Array(8).fill(-1), squares: [], moves: 0, solved: false };

function playRender(hint = null) {
  const conf = conflictsOf(play.n, play.cols);
  const placed = play.cols.filter((c) => c >= 0).length;
  const solved = placed === play.n && conf === 0;
  paint($('play-board'), play.squares, play.n, play.cols, { hint, solved });
  $('chip-conf').textContent = `${conf}`;
  $('chip-steps').textContent = `${play.moves}`;
  $('play-status').textContent = solved
    ? `Solved in ${play.moves} moves. Every row has a queen and nothing attacks anything.`
    : `Placed ${placed} of ${play.n} queens, ${conf} conflict${conf === 1 ? '' : 's'}`;
  if (solved && !play.solved) {
    play.solved = true;
    banner('SOLVED', `${play.n} queens, no two attacking, in ${play.moves} moves`, '#00ff88');
    burst($('play-board'), { count: 90 });
  }
  if (!solved) play.solved = false;
}

function playClick(r, c) {
  if (play.cols[r] === c) {
    play.cols[r] = -1; // the same square takes the queen back
  } else {
    play.cols[r] = c; // a square in a row that has a queen moves it
  }
  play.moves++;
  $('play-hint-text').textContent = 'Press HINT. The hint is the min-conflicts move: take a conflicted queen and put it on the column with the fewest attacks.';
  playRender();
}

function playHint() {
  const h = hintMove(play.n, play.cols);
  if (h.kind === 'done') {
    $('play-hint-text').textContent = 'Nothing to fix: every row has a queen and no queen is attacked.';
    playRender();
    return;
  }
  if (h.kind === 'move') {
    $('play-hint-text').textContent = `Move the row ${h.r + 1} queen from ${sqName(h.r, h.from)} to ${sqName(h.r, h.to)}: ` +
      `${h.before} attack${h.before === 1 ? '' : 's'} now, ${h.after} there.`;
    playRender({ r: h.r, c: h.to });
  } else {
    $('play-hint-text').textContent = `Row ${h.r + 1} needs a queen. ${sqName(h.r, h.to)} is attacked by ${h.after} queen${h.after === 1 ? '' : 's'}; the least attacked square in the row.`;
    playRender({ r: h.r, c: h.to });
  }
}

function playShowSolution() {
  // A full min-conflicts run from a greedy start at 8 queens, shown on the board.
  const rng = mulberry32(Math.floor(Math.random() * 1e9));
  const run = localMinConflicts(8, rng);
  play.cols = Array.from(run.cols);
  play.moves = 0;
  $('play-hint-text').textContent = `A min-conflicts solution, found in ${run.steps} repair${run.steps === 1 ? '' : 's'}.`;
  playRender();
}

function playClear() {
  play.cols = new Array(play.n).fill(-1);
  play.moves = 0;
  play.solved = false;
  $('play-hint-text').textContent = 'Press HINT. The hint is the min-conflicts move: take a conflicted queen and put it on the column with the fewest attacks.';
  playRender();
}

// ── Min-conflicts in the browser (the same algorithm as queens/agents.py) ─

// A small seeded generator, so a seed always gives the same run.
function mulberry32(a) {
  return function () {
    a |= 0; a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

const GREEDY_TRIES = 4096;
const REJECTION_TRIES = 4000;

// The solver state for one board. Arrays match Board: col, d1 (r + c), d2 (r - c + n - 1).
class MinConflicts {
  constructor(n, seed) {
    this.n = n;
    this.rng = mulberry32(seed);
    this.col = new Int32Array(n);
    this.d1 = new Int32Array(2 * n - 1);
    this.d2 = new Int32Array(2 * n - 1);
    this.cols = new Int32Array(n).fill(-1);
    this.steps = 0;
    this.restarts = 0;
    this.start();
  }

  // Greedy start: each row takes the least attacked of up to GREEDY_TRIES random unused columns.
  // Unused columns are kept in a list, because only an unused column can be attack-free.
  start() {
    const { n, col, d1, d2, cols, rng } = this;
    col.fill(0); d1.fill(0); d2.fill(0); cols.fill(-1);
    this.conflicts = 0;
    const free = Array.from({ length: n }, (_, i) => i);
    const pos = Array.from({ length: n }, (_, i) => i);
    for (let r = 0; r < n; r++) {
      let bestC = -1, bestA = Infinity;
      const m = free.length;
      for (let t = 0; t < GREEDY_TRIES; t++) {
        const c = free[Math.floor(rng() * m)];
        const a = d1[r + c] + d2[r - c + n - 1];
        if (a < bestA) { bestA = a; bestC = c; if (a === 0) break; }
      }
      const i = pos[bestC], last = free[free.length - 1];
      free[i] = last; pos[last] = i; free.pop();
      this.conflicts += col[bestC] + d1[r + bestC] + d2[r - bestC + n - 1];
      this.place(r, bestC);
    }
    this.best = this.conflicts;
    this.stall = 0;
  }

  place(r, c) {
    const n = this.n;
    this.col[c]++; this.d1[r + c]++; this.d2[r - c + n - 1]++;
    this.cols[r] = c;
  }

  drop(r, c) {
    const n = this.n;
    this.col[c]--; this.d1[r + c]--; this.d2[r - c + n - 1]--;
    this.cols[r] = -1;
    return this.col[c] + this.d1[r + c] + this.d2[r - c + n - 1];
  }

  // Conflicted rows: those whose queen has more than its own three counts.
  queenSum(r) {
    const c = this.cols[r], n = this.n;
    return this.col[c] + this.d1[r + c] + this.d2[r - c + n - 1];
  }

  // One repair: pick a conflicted queen (uniform), lift it, score its row, put it on a least-attacked column.
  repair() {
    const { n, col, d1, d2, cols, rng } = this;
    let r = -1;
    for (let t = 0; t < REJECTION_TRIES; t++) {
      const cand = Math.floor(rng() * n);
      if (this.queenSum(cand) > 3) { r = cand; break; }
    }
    if (r < 0) {
      const bad = [];
      for (let rr = 0; rr < n; rr++) if (this.queenSum(rr) > 3) bad.push(rr);
      r = bad[Math.floor(rng() * bad.length)];
    }
    const from = cols[r];
    this.conflicts -= this.drop(r, from);
    // Score every column for row r; ties are broken uniformly by reservoir sampling.
    let best = Infinity, to = from, ties = 0;
    for (let c = 0; c < n; c++) {
      const v = col[c] + d1[r + c] + d2[r - c + n - 1];
      if (v < best) { best = v; to = c; ties = 1; }
      else if (v === best) { ties++; if (rng() * ties < 1) to = c; }
    }
    this.conflicts += col[to] + d1[r + to] + d2[r - to + n - 1];
    this.place(r, to);
    this.steps++;
    if (this.conflicts < this.best) { this.best = this.conflicts; this.stall = 0; }
    else this.stall++;
    // A start that has not reached a new low for max(200, 20 n) repairs is cycling on a plateau.
    if (this.stall > Math.max(200, 20 * n) && this.conflicts > 0) {
      this.restarts++;
      this.start();
    }
    return { r, from, to };
  }
}

function localMinConflicts(n, rng) {
  // Used for PLAY's "show a solution": the same loop as the class, with a fixed generator.
  const s = new MinConflicts(n, Math.floor(rng() * 1e9));
  while (s.conflicts > 0 && s.steps < 100000) s.repair();
  return { cols: s.cols, steps: s.steps };
}

// ── BIG N: live min-conflicts ────────────────────────────────────────────

const big = { n: 1000, seed: 1, solver: null, squares: null, running: false, raf: 0, history: [], lastMoved: null, solved: false };

function bigNew() {
  stopBig();
  big.n = +$('big-n').value;
  big.seed = Math.floor(Math.random() * 1e9) + 1;
  big.solver = new MinConflicts(big.n, big.seed);
  big.history = [big.solver.conflicts];
  big.steps = [0];
  big.solved = false;
  big.lastMoved = null;
  const canvas = $('big-canvas');
  canvas.width = big.n;
  canvas.height = big.n;
  big.ctx = canvas.getContext('2d');
  big.ctx.fillStyle = '#02060f';
  big.ctx.fillRect(0, 0, big.n, big.n);
  big.ctx.fillStyle = '#00f5ff';
  for (let r = 0; r < big.n; r++) big.ctx.fillRect(big.solver.cols[r], r, 1, 1);
  big.cols = big.solver.cols;
  $('big-play').querySelector('.btn-txt').textContent = '▶ START';
  $('big-play').setAttribute('aria-pressed', 'false');
  bigStatus();
  bigChart();
}

function bigStatus() {
  const s = big.solver;
  $('chip-conf').textContent = `${s.conflicts}`;
  $('chip-steps').textContent = `${s.steps}`;
  if (big.solved) {
    $('big-status').textContent = `Solved: ${big.n.toLocaleString('en-US')} queens with ${s.steps} repairs (${s.restarts} restarts).`;
  } else {
    const left = s.conflicts.toLocaleString('en-US');
    $('big-status').textContent = `${big.n.toLocaleString('en-US')} queens, ${left} conflict${s.conflicts === 1 ? '' : 's'} left, ${s.steps.toLocaleString('en-US')} repairs so far.`;
  }
}

function bigChart() {
  // Keep the chart to a few hundred points by sampling the history.
  const h = big.history, st = big.steps;
  const stride = Math.max(1, Math.ceil(h.length / 600));
  const xs = [], ys = [];
  for (let i = 0; i < h.length; i += stride) { xs.push(st[i]); ys.push(h[i]); }
  if (xs[xs.length - 1] !== st[st.length - 1]) { xs.push(st[st.length - 1]); ys.push(h[h.length - 1]); }
  drawChart($('big-chart'), xs, ys, -1, '#00ff88', 'repair');
  $('big-chart-title').textContent = `Conflicts remaining, one point per repair (${big.n.toLocaleString('en-US')} queens)`;
}

function bigTick() {
  const s = big.solver;
  const per = +$('big-speed').value;
  for (let k = 0; k < per && s.conflicts > 0; k++) {
    const { r, from, to } = s.repair();
    const ctx = big.ctx;
    // Clear the old pixel, light the new one, and mark the moved queen white for one frame.
    ctx.fillStyle = '#02060f'; ctx.fillRect(from, r, 1, 1);
    ctx.fillStyle = '#00f5ff'; ctx.fillRect(to, r, 1, 1);
    big.history.push(s.conflicts);
    big.steps.push(s.steps);
    big.lastMoved = { r, to };
  }
  if (big.lastMoved) {
    const { r, to } = big.lastMoved;
    big.ctx.fillStyle = '#ffffff';
    big.ctx.fillRect(to, r, 1, 1);
  }
  // Check for the solve before writing the status, so the last line says Solved and not the old count.
  if (s.conflicts === 0 && !big.solved) {
    big.solved = true;
    stopBig();
    banner('SOLVED', `${big.n.toLocaleString('en-US')} queens in ${s.steps} repairs`, '#00ff88');
    burst($('big-canvas'), { count: 120 });
  }
  bigStatus();
  bigChart();
}

function bigLoop() {
  if (!big.running) return;
  bigTick();
  if (big.running) big.raf = requestAnimationFrame(bigLoop);
}

function startBig() {
  if (big.solved) bigNew();
  big.running = true;
  $('big-play').querySelector('.btn-txt').textContent = '❚❚ PAUSE';
  $('big-play').setAttribute('aria-pressed', 'true');
  big.raf = requestAnimationFrame(bigLoop);
}

function stopBig() {
  big.running = false;
  cancelAnimationFrame(big.raf);
  const b = $('big-play');
  if (b) {
    b.querySelector('.btn-txt').textContent = '▶ START';
    b.setAttribute('aria-pressed', 'false');
  }
}

// ── WATCH ────────────────────────────────────────────────────────────────

const watch = { n: 8, agent: 'minconf', frames: [], steps: [], values: [], idx: 0, timer: 0 };

function watchValues(agent, n, frames) {
  // Backtracking never has a conflict, so its chart counts the rows still open.
  return frames.map((f) => (agent === 'backtrack' ? f.filter((c) => c < 0).length : conflictsOf(n, f)));
}

function watchSay(i) {
  const { frames, steps, values, agent, n } = watch;
  const a = AGENTS[agent];
  if (i === 0) return `Start. ${a.blurb}`;
  const prev = frames[i - 1], cur = frames[i];
  const changed = [];
  for (let r = 0; r < n; r++) if (prev[r] !== cur[r]) changed.push(r);
  const measure = agent === 'backtrack' ? 'rows open' : 'conflicts';
  let text;
  if (changed.length === 0) {
    text = agent === 'anneal' ? 'Proposal rejected: the uphill move was not accepted this time.' : 'No change.';
  } else if (changed.length === 1) {
    const r = changed[0];
    if (prev[r] < 0) text = `Row ${r + 1}: queen placed on ${sqName(r, cur[r])}.`;
    else if (cur[r] < 0) text = `Row ${r + 1}: queen taken back from ${sqName(r, prev[r])}.`;
    else text = `Row ${r + 1}: queen moved ${sqName(r, prev[r])} to ${sqName(r, cur[r])}.`;
  } else {
    text = `${changed.length} rows changed${agent === 'hill' ? ' (a restart from a random board)' : ''}.`;
  }
  text += ` ${measure}: ${values[i - 1]} to ${values[i]}.`;
  const gap = steps[i] - steps[i - 1];
  if (gap > 1) text += ` (${gap} steps since the last frame)`;
  return `${text} ${a.tip}`;
}

function watchShow(i) {
  const { frames, values, n, agent } = watch;
  if (!frames.length) return;
  watch.idx = Math.max(0, Math.min(frames.length - 1, i));
  const cur = frames[watch.idx];
  const solved = agent !== 'backtrack' ? values[watch.idx] === 0 && cur.every((c) => c >= 0)
    : cur.every((c) => c >= 0);
  const moved = watch.idx > 0 ? changedRow(frames[watch.idx - 1], cur) : null;
  paint($('watch-board'), watch.squares, n, cur, { moved, solved });
  $('watch-scrub').value = watch.idx;
  $('watch-count').textContent = `frame ${watch.idx + 1} of ${frames.length} · step ${watch.steps[watch.idx]}`;
  $('chip-conf').textContent = `${values[watch.idx]}`;
  $('chip-steps').textContent = `${watch.steps[watch.idx]}`;
  $('watch-say').textContent = watchSay(watch.idx);
  drawChart($('watch-chart'), watch.steps, values, watch.idx, agent === 'backtrack' ? '#ffe600' : '#00f5ff',
    'step');
  $('watch-chart-title').textContent = agent === 'backtrack' ? 'Rows still open' : 'Conflicts remaining';
  if (solved && !watch.solvedShown) {
    watch.solvedShown = true;
    banner('SOLVED', `${n} queens in ${watch.steps[watch.idx]} steps`, '#00ff88');
  }
}

function changedRow(prev, cur) {
  for (let r = 0; r < cur.length; r++) if (prev[r] !== cur[r]) return r;
  return null;
}

function stopWatch() {
  clearInterval(watch.timer);
  watch.timer = 0;
  const b = $('watch-play');
  if (b) {
    b.querySelector('.btn-txt').textContent = '▶ PLAY';
    b.setAttribute('aria-pressed', 'false');
  }
}

function watchToggle() {
  if (watch.timer) { stopWatch(); return; }
  if (!watch.frames.length) return;
  const b = $('watch-play');
  b.querySelector('.btn-txt').textContent = '❚❚ PAUSE';
  b.setAttribute('aria-pressed', 'true');
  watch.timer = setInterval(() => {
    if (watch.idx >= watch.frames.length - 1) { stopWatch(); return; }
    watchShow(watch.idx + 1);
  }, 260);
}

async function watchRun() {
  stopWatch();
  $('watch-err').textContent = '';
  const agent = $('watch-agent').value;
  const n = +$('watch-n').value;
  const seed = +$('watch-seed').value || 0;
  watch.agent = agent;
  watch.n = n;
  watch.solvedShown = false;
  $('watch-agent').title = AGENTS[agent].tip;
  $('watch-count').textContent = 'running…';
  try {
    const res = await postJSON('/api/queens/solve', { agent, n, seed, frames: true });
    watch.frames = res.frames || [];
    watch.steps = res.frame_steps || [];
    watch.values = watchValues(agent, n, watch.frames);
    watch.squares = buildGrid($('watch-board'), n);
    $('watch-scrub').max = Math.max(0, watch.frames.length - 1);
    const note = res.solved ? '' : ` It did not finish: ${res.reason}.`;
    $('watch-err').textContent = note ? note.trim() : '';
    watchShow(0);
    $('watch-count').textContent += ` · ${res.solved ? 'solved' : res.reason}`;
  } catch (e) {
    $('watch-err').textContent = e.message;
    $('watch-count').textContent = 'no run yet';
  }
}

// ── RACE ─────────────────────────────────────────────────────────────────

function raceCardHTML(agent, res, n) {
  const a = AGENTS[agent];
  if (!res) {
    return `<div class="qn-card na"><h3>${a.label}</h3><div class="big">—</div><div class="small">not served above ${maxN[agent].toLocaleString('en-US')} queens</div></div>`;
  }
  const cls = res.solved ? 'win' : 'lose';
  const verdict = res.solved ? 'SOLVED' : res.reason.toUpperCase();
  return `<div class="qn-card ${cls}">
    <h3>${a.label}</h3>
    <div class="big">${verdict}</div>
    <div class="small">${res.steps.toLocaleString('en-US')} ${a.steps}</div>
    <div class="small">${res.evaluations.toLocaleString('en-US')} ${a.work}</div>
    <div class="small">${res.seconds.toFixed(res.seconds < 1 ? 3 : 2)} s${res.restarts ? ` · ${res.restarts} restarts` : ''}</div>
  </div>`;
}

async function raceRun() {
  $('race-err').textContent = '';
  const n = +$('race-n').value;
  const seed = +$('race-seed').value || 0;
  $('race-title').textContent = `${n.toLocaleString('en-US')} queens, seed ${seed}: racing…`;
  $('race-cards').innerHTML = '';
  $('race-bars').innerHTML = '';
  const jobs = AGENT_ORDER.map((agent) => (n > maxN[agent]
    ? Promise.resolve(null)
    : postJSON('/api/queens/solve', { agent, n, seed, frames: false }).catch((e) => ({ error: e.message }))));
  const results = await Promise.all(jobs);
  $('race-cards').innerHTML = AGENT_ORDER.map((agent, i) => raceCardHTML(agent, results[i] && !results[i].error ? results[i] : null, n)).join('');
  const errors = results.filter((r) => r && r.error).map((r) => r.error);
  if (errors.length) $('race-err').textContent = errors[0];
  const served = AGENT_ORDER.map((agent, i) => ({ agent, res: results[i] })).filter((x) => x.res && !x.res.error);
  const maxSec = Math.max(0.001, ...served.map((x) => x.res.seconds));
  $('race-bars').innerHTML = served.map(({ agent, res }) => `
    <div class="qn-bar-row ${res.solved ? 'win' : ''}">
      <span>${AGENTS[agent].label}</span>
      <span class="bar"><i style="width:${Math.max(2, (100 * res.seconds) / maxSec).toFixed(1)}%"></i></span>
      <span class="sc">${res.seconds.toFixed(res.seconds < 1 ? 3 : 2)} s</span>
    </div>`).join('') || '<div class="field-hint">No agent is served at this size.</div>';
  const winners = served.filter((x) => x.res.solved).sort((a, b) => a.res.seconds - b.res.seconds);
  $('race-title').textContent = winners.length
    ? `${n.toLocaleString('en-US')} queens: ${AGENTS[winners[0].agent].label} finished first, in ${winners[0].res.seconds.toFixed(3)} s.`
    : `${n.toLocaleString('en-US')} queens: no agent finished inside its time cap.`;
}

// ── Agent comparison table ───────────────────────────────────────────────

function renderAgentTable() {
  const rows = AGENT_ORDER.map((a) => `<tr>
      <td>${AGENTS[a].label}</td>
      <td>${AGENTS[a].steps}</td>
      <td>${AGENTS[a].work}</td>
    </tr>`).join('');
  $('agent-table').innerHTML = `<table class="qn-table"><thead><tr><th>AGENT</th><th>ONE STEP IS</th><th>WORK COUNTED</th></tr></thead><tbody>${rows}</tbody></table>`;
}

// ── Wiring ───────────────────────────────────────────────────────────────

function init() {
  play.squares = buildGrid($('play-board'), play.n, playClick);
  playRender();
  for (const m of modes) $(`tab-${m}`).addEventListener('click', () => setMode(m));
  $('play-hint').addEventListener('click', playHint);
  $('play-clear').addEventListener('click', playClear);
  $('play-solve').addEventListener('click', playShowSolution);

  $('watch-run').addEventListener('click', watchRun);
  $('watch-play').addEventListener('click', watchToggle);
  $('watch-prev').addEventListener('click', () => { stopWatch(); watchShow(watch.idx - 1); });
  $('watch-next').addEventListener('click', () => { stopWatch(); watchShow(watch.idx + 1); });
  $('watch-scrub').addEventListener('input', (e) => { stopWatch(); watchShow(+e.target.value); });
  $('watch-agent').addEventListener('change', (e) => { e.target.title = AGENTS[e.target.value].tip; });
  $('watch-shuffle').addEventListener('click', () => { $('watch-seed').value = Math.floor(Math.random() * 1000); });
  $('watch-agent').title = AGENTS[$('watch-agent').value].tip;

  $('race-run').addEventListener('click', raceRun);

  $('big-play').addEventListener('click', () => (big.running ? stopBig() : startBig()));
  $('big-new').addEventListener('click', bigNew);
  $('big-n').addEventListener('change', bigNew);
  $('big-speed').addEventListener('input', (e) => { $('big-speed-out').textContent = e.target.value; });

  renderAgentTable();
  getJSON('/api/queens/meta').then((meta) => {
    for (const a of AGENT_ORDER) if (meta.agents && meta.agents[a]) maxN[a] = meta.agents[a].max_n;
  }).catch(() => {});
  setMode('play');
}

init();
