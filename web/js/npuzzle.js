// N-Puzzle page: playable board, live search streaming, solution playback, and comparison.
//
// Boards use the backend's layout: a flat array read row by row, 0 = blank,
// goal = [0, 1, ..., n*n - 1]. Actions name the direction the blank moves.

import { SearchSocket, formatNumber, formatSeconds, getJSON } from './api.js';

const $ = (id) => document.getElementById(id);
const socket = new SearchSocket('/ws/npuzzle');

const state = {
  n: 3,
  board: [],
  meta: null,
  solution: null,   // { start, path, boards }
  step: 0,
  playTimer: null,
  yourMoves: 0,
  trace: [],        // [{ board, g, h }] in expansion order
  replayTimer: null,
  comparing: false,
};

// ── Board helpers ────────────────────────────────────────────────────────

const goal = (n) => Array.from({ length: n * n }, (_, i) => i);
const isGoal = (b) => b.every((t, i) => t === i);

function manhattan(b, n) {
  let d = 0;
  for (let i = 0; i < b.length; i++) {
    const t = b[i];
    if (t) d += Math.abs(Math.floor(i / n) - Math.floor(t / n)) + Math.abs((i % n) - (t % n));
  }
  return d;
}

function isSolvable(b, n) {
  const tiles = b.filter((t) => t !== 0);
  let inv = 0;
  for (let i = 0; i < tiles.length; i++)
    for (let j = i + 1; j < tiles.length; j++) if (tiles[i] > tiles[j]) inv++;
  if (n % 2 === 1) return inv % 2 === 0;
  // Goal has the blank on row 0; each vertical move flips inversion parity.
  return (inv + Math.floor(b.indexOf(0) / n)) % 2 === 0;
}

const OFFSETS = { Up: [-1, 0], Down: [1, 0], Left: [0, -1], Right: [0, 1] };

function applyMove(b, action, n) {
  const blank = b.indexOf(0);
  const [dr, dc] = OFFSETS[action];
  const r = Math.floor(blank / n) + dr, c = (blank % n) + dc;
  if (r < 0 || r >= n || c < 0 || c >= n) return null;
  const next = b.slice();
  const target = r * n + c;
  next[blank] = next[target];
  next[target] = 0;
  return next;
}

function scramble(n, moves) {
  const inverse = { Up: 'Down', Down: 'Up', Left: 'Right', Right: 'Left' };
  let b = goal(n), prev = null;
  for (let i = 0; i < moves; i++) {
    const options = Object.keys(OFFSETS).filter((a) => a !== inverse[prev] && applyMove(b, a, n));
    prev = options[Math.floor(Math.random() * options.length)];
    b = applyMove(b, prev, n);
  }
  return b;
}

function uniformRandom(n) {
  for (;;) {
    const b = goal(n);
    for (let i = b.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [b[i], b[j]] = [b[j], b[i]];
    }
    if (isSolvable(b, n)) return b;
  }
}

// ── Rendering ────────────────────────────────────────────────────────────

function renderBoard(movedIndex = -1) {
  const { n, board } = state;
  const el = $('board');
  const size = n === 3 ? 96 : 78;
  el.style.gridTemplateColumns = `repeat(${n}, ${size}px)`;
  el.classList.toggle('solved', isGoal(board));
  const blank = board.indexOf(0);
  const busy = socket.busy || state.playTimer;
  el.innerHTML = '';
  board.forEach((tile, i) => {
    const div = document.createElement('div');
    const adjacent = Math.abs(Math.floor(i / n) - Math.floor(blank / n)) + Math.abs((i % n) - (blank % n)) === 1;
    div.className = 'np-tile' + (tile === 0 ? ' np-blank' : '') + (i === movedIndex ? ' np-moved' : '')
      + (adjacent && !busy ? ' clickable' : '');
    div.style.width = div.style.height = `${size}px`;
    div.style.fontSize = n === 3 ? '1.9rem' : '1.5rem';
    div.textContent = tile || '';
    if (adjacent) div.onclick = () => playerMove(i);
    el.appendChild(div);
  });
}

function renderMini(b, g) {
  const n = Math.round(Math.sqrt(b.length));
  const el = $('mini-board');
  el.style.gridTemplateColumns = `repeat(${n}, 30px)`;
  el.innerHTML = '';
  b.forEach((tile, i) => {
    const div = document.createElement('div');
    div.className = 'mini-tile' + (tile === 0 ? ' blank' : tile === i ? ' home' : '');
    div.textContent = tile || '';
    el.appendChild(div);
  });
  $('mini-info').textContent = `g = ${g ?? '—'} · h = ${manhattan(b, n)}`;
}

function setChip(id, text) { $(id).textContent = text; }

function log(text, cls = 'log-info') {
  const el = $('log');
  const line = document.createElement('div');
  line.className = cls;
  line.textContent = text;
  el.appendChild(line);
  while (el.childElementCount > 300) el.firstChild.remove();
  el.scrollTop = el.scrollHeight;
}

// ── Canvases ─────────────────────────────────────────────────────────────

function setupCanvas(canvas) {
  const dpr = window.devicePixelRatio || 1;
  const w = canvas.clientWidth, h = canvas.clientHeight;
  if (canvas.width !== w * dpr || canvas.height !== h * dpr) {
    canvas.width = w * dpr;
    canvas.height = h * dpr;
  }
  const ctx = canvas.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);
  return { ctx, w, h };
}

function axes(ctx, w, h, pad, xLabel, yLabel) {
  ctx.strokeStyle = 'rgba(0,245,255,0.25)';
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(pad, 6); ctx.lineTo(pad, h - pad); ctx.lineTo(w - 6, h - pad);
  ctx.stroke();
  ctx.fillStyle = '#4a7a9b';
  ctx.font = '10px JetBrains Mono, monospace';
  ctx.fillText(xLabel, w - 6 - ctx.measureText(xLabel).width, h - 6);
  ctx.save(); ctx.translate(10, 12); ctx.fillText(yLabel, 0, 0); ctx.restore();
}

function lerpColor(t) {
  // cyan (0,245,255) to pink (255,0,160)
  const r = Math.round(255 * t), g = Math.round(245 * (1 - t)), b = Math.round(255 - 95 * t);
  return `rgb(${r},${g},${b})`;
}

function drawScatter(upTo = state.trace.length) {
  const { ctx, w, h } = setupCanvas($('scatter'));
  const pad = 24;
  axes(ctx, w, h, pad, 'g (depth)', 'h');
  const pts = state.trace;
  if (!pts.length) return;
  const maxG = Math.max(1, ...pts.map((p) => p.g)), maxH = Math.max(1, ...pts.map((p) => p.h));
  const sx = (g) => pad + (g / maxG) * (w - pad - 10);
  const sy = (v) => h - pad - (v / maxH) * (h - pad - 10);
  for (let i = 0; i < upTo; i++) {
    ctx.fillStyle = lerpColor(i / Math.max(1, pts.length - 1));
    ctx.globalAlpha = 0.55;
    ctx.fillRect(sx(pts[i].g) - 1.5, sy(pts[i].h) - 1.5, 3, 3);
  }
  ctx.globalAlpha = 1;
  if (upTo > 0) {
    const p = pts[upTo - 1];
    ctx.strokeStyle = '#ffe600';
    ctx.strokeRect(sx(p.g) - 4, sy(p.h) - 4, 8, 8);
  }
}

function drawHistogram(upTo = state.trace.length) {
  const { ctx, w, h } = setupCanvas($('histogram'));
  const pad = 24;
  axes(ctx, w, h, pad, 'g (depth)', 'nodes');
  if (!state.trace.length) return;
  const counts = [];
  for (let i = 0; i < upTo; i++) {
    const g = state.trace[i].g;
    counts[g] = (counts[g] || 0) + 1;
  }
  const maxG = Math.max(...state.trace.map((p) => p.g));
  const maxC = Math.max(1, ...counts.filter(Boolean));
  const bw = (w - pad - 10) / (maxG + 1);
  for (let g = 0; g <= maxG; g++) {
    const c = counts[g] || 0;
    const bh = (c / maxC) * (h - pad - 12);
    ctx.fillStyle = lerpColor(g / Math.max(1, maxG));
    ctx.fillRect(pad + g * bw + 1, h - pad - bh, Math.max(1, bw - 2), bh);
  }
}

let drawQueued = false;
function queueDraw() {
  if (drawQueued) return;
  drawQueued = true;
  requestAnimationFrame(() => { drawQueued = false; drawScatter(); drawHistogram(); });
}

// ── Controls ─────────────────────────────────────────────────────────────

function heuristicChoices(algorithm, n) {
  const all = state.meta.heuristics.filter((h) => h.sizes.includes(n)).map((h) => h.name);
  if (algorithm === 'bwas') return all.filter((h) => h !== 'linear_conflict');
  return all.filter((h) => h !== 'neural');
}

function updateControls() {
  if (!state.meta) return;
  const algo = $('algorithm').value;
  const info = state.meta.algorithms.find((a) => a.name === algo);
  $('algo-desc').textContent = info.description;
  $('heuristic-wrap').classList.toggle('hidden', !info.informed);
  $('weight-wrap').classList.toggle('hidden', algo !== 'wastar');
  $('bwas-wrap').classList.toggle('hidden', algo !== 'bwas');

  const sel = $('heuristic');
  const previous = sel.value;
  const choices = heuristicChoices(algo, state.n);
  sel.innerHTML = '';
  for (const name of choices) {
    const opt = document.createElement('option');
    opt.value = name;
    opt.textContent = { manhattan: 'Manhattan distance', linear_conflict: 'Linear conflict',
                        pdb: 'Pattern database (5-5-5)', neural: 'Neural (learned)' }[name];
    sel.appendChild(opt);
  }
  sel.value = choices.includes(previous) ? previous
    : (algo === 'bwas' && choices.includes('neural') ? 'neural' : choices[0]);
  const h = state.meta.heuristics.find((x) => x.name === sel.value);
  $('heur-desc').textContent = info.informed && h ? h.description : '';
}

function request() {
  const algorithm = $('algorithm').value;
  const req = { board: state.board, algorithm };
  const info = state.meta.algorithms.find((a) => a.name === algorithm);
  if (info.informed) req.heuristic = $('heuristic').value;
  if (algorithm === 'wastar') req.weight = Number($('weight').value);
  if (algorithm === 'bwas') {
    req.batch_size = Number($('batch').value);
    req.g_weight = Number($('gweight').value);
  }
  return req;
}

function label(req) {
  const short = { manhattan: 'Manhattan', linear_conflict: 'LC', pdb: 'PDB', neural: 'Neural' };
  const names = { bfs: 'BFS', dfs: 'DFS', ids: 'IDS', ucs: 'UCS', bibfs: 'Bidir BFS', greedy: 'Greedy',
                  astar: 'A*', wastar: 'Weighted A*', idastar: 'IDA*', bwas: 'Batch WA*' };
  return names[req.algorithm] + (req.heuristic ? ` + ${short[req.heuristic]}` : '');
}

function setBusy(busy) {
  for (const id of ['btn-solve', 'btn-scramble', 'btn-random', 'btn-load', 'btn-reset', 'size',
                    'algorithm', 'heuristic', 'btn-compare', 'btn-replay'])
    $(id).disabled = busy;
  $('btn-cancel').disabled = !busy;
  if (busy) for (const id of ['btn-back', 'btn-fwd', 'btn-play']) $(id).disabled = true;
  else updatePlayback();
  renderBoard();
}

function setBoard(b, why) {
  stopPlayback();
  stopReplay();
  state.n = Math.round(Math.sqrt(b.length));
  $('size').value = String(state.n);
  state.board = b;
  state.solution = null;
  state.step = 0;
  state.yourMoves = 0;
  $('st-yours').textContent = '0';
  $('moves-list').textContent = '';
  updatePlayback();
  renderBoard();
  renderMini(b, 0);
  updateControls();
  renderPresets();
  $('cmp-body').innerHTML = '';
  $('cmp-status').textContent = '';
  if (why) log(`// ${why}: ${b.join(',')}`);
}

function playerMove(index) {
  if (socket.busy || state.playTimer) return;
  const blank = state.board.indexOf(0);
  const next = state.board.slice();
  next[blank] = next[index];
  next[index] = 0;
  state.board = next;
  state.solution = null;
  updatePlayback();
  state.yourMoves++;
  $('st-yours').textContent = String(state.yourMoves);
  renderBoard(blank);
  if (isGoal(next)) {
    log(`★ Solved by hand in ${state.yourMoves} moves.`, 'log-best');
    setChip('chip-status', 'SOLVED');
  }
}

// ── Solving ──────────────────────────────────────────────────────────────

function resetStats() {
  for (const id of ['st-path', 'st-nodes', 'st-rate', 'st-frontier', 'st-depth', 'st-time']) $(id).textContent = '—';
}

async function solve() {
  if (!state.meta) return;
  const req = request();
  stopPlayback();
  stopReplay();
  state.trace = [];
  queueDraw();
  resetStats();
  setChip('chip-algo', label(req));
  setChip('chip-status', 'SEARCHING');
  setChip('chip-nodes', '0');
  log(`▶ ${label(req)} on ${req.board.join(',')}`, 'log-move');
  setBusy(true);
  let maxFrontier = 0;
  try {
    const result = await socket.solve(req, (msg) => {
      if (msg.type !== 'progress') return;
      maxFrontier = Math.max(maxFrontier, msg.frontier);
      $('st-nodes').textContent = formatNumber(msg.nodes);
      $('st-rate').textContent = msg.elapsed > 0 ? formatNumber(msg.nodes / msg.elapsed) : '—';
      $('st-frontier').textContent = formatNumber(maxFrontier);
      $('st-time').textContent = formatSeconds(msg.elapsed);
      setChip('chip-nodes', formatNumber(msg.nodes));
      const n = state.n;
      for (const [b, g] of msg.expanded) state.trace.push({ board: b, g, h: manhattan(b, n) });
      const last = msg.expanded.length ? msg.expanded[msg.expanded.length - 1] : [msg.board, null];
      renderMini(last[0], last[1]);
      queueDraw();
    });
    showResult(req, result);
  } catch (err) {
    setChip('chip-status', 'ERROR');
    log(`✗ ${err.message}`, 'log-err');
  } finally {
    setBusy(false);
    $('btn-replay').disabled = state.trace.length === 0;
    queueDraw();
  }
}

function showResult(req, r) {
  $('st-nodes').textContent = formatNumber(r.expanded);
  if (r.max_frontier !== undefined) $('st-frontier').textContent = formatNumber(r.max_frontier);
  if (r.max_depth !== undefined) $('st-depth').textContent = formatNumber(r.max_depth);
  if (r.seconds !== undefined) {
    $('st-time').textContent = formatSeconds(r.seconds);
    $('st-rate').textContent = r.seconds > 0 ? formatNumber(r.expanded / r.seconds) : '—';
  }
  setChip('chip-nodes', formatNumber(r.expanded));
  setChip('chip-status', r.status.toUpperCase());

  if (r.status !== 'solved') {
    const why = { limit: 'hit the server time or memory limit', unsolvable: 'this board cannot reach the goal',
                  cancelled: 'stopped', exhausted: 'searched everything without reaching the goal' }[r.status];
    log(`■ ${label(req)}: ${why} after ${formatNumber(r.expanded)} nodes.`, 'log-adv');
    return;
  }
  $('st-path').textContent = String(r.cost);
  setChip('chip-moves', String(r.cost));
  const boards = [req.board];
  for (const a of r.path) boards.push(applyMove(boards[boards.length - 1], a, state.n));
  state.solution = { start: req.board, path: r.path, boards };
  state.step = 0;
  updatePlayback();
  log(`✓ ${label(req)}: ${r.cost} moves, ${formatNumber(r.expanded)} nodes, ${formatSeconds(r.seconds)}.`, 'log-best');
}

// ── Playback ─────────────────────────────────────────────────────────────

function updatePlayback() {
  const sol = state.solution;
  const total = sol ? sol.path.length : 0;
  $('progress-lbl').textContent = `${state.step} / ${total}`;
  $('progress-bar').style.width = total ? `${(100 * state.step) / total}%` : '0%';
  $('btn-back').disabled = !sol || state.step === 0;
  $('btn-fwd').disabled = !sol || state.step === total;
  $('btn-play').disabled = !sol;
  $('btn-play').querySelector('.btn-txt').textContent = state.playTimer ? '⏸ PAUSE' : '▶ PLAY';
  const list = $('moves-list');
  list.innerHTML = '';
  if (sol) {
    sol.path.forEach((a, i) => {
      const span = document.createElement('span');
      span.textContent = { Up: '↑', Down: '↓', Left: '←', Right: '→' }[a] + ' ';
      if (i === state.step - 1) span.className = 'cur';
      list.appendChild(span);
    });
  }
}

function goToStep(step) {
  const sol = state.solution;
  step = Math.max(0, Math.min(sol.path.length, step));
  const prevBlank = state.board.indexOf(0);
  state.step = step;
  state.board = sol.boards[step];
  renderBoard(step > 0 ? prevBlank : -1);
  updatePlayback();
}

function stopPlayback() {
  clearInterval(state.playTimer);
  state.playTimer = null;
}

function togglePlay() {
  if (state.playTimer) {
    stopPlayback();
    updatePlayback();
    renderBoard();
    return;
  }
  if (state.step === state.solution.path.length) goToStep(0);
  state.playTimer = setInterval(() => {
    if (state.step >= state.solution.path.length) {
      stopPlayback();
      updatePlayback();
      renderBoard();
      return;
    }
    goToStep(state.step + 1);
  }, state.n === 3 ? 260 : 180);
  updatePlayback();
}

// Re-animate the recorded expansions on the mini board and charts.
function replaySearch() {
  stopReplay();
  const total = state.trace.length;
  if (!total) return;
  let i = 0;
  const perFrame = Math.max(1, Math.round(total / 300));
  state.replayTimer = setInterval(() => {
    i = Math.min(total, i + perFrame);
    const p = state.trace[i - 1];
    renderMini(p.board, p.g);
    drawScatter(i);
    drawHistogram(i);
    if (i >= total) stopReplay();
  }, 30);
}

function stopReplay() {
  clearInterval(state.replayTimer);
  state.replayTimer = null;
}

// ── Compare ──────────────────────────────────────────────────────────────

const PRESETS = {
  3: [
    { algorithm: 'bfs' }, { algorithm: 'bibfs' }, { algorithm: 'ids' }, { algorithm: 'dfs' },
    { algorithm: 'greedy', heuristic: 'manhattan' }, { algorithm: 'astar', heuristic: 'manhattan' },
    { algorithm: 'astar', heuristic: 'linear_conflict' }, { algorithm: 'idastar', heuristic: 'manhattan' },
  ],
  4: [
    { algorithm: 'greedy', heuristic: 'manhattan' }, { algorithm: 'astar', heuristic: 'manhattan' },
    { algorithm: 'idastar', heuristic: 'manhattan' }, { algorithm: 'idastar', heuristic: 'linear_conflict' },
    { algorithm: 'idastar', heuristic: 'pdb' }, { algorithm: 'astar', heuristic: 'pdb' },
    { algorithm: 'bwas', heuristic: 'pdb' }, { algorithm: 'bwas', heuristic: 'neural' },
  ],
};
const DEFAULT_ON = { 3: [0, 1, 5, 7], 4: [1, 2, 4, 7] };

function renderPresets() {
  const el = $('cmp-presets');
  el.innerHTML = '';
  PRESETS[state.n].forEach((p, i) => {
    const lbl = document.createElement('label');
    const box = document.createElement('input');
    box.type = 'checkbox';
    box.checked = DEFAULT_ON[state.n].includes(i);
    box.dataset.index = i;
    lbl.append(box, label(p));
    el.appendChild(lbl);
  });
}

async function compare() {
  const chosen = [...$('cmp-presets').querySelectorAll('input:checked')].map((b) => PRESETS[state.n][b.dataset.index]);
  if (!chosen.length) return;
  stopPlayback();
  stopReplay();
  const board = state.board.slice();
  const body = $('cmp-body');
  body.innerHTML = '';
  const rows = [];
  setBusy(true);
  state.comparing = true;
  log(`⇶ Comparing ${chosen.length} configurations on ${board.join(',')}`, 'log-move');
  for (const [i, preset] of chosen.entries()) {
    const req = { board, ...preset };
    $('cmp-status').textContent = `running ${i + 1}/${chosen.length}: ${label(req)}`;
    let r;
    try {
      r = await socket.solve(req, (msg) => {
        if (msg.type === 'progress') $('cmp-status').textContent = `running ${i + 1}/${chosen.length}: ${label(req)} · ${formatNumber(msg.nodes)} nodes`;
      });
    } catch (err) {
      r = { status: 'error', expanded: null, seconds: null, message: err.message };
    }
    rows.push({ req, r });
    renderCompare(rows);
    if (!state.comparing) break;  // STOP pressed
  }
  state.comparing = false;
  $('cmp-status').textContent = 'done';
  setBusy(false);
}

function renderCompare(rows) {
  const body = $('cmp-body');
  body.innerHTML = '';
  const solved = rows.filter((x) => x.r.status === 'solved');
  const maxNodes = Math.max(10, ...rows.map((x) => x.r.expanded || 0));
  const fewest = solved.length ? Math.min(...solved.map((x) => x.r.expanded)) : -1;
  for (const { req, r } of rows) {
    const tr = document.createElement('tr');
    if (r.status === 'solved' && r.expanded === fewest) tr.className = 'best';
    const pct = r.expanded ? (100 * Math.log10(r.expanded + 1)) / Math.log10(maxNodes + 1) : 0;
    tr.innerHTML = `
      <td>${label(req)}</td>
      <td class="status-${r.status}">${r.status}</td>
      <td class="num">${r.status === 'solved' ? r.cost : '—'}</td>
      <td class="num">${formatNumber(r.expanded)}</td>
      <td><div class="cmp-bar" style="width:${pct.toFixed(1)}%"></div></td>
      <td class="num">${formatSeconds(r.seconds)}</td>`;
    body.appendChild(tr);
  }
}

// ── Wiring ───────────────────────────────────────────────────────────────

function parseCustom() {
  const text = $('custom').value.trim();
  const tiles = text.split(/[\s,]+/).filter(Boolean).map(Number);
  const n = Math.round(Math.sqrt(tiles.length));
  if (![3, 4].includes(n) || n * n !== tiles.length) return 'enter 9 or 16 tiles';
  if ([...tiles].sort((a, b) => a - b).some((t, i) => t !== i)) return `use each tile 0–${n * n - 1} once`;
  if (!isSolvable(tiles, n)) return 'that board cannot reach the goal';
  return tiles;
}

function wire() {
  $('btn-solve').onclick = solve;
  $('btn-cancel').onclick = () => { state.comparing = false; socket.cancel(); };
  $('btn-back').onclick = () => { stopPlayback(); goToStep(state.step - 1); };
  $('btn-fwd').onclick = () => { stopPlayback(); goToStep(state.step + 1); };
  $('btn-play').onclick = togglePlay;
  $('btn-replay').onclick = replaySearch;
  $('btn-compare').onclick = compare;
  $('btn-reset').onclick = () => {
    stopPlayback();
    const start = state.solution ? state.solution.start : null;
    if (start) { goToStep(0); } else { setBoard(goal(state.n), 'Reset to goal'); }
  };
  $('btn-scramble').onclick = () => setBoard(scramble(state.n, Number($('depth').value)), `Scrambled ${$('depth').value} moves`);
  $('btn-random').onclick = () => setBoard(uniformRandom(state.n), 'Uniformly random board');
  $('btn-load').onclick = () => {
    const parsed = parseCustom();
    $('custom-err').textContent = typeof parsed === 'string' ? parsed : '';
    if (typeof parsed !== 'string') setBoard(parsed, 'Loaded custom board');
  };
  $('size').onchange = () => {
    state.n = Number($('size').value);
    $('depth').max = state.n === 3 ? 40 : 80;
    setBoard(scramble(state.n, Number($('depth').value)), `New ${state.n}×${state.n} board`);
  };
  $('depth').oninput = () => { $('depth-lbl').textContent = $('depth').value; };
  $('weight').oninput = () => { $('weight-lbl').textContent = Number($('weight').value).toFixed(1); };
  $('batch').oninput = () => { $('batch-lbl').textContent = $('batch').value; };
  $('gweight').oninput = () => { $('gw-lbl').textContent = Number($('gweight').value).toFixed(2); };
  $('algorithm').onchange = updateControls;
  $('heuristic').onchange = updateControls;
  window.addEventListener('resize', queueDraw);
}

async function init() {
  wire();
  setBoard(scramble(3, 20));
  try {
    state.meta = await getJSON('/api/npuzzle/meta');
  } catch (err) {
    log(`✗ Search server unreachable (${err.message}). The board still works by hand.`, 'log-err');
    $('btn-solve').disabled = $('btn-compare').disabled = true;
    return;
  }
  const sel = $('algorithm');
  for (const a of state.meta.algorithms) {
    const opt = document.createElement('option');
    opt.value = a.name;
    opt.textContent = label({ algorithm: a.name });
    sel.appendChild(opt);
  }
  sel.value = 'astar';
  updateControls();
  $('log').innerHTML = '';
  log('// Connected. Scramble a board, pick an agent, and press SOLVE.');
}

init();
