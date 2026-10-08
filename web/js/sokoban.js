// Sokoban page: play by hand, read the search's bounds for each position, animate the solver's
// plan, and compare node counts with deadlock pruning off and on.
//
// Positions are interior indices i = row * cols + col (see server/sokoban_api.py). A cell that is
// not in `floor` is a wall. Each box keeps its slot in S.boxes as it is pushed, so its element can
// slide to the next cell. The grid cells are built once per level; only sprites and classes change.

import { getJSON, postJSON, formatNumber, formatSeconds } from './api.js';
import { banner, burst, shake } from './fx.js';

const $ = (id) => document.getElementById(id);
const SVG_NS = 'http://www.w3.org/2000/svg';
const GAP = 3; // px between cells
const BUDGET = 20000; // node budget per search request, the same cap the server enforces
const REDUCED = matchMedia('(prefers-reduced-motion: reduce)').matches;
const DIRS = { w: [-1, 0], s: [1, 0], a: [0, -1], d: [0, 1] };
const ARROWS = { ArrowUp: 'w', ArrowDown: 's', ArrowLeft: 'a', ArrowRight: 'd' };
const ALGO_LABEL = { astar: 'A*', greedy: 'GREEDY', bfs: 'BFS' };
const HEUR_LABEL = { matching: 'MATCHING', simple: 'SIMPLE' };
const STATUS_TEXT = {
  budget: `The node budget (${formatNumber(BUDGET)} expansions) ran out. A* with the matching bound needs far fewer on most levels.`,
  time: 'The 5 second limit on the server ran out. Try A* with the matching bound.',
  exhausted: 'The search emptied its frontier: this position has no solution at all.',
};

const S = {
  n: 1,
  cols: 0,
  rows: 0,
  cell: 40,
  floor: null, // Uint8Array over interior cells
  goals: new Set(),
  goalList: [], // sorted, to compare with the sorted box list
  dead: new Set(),
  nearest: [],
  start: null,
  player: 0,
  boxes: [],
  moves: '', // lowercase walks, uppercase pushes, like the server
  face: 'd',
  history: [],
  frozen: new Set(),
  pairs: [],
  deadlock: null,
  analysis: null,
  playerEl: null,
  boxEls: [],
  busy: false,
  stopping: false,
  wasSolved: false,
  lastDeadlock: false,
  prune: true,
  evalTimer: 0,
  evalBusy: false,
  evalPending: false,
};

const rowOf = (i) => Math.floor(i / S.cols);
const colOf = (i) => i % S.cols;
const label = (i) => `${String.fromCharCode(97 + colOf(i))}${rowOf(i) + 1}`;
const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

// Index of an open cell at (r, c), or -1 for a wall or outside the grid.
function openAt(r, c) {
  if (r < 0 || c < 0 || r >= S.rows || c >= S.cols) return -1;
  const i = r * S.cols + c;
  return S.floor[i] ? i : -1;
}

// One walk or push. The rule: stepping into a box pushes it only if the cell beyond is open and empty.
// Returns null when blocked, otherwise {pushed, box} where box indexes S.boxes.
function stepTo(letter) {
  const [dr, dc] = DIRS[letter];
  const r = rowOf(S.player), c = colOf(S.player);
  const target = openAt(r + dr, c + dc);
  if (target < 0) return null;
  const bi = S.boxes.indexOf(target);
  let beyond = -1;
  if (bi >= 0) {
    beyond = openAt(r + 2 * dr, c + 2 * dc);
    if (beyond < 0 || S.boxes.includes(beyond)) return null;
  }
  S.history.push({ player: S.player, boxes: S.boxes.slice(), moves: S.moves, face: S.face });
  S.player = target;
  S.face = letter;
  if (bi >= 0) {
    S.boxes[bi] = beyond;
    S.moves += letter.toUpperCase();
    return { pushed: true, box: bi };
  }
  S.moves += letter;
  return { pushed: false, box: -1 };
}

function solvedNow() {
  if (S.boxes.length !== S.goalList.length) return false;
  const sorted = S.boxes.slice().sort((a, b) => a - b);
  return sorted.every((v, k) => v === S.goalList[k]);
}

const pushCount = () => (S.moves.match(/[A-Z]/g) || []).length;

function boxKey() {
  return S.boxes.slice().sort((a, b) => a - b).join(',');
}

function setNote(msg) {
  $('sok-note').textContent = msg;
}

function log(msg) {
  const box = $('log');
  const line = document.createElement('div');
  line.textContent = msg;
  box.appendChild(line);
  while (box.childElementCount > 80) box.firstElementChild.remove();
  box.scrollTop = box.scrollHeight;
}

// ── Rendering ────────────────────────────────────────────────────────────

// Build the static grid and the sprites for the current level. Called once per level.
function buildBoard() {
  const board = $('sok-board');
  board.replaceChildren();
  board.style.setProperty('--cols', String(S.cols));
  const frag = document.createDocumentFragment();
  for (let i = 0; i < S.rows * S.cols; i++) {
    const cell = document.createElement('div');
    if (!S.floor[i]) {
      cell.className = 'sok-cell wall';
    } else {
      const classes = ['sok-cell', 'floor'];
      if (S.goals.has(i)) classes.push('goal');
      if (S.dead.has(i)) classes.push('dead');
      cell.className = classes.join(' ');
      // The push distance overlay: shown or hidden by a class on the board, so it costs nothing per move.
      const num = document.createElement('span');
      num.className = 'sok-num';
      const v = S.nearest[i];
      num.textContent = v >= 0 ? String(v) : 'x';
      cell.appendChild(num);
    }
    frag.appendChild(cell);
  }
  board.appendChild(frag);
  S.playerEl = document.createElement('div');
  S.playerEl.className = 'sok-sprite sok-player';
  board.appendChild(S.playerEl);
  S.boxEls = S.boxes.map(() => {
    const el = document.createElement('div');
    el.className = 'sok-sprite sok-box';
    board.appendChild(el);
    return el;
  });
}

// Fit the grid to the stage width. Measured on the stage, not the frame, because the frame's width
// depends on the grid it contains.
function layout() {
  const stage = $('sok-wrap').parentElement;
  const avail = stage.clientWidth - 40; // frame padding and border, with a little slack
  const cell = Math.max(22, Math.min(60, Math.floor((avail - GAP * (S.cols - 1)) / S.cols)));
  S.cell = cell;
  const board = $('sok-board');
  board.style.setProperty('--cell', `${cell}px`);
  board.style.setProperty('--gap', `${GAP}px`);
  const w = S.cols * cell + (S.cols - 1) * GAP;
  const h = S.rows * cell + (S.rows - 1) * GAP;
  const svg = $('sok-svg');
  svg.setAttribute('width', String(w));
  svg.setAttribute('height', String(h));
  svg.setAttribute('viewBox', `0 0 ${w} ${h}`);
  renderSprites();
  drawPairs();
}

function place(el, i, face) {
  const x = colOf(i) * (S.cell + GAP);
  const y = rowOf(i) * (S.cell + GAP);
  el.style.transform = `translate(${x}px, ${y}px)`;
  if (face) el.dataset.face = face;
}

function renderSprites() {
  if (!S.playerEl) return;
  place(S.playerEl, S.player, S.face);
  S.boxes.forEach((pos, k) => place(S.boxEls[k], pos));
}

// Box classes: on a goal, on a dead square, or frozen (from the last analysis).
function paintBoxes() {
  S.boxes.forEach((pos, k) => {
    const el = S.boxEls[k];
    if (!el) return;
    el.classList.toggle('on-goal', S.goals.has(pos));
    el.classList.toggle('dead-box', S.dead.has(pos));
    el.classList.toggle('frozen', S.frozen.has(pos));
  });
}

function updateChips() {
  $('chip-pushes').textContent = formatNumber(pushCount());
  $('chip-moves').textContent = formatNumber(S.moves.length);
  const solved = solvedNow();
  $('chip-status').textContent = solved ? 'SOLVED' : S.deadlock ? 'DEADLOCK' : 'PLAYING';
}

function renderState() {
  renderSprites();
  paintBoxes();
  const solved = solvedNow();
  $('sok-wrap').classList.toggle('solved', solved);
  updateChips();
  if (solved && !S.wasSolved) {
    banner('SOLVED', `${pushCount()} pushes, ${S.moves.length} moves`, '#00ff88');
    burst($('sok-board'));
    log(`solved in ${pushCount()} pushes (${S.moves.length} moves)`);
  }
  S.wasSolved = solved;
}

// Drop the analysis when the boxes move: its deadlock, frozen set and matching lines are for the old layout.
function markStale() {
  S.analysis = null;
  S.deadlock = null;
  S.frozen = new Set();
  S.pairs = [];
  $('sok-wrap').classList.remove('deadlocked');
  drawPairs();
}

function thumpBox(k) {
  const el = S.boxEls[k];
  if (!el || REDUCED) return;
  el.classList.remove('thump');
  void el.offsetWidth; // restart the animation
  el.classList.add('thump');
}

// The matching lines: each box to the goal the cheapest pairing gives it, with the pushes labelled.
function drawPairs() {
  const svg = $('sok-svg');
  svg.replaceChildren();
  const pairs = S.analysis ? S.analysis.pairs : [];
  const cx = (i) => colOf(i) * (S.cell + GAP) + S.cell / 2;
  const cy = (i) => rowOf(i) * (S.cell + GAP) + S.cell / 2;
  for (const [box, goal, n] of pairs) {
    if (box === goal) continue; // already home: nothing to draw
    const line = document.createElementNS(SVG_NS, 'line');
    line.setAttribute('x1', String(cx(box)));
    line.setAttribute('y1', String(cy(box)));
    line.setAttribute('x2', String(cx(goal)));
    line.setAttribute('y2', String(cy(goal)));
    svg.appendChild(line);
    const dot = document.createElementNS(SVG_NS, 'circle');
    dot.setAttribute('cx', String(cx(goal)));
    dot.setAttribute('cy', String(cy(goal)));
    dot.setAttribute('r', '3.5');
    svg.appendChild(dot);
    const text = document.createElementNS(SVG_NS, 'text');
    text.setAttribute('x', String((cx(box) + cx(goal)) / 2 + 4));
    text.setAttribute('y', String((cy(box) + cy(goal)) / 2 - 4));
    text.textContent = String(n);
    svg.appendChild(text);
  }
}

// ── Analysis: the search's view of the current layout ───────────────────

function para(text, cls) {
  const p = document.createElement('p');
  if (cls) p.className = cls;
  p.textContent = text;
  return p;
}

function thinkPanel(a) {
  const title = $('think-title');
  const body = [];
  if (solvedNow()) {
    title.textContent = 'Solved: no pushes left.';
    body.push(para('Every box is on a goal, so the search has nothing left to bound.', 'good'));
    $('think-body').replaceChildren(...body);
    return;
  }
  if (a.deadlock === 'dead square') {
    title.textContent = 'Deadlock: a box sits on a dead square.';
    body.push(para('No pull from any goal reaches this square, so no sequence of pushes can bring the box to a goal. ' +
      'Every position after this one is hopeless. Pruning drops it when it is generated. Without pruning the search ' +
      'keeps it and only finds out when it runs out of moves.', 'bad'));
  } else if (a.deadlock === 'frozen box') {
    title.textContent = 'Deadlock: a frozen box is off every goal.';
    body.push(para('Walls or other frozen boxes block this box on both axes, so it can never move again, and it is not on a goal.', 'bad'));
  } else {
    title.textContent = `At least ${a.h_matching} ${a.h_matching === 1 ? 'push' : 'pushes'} left (matching bound).`;
  }
  if (a.pairs.length) {
    const pairs = a.pairs.map(([b, g, n]) => `${label(b)} to ${label(g)}: ${n}`).join('; ');
    body.push(para(`Matching bound ${a.h_matching}: the cheapest one-to-one pairing of boxes and goals, with the pushes each pair needs. ${pairs}.`, 'hot'));
  }
  body.push(para(`Simple bound ${a.h_simple}: each box alone counts the pushes to its nearest goal, so two boxes may claim the same goal. ` +
    'The matching bound is never smaller, and it is the one A* uses.'));
  body.push(para(`This level has ${S.dead.size} dead squares. The red tint shows them, and a box on one of them is a deadlock.`));
  $('think-body').replaceChildren(...body);
}

function applyAnalysis(a) {
  S.analysis = a;
  S.deadlock = a.deadlock;
  S.frozen = new Set(a.frozen);
  S.pairs = a.pairs;
  $('chip-bound').textContent = formatNumber(a.h_matching);
  $('sok-wrap').classList.toggle('deadlocked', Boolean(a.deadlock));
  drawPairs();
  paintBoxes();
  updateChips();
  thinkPanel(a);
  if (a.deadlock && !S.lastDeadlock) {
    const why = a.deadlock === 'dead square' ? 'a box can never reach a goal from here' : 'a frozen box is off every goal';
    banner('DEADLOCK', why, '#ff3b3b');
    log(`deadlock: ${why}`);
  }
  S.lastDeadlock = Boolean(a.deadlock);
}

// Evaluate after the last move, 250 ms later, and never more than one request in flight.
function scheduleEval(delay = 250) {
  clearTimeout(S.evalTimer);
  S.evalTimer = setTimeout(runEval, delay);
}

async function runEval() {
  if (S.evalBusy) {
    S.evalPending = true;
    return;
  }
  const level = S.n;
  const key = boxKey();
  S.evalBusy = true;
  try {
    const a = await postJSON('/api/sokoban/evaluate', { level, boxes: S.boxes.slice() });
    if (level === S.n && key === boxKey()) applyAnalysis(a); // ignore answers for a layout already left behind
  } catch (err) {
    // A 429 from the rate limiter is expected when playing fast: skip it quietly.
    if (!/rate limit/i.test(err.message)) setNote(`analysis unavailable: ${err.message}`);
  } finally {
    S.evalBusy = false;
    if (S.evalPending) {
      S.evalPending = false;
      scheduleEval(0);
    }
  }
}

// ── Play ─────────────────────────────────────────────────────────────────

function playMove(letter) {
  if (solvedNow()) return;
  const r = stepTo(letter);
  if (!r) {
    shake($('sok-board'), 'small');
    return;
  }
  markStale();
  renderState();
  if (r.pushed) thumpBox(r.box);
  scheduleEval();
}

function undo() {
  if (S.busy) return;
  const h = S.history.pop();
  if (!h) {
    setNote('nothing to undo');
    return;
  }
  Object.assign(S, { player: h.player, boxes: h.boxes, moves: h.moves, face: h.face });
  markStale();
  renderState();
  scheduleEval();
}

function restart() {
  if (S.busy) return;
  S.player = S.start.player;
  S.boxes = S.start.boxes.slice();
  S.moves = '';
  S.face = 'd';
  S.history = [];
  S.lastDeadlock = false;
  markStale();
  renderState();
  scheduleEval();
}

// ── Search: solve and hint ───────────────────────────────────────────────

function setBusy(on) {
  S.busy = on;
  for (const id of ['btn-undo', 'btn-restart', 'btn-hint', 'btn-solve', 'btn-compare', 'sel-level']) {
    $(id).disabled = on;
  }
}

function heuristicFor(algo) {
  return algo === 'bfs' ? 'none' : $('sel-heur').value;
}

// Explain a result in the plan panel: the moves, with pushes in pink, or why there is no plan.
function showPlan(res) {
  const title = $('plan-title');
  const list = $('plan-list');
  list.replaceChildren();
  const who = `${ALGO_LABEL[res.algorithm]}${res.heuristic !== 'none' ? ` + ${HEUR_LABEL[res.heuristic]}` : ''}, pruning ${res.prune ? 'on' : 'off'}`;
  if (!res.solved) {
    title.textContent = `No plan: ${who} stopped with status ${res.status} after ${formatNumber(res.expanded)} nodes.`;
    list.textContent = STATUS_TEXT[res.status] || '';
    return;
  }
  title.textContent = `${who}: ${res.pushes} pushes, ${formatNumber(res.expanded)} nodes expanded, ${formatSeconds(res.seconds)}.`;
  // Group the letters into runs: lowercase walks as text, each push as its own pink span.
  let walk = '';
  for (const ch of res.moves) {
    if (ch >= 'A' && ch <= 'Z') {
      if (walk) list.appendChild(document.createTextNode(`${walk} `));
      walk = '';
      const span = document.createElement('span');
      span.className = 'push';
      span.textContent = `${ch} `;
      list.appendChild(span);
    } else {
      walk += ch;
    }
  }
  if (walk) list.appendChild(document.createTextNode(walk));
}

async function animate(moves) {
  S.stopping = false;
  $('btn-stop').disabled = false;
  for (const ch of moves) {
    if (S.stopping) break;
    const push = ch >= 'A' && ch <= 'Z';
    await sleep(REDUCED ? 0 : push ? 230 : 70);
    if (S.stopping) break;
    const r = stepTo(ch.toLowerCase());
    if (!r) {
      setNote('the plan no longer fits the board');
      break;
    }
    markStale();
    renderState();
    if (r.pushed) thumpBox(r.box);
  }
  S.stopping = false;
  $('btn-stop').disabled = true;
}

// SOLVE animates the whole plan. HINT applies only the first push, with the walk that leads to it.
async function searchFromHere(hintOnly) {
  if (S.busy || solvedNow()) return;
  setBusy(true);
  const algo = $('sel-algo').value;
  setNote(hintOnly ? 'searching for the next push...' : 'searching from this position...');
  try {
    const res = await postJSON('/api/sokoban/solve', {
      level: S.n,
      boxes: S.boxes.slice(),
      player: S.player,
      algorithm: algo,
      heuristic: heuristicFor(algo),
      prune: S.prune,
      budget: BUDGET,
    });
    showPlan(res);
    if (!res.solved) {
      setNote(`no plan: ${res.status}`);
      return;
    }
    let moves = res.moves;
    if (hintOnly) {
      const k = moves.search(/[A-Z]/);
      if (k >= 0) {
        setNote(`hint: walk ${moves.slice(0, k).toLowerCase() || 'nothing'}, then push ${moves[k]}`);
        moves = moves.slice(0, k + 1);
      }
    } else {
      setNote(`animating ${res.pushes} pushes`);
    }
    await animate(moves);
  } catch (err) {
    setNote(`search failed: ${err.message}`);
  } finally {
    setBusy(false);
    markStale();
    renderState();
    scheduleEval();
  }
}

// ── Pruning comparison ───────────────────────────────────────────────────

function resetCounters() {
  for (const side of ['off', 'on']) {
    $(`cmp-${side}-num`).textContent = '0';
    $(`cmp-${side}-bar`).style.width = '0%';
    $(`cmp-${side}-info`).textContent = 'no run yet';
  }
}

// Count up through the run's progress samples on a clock shared by both runs, so the faster run
// finishes first. The samples are the server's record of (nodes expanded, seconds), not live numbers.
function countUp(side, res, maxSeconds, scaleMax) {
  const numEl = $(`cmp-${side}-num`);
  const barEl = $(`cmp-${side}-bar`);
  const pts = res.samples.map(([n, s]) => [n, s]);
  pts.push([res.expanded, res.seconds]);
  const DURATION = REDUCED ? 0 : 2000;
  const at = (s) => (maxSeconds > 0 ? (s / maxSeconds) * DURATION : DURATION);
  return new Promise((resolve) => {
    const t0 = performance.now();
    const tick = (now) => {
      const elapsed = now - t0;
      let v = 0;
      for (const [n, s] of pts) {
        if (at(s) <= elapsed) v = n;
        else break;
      }
      if (elapsed >= DURATION) v = res.expanded;
      numEl.textContent = formatNumber(v);
      barEl.style.width = `${(v / scaleMax) * 100}%`;
      if (elapsed < DURATION) requestAnimationFrame(tick);
      else resolve();
    };
    requestAnimationFrame(tick);
  });
}

function describeRun(side, res) {
  $(`cmp-${side}-info`).textContent =
    `${res.status} · ${formatSeconds(res.seconds)} · ${formatNumber(res.generated)} generated · ` +
    `${formatNumber(res.pruned)} pruned · ${res.pushes ?? '—'} pushes`;
}

async function compare() {
  if (S.busy) return;
  setBusy(true);
  $('cmp-spin').hidden = false;
  setNote('running both searches...');
  const algo = $('sel-algo').value;
  const base = {
    level: S.n,
    boxes: S.boxes.slice(),
    player: S.player,
    algorithm: algo,
    heuristic: heuristicFor(algo),
    budget: BUDGET,
  };
  try {
    const [off, on] = await Promise.all([
      postJSON('/api/sokoban/solve', { ...base, prune: false }),
      postJSON('/api/sokoban/solve', { ...base, prune: true }),
    ]);
    $('cmp-title').textContent = 'Replayed from each run\'s progress samples on one clock, then settled on the measured count.';
    const maxSeconds = Math.max(off.seconds, on.seconds, ...off.samples.map((p) => p[1]), ...on.samples.map((p) => p[1]));
    const scaleMax = Math.max(off.expanded, on.expanded, 1);
    await Promise.all([countUp('off', off, maxSeconds, scaleMax), countUp('on', on, maxSeconds, scaleMax)]);
    describeRun('off', off);
    describeRun('on', on);
    const diff = off.expanded > 0 ? Math.round(Math.abs(1 - on.expanded / off.expanded) * 100) : 0;
    const dir = on.expanded <= off.expanded ? 'fewer' : 'more';
    setNote(`same ${ALGO_LABEL[algo]} search from this position: with pruning ${formatNumber(on.expanded)} nodes, without ${formatNumber(off.expanded)} (${diff}% ${dir} with pruning)`);
  } catch (err) {
    setNote(`comparison failed: ${err.message}`);
  } finally {
    $('cmp-spin').hidden = true;
    setBusy(false);
  }
}

// ── Levels and setup ─────────────────────────────────────────────────────

async function loadLevel(n) {
  if (S.busy) return;
  clearTimeout(S.evalTimer);
  setNote('loading level...');
  const lv = await getJSON(`/api/sokoban/level/${n}`);
  S.n = n;
  S.cols = lv.cols;
  S.rows = lv.rows;
  S.floor = new Uint8Array(S.cols * S.rows);
  for (const i of lv.floor) S.floor[i] = 1;
  S.goals = new Set(lv.goals);
  S.goalList = lv.goals.slice().sort((a, b) => a - b);
  S.dead = new Set(lv.dead);
  S.nearest = lv.nearest;
  S.start = { player: lv.player, boxes: lv.boxes.slice() };
  S.player = lv.player;
  S.boxes = lv.boxes.slice();
  S.moves = '';
  S.face = 'd';
  S.history = [];
  S.wasSolved = false;
  S.lastDeadlock = false;
  markStale();
  $('sel-level').value = String(n);
  $('chip-optimal').textContent = formatNumber(lv.optimal_pushes);
  $('chip-bound').textContent = '—';
  $('level-note').textContent = `${lv.boxes.length} ${lv.boxes.length === 1 ? 'box' : 'boxes'}. The fewest pushes is ${lv.optimal_pushes}.`;
  buildBoard();
  layout();
  renderState();
  $('think-title').textContent = 'Waiting for the first analysis.';
  $('think-body').replaceChildren();
  $('plan-title').textContent = 'Press SOLVE or HINT to search from this position.';
  $('plan-list').replaceChildren();
  resetCounters();
  log(`level ${n}: ${lv.name}`);
  setNote('Arrow keys or W A S D walk and push. U undo, R restart.');
  scheduleEval(0);
}

function toggleButton(id, on) {
  const btn = $(id);
  btn.setAttribute('aria-pressed', String(on));
  return btn;
}

async function init() {
  $('sel-level').addEventListener('change', (e) => {
    loadLevel(Number(e.target.value)).catch((err) => setNote(`could not load the level: ${err.message}`));
  });
  $('btn-undo').addEventListener('click', undo);
  $('btn-restart').addEventListener('click', restart);
  $('btn-hint').addEventListener('click', () => searchFromHere(true));
  $('btn-solve').addEventListener('click', () => searchFromHere(false));
  $('btn-stop').addEventListener('click', () => { S.stopping = true; });
  $('btn-compare').addEventListener('click', compare);
  $('tog-dead').addEventListener('click', (e) => {
    const on = e.currentTarget.getAttribute('aria-pressed') !== 'true';
    toggleButton('tog-dead', on);
    $('sok-board').classList.toggle('show-dead', on);
  });
  $('tog-dist').addEventListener('click', (e) => {
    const on = e.currentTarget.getAttribute('aria-pressed') !== 'true';
    toggleButton('tog-dist', on);
    $('sok-board').classList.toggle('show-dist', on);
  });
  $('tog-pairs').addEventListener('click', (e) => {
    const on = e.currentTarget.getAttribute('aria-pressed') !== 'true';
    toggleButton('tog-pairs', on);
    $('sok-wrap').classList.toggle('no-pairs', !on);
  });
  $('btn-prune').addEventListener('click', () => {
    S.prune = !S.prune;
    toggleButton('btn-prune', S.prune);
    $('btn-prune').querySelector('.btn-txt').textContent = `DEADLOCK PRUNING: ${S.prune ? 'ON' : 'OFF'}`;
  });
  $('sel-algo').addEventListener('change', (e) => {
    $('heur-wrap').hidden = e.target.value === 'bfs';
  });
  document.addEventListener('keydown', (e) => {
    // Keys belong to the form controls while one of them has focus.
    const tag = e.target && e.target.tagName;
    if (tag === 'SELECT' || tag === 'INPUT' || tag === 'TEXTAREA') return;
    if (e.altKey || e.ctrlKey || e.metaKey) return;
    const letter = ARROWS[e.key] || (/^[wasd]$/i.test(e.key) ? e.key.toLowerCase() : null);
    if (letter) {
      e.preventDefault();
      if (!S.busy) playMove(letter);
      return;
    }
    const k = e.key.toLowerCase();
    if (k === 'u') undo();
    else if (k === 'r') restart();
  });
  let frame = 0;
  window.addEventListener('resize', () => {
    cancelAnimationFrame(frame);
    frame = requestAnimationFrame(() => {
      if (S.floor) {
        layout();
      }
    });
  });

  try {
    const { levels } = await getJSON('/api/sokoban/levels');
    const sel = $('sel-level');
    sel.replaceChildren();
    for (const l of levels) {
      sel.appendChild(new Option(`${l.number}. ${l.name} (${l.optimal_pushes} pushes)`, String(l.number)));
    }
    const asked = Number(new URLSearchParams(location.search).get('level'));
    await loadLevel(asked >= 1 && asked <= levels.length ? asked : 1);
  } catch (err) {
    setNote(`could not reach the search server: ${err.message}`);
    $('chip-status').textContent = 'OFFLINE';
  }
}

init();
