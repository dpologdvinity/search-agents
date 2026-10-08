// Hex page.
//
// The server holds the rules and does the search. This page keeps the move history, sends it with each
// request, and draws what comes back. The board is redrawn from the history after every change, so the
// page never has its own copy of the position to fall out of step with the server.
//
// A request is POST /api/hexgame/move with the history and the agent that should move (or null when it is
// the player's turn). The response says who won (and the winning chain), and if the game is open and an
// agent was named, that agent's move and the search behind it.

import { postJSON } from './api.js';
import { banner, burst, pop, shake } from './fx.js';

const $ = (id) => document.getElementById(id);
const SVG = 'http://www.w3.org/2000/svg';
const DOWN = 1, ACROSS = 2, SWAP = -1;
const AGENT_NAME = { rave: 'RAVE-MCTS', uct: 'UCT', shortest: 'SHORTEST', random: 'RANDOM' };
const WATCH_MS = 450;     // pause between moves when two agents play
const K_RAVE = 300;       // the server's RAVE equivalence parameter, used only to explain beta on the page

const state = {
  n: 7,
  swap: false,
  mode: 'vs',          // 'vs' (you against the agent) or 'watch' (two agents)
  you: DOWN,
  moves: [],           // the history sent to the server; -1 is the swap
  cells: [],           // replayed from moves, for drawing: 0 empty, 1 DOWN, 2 ACROSS
  to_move: DOWN,
  winner: null,
  chain: null,
  busy: false,
  watching: false,
  analysis: null,      // the last search shown in the panel
  analysisSide: null,  // the side that search was for
  hintCell: null,
  lastMove: null,
  heat: 'visits',
  sims: null,
  speed: null,
};

// ── Geometry ─────────────────────────────────────────────────────────────
// Pointy-top hexagons. Row r is shifted right by half a hexagon per row, which makes the rhombus.
// Vertex k sits at angle -90 + 60k degrees. Edge k joins vertex k and k+1; edges 5 and 0 face up,
// 2 and 3 face down, 4 faces left and 1 faces right.

function layout(n) {
  const R = Math.min(34, 560 / (Math.sqrt(3) * (1.5 * n + 1)));
  const w = Math.sqrt(3) * R;
  const dy = 1.5 * R;
  const pad = 14;
  const cx = (r, c) => pad + w / 2 + (c + r / 2) * w;
  const cy = (r) => pad + R + r * dy;
  return {
    R, w, cx, cy,
    width: 2 * pad + w + 1.5 * (n - 1) * w,
    height: 2 * pad + 2 * R + (n - 1) * dy,
  };
}

function corners(x, y, R) {
  const pts = [];
  for (let k = 0; k < 6; k++) {
    const a = ((-90 + 60 * k) * Math.PI) / 180;
    pts.push(`${(x + R * Math.cos(a)).toFixed(2)},${(y + R * Math.sin(a)).toFixed(2)}`);
  }
  return pts.join(' ');
}

function edgeSegment(x, y, R, k) {
  const pts = [k, (k + 1) % 6].map((j) => {
    const a = ((-90 + 60 * j) * Math.PI) / 180;
    return [x + R * Math.cos(a), y + R * Math.sin(a)];
  });
  return `M${pts[0][0].toFixed(2)} ${pts[0][1].toFixed(2)} L${pts[1][0].toFixed(2)} ${pts[1][1].toFixed(2)}`;
}

const label = (cell, n) => `${String.fromCharCode(97 + (cell % n))}${Math.floor(cell / n) + 1}`;

// Rebuild the cells from the history, as the server would: stones alternate, and a swap flips the first stone.
function replay(n, moves) {
  const cells = new Array(n * n).fill(0);
  let p = DOWN;
  for (const m of moves) {
    if (m === SWAP) {
      const first = moves[0];
      cells[first] = 3 - cells[first];
    } else {
      cells[m] = p;
    }
    p = 3 - p;
  }
  return { cells, to_move: p };
}

// ── Drawing ──────────────────────────────────────────────────────────────

function el(tag, attrs = {}, parent = null) {
  const node = document.createElementNS(SVG, tag);
  for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, String(v));
  if (parent) parent.appendChild(node);
  return node;
}

// Gradients and glow filters for the stones, defined once per board build.
function defs(svg) {
  const d = el('defs', {}, svg);
  const grad = (id, c1, c2, c3) => {
    const g = el('radialGradient', { id, cx: '35%', cy: '30%', r: '75%' }, d);
    el('stop', { offset: '0%', 'stop-color': c1 }, g);
    el('stop', { offset: '45%', 'stop-color': c2 }, g);
    el('stop', { offset: '100%', 'stop-color': c3 }, g);
  };
  grad('hx-grad-down', '#ff8ad0', '#ff00a0', '#8a0058');
  grad('hx-grad-across', '#ffffff', '#9feeff', '#2b7f99');
  for (const [id, color] of [['hx-glow-pink', '#ff00a0'], ['hx-glow-cyan', '#00f5ff']]) {
    const f = el('filter', { id, x: '-50%', y: '-50%', width: '200%', height: '200%' }, d);
    el('feDropShadow', { dx: 0, dy: 0, stdDeviation: 3, 'flood-color': color, 'flood-opacity': 0.9 }, f);
  }
}

// Heat colour for a cell: visit share (or RAVE rate) turns the empty cell from dark to lime.
function heatFill(t) {
  const a = (0.08 + 0.72 * Math.max(0, Math.min(1, t))).toFixed(3);
  return `rgba(0,255,136,${a})`;
}

function drawBoard() {
  const svg = $('hx-board');
  const n = state.n;
  const L = layout(n);
  svg.innerHTML = '';
  svg.setAttribute('viewBox', `0 0 ${L.width.toFixed(1)} ${L.height.toFixed(1)}`);
  defs(svg);
  const an = state.analysis;
  const haveSearch = an && an.visits;
  // Heat is scaled between the least and most visited empty cell, so the colours separate the candidates.
  const emptyVisits = haveSearch ? an.visits.filter((v, i) => state.cells[i] === 0) : [];
  const minVisits = emptyVisits.length ? Math.min(...emptyVisits) : 0;
  const maxVisits = emptyVisits.length ? Math.max(1, ...emptyVisits) : 1;
  const pvIndex = new Map(haveSearch && an.pv ? an.pv.map((c, i) => [c, i + 1]) : []);
  const chainIndex = new Map(state.chain ? state.chain.map((c, i) => [c, i]) : []);

  const cellsGroup = el('g', {}, svg);
  for (let r = 0; r < n; r++) {
    for (let c = 0; c < n; c++) {
      const i = r * n + c;
      const x = L.cx(r, c), y = L.cy(r);
      const owner = state.cells[i];
      const open = owner === 0 && !state.winner && !state.busy;
      const classes = ['hx-cell'];
      if (owner === DOWN) classes.push('down');
      if (owner === ACROSS) classes.push('across');
      if (state.lastMove === i) classes.push('last');
      if (state.hintCell === i && owner === 0) classes.push('hint');
      if (chainIndex.has(i)) classes.push('chain');
      const poly = el('polygon', {
        class: classes.join(' '), points: corners(x, y, L.R), 'data-cell': i,
        role: 'button', 'aria-label': `${label(i, n)}, ${owner ? (owner === DOWN ? 'DOWN stone' : 'ACROSS stone') : 'empty'}`,
      }, cellsGroup);
      if (open) poly.setAttribute('tabindex', '0');
      if (chainIndex.has(i)) poly.style.animationDelay = `${chainIndex.get(i) * 70}ms`;
      el('title', {}, poly).textContent = `${label(i, n)}${owner ? '' : ' (empty)'}`;
    }
  }

  // Heat overlay and principal variation, drawn over the cells but never clickable.
  if (haveSearch && !state.winner) {
    const rave = an.rave_rate || [];
    const overlay = el('g', { class: 'hx-heat-layer' }, svg);
    for (let r = 0; r < n; r++) {
      for (let c = 0; c < n; c++) {
        const i = r * n + c;
        if (state.cells[i] !== 0) continue;
        const x = L.cx(r, c), y = L.cy(r);
        let t = null, text = '';
        if (state.heat === 'visits') {
          const v = an.visits[i];
          if (v > 0) {
            t = maxVisits > minVisits ? (v - minVisits) / (maxVisits - minVisits) : 1;
            text = v >= 1000 ? `${(v / 1000).toFixed(1)}k` : String(v);
          }
        } else if (rave[i] !== null && rave[i] !== undefined) {
          const rs = rave.filter((x, j) => state.cells[j] === 0 && x !== null && x !== undefined);
          const lo = Math.min(...rs), hi = Math.max(...rs);
          t = hi > lo ? (rave[i] - lo) / (hi - lo) : 1;
          text = `${Math.round(100 * rave[i])}`;
        }
        if (t !== null) {
          el('polygon', { class: 'hx-heat', points: corners(x, y, L.R), fill: heatFill(t) }, overlay);
          const tx = el('text', { class: 'hx-heat-lbl', x: x.toFixed(2), y: (y + L.R * 0.15).toFixed(2),
            'font-size': (L.R * 0.5).toFixed(1) }, overlay);
          tx.textContent = text;
        }
        if (pvIndex.has(i)) {
          el('text', { class: 'hx-pv-num', x: x.toFixed(2), y: (y + L.R * 0.36).toFixed(2),
            'font-size': (L.R * 0.95).toFixed(1) }, overlay).textContent = String(pvIndex.get(i));
        }
      }
    }
  }

  // Goal edges: DOWN's start and goal are the top and bottom rows; ACROSS's are the left and right columns.
  const edges = el('g', { 'pointer-events': 'none' }, svg);
  for (let r = 0; r < n; r++) {
    for (let c = 0; c < n; c++) {
      const x = L.cx(r, c), y = L.cy(r);
      const add = (k, cls) => el('path', { class: `hx-edge ${cls}`, d: edgeSegment(x, y, L.R, k) }, edges);
      if (r === 0) { add(5, 'down'); add(0, 'down'); }
      if (r === n - 1) { add(2, 'down'); add(3, 'down'); }
      if (c === 0) { add(4, 'across'); add(3, 'across'); }
      if (c === n - 1) { add(1, 'across'); add(0, 'across'); }
    }
  }
  svg.classList.toggle('won', !!state.winner);
}

// ── Panel ────────────────────────────────────────────────────────────────

function renderPanel() {
  const an = state.analysis;
  const n = state.n;
  const tbody = $('cand-body');
  const pvLine = $('pv-line');
  const note = $('think-note');
  if (!an) {
    $('think-title').textContent = 'Make a move, or press HINT, to see the search.';
    tbody.innerHTML = '<tr><td colspan="5" class="field-hint">No search yet.</td></tr>';
    pvLine.textContent = '';
    note.textContent = '';
    return;
  }
  const who = state.analysisSide === DOWN ? 'DOWN' : 'ACROSS';
  const agentName = AGENT_NAME[an.agent] || an.agent;
  $('think-title').textContent = `${agentName} searched ${an.sims} simulations for ${who} (${an.playouts} random playouts).`;

  if (an.visits) {
    const rows = [];
    an.visits.forEach((v, i) => { if (v > 0) rows.push([i, v]); });
    rows.sort((a, b) => b[1] - a[1]);
    const maxV = Math.max(1, rows.length ? rows[0][1] : 1);
    tbody.innerHTML = '';
    for (const [i, v] of rows.slice(0, 6)) {
      const tr = document.createElement('tr');
      if (i === an.move) tr.className = 'top';
      const wr = an.win_rate[i];
      const rv = an.rave_rate ? an.rave_rate[i] : null;
      tr.innerHTML = `<td>${label(i, n)}</td><td class="num">${v}</td>`
        + `<td class="num">${wr === null ? '—' : Math.round(100 * wr) + '%'}</td>`
        + `<td class="num">${rv === null || rv === undefined ? '—' : Math.round(100 * rv) + '%'}</td>`
        + `<td><div class="bar"><i style="width:${(100 * v / maxV).toFixed(1)}%"></i></div></td>`;
      tbody.appendChild(tr);
    }
  } else {
    tbody.innerHTML = '<tr><td colspan="5" class="field-hint">No visit table for this move.</td></tr>';
  }

  pvLine.textContent = an.pv_labels && an.pv_labels.length
    ? `Principal variation: ${an.pv_labels.slice(0, 10).join(' → ')}` : '';

  // Explain the choice in the agent's own terms.
  if (an.agent === 'shortest') {
    note.textContent = 'This agent does not search. It plays the cell that most lengthens the opponent\'s shortest route '
      + 'and most shortens its own.';
  } else if (an.agent === 'random') {
    note.textContent = 'This agent plays a uniformly random legal move.';
  } else if (an.swap) {
    const s = an.swap;
    note.textContent = `Swap check: win chance ${Math.round(100 * s.swap)}% after swapping, ${Math.round(100 * s.stay)}% if it stays. `
      + (s.chose === 'swap' ? 'It swaps.' : 'It plays on.');
  } else if (an.agent === 'rave' && an.visits && an.move !== null && an.move >= 0) {
    const v = an.visits[an.move];
    const beta = Math.sqrt(K_RAVE / (3 * v + K_RAVE));
    const rv = an.rave_rate ? an.rave_rate[an.move] : null;
    note.textContent = `${label(an.move, n)}: ${v} visits. Its own win rate is ${Math.round(100 * (an.win_rate[an.move] || 0))}%`
      + (rv !== null && rv !== undefined ? `, its RAVE rate ${Math.round(100 * rv)}%` : '')
      + `. At this many visits beta = ${beta.toFixed(2)}: the shared estimate carries ${Math.round(100 * beta)}% of the score.`;
  } else if (an.visits && an.move !== null && an.move >= 0) {
    const v = an.visits[an.move];
    note.textContent = `${label(an.move, n)}: ${v} visits. Plain UCT has no shared estimates, so its win rate is the only signal.`;
  } else {
    note.textContent = '';
  }
}

function renderGauge() {
  const an = state.analysis;
  let pDown = 0.5;
  if (an && typeof an.value === 'number') {
    pDown = state.analysisSide === DOWN ? an.value : 1 - an.value;
  }
  if (state.winner) pDown = state.winner === DOWN ? 1 : 0;
  const pct = (x) => `${Math.round(100 * x)}%`;
  $('gauge-down').style.width = pct(pDown);
  $('gauge-across').style.width = pct(1 - pDown);
  $('gauge-down-lbl').textContent = `DOWN ${pct(pDown)}`;
  $('gauge-across-lbl').textContent = `ACROSS ${pct(1 - pDown)}`;
}

// ── Status ───────────────────────────────────────────────────────────────

function statusText() {
  if (state.winner) return 'FINISHED';
  if (state.busy) return 'THINKING';
  if (state.watching) return 'WATCHING';
  return 'PLAYING';
}

function renderChrome() {
  $('chip-status').textContent = statusText();
  $('chip-moves').textContent = String(state.moves.length);
  $('chip-sims').textContent = state.sims ? String(state.sims) : '—';
  $('chip-speed').textContent = state.speed ? `${state.speed} /s` : '—';
  $('tag-down').textContent = state.mode === 'vs' && state.you === DOWN ? 'YOU · DOWN · X · TOP TO BOTTOM' : 'DOWN · X · TOP TO BOTTOM';
  $('tag-across').textContent = state.mode === 'vs' && state.you === ACROSS ? 'YOU · ACROSS · O · LEFT TO RIGHT' : 'ACROSS · O · LEFT TO RIGHT';
  const idle = !state.busy && !state.winner && !state.watching;
  $('btn-hint').disabled = !idle;
  const canSwap = state.swap && state.mode === 'vs' && state.moves.length === 1 && state.to_move === state.you
    && state.you === ACROSS && !state.winner;
  $('btn-swap').hidden = !canSwap;
  $('btn-swap').disabled = state.busy;
  $('btn-watch').hidden = state.mode !== 'watch';
  $('btn-watch').querySelector('.btn-txt').textContent = state.watching ? '■ STOP' : '▶ START';
  $('btn-watch').setAttribute('aria-pressed', String(state.watching));
  $('btn-new').disabled = state.busy && !state.watching;
  $('you-row').hidden = state.mode !== 'vs';
  $('agent2-row').hidden = state.mode !== 'watch';
  $('agent-lbl').textContent = state.mode === 'watch' ? 'DOWN AGENT' : 'AGENT';
  $('heat-visits').setAttribute('aria-pressed', String(state.heat === 'visits'));
  $('heat-rave').setAttribute('aria-pressed', String(state.heat === 'rave'));
}

function render() {
  const replayed = replay(state.n, state.moves);
  state.cells = replayed.cells;
  state.to_move = replayed.to_move;
  drawBoard();
  renderPanel();
  renderGauge();
  renderChrome();
  $('hx-note').textContent = noteText();
}

function noteText() {
  if (state.winner) {
    if (state.mode === 'watch') return `${NAME(state.winner)} connected its edges.`;
    return state.winner === state.you ? 'You connected your edges. You win.' : 'The agent connected its edges. The agent wins.';
  }
  if (state.mode === 'watch') return state.watching ? 'Two agents are playing.' : 'Press START to watch the two agents.';
  if (state.to_move === state.you) return 'Your move. Click a hexagon. Your goal is the coloured edges.';
  return 'The agent is thinking.';
}

const NAME = (side) => (side === DOWN ? 'DOWN' : 'ACROSS');

function log(text, cls = '') {
  const box = $('log');
  const line = document.createElement('div');
  if (cls) line.className = cls;
  line.textContent = text;
  box.appendChild(line);
  box.scrollTop = box.scrollHeight;
}

// ── Game flow ────────────────────────────────────────────────────────────

function setBusy(b) {
  state.busy = b;
}

function sleep(ms) { return new Promise((r) => setTimeout(r, ms)); }

// Which agent moves for `side`, or null when that side is the player.
function agentFor(side) {
  if (state.mode === 'watch') return side === DOWN ? $('agent').value : $('agent2').value;
  return side === state.you ? null : $('agent').value;
}

async function request(agent) {
  return postJSON('/api/hexgame/move', {
    size: state.n,
    moves: state.moves,
    swap: state.swap,
    agent,
    level: Number($('level').value),
  });
}

// Ask for the next move and apply it, until the game ends or it is the player's turn (or watching stops).
async function play() {
  setBusy(true);
  renderChrome();
  try {
    for (;;) {
      if (state.winner) break;
      const side = state.to_move;
      const agent = agentFor(side);
      if (state.mode === 'watch' && !state.watching) break;
      const res = await request(agent);
      if (res.winner) {
        finish(res);
        break;
      }
      if (res.move === null || res.move === undefined) break; // the player's turn
      const swapped = res.move === SWAP;
      state.moves.push(res.move);
      state.lastMove = swapped ? null : res.move;
      state.hintCell = null;
      state.analysis = res.analysis;
      state.analysisSide = side;
      state.sims = res.analysis && res.analysis.sims ? res.analysis.sims : state.sims;
      if (res.analysis && res.analysis.seconds > 0 && res.analysis.sims) {
        state.speed = Math.round(res.analysis.sims / res.analysis.seconds);
      }
      const text = swapped
        ? `${AGENT_NAME[agent] || agent} swaps: it takes your stone`
        : `${AGENT_NAME[agent] || agent} plays ${res.label}`
          + (res.analysis && res.analysis.visits ? ` (${res.analysis.visits[res.move]} visits)` : '');
      log(`${NAME(side)}: ${text}`, side === DOWN ? 'hx-pink' : 'hx-cyan');
      render();
      if (swapped) shake($('hx-board'));
      else {
        const cell = document.querySelector(`#hx-board polygon[data-cell="${res.move}"]`);
        if (cell) pop(cell, res.label, side === DOWN ? '#ff00a0' : '#00f5ff');
      }
      if (state.mode === 'watch') await sleep(WATCH_MS);
    }
  } catch (e) {
    log(`error: ${e.message}`, 'hx-pink');
    $('hx-note').textContent = e.message;
    state.watching = false;
  } finally {
    setBusy(false);
    render();
  }
}

function finish(res) {
  state.winner = res.winner;
  state.chain = res.chain;
  state.watching = false;
  const who = state.mode === 'watch' ? `${NAME(res.winner)} wins`
    : res.winner === state.you ? 'YOU WIN' : 'AGENT WINS';
  const sub = `${NAME(res.winner)} connected in ${state.moves.length} moves`;
  log(`${NAME(res.winner)} wins with ${res.chain.length} stones on the chain`, 'hx-cyan');
  if (res.chain) {
    // The chain's stones pulse in turn (drawBoard staggers them by their place on the chain).
    burst(document.querySelector(`#hx-board polygon[data-cell="${res.chain[res.chain.length - 1]}"]`), { count: 80 });
  }
  banner(who, sub, res.winner === DOWN ? '#ff00a0' : '#00f5ff');
}

function newGame() {
  if (state.watching) state.watching = false;
  state.moves = [];
  state.winner = null;
  state.chain = null;
  state.analysis = null;
  state.analysisSide = null;
  state.hintCell = null;
  state.lastMove = null;
  state.sims = null;
  state.speed = null;
  $('log').innerHTML = '';
  render();
  if (state.mode === 'vs' && state.you === ACROSS) play(); // the agent opens as DOWN
}

async function clickCell(i) {
  if (state.busy || state.winner || state.watching) return;
  if (state.mode !== 'vs' || state.to_move !== state.you) return;
  if (state.cells[i] !== 0) return;
  state.moves.push(i);
  state.lastMove = i;
  state.hintCell = null;
  state.analysis = null;
  log(`${NAME(state.you)}: you play ${label(i, state.n)}`, state.you === DOWN ? 'hx-pink' : 'hx-cyan');
  render();
  await play();
}

async function swapByPlayer() {
  if (state.busy || state.winner) return;
  state.moves.push(SWAP);
  state.analysis = null;
  log('ACROSS: you swap. The stone is yours now.', 'hx-cyan');
  render();
  await play();
}

async function hint() {
  if (state.busy || state.winner || state.watching) return;
  const side = state.to_move;
  setBusy(true);
  renderChrome();
  try {
    const res = await request('rave');
    state.analysis = res.analysis;
    state.analysisSide = side;
    state.hintCell = res.move === SWAP || res.move === null ? null : res.move;
    state.sims = res.analysis ? res.analysis.sims : state.sims;
    if (res.analysis && res.analysis.seconds > 0) state.speed = Math.round(res.analysis.sims / res.analysis.seconds);
    $('hx-note').textContent = res.move === SWAP ? 'Hint: the search says swap.'
      : `Hint: ${res.label} (not played).`;
  } catch (e) {
    $('hx-note').textContent = e.message;
  } finally {
    setBusy(false);
    render();
  }
}

// ── Wiring ───────────────────────────────────────────────────────────────

function fillSelect(select, from, to) {
  select.innerHTML = '';
  for (let v = from; v <= to; v++) {
    const o = document.createElement('option');
    o.value = String(v);
    o.textContent = `${v} × ${v}`;
    if (v === state.n) o.selected = true;
    select.appendChild(o);
  }
}

function init() {
  fillSelect($('size'), 5, 11);
  $('size').value = String(state.n);
  $('size').addEventListener('change', () => { state.n = Number($('size').value); newGame(); });
  $('mode').addEventListener('change', () => {
    state.mode = $('mode').value;
    state.watching = false;
    newGame();
  });
  $('you').addEventListener('change', () => { state.you = Number($('you').value); newGame(); });
  $('swap-rule').addEventListener('change', () => { state.swap = $('swap-rule').checked; newGame(); });
  $('btn-new').addEventListener('click', () => newGame());
  $('btn-hint').addEventListener('click', () => hint());
  $('btn-swap').addEventListener('click', () => swapByPlayer());
  $('btn-watch').addEventListener('click', () => {
    if (state.watching) { state.watching = false; renderChrome(); return; }
    if (state.winner) newGame();
    state.watching = true;
    renderChrome();
    play();
  });
  $('heat-visits').addEventListener('click', () => { state.heat = 'visits'; render(); });
  $('heat-rave').addEventListener('click', () => { state.heat = 'rave'; render(); });

  const svg = $('hx-board');
  svg.addEventListener('click', (e) => {
    const t = e.target.closest('[data-cell]');
    if (t) clickCell(Number(t.dataset.cell));
  });
  svg.addEventListener('keydown', (e) => {
    if (e.key !== 'Enter' && e.key !== ' ') return;
    const t = e.target.closest('[data-cell]');
    if (t) { e.preventDefault(); clickCell(Number(t.dataset.cell)); }
  });
  newGame();
}

init();
