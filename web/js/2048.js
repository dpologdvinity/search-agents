// 2048 page: the game runs in the browser; each AI move is one request to the Python agents.

import { getJSON, postJSON } from './api.js';
import { banner, burst, pop, shake } from './fx.js';

const $ = (id) => document.getElementById(id);
const DIRS = ['Up', 'Down', 'Left', 'Right'];
const NAMES = { expectimax: 'Expectimax (tuned features)',
                ntuple: 'N-tuple network (no search)', ntuple_search: 'N-tuple + expectimax' };

const state = { grid: [], score: 0, moves: 0, aiMoves: 0, playing: false, busy: false, meta: null, fresh: -1 };

// ── Rules ────────────────────────────────────────────────────────────────

function slideRow(row) {
  const tiles = row.filter((v) => v);
  const out = [];
  let gained = 0;
  for (let i = 0; i < tiles.length; i++) {
    if (i + 1 < tiles.length && tiles[i] === tiles[i + 1]) {
      out.push(tiles[i] * 2);
      gained += tiles[i] * 2;
      i++;
    } else out.push(tiles[i]);
  }
  while (out.length < 4) out.push(0);
  return [out, gained];
}

function moveGrid(grid, dir) {
  const g = grid.map((r) => r.slice());
  let gained = 0;
  for (let k = 0; k < 4; k++) {
    let line;
    if (dir === 2) line = g[k];
    else if (dir === 3) line = g[k].slice().reverse();
    else if (dir === 0) line = [0, 1, 2, 3].map((r) => g[r][k]);
    else line = [3, 2, 1, 0].map((r) => g[r][k]);
    const [moved, s] = slideRow(line);
    gained += s;
    for (let i = 0; i < 4; i++) {
      if (dir === 2) g[k][i] = moved[i];
      else if (dir === 3) g[k][3 - i] = moved[i];
      else if (dir === 0) g[i][k] = moved[i];
      else g[3 - i][k] = moved[i];
    }
  }
  const changed = g.some((r, i) => r.some((v, j) => v !== grid[i][j]));
  return changed ? [g, gained] : null;
}

function spawn(grid) {
  const empty = [];
  grid.forEach((r, i) => r.forEach((v, j) => { if (!v) empty.push([i, j]); }));
  if (!empty.length) return -1;
  const [i, j] = empty[Math.floor(Math.random() * empty.length)];
  grid[i][j] = Math.random() < 0.9 ? 2 : 4;
  return 4 * i + j;
}

const canMove = (grid) => [0, 1, 2, 3].some((d) => moveGrid(grid, d));

// ── Rendering ────────────────────────────────────────────────────────────

function tileStyle(v) {
  if (!v) return '';
  const k = Math.log2(v);
  const hue = 280 - Math.min(k, 13) * 18; // purple -> pink -> orange -> yellow
  const glow = Math.min(0.15 + k * 0.05, 0.85);
  const size = v < 128 ? '1.7rem' : v < 1024 ? '1.4rem' : v < 16384 ? '1.1rem' : '0.95rem';
  return `background:hsla(${hue},100%,55%,${0.12 + k * 0.045});color:hsl(${hue},100%,72%);`
    + `text-shadow:0 0 10px hsl(${hue},100%,60%);box-shadow:0 0 ${8 + k}px hsla(${hue},100%,55%,${glow});font-size:${size}`;
}

function render() {
  const el = $('g2048');
  el.innerHTML = '';
  state.grid.forEach((row, i) => row.forEach((v, j) => {
    const cell = document.createElement('div');
    cell.className = 'g-cell' + (4 * i + j === state.fresh ? ' new' : '');
    cell.style.cssText = tileStyle(v);
    cell.textContent = v || '';
    el.appendChild(cell);
  }));
  $('chip-score').textContent = state.score.toLocaleString();
  $('chip-moves').textContent = String(state.moves);
  $('chip-best').textContent = String(Math.max(...state.grid.flat()));
  $('st-ai').textContent = String(state.aiMoves);
}

function log(text, cls = 'log-info') {
  const el = $('log');
  const line = document.createElement('div');
  line.className = cls;
  line.textContent = text;
  el.appendChild(line);
  while (el.childElementCount > 200) el.firstChild.remove();
  el.scrollTop = el.scrollHeight;
}

function showValues(result) {
  const bars = $('an-bars');
  bars.innerHTML = '';
  const vals = DIRS.map((d) => result.values[d]);
  const legal = vals.filter((v) => v !== undefined);
  const lo = Math.min(...legal), hi = Math.max(...legal);
  DIRS.forEach((d, i) => {
    const v = vals[i];
    const col = document.createElement('div');
    col.className = 'dir-col' + (d === result.direction ? ' chosen' : '');
    const bar = document.createElement('div');
    bar.className = 'dir-bar';
    const frac = v === undefined ? 0 : hi === lo ? 1 : 0.15 + (0.85 * (v - lo)) / (hi - lo);
    bar.style.height = `${frac * 100}px`;
    const lbl = document.createElement('div');
    lbl.className = 'dir-lbl';
    lbl.innerHTML = `${{ Up: '↑', Down: '↓', Left: '←', Right: '→' }[d]} ${d}<br>${v === undefined ? 'illegal' : v.toFixed(1)}`;
    col.append(bar, lbl);
    bars.appendChild(col);
  });
  $('an-title').textContent = `${NAMES[result.agent]} → ${result.direction} · depth ${result.depth} · ${result.nodes.toLocaleString()} nodes`;
  $('an-note').textContent = result.agent.startsWith('ntuple')
    ? 'Values are expected future points, learned from self-play.'
    : 'Values are the hand-tuned evaluation, averaged over random tile spawns.';
  $('st-depth').textContent = String(result.depth);
  $('st-nodes').textContent = result.nodes.toLocaleString();
  $('st-time').textContent = `${(result.seconds * 1000).toFixed(0)} ms`;
}

// ── Play ─────────────────────────────────────────────────────────────────

function apply(dir, who) {
  const res = moveGrid(state.grid, dir);
  if (!res) return false;
  const before = Math.max(...state.grid.flat());
  [state.grid] = res;
  state.score += res[1];
  state.moves++;
  state.fresh = spawn(state.grid);
  render();
  const board = $('g2048'), best = Math.max(...state.grid.flat());
  if (res[1] >= 8) pop(board, `+${res[1]}`, res[1] >= 256 ? '#ff00a0' : '#ffe600');
  if (best > before && best >= 128) {
    burst(board, { count: Math.min(30 + Math.log2(best) * 10, 160) });
    if (best >= 512) { shake(board, 'big'); banner(`${best}`, `new highest tile at move ${state.moves}`, '#9b00ff'); }
  }
  if (!canMove(state.grid)) {
    state.playing = false;
    $('chip-status').textContent = 'GAME OVER';
    $('btn-play').querySelector('.btn-txt').textContent = '▶ AI PLAY';
    log(`■ Game over: score ${state.score.toLocaleString()}, best tile ${Math.max(...state.grid.flat())} (${who}).`, 'log-best');
    shake($('g2048'), 'big');
    banner('GAME OVER', `score ${state.score.toLocaleString()} · best tile ${Math.max(...state.grid.flat())}`, '#ff00a0');
  }
  return true;
}

async function aiStep() {
  if (state.busy || !canMove(state.grid)) return false;
  state.busy = true;
  try {
    const result = await postJSON('/api/2048/move', { grid: state.grid, agent: $('agent').value });
    showValues(result);
    state.aiMoves++;
    const before = Math.max(...state.grid.flat());
    apply(result.move, NAMES[result.agent]);
    const after = Math.max(...state.grid.flat());
    if (after > before && after >= 256) log(`★ ${after} tile reached at move ${state.moves}`, 'log-best');
    return true;
  } catch (err) {
    log(`✗ ${err.message}`, 'log-err');
    state.playing = false;
    return false;
  } finally {
    state.busy = false;
  }
}

async function autoplay() {
  if (state.playing) {
    state.playing = false;
    $('btn-play').querySelector('.btn-txt').textContent = '▶ AI PLAY';
    $('chip-status').textContent = 'PAUSED';
    return;
  }
  state.playing = true;
  $('btn-play').querySelector('.btn-txt').textContent = '⏸ PAUSE';
  $('chip-status').textContent = 'AI PLAYING';
  while (state.playing && canMove(state.grid)) {
    if (!(await aiStep())) break;
    await new Promise((r) => setTimeout(r, Number($('delay').value)));
  }
}

function newGame() {
  state.playing = false;
  state.grid = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]];
  state.score = state.moves = state.aiMoves = 0;
  spawn(state.grid);
  state.fresh = spawn(state.grid);
  $('btn-play').querySelector('.btn-txt').textContent = '▶ AI PLAY';
  $('chip-status').textContent = 'READY';
  render();
}

async function init() {
  $('btn-new').onclick = newGame;
  $('btn-step').onclick = aiStep;
  $('btn-play').onclick = autoplay;
  $('delay').oninput = () => { $('delay-lbl').textContent = $('delay').value; };
  $('agent').onchange = () => {
    const a = state.meta.agents.find((x) => x.name === $('agent').value);
    $('agent-desc').textContent = a ? a.description : '';
  };
  const keys = { ArrowUp: 0, ArrowDown: 1, ArrowLeft: 2, ArrowRight: 3, w: 0, s: 1, a: 2, d: 3 };
  window.addEventListener('keydown', (e) => {
    if (!(e.key in keys) || state.playing || e.target.tagName === 'SELECT') return;
    e.preventDefault();
    if (apply(keys[e.key], 'you')) $('chip-status').textContent = 'YOU';
  });
  newGame();
  try {
    state.meta = await getJSON('/api/2048/meta');
  } catch (err) {
    log(`✗ Server unreachable (${err.message}). You can still play by hand.`, 'log-err');
    return;
  }
  for (const a of state.meta.agents) {
    const opt = document.createElement('option');
    opt.value = a.name;
    opt.textContent = NAMES[a.name] + (a.available ? '' : ' (not trained yet)');
    opt.disabled = !a.available;
    $('agent').appendChild(opt);
  }
  const best = state.meta.agents.find((a) => a.name === 'ntuple_search' && a.available);
  $('agent').value = best ? 'ntuple_search' : 'expectimax';
  $('agent').onchange();
}

init();
