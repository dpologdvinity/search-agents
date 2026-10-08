// Battleship page.
//
// Two fleets, two owners. The agent's fleet is random and lives on the server (POST /new); the server answers the
// player's shots at it (POST /shot) and reveals it only when the player wins. The player's fleet lives here: the
// agent's shots are resolved on this page, and the page sends the history of those shots to the server
// (POST /agent-shot), which returns the agent's odds over this fleet and the cell it fires at next.
//
// So the page holds every answer about the player's fleet, and the server holds every answer about the agent's.
// Each agent turn is: aim (the reticle sits on the cell the server picked), fire (resolved locally), then ask
// the server for fresh odds from the updated history.

import { getJSON, postJSON } from './api.js';
import { banner, burst, pop, shake } from './fx.js';

const $ = (id) => document.getElementById(id);
const SIZE = 10;
const COLS = 'ABCDEFGHIJ';
const AIM_MS = 650; // reticle time before each agent shot in battle
// Time per agent shot when watching. Each turn also makes one request, and the server allows 120 moves a minute
// per client, so the fast setting stays above about half a second.
const PAUSE = { slow: 1200, normal: 800, fast: 500 };

const cellName = (i) => `${COLS[i % SIZE]}${Math.floor(i / SIZE) + 1}`;
const cellIndex = (name) => COLS.indexOf(name[0]) + (parseInt(name.slice(1), 10) - 1) * SIZE;
const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

const state = {
  meta: null,
  phase: 'place', // place | battle | watch | over
  vertical: false, // orientation for the next ship placed
  selected: -1, // index into the fleet of the ship being placed
  placed: [], // ships on your grid while placing: {idx, name, length, cells}
  hover: null, // placement preview under the pointer: {cells, ok}
  mine: [], // your ships once the battle starts: {idx, name, length, cells}
  mineStatus: Array(SIZE * SIZE).fill(null), // per cell of your grid: null | miss | hit | sunk
  shipHits: [], // hits taken per ship of yours
  reports: [], // the agent's shots at your fleet, in order, in the form /agent-shot expects
  odds: null, // last /agent-shot response
  aim: null, // cell the reticle is on, while the agent aims
  gameId: null,
  enemyStatus: Array(SIZE * SIZE).fill(null), // per cell of enemy waters: null | miss | hit | sunk
  enemySunk: 0,
  enemyFleet: null, // the agent's fleet, revealed on a win: [{name, cells: [names]}]
  shotsMine: 0,
  shotsAgent: 0,
  heat: true,
  busy: false,
  speed: 'normal',
  token: 0, // bumped by NEW MATCH so that loops still running from an old match stop
  endTitle: '', // banner text of the finished match, kept on the status chip
};

const boards = { me: [], enemy: [] };

// ── Geometry and placement ───────────────────────────────────────────────

/** Cells of a ship of `length` starting at `start`, or null if it would leave the board. */
function shipCells(start, length, vertical) {
  const r0 = Math.floor(start / SIZE), c0 = start % SIZE;
  const cells = [];
  for (let k = 0; k < length; k++) {
    const r = r0 + (vertical ? k : 0), c = c0 + (vertical ? 0 : k);
    if (r >= SIZE || c >= SIZE) return null;
    cells.push(r * SIZE + c);
  }
  return cells;
}

function fleetDef() {
  return state.meta ? state.meta.fleet : [];
}

function occupied(except = null) {
  const taken = new Set();
  for (const s of state.placed) if (s !== except) s.cells.forEach((c) => taken.add(c));
  return taken;
}

/** Preview of the selected ship under the pointer: its cells, and whether it fits. */
function previewAt(i) {
  if (state.selected < 0) return null;
  const def = fleetDef()[state.selected];
  const cells = shipCells(i, def.length, state.vertical);
  if (!cells) return { cells: [], ok: false };
  const taken = occupied();
  return { cells, ok: cells.every((c) => !taken.has(c)) };
}

/** Random legal fleet: each ship gets random spots until one fits. Gives up and retries after too many misses. */
function randomFleet() {
  for (let attempt = 0; attempt < 200; attempt++) {
    const placed = [];
    const taken = new Set();
    let ok = true;
    fleetDef().forEach((def, idx) => {
      if (!ok) return;
      for (let t = 0; t < 300; t++) {
        const cells = shipCells(Math.floor(Math.random() * SIZE * SIZE), def.length, Math.random() < 0.5);
        if (cells && cells.every((c) => !taken.has(c))) {
          cells.forEach((c) => taken.add(c));
          placed.push({ idx, name: def.name, length: def.length, cells });
          return;
        }
      }
      ok = false;
    });
    if (ok) return placed;
  }
  return [];
}

function fleetComplete() {
  return state.placed.length === fleetDef().length;
}

function nextUnplaced() {
  const done = new Set(state.placed.map((s) => s.idx));
  return fleetDef().findIndex((_, idx) => !done.has(idx));
}

/** Take a placed ship back off the grid so it can be moved. */
function pickUp(ship) {
  state.placed = state.placed.filter((s) => s !== ship);
  state.selected = ship.idx;
}

// ── Boards ───────────────────────────────────────────────────────────────

function buildBoard(id, key) {
  const el = $(id);
  el.innerHTML = '';
  const label = (text) => {
    const d = document.createElement('div');
    d.className = 'bs-lbl';
    d.textContent = text;
    return d;
  };
  el.appendChild(label(''));
  for (const c of COLS) el.appendChild(label(c));
  for (let r = 0; r < SIZE; r++) {
    el.appendChild(label(String(r + 1)));
    for (let c = 0; c < SIZE; c++) {
      const i = r * SIZE + c;
      const cell = document.createElement('div');
      cell.className = 'bs-cell';
      cell.dataset.i = i;
      cell.title = cellName(i);
      cell.setAttribute('role', 'gridcell');
      cell.appendChild(Object.assign(document.createElement('div'), { className: 'bs-heat' }));
      el.appendChild(cell);
      boards[key][i] = cell;
    }
  }
}

/**
 * Glow for the agent's odds. Early on the odds are nearly flat, so the wash is measured from the mean: cells
 * above average glow, the hottest brightest, and cells at or below average stay clear.
 */
function heatAlphas() {
  const out = Array(SIZE * SIZE).fill(0);
  if (!state.heat || !state.odds || state.phase === 'place') return out;
  const unknown = [];
  for (let i = 0; i < SIZE * SIZE; i++) if (!state.mineStatus[i]) unknown.push(i);
  if (!unknown.length) return out;
  const p = state.odds.probs;
  const mean = unknown.reduce((sum, i) => sum + p[i], 0) / unknown.length;
  const top = Math.max(...unknown.map((i) => p[i]));
  const span = Math.max(1e-9, top - mean);
  for (const i of unknown) out[i] = 0.8 * Math.min(1, Math.max(0, (p[i] - mean) / span));
  return out;
}

function paintMine() {
  const ships = state.phase === 'place' ? state.placed : state.mine;
  const hull = new Set(ships.flatMap((s) => s.cells));
  const alphas = heatAlphas();
  for (let i = 0; i < SIZE * SIZE; i++) {
    const cell = boards.me[i];
    const cls = ['bs-cell'];
    const st = state.mineStatus[i];
    if (st) cls.push(st);
    else if (hull.has(i)) cls.push('ship');
    if (state.hover && state.hover.cells.includes(i)) cls.push(state.hover.ok ? 'prev-ok' : 'prev-bad');
    if (state.aim === i) cls.push('aim');
    cell.className = cls.join(' ');
    const a = alphas[i];
    cell.firstChild.style.background = a > 0.005 ? `rgba(255,0,160,${a.toFixed(3)})` : '';
  }
}

function paintEnemy() {
  const reveal = new Set(state.enemyFleet ? state.enemyFleet.flatMap((s) => s.cells.map(cellIndex)) : []);
  for (let i = 0; i < SIZE * SIZE; i++) {
    const cls = ['bs-cell'];
    const st = state.enemyStatus[i];
    if (st) cls.push(st);
    else if (reveal.has(i)) cls.push('ship');
    boards.enemy[i].className = cls.join(' ');
  }
  $('enemy').classList.toggle('live', state.phase === 'battle' && !state.busy);
}

function paintTray() {
  const tray = $('tray');
  tray.innerHTML = '';
  fleetDef().forEach((def, idx) => {
    const btn = document.createElement('button');
    btn.type = 'button';
    const placed = state.placed.some((s) => s.idx === idx);
    btn.className = 'bs-ship' + (placed ? ' placed' : '') + (state.selected === idx ? ' active' : '');
    btn.disabled = state.phase !== 'place';
    const name = document.createElement('span');
    name.textContent = def.name;
    const blocks = document.createElement('span');
    blocks.className = 'blocks';
    for (let k = 0; k < def.length; k++) blocks.appendChild(document.createElement('i'));
    btn.append(name, blocks);
    btn.onclick = () => {
      const ship = state.placed.find((s) => s.idx === idx);
      if (ship) pickUp(ship);
      else state.selected = idx;
      paint();
    };
    tray.appendChild(btn);
  });
}

function paintStats() {
  const sunkMine = state.mine.filter((s, k) => state.shipHits[k] >= s.length).length;
  const youAfloat = state.phase === 'place' ? state.placed.length : state.mine.length - sunkMine;
  $('chip-you').textContent = `${youAfloat} / ${fleetDef().length}`;
  $('chip-enemy').textContent = state.phase === 'battle' || state.phase === 'over' ? `${fleetDef().length - state.enemySunk} / ${fleetDef().length}` : '?';
  $('chip-shots').textContent = `${state.shotsMine} / ${state.shotsAgent}`;
  $('chip-status').textContent = state.phase === 'over' ? state.endTitle : STATUS[state.phase] || '';
  $('btn-launch').disabled = !(state.phase === 'place' && fleetComplete()) || state.busy;
  $('btn-random').disabled = state.phase !== 'place' || state.busy;
  $('btn-rotate').disabled = state.phase !== 'place';
  $('btn-watch').disabled = state.phase === 'battle' || state.busy;
  $('enemy-wrap').hidden = state.phase === 'watch';
  $('bs-hint').textContent = HINT[state.phase] || '';
  paintOdds();
}

const STATUS = { place: 'PLACE YOUR FLEET', battle: 'YOUR SHOT', watch: 'AGENT HUNTING', over: 'GAME OVER' };
const HINT = {
  place: 'Pick a ship from the tray, then click your grid. Click a placed ship to pick it up. R rotates.',
  battle: 'Click enemy waters to fire. The agent answers after its reticle settles on a cell.',
  watch: 'The agent is hunting your random fleet. Its odds glow over your grid.',
  over: 'Start a new match to play again.',
};

/** The odds panel: layouts counted, the method, ships afloat, and the top cells with their probabilities. */
function paintOdds() {
  const o = state.odds;
  const top = $('top');
  top.innerHTML = '';
  if (!o) {
    $('m-count').textContent = '—';
    $('m-method').textContent = '—';
    $('m-ships').textContent = '—';
    $('an-note').textContent = '';
    return;
  }
  const method = { exact: 'exact count', sampled: 'importance sampling', done: 'no ships afloat', none: 'no consistent layout' }[o.method] || o.method;
  $('m-count').textContent = o.method === 'exact' ? o.count.toLocaleString('en-US')
    : o.method === 'sampled' ? `${o.count.toLocaleString('en-US')} samples` : '—';
  $('m-method').textContent = method;
  $('m-ships').textContent = o.remaining.length ? o.remaining.join(' · ') : 'none';
  const pmax = Math.max(1e-9, o.top[0] ? o.top[0].probability : 1e-9);
  for (const t of o.top) {
    const li = document.createElement('li');
    if (t.cell === o.choice) li.className = 'chosen';
    li.innerHTML = '<span class="mv"></span><span class="bar"><i></i></span><span class="sc"></span>';
    li.querySelector('.mv').textContent = t.cell;
    li.querySelector('.bar i').style.width = `${Math.max(4, (100 * t.probability) / pmax)}%`;
    li.querySelector('.sc').textContent = `${Math.round(100 * t.probability)}%`;
    top.appendChild(li);
  }
  $('an-title').textContent = state.phase === 'battle' || state.phase === 'watch'
    ? `Next shot: ${o.choice}, a ${Math.round(100 * o.probability)}% chance of a ship`
    : 'Its odds of your ships appear once the battle starts.';
  $('an-note').textContent = 'Odds are the share of consistent fleet layouts that cover each cell. Squares are numbered 1–10 down and A–J across.';
  $('agent-desc').textContent = state.meta ? state.meta.agent.description : '';
}

function paint() {
  paintTray();
  paintMine();
  paintEnemy();
  paintStats();
}

// ── Log and effects ──────────────────────────────────────────────────────

function log(text, cls = 'log-info') {
  const line = document.createElement('div');
  line.className = cls;
  line.textContent = text;
  $('log').appendChild(line);
  $('log').scrollTop = $('log').scrollHeight;
}

// ── Match flow ───────────────────────────────────────────────────────────

function allMineSunk() {
  return state.mine.length > 0 && state.mine.every((s, k) => state.shipHits[k] >= s.length);
}

/** Resolve one agent shot against your fleet, locally. Returns the report for /agent-shot and what happened. */
function resolveAgentShot(i) {
  const name = cellName(i);
  const k = state.mine.findIndex((s) => s.cells.includes(i));
  if (k < 0) {
    state.mineStatus[i] = 'miss';
    return { report: { cell: name, result: 'miss' }, kind: 'miss' };
  }
  const ship = state.mine[k];
  state.shipHits[k] += 1;
  if (state.shipHits[k] < ship.length) {
    state.mineStatus[i] = 'hit';
    return { report: { cell: name, result: 'hit' }, kind: 'hit', ship };
  }
  for (const c of ship.cells) state.mineStatus[c] = 'sunk';
  return { report: { cell: name, result: 'sunk', ship: { cells: ship.cells.map(cellName) } }, kind: 'sunk', ship };
}

/** Ask the server for the agent's odds, given every shot it has taken so far. */
async function refreshOdds() {
  state.odds = await postJSON('/api/battleship/agent-shot', { shots: state.reports });
}

/** One agent turn: aim, fire at your fleet, show it, then refresh the odds. Returns false if the match ended. */
async function agentTurn(token) {
  const cell = state.odds.choice_index;
  state.aim = cell;
  paintMine();
  await sleep(state.phase === 'watch' ? PAUSE[state.speed] : AIM_MS);
  if (token !== state.token) return false;
  state.aim = null;
  const r = resolveAgentShot(cell);
  state.shotsAgent += 1;
  state.reports.push(r.report);
  const board = $('me');
  const target = boards.me[cell];
  if (r.kind === 'miss') {
    log(`The agent fires at ${cellName(cell)}: miss.`, 'log-info');
  } else if (r.kind === 'hit') {
    log(`The agent hits your ${r.ship.name} at ${cellName(cell)}.`, 'log-adv');
    burst(target, { count: 36, colors: ['#ff00a0', '#ffe600', '#9b00ff'] });
    shake(board, 'small');
    pop(target, 'HIT', '#ff00a0');
  } else {
    log(`The agent sank your ${r.ship.name}!`, 'log-adv');
    burst(target, { count: 90, colors: ['#ff3b3b', '#ff00a0', '#ffe600'] });
    shake(board, 'big');
    banner(`${r.ship.name.toUpperCase()} SUNK`, `${state.mine.filter((s, k) => state.shipHits[k] < s.length).length} of yours afloat`, '#ff3b3b');
  }
  paintMine();
  paintStats();
  if (allMineSunk()) {
    endMatch(false, state.phase === 'watch' ? 'FLEET SUNK' : 'DEFEAT', `the agent sank your fleet in ${state.shotsAgent} shots`);
    return false;
  }
  await refreshOdds();
  paint();
  return true;
}

/** End the match: banner, celebration or sting, and the agent's fleet revealed when the player wins. */
function endMatch(won, title, sub) {
  state.phase = 'over';
  state.aim = null;
  state.endTitle = title;
  const color = won ? '#00ff88' : '#ff3b3b';
  const board = $(won ? 'enemy' : 'me');
  burst(board, { count: 150, colors: won ? undefined : ['#ff3b3b', '#ff00a0'] });
  shake(board, 'big');
  banner(title, sub, color);
  log(`★ ${title}: ${sub}`, 'log-best');
  paint();
}

/** Player fires at the agent's fleet. The agent answers after its reticle has had time to settle. */
async function fire(i) {
  if (state.phase !== 'battle' || state.busy || state.enemyStatus[i]) return;
  const token = state.token;
  state.busy = true;
  paint();
  try {
    const res = await postJSON('/api/battleship/shot', { id: state.gameId, cell: cellName(i) });
    if (token !== state.token) return;
    state.shotsMine += 1;
    const target = boards.enemy[i];
    if (res.result === 'miss') {
      state.enemyStatus[i] = 'miss';
      log(`You fire at ${res.cell}: miss.`, 'log-info');
    } else if (res.result === 'hit') {
      state.enemyStatus[i] = 'hit';
      log(`You hit at ${res.cell}!`, 'log-move');
      burst(target, { count: 40, colors: ['#00f5ff', '#ffe600', '#ff00a0'] });
      shake($('enemy'), 'small');
      pop(target, 'HIT', '#00f5ff');
    } else {
      state.enemySunk += 1;
      for (const n of res.ship.cells) state.enemyStatus[cellIndex(n)] = 'sunk';
      log(`★ You sank their ${res.ship.name}!`, 'log-best');
      burst(target, { count: 100, colors: ['#ffe600', '#00f5ff', '#ff00a0'] });
      shake($('enemy'), 'big');
      banner(`${res.ship.name.toUpperCase()} SUNK`, `${fleetDef().length - state.enemySunk} of theirs afloat`, '#ffe600');
    }
    paint();
    if (res.won) {
      state.enemyFleet = res.fleet;
      endMatch(true, 'VICTORY', `their fleet is gone after ${state.shotsMine} shots`);
      return;
    }
    if (!(await agentTurn(token))) return;
  } catch (err) {
    log(`✗ ${err.message}`, 'log-err');
  } finally {
    if (token === state.token) {
      state.busy = false;
      paint();
    }
  }
}

/** Start a battle: the server makes the agent's fleet, and the odds for its first shot are fetched. */
async function launch() {
  if (state.phase !== 'place' || !fleetComplete() || state.busy) return;
  const token = ++state.token;
  state.mine = [...state.placed].sort((a, b) => a.idx - b.idx);
  state.phase = 'battle';
  state.mineStatus = Array(SIZE * SIZE).fill(null);
  state.enemyStatus = Array(SIZE * SIZE).fill(null);
  state.shipHits = state.mine.map(() => 0);
  state.reports = [];
  state.hover = null;
  state.busy = true;
  paint();
  try {
    const game = await postJSON('/api/battleship/new', {});
    if (token !== state.token) return;
    state.gameId = game.id;
    await refreshOdds();
    log('Battle begins. The agent has its own fleet hidden on the server.', 'log-best');
  } catch (err) {
    log(`✗ ${err.message}`, 'log-err');
    state.phase = 'place';
  } finally {
    if (token === state.token) {
      state.busy = false;
      paint();
    }
  }
}

/** Watch mode: the agent hunts a fleet that you did not have to place (a random one if none is set). */
async function watch() {
  if (state.phase === 'battle' || state.busy) return;
  // Watching after a finished match hunts a fresh random fleet; otherwise it uses the fleet you placed.
  if (state.phase === 'over' || !fleetComplete()) state.placed = randomFleet();
  const token = ++state.token;
  state.mine = [...state.placed].sort((a, b) => a.idx - b.idx);
  state.phase = 'watch';
  state.mineStatus = Array(SIZE * SIZE).fill(null);
  state.shipHits = state.mine.map(() => 0);
  state.reports = [];
  state.hover = null;
  state.enemyFleet = null;
  state.busy = true;
  paint();
  try {
    await refreshOdds();
    log('Watching the agent hunt a fleet.', 'log-move');
  } catch (err) {
    log(`✗ ${err.message}`, 'log-err');
    state.phase = 'place';
    return;
  } finally {
    if (token === state.token) {
      state.busy = false;
      paint();
    }
  }
  try {
    while (token === state.token && state.phase === 'watch') {
      if (!(await agentTurn(token))) return;
    }
  } catch (err) {
    // A failed request (for example the rate limit) stops the watch rather than leaving it half-running.
    if (token === state.token) {
      log(`✗ ${err.message}`, 'log-err');
      endMatch(false, 'STOPPED', 'the watch stopped on an error; start again to continue');
    }
  }
}

function newMatch() {
  state.token++;
  Object.assign(state, {
    phase: 'place', placed: [], mine: [], hover: null, aim: null, odds: null, gameId: null, enemyFleet: null,
    reports: [], shipHits: [], enemySunk: 0, shotsMine: 0, shotsAgent: 0, busy: false, selected: 0,
    mineStatus: Array(SIZE * SIZE).fill(null), enemyStatus: Array(SIZE * SIZE).fill(null),
  });
  $('log').innerHTML = '';
  $('an-title').textContent = 'Its odds of your ships appear once the battle starts.';
  paint();
}

// ── Input ────────────────────────────────────────────────────────────────

function onMineClick(i) {
  if (state.phase !== 'place') return;
  const owner = state.placed.find((s) => s.cells.includes(i));
  if (owner) {
    pickUp(owner);
  } else {
    const p = previewAt(i);
    if (!p || !p.ok) return;
    const def = fleetDef()[state.selected];
    state.placed.push({ idx: state.selected, name: def.name, length: def.length, cells: p.cells });
    state.selected = nextUnplaced();
    state.hover = null;
  }
  paint();
}

function wire() {
  buildBoard('me', 'me');
  buildBoard('enemy', 'enemy');

  $('me').addEventListener('click', (e) => {
    const cell = e.target.closest('.bs-cell');
    if (cell) onMineClick(Number(cell.dataset.i));
  });
  $('me').addEventListener('mouseover', (e) => {
    const cell = e.target.closest('.bs-cell');
    if (!cell || state.phase !== 'place') return;
    state.hover = previewAt(Number(cell.dataset.i));
    paintMine();
  });
  $('me').addEventListener('mouseleave', () => {
    state.hover = null;
    paintMine();
  });
  $('enemy').addEventListener('click', (e) => {
    const cell = e.target.closest('.bs-cell');
    if (cell) fire(Number(cell.dataset.i));
  });

  $('btn-launch').onclick = launch;
  $('btn-random').onclick = () => {
    if (state.phase !== 'place') return;
    state.placed = randomFleet();
    state.selected = -1;
    paint();
  };
  $('btn-rotate').onclick = () => {
    state.vertical = !state.vertical;
    paint();
  };
  $('btn-watch').onclick = watch;
  $('btn-new').onclick = newMatch;
  $('heat').onchange = (e) => {
    state.heat = e.target.checked;
    paintMine();
  };
  $('speed').onchange = (e) => {
    state.speed = e.target.value;
  };
  addEventListener('keydown', (e) => {
    if ((e.key === 'r' || e.key === 'R') && state.phase === 'place') {
      state.vertical = !state.vertical;
      paint();
    }
  });
}

async function init() {
  wire();
  try {
    state.meta = await getJSON('/api/battleship/meta');
  } catch (err) {
    log(`✗ Server unreachable (${err.message}).`, 'log-err');
    return;
  }
  state.selected = 0;
  newMatch();
}

init();
