// Battleship page.
//
// Two fleets, two owners, and two shooters per match. The opponent is either Probability (Bayesian, the server's
// counting agent) or Chance (fixed odds, a weighted draw from a table the server publishes). Both run on the server
// through POST /agent-shot, which takes the shots a side has fired and returns its next cell with the odds behind it.
//
// Battle (you vs the opponent): your fleet lives on this page. The opponent's shots are resolved here, and the page
// sends their history to the server for fresh odds. Your shots go to a fleet on the server (POST /new, POST /shot),
// which is revealed only when you win.
//
// Watch (AI vs the opponent): the AI fires at a fleet on the server, the opponent fires at a fleet on this page, and
// the shots alternate until one side's fleet is gone. The score counts the AI's wins and losses across races.
//
// Each shot is: aim (the reticle sits on the cell the server picked), fire (resolved here or on the server), then
// ask for fresh odds from the updated history.

import { getJSON, postJSON } from './api.js';
import { banner, burst, pop, shake } from './fx.js';
import { SIZE, randomFleet as drawFleet, shipCells } from './battleship-core.js';

const $ = (id) => document.getElementById(id);
const COLS = 'ABCDEFGHIJ';
const AIM_MS = 650; // reticle time before each shot in battle
// Time between shots in a watched race. An AI shot costs two requests (the shot and the AI's next odds) and an
// opponent shot one, so a shot averages 1.5 requests. The server allows 120 moves a minute per client, so the fast
// setting (800 ms) stays under that at about 112 requests a minute.
const PAUSE = { slow: 1500, normal: 1000, fast: 800 };
const RESTART_MS = 1800; // pause on the result before an automatic restart
const LABEL = { probability: 'Probability (Bayesian)', chance: 'Chance (fixed odds)' };

const cellName = (i) => `${COLS[i % SIZE]}${Math.floor(i / SIZE) + 1}`;
const cellIndex = (name) => COLS.indexOf(name[0]) + (parseInt(name.slice(1), 10) - 1) * SIZE;
const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
const blank = () => Array(SIZE * SIZE).fill(null);
const randomSeed = () => Math.floor(Math.random() * 2 ** 31);

const state = {
  meta: null,
  phase: 'place', // place | battle | watch | over
  vertical: false, // orientation for the next ship placed
  selected: -1, // index into the fleet of the ship being placed
  placed: [], // ships on your grid while placing: {idx, name, length, cells}
  hover: null, // placement preview under the pointer: {cells, ok}
  mine: [], // the fleet being shot at by the opponent once a match starts: {idx, name, length, cells}
  mineStatus: blank(), // per cell of that fleet: null | miss | hit | sunk
  shipHits: [], // hits taken per ship of that fleet
  reports: [], // the opponent's shots at that fleet, in order, in the form /agent-shot expects
  odds: null, // last /agent-shot response for the opponent
  aim: null, // cell the reticle is on, while the opponent aims at your fleet
  gameId: null,
  enemyStatus: blank(), // per cell of the server's fleet: null | miss | hit | sunk
  enemySunk: 0,
  enemyFleet: null, // the server's fleet, revealed when it is sunk: [{name, cells: [names]}]
  aiReports: [], // the AI's shots at the server's fleet, in the form /agent-shot expects (watch only)
  aiOdds: null, // last /agent-shot response for the AI (watch only)
  opponent: 'probability', // who fires at your fleet: probability | chance
  seed: 0, // chance's seed for this match or race
  shotsMine: 0, // shots fired by you (battle) or by the AI (watch)
  shotsAgent: 0, // shots fired by the opponent
  heat: true,
  busy: false,
  speed: 'normal',
  watching: false, // a race session is running, so races restart when they finish if auto-restart is on
  solo: false, // watch mode is "AI alone": the AI hunts your fleet with nobody shooting back
  paused: false,
  stepOnce: false, // one shot is allowed through while paused
  turn: 'ai', // whose shot comes next in a race
  aiFirst: false, // alternates per race, so neither side always fires first
  score: { ai: 0, opp: 0 }, // races the AI won and lost in this watch session
  token: 0, // bumped by NEW MATCH and by each new race so that loops still running from an old one stop
  endTitle: '', // banner text of the finished match, kept on the status chip
};

const boards = { me: [], enemy: [] };

// ── Geometry and placement ───────────────────────────────────────────────

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

/** Random legal fleet for the page's ships, drawn uniformly over legal fleets (see randomFleet in battleship-core.js). */
function randomFleet() {
  const def = fleetDef();
  return drawFleet(def.map((d) => d.length)).map((cells, idx) => ({ idx, name: def[idx].name, length: def[idx].length, cells }));
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

/** Name of the opponent as shown to the player, e.g. "Chance (fixed odds)". */
function oppLabel() {
  return LABEL[state.opponent];
}

/** Short name used in log lines and banners, e.g. "Chance". */
function oppName() {
  return state.opponent === 'chance' ? 'Chance' : 'Probability';
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
 * Glow for one side's odds over a board. Early on the odds are nearly flat, so the wash is measured from the mean:
 * cells above average glow, the hottest brightest, and cells at or below average stay clear. Known cells never glow.
 */
function heatAlphas(status, odds) {
  const out = Array(SIZE * SIZE).fill(0);
  if (!state.heat || !odds) return out;
  const unknown = [];
  for (let i = 0; i < SIZE * SIZE; i++) if (!status[i]) unknown.push(i);
  if (!unknown.length) return out;
  const p = odds.probs;
  const mean = unknown.reduce((sum, i) => sum + p[i], 0) / unknown.length;
  const top = Math.max(...unknown.map((i) => p[i]));
  const span = Math.max(1e-9, top - mean);
  for (const i of unknown) out[i] = 0.8 * Math.min(1, Math.max(0, (p[i] - mean) / span));
  return out;
}

function paintMine() {
  const ships = state.phase === 'place' ? state.placed : state.mine;
  const hull = new Set(ships.flatMap((s) => s.cells));
  const alphas = heatAlphas(state.mineStatus, state.phase === 'place' ? null : state.odds);
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
  const alphas = heatAlphas(state.enemyStatus, state.phase === 'watch' ? state.aiOdds : null);
  for (let i = 0; i < SIZE * SIZE; i++) {
    const cls = ['bs-cell'];
    const st = state.enemyStatus[i];
    if (st) cls.push(st);
    else if (reveal.has(i)) cls.push('ship');
    boards.enemy[i].className = cls.join(' ');
    const a = alphas[i];
    boards.enemy[i].firstChild.style.background = a > 0.005 ? `rgba(0,245,255,${a.toFixed(3)})` : '';
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

const STATUS = { place: 'PLACE YOUR FLEET', battle: 'YOUR SHOT', watch: 'RACE', over: 'GAME OVER' };

/** The live status line: names which algorithm each side uses, so the page says who is shooting how. */
function hintText() {
  if (state.phase === 'place') return 'Pick a ship from the tray, then click your grid. Click a placed ship to pick it up. R rotates.';
  if (state.phase === 'battle') return `You fire at enemy waters. ${oppLabel()} answers from its own odds once its reticle settles.`;
  if (state.phase === 'watch') {
    const paused = state.paused ? ' Paused: step for one shot or resume.' : '';
    if (state.solo) return `AI alone: ${LABEL.probability} hunts your fleet, and nothing shoots back. Its odds glow over your grid.${paused}`;
    const tally = state.score.ai + state.score.opp > 0 ? ` Races: AI ${state.score.ai}, ${oppName()} ${state.score.opp}.` : '';
    return `AI: ${LABEL.probability} fires at the server's fleet. ${oppLabel()} fires at the fleet on this page. Shots alternate.${tally}${paused}`;
  }
  return 'Start a new match to play again.';
}

function paintStats() {
  // A race keeps the watch layout until NEW MATCH, so the labels stay on the AI and the opponent between races.
  const watching = state.watching;
  const sunkMine = state.mine.filter((s, k) => state.shipHits[k] >= s.length).length;
  const youAfloat = state.phase === 'place' ? state.placed.length : state.mine.length - sunkMine;
  $('lbl-you').textContent = watching ? `${oppName().toUpperCase()}'S FLEET` : 'YOUR FLEET';
  $('lbl-enemy').textContent = watching ? 'AI FLEET' : 'ENEMY';
  $('lbl-shots').textContent = watching ? 'AI / OPP' : 'SHOTS';
  $('chip-you').textContent = `${youAfloat} / ${fleetDef().length}`;
  $('chip-enemy').textContent = state.phase === 'battle' || state.phase === 'over' || watching
    ? `${fleetDef().length - state.enemySunk} / ${fleetDef().length}` : '?';
  $('chip-shots').textContent = `${state.shotsMine} / ${state.shotsAgent}`;
  $('chip-status').textContent = state.phase === 'over' ? state.endTitle
    : state.phase === 'watch' ? (state.paused ? 'PAUSED' : state.solo ? 'WATCHING' : STATUS.watch) : STATUS[state.phase] || '';
  $('chance-block').hidden = $('opponent').value !== 'chance';
  $('btn-launch').disabled = !(state.phase === 'place' && fleetComplete()) || state.busy;
  $('btn-random').disabled = state.phase !== 'place' || state.busy;
  $('btn-rotate').disabled = state.phase !== 'place';
  $('btn-watch').disabled = state.phase === 'battle' || state.phase === 'watch' || state.busy;
  $('btn-pause').hidden = state.phase !== 'watch';
  $('btn-step').hidden = state.phase !== 'watch';
  $('enemy-wrap').hidden = state.phase === 'watch' && state.solo;
  $('watch-mode').disabled = state.phase === 'battle' || state.phase === 'watch' || state.busy;
  $('btn-pause').querySelector('.btn-txt').textContent = state.paused ? '▶ RESUME' : '❚❚ PAUSE';
  $('opponent').disabled = state.phase === 'battle' || state.phase === 'watch' || state.busy;
  $('me-title').textContent = watching ? `${oppName().toUpperCase()}'S TARGET` : 'YOUR FLEET';
  $('enemy-title').textContent = watching ? "AI'S TARGET" : 'ENEMY WATERS';
  $('bs-hint').textContent = hintText();
  paintOdds();
}

/** The odds panel: layouts counted, the method, ships afloat, and the top cells. Shows the AI in a race. */
function paintOdds() {
  const watching = state.watching;
  const o = watching ? state.aiOdds : state.odds;
  const top = $('top');
  top.innerHTML = '';
  $('an-title').textContent = 'Its odds appear once the battle starts.';
  if (!o) {
    $('m-count').textContent = '—';
    $('m-method').textContent = '—';
    $('m-ships').textContent = '—';
    $('an-note').textContent = '';
    return;
  }
  const fixed = o.method === 'fixed odds';
  const method = { exact: 'exact count', sampled: 'importance sampling', done: 'no ships afloat', none: 'no consistent layout', 'fixed odds': 'fixed table' }[o.method] || o.method;
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
    li.querySelector('.sc').textContent = `${(100 * t.probability).toFixed(1)}%`;
    top.appendChild(li);
  }
  const who = watching ? 'The AI' : oppName();
  const share = fixed ? `${(100 * o.probability).toFixed(1)}% of its fixed table` : `a ${Math.round(100 * o.probability)}% chance of a ship`;
  $('an-title').textContent = `${who} next fires at ${o.choice}: ${share}`;
  $('an-note').textContent = fixed
    ? 'Chance ignores its hits and misses: its odds are the fixed table renormalised over the cells it has not fired at.'
    : 'Odds are the share of consistent fleet layouts that cover each cell. Squares are numbered 1–10 down and A–J across.';
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

/** Resolve one shot against the fleet on this page, locally. Returns the report for /agent-shot and what happened. */
function resolveOppShot(i) {
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

/** Ask the server for one side's odds, given every shot that side has taken so far. */
function askOdds(agent, shots, seed) {
  return postJSON('/api/battleship/agent-shot', { shots, agent, seed });
}

/** The opponent fires at the fleet on this page: aim (battle only), fire, show it, then refresh its odds. */
async function oppShot(token) {
  const cell = state.odds.choice_index;
  if (state.phase === 'battle') {
    state.aim = cell;
    paintMine();
    await sleep(AIM_MS);
    if (token !== state.token) return false;
  }
  state.aim = null;
  const r = resolveOppShot(cell);
  state.shotsAgent += 1;
  state.reports.push(r.report);
  const target = boards.me[cell];
  if (r.kind === 'miss') {
    log(`${oppName()} fires at ${cellName(cell)}: miss.`, 'log-info');
  } else if (r.kind === 'hit') {
    log(`${oppName()} hits your ${r.ship.name} at ${cellName(cell)}.`, 'log-adv');
    burst(target, { count: 36, colors: ['#ff00a0', '#ffe600', '#9b00ff'] });
    shake($('me'), 'small');
    pop(target, 'HIT', '#ff00a0');
  } else {
    log(`${oppName()} sank your ${r.ship.name}!`, 'log-adv');
    burst(target, { count: 90, colors: ['#ff3b3b', '#ff00a0', '#ffe600'] });
    shake($('me'), 'big');
    banner(`${r.ship.name.toUpperCase()} SUNK`, `${state.mine.filter((s, k) => state.shipHits[k] < s.length).length} of yours afloat`, '#ff3b3b');
  }
  paintMine();
  paintStats();
  if (allMineSunk()) {
    if (state.solo) endMatch(false, 'FLEET SUNK', `${oppName()} sank your fleet in ${state.shotsAgent} shots`);
    else if (state.phase === 'watch') finishRace('opp');
    else endMatch(false, 'DEFEAT', `${oppName()} sank your fleet in ${state.shotsAgent} shots`);
    return false;
  }
  state.odds = await askOdds(state.opponent, state.reports, state.seed);
  paint();
  return true;
}

/** The AI fires at the server's fleet: the server answers, then the AI's odds are refreshed from its history. */
async function aiShot(token) {
  const cell = state.aiOdds.choice_index;
  const res = await postJSON('/api/battleship/shot', { id: state.gameId, cell: cellName(cell) });
  if (token !== state.token) return false;
  state.shotsMine += 1;
  const target = boards.enemy[cell];
  if (res.result === 'miss') {
    state.enemyStatus[cell] = 'miss';
    log(`AI fires at ${res.cell}: miss.`, 'log-info');
  } else if (res.result === 'hit') {
    state.enemyStatus[cell] = 'hit';
    log(`AI hits at ${res.cell}.`, 'log-move');
    burst(target, { count: 40, colors: ['#00f5ff', '#ffe600', '#ff00a0'] });
    shake($('enemy'), 'small');
    pop(target, 'HIT', '#00f5ff');
  } else {
    state.enemySunk += 1;
    for (const n of res.ship.cells) state.enemyStatus[cellIndex(n)] = 'sunk';
    log(`★ AI sank the ${res.ship.name}!`, 'log-best');
    burst(target, { count: 100, colors: ['#ffe600', '#00f5ff', '#ff00a0'] });
    shake($('enemy'), 'big');
    banner(`${res.ship.name.toUpperCase()} SUNK`, `${fleetDef().length - state.enemySunk} of theirs afloat`, '#ffe600');
  }
  state.aiReports.push(res.result === 'sunk'
    ? { cell: res.cell, result: 'sunk', ship: { cells: res.ship.cells } }
    : { cell: res.cell, result: res.result });
  paint();
  if (res.won) {
    state.enemyFleet = res.fleet;
    finishRace('ai');
    return false;
  }
  state.aiOdds = await askOdds('probability', state.aiReports, 0);
  paint();
  return true;
}

/** End the match: banner, celebration or sting, and the server's fleet revealed when the AI or you sink it. */
function endMatch(won, title, sub) {
  state.phase = 'over';
  state.aim = null;
  state.paused = false;
  state.endTitle = title;
  const color = won ? '#00ff88' : '#ff3b3b';
  const board = $(won ? 'enemy' : 'me');
  burst(board, { count: 150, colors: won ? undefined : ['#ff3b3b', '#ff00a0'] });
  shake(board, 'big');
  banner(title, sub, color);
  log(`★ ${title}: ${sub}`, 'log-best');
  paint();
}

/** A race is over. Score it, then restart with a fresh random fleet on each side if auto-restart is on. */
function finishRace(winner) {
  if (winner === 'ai') {
    state.score.ai += 1;
    endMatch(true, 'AI WINS', `the AI sank the ${oppName()} fleet in ${state.shotsMine} shots`);
  } else {
    state.score.opp += 1;
    endMatch(false, `${oppName().toUpperCase()} WINS`, `${oppName()} sank your fleet in ${state.shotsAgent} shots`);
  }
  if ($('auto').checked) setTimeout(nextRace, RESTART_MS);
}

function nextRace() {
  if (!state.watching || state.phase !== 'over' || !$('auto').checked) return;
  state.placed = randomFleet();
  startRace();
}

/** Player fires at the server's fleet. The opponent answers after its reticle has had time to settle. */
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
    if (!(await oppShot(token))) return;
  } catch (err) {
    log(`✗ ${err.message}`, 'log-err');
  } finally {
    if (token === state.token) {
      state.busy = false;
      paint();
    }
  }
}

/** Start a battle: the server makes its fleet, and the odds for the opponent's first shot are fetched. */
async function launch() {
  if (state.phase !== 'place' || !fleetComplete() || state.busy) return;
  const token = ++state.token;
  state.opponent = $('opponent').value;
  state.seed = randomSeed();
  state.mine = [...state.placed].sort((a, b) => a.idx - b.idx);
  state.phase = 'battle';
  state.mineStatus = blank();
  state.enemyStatus = blank();
  state.shipHits = state.mine.map(() => 0);
  state.reports = [];
  state.hover = null;
  state.busy = true;
  paint();
  try {
    const game = await postJSON('/api/battleship/new', {});
    if (token !== state.token) return;
    state.gameId = game.id;
    state.odds = await askOdds(state.opponent, state.reports, state.seed);
    log(`Battle begins. ${oppLabel()} fires at your fleet; its fleet stays hidden on the server.`, 'log-best');
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

/**
 * Watch, in one of two modes. "AI alone" lets the AI hunt your fleet with nothing firing back (the page's first
 * watch). "AI vs opponent race" also has the opponent fire at your fleet while the AI fires at the server's fleet.
 * Either mode starts with the fleet you placed, or a random one after a finished match or race.
 */
async function watch() {
  if (state.phase === 'battle' || state.phase === 'watch' || state.busy) return;
  if (state.phase === 'over' || !fleetComplete()) state.placed = randomFleet();
  state.paused = false;
  state.solo = $('watch-mode').value === 'solo';
  if (state.solo) {
    state.opponent = 'probability';
    state.watching = false;
    await startSolo();
    return;
  }
  state.opponent = $('opponent').value;
  state.watching = true;
  await startRace();
}

/** AI alone: the AI's odds over your fleet, then one AI shot at a time until your fleet is gone. No server fleet. */
async function startSolo() {
  const token = ++state.token;
  state.mine = [...state.placed].sort((a, b) => a.idx - b.idx);
  state.phase = 'watch';
  state.mineStatus = blank();
  state.enemyStatus = blank();
  state.shipHits = state.mine.map(() => 0);
  state.reports = [];
  state.aiReports = [];
  state.hover = null;
  state.enemyFleet = null;
  state.enemySunk = 0;
  state.shotsMine = 0;
  state.shotsAgent = 0;
  state.endTitle = '';
  state.odds = null;
  state.aiOdds = null;
  state.busy = true;
  paint();
  try {
    state.odds = await askOdds('probability', state.reports, 0);
    if (token !== state.token) return;
    log('Watching the AI hunt a fleet.', 'log-move');
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
    await raceLoop(token);
  } catch (err) {
    if (token === state.token) {
      log(`✗ ${err.message}`, 'log-err');
      endMatch(false, 'STOPPED', 'the watch stopped on an error; start again to continue');
    }
  }
}

/** Set up one race: a fresh server fleet for the AI, the opponent's odds, then the shots alternate until one fleet is gone. */
async function startRace() {
  const token = ++state.token;
  state.mine = [...state.placed].sort((a, b) => a.idx - b.idx);
  state.phase = 'watch';
  state.mineStatus = blank();
  state.enemyStatus = blank();
  state.shipHits = state.mine.map(() => 0);
  state.reports = [];
  state.aiReports = [];
  state.hover = null;
  state.enemyFleet = null;
  state.enemySunk = 0;
  state.shotsMine = 0;
  state.shotsAgent = 0;
  state.endTitle = '';
  state.aiFirst = !state.aiFirst;
  state.turn = state.aiFirst ? 'ai' : 'opp';
  state.seed = randomSeed();
  state.busy = true;
  paint();
  try {
    const game = await postJSON('/api/battleship/new', {});
    if (token !== state.token) return;
    state.gameId = game.id;
    state.aiOdds = await askOdds('probability', state.aiReports, 0);
    if (token !== state.token) return;
    state.odds = await askOdds(state.opponent, state.reports, state.seed);
    log(`Race begins: AI (${LABEL.probability}) against ${oppLabel()}.`, 'log-best');
  } catch (err) {
    log(`✗ ${err.message}`, 'log-err');
    state.phase = 'place';
    state.watching = false;
    return;
  } finally {
    if (token === state.token) {
      state.busy = false;
      paint();
    }
  }
  try {
    await raceLoop(token);
  } catch (err) {
    // A failed request (for example the rate limit) stops the watch rather than leaving it half-running.
    if (token === state.token) {
      state.watching = false;
      log(`✗ ${err.message}`, 'log-err');
      endMatch(false, 'STOPPED', 'the watch stopped on an error; start again to continue');
    }
  }
}

/**
 * Fire one shot at a time, alternating sides. Pause holds the race; step lets exactly one shot through.
 * Between shots the loop waits for the chosen speed, so the race is readable and stays under the request limit.
 */
async function raceLoop(token) {
  while (token === state.token && state.phase === 'watch') {
    if (state.paused && !state.stepOnce) {
      await sleep(150);
      continue;
    }
    const stepping = state.stepOnce;
    state.stepOnce = false;
    // Solo watch has only the opponent slot firing (the AI hunting your fleet), so every turn is that side's.
    const side = state.solo ? 'opp' : state.turn;
    if (!state.solo) state.turn = side === 'ai' ? 'opp' : 'ai';
    const ok = side === 'ai' ? await aiShot(token) : await oppShot(token);
    if (!ok) return;
    if (!stepping && token === state.token && !state.paused) await sleep(PAUSE[state.speed]);
  }
}

function newMatch() {
  state.token++;
  Object.assign(state, {
    phase: 'place', placed: [], mine: [], hover: null, aim: null, odds: null, aiOdds: null,
    gameId: null, enemyFleet: null, reports: [], aiReports: [], shipHits: [], enemySunk: 0, shotsMine: 0,
    shotsAgent: 0, busy: false, selected: 0, watching: false, solo: false, paused: false, stepOnce: false,
    score: { ai: 0, opp: 0 }, endTitle: '', mineStatus: blank(), enemyStatus: blank(),
  });
  $('log').innerHTML = '';
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

/** The chance table: the cell weights as a 10 x 10 grid, shaded so the middle of the board reads as hotter. */
function paintChanceTable() {
  const c = state.meta.chance;
  const grid = $('chance-table');
  grid.innerHTML = '';
  const weight = (i) => c.profile[Math.floor(i / SIZE)] * c.profile[i % SIZE];
  const max = Math.max(...Array.from({ length: SIZE * SIZE }, (_, i) => weight(i)));
  for (let i = 0; i < SIZE * SIZE; i++) {
    const span = document.createElement('span');
    span.textContent = weight(i);
    span.title = `${cellName(i)}: weight ${weight(i)}`;
    span.style.background = `rgba(0,245,255,${(0.08 + 0.5 * (weight(i) / max)).toFixed(3)})`;
    grid.appendChild(span);
  }
  grid.title = `profile = ${c.profile.join(' ')}. ${c.rule}.`;
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
  $('btn-pause').onclick = () => {
    state.paused = !state.paused;
    paint();
  };
  $('btn-step').onclick = () => {
    state.paused = true;
    state.stepOnce = true;
    paint();
  };
  $('btn-new').onclick = newMatch;
  $('heat').onchange = (e) => {
    state.heat = e.target.checked;
    paint();
  };
  $('speed').onchange = (e) => {
    state.speed = e.target.value;
  };
  $('opponent').onchange = paint;
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
  paintChanceTable();
  state.selected = 0;
  newMatch();
}

init();
