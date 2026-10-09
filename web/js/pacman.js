// Pac-Man page. The server owns the rules and the agents: this file draws the maze, moves
// the sprites from the position before each turn to the position after it, and shows what
// the Q-agent saw (its value for every move, the features behind it, and the learned weights).
//
// Two modes share one renderer:
//   watch  POST /api/pacman/episode returns the whole game; playback steps through its turns.
//   play   POST /api/pacman/play replays the moves so far (the server keeps no session), and
//          returns the new state, the legal moves next, and the turn just played.
// The GHOSTS menu sends "ghosts": "ai" (A* routes) or "chance" (fixed odds) with every request.
// The server draws the chance moves from the game's seed, so this file holds no chance logic.

import { getJSON, postJSON } from './api.js';
import { banner, burst, pop, shake } from './fx.js';

const $ = (id) => document.getElementById(id);
const CELL = 36; // canvas pixels per maze cell
const SCARED = '#3d5cff';
const GHOST_COLOR = { chaser: '#ff3b6b', ambusher: '#00f5ff', scatter: '#ffe600' };
const ANGLE = { N: -Math.PI / 2, E: 0, S: Math.PI / 2, W: Math.PI }; // canvas y grows downward
const STEP = { N: [-1, 0], E: [0, 1], S: [1, 0], W: [0, -1] };
const LETTERS = ['N', 'E', 'S', 'W'];
const KEY_MOVES = {
  w: 'N', arrowup: 'N', s: 'S', arrowdown: 'S', a: 'W', arrowleft: 'W', d: 'E', arrowright: 'E',
};
const DEATH_MS = 700; // how long Pac-Man takes to shrink away
const ODDS_NAMES = { straight: 'Keep going', left: 'Turn left', right: 'Turn right', back: 'Reverse' };
const POLICY_CHIP = { ai: 'AI', chance: 'CHANCE' };
const NEXT_GAME_MS = 900; // pause between games when auto-restart is on, so the last board can be read
const HINT_PLAY = 'Use W A S D or the arrow keys to move. Each key is one turn. Ghosts move after you.';
const HINT_WATCH = 'Press Run to watch the agent play the seeded game. Pause and step to read its values turn by turn.';

const canvas = $('pm-canvas');
const ctx = canvas.getContext('2d');

// Everything on screen. `from` holds the sprite positions an animation starts from; the
// sprites glide between `from` and their current cells over `animDur` milliseconds.
const S = {
  meta: null, rows: [], busy: false,
  pac: [0, 0], ghosts: [], pellets: new Set(), plans: [],
  q: null, features: null, qAt: null, // qAt: the cell the Q-values were computed from
  lastAction: 'E', lastChosen: null, lastAgent: null,
  points: 0, turn: 0, status: 'ready',
  from: null, animStart: 0, animDur: 0, deathAt: 0,
  legal: [], actions: [],
  episode: null, index: 0, playing: false, timer: 0,
  policy: 'ai', // the ghost policy of the board on screen
  tally: { games: 0, won: 0, lost: 0, timeout: 0 }, // watch-mode results for the current settings
};

const key = (r, c) => `${r},${c}`;
const solid = (r, c) => r >= 0 && c >= 0 && r < S.rows.length && c < S.rows[0].length && S.rows[r][c] === '#';
const speedMs = () => Number($('speed').value);
const randomSeed = () => { $('seed').value = Math.floor(Math.random() * 1_000_000); };
const smooth = (t) => t * t * (3 - 2 * t);
const lerp = (a, b, k) => a + (b - a) * k;

// ── Drawing ──────────────────────────────────────────────────────────────

function sizeCanvas() {
  const w = S.rows[0].length, h = S.rows.length;
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  canvas.width = w * CELL * dpr;
  canvas.height = h * CELL * dpr;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
}

// Walls are neon outlines on the edges that border open space, so the maze reads as lines of
// light rather than solid blocks. Cells beyond the board count as solid, which keeps the frame lit.
function drawWalls() {
  ctx.save();
  ctx.strokeStyle = '#00f5ff';
  ctx.lineWidth = 2;
  ctx.shadowColor = '#00f5ff';
  ctx.shadowBlur = 10;
  ctx.beginPath();
  for (let r = 0; r < S.rows.length; r++) {
    for (let c = 0; c < S.rows[0].length; c++) {
      if (!solid(r, c)) continue;
      const x = c * CELL, y = r * CELL;
      if (!solid(r - 1, c)) { ctx.moveTo(x, y); ctx.lineTo(x + CELL, y); }
      if (!solid(r + 1, c)) { ctx.moveTo(x, y + CELL); ctx.lineTo(x + CELL, y + CELL); }
      if (!solid(r, c - 1)) { ctx.moveTo(x, y); ctx.lineTo(x, y + CELL); }
      if (!solid(r, c + 1)) { ctx.moveTo(x + CELL, y); ctx.lineTo(x + CELL, y + CELL); }
    }
  }
  ctx.stroke();
  ctx.restore();
}

function drawPellets(now) {
  ctx.save();
  for (const k of S.pellets) {
    const [r, c] = k.split(',').map(Number);
    const x = (c + 0.5) * CELL, y = (r + 0.5) * CELL;
    if (S.rows[r][c] === 'o') {
      // Power pellets pulse so they stand out from the dots.
      ctx.fillStyle = '#ff00a0';
      ctx.shadowColor = '#ff00a0';
      ctx.shadowBlur = 14;
      ctx.beginPath(); ctx.arc(x, y, 6 + 1.5 * Math.sin(now / 180), 0, Math.PI * 2); ctx.fill();
    } else {
      ctx.fillStyle = '#c0e8ff';
      ctx.shadowColor = '#c0e8ff';
      ctx.shadowBlur = 4;
      ctx.beginPath(); ctx.arc(x, y, 2.6, 0, Math.PI * 2); ctx.fill();
    }
  }
  ctx.restore();
}

// The routes the ghosts planned this turn: dashed glowing lines that march toward each target.
// A ring marks the target, so you can see where each A* search is heading.
function drawRoutes(now) {
  ctx.save();
  ctx.lineWidth = 3;
  ctx.setLineDash([6, 6]);
  ctx.lineDashOffset = -now / 40;
  ctx.globalAlpha = 0.85;
  for (const plan of S.plans) {
    if (plan.path.length < 2) continue;
    const color = GHOST_COLOR[S.ghosts[plan.ghost]?.personality] || '#ffffff';
    ctx.strokeStyle = color;
    ctx.shadowColor = color;
    ctx.shadowBlur = 12;
    ctx.beginPath();
    plan.path.forEach(([r, c], i) => {
      const x = (c + 0.5) * CELL, y = (r + 0.5) * CELL;
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    });
    ctx.stroke();
    if (plan.target) {
      ctx.setLineDash([]);
      ctx.beginPath();
      ctx.arc((plan.target[1] + 0.5) * CELL, (plan.target[0] + 0.5) * CELL, 8, 0, Math.PI * 2);
      ctx.stroke();
      ctx.setLineDash([6, 6]);
    }
  }
  ctx.restore();
}

// The Q-agent's value for each legal move, written on the cell that move leads to. The move it
// chose is green; the others shade from dim (lowest value) to bright (highest).
function drawQLabels() {
  if (!S.q || !S.qAt) return;
  const values = Object.values(S.q);
  const lo = Math.min(...values), hi = Math.max(...values);
  ctx.save();
  ctx.font = '600 10px "JetBrains Mono", monospace';
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  for (const letter of LETTERS) {
    if (!(letter in S.q)) continue;
    const [dr, dc] = STEP[letter];
    const x = (S.qAt[1] + dc + 0.5) * CELL, y = (S.qAt[0] + dr + 0.5) * CELL;
    const v = S.q[letter];
    const chosen = letter === S.lastChosen;
    const t = hi === lo ? 1 : (v - lo) / (hi - lo);
    const color = chosen ? '#00ff88' : `rgba(0, 245, 255, ${0.45 + 0.5 * t})`;
    ctx.fillStyle = 'rgba(2, 6, 16, 0.85)';
    ctx.fillRect(x - 17, y - 8, 34, 16);
    ctx.strokeStyle = color;
    ctx.lineWidth = chosen ? 2 : 1;
    ctx.strokeRect(x - 17, y - 8, 34, 16);
    ctx.fillStyle = color;
    ctx.shadowColor = color;
    ctx.shadowBlur = chosen ? 10 : 0;
    ctx.fillText(v.toFixed(1), x, y);
  }
  ctx.restore();
}

// A ghost: a rounded top, a wavy skirt, and two eyes. Scared ghosts turn blue and flicker white
// in their last two turns of fright, which is how you know the power is running out.
function drawGhost(x, y, r, color, scared, flicker) {
  const body = scared ? (flicker ? '#ffffff' : SCARED) : color;
  ctx.save();
  ctx.fillStyle = body;
  ctx.shadowColor = body;
  ctx.shadowBlur = 12;
  ctx.beginPath();
  ctx.arc(x, y - r * 0.15, r, Math.PI, 0);
  let px = x + r;
  const step = (2 * r) / 3;
  ctx.lineTo(px, y + r * 0.8);
  for (let i = 0; i < 3; i++) {
    ctx.quadraticCurveTo(px - step / 2, y + r, px - step, y + r * 0.8);
    px -= step;
  }
  ctx.closePath();
  ctx.fill();
  ctx.shadowBlur = 0;
  for (const dx of [-r * 0.35, r * 0.35]) {
    ctx.fillStyle = '#ffffff';
    ctx.beginPath(); ctx.arc(x + dx, y - r * 0.2, r * 0.24, 0, Math.PI * 2); ctx.fill();
    if (!scared) {
      ctx.fillStyle = '#020610';
      ctx.beginPath(); ctx.arc(x + dx + r * 0.08, y - r * 0.18, r * 0.12, 0, Math.PI * 2); ctx.fill();
    }
  }
  ctx.restore();
}

// Pac-Man faces the way he last moved and chomps while he moves. When caught he shrinks away.
function drawPac(x, y, r, now) {
  const dying = S.deathAt && now - S.deathAt < DEATH_MS;
  const shrink = dying ? Math.max(0, 1 - (now - S.deathAt) / DEATH_MS) : 1;
  const mouth = dying ? 0.1 + (1 - shrink) * Math.PI : 0.25 + 0.2 * Math.abs(Math.sin(now / 140));
  const angle = ANGLE[S.lastAction] ?? 0;
  ctx.save();
  ctx.fillStyle = '#ffe600';
  ctx.shadowColor = '#ffe600';
  ctx.shadowBlur = 16;
  ctx.beginPath();
  ctx.moveTo(x, y);
  ctx.arc(x, y, r * shrink, angle + mouth, angle + Math.PI * 2 - mouth);
  ctx.closePath();
  ctx.fill();
  ctx.restore();
}

// Where each sprite is right now: partway between its starting cell and its new cell.
function positions(now) {
  if (!S.from || !S.animDur) return { pac: S.pac, ghosts: S.ghosts.map((g) => g.pos) };
  const k = smooth(Math.min(1, Math.max(0, (now - S.animStart) / S.animDur)));
  const place = (a, b) => {
    // A ghost that was eaten jumps back home; gliding it across the maze would look wrong.
    const jump = Math.abs(a[0] - b[0]) + Math.abs(a[1] - b[1]) > 2;
    const kk = jump ? 1 : k;
    return [lerp(a[0], b[0], kk), lerp(a[1], b[1], kk)];
  };
  return {
    pac: place(S.from.pac, S.pac),
    ghosts: S.ghosts.map((g, i) => place(S.from.ghosts[i]?.pos ?? g.pos, g.pos)),
  };
}

function draw(now) {
  if (!S.rows.length) return;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  drawWalls();
  drawPellets(now);
  const p = positions(now);
  drawRoutes(now);
  drawQLabels();
  const center = (rc) => [(rc[1] + 0.5) * CELL, (rc[0] + 0.5) * CELL];
  const flicker = S.ghosts.some((g) => g.scared && g.scared <= 2) && Math.floor(now / 150) % 2 === 0;
  drawPac(...center(p.pac), CELL * 0.42, now);
  S.ghosts.forEach((g, i) => {
    drawGhost(...center(p.ghosts[i]), CELL * 0.38, GHOST_COLOR[g.personality] || '#ffffff', g.scared > 0, flicker);
  });
}

function frame(now) {
  draw(now);
  requestAnimationFrame(frame);
}

// ── Panels ───────────────────────────────────────────────────────────────

function log(text, cls = 'log-info') {
  const line = document.createElement('div');
  line.className = cls;
  line.textContent = text;
  $('log').appendChild(line);
  $('log').scrollTop = $('log').scrollHeight;
}

// Screen position of a cell, for the particle and text effects in fx.js.
function screenXY(rc) {
  const rect = canvas.getBoundingClientRect();
  const s = rect.width / (S.rows[0].length * CELL);
  return [rect.left + (rc[1] + 0.5) * CELL * s, rect.top + (rc[0] + 0.5) * CELL * s];
}

function ghostInfo(policy) {
  return S.meta?.ghost_policies.find((p) => p.name === policy);
}

// Names the algorithm on each side. Shown under the title and in the GHOSTS chip, so the page
// always says what Pac-Man and the ghosts are running.
function updateMatchup() {
  const policy = $('ghosts').value;
  const ghosts = ghostInfo(policy)?.label ?? policy;
  const pacman = $('mode').value === 'play' ? 'You play Pac-Man' : `${$('agent').selectedOptions[0].textContent} plays Pac-Man`;
  $('matchup').textContent = `${pacman} against ${ghosts}.`;
  $('ghosts-desc').textContent = ghostInfo(policy)?.description ?? '';
  $('chip-ghosts').textContent = POLICY_CHIP[policy];
}

// The odds as a two-column table: each way a chance ghost can turn and its share of the draw.
function oddsTable() {
  const odds = S.meta?.chance_odds;
  const table = document.createElement('table');
  table.className = 'pm-feats';
  table.innerHTML = '<thead><tr><th>TURN</th><th>SHARE</th></tr></thead>';
  const body = document.createElement('tbody');
  for (const [name, share] of Object.entries(odds ?? {})) {
    const tr = document.createElement('tr');
    const turn = document.createElement('td');
    turn.textContent = ODDS_NAMES[name];
    const pct = document.createElement('td');
    pct.textContent = `${Math.round(100 * share)}%`;
    tr.append(turn, pct);
    body.appendChild(tr);
  }
  table.appendChild(body);
  return table;
}

// Score across watch-mode games for the current settings: wins, losses, and time-ups.
function renderTally() {
  const t = S.tally;
  const rate = t.games ? `${((100 * t.won) / t.games).toFixed(1)}%` : '—';
  $('score').replaceChildren(...[['GAMES', t.games], ['WON', t.won], ['LOST', t.lost], ['TIME UP', t.timeout], ['WIN RATE', rate]]
    .map(([label, value]) => {
      const chip = document.createElement('span');
      chip.className = 'stat-chip';
      chip.textContent = `${label} `;
      const val = document.createElement('span');
      val.className = 'val';
      val.textContent = String(value);
      chip.appendChild(val);
      return chip;
    }));
}

function resetTally() {
  S.tally = { games: 0, won: 0, lost: 0, timeout: 0 };
  renderTally();
}

// Counts a finished watch game once: a win, a loss, or running out of turns.
function countGame(result) {
  S.tally.games += 1;
  if (result.won) S.tally.won += 1;
  else if (result.status === 'timeout') S.tally.timeout += 1;
  else S.tally.lost += 1;
  renderTally();
}

function updateChips() {
  $('chip-mode').textContent = $('mode').value === 'play' ? 'PLAY' : 'WATCH';
  $('chip-ghosts').textContent = POLICY_CHIP[S.policy];
  $('chip-score').textContent = S.points.toLocaleString('en-US');
  $('chip-pellets').textContent = String(S.pellets.size);
  $('chip-turn').textContent = String(S.turn);
  $('chip-status').textContent = S.status.toUpperCase();
}

function updateBrain() {
  const list = $('brain-q');
  const table = $('brain-feats');
  const title = $('brain-title');
  const note = $('brain-note');
  list.innerHTML = '';
  if (!S.q) {
    table.hidden = true;
    note.textContent = '';
    if ($('mode').value === 'play') {
      title.textContent = 'You are playing, so no agent is scoring moves. Switch to watch mode to see its values.';
    } else if (S.lastAgent === 'reflex') {
      title.textContent = 'The reflex agent scores each move with a fixed rule, so it has no learned values.';
    } else if (S.lastAgent === 'random') {
      title.textContent = 'The random agent picks any legal move, so there are no values to show.';
    } else {
      title.textContent = 'Run the Q-learning agent to see its value for each move.';
    }
    return;
  }
  const values = Object.values(S.q);
  const lo = Math.min(...values), hi = Math.max(...values);
  title.textContent = `Q-value of each legal move. Chosen: ${S.lastChosen}`;
  for (const letter of LETTERS) {
    if (!(letter in S.q)) continue;
    const v = S.q[letter];
    const pct = hi === lo ? 100 : 12 + (88 * (v - lo)) / (hi - lo);
    const li = document.createElement('li');
    if (letter === S.lastChosen) li.className = 'chosen';
    li.innerHTML = '<span class="mv"></span><span class="bar"><i></i></span><span class="val"></span>';
    li.querySelector('.mv').textContent = letter;
    li.querySelector('.bar i').style.width = `${pct}%`;
    li.querySelector('.val').textContent = v.toFixed(2);
    list.appendChild(li);
  }
  const chosenFeats = S.features?.[S.lastChosen];
  const names = S.meta?.features ?? [];
  const weights = S.meta?.weights?.weights ?? [];
  if (!chosenFeats || !names.length) {
    table.hidden = true;
    return;
  }
  table.hidden = false;
  const body = table.querySelector('tbody');
  body.innerHTML = '';
  let total = 0;
  names.forEach((f, i) => {
    const value = chosenFeats[i];
    const w = weights[i] ?? 0;
    const contribution = value * w;
    total += contribution;
    const tr = document.createElement('tr');
    tr.innerHTML = '<td></td><td></td><td></td><td></td>';
    tr.children[0].textContent = f.name;
    tr.children[1].textContent = value.toFixed(3);
    tr.children[2].textContent = w.toFixed(2);
    tr.children[3].textContent = contribution.toFixed(2);
    tr.children[3].className = contribution > 0 ? 'pos' : contribution < 0 ? 'neg' : '';
    body.appendChild(tr);
  });
  note.textContent = `The contributions add up to this move's Q-value (${total.toFixed(2)}). Green pushes the move up, pink pushes it down.`;
  if (S.policy === 'chance') {
    note.textContent += ' The learned model was trained against the A* ghosts, so its survival look-ahead assumes their moves, not the chance odds.';
  }
}

function updateGhosts() {
  const panel = $('ghost-panel');
  panel.innerHTML = '';
  if (!S.ghosts.length) {
    panel.textContent = 'No ghosts yet.';
    return;
  }
  S.ghosts.forEach((g, i) => {
    const plan = S.plans.find((p) => p.ghost === i);
    let state = 'waiting';
    if (g.scared) {
      state = `scared for ${g.scared} more turn${g.scared === 1 ? '' : 's'}: fleeing`;
    } else if (plan && plan.target) {
      const atTarget = plan.target[0] === g.pos[0] && plan.target[1] === g.pos[1];
      state = atTarget
        ? 'on its target'
        : `heading for (${plan.target[0]}, ${plan.target[1]}), ${plan.path.length - 1} steps`;
    } else if (S.policy === 'chance') {
      state = 'moves by the fixed odds';
    }
    const row = document.createElement('div');
    row.className = 'pm-ghost';
    row.innerHTML = '<span class="sw"></span><span class="name"></span><span class="state"></span>';
    row.querySelector('.sw').style.color = GHOST_COLOR[g.personality] || '#ffffff';
    // Chance ghosts have no personality, so they are all just GHOST; the colour still tells them apart.
    const role = S.policy === 'chance' ? 'GHOST' : g.personality.toUpperCase();
    row.querySelector('.name').textContent = `${i + 1} · ${role}`;
    row.querySelector('.state').textContent = state;
    panel.appendChild(row);
  });
  const hint = document.createElement('div');
  hint.className = 'field-hint';
  hint.style.marginTop = '0.4rem';
  if (S.policy === 'chance') {
    hint.textContent = 'Chance ghosts ignore Pac-Man and the maze: each one keeps going, turns or reverses by the odds below.';
    panel.appendChild(hint);
    panel.appendChild(oddsTable());
    return;
  }
  hint.textContent = 'Chasers head for Pac-Man. The ambusher aims ahead of him. Scatter ghosts retreat to a corner when close.';
  panel.appendChild(hint);
}

function renderWeights() {
  const list = $('weights');
  const w = S.meta?.weights;
  list.innerHTML = '';
  if (!w) {
    $('weights-note').textContent = 'No trained weights yet. Run: python -m pacman train';
    return;
  }
  const max = Math.max(...w.weights.map(Math.abs), 1e-9);
  $('weights-note').textContent = `Learned over ${w.trained?.episodes ?? '?'} self-play games. Green weights reward a feature, pink ones punish it.`;
  w.features.forEach((name, i) => {
    const v = w.weights[i];
    const li = document.createElement('li');
    li.innerHTML = '<span class="nm"></span><span class="wbar"><i></i></span><span class="num"></span>';
    li.querySelector('.nm').textContent = name;
    const bar = li.querySelector('.wbar i');
    bar.className = v >= 0 ? 'pos' : 'neg';
    bar.style.width = `${(50 * Math.abs(v)) / max}%`;
    li.querySelector('.num').textContent = v.toFixed(1);
    list.appendChild(li);
  });
}

function updateAll() {
  updateChips();
  updateBrain();
  updateGhosts();
}

// ── State changes ────────────────────────────────────────────────────────

// Starts a fresh board from a start state (from the server), with no animation.
function setBoard(maze, start, policy) {
  S.policy = policy;
  S.rows = maze.rows;
  sizeCanvas();
  S.pac = start.pac;
  S.ghosts = start.ghosts.map((g) => ({ ...g }));
  S.pellets = new Set(start.pellets.map(([r, c]) => key(r, c)));
  S.plans = [];
  S.q = null; S.features = null; S.qAt = null;
  S.lastAction = 'E'; S.lastChosen = null; S.lastAgent = null;
  S.points = 0; S.turn = 0; S.status = 'playing';
  S.from = null; S.animDur = 0; S.deathAt = 0;
  updateAll();
}

// Begins the glide to a new state. The sprites start from wherever they are on screen, so a
// turn that arrives mid-animation does not snap.
function beginTurn(now, next) {
  const current = positions(now);
  S.from = { pac: current.pac, ghosts: current.ghosts.map((pos) => ({ pos })) };
  S.pac = next.pac;
  S.ghosts = next.ghosts.map((g) => ({ ...g }));
  S.animStart = now;
  S.animDur = Math.max(120, speedMs() * 0.85);
}

// Power bursts, eaten ghosts, deaths and the end of the maze, with a matching log line.
function reactToEvents(events) {
  const now = performance.now();
  for (const e of events) {
    if (e === 'power') {
      burst(screenXY(S.pac), { count: 70, colors: ['#3d5cff', '#00f5ff', '#ffffff'] });
      banner('POWER', 'ghosts turn blue for a while', '#3d5cff');
      log('power pellet: the ghosts are scared', 'log-best');
    } else if (e === 'ghost') {
      pop(screenXY(S.pac), `+${S.meta?.constants.ghost_points ?? 200}`, '#00ff88');
      burst(screenXY(S.pac), { count: 50, colors: ['#00ff88', '#00f5ff'] });
      log('ate a scared ghost', 'log-move');
    } else if (e === 'death') {
      S.deathAt = now;
      shake(canvas, 'big');
      banner('CAUGHT', `score ${S.points}`, '#ff3b3b');
      log('caught by a ghost', 'log-err');
    } else if (e === 'win') {
      burst(canvas, { count: 160, colors: ['#00ff88', '#ffe600', '#00f5ff'] });
      banner('MAZE CLEARED', `score ${S.points}`, '#00ff88');
      log(`all pellets eaten: ${S.points} points`, 'log-best');
    } else if (e === 'timeout') {
      banner('TIME UP', `score ${S.points}`, '#ffe600');
      log('out of turns', 'log-info');
    }
  }
}

// Shows one watch-mode turn: the move, the Q-values behind it, the routes, and the events.
function applyWatchTurn(turn) {
  const now = performance.now();
  S.qAt = [...S.pac]; // the decision was made from the cell Pac-Man is leaving
  beginTurn(now, { pac: turn.pac, ghosts: turn.ghosts });
  if (turn.eaten) S.pellets.delete(key(turn.eaten[0], turn.eaten[1]));
  S.lastAction = turn.action;
  S.lastChosen = turn.action;
  S.plans = turn.plans;
  S.q = turn.q;
  S.features = turn.features;
  S.points = turn.score;
  S.turn = S.index + 1;
  S.status = turn.status;
  S.lastAgent = S.episode.agent;
  updateAll();
  reactToEvents(turn.events);
  const q = turn.q ? ` (Q ${turn.q[turn.action].toFixed(1)})` : '';
  log(`${S.episode.agent} plays ${turn.action}${q}`, 'log-move');
}

// Shows the state after a human move; the server's record of that turn supplies the events and routes.
function applyHumanState(res, letter) {
  const { state, last } = res;
  beginTurn(performance.now(), { pac: state.pac, ghosts: state.ghosts });
  S.lastAction = letter;
  S.lastChosen = null;
  S.qAt = null;
  S.pellets = new Set(state.pellets.map(([r, c]) => key(r, c)));
  S.plans = last ? last.plans : [];
  S.q = null;
  S.features = null;
  S.points = state.points;
  S.turn = state.turn;
  S.status = state.status;
  S.lastAgent = null;
  S.legal = res.legal;
  updateAll();
  if (last) reactToEvents(last.events);
}

// ── Modes ────────────────────────────────────────────────────────────────

function seedValue() {
  const v = Number($('seed').value);
  return Number.isInteger(v) && v >= 0 ? v : 7;
}

function stopPlayback() {
  S.playing = false;
  clearTimeout(S.timer);
  $('btn-toggle').querySelector('.btn-txt').textContent = '▶ RESUME';
}

function setBusy(flag) {
  S.busy = flag;
  $('btn-run').disabled = flag;
  $('btn-new').disabled = flag;
}

// Loads the board for the current maze and seed. Both modes start from the same opening position.
async function loadBoard() {
  const res = await postJSON('/api/pacman/play', {
    maze: $('maze').value, seed: seedValue(), actions: [], ghosts: $('ghosts').value,
  });
  stopPlayback();
  S.actions = [];
  S.episode = null;
  S.legal = res.legal;
  setBoard(res.maze, res.state, res.ghosts);
  updateMatchup();
}

function setMode(mode) {
  const watch = mode === 'watch';
  for (const id of ['btn-run', 'btn-toggle', 'btn-step']) $(id).hidden = !watch;
  $('agent-row').hidden = !watch;
  $('speed').closest('div').hidden = !watch;
  $('score-row').hidden = !watch;
  $('pm-hint').textContent = watch ? HINT_WATCH : HINT_PLAY;
  $('btn-new').querySelector('.btn-txt').textContent = watch ? '↺ NEW SEED' : '↺ NEW GAME';
  loadBoard().catch((e) => log(e.message, 'log-err'));
}

async function runAgent() {
  stopPlayback();
  setBusy(true);
  const seed = seedValue();
  try {
    const episode = await postJSON('/api/pacman/episode', {
      agent: $('agent').value, maze: $('maze').value, seed, max_turns: 300, ghosts: $('ghosts').value,
    });
    S.episode = episode;
    S.index = 0;
    setBoard(episode.maze, episode.start, episode.ghosts);
    updateMatchup();
    log(`${episode.agent} vs ${episode.ghosts_label} on ${episode.maze.title}, seed ${seed}: ${episode.turns.length} turns`, 'log-info');
    playFrom();
  } catch (e) {
    log(e.message, 'log-err');
  } finally {
    setBusy(false);
  }
}

// Advances watch playback by one turn. Returns false once the game's turns are used up.
function stepWatch() {
  const turns = S.episode?.turns ?? [];
  if (S.index >= turns.length) {
    stopPlayback();
    const r = S.episode?.result;
    if (r) log(`game over: ${r.status}, score ${r.score} in ${r.turns} turns`, r.won ? 'log-best' : 'log-info');
    // Each game is counted once, however many times the end is reached (step, then play).
    if (r && !S.episode.scored) {
      S.episode.scored = true;
      countGame(r);
      if ($('auto-restart').checked) {
        S.timer = setTimeout(() => {
          if ($('mode').value === 'watch' && $('auto-restart').checked) {
            randomSeed();
            runAgent();
          }
        }, NEXT_GAME_MS);
      }
    }
    return false;
  }
  applyWatchTurn(turns[S.index]);
  S.index += 1;
  return true;
}

function playFrom() {
  if (!S.episode) return;
  S.playing = true;
  $('btn-toggle').querySelector('.btn-txt').textContent = '❚❚ PAUSE';
  const tick = () => {
    if (!S.playing || !stepWatch()) return;
    S.timer = setTimeout(tick, speedMs());
  };
  tick();
}

async function humanMove(letter) {
  if (S.busy) return;
  if (!S.legal.includes(letter)) {
    shake(canvas);
    $('pm-hint').textContent = 'A wall is in the way. Pick another direction.';
    return;
  }
  setBusy(true);
  try {
    const actions = [...S.actions, letter];
    const res = await postJSON('/api/pacman/play', {
      maze: $('maze').value, seed: seedValue(), actions, ghosts: S.policy,
    });
    S.actions = actions;
    applyHumanState(res, letter);
    if (S.status === 'playing') {
      $('pm-hint').textContent = HINT_PLAY;
    } else {
      $('pm-hint').textContent = 'Game over. Press New game to try again.';
      log(`game over: ${S.status}, score ${S.points}`, S.status === 'won' ? 'log-best' : 'log-info');
    }
  } catch (e) {
    log(e.message, 'log-err');
  } finally {
    setBusy(false);
  }
}

// ── Wiring ───────────────────────────────────────────────────────────────

function wire() {
  const reload = () => loadBoard().catch((e) => log(e.message, 'log-err'));

  $('mode').addEventListener('change', () => setMode($('mode').value));
  $('maze').addEventListener('change', () => { resetTally(); reload(); });
  $('seed').addEventListener('change', () => { if ($('mode').value === 'play') reload(); });
  $('agent').addEventListener('change', () => {
    const info = S.meta?.agents.find((a) => a.name === $('agent').value);
    $('agent-desc').textContent = info ? info.description : '';
    resetTally();
    updateMatchup();
  });
  $('ghosts').addEventListener('change', () => {
    // The board and any watched game were built under the old ghosts, so start again under the new ones.
    resetTally();
    stopPlayback();
    loadBoard().catch((e) => log(e.message, 'log-err'));
  });
  $('btn-reset-score').addEventListener('click', resetTally);
  $('speed').addEventListener('input', () => { $('speed-val').textContent = `${speedMs()} ms per turn`; });
  $('btn-run').addEventListener('click', runAgent);
  $('btn-toggle').addEventListener('click', () => {
    if (!S.episode) return;
    if (S.playing) stopPlayback(); else playFrom();
  });
  $('btn-step').addEventListener('click', () => {
    stopPlayback();
    stepWatch();
  });
  $('btn-new').addEventListener('click', () => {
    randomSeed();
    if ($('mode').value === 'play') reload(); else runAgent();
  });
  $('btn-seed').addEventListener('click', () => {
    randomSeed();
    if ($('mode').value === 'play') reload();
  });
  addEventListener('keydown', (e) => {
    if (e.target.matches('input, select, textarea')) return;
    const letter = KEY_MOVES[e.key.toLowerCase()];
    if (!letter || $('mode').value !== 'play') return;
    e.preventDefault();
    humanMove(letter);
  });
}

async function init() {
  wire();
  $('speed-val').textContent = `${speedMs()} ms per turn`;
  requestAnimationFrame(frame);
  try {
    S.meta = await getJSON('/api/pacman/meta');
  } catch (e) {
    log(`could not reach the server: ${e.message}`, 'log-err');
    return;
  }
  for (const m of S.meta.mazes) {
    const opt = document.createElement('option');
    opt.value = m.name;
    opt.textContent = m.title;
    $('maze').appendChild(opt);
  }
  $('agent-desc').textContent = S.meta.agents[0].description;
  $('odds-table').replaceChildren(oddsTable());
  renderTally();
  renderWeights();
  updateMatchup();
  setMode($('mode').value);
  log('ready. Watch the agent, or switch to You play and use the arrow keys or W A S D.', 'log-info');
}

init();
