// Endgame page: a king and queen (or rook) against a lone king, with every position solved.
//
// The page never searches. Each position goes to /api/endgame/analyze, which looks it up in a tablebase
// built by retrograde analysis: working backwards from checkmate, every position gets an exact distance
// to mate (DTM), or a draw. The reply lists every legal move with its own DTM, so the board can tag each
// move, and the tablebase picks its reply from the same list. The side with the piece wants the smallest DTM;
// the lone king wants the largest: the longest resistance.
//
// The other opponent is the Chance (fixed odds) player from endgame-core.js: it draws a move from the same
// list by fixed weights on each move's kind (captures, checks, king steps to the centre, other moves), and
// never reads the tablebase. Two modes use them: you play the agent, or watch the tablebase play chance.
//
// Positions are {wk, wp, bk, stm} with squares 0..63 (a1 = 0, h8 = 63) and stm 0 = White, 1 = Black.

import { getJSON, postJSON } from './api.js';
import { banner, burst, pop, shake } from './fx.js';
import { pickAt, probabilities, seededRandom } from './endgame-core.js';

const $ = (id) => document.getElementById(id);
const WHITE = 0, BLACK = 1;
const FILES = 'abcdefgh';
const GLYPH = { K: '♔', Q: '♕', R: '♖', k: '♚' };
const TAG_CLASSES = ['tag-win', 'tag-draw', 'tag-loss', 'tag-mate'];
const AGENT_DELAY = 650; // pause before the agent answers, so the reply can be read
const HISTORY_LIMIT = 200;
const WATCH_MAX_PLIES = 300; // a watched game stops here; the chance side can hold out in an endless line
const WATCH_RESTART_DELAY = 1800; // pause between the end of one watched game and the next
const OPP_LABEL = { tablebase: 'Tablebase (exact)', chance: 'Chance (fixed odds)' };
const OPP_SHORT = { tablebase: 'TABLEBASE', chance: 'CHANCE' };

const state = {
  meta: null,
  piece: 'Q',         // the white piece: Q or R
  human: BLACK,       // BLACK: you are the lone king; WHITE: you have the king and the piece
  opponent: 'tablebase', // who answers you: 'tablebase' (exact) or 'chance' (fixed odds)
  watching: false,    // MODE: watch the tablebase play the chance opponent, with no moves from you
  tbSide: WHITE,      // watch mode: the side the tablebase plays (the other side is chance)
  paused: false,      // watch mode: PAUSE stops after the current move
  speed: 600,         // watch mode: milliseconds between moves
  autoRestart: true,  // watch mode: start the next game when one ends
  score: { win: 0, draw: 0, loss: 0, unfinished: 0 }, // watch mode: the tablebase's results
  roll: null,         // seeded uniform generator for the chance opponent's draws, reset for each game
  seed: 0,            // seed of the current game; each new game takes the next one
  plies: 0,           // moves made in the current game (watch mode stops at WATCH_MAX_PLIES)
  games: 0,           // watch mode: games started since the score was last reset
  pos: null,          // the current position
  info: null,         // the analyze reply for pos: value, moves, best
  history: [],        // states before each of your moves, for UNDO
  selected: null,     // square of the piece you have picked up
  showHint: false,    // HINT: list every move and mark the best one
  heat: false,        // HEAT MAP: colour the lone king's squares by distance to mate
  heatData: null,     // the heat-map reply for the current pieces and side to move
  last: null,         // {from, to} of the last move played
  over: null,         // 'checkmate' | 'stalemate' | 'draw' once the game has ended
  busy: false,
  timer: 0,           // pending agent move
  moves: 0,           // moves you have made
  cells: [],          // 64 entries, indexed by square: {el, piece, tag, heat}
};

const sqName = (sq) => FILES[sq & 7] + ((sq >> 3) + 1);
const agentSide = () => 1 - state.human;
const sideName = (stm) => (stm === WHITE ? 'White' : 'Black');

// Which opponent moves for side stm. Watch mode has one per side (the tablebase on tbSide, chance on the
// other). Play mode has the agent on its side, and nothing on yours.
function opponentFor(stm) {
  if (state.watching) return stm === state.tbSide ? 'tablebase' : 'chance';
  return stm === agentSide() ? state.opponent : null;
}
// True when the engine, not you, is to move: always in watch mode.
const isAgentTurn = () => !!state.pos && (state.watching || state.pos.stm === agentSide());

// You can pick up the king and piece when you are White, and the lone king when you are Black. Never in watch mode.
function ownPiece(sq) {
  const p = state.pos;
  if (!p || state.watching || p.stm !== state.human || state.over) return false;
  return state.human === WHITE ? sq === p.wk || sq === p.wp : sq === p.bk;
}

// The legal moves that start on a square, from the analyze reply.
const movesFrom = (sq) => (state.info ? state.info.moves.filter((m) => m.from === sq) : []);

// The position after a move from the analyze reply (captures are handled separately).
function applyMove(pos, mv) {
  const next = { ...pos, stm: 1 - pos.stm };
  if (mv.mover === 'K') next.wk = mv.to;
  else if (mv.mover === 'k') next.bk = mv.to;
  else next.wp = mv.to;
  return next;
}

// ── Labels and rankings ─────────────────────────────────────────────────

// The tag for a move, as the mover sees it: M3 (mate in 3), = (drawn), L9 (mated in 9 at most), # (checkmate).
function tagFor(mv) {
  if (mv.mate) return { text: '#', cls: 'tag-mate' };
  if (mv.outcome === 'draw') return { text: '=', cls: 'tag-draw' };
  if (mv.outcome === 'win') return { text: `M${mv.mate_in}`, cls: 'tag-win' };
  return { text: `L${mv.mate_in}`, cls: 'tag-loss' };
}

// Higher is better for the mover. Same order as endgame/tablebase.py rank_key: fast wins, draws, long losses.
function rankOf(mv) {
  if (mv.outcome === 'win') return [2, -mv.plies];
  if (mv.outcome === 'draw') return [1, 0];
  return [0, mv.plies];
}
function compareRank(a, b) {
  const [a0, a1] = rankOf(a), [b0, b1] = rankOf(b);
  return a0 !== b0 ? a0 - b0 : a1 - b1;
}

function describeMove(mv) {
  if (mv.outcome === 'draw') return 'draw';
  return mv.outcome === 'win' ? `mate in ${mv.mate_in}` : `mated in ${mv.mate_in} at most`;
}

// ── Board ───────────────────────────────────────────────────────────────

// Build the 64 square buttons once, rank 8 at the top and file a on the left, and keep them by square.
function buildBoard() {
  const board = $('eg-board');
  board.innerHTML = '';
  state.cells = new Array(64);
  for (let r = 7; r >= 0; r--) {
    for (let f = 0; f < 8; f++) {
      const sq = r * 8 + f;
      const el = document.createElement('button');
      el.type = 'button';
      el.className = `eg-sq ${(r + f) % 2 ? 'dark' : 'light'}`;
      el.dataset.sq = String(sq);
      const piece = document.createElement('span');
      piece.className = 'eg-piece';
      const tag = document.createElement('span');
      tag.className = 'eg-tag';
      tag.hidden = true;
      const heat = document.createElement('span');
      heat.className = 'eg-heat-n';
      el.append(piece, tag, heat);
      if (f === 0) el.insertAdjacentHTML('beforeend', `<span class="eg-rank">${r + 1}</span>`);
      if (r === 0) el.insertAdjacentHTML('beforeend', `<span class="eg-coord">${FILES[f]}</span>`);
      board.appendChild(el);
      state.cells[sq] = { el, piece, tag, heat };
    }
  }
}

// Repaint the board from state. Cheap enough to call after every change.
function paint() {
  const p = state.pos;
  const info = state.info;
  const agentTurn = isAgentTurn();
  // Tags on the squares the picked-up piece can reach.
  const targets = new Map();
  if (info && state.selected !== null && !agentTurn) {
    for (const mv of movesFrom(state.selected)) targets.set(mv.to, mv);
  }
  // HINT: the best move's squares are outlined in green.
  const best = info && info.best && !agentTurn ? info.moves.find((m) => m.uci === info.best.uci) : null;

  for (let sq = 0; sq < 64; sq++) {
    const { el, piece, tag, heat } = state.cells[sq];
    let glyph = '', color = '';
    if (p) {
      if (sq === p.wk) { glyph = GLYPH.K; color = 'white'; }
      else if (sq === p.wp) { glyph = GLYPH[state.piece]; color = 'white'; }
      else if (sq === p.bk) { glyph = GLYPH.k; color = 'black'; }
    }
    piece.textContent = glyph;
    piece.className = `eg-piece ${color}`;
    el.classList.toggle('own', ownPiece(sq));
    el.classList.toggle('selected', state.selected === sq);
    el.classList.toggle('last', !!state.last && (state.last.from === sq || state.last.to === sq));
    el.classList.toggle('hint-from', !!best && state.showHint && best.from === sq);
    el.classList.toggle('hint-to', !!best && state.showHint && best.to === sq);
    el.classList.toggle('eg-target', targets.has(sq));
    el.setAttribute('aria-label', `${sqName(sq)}${glyph ? ' ' + glyph : ''}`);

    el.classList.remove(...TAG_CLASSES);
    const mv = targets.get(sq);
    if (mv) {
      const t = tagFor(mv);
      el.classList.add(t.cls);
      tag.textContent = t.text;
      tag.className = `eg-tag ${t.cls}`;
      tag.hidden = false;
    } else {
      tag.hidden = true;
    }

    // Heat map: a square the lone king may stand on, tinted by its distance to mate for the side to move.
    el.classList.remove('heat-win', 'heat-loss', 'heat-draw', 'heat-illegal');
    heat.textContent = '';
    el.style.removeProperty('--h');
    if (state.heat && state.heatData) {
      const outcome = state.heatData.outcome[sq];
      if (outcome === null) {
        el.classList.add('heat-illegal');
      } else if (outcome === 'draw') {
        el.classList.add('heat-draw');
        heat.textContent = '=';
      } else {
        const moves = state.heatData.moves[sq];
        const max = Math.max(state.meta ? state.meta.pieces[state.piece].max_win_moves : 10, 1);
        el.style.setProperty('--h', String(Math.min(1, moves / max)));
        el.classList.add(outcome === 'win' ? 'heat-win' : 'heat-loss');
        heat.textContent = String(moves);
      }
    }
  }
  $('eg-board').classList.toggle('over', !!state.over);
  renderChips();
  renderCountdown();
  renderSides();
  renderThink();
}

// The line under the board: who plays each side, by name, so the algorithm in use is always visible.
function renderSides() {
  if (state.watching) {
    $('eg-sides').textContent = `White: ${OPP_LABEL[opponentFor(WHITE)]}. Black: ${OPP_LABEL[opponentFor(BLACK)]}.`;
    return;
  }
  $('eg-sides').textContent = `You play ${sideName(state.human)}. The agent plays ${sideName(agentSide())}: `
    + `${OPP_LABEL[state.opponent]}.`;
}

// ── Panels ──────────────────────────────────────────────────────────────

function statusText() {
  if (state.busy) return 'THINKING';
  if (state.over === 'checkmate') return 'CHECKMATE';
  if (state.over === 'stalemate') return 'STALEMATE';
  if (state.over === 'draw') return 'DRAWN';
  if (!state.info) return 'LOADING';
  if (state.info.value.outcome === 'draw') return 'DRAWN';
  if (isAgentTurn()) return `${OPP_SHORT[opponentFor(state.pos.stm)]} TO MOVE`;
  return 'YOUR MOVE';
}

// The side-to-move tag in the TO MOVE chip: who moves for that side.
function turnTag(stm) {
  if (state.watching) return ` (${OPP_SHORT[opponentFor(stm)]})`;
  return stm === state.human ? ' (you)' : ' (agent)';
}

// Moves the side with the piece needs to mate from here, or null when the position is drawn.
function mateNumber() {
  if (!state.info || state.info.value.outcome === 'draw') return null;
  return state.info.value.moves;
}

function renderChips() {
  const p = state.pos;
  const n = mateNumber();
  $('chip-status').textContent = statusText();
  $('chip-turn').textContent = p ? `${sideName(p.stm)}${turnTag(p.stm)}` : '—';
  $('chip-mate').textContent = n === null || n === undefined ? '—' : String(n);
  $('chip-moves').textContent = String(state.moves);
  $('chip-pos').textContent = state.meta ? state.meta.pieces[state.piece].legal.toLocaleString('en-US') : '—';
}

// The big number above the board: moves left for the side with the piece, with best play from both sides.
function renderCountdown() {
  const box = $('eg-countdown');
  const info = state.info;
  const n = mateNumber();
  const drawn = !info || info.value.outcome === 'draw';
  // Red when the agent is mating you, green when you are mating the agent, yellow for a draw.
  const youAreMated = !state.watching && !!info && info.value.outcome === 'loss' && state.pos.stm === state.human;
  box.classList.toggle('drawn', drawn);
  box.classList.toggle('lost', youAreMated);
  box.classList.remove('tick');
  if (!info) {
    $('eg-cd-num').textContent = '—';
    $('eg-cd-txt').textContent = 'loading the tablebase';
    return;
  }
  if (n === null) {
    $('eg-cd-num').textContent = '½';
    $('eg-cd-txt').textContent = 'neither side can force mate: a draw';
    return;
  }
  if (state.over === 'checkmate') {
    $('eg-cd-num').textContent = '0';
    $('eg-cd-txt').textContent = 'checkmate';
    return;
  }
  const who = state.watching ? 'White mates' : state.human === WHITE ? 'you mate' : 'the agent mates you';
  $('eg-cd-num').textContent = String(n);
  $('eg-cd-txt').textContent = `${n === 1 ? 'move' : 'moves'} left: ${who}, with best play from both sides`;
  void box.offsetWidth; // restart the tick animation
  box.classList.add('tick');
}

// The reason the tablebase picks its move, by the side it plays.
function tablebaseWhy(stm) {
  return stm === WHITE
    ? 'The tablebase has the king and the piece, so it picks the move that leaves the fewest plies to mate.'
    : 'The tablebase is the lone king, so it picks the move that lasts longest: a drawing move if there is one, '
      + 'otherwise the largest distance to mate.';
}

// THE AGENT IS THINKING: what the table says about this position, why the agent picks what it picks,
// and the alternatives it compared. For chance, the alternatives are listed with their odds.
function renderThink() {
  const body = $('think-body');
  const title = $('think-title');
  body.textContent = '';
  const info = state.info;
  if (!info) return;
  const add = (text, cls = '') => {
    const p = document.createElement('p');
    p.textContent = text;
    if (cls) p.className = cls;
    body.appendChild(p);
  };
  const v = info.value;
  if (v.outcome === 'win') {
    title.textContent = `${sideName(state.pos.stm)} to move wins: mate in ${v.moves}.`;
    add(`The table has this position at ${v.plies} plies to mate. Each winning move leaves the opponent a position one ply closer to mate, so the count drops by one on every best move.`);
  } else if (v.outcome === 'loss') {
    title.textContent = `${sideName(state.pos.stm)} to move is mated in ${v.moves} at most.`;
    add(`Every move loses. The table has the slowest defence at ${v.plies} plies.`);
  } else if (v.outcome === 'draw') {
    title.textContent = 'Neither side can force mate from here.';
    add('The table has no winning line for either side, so the position is drawn with best play.');
  }

  const ranked = info.moves.slice().sort((a, b) => compareRank(b, a));
  const who = isAgentTurn() ? opponentFor(state.pos.stm) : null;
  // Chance: each legal move's odds, largest first. The tablebase: its ranking, with its best move marked.
  const odds = new Map();
  if (who === 'chance') {
    const probs = probabilities(info.moves, state.pos.wp);
    info.moves.forEach((mv, i) => odds.set(mv.uci, probs[i]));
    ranked.sort((a, b) => odds.get(b.uci) - odds.get(a.uci));
    const ways = info.moves.length;
    add(`Chance (fixed odds) draws this move at random. Each legal move's weight is its kind's odds: captures 4, `
      + `checks 2, king steps toward the centre 2, other moves 1. Its chance is that weight over the total of the `
      + `${ways} legal ${ways === 1 ? 'move' : 'moves'}. It does not search and does not read the tablebase.`, 'eg-why');
  } else if (who === 'tablebase' && info.best) {
    add(tablebaseWhy(state.pos.stm), 'eg-why');
  }
  const list = document.createElement('div');
  list.className = 'eg-alt-list';
  const bestUci = who === 'tablebase' && info.best ? info.best.uci : null;
  const shown = state.showHint ? ranked : ranked.slice(0, 6);
  for (const mv of shown) {
    const row = document.createElement('div');
    row.className = `eg-alt ${mv.outcome}${mv.uci === bestUci ? ' best' : ''}`;
    const name = document.createElement('span');
    name.textContent = mv.san;
    const d = document.createElement('span');
    d.className = 'eg-alt-d';
    d.textContent = odds.has(mv.uci)
      ? `${(odds.get(mv.uci) * 100).toFixed(1)}% · ${describeMove(mv)}`
      : describeMove(mv);
    row.append(name, d);
    list.appendChild(row);
  }
  if (shown.length) body.appendChild(list);
  if (!state.showHint && ranked.length > shown.length) {
    add(`…and ${ranked.length - shown.length} more. Press HINT to list them all.`, 'eg-why');
  }
}

// The histogram of moves to mate over every position of the current piece, with a marker at this one.
function drawHistogram() {
  const canvas = $('eg-hist');
  const ctx = canvas.getContext('2d');
  const W = canvas.width, H = canvas.height;
  ctx.clearRect(0, 0, W, H);
  if (!state.meta) return;
  const data = state.meta.pieces[state.piece];
  const win = data.win_histogram, loss = data.loss_histogram;
  const maxK = Math.max(win.length, loss.length);
  const peak = Math.max(...win, ...loss, 1);
  const pad = { l: 40, r: 8, t: 12, b: 26 };
  const plotW = W - pad.l - pad.r, plotH = H - pad.t - pad.b;
  const slot = plotW / Math.max(maxK, 1);
  ctx.font = '10px JetBrains Mono, monospace';
  // For each move count: green is the side to move winning in k moves, pink is it being mated in k moves.
  for (let k = 1; k < maxK; k++) {
    const x = pad.l + k * slot;
    const hw = ((win[k] || 0) / peak) * plotH;
    const hl = ((loss[k] || 0) / peak) * plotH;
    ctx.fillStyle = 'rgba(0,255,136,0.75)';
    ctx.fillRect(x + 1, pad.t + plotH - hw, slot * 0.42, hw);
    ctx.fillStyle = 'rgba(255,0,160,0.75)';
    ctx.fillRect(x + slot * 0.46, pad.t + plotH - hl, slot * 0.42, hl);
    if (k % 2 === 0 || maxK <= 11) {
      ctx.fillStyle = '#4a7a9b';
      ctx.textAlign = 'center';
      ctx.fillText(String(k), x + slot / 2, H - 10);
    }
  }
  ctx.fillStyle = '#4a7a9b';
  ctx.textAlign = 'right';
  ctx.fillText(peak.toLocaleString('en-US'), pad.l - 4, pad.t + 8);
  ctx.fillText('0', pad.l - 4, pad.t + plotH);
  const n = mateNumber();
  if (n !== null && n !== undefined) {
    const x = pad.l + (n + 0.5) * slot;
    ctx.strokeStyle = '#ffe600';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(x, pad.t);
    ctx.lineTo(x, pad.t + plotH);
    ctx.stroke();
    ctx.fillStyle = '#ffe600';
    ctx.textAlign = 'left';
    ctx.fillText(`you are here: ${n}`, Math.min(x + 4, W - 110), pad.t + 10);
  }
  $('hist-title').textContent = `${data.name}: ${data.legal.toLocaleString('en-US')} legal positions`;
  $('hist-note').textContent = `Green: the side to move wins in k moves (longest ${data.max_win_moves}). `
    + `Pink: it is mated in k moves (longest ${data.max_loss_moves}).`;
}

// The "how it is built" cards, filled with the real numbers from the server.
function renderHow() {
  const { Q: q, R: r } = state.meta.pieces;
  const cards = [
    ['01 · THE SEEDS', 'cyan', 'A position where the side to move is checkmated is decided at once: zero plies from mate. A stalemate is a draw. Everything else is decided by what it leads to.'],
    ['02 · STEP BACKWARDS', 'pink', 'For each decided position, find its predecessors: the positions one quiet move earlier that lead to it. A piece steps back along the lines it could have slid along; a king steps to a neighbouring square.'],
    ['03 · COUNT THE ESCAPES', 'green', 'Every undecided position counts its legal moves. A predecessor where the side to move has a move into a lost position is a win at once. A predecessor whose moves all lead to wins is lost once its last such move is decided.'],
    ['04 · ORDER BY DISTANCE', 'yellow', 'Positions are processed in order of distance to mate. So the first win found is the fastest, and the last move to be decided in a lost position is its longest defence. Whatever is never decided is a draw.'],
    ['05 · THE RESULT', 'cyan', `KQK: ${q.legal.toLocaleString('en-US')} legal positions, longest mate ${q.max_win_moves} moves. KRK: ${r.legal.toLocaleString('en-US')} positions, longest mate ${r.max_win_moves} moves. Each table builds in seconds, and every entry was re-checked against the rules.`],
  ];
  const wrap = $('how');
  wrap.textContent = '';
  for (const [title, color, text] of cards) {
    const card = document.createElement('div');
    card.className = 'heuristic-card';
    card.style.padding = '0.7rem';
    const t = document.createElement('div');
    t.className = 'eg-how-n';
    t.style.color = `var(--${color})`;
    t.textContent = title;
    const d = document.createElement('div');
    d.className = 'field-hint';
    d.textContent = text;
    card.append(t, d);
    wrap.appendChild(card);
  }
}

function log(text, cls = 'log-info') {
  const line = document.createElement('div');
  line.className = cls;
  line.textContent = text;
  $('log').appendChild(line);
  $('log').scrollTop = $('log').scrollHeight;
}

function note(text) {
  $('eg-note').textContent = text;
}

// ── Game flow ───────────────────────────────────────────────────────────

// Make `pos` the current position. `info` is its analyze reply if we already have one.
async function show(pos, info = null) {
  state.busy = true;
  state.selected = null;
  paint();
  try {
    state.info = info || (await postJSON('/api/endgame/analyze', {
      piece: state.piece, wk: pos.wk, wp: pos.wp, bk: pos.bk, stm: pos.stm,
    }));
    state.pos = pos;
    if (state.heat && !state.watching) { // watch mode skips the heat map: it would double the requests per move
      state.heatData = await getJSON('/api/endgame/heatmap', {
        piece: state.piece, wk: pos.wk, wp: pos.wp, stm: pos.stm,
      });
    }
  } catch (err) {
    log(`✗ ${err.message}`, 'log-err');
    state.busy = false;
    if (state.watching) pauseWatch(`paused: ${err.message}`); // usually the move rate limit: PLAY resumes
    paint();
    return false;
  }
  state.busy = false;
  paint();
  drawHistogram();
  if (state.info.moves.length === 0) {
    endFromNoMoves();
  } else if (isAgentTurn()) {
    scheduleAgent();
  } else {
    note('Click or drag one of your pieces, then a highlighted square. Tags show each move\'s distance to mate.');
  }
  return true;
}

// No legal moves: checkmate when the side to move is in check (the table's value is loss in 0), else stalemate.
function endFromNoMoves() {
  if (state.info.value.outcome === 'loss') {
    if (state.watching) {
      finish('checkmate', `${sideName(state.pos.stm)} is checkmated.`, null);
      return;
    }
    const youLost = state.pos.stm === state.human;
    finish('checkmate', youLost ? 'The agent mates you.' : 'You mate the agent.', youLost);
  } else {
    finish('stalemate', 'Stalemate: a draw.', null);
  }
}

// The tablebase's result for a watched game that ended as kind: it wins when the chance side is mated.
function watchResult(kind) {
  if (kind === 'checkmate') return state.pos.stm === state.tbSide ? 'loss' : 'win';
  return kind === 'unfinished' ? 'unfinished' : 'draw';
}

// End the game. `youLost` is true or false for checkmate, null for a draw or a watched game.
function finish(kind, text, youLost) {
  state.over = kind;
  stopTimer();
  state.selected = null;
  note(text);
  log(`★ ${text}`, 'log-best');
  paint();
  if (state.watching) {
    state.score[watchResult(kind)] += 1;
    renderScore();
    if (state.autoRestart) state.timer = setTimeout(newWatchGame, WATCH_RESTART_DELAY);
  }
  if (youLost === false) {
    burst($('eg-board'), { count: 150, colors: ['#00ff88', '#00f5ff', '#ffe600'] });
    shake($('eg-board'), 'big');
    banner('CHECKMATE', text, '#00ff88');
  } else if (youLost === true) {
    shake($('eg-board'), 'big');
    banner('CHECKMATE', text, '#ff00a0');
  } else if (kind === 'checkmate') {
    shake($('eg-board'), 'big');
    banner('CHECKMATE', text, '#00f5ff');
  } else {
    shake($('eg-board'), 'small');
    banner(kind === 'stalemate' ? 'STALEMATE' : kind === 'unfinished' ? 'UNFINISHED' : 'DRAWN', text, '#ffe600');
  }
}

function stopTimer() {
  if (state.timer) clearTimeout(state.timer);
  state.timer = 0;
}

function scheduleAgent() {
  stopTimer();
  if (state.watching) {
    note(state.paused
      ? 'Paused. STEP plays the next move, PLAY resumes.'
      : 'Watching: the tablebase and the chance opponent take turns. PAUSE stops after the current move.');
    if (state.paused) return;
  }
  state.timer = setTimeout(agentMove, state.watching ? state.speed : AGENT_DELAY);
}

// The move the side to move plays: the tablebase's best move, or a draw from the chance odds. The label in
// the log says which, so a watched game reads clearly.
function agentMove() {
  state.timer = 0;
  const info = state.info;
  if (!info || state.over || !info.best) return;
  if (state.watching && state.plies >= WATCH_MAX_PLIES) {
    finish('unfinished', `Stopped after ${WATCH_MAX_PLIES} plies: neither side has finished.`, null);
    return;
  }
  const who = opponentFor(state.pos.stm);
  const mv = who === 'chance'
    ? pickAt(info.moves, state.pos.wp, state.roll())
    : info.moves.find((m) => m.uci === info.best.uci);
  if (!mv) return;
  state.plies += 1;
  state.last = { from: mv.from, to: mv.to };
  log(`${OPP_SHORT[who].toLowerCase()}: ${mv.san}  (${describeMove(mv)})`, 'log-adv');
  if (mv.capture) {
    captured(mv);
    return;
  }
  show(applyMove(state.pos, mv));
}

// The black king took the white piece: king against king. Show the capture, then end as a draw.
function captured(mv) {
  const before = state.pos;
  state.pos = { ...before, bk: mv.to, wp: -1, stm: 1 - before.stm };
  state.info = { ...state.info, moves: [], best: null, value: { outcome: 'draw', plies: null, moves: null } };
  finish('draw', 'The lone king took the piece: king against king, a draw.', null);
}

// Your move: judge it against the table, log it, and play it.
function humanMove(mv) {
  if (state.busy || state.over || !state.info) return;
  const info = state.info;
  const best = info.best ? info.moves.find((m) => m.uci === info.best.uci) : null;
  state.history.push({ pos: state.pos, info, last: state.last, moves: state.moves });
  if (state.history.length > HISTORY_LIMIT) state.history.shift();
  state.moves += 1;
  state.plies += 1;
  state.last = { from: mv.from, to: mv.to };

  let verdict = '';
  if (state.human === WHITE && best) {
    if (mv.uci === best.uci) verdict = ' · best move';
    else if (mv.outcome === 'win') verdict = ` · not the fastest: ${best.san} was mate in ${best.mate_in}`;
    else if (best.outcome === 'win') verdict = ` · blunder: ${best.san} was mate in ${best.mate_in}`;
    else verdict = ' · blunder: this loses';
  }
  log(`you: ${mv.san}  (${describeMove(mv)})${verdict}`, verdict.includes('blunder') ? 'log-err' : 'log-move');
  const label = mv.outcome === 'win' ? `M${mv.mate_in}` : mv.outcome === 'draw' ? '=' : `L${mv.mate_in}`;
  const color = mv.outcome === 'win' ? '#00ff88' : mv.outcome === 'draw' ? '#ffe600' : '#ff00a0';
  pop(state.cells[mv.to].el, label, color);

  if (mv.capture) {
    captured(mv);
    return;
  }
  show(applyMove(state.pos, mv));
}

// ── Input ───────────────────────────────────────────────────────────────

// Click handling: the first click picks up your piece; the second on a highlighted square moves it.
function onSquare(sq) {
  if (state.busy || state.over || isAgentTurn() || !state.info) return;
  if (state.selected !== null) {
    const mv = movesFrom(state.selected).find((m) => m.to === sq);
    if (mv) {
      humanMove(mv);
      return;
    }
  }
  state.selected = ownPiece(sq) ? sq : null;
  paint();
}

// Drag: pick a piece up, carry a ghost of it under the pointer, drop it on a legal target.
let drag = null;

function wireBoard() {
  const board = $('eg-board');
  board.addEventListener('click', (e) => {
    const cell = e.target.closest('.eg-sq');
    if (cell) onSquare(Number(cell.dataset.sq));
  });
  board.addEventListener('pointerdown', (e) => {
    const cell = e.target.closest('.eg-sq');
    if (!cell || state.busy || state.over || isAgentTurn() || !state.info) return;
    const sq = Number(cell.dataset.sq);
    if (!ownPiece(sq)) return;
    state.selected = sq;
    paint();
    const ghost = $('eg-ghost');
    ghost.textContent = cell.querySelector('.eg-piece').textContent;
    ghost.className = `eg-ghost ${state.human === WHITE ? 'white' : 'black'}`;
    drag = { sq, x: e.clientX, y: e.clientY, moved: false };
    board.setPointerCapture?.(e.pointerId);
  });
  board.addEventListener('pointermove', (e) => {
    if (!drag) return;
    if (!drag.moved && Math.hypot(e.clientX - drag.x, e.clientY - drag.y) < 6) return;
    drag.moved = true;
    const ghost = $('eg-ghost');
    ghost.classList.add('on');
    ghost.style.left = `${e.clientX}px`;
    ghost.style.top = `${e.clientY}px`;
    const over = document.elementFromPoint(e.clientX, e.clientY)?.closest?.('.eg-sq');
    for (const { el } of state.cells) el.classList.toggle('dragover', el === over);
  });
  const release = (e) => {
    if (!drag) return;
    const d = drag;
    drag = null;
    $('eg-ghost').classList.remove('on');
    for (const { el } of state.cells) el.classList.remove('dragover');
    if (!d.moved) return; // a plain click: the click handler does the work
    const over = document.elementFromPoint(e.clientX, e.clientY)?.closest?.('.eg-sq');
    const to = over ? Number(over.dataset.sq) : null;
    const mv = to === null ? undefined : movesFrom(d.sq).find((m) => m.to === to);
    if (mv) humanMove(mv);
    else paint();
  };
  board.addEventListener('pointerup', release);
  board.addEventListener('pointercancel', release);
}

// Start a fresh random position where the side with the piece wins, with you to move.
async function newPosition() {
  stopTimer();
  try {
    const reply = await getJSON('/api/endgame/random', { piece: state.piece, want: 'strong_wins', stm: state.human });
    await showReply(reply, 'new position');
  } catch (err) {
    log(`✗ ${err.message}`, 'log-err');
  }
}

// Take a reply from the random or FEN endpoint as the new game. Each game gets the next seed, so the chance
// opponent's draws in it can be repeated.
async function showReply(reply, what) {
  stopTimer();
  state.piece = reply.piece;
  $('eg-piece').value = reply.piece;
  state.over = null;
  state.history = [];
  state.moves = 0;
  state.plies = 0;
  state.last = null;
  state.seed += 1;
  state.roll = seededRandom(state.seed);
  await show({ wk: reply.wk, wp: reply.wp, bk: reply.bk, stm: reply.stm }, reply);
  log(`${what}: ${reply.fen}`, 'log-info');
}

// Watch mode: a random winning position for the piece side, with the tablebase on tbSide and chance on the other.
async function newWatchGame() {
  stopTimer();
  state.tbSide = Number($('eg-watch-seat').value);
  try {
    const reply = await getJSON('/api/endgame/random', { piece: state.piece, want: 'strong_wins' });
    state.games += 1;
    await showReply(reply, `game ${state.games}`);
  } catch (err) {
    log(`✗ ${err.message}`, 'log-err');
    pauseWatch(`paused: ${err.message}`);
  }
}

// The big button: a new position in play mode, the next game in watch mode.
function startNew() {
  return state.watching ? newWatchGame() : newPosition();
}

function renderScore() {
  const s = state.score;
  const finished = s.win + s.draw + s.loss;
  $('eg-score').textContent = `Tablebase: ${s.win} won · ${s.draw} drawn · ${s.loss} lost`
    + (s.unfinished ? ` · ${s.unfinished} unfinished` : '');
  $('eg-score-note').textContent = finished
    ? `Tablebase win rate ${Math.round((100 * s.win) / finished)}% over ${finished} finished ${finished === 1 ? 'game' : 'games'}.`
    : 'Games appear here as they finish.';
}

function resetScore() {
  state.score = { win: 0, draw: 0, loss: 0, unfinished: 0 };
  state.games = 0;
  renderScore();
}

// Watch mode pauses itself on an error (usually the move rate limit) so the next request is the user's choice.
function pauseWatch(text) {
  state.paused = true;
  stopTimer();
  updatePauseUI();
  note(text);
}

function updatePauseUI() {
  $('btn-pause').querySelector('.btn-txt').textContent = state.paused ? '▶ PLAY' : '❚❚ PAUSE';
}

function togglePause() {
  if (!state.watching) return;
  state.paused = !state.paused;
  updatePauseUI();
  if (state.paused) {
    stopTimer();
    note('Paused. STEP plays the next move, PLAY resumes.');
  } else if (state.over) {
    if (state.autoRestart) newWatchGame();
  } else if (state.info && !state.busy) {
    scheduleAgent();
  }
}

// STEP: pause if running, then play one move.
function stepWatch() {
  if (!state.watching || state.busy || state.over || !state.info) return;
  if (!state.paused) togglePause();
  agentMove();
}

// MODE: play against the agent, or watch the tablebase and chance play each other.
async function setMode(mode) {
  stopTimer();
  state.watching = mode === 'watch';
  state.paused = false;
  state.over = null;
  $('eg-side-wrap').hidden = state.watching;
  $('eg-opp-wrap').hidden = state.watching;
  $('eg-watch-wrap').hidden = !state.watching;
  $('eg-watch-controls').hidden = !state.watching;
  $('eg-score-box').hidden = !state.watching;
  $('btn-heat').disabled = state.watching;
  $('btn-undo').disabled = state.watching;
  $('btn-random').querySelector('.btn-txt').textContent = state.watching ? '↺ NEXT GAME' : '↺ RANDOM';
  updatePauseUI();
  if (state.watching) {
    if (state.heat) await toggleHeat();
    resetScore();
    await newWatchGame();
  } else {
    await newPosition();
  }
}

async function setFen() {
  const text = $('eg-fen').value.trim();
  $('eg-fen-err').textContent = '';
  if (!text) return;
  try {
    const reply = await postJSON('/api/endgame/analyze', { fen: text });
    await showReply(reply, 'set up');
  } catch (err) {
    $('eg-fen-err').textContent = err.message;
  }
}

function undo() {
  if (state.busy || !state.history.length) return;
  stopTimer();
  const prev = state.history.pop();
  state.over = null;
  state.moves = prev.moves;
  state.last = prev.last;
  state.pos = prev.pos;
  state.info = prev.info;
  state.selected = null;
  log('undo: back to your move', 'log-info');
  paint();
  drawHistogram();
}

async function toggleHeat() {
  if (state.watching) return; // the heat map is for playing: it doubles the requests of a watched move
  state.heat = !state.heat;
  $('btn-heat').setAttribute('aria-pressed', String(state.heat));
  $('btn-heat').querySelector('.btn-txt').textContent = state.heat ? '■ HEAT MAP' : '▦ HEAT MAP';
  if (state.heat && state.pos) {
    try {
      state.heatData = await getJSON('/api/endgame/heatmap', {
        piece: state.piece, wk: state.pos.wk, wp: state.pos.wp, stm: state.pos.stm,
      });
    } catch (err) {
      log(`✗ ${err.message}`, 'log-err');
    }
  }
  note(state.heat
    ? 'Heat map: each square the lone king could stand on, shaded by moves to mate for the side to move.'
    : 'Click or drag one of your pieces, then a highlighted square. Tags show each move\'s distance to mate.');
  paint();
}

function toggleHint() {
  state.showHint = !state.showHint;
  $('btn-hint').querySelector('.btn-txt').textContent = state.showHint ? '■ HIDE HINT' : '✦ HINT';
  paint();
}

// ── Startup ─────────────────────────────────────────────────────────────

async function init() {
  buildBoard();
  wireBoard();
  $('btn-random').onclick = startNew;
  $('btn-undo').onclick = undo;
  $('btn-hint').onclick = toggleHint;
  $('btn-heat').onclick = toggleHeat;
  $('btn-fen').onclick = setFen;
  $('btn-pause').onclick = togglePause;
  $('btn-step').onclick = stepWatch;
  $('eg-fen').addEventListener('keydown', (e) => {
    if (e.key === 'Enter') setFen();
  });
  $('eg-piece').onchange = (e) => {
    state.piece = e.target.value;
    startNew();
  };
  $('eg-side').onchange = (e) => {
    state.human = Number(e.target.value);
    newPosition();
  };
  $('eg-side').value = String(state.human);
  $('eg-mode').onchange = (e) => setMode(e.target.value);
  // The opponent can change mid-game: the next agent move is drawn from the new one.
  $('eg-opponent').onchange = (e) => {
    state.opponent = e.target.value;
    paint();
  };
  $('eg-opponent').value = state.opponent;
  $('eg-watch-seat').onchange = () => {
    resetScore();
    newWatchGame();
  };
  $('eg-speed').onchange = (e) => {
    state.speed = Number(e.target.value);
  };
  state.speed = Number($('eg-speed').value);
  $('eg-auto').onchange = (e) => {
    state.autoRestart = e.target.checked;
  };
  state.autoRestart = $('eg-auto').checked;
  renderScore();
  try {
    state.meta = await getJSON('/api/endgame/meta');
  } catch (err) {
    log(`✗ Server unreachable (${err.message}).`, 'log-err');
    $('chip-status').textContent = 'OFFLINE';
    return;
  }
  renderHow();
  await newPosition();
}

init();
