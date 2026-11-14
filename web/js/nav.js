// Site navigation: MAIN, a GAMES menu, a LABS menu, and the current page as the active item.
// The lists live here: a new game is one line in GAMES, a new lab one line in LABS. Each page ships a plain MAIN
// link as its no-JavaScript fallback, and this module replaces it with the full menu. It also adds the sound
// toggle (sfx.js).
import { isEnabled, setEnabled, subscribe } from './sfx.js';
// Overdrive: the Konami code and seven logo taps, live on every page (overdrive.js).
import './overdrive.js';
// Search-wave reveal: each page arrives with a BFS wave from the last click (reveal.js).
import './reveal.js';

export const GAMES = [
  { href: 'npuzzle.html', label: 'N-PUZZLE' },
  { href: 'connect4.html', label: 'CONNECT FOUR' },
  { href: 'checkers.html', label: 'CHECKERS' },
  { href: 'routes.html', label: 'ROUTES' },
  { href: '2048.html', label: '2048' },
  { href: 'sudoku.html', label: 'SUDOKU' },
  { href: 'lightsout.html', label: 'LIGHTS OUT' },
  { href: 'blackjack.html', label: 'BLACKJACK' },
  { href: 'battleship.html', label: 'BATTLESHIP' },
  { href: 'pacman.html', label: 'PAC-MAN' },
  { href: 'warehouse.html', label: 'WAREHOUSE' },
  { href: 'wordle.html', label: 'WORDLE' },
  { href: 'minesweeper.html', label: 'MINESWEEPER' },
  { href: 'hexgame.html', label: 'HEX' },
  { href: 'queens.html', label: 'N-QUEENS' },
  { href: 'snake.html', label: 'SNAKE' },
  { href: 'tetris.html', label: 'TETRIS' },
  { href: 'nonogram.html', label: 'NONOGRAM' },
  { href: 'tictactoe.html', label: 'TIC-TAC-TOE' },
];

// Games hidden from the menus for now. Their pages, APIs and landing sims still exist and their URLs still work;
// this list only gives a hidden page its own name in the nav when someone opens it directly. It is not exported,
// so the 404 page's game links stay limited to GAMES. To bring a game back:
// 1. move its line from HIDDEN to GAMES;
// 2. restore its cabinet in web/index.html and its row in the README game table, from the parent of the commit
//    "chore(web): hide eight games from the menus and landing page for now";
// 3. add its algorithm back to the two meta descriptions in web/index.html, and update the game counts there and
//    in the README.
const HIDDEN = [
  { href: 'endgame.html', label: 'ENDGAME' },
  { href: 'sokoban.html', label: 'SOKOBAN' },
  { href: 'poker.html', label: 'POKER' },
  { href: 'bandits.html', label: 'BANDITS' },
  { href: 'cartpole.html', label: 'CART POLE' },
  { href: 'rover.html', label: 'ROVER' },
  { href: 'ghosthunt.html', label: 'GHOST HUNT' },
  { href: 'localize.html', label: 'LOST ROBOT' },
];

// Labs are playgrounds where you change the inputs and watch an algorithm respond. They are not games, so
// they have their own menu and do not count toward the game total.
const LABS = [
  { href: 'clusters.html', label: 'CLUSTER LAB' },
  { href: 'nnlab.html', label: 'NEURAL NET LAB' },
  { href: 'pathfind.html', label: 'PATHFIND' },
  { href: 'mdplab.html', label: 'MDP LAB' },
  { href: 'optlab.html', label: 'OPTIMIZER RACE' },
  { href: 'treelab.html', label: 'TREE LAB' },
  { href: 'markov.html', label: 'MARKOV TEXT' },
  { href: 'regression.html', label: 'REGRESSION' },
  { href: 'walkers.html', label: 'WALKERS' },
];

const list = document.getElementById('nav-links');
if (list) {
  const here = location.pathname.split('/').pop() || 'index.html';
  const item = (href, label, active) =>
    `<li><a href="${href}"${active ? ' class="act" aria-current="page"' : ''}>${label}</a></li>`;
  const dropdown = (title, entries) =>
    `<li class="games-menu"><details><summary>${title} <span aria-hidden="true">▾</span></summary></details>`
    + `<ul class="games-panel">${entries.map((g) => item(g.href, g.label, g.href === here)).join('')}</ul></li>`;
  const current = [...GAMES, ...LABS, ...HIDDEN].find((g) => g.href === here);
  list.innerHTML = item('index.html', 'MAIN', here === 'index.html')
    + dropdown('GAMES', GAMES)
    + dropdown('LABS', LABS)
    + (current ? item(current.href, current.label, true) : '');
  // On phones the links scroll sideways, so bring the current page into view. It waits for the web fonts, because
  // the link widths change when they load. On desktop the links all fit, so nothing moves.
  const showCurrent = () => list.querySelector(':scope > li > a.act')?.scrollIntoView({ block: 'nearest', inline: 'nearest' });
  document.fonts.ready.then(showCurrent);

  // Escape closes the open menu and returns focus to its summary; a click anywhere else closes every open menu.
  const menus = [...list.querySelectorAll('.games-menu')].map((menu) => ({ menu, details: menu.querySelector('details') }));
  for (const { menu, details } of menus) {
    menu.addEventListener('keydown', (e) => {
      if (e.key === 'Escape' && details.open) {
        details.open = false;
        details.querySelector('summary').focus();
      }
    });
  }
  document.addEventListener('click', (e) => {
    for (const { menu, details } of menus) {
      if (details.open && !menu.contains(e.target)) details.open = false;
    }
  });

  // Sound toggle: the last item of the link list, so the nav's space-between layout is unchanged. Off by default.
  const sound = document.createElement('li');
  sound.className = 'sound-li';
  sound.innerHTML = '<button type="button" class="sound-toggle" aria-pressed="false" title="Sound effects">'
    + '<svg viewBox="0 0 24 24" aria-hidden="true" focusable="false">'
    + '<path class="body" d="M3 9v6h4l5 4V5L7 9z"/>'
    + '<path class="wave" d="M16 8.5a5 5 0 0 1 0 7M18.5 6a8.5 8.5 0 0 1 0 12"/>'
    + '<path class="slash" d="M4 4l16 16"/></svg>'
    + '<span class="sr-only">Sound effects</span></button>';
  list.appendChild(sound);
  const toggle = sound.querySelector('button');
  const sync = (on) => { toggle.setAttribute('aria-pressed', String(on)); };
  sync(isEnabled());
  subscribe(sync);
  toggle.addEventListener('click', () => setEnabled(!isEnabled()));
}
