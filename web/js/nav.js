// Site navigation: MAIN, a GAMES menu, and the current game as the active item.
// The game list lives here, so a new game is one line in GAMES. Each page ships a plain MAIN link as its
// no-JavaScript fallback, and this module replaces it with the full menu.
const GAMES = [
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
  { href: 'endgame.html', label: 'ENDGAME' },
  { href: 'sokoban.html', label: 'SOKOBAN' },
  { href: 'wordle.html', label: 'WORDLE' },
  { href: 'poker.html', label: 'POKER' },
  { href: 'minesweeper.html', label: 'MINESWEEPER' },
  { href: 'hexgame.html', label: 'HEX' },
  { href: 'bandits.html', label: 'BANDITS' },
  { href: 'cartpole.html', label: 'CART POLE' },
  { href: 'queens.html', label: 'N-QUEENS' },
  { href: 'snake.html', label: 'SNAKE' },
  { href: 'rover.html', label: 'ROVER' },
  { href: 'tetris.html', label: 'TETRIS' },
  { href: 'nonogram.html', label: 'NONOGRAM' },
];

const list = document.getElementById('nav-links');
if (list) {
  const here = location.pathname.split('/').pop() || 'index.html';
  const item = (href, label, active) =>
    `<li><a href="${href}"${active ? ' class="act" aria-current="page"' : ''}>${label}</a></li>`;
  const current = GAMES.find((g) => g.href === here);
  list.innerHTML = item('index.html', 'MAIN', here === 'index.html')
    + '<li class="games-menu"><details><summary>GAMES <span aria-hidden="true">▾</span></summary></details>'
    + `<ul class="games-panel">${GAMES.map((g) => item(g.href, g.label, g.href === here)).join('')}</ul></li>`
    + (current ? item(current.href, current.label, true) : '');

  // Escape closes the menu and returns focus to its summary; a click anywhere else closes it too.
  const menu = list.querySelector('.games-menu');
  const details = menu.querySelector('details');
  menu.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && details.open) {
      details.open = false;
      details.querySelector('summary').focus();
    }
  });
  document.addEventListener('click', (e) => {
    if (details.open && !menu.contains(e.target)) details.open = false;
  });
}
