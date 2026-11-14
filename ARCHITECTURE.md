# Architecture

search-agents is a set of games and labs, each played or solved by a classic AI algorithm. Every algorithm is
written once in Python, usable from the terminal, and shown in the browser by a small static frontend talking to a
FastAPI server. This file covers how the parts fit together; the README's "How it works" section covers each
algorithm.

## Layout

```
web/            static HTML/CSS/JS, no build step; one page per game or lab
server/         FastAPI app: REST endpoints per game, a WebSocket for N-Puzzle searches, request guards
<game>/         one Python package per game or lab: rules, agents, CLI (__main__.py), benchmarks, data files
tests/          pytest suites per package and per server router; tests/web/ holds node tests and layout checks
results/        committed benchmark output (JSON and Markdown) that the README and pages quote
docs/           longer write-ups (e.g. connect4-training.md)
```

The game and lab packages are `npuzzle connect4 checkers routes game2048 sudoku lightsout blackjack battleship
pacman warehouse endgame sokoban wordle poker minesweeper hexgame bandits cartpole queens snake rover ghosthunt
tetris nonogram localize clusters nnlab pathfind mdplab optlab treelab markov regression walkers tictactoe`. Eight
of them (endgame, sokoban, poker, bandits, cartpole, rover, ghosthunt, localize) are currently hidden from the
site's menus; their pages, APIs and tests still exist (see the `HIDDEN` list in `web/js/nav.js`).

## Module boundaries

- **Game packages own the algorithms.** A package has no web or server code: rules, search or learning, a CLI and
  its benchmark live together, and the package runs on its own (`python -m <package>`). Packages do not import each
  other, except shared helpers that are imported explicitly (for example the seeded RNG in `bandits/rng.py`).
- **`server/` is a thin adapter.** Each `<game>_api.py` router validates the request, calls into its package in a
  worker thread, and shapes the JSON. `server/limits.py` defines the guards every router uses (client key, body
  size, finite-number JSON, rate limiters, search slots), so the rules exist once. `server/app.py` wires routers,
  middleware and the static files together.
- **`web/` is static.** Pages are plain HTML with ES modules, no bundler. Shared modules: `nav.js` (the game and
  lab lists, the only place they live), `api.js` (server calls), `fx.js` and `sfx.js` (effects and sound),
  `play-layout.js` (fitting boards to the screen), and `explain.js` (formulas, pseudocode and folding sections).
- **Browser ports of algorithms.** Some pages run their algorithm in the browser so they need no server round trip
  (tic-tac-toe, most labs, the landing-page sims). Each port is a pure `*-core.js` module mirroring its Python
  package, and a parity test (`tests/test_*_parity.py`, `tests/web/*.test.mjs`) checks they agree.

## Request lifecycle

1. The browser loads a static page from `web/` (served by the same FastAPI app, so there is one origin).
2. Page JS calls a REST endpoint under `/api/<game>/...`, or for N-Puzzle opens the `/ws/npuzzle` WebSocket, which
   streams the search's expansions as they happen.
3. The router checks the body (size cap, finite JSON, value ranges), the client's rate limit, and for expensive
   work takes one of the search slots (`SEARCH_SLOTS`, default 2); if none frees up within a few seconds the
   request gets a 503 (on the WebSocket, an error message) instead of queuing forever.
4. The work runs in a worker thread with node and time limits, and the result goes back as JSON.

Unknown pages get `web/404.html` (a tic-tac-toe game) with status 404; unknown paths under `/api/` stay JSON 404s.

## Data and models

- **Generated, committed data.** Pattern databases, puzzle sets, word lists, trained weights and benchmark results
  are committed, and each has the command that regenerates it (for example `python -m npuzzle.pdb`,
  `python -m sudoku.generate`). The server loads the heavy ones at startup so the first request is not slow.
- **No torch when serving.** Models trained with torch (the AlphaZero Connect Four net, the N-Puzzle neural
  heuristic) are exported to `.npz` and evaluated in NumPy, with batch norm folded into the weights. torch is only
  in the `train` extra, which keeps torch (about 350 MB) out of the server image; every model loads in about
  120 MB of RAM.
- **Trained sizes are fixed.** A model trained on one board size is only offered at that size (Connect Four 7x6,
  2048 4x4, Snake 12x12, the Pac-Man Q-agent's 15x11 mazes). Size-agnostic algorithms get the other sizes.

## Constraints that shaped the design

- **One small machine.** The site runs on one Fly.io machine with 1 GB of RAM, so searches that store every state
  (about 1 KB per expanded node) are capped (250,000 expanded nodes up to 4x4 N-Puzzle boards, less on larger boards), IDS and IDA\* get a larger
  cap because their memory is linear in depth, and lookup tables use compact containers (`bytes`, `array`).
- **A public demo.** Every endpoint assumes hostile input: bounded bodies, validated sizes and ids, per-client rate
  limits, and capped work per request.
- **No build step.** The frontend is plain files so a page can be read, changed and reloaded directly.

## Tooling

CI (`.github/workflows/tests.yml`) runs ruff lint and format checks, typos, the pytest suite, the Node frontend tests
(`tests/web/*.test.mjs`), and a gitleaks scan of each push's or pull request's commits. The same checks run locally
through lefthook git hooks; see `CONTRIBUTING.md`.
