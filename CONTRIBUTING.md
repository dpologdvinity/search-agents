# Contributing

How to set up the project, run it, check a change, and add a new game. `ARCHITECTURE.md` explains how the parts fit
together.

## Setup

Python 3.11+, [uv](https://docs.astral.sh/uv/), and Node 18+ (for the frontend tests).

```bash
uv venv && uv pip install -e ".[dev]"      # add ".[train]" for torch and matplotlib (training only)
```

The git hooks also need [lefthook](https://lefthook.dev), [typos](https://github.com/crate-ci/typos) and
[gitleaks](https://github.com/gitleaks/gitleaks) on your PATH, and ruff at the version pinned in `pyproject.toml`
(formatter output can change between ruff versions). Then:

```bash
lefthook install                           # lint, format, spelling and secret checks on staged files
```

Skip the hooks for one commit with `LEFTHOOK=0 git commit ...` only when you will fix the finding right after.

## Running

```bash
uvicorn server.app:app --reload            # the site and API on http://localhost:8000
python -m <package>                        # a game or lab in the terminal, e.g. python -m connect4
python -m <package> --help                 # its commands (benchmark, train, match, ...)
```

Run modules from the repository root so the packages import.

## Checks

These are the checks CI runs; run them before you push.

```bash
ruff check . && ruff format --check . && typos   # lint, formatting, spelling
python -m pytest -q                              # all Python tests (tests that need torch skip without it)
TTT_PYTHON=.venv/bin/python node --test tests/web/*.test.mjs   # frontend modules (TTT_PYTHON: the Python for the tic-tac-toe parity check)
gitleaks git --redact .                          # secrets anywhere in the history
```

Pages with a fitted board layout have a browser check. It needs Playwright
(`uv pip install playwright && playwright install chromium`) and a running server:

```bash
python tests/web/play_layout.py http://127.0.0.1:8000
```

## Branches and commits

- `master` is the main branch. Work on a short-lived branch and fast-forward `master` onto it when it is done;
  do not rewrite commits that are already pushed.
- Commits follow [Conventional Commits](https://www.conventionalcommits.org/): `type(scope): description`, for
  example `fix(npuzzle): report the per-size node cap`. Keep each commit to one logical change.
- Bug fixes start with a failing test that reproduces the bug. New logic comes with tests.

## Code style

- Every module and public function has a docstring saying what it does and why. Comment the non-obvious steps
  (rules, formulas, tricks), not the obvious lines.
- Browser ports of a Python algorithm (`web/js/*-core.js`) stay pure (no DOM) and get a parity test against the
  Python version.
- Server endpoints validate every input at the boundary (sizes, ids, ranges) and use the guards in
  `server/limits.py`; work that can run long takes a search slot and has node and time limits.
- Code that links to source lines on GitHub (`<a class="ref">` in the pages) must be updated when those lines move.

## Adding a game

A new game touches these places:

1. A package `<game>/` with the rules, agents, a CLI (`__main__.py`) and tests in `tests/`.
2. A router `server/<game>_api.py`, included in `server/app.py` (skip if the game runs entirely in the browser).
3. A page `web/<game>.html` (+ CSS and JS), with the same static fallback nav as the other pages.
4. One line in `GAMES` in `web/js/nav.js`, the only list of games for the menus and the 404 page.
5. A landing cabinet in `web/index.html` with its sim in `web/js/landing.js`, and the game count in the landing
   page's hero and meta descriptions.
6. The package in `pyproject.toml` (`packages`) and a `COPY` line in the `Dockerfile`.
7. A row in the README's game table, and its sections under Results and How it works.

Generated data files (pattern databases, puzzle sets, trained weights, benchmark results) are committed. Regenerate
them with their package's command and commit the output with the code that produced it.
