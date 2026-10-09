"""Play Snake in the terminal, watch an agent play, train the agents, or run the benchmark and action counts.

    python -m snake                              # play yourself, turn by turn (w a s d, one step per line)
    python -m snake play --delay 0.15            # timed: the snake keeps moving; w a s d set its heading
    python -m snake watch --agent evolved --seed 3 --delay 0.08
    python -m snake watch --agent planner --seed 3 --steps 400
    python -m snake evolve                       # the committed run (about 10 minutes on two cores)
    python -m snake evolve --generations 20 --pop 40 --games 2 --no-save    # a quick sketch
    python -m snake cem                          # cross-entropy over the evaluator's 8 weights (about 55 minutes)
    python -m snake benchmark --games 200 --write
    python -m snake actions --games 40 --write   # how often the net picks left, straight, right

Keys in turn mode: w a s d set the next move's direction (w is north). A blank line keeps the
current heading. h shows the planner's suggested move, q quits. In timed mode the same keys set the
heading and q quits. The snake cannot reverse, so the opposite key is ignored.
"""

from __future__ import annotations

import argparse
import json
import os
import select
import sys
import time
from pathlib import Path

from .agents import AGENT_NAMES, DESCRIPTIONS, make_policy, planner_path
from .benchmark import (
    DEFAULT_GAMES,
    RESULTS_DIR,
    action_counts,
    run_benchmark,
    to_markdown,
    write_action_counts,
    write_results,
)
from .board import ACTION_NAMES, BOARD, HEADING_NAMES, STRAIGHT, Game, action_toward
from .cem import CEMSettings, cem, write_weights
from .cem import settings_dict as cem_settings_dict
from .evaluator import WEIGHTS_PATH
from .evolve import Settings, evolve, settings_dict
from .net import CHAMPION_PATH, Net

COMMANDS = ("play", "watch", "evolve", "cem", "benchmark", "actions")
KEY_HEADINGS = {"w": 0, "d": 1, "s": 2, "a": 3}  # w north, d east, s south, a west
ARROWS = {0: "^", 1: ">", 2: "v", 3: "<"}


def render(game: Game) -> str:
    """ASCII board: `@`-like head as an arrow for its heading, `o` body, `*` food, `.` empty."""
    rows = []
    cells = {}
    for i, cell in enumerate(game.body):
        cells[cell] = ARROWS[game.heading] if i == 0 else "o"
    if game.food is not None:
        cells[game.food] = "*"
    rows.append("   " + " ".join(chr(ord("a") + c) for c in range(game.size)))
    for y in range(game.size):
        line = " ".join(cells.get((x, y), ".") for x in range(game.size))
        rows.append(f"{y + 1:>2} {line}")
    return "\n".join(rows)


def status(game: Game) -> str:
    """One line under the board: apples, steps, heading, and the game's end once it has one."""
    text = f"apples {game.apples}  steps {game.steps}  heading {HEADING_NAMES[game.heading]}  length {game.length}"
    if not game.alive:
        text += f"  game over: {CAUSE_TEXT.get(game.cause, game.cause)}"
    return text


CAUSE_TEXT = {
    "wall": "hit a wall",
    "body": "hit its own body",
    "starved": "no apple for too long",
    "full": "the board is full: you win",
}


def play_turns(seed: int = 0, read=input, out=print, size: int = BOARD) -> int:
    """Turn-based play: each line is one move. Returns the apples eaten. The game also ends on EOF or `q`."""
    game = Game(seed, size)
    out(render(game))
    out("w a s d = set direction for the next move, blank = straight on, h = planner's hint, q = quit")
    while game.alive:
        try:
            text = read("> ").strip().lower()
        except EOFError:
            break
        if text in ("q", "quit", "exit"):
            break
        if text == "h":
            path = planner_path(game)
            if path:
                out(f"hint: the planner heads for the food, next cell {path[0]}")
            else:
                out("hint: the planner sees no route to the food")
            continue
        if text == "":
            action = STRAIGHT
        elif text[0] in KEY_HEADINGS:
            action = action_toward(game.heading, KEY_HEADINGS[text[0]])
            if action is None:
                out("cannot reverse into your own neck")
                continue
        else:
            out(f"'{text}' is not a key: use w a s d, a blank line, h or q")
            continue
        game.step(action)
        out(render(game))
        out(status(game) + f"  (last move: {ACTION_NAMES[action]})")
    out(status(game))
    return game.apples


def play_timed(seed: int = 0, delay: float = 0.15, out=print, size: int = BOARD) -> int:
    """Timed play on a terminal: the snake moves every `delay` seconds, and w a s d set its heading.

    The terminal is put in cbreak mode (single keys, no echo) and restored afterwards. Uses select() on
    stdin, so it needs a real terminal (Linux or macOS).
    """
    import termios
    import tty

    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    game = Game(seed, size)
    desired = game.heading
    try:
        tty.setcbreak(fd)
        next_tick = time.monotonic() + delay
        while game.alive:
            timeout = max(0.0, next_tick - time.monotonic())
            ready, _, _ = select.select([fd], [], [], timeout)
            if ready:
                key = os.read(fd, 1).decode(errors="ignore").lower()
                if key == "q":
                    break
                if key in KEY_HEADINGS:
                    desired = KEY_HEADINGS[key]
            if time.monotonic() >= next_tick:
                # A reversal request is ignored: the snake keeps its heading for that tick.
                action = action_toward(game.heading, desired)
                game.step(STRAIGHT if action is None else action)
                out("\x1b[2J\x1b[H" + render(game) + "\n" + status(game))
                next_tick += delay
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
    out(status(game))
    return game.apples


def watch(agent: str, seed: int, delay: float, steps: int | None, out=print, size: int = BOARD) -> Game:
    """Run one agent on one seeded game, printing each frame. `steps` caps the number of moves shown."""
    game = Game(seed, size)
    policy = make_policy(agent, seed=seed)
    out(f"{agent}: {DESCRIPTIONS[agent]}")
    while game.alive and (steps is None or game.steps < steps):
        action = policy(game)
        game.step(action)
        out("\x1b[2J\x1b[H" + render(game) + f"\n{status(game)}  move: {ACTION_NAMES[action]}")
        if delay > 0:
            time.sleep(delay)
    out(status(game))
    return game


def _write_champion(path: Path, genome, meta: dict) -> None:
    net = Net.from_genome(genome)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(net.to_json(**meta), separators=(",", ":")) + "\n")


def _evolve_cmd(args) -> int:
    s = Settings(pop=args.pop, games=args.games, generations=args.generations, elites=args.elites,
                 tournament=args.tournament, mutation_rate=args.mutation_rate, mutation_sigma=args.mutation_sigma,
                 seed=args.seed, workers=args.workers, starve_penalty=args.starve_penalty)
    t0 = time.perf_counter()
    checkpoint_dir = Path(args.checkpoint_dir) if args.checkpoint_dir else None
    records: list[dict] = []

    def show(rec, best_genome):
        records.append(rec)
        print(f"gen {rec['generation']:>3}  best {rec['best_fitness']:7.3f}  mean {rec['mean_fitness']:7.3f}  "
              f"best apples {rec['best_apples']:6.2f}  mean apples {rec['mean_apples']:6.2f}  "
              f"best steps {rec['best_steps']:7.1f}  {rec['elapsed_s']:7.1f}s", flush=True)
        # Checkpoints hold the best-ever training genome, not the validated champion, so they are marked as such.
        # They let a partial run be used or inspected without waiting for the end.
        if checkpoint_dir and rec["generation"] % args.checkpoint_every == 0:
            meta = {"checkpoint": True, "generation": rec["generation"], "settings": settings_dict(s)}
            _write_champion(checkpoint_dir / f"champion_gen{rec['generation']:03d}.json", best_genome, meta)
            (checkpoint_dir / "log.jsonl").write_text("".join(json.dumps(h) + "\n" for h in records))

    genome, score, history, best_gen = evolve(s, on_generation=show)
    print(f"done in {time.perf_counter() - t0:.0f}s: champion from generation {best_gen}, "
          f"validation fitness {score.fitness:.3f}, apples {score.apples:.2f} per game")
    if args.no_save:
        return 0
    meta = {
        "settings": settings_dict(s),
        "best_generation": best_gen,
        "train_fitness": round(score.fitness, 4),
        "train_apples": round(score.apples, 3),
        "train_steps": round(score.steps, 1),
        "history": history,
    }
    champion = Path(args.champion) if args.champion else CHAMPION_PATH
    log = Path(args.log) if args.log else RESULTS_DIR / "snake_train_log.jsonl"
    _write_champion(champion, genome, meta)
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text("".join(json.dumps(r) + "\n" for r in history))
    print(f"wrote {champion} and {log}")
    return 0


def _cem_cmd(args) -> int:
    s = CEMSettings(pop=args.pop, games=args.games, generations=args.generations, elites=args.elites,
                    init_sigma=args.init_sigma, min_sigma=args.min_sigma, seed=args.seed, workers=args.workers)
    t0 = time.perf_counter()

    def show(rec):
        print(f"gen {rec['generation']:>3}  best {rec['best_fitness']:7.3f}  mean {rec['mean_fitness']:7.3f}  "
              f"best apples {rec['best_apples']:6.2f}  mean apples {rec['mean_apples']:6.2f}  "
              f"best steps {rec['best_steps']:7.1f}  sigma {rec['sigma_mean']:.3f}  {rec['elapsed_s']:7.1f}s",
              flush=True)

    weights, score, history, best_gen = cem(s, on_generation=show)
    print(f"done in {time.perf_counter() - t0:.0f}s: champion from generation {best_gen}, validation fitness "
          f"{score.fitness:.3f}, apples {score.apples:.2f} per game")
    if args.no_save:
        return 0
    meta = {
        "settings": cem_settings_dict(s),
        "best_generation": best_gen,
        "validation_apples": round(score.apples, 3),
        "validation_steps": round(score.steps, 1),
        "history": history,
    }
    write_weights(WEIGHTS_PATH, weights, meta)
    log = RESULTS_DIR / "snake_eval_train_log.jsonl"
    log.write_text("".join(json.dumps(r) + "\n" for r in history))
    print(f"wrote {WEIGHTS_PATH} and {log}")
    return 0


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    # A bare `python -m snake` (or one that starts with a flag) means `play`.
    if not argv or (argv[0] not in COMMANDS and argv[0] not in ("-h", "--help")):
        argv = ["play", *argv]
    parser = argparse.ArgumentParser(
        prog="python -m snake",
        description="Snake: play it, watch the agents, train the net and the evaluation function, or benchmark them.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("play", help="play yourself (turn by turn, or timed with --delay)")
    p.add_argument("--seed", type=int, default=0, help="food seed, for a repeatable game")
    p.add_argument("--delay", type=float, default=0.0, help="seconds per step; 0 means turn by turn")

    w = sub.add_parser("watch", help="watch an agent play one seeded game")
    w.add_argument("--agent", choices=AGENT_NAMES, default="evolved")
    w.add_argument("--seed", type=int, default=0)
    w.add_argument("--delay", type=float, default=0.1, help="seconds between frames")
    w.add_argument("--steps", type=int, default=None, help="stop after this many moves")

    e = sub.add_parser("evolve", help="run the genetic algorithm and save the champion")
    d = Settings()
    e.add_argument("--pop", type=int, default=d.pop, help="population size")
    e.add_argument("--games", type=int, default=d.games, help="training games per genome")
    e.add_argument("--generations", type=int, default=d.generations)
    e.add_argument("--elites", type=int, default=d.elites, help="genomes copied unchanged each generation")
    e.add_argument("--tournament", type=int, default=d.tournament, help="contestants per parent pick")
    e.add_argument("--mutation-rate", type=float, default=d.mutation_rate)
    e.add_argument("--mutation-sigma", type=float, default=d.mutation_sigma)
    e.add_argument("--seed", type=int, default=d.seed)
    e.add_argument("--workers", type=int, default=d.workers, help="processes scoring genomes in parallel")
    e.add_argument("--starve-penalty", type=float, default=d.starve_penalty,
                   help="fitness lost by a game that ends by starvation")
    e.add_argument("--champion", help="where to write the champion (default: snake/data/champion.json)")
    e.add_argument("--log", help="where to write the per-generation log (default: results/snake_train_log.jsonl)")
    e.add_argument("--checkpoint-dir",
                   help="also write the best-so-far genome here, every --checkpoint-every generations")
    e.add_argument("--checkpoint-every", type=int, default=20)
    e.add_argument("--no-save", action="store_true", help="print the run but leave the committed files alone")

    c = sub.add_parser("cem", help="cross-entropy method over the evaluation function's weights")
    cd = CEMSettings()
    c.add_argument("--pop", type=int, default=cd.pop, help="candidate weight vectors per generation")
    c.add_argument("--games", type=int, default=cd.games, help="training games per candidate")
    c.add_argument("--generations", type=int, default=cd.generations)
    c.add_argument("--elites", type=int, default=cd.elites, help="candidates the Gaussian is refit to")
    c.add_argument("--init-sigma", type=float, default=cd.init_sigma)
    c.add_argument("--min-sigma", type=float, default=cd.min_sigma)
    c.add_argument("--seed", type=int, default=cd.seed)
    c.add_argument("--workers", type=int, default=cd.workers, help="processes scoring candidates in parallel")
    c.add_argument("--no-save", action="store_true", help="print the run but leave the committed files alone")

    b = sub.add_parser("benchmark", help="win and death statistics for every agent on the same boards")
    b.add_argument("--games", type=int, default=DEFAULT_GAMES)
    b.add_argument("--agents", nargs="+", choices=AGENT_NAMES, default=list(AGENT_NAMES))
    b.add_argument("--write", action="store_true", help="also write results/snake_benchmark.{json,md}")

    a = sub.add_parser("actions", help="how often the committed net picks left, straight, and right")
    a.add_argument("--games", type=int, default=40, help="benchmark boards to play (from seed 50000)")
    a.add_argument("--write", action="store_true", help="also write results/snake_net_actions.json")

    args = parser.parse_args(argv)
    if args.command == "play":
        if args.delay > 0:
            play_timed(args.seed, args.delay)
        else:
            play_turns(args.seed)
        return 0
    if args.command == "watch":
        watch(args.agent, args.seed, args.delay, args.steps)
        return 0
    if args.command == "evolve":
        return _evolve_cmd(args)
    if args.command == "cem":
        return _cem_cmd(args)
    if args.command == "benchmark":
        result = run_benchmark(args.games, tuple(args.agents),
                               log=lambda name, s: print(f"{name}: {s['mean_apples']:.2f} mean apples", flush=True))
        print()
        print(to_markdown(result))
        if args.write:
            write_results(result)
            print(f"wrote {RESULTS_DIR / 'snake_benchmark.json'} and .md")
        return 0
    if args.command == "actions":
        result = action_counts(args.games)
        print(json.dumps(result, indent=2))
        if args.write:
            write_action_counts(result)
            print(f"wrote {RESULTS_DIR / 'snake_net_actions.json'}")
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
