"""Terminal play: walk and push with w a s d, undo, restart, hints from the solver, and animated solutions.

The game state is a stack of positions, so undo is a pop. Hints and the solution come from the same
search as the benchmark (A* with the matching heuristic, deadlock pruning on), started from wherever
the player is now, not from the level's start. Pushes are shown in uppercase, walks in lowercase.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field

from .analysis import describe
from .board import LETTERS, Level, apply_move, is_solved, render
from .levels import NAMES, all_levels
from .search import solve

HELP = (
    "w a s d  move (walk or push)     u  undo     r  restart     h  hint (next push)\n"
    "x  solve and animate the rest    l  choose level     q  quit     ?  this help\n"
    "(s is down, so the solver is on x; a run of moves such as dddd works too)"
)


@dataclass
class Game:
    """One level in progress. `history` holds (player, boxes, moves) so undo can step back."""

    level: Level
    player: int
    boxes: set[int]
    moves: str = ""
    history: list = field(default_factory=list)

    @classmethod
    def start(cls, level: Level) -> Game:
        return cls(level=level, player=level.start_player, boxes=set(level.start_boxes))

    @property
    def pushes(self) -> int:
        return sum(1 for ch in self.moves if ch.isupper())

    @property
    def solved(self) -> bool:
        return is_solved(self.level, self.boxes)

    def move(self, letter: str) -> str:
        """Apply a walk or push. Returns "" on success or a short reason it was blocked."""
        result = apply_move(self.level, self.player, self.boxes, letter)
        if result is None:
            return "blocked: a wall or another box is in the way"
        self.history.append((self.player, set(self.boxes), self.moves))
        self.player, self.boxes, pushed = result
        self.moves += letter.upper() if pushed else letter.lower()
        return ""

    def undo(self) -> bool:
        if not self.history:
            return False
        self.player, self.boxes, self.moves = self.history.pop()
        return True

    def restart(self) -> None:
        self.history.clear()
        self.player = self.level.start_player
        self.boxes = set(self.level.start_boxes)
        self.moves = ""

    def plan_from_here(self):
        """A* with the matching heuristic and deadlock pruning, from the current position."""
        return solve(self.level.with_state(self.player, self.boxes), "astar", "matching", True,
                     node_budget=20_000, time_limit=10.0)


def _segment(moves: str) -> str:
    """The first push of a plan, with the walk that leads to it: e.g. "dd then S"."""
    for i, ch in enumerate(moves):
        if ch.isupper():
            walk = moves[:i]
            return (f"walk {walk} then push {ch}" if walk else f"push {ch}")
    return "no push left"


def animate(game: Game, moves: str, out: Callable[[str], None], delay: float) -> None:
    """Replay a move string on the game, printing the board after each push (walks are quick)."""
    for ch in moves:
        if delay:
            time.sleep(delay)
        game.move(ch)
        if ch.isupper() or game.solved:
            out("\n" + "\n".join(render(game.level, game.player, game.boxes)))


def play(level_number: int = 1, read: Callable[[str], str] = input, out: Callable[[str], None] = print,
         delay: float = 0.12) -> int:
    """Interactive game. Returns the number of levels solved. Ends on q or when input runs out."""
    levels = all_levels()
    game = Game.start(levels[level_number - 1])
    solved_count = 0
    out(f"Sokoban: level {level_number}/{len(levels)} '{game.level.name}'. {HELP}")
    while True:
        out("\n" + "\n".join(render(game.level, game.player, game.boxes)))
        analysis = describe(game.level, sorted(game.boxes))
        status = f"pushes {game.pushes}  moves {len(game.moves)}"
        out(status)
        if analysis.deadlock:
            out(f"DEADLOCK ({analysis.deadlock}): no push sequence solves this now. Press u to undo or r to restart.")
        try:
            text = read("> ").strip()
        except EOFError:
            return solved_count
        text = text.lower()
        key = text[:1]
        if key == "q":
            return solved_count
        if text and all(ch in LETTERS for ch in text):
            # One letter, or a run such as "dddd": each step is applied in turn, stopping at the first block.
            for ch in text:
                msg = game.move(ch)
                if msg:
                    out(msg)
                    break
                if game.solved:
                    break
            if game.solved:
                solved_count += 1
                out("\n" + "\n".join(render(game.level, game.player, game.boxes)))
                out(f"Solved '{game.level.name}' in {game.pushes} pushes ({len(game.moves)} moves).")
                return solved_count
            continue
        if key == "u":
            out("" if game.undo() else "nothing to undo")
        elif key == "r":
            game.restart()
        elif key == "h":
            res = game.plan_from_here()
            if not res.solved:
                out(f"no hint: the search stopped with status {res.status}")
            else:
                out(f"hint: {_segment(res.moves)}. The solver needs {res.pushes} pushes from here.")
        elif key == "x":
            res = game.plan_from_here()
            if not res.solved:
                out(f"no solution found: the search stopped with status {res.status}")
            else:
                out(f"solution: {res.moves} ({res.pushes} pushes). Animating.")
                animate(game, res.moves, out, delay)
                if game.solved:
                    solved_count += 1
                    out(f"Solved '{game.level.name}' in {game.pushes} pushes.")
                    return solved_count
        elif key == "l":
            choice = read(f"level 1-{len(levels)}> ").strip()
            if choice.isdigit() and 1 <= int(choice) <= len(levels):
                level_number = int(choice)
                game = Game.start(levels[level_number - 1])
                out(f"level {level_number}: '{game.level.name}'")
            else:
                out("pick a number from the list (see `python -m sokoban levels`)")
        elif key == "?":
            out(HELP)
        elif text:
            out(f"'{text}' is not a command. {HELP}")


def level_list() -> list[str]:
    """The levels as lines for the terminal: number, name, optimal pushes."""
    from .levels import OPTIMAL_PUSHES

    return [f"{i:>2}  {name:<20} {OPTIMAL_PUSHES[name]:>3} pushes" for i, name in enumerate(NAMES, 1)]
