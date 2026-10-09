"""Terminal front end: ASCII rendering, human play, and an animated watch mode.

Legend: # wall, . pellet, o power pellet, P Pac-Man, 1-4 ghosts (~ while they are scared).
The human plays with w/a/s/d and Enter. A move into a wall is refused without using a turn.
`ghosts` picks the opponent: "ai" (A* routes, the default) or "chance" (fixed odds).
"""

from __future__ import annotations

import time
from collections.abc import Callable

from .agents import make_agent
from .engine import Game, State, Turn, legal_actions
from .episode import run_episode
from .ghosts import GHOST_LABELS
from .mazes import ACTIONS, Maze, get

KEYS = {"w": "N", "d": "E", "s": "S", "a": "W"}
CLEAR = "\033[H\033[2J"  # ANSI: move the cursor home and clear the screen


def render(state: State) -> str:
    """The board as text, with a status line underneath."""
    maze = state.maze
    chars = []
    for cell in range(maze.cells):
        if not maze.is_open[cell]:
            chars.append("#")
        elif cell in state.pellets:
            chars.append("o" if cell in maze.powers else ".")
        else:
            chars.append(" ")
    for i, g in enumerate(state.ghosts):
        chars[g.pos] = "~" if g.scared else str((i + 1) % 10)
    chars[state.pac] = "P"
    rows = ["".join(chars[r * maze.width:(r + 1) * maze.width]) for r in range(maze.height)]
    status = f"{maze.title}  turn {state.turn}  score {state.points}  pellets left {len(state.pellets)}"
    if state.over:
        status += f"  [{state.status.upper()}]"
    return "\n".join(rows) + "\n" + status


def describe(turn: Turn) -> str:
    """One line of what happened this turn, empty if nothing noteworthy did."""
    words = {
        "power": "power pellet: ghosts turn scared",
        "ghost": "ate a ghost",
        "death": "caught by a ghost",
        "win": "all pellets eaten",
        "timeout": "out of turns",
    }
    return "; ".join(words[e] for e in turn.events if e in words)


def play_human(maze: Maze | str, seed: int, read: Callable[[str], str] = input,
               out: Callable[[str], None] = print, ghosts: str = "ai") -> str:
    """One interactive game against the chosen ghost policy. Returns the final status, or "quit"."""
    game = Game(maze, seed, ghosts=ghosts)
    while not game.state.over:
        out("\n" + render(game.state))
        text = read("move (w/a/s/d, q to quit): ").strip().lower()
        if text == "q":
            return "quit"
        if text not in KEYS:
            out("  use w (up), a (left), s (down), d (right)")
            continue
        action = ACTIONS.index(KEYS[text])
        if action not in legal_actions(game.state):
            out("  a wall is in the way")
            continue
        turn = game.step(action)
        if describe(turn):
            out(f"  {describe(turn)}")
    out("\n" + render(game.state))
    return game.state.status


def watch(agent_name: str, maze: Maze | str, seed: int, delay: float = 0.25,
          out: Callable[[str], None] = print, sleep: Callable[[float], None] = time.sleep,
          clear: bool = True, ghosts: str = "ai") -> str:
    """Play `agent_name` on `maze` and redraw the board after every turn. Returns the final status."""
    m = get(maze) if isinstance(maze, str) else maze
    agent = make_agent(agent_name)
    episode = run_episode(agent, m, seed, agent_name, ghosts=ghosts)
    for turn, decision in zip(episode.turns, episode.decisions, strict=True):
        lines = [render(turn.after), f"{agent_name} plays {ACTIONS[turn.action]}"]
        if decision.values:
            lines.append("Q: " + "  ".join(f"{ACTIONS[a]} {v:.1f}" for a, v in decision.values.items()))
        if describe(turn):
            lines.append(describe(turn))
        out((CLEAR if clear else "\n") + "\n".join(lines))
        sleep(delay)
    final = episode.final
    out(f"{agent_name} vs {GHOST_LABELS[ghosts]}: {final.status}, score {final.points}, {final.turn} turns")
    return final.status
