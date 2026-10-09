"""Play one full game with an agent and keep every turn, for the web page, the benchmark, and the tests.

The agent gets its own random generator, derived from the seed but separate from the
game's, so changing the agent never changes how the ghosts move. The ghost policy ("ai" or
"chance") is passed through to the game and recorded on the episode.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from .agents import Decision
from .engine import MAX_TURNS, Game, State, Turn, legal_actions
from .mazes import Maze


@dataclass(frozen=True)
class Episode:
    maze: Maze
    seed: int
    agent: str
    turns: tuple[Turn, ...]
    decisions: tuple[Decision, ...]  # one per turn, in order
    final: State
    ghosts: str = "ai"  # the ghost policy the game was played under

    @property
    def won(self) -> bool:
        return self.final.status == "won"

    @property
    def score(self) -> int:
        return self.final.points


def agent_rng(seed: int) -> random.Random:
    """The agent's generator for a game seeded with `seed`. Offset so it differs from the game's own RNG."""
    return random.Random(seed * 7919 + 1)


def run_episode(agent, maze: Maze, seed: int, name: str = "", max_turns: int = MAX_TURNS,
                ghosts: str = "ai") -> Episode:
    """Play `agent` on `maze` with ghost randomness from `seed` until the game ends.

    `agent` is any object with act(state, rng) -> Decision; `name` is recorded for display.
    `ghosts` picks the ghost policy: "ai" (A* routes) or "chance" (fixed odds).
    """
    game = Game(maze, seed=seed, max_turns=max_turns, ghosts=ghosts)
    rng = agent_rng(seed)
    turns: list[Turn] = []
    decisions: list[Decision] = []
    while not game.state.over:
        decision = agent.act(game.state, rng)
        if decision.action not in legal_actions(game.state):  # guards against a buggy agent
            raise RuntimeError(f"agent {name!r} chose an illegal action")
        decisions.append(decision)
        turns.append(game.step(decision.action))
    return Episode(
        maze=maze,
        seed=seed,
        agent=name or getattr(agent, "name", "agent"),
        turns=tuple(turns),
        decisions=tuple(decisions),
        final=game.state,
        ghosts=game.ghost_policy,
    )
