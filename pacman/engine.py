"""Pac-Man rules: a turn-based game whose outcome depends only on the maze, the seed, and the moves.

Each turn Pac-Man picks one open neighbour and every ghost then moves one step:

 1. Pac-Man moves. Pellet points are awarded. A power pellet scares every ghost for
    SCARED_TURNS turns.
 2. Any ghost already on Pac-Man's new cell is collided with. A scared ghost is eaten
    (points, and it returns to its home cell). A normal ghost kills Pac-Man. Because
    ghosts always move, this check also catches a ghost that swaps places with him.
 3. If no pellets remain, Pac-Man has won and the ghosts do not move.
 4. Each ghost chooses a move from the same snapshot, then all of them move together. With
    the "ai" policy (the default) that is the A* route (ghosts.choose_move). With the
    "chance" policy it is a draw from fixed odds that ignore the board (ghosts.chance_move).
 5. Collision check again for ghosts that stepped onto Pac-Man's cell.
 6. Scared timers count down, the turn counter advances, and the game times out
    after max_turns.

`Game.step` records everything a viewer needs (positions before and after, each ghost's
target and route, the points and events) in a Turn. Random tie-breaks use `Game.rng`,
so the same seed and the same moves always produce the same game.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, replace

from .ghosts import GHOST_POLICIES, NO_HEADING, Ghost, chance_move, choose_move, personality_for
from .mazes import NO_CELL, Maze, get
from .search import distances

PLAYING, WON, LOST, TIMEOUT = "playing", "won", "lost", "timeout"
NO_DIRECTION = -1

PELLET_POINTS = 10
POWER_POINTS = 50
GHOST_POINTS = 200
WIN_POINTS = 500
SCARED_TURNS = 8
MAX_TURNS = 300

# Learning-only shaping (the score shown to players does not include these):
STEP_COST = 1.0        # small per-turn cost, so a quicker clear is preferred
DEATH_PENALTY = 300.0  # a lost life must outweigh the points available before it


@dataclass(frozen=True)
class State:
    """A snapshot of the game. Immutable: `Game.step` builds a new State each turn."""

    maze: Maze
    pac: int
    ghosts: tuple[Ghost, ...]
    pellets: frozenset[int]
    points: int = 0
    turn: int = 0
    facing: int = NO_DIRECTION  # action index of Pac-Man's last move
    status: str = PLAYING

    @property
    def over(self) -> bool:
        return self.status != PLAYING


@dataclass(frozen=True)
class Plan:
    """What one ghost planned on a turn: its target (None while fleeing) and the route it took."""

    ghost: int
    target: int | None
    path: tuple[int, ...]


@dataclass(frozen=True)
class Turn:
    """One complete turn: the state before and after, the ghosts' plans, and what happened."""

    action: int
    before: State
    after: State
    plans: tuple[Plan, ...]
    points: int
    events: tuple[str, ...]  # any of: pellet, power, ghost, death, win, timeout
    eaten: int | None  # the pellet cell eaten this turn, if any


def initial_state(maze: Maze) -> State:
    ghosts = tuple(
        Ghost(pos=start, personality=personality_for(i), home=start) for i, start in enumerate(maze.ghost_starts)
    )
    return State(maze=maze, pac=maze.pac_start, ghosts=ghosts, pellets=maze.pellets)


def legal_actions(state: State) -> list[int]:
    """Action indices of the moves that do not run into a wall."""
    return [a for a, nb in enumerate(state.maze.nbr[state.pac]) if nb != NO_CELL]


def learning_reward(turn: Turn) -> float:
    """The reward the agent learns from: points, minus a small per-turn cost, minus a large penalty for dying."""
    reward = turn.points - STEP_COST
    if "death" in turn.events:
        reward -= DEATH_PENALTY
    return reward


def _meet(ghosts: tuple[Ghost, ...], cell: int) -> tuple[tuple[Ghost, ...], int, bool, list[str]]:
    """Resolve ghosts standing on `cell`. Returns (ghosts, points gained, Pac-Man died, events)."""
    gained, died, events = 0, False, []
    out = []
    for g in ghosts:
        if g.pos != cell:
            out.append(g)
        elif g.scared:
            out.append(replace(g, pos=g.home, scared=0, heading=NO_HEADING))  # eaten: back home, no heading
            gained += GHOST_POINTS
            events.append("ghost")
        else:
            died = True
            out.append(g)
    return tuple(out), gained, died, events


class Game:
    """One game of Pac-Man on a named maze, deterministic for a given seed.

    `ghosts` picks the ghost policy: "ai" (A* routes, the default and the one the agents were
    trained on) or "chance" (every ghost moves by the fixed odds in ghosts.CHANCE_ODDS).
    """

    def __init__(self, maze: Maze | str, seed: int = 0, max_turns: int = MAX_TURNS, ghosts: str = "ai"):
        if ghosts not in GHOST_POLICIES:
            raise ValueError(f"unknown ghost policy {ghosts!r}; choose from {', '.join(GHOST_POLICIES)}")
        self.maze = get(maze) if isinstance(maze, str) else maze
        self.seed = seed
        self.max_turns = max_turns
        self.ghost_policy = ghosts
        self.rng = random.Random(seed)
        self.state = initial_state(self.maze)

    def legal_actions(self) -> list[int]:
        return legal_actions(self.state)

    def step(self, action: int) -> Turn:
        """Play one turn with `action` (an index into ACTIONS) and return what happened."""
        before = self.state
        if before.over:
            raise RuntimeError("the game is over")
        if action not in legal_actions(before):
            raise ValueError(f"action {action} runs into a wall")
        maze = before.maze
        pac = maze.nbr[before.pac][action]
        points, events = 0, []
        eaten = None
        pellets = before.pellets
        if pac in pellets:
            pellets = pellets - {pac}
            eaten = pac
            if pac in maze.powers:
                points += POWER_POINTS
                events.append("power")
            else:
                points += PELLET_POINTS
                events.append("pellet")

        ghosts = before.ghosts
        if pac in maze.powers:  # the power pellet frightens every ghost, eaten or not
            ghosts = tuple(replace(g, scared=SCARED_TURNS) for g in ghosts)

        # Steps 2 and 3: collisions with ghosts already here, then the win check.
        ghosts, gained, died, hit = _meet(ghosts, pac)
        points += gained
        events += hit
        if died:
            return self._finish(before, action, pac, ghosts, pellets, points, events, LOST, eaten, ())
        if not pellets:
            return self._finish(before, action, pac, ghosts, pellets, points + WIN_POINTS, events, WON, eaten, ())

        # Step 4: every ghost picks a move from the same snapshot, then they all move.
        # Under the "ai" policy, scared ghosts flee from the BFS distances out of Pac-Man's new cell.
        # Under the "chance" policy, every ghost draws its move from the fixed odds instead, and
        # has no route to show (an empty path).
        pac_dist = distances(maze, pac) if any(g.scared for g in ghosts) else None
        moved, plans = [], []
        for i, g in enumerate(ghosts):
            if self.ghost_policy == "chance":
                nxt, heading = chance_move(maze, g, self.rng)
                target, route = None, ()
            else:
                nxt, target, route = choose_move(maze, g, i, pac, action, self.rng, pac_dist or [])
                heading = g.heading  # A* ghosts never read a heading; leaving it alone keeps their states unchanged
            # Chance ghosts record the direction of this move, so their next draw can keep going.
            moved.append(replace(g, pos=nxt, heading=heading))
            plans.append(Plan(ghost=i, target=target, path=route))

        # Step 5: collisions with ghosts that just stepped onto Pac-Man's cell.
        ghosts, gained, died, hit = _meet(tuple(moved), pac)
        points += gained
        events += hit
        if died:
            return self._finish(before, action, pac, ghosts, pellets, points, events, LOST, eaten, tuple(plans))
        # Step 6: fright wears off.
        ghosts = tuple(replace(g, scared=max(0, g.scared - 1)) for g in ghosts)
        status = TIMEOUT if before.turn + 1 >= self.max_turns else PLAYING
        if status == TIMEOUT:
            events.append("timeout")
        return self._finish(before, action, pac, ghosts, pellets, points, events, status, eaten, tuple(plans))

    def _finish(self, before, action, pac, ghosts, pellets, points, events, status, eaten, plans) -> Turn:
        after = State(
            maze=before.maze,
            pac=pac,
            ghosts=ghosts,
            pellets=pellets,
            points=before.points + points,
            turn=before.turn + 1,
            facing=action,
            status=status,
        )
        if status == LOST:
            events.append("death")
        elif status == WON:
            events.append("win")
        self.state = after
        return Turn(
            action=action,
            before=before,
            after=after,
            plans=plans,
            points=points,
            events=tuple(events),
            eaten=eaten,
        )
