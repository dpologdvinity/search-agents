"""One match: an agent fires at a fleet until every ship is sunk.

This is the loop the benchmark and the watch mode share. The fleet answers each shot, the agent's knowledge
records the answer, and the agent chooses again from that knowledge alone.
"""

from __future__ import annotations

from .agents import Agent
from .board import Fleet, Knowledge


def play_out(agent: Agent, ships) -> int:
    """Shots the agent needs to sink every ship in `ships`. The agent never sees the fleet, only the answers."""
    fleet = Fleet(ships)
    k = Knowledge()
    while not fleet.all_sunk:
        cell = agent.choose(k)
        shot = fleet.fire(cell)
        k.observe(cell, shot.result, shot.ship)
    return len(fleet.fired)
