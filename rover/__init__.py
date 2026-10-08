"""Rover in an unknown maze: D* Lite incremental replanning, compared with A* replanned from scratch.

world.py       the true map, seeded generation (mulberry32), the sensor square, ASCII rendering
dstar.py       D* Lite: g/rhs values, a priority queue with km offsets, and repair after each map change
astar.py       the baseline: A* from the rover cell, rerun whenever the sensed map changes
explorer.py    the rover loop: sense, replan both planners on the same changes, then move one cell
benchmark.py   seeded route benchmark: expansions and planning time per planner
"""
