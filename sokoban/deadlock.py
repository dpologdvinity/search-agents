"""Deadlock tests: a position from which no push sequence can solve the level.

Two kinds, from cheap to more expensive:

1. Dead squares (precomputed in tables.py). A box on a dead square can never reach a goal.
2. Frozen boxes. A box is frozen when no push can ever move it. Along each axis, a box needs both
   neighbours free to be pushed (one to stand on, one to slide into). If a wall or another frozen
   box sits on one side of it along both axes, it is stuck for good. A frozen box that is not on a
   goal is a deadlock. A 2x2 block of boxes and walls with a box off-goal is the classic case.

The frozen set is the greatest fixed point: start by assuming every box is frozen, then drop any box
that has a free side along some axis, and repeat until nothing changes. Dropping is always justified
(a box with a free side on an axis can move there if the player can get to the other side), and
every box left in the set really is stuck: its first move would need a blocking neighbour to move
first, and by induction none of them ever does. So the test is sound (it never flags a solvable
position) even though it ignores the player.
"""

from __future__ import annotations

from .board import Level


def frozen_boxes(level: Level, boxes: set[int] | frozenset[int]) -> set[int]:
    """The boxes that can never move again, by the greatest fixed point described above."""
    floor = level.floor
    width = level.width
    frozen = set(boxes)
    changed = True
    while changed:
        changed = False
        for b in list(frozen):
            # A side is blocked when it is a wall or a box that is itself frozen.
            left_blocked = not floor[b - 1] or (b - 1) in frozen
            right_blocked = not floor[b + 1] or (b + 1) in frozen
            up_blocked = not floor[b - width] or (b - width) in frozen
            down_blocked = not floor[b + width] or (b + width) in frozen
            if not ((left_blocked or right_blocked) and (up_blocked or down_blocked)):
                frozen.discard(b)
                changed = True
    return frozen


def is_deadlock(level: Level, dead: bytes, boxes: set[int] | frozenset[int]) -> bool:
    """True when some box is on a dead square, or a frozen box is off every goal."""
    for b in boxes:
        if dead[b]:
            return True
    goals = set(level.goals)
    return any(b not in goals for b in frozen_boxes(level, boxes))
