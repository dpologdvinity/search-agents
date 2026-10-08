"""The built-in levels, in order of difficulty (fewest pushes in the optimal solution).

Levels 1 to 7 are drawn by hand. Levels 8 to 12 are generated: the generator starts from a goal
layout, applies random reverse pushes (each undoes a forward push, so the scramble is solvable by
construction), and keeps the scramble with the most optimal pushes among 40 seeds that A* with the
matching heuristic solves. Every level here is solved by the test suite, so none can be unsolvable.

The push counts below are the optimal ones (A* with the matching heuristic); the tests check them.
"""

from __future__ import annotations

from functools import cache

from .board import Level, parse_level

# (name, optimal pushes, text). The text uses the symbols in board.py.
LEVEL_TEXT = (
    ("Warm-up", 1, """
#####
#@$.#
#####
"""),
    ("Around the corner", 3, """
#######
#     #
# $   #
#   . #
#  @  #
#######
"""),
    ("Pillar", 3, """
########
#  .   #
# #$#  #
#  $ . #
#  @   #
#      #
########
"""),
    ("Two in a row", 4, """
########
#  ..  #
# $$ @ #
#      #
########
"""),
    ("Side step", 6, """
#########
#  .#   #
#  $ $  #
#  #@   #
#   .   #
#########
"""),
    ("Crossroads", 8, """
#########
#   .   #
# $ # $ #
#   @   #
# $ # . #
#   .   #
#########
"""),
    ("Open room", 9, """
#######
#     #
# $ $ #
#  @  #
# $ . #
#.  . #
#######
"""),
    ("Stacks", 11, """
########
# $.   #
# $  $ #
# #  # #
#   .  #
#  #   #
#  .@  #
########
"""),
    ("Forklift", 13, """
##########
#  .  #  .#
#     $  ##
# $#  # $ #
# $.   # @#
# #    .  #
#   #      #
##########
"""),
    ("Shelf", 15, """
#########
#.  #  @#
# $ #$$ #
#       #
#.  # * #
#  #    #
#  .    #
#########
"""),
    ("Aisle", 18, """
###########
#  .  #  .#
# $      .#
#  #  #   #
# $#$$#  ##
#@ .      #
###########
"""),
    ("Dock", 19, """
  ########
  #      #
###  ##  ###
# $  .  .   #
#@ ##   ## *#
# $         #
#  ##   ## *#
# $  .  .   #
### $##  ###
  #      #
  ########
"""),
)

NAMES = tuple(name for name, _, _ in LEVEL_TEXT)
OPTIMAL_PUSHES = {name: pushes for name, pushes, _ in LEVEL_TEXT}


@cache
def all_levels() -> tuple[Level, ...]:
    """Every built-in level, parsed. Index 0 is level 1."""
    return tuple(parse_level(text.strip("\n"), name) for name, _, text in LEVEL_TEXT)


def get_level(number: int) -> Level:
    """Level by its 1-based number. Raises ValueError for a number outside the set."""
    levels = all_levels()
    if not 1 <= number <= len(levels):
        raise ValueError(f"there are {len(levels)} levels; pick 1 to {len(levels)}")
    return levels[number - 1]
