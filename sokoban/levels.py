"""The built-in levels, in order of difficulty (fewest pushes in the optimal solution).

All twelve levels are fixed in this file: the boards are written out below, not generated at run time.
Each optimal push count is what the solver finds (A* with the matching heuristic), and the tests check
that the solver still reaches it. The tests also solve every level, so none can be unsolvable.
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
