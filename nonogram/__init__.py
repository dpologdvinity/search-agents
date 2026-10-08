"""Nonograms (picture logic puzzles), solved three ways: line solving, a hybrid search, and SAT.

Modules, in the order a reader meets them:
  automaton  a clue as a finite automaton over 0 (empty) and 1 (filled)
  lines      one line's forced cells, by forward and backward reachability over positions
  puzzle     clues, grids, and the terminal renderer
  solvers    the method drivers, the uniqueness check, and the hint
  cnf        the whole puzzle as CNF (the automaton encoding)
  dpll       the DPLL solver (watched literals, pure literals, VSIDS-lite, phase saving)
  generate   random pictures that have exactly one solution
  library    twelve hand-drawn pictures
  benchmark  the measurements in results/nonogram_benchmark.md
"""

from .puzzle import Puzzle, render
from .solvers import hint, solve

__all__ = ["Puzzle", "render", "solve", "hint"]
