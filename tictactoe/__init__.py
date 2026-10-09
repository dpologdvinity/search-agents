"""Tic-tac-toe with a minimax agent that uses alpha-beta pruning.

The game is small enough to search exhaustively, so the agent is unbeatable: perfect play from the
empty board is a draw, and the agent never loses. This package is the Python reference for the
404 page's game (web/js/tictactoe-core.js). Both implement the same search in the same move order,
and tests/web/tictactoe.test.mjs checks that they agree on every reachable position.
"""

from .core import Search, legal_moves, minimax, reachable_positions, search, winner

__all__ = ["Search", "legal_moves", "minimax", "reachable_positions", "search", "winner"]
