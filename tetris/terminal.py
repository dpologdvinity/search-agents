"""Turn-based Tetris in the terminal, with the agent's placement as a hint.

Input is a line of keys, and each key is one action, so "aa w" moves left twice and rotates:
  a / d   move left / right          w   rotate clockwise (kicks off the wall if needed)
  s       soft drop one row          space  hard drop (locks the piece, ends the turn)
  h       show the agent's choice: the ghost 'o' and how many rotations and which column to use
  q       quit

Nothing falls while you think: each piece waits until you move it. The hint uses the same search as the
AI, with the tuned weights and one-piece lookahead on the preview piece.
"""

from __future__ import annotations

import random
import sys

from .board import H, W, empty_board, fits, lock, render
from .game import PieceBag
from .pieces import ROTATIONS
from .search import best_move
from .tuned import load_tuned

KICKS = (0, -1, 1, -2, 2)  # horizontal offsets tried after a rotation, so a piece can turn against a wall


class Game:
    """The state of one terminal game: board, falling piece, preview, and score."""

    def __init__(self, seed: int, weights):
        self.weights = weights
        self.bag = PieceBag(seed)
        self.board = empty_board()
        self.lines = 0
        self.pieces = 0
        self.over = False
        self.piece = None
        self.rot = 0
        self.x = 0
        self.y = 0
        self._spawn()

    def _spawn(self) -> None:
        self.piece = self.bag.take()
        self.rot = 0
        rot = self.rot_obj()
        self.x = (W - rot.width) // 2
        self.y = H - rot.height  # top row of the box is the top row of the board
        if not fits(self.board, rot, self.x, self.y):
            self.over = True  # the spawn area is blocked
        self.pieces += 1

    def rot_obj(self, rot: int | None = None):
        return ROTATIONS[self.piece][self.rot if rot is None else rot]

    def move(self, dx: int) -> bool:
        if fits(self.board, self.rot_obj(), self.x + dx, self.y):
            self.x += dx
            return True
        return False

    def rotate(self) -> bool:
        nxt = (self.rot + 1) % len(ROTATIONS[self.piece])
        for k in KICKS:
            if fits(self.board, self.rot_obj(nxt), self.x + k, self.y):
                self.rot, self.x = nxt, self.x + k
                return True
        return False

    def soft_drop(self) -> bool:
        if fits(self.board, self.rot_obj(), self.x, self.y - 1):
            self.y -= 1
            return True
        return False

    def hard_drop(self) -> None:
        while self.soft_drop():
            pass
        self._lock()

    def _lock(self) -> None:
        self.board, lines, _, topped = lock(self.board, self.rot_obj(), self.x, self.y)
        self.lines += lines
        if topped:
            self.over = True
            return
        if not self.over:
            self._spawn()

    def hint(self):
        """The agent's choice for the falling piece, as a Move (or None if it has no legal placement)."""
        move, _ = best_move(self.board, self.piece, self.weights, preview=self.bag.peek(), top_k=6)
        return move


def describe_hint(game: Game, move) -> str:
    """How to reach the hint from the current piece: w presses, then a and d steps, then space."""
    if move is None:
        return "the agent has no legal placement"
    turns = (move.rot - game.rot) % len(ROTATIONS[game.piece])
    dx = move.x - game.x
    parts = []
    if turns:
        parts.append(f"w x{turns}")
    if dx:
        parts.append(f"{'d' if dx > 0 else 'a'} x{abs(dx)}")
    parts.append("space")
    return f"agent: {' '.join(parts)}   (it clears {move.lines} line(s))"


def _status(game: Game) -> str:
    return (f"piece {game.piece}   next {game.bag.peek()}   lines {game.lines}   "
            f"pieces {game.pieces}")


def run(seed: int | None = None, weights=None, stdin=None, stdout=None) -> int:
    """Play an interactive game. Returns 0 when the player quits or the game ends."""
    stdin = stdin or sys.stdin
    stdout = stdout or sys.stdout
    weights = weights if weights is not None else load_tuned()["weights"]
    seed = seed if seed is not None else random.randrange(1 << 30)
    game = Game(seed, weights)
    hint = None

    def show(msg: str = "") -> None:
        overlays = [(game.rot_obj(), game.x, game.y, "@")]
        if hint is not None:
            overlays.append((ROTATIONS[hint.piece][hint.rot], hint.x, hint.y, "o"))
        print(render(game.board, overlays), file=stdout)
        print(_status(game), file=stdout)
        if msg:
            print(msg, file=stdout)

    print("Keys: a d move, w rotate, s soft drop, space hard drop, h hint, q quit. Type keys then Enter.",
          file=stdout)
    show()
    while not game.over:
        print("keys> ", end="", file=stdout, flush=True)
        line = stdin.readline()
        if not line:
            break
        msg = ""
        for ch in line.rstrip("\n"):
            if ch == "q":
                print(f"quit after {game.lines} lines in {game.pieces} pieces.", file=stdout)
                return 0
            if ch == "a":
                game.move(-1)
            elif ch == "d":
                game.move(1)
            elif ch == "w":
                game.rotate()
            elif ch == "s":
                game.soft_drop()
            elif ch == " ":
                hint = None
                game.hard_drop()
                if game.over:
                    break
            elif ch == "h":
                hint = game.hint()
                msg = describe_hint(game, hint)
            # Other keys are ignored.
        if game.over:
            break
        show(msg)
    show()
    if game.over:
        print(f"game over: {game.lines} lines in {game.pieces} pieces.", file=stdout)
    else:
        # The loop also ends when stdin closes (a pipe that ran out, or Ctrl-D): that is not a game over.
        print(f"input ended after {game.lines} lines in {game.pieces} pieces.", file=stdout)
    return 0
