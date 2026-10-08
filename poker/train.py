"""Training runs: build a game's tree, run CFR or CFR+, measure exploitability on a schedule, and save.

`python -m poker train` calls `run` and then `save_leduc`. The committed files are:
  poker/data/leduc_strategy.json   the average strategy the bot and the hints read
  poker/data/leduc_training.json   exploitability against iterations for CFR and CFR+ (the page's chart)
  results/poker_train_log.jsonl    the same curves, one JSON object per measured point
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

from . import kuhn, leduc, strategy
from .cfr import Solver, train
from .cfr import table as average_table
from .tree import GameTree, build_tree, expected_value, exploitability

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
GAME_ROOTS = {"kuhn": kuhn.root, "leduc": leduc.root}
KUHN_VALUE = -1 / 18  # game value for the first player at equilibrium (Kuhn, 1950)

_trees: dict[str, GameTree] = {}


def tree_for(game: str) -> GameTree:
    """The game's tree, built once per process."""
    if game not in _trees:
        _trees[game] = build_tree(GAME_ROOTS[game]())
    return _trees[game]


@dataclass
class Run:
    game: str
    algorithm: str
    iterations: int
    points: list[tuple[int, float]] = field(default_factory=list)  # (iteration, exploitability)
    value: float = 0.0          # seat 0's expected payoff under the average strategy
    solver: Solver | None = None
    seconds: float = 0.0

    def as_dict(self) -> dict:
        return {
            "game": self.game,
            "algorithm": self.algorithm,
            "iterations": self.iterations,
            "exploitability_points": [[t, round(e, 8)] for t, e in self.points],
            "final_exploitability": round(self.points[-1][1], 8),
            "value_seat0": round(self.value, 8),
        }


def run(game: str, algorithm: str, iterations: int, progress=None) -> Run:
    """Train `algorithm` ("cfr" or "cfr+") for `iterations` on `game`. `progress(t, expl)` is called per checkpoint."""
    tree = tree_for(game)
    result = Run(game, algorithm, iterations)
    start = time.time()

    def on_point(t, expl):
        result.points.append((t, expl))
        if progress is not None:
            progress(t, expl)

    solver = train(tree, algorithm, iterations, on_point=on_point)
    result.solver = solver
    result.value = expected_value(tree, solver.average())
    result.seconds = time.time() - start
    return result


def kuhn_check(iterations: int = 10_000) -> dict:
    """Train both algorithms on Kuhn and compare the value with the known -1/18."""
    out = {}
    for algo in ("cfr", "cfr+"):
        r = run("kuhn", algo, iterations)
        out[algo] = {"value": r.value, "exact": KUHN_VALUE, "value_error": abs(r.value - KUHN_VALUE),
                     "exploitability": r.points[-1][1]}
    return out


def save_leduc(r: Run, strategy_path: Path, training_path: Path, log_path: Path | None,
               runs: list[Run], extra: dict | None = None) -> dict:
    """Write the strategy file for a Leduc CFR+ run, the training curves, and the JSONL log.

    The exploitability stored with the strategy is measured on the rounded table that the file holds,
    not on the float solver state, so the number matches what the bot plays.
    """
    tree = tree_for("leduc")
    table = strategy.round_table(average_table(tree, r.solver.average()))
    sigma = strategy.profile(tree, table)
    expl, br0, br1 = exploitability(tree, sigma)
    value = expected_value(tree, sigma)
    meta = {
        "game": "leduc",
        "algorithm": r.algorithm.upper(),
        "iterations": r.iterations,
        "seconds": round(r.seconds, 1),
        "big_blind": leduc.BET_SIZES[0],
        "infosets": len(tree.infosets),
        "value_seat0": round(value, 6),
        "exploitability": round(expl, 8),
        "best_response_seat0": round(br0, 8),
        "best_response_seat1": round(br1, 8),
    }
    strategy.save(strategy_path, meta, table)
    curves = training_curves(runs)
    curves.update(extra or {})
    training_path.write_text(json.dumps(curves, separators=(",", ":")) + "\n")
    if log_path is not None:
        with log_path.open("w") as f:
            for run_ in runs:
                for t, e in run_.points:
                    f.write(json.dumps({"game": run_.game, "algorithm": run_.algorithm, "iteration": t,
                                        "exploitability": e}) + "\n")
    return meta


def training_curves(runs: list[Run]) -> dict:
    """The JSON the page charts: each algorithm's exploitability at each checkpoint, plus the Kuhn check."""
    return {
        "game": "leduc",
        "runs": [
            {"algorithm": r.algorithm.upper(), "iterations": [t for t, _ in r.points],
             "exploitability": [round(e, 8) for _, e in r.points]}
            for r in runs
        ],
    }

