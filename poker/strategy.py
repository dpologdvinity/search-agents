"""The trained average strategy as a table, its JSON file, and how the bot and hints read it.

A table maps an information-set key (see `LeducState.infoset`) to {action: probability}. Both seats'
information sets are in one table, since the key already says whose move it is. The file that ships
with the package is `poker/data/leduc_strategy.json`, written by `python -m poker train --save`, so the
server and the CLI never need to train: they look up probabilities.

The table is rounded to 4 decimals on disk. Probabilities are renormalized after loading, so a
rounded row still sums to exactly one, and the exploitability in the file is measured on the rounded
table, which is the one the bot actually plays.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

DATA = Path(__file__).resolve().parent / "data"
LEDUC_STRATEGY = DATA / "leduc_strategy.json"
LEDUC_TRAINING = DATA / "leduc_training.json"
DECIMALS = 4

ACTION_NAMES = {"f": "fold", "k": "check", "c": "call", "b": "bet", "r": "raise"}


def round_table(table: dict[str, dict[str, float]]) -> dict[str, dict[str, float]]:
    """Round every row to DECIMALS and renormalize, so each row sums to exactly one."""
    out = {}
    for key, row in table.items():
        rounded = {a: round(p, DECIMALS) for a, p in row.items()}
        total = sum(rounded.values())
        out[key] = {a: round(p / total, DECIMALS) if total > 0 else 1 / len(row) for a, p in rounded.items()}
    return out


def save(path: Path, meta: dict, table: dict[str, dict[str, float]]) -> None:
    """Write the strategy as compact JSON: metadata, then {key: {action: probability}}."""
    payload = dict(meta, strategy=round_table(table))
    path.write_text(json.dumps(payload, separators=(",", ":")) + "\n")


def load(path: Path = LEDUC_STRATEGY) -> dict:
    """The whole file: metadata and the strategy table, with each row renormalized."""
    raw = json.loads(path.read_text())
    raw["strategy"] = {
        key: {a: p / sum(row.values()) for a, p in row.items()} for key, row in raw["strategy"].items()
    }
    return raw


def sample(row: dict[str, float], rng: random.Random) -> str:
    """Draw one action from a probability row. Dict order is fixed, so a seeded rng repeats exactly."""
    r = rng.random()
    acc = 0.0
    last = None
    for action, p in row.items():
        acc += p
        last = action
        if r < acc:
            return action
    return last  # guards against float round-off leaving acc just under 1


def describe_row(row: dict[str, float]) -> str:
    """A row as text for the CLI, e.g. "bet 67%, check 33%", most likely action first."""
    parts = [f"{ACTION_NAMES[a]} {p:.0%}" for a, p in sorted(row.items(), key=lambda kv: -kv[1]) if p > 0.0005]
    return ", ".join(parts)


def profile(tree, table: dict[str, dict[str, float]]) -> list[list[float]]:
    """Turn a {key: {action: prob}} table into the per-information-set arrays the tree uses."""
    out = []
    for iid, key in enumerate(tree.infosets):
        row = table[key]
        out.append([row.get(a, 0.0) for a in tree.info_actions[iid]])
    return out
