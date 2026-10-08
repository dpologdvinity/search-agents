"""Head-to-head benchmark: the CFR+ bot against the baselines on Leduc, written to results/.

Every matchup plays the same seeded deals, each twice (once per seat), so the luck of the cards
cancels. The CFR bot's mixed strategy is sampled at every decision, so its results are real play, not
an expectation. The self-play row (CFR+ against CFR+) should be about zero: at equilibrium neither side
can win in expectation, which is a check on the whole pipeline.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from . import strategy
from .bots import AlwaysCall, CFRBot, HandStrength, RandomBot, head_to_head
from .leduc import BET_SIZES

RESULTS = Path(__file__).resolve().parent.parent / "results"


def run_benchmark(deals: int, seed: int, table: dict | None = None) -> dict:
    """Run every matchup and return the results as a dict (also what the JSON file holds)."""
    if table is None:
        table = strategy.load()["strategy"]
    cfr = CFRBot(table)
    opponents = [RandomBot(), AlwaysCall(), HandStrength(), CFRBot(table)]
    rows = []
    for i, opp in enumerate(opponents):
        start = time.time()
        r = head_to_head(cfr, opp, deals, seed + i)
        r.seconds = round(time.time() - start, 1)
        rows.append(r)
    names = ["random", "always-call", "hand-strength", "cfr (self-play)"]
    return {
        "game": "leduc",
        "big_blind_chips": BET_SIZES[0],
        "deals_per_matchup": deals,
        "seed": seed,
        "unit": "milli-big-blinds per hand, from the CFR+ bot's side (positive = the bot wins)",
        "matchups": [
            {
                "opponent": names[i],
                "hands": r.hands,
                "mbb_per_hand": round(r.mbb_per_hand, 2),
                "ci95_mbb": [round(x, 2) for x in r.ci95_mbb()],
                "chips_per_hand": round(r.mean_chips, 5),
                "seconds": r.seconds,
            }
            for i, r in enumerate(rows)
        ],
    }


def markdown(result: dict) -> str:
    """The benchmark as a table for the README and results/poker_benchmark.md."""
    lines = [
        f"Leduc hold'em, CFR+ bot against each opponent: {result['deals_per_matchup']:,} seeded deals per "
        f"matchup, each played in both seats ({result['matchups'][0]['hands']:,} hands).",
        "",
        "| Opponent | Hands | mbb/hand (CFR+ side) | 95% CI | chips/hand |",
        "|---|---:|---:|---:|---:|",
    ]
    for m in result["matchups"]:
        lo, hi = m["ci95_mbb"]
        lines.append(f"| {m['opponent']} | {m['hands']:,} | {m['mbb_per_hand']:+.1f} | "
                     f"[{lo:+.1f}, {hi:+.1f}] | {m['chips_per_hand']:+.4f} |")
    return "\n".join(lines) + "\n"


def write_results(result: dict, out_dir: Path = RESULTS) -> None:
    """Write the JSON and the Markdown table under results/."""
    out_dir.mkdir(exist_ok=True)
    (out_dir / "poker_benchmark.json").write_text(json.dumps(result, indent=1) + "\n")
    (out_dir / "poker_benchmark.md").write_text(markdown(result))
