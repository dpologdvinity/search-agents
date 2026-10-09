# Poker match: CFR+ against Chance (fixed odds)

Reproduce from the repo root with:

```
PYTHONPATH=. python -m poker match --opponent chance --hands 200000 --seed 1
```

Seeded: the deals, the CFR+ bot's sampled actions, and the chance player's sampled actions all come from seed 1.
Each deal is played in both seats. The timing line depends on the machine (this run: 22.2 s on a shared 3-core slice).

```
Poker match, seed 1: CFR+ average strategy (cfr) vs Chance (fixed odds, ignores its card) (chance)
  200,000 hands (100,000 deals, each played in both seats), 22.2 s
  CFR+ side: +358.0 mbb/hand, 95% CI [+350.4, +365.6]
  CFR+ side: +0.7160 chips/hand, +143210.0 chips over the match
```

Chance's fixed odds are fold 15%, call or check 55%, bet or raise 30%, renormalised over the legal actions.
The CFR+ side wins about 0.36 big blinds per hand against chance. The random row in `results/poker_benchmark.md`
uses the same seed and deals, but its opponent draws different actions, so it is a different measurement
(at 20,000 deals the two differ by about 1 mbb).
