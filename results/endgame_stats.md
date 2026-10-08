# Endgame tablebases: KQK and KRK

Produced by `python -m endgame build` (solve, then check every entry) and `python -m endgame stats`.
Raw numbers: `results/endgame_stats.json`. Distances are in plies (half-moves) for the table; the tables below
use moves, where a win in d plies is mate in (d + 1) / 2 moves for the side to move.

## Positions

| | KQK | KRK |
|---|---:|---:|
| Legal positions | 368,452 | 399,112 |
| White to move (all won) | 144,508 | 175,168 |
| Black to move: lost | 200,896 | 201,700 |
| Black to move: drawn | 23,048 | 22,244 |
| Longest forced mate (side with the piece) | 10 moves (19 plies) | 16 moves (31 plies) |
| Longest defence (lone king) | 10 moves (20 plies) | 16 moves (32 plies) |
| Solve time, first run on this laptop | 4.9 s | 4.4 s |
| Solve time, the run that wrote the committed files | 25.0 s | 34.3 s |
| Entries checked against the rules | 368,452 of 368,452, 0 problems | 399,112 of 399,112, 0 problems |
| Data file | 145 KB (`endgame/data/kqk.npz`) | 153 KB (`endgame/data/krk.npz`) |

The solve times differ by about 6x between runs because the laptop was shared (load average 15 to 27 during
the later run). The solver is the same code in both runs.

Longest mates: KQK `8/8/8/5k2/8/8/1Q6/K7 w - - 0 1` and KRK `8/8/8/8/8/2k5/1R6/K7 w - - 0 1`, both with White to move.

## Distance-to-mate histogram

Wins are for the side to move; losses are for the side to move being mated (0 = already checkmated).

| Moves | KQK win | KQK mated | KRK win | KRK mated |
|---:|---:|---:|---:|---:|
| 0 | - | 364 | - | 216 |
| 1 | 2,448 | 1,352 | 1,512 | 624 |
| 2 | 5,012 | 2,956 | 4,676 | 1,948 |
| 3 | 9,064 | 7,480 | 3,852 | 648 |
| 4 | 19,964 | 14,144 | 1,900 | 1,584 |
| 5 | 26,164 | 25,484 | 4,848 | 3,768 |
| 6 | 32,064 | 39,908 | 8,708 | 4,728 |
| 7 | 32,104 | 54,052 | 11,320 | 5,444 |
| 8 | 15,000 | 43,800 | 17,172 | 11,448 |
| 9 | 2,680 | 11,300 | 20,088 | 13,672 |
| 10 | 8 | 56 | 19,016 | 15,872 |
| 11 | - | - | 20,476 | 22,788 |
| 12 | - | - | 21,480 | 28,732 |
| 13 | - | - | 17,824 | 33,516 |
| 14 | - | - | 16,136 | 36,372 |
| 15 | - | - | 5,244 | 17,284 |
| 16 | - | - | 916 | 3,056 |

Both columns sum to the win and loss totals above. The KQK and KRK maxima (10 and 16 moves) are the known
values for these endgames.

## Method

Retrograde analysis: checkmates are seeded at distance 0, and each decided position is walked back to its
predecessors. A predecessor where the side to move can move into a loss is a win one ply deeper; a predecessor
whose moves all lose is a loss, decided by its last-resolved move. Positions never decided are draws.
The stored value is the distance to mate in plies. Captures leave king against king, so they are draws and
never decrement a counter. The 50-move rule is not applied (pure distance to mate).

## Verification

`python -m endgame build` re-checks every legal position against the definition of distance to mate: a
win must have a move to a loss one ply shallower and no faster winning move; a loss must have only winning
replies, with the slowest at one ply less; a draw must have no winning move. This check does not use the
solver. As a sanity test of the checker, 20 entries were corrupted in a copy of the table and 51 problems
were reported. The predecessor generator was checked exhaustively during development to be the exact inverse
of the move generator: 4,869,496 quiet edges in each direction for KQK, with no mismatches. The unit tests
repeat the inverse check on random positions.
