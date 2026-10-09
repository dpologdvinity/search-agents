# Endgame: tablebase against the chance opponent

Command (from the repo root, with `PYTHONPATH=.`):

    python -m endgame match --games 200 --piece both --seed 0 --json > results/endgame_chance_match.json

Same numbers as text: drop `--json`. Each position is a random winning position for the side with the piece,
played twice: once with the tablebase as the strong side (White, with the piece) and once with it as the lone
black king. Position i and the chance rolls both use seed + i, so the run is repeatable.

- Tablebase (exact): plays its best move from the solved table.
- Chance (fixed odds): draws each legal move by its kind's weight (captures 4, checks 2, king steps toward the
  centre 2, other moves 1), renormalised over the legal moves. No search, no table lookup.

| Piece | Tablebase seat | Chance seat | Tablebase won | Drawn | Lost | Mean length of the tablebase's wins |
|-------|----------------|-------------|---------------|-------|------|-------------------------------------|
| KQK   | White (strong) | Black       | 200 / 200     | 0     | 0    | 4.95 moves                          |
| KQK   | Black (weak)   | White       | 0 / 200       | 199   | 1    | (none); the one loss lasted 1 move  |
| KRK   | White (strong) | Black       | 200 / 200     | 0     | 0    | 6.66 moves                          |
| KRK   | Black (weak)   | White       | 0 / 200       | 200   | 0    | (none)                              |

Reading the table: with the piece, the tablebase mates every time. Its wins average 4.95 moves in KQK and 6.66
in KRK (the longest forced mates are 10 and 16 moves), because the table forces mate whatever the defence.
As the lone king, the tablebase survives: the chance side hangs its piece often, and the lone king takes it,
which is a draw. In the traces inspected while building this run, every draw was that capture; the report does
not count draws by reason. The one KQK loss is a mate in one that the random side happened to play.
