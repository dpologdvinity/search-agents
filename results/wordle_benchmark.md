# Wordle benchmark

All 3568 answers, every allowed guess (6748 words) available to each strategy.
Guess counts include the winning guess; 7 marks a game not solved in 6.

| strategy | mean guesses | 1 | 2 | 3 | 4 | 5 | 6 | fail (>6) | failure rate | seconds |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| entropy | 3.627 | 0 | 59 | 1513 | 1710 | 272 | 14 | 0 | 0.00% | 241.8 |
| minimax | 3.739 | 0 | 50 | 1223 | 1927 | 344 | 24 | 0 | 0.00% | 332.6 |
| random | 4.412 | 1 | 94 | 699 | 1274 | 886 | 368 | 246 | 6.89% | 0.8 |

## Best opening word (all answers still possible)

| rank | word | expected bits | worst bucket | expected left | in answers |
|---:|---|---:|---:|---:|---|
| 1 | tares | 6.230 | 251 | 78.7 | no |
| 2 | lares | 6.165 | 211 | 74.9 | no |
| 3 | rales | 6.146 | 211 | 75.0 | no |
| 4 | tales | 6.138 | 213 | 80.4 | yes |
| 5 | rates | 6.122 | 251 | 81.3 | yes |
| 6 | tears | 6.075 | 251 | 90.2 | yes |
| 7 | tries | 6.054 | 281 | 90.9 | yes |
| 8 | tires | 6.048 | 281 | 91.7 | yes |
| 9 | reals | 6.046 | 211 | 84.2 | yes |
| 10 | slate | 6.033 | 245 | 91.9 | yes |
