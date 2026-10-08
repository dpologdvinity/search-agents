Hex 7x7, 400 simulations per move, 40 seeded games per matchup (colours alternate). Win rate is for the first agent, with a 95% Wilson interval.

| matchup | first agent win rate (95% CI) | wins as DOWN | wins as ACROSS | mean moves |
|---|---|---|---|---|
| rave vs uct | 100.0% (91.2%-100.0%) | 20/20 | 20/20 | 15.9 |
| rave vs shortest | 97.5% (87.1%-99.6%) | 19/20 | 20/20 | 16.8 |
| uct vs shortest | 32.5% (20.1%-48.0%) | 6/20 | 7/20 | 18.7 |
| rave vs random | 100.0% (91.2%-100.0%) | 20/20 | 20/20 | 14.8 |
| uct vs random | 100.0% (91.2%-100.0%) | 20/20 | 20/20 | 25.5 |

Speed on an empty 7x7 board: RAVE 5811 simulations/s, UCT 5783 simulations/s, random fill playouts 10739/s (single core, pure Python).
