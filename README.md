# search-agents

Nineteen puzzles and games, each played or solved by a classic AI algorithm: search guided by hand-built heuristics, pattern databases, or models trained from self-generated data; exact solvers, including retrograde analysis for endgame tablebases; reinforcement learning; and multi-agent pathfinding. Every algorithm runs in Python, in the terminal and behind a browser frontend.

| Domain | Classic search | Learned guidance | Live demo shows |
|---|---|---|---|
| **N-Puzzle** | BFS, DFS, IDS, UCS, bidirectional BFS, greedy, A\*, weighted A\*, IDA\* | Neural cost-to-go heuristic trained by approximate value iteration on top of a 5-5-5 pattern database | Every expansion as it happens, depth vs heuristic plots, side-by-side algorithm comparison |
| **Connect Four** | Alpha-beta negamax, UCT Monte Carlo tree search | AlphaZero-style PUCT with a policy-value ResNet trained only by self-play | The network's prior vs search visits per column, and its win probability |
| **Checkers** | Plain minimax, alpha-beta with iterative deepening, a transposition table, and move ordering | Hand-built evaluation (material, advancement, home row, centre) | How many positions it searched and how many branches pruning cut, with every candidate move's score |
| **Battleship** | Bayesian probability targeting: counts every fleet layout that fits what it has seen, exactly when that is cheap and by importance sampling otherwise | Hunt/target (checkerboard parity, then the neighbours of wounded ships) and random firing | Its odds of every cell of your fleet glowing live, and the cell it fires at next |
| **Blackjack** | Exact dynamic programming over the dealer's and player's hands (a Markov decision process), plus Monte Carlo control | None needed: the values are exact for the infinite-deck model | The full basic-strategy table, the expected value of each move for your hand, and a learner converging on the same table |
| **Route planner** (traveling salesman) | Nearest neighbour + 2-opt; exact Held-Karp dynamic programming up to 12 cities | Simulated annealing and a genetic algorithm (order crossover, inversion mutation, elitism) | Each route untangling live, length and temperature charts, a three-way race, and drawing your own route to compare |
| **2048** | Expectimax with six hand-crafted features, weights tuned by the cross-entropy method | N-tuple network trained by TD(0) on afterstates, used greedily or inside expectimax | Expected value of each move, search depth |
| **Sudoku** | Backtracking, MRV + forward checking | Constraint propagation (naked and hidden singles) | Every guess, forced fill, and backtrack, replayed |
| **Lights Out** | Gaussian elimination over GF(2), exact (no search): the null space gives every solution and the lightest one is the answer | None: the answer is exact, so there is nothing to learn | The augmented matrix reducing one pivot at a time, the rank, and which boards can never be cleared |
| **Endgame tablebases** (king and queen or rook against a lone king) | Retrograde analysis: every position's exact distance to mate, built backwards from checkmate. No search at play time | None: the tables are exact, so there is nothing to learn | Each legal move's distance to mate, the agent's choice against the alternatives, the lone king's heat map, and the distance-to-mate histogram over all 368,452 KQK positions |
| **Sokoban** | Search over pushes (player walks are free, so states are box layouts and the player's region): BFS, greedy best-first, and A\* with a sum-of-nearest-goal bound or a minimum-cost box-to-goal matching (Hungarian), each with or without deadlock pruning (dead squares, frozen boxes) | None needed: the search is exact, and both bounds are admissible (they never overestimate) | The next push the solver suggests, the matching lines from each box to its goal, dead squares tinted red, and the nodes expanded with and without deadlock pruning, side by side |
| **Pac-Man** | A* routes for ghosts with chaser, ambusher and scatter personalities; random and greedy reflex baselines | Approximate Q-learning over 13 hand-built features (one is a six-turn survival search over the ghosts' real moves), trained by epsilon-greedy self-play | Each ghost's A* route as glowing lines, every move's Q-value, and the feature contributions behind it |
| **Warehouse robots** (multi-agent pathfinding) | Independent A\*, prioritized planning (robots planned one at a time, earlier paths as moving obstacles) | Conflict-Based Search: optimal sum of costs, branching on each collision and replanning only the constrained robot with space-time A\* | Each robot's path step by step with glowing trails, collisions flashing, the constraint tree growing as CBS branches, and a race between the three planners |
| **Wordle** | Information theory: pick the guess with the most expected bits (entropy of the feedback patterns over the candidates); minimax (smallest worst-case bucket); random consistent word as the baseline | None needed: a guess is scored exactly against every answer, so the search is exhaustive over the allowed guesses | Every candidate's feedback and bucket size, the bits each guess is expected to give against the bits it actually gave, and the remaining words after each guess |
| **Poker** (Leduc hold'em, imperfect information) | Exact best response and exploitability on the game tree (no search); the Kuhn value is checked against -1/18 | Counterfactual regret minimization by self-play: vanilla CFR and CFR+, with the average strategy as the answer | The bot's probability bars for every decision, a hint with the equilibrium mix for your card, why its bets are bluffs, and exploitability falling over iterations |
| **Minesweeper** | Constraint satisfaction on the frontier, then exact probabilities: each number is a constraint, components are counted by memoized backtracking, and the global mine count weights every layout | None: the counts are exact, so the agent proves what it can and prices the rest | Which cells are proven safe or mines, the constraint components, and the exact mine probability of every covered cell |
| **Hex** | Monte Carlo tree search with RAVE: UCT statistics blended with all-moves-as-first statistics, beta = sqrt(k / (3n + k)); shortest-path and random baselines | None: every playout is a random fill, so nothing is learned | The agent's visit heat map and principal variation, a hint with visits and win rates, and agent-versus-agent games |
| **Bandits** (slot machines) | Greedy, epsilon-greedy (fixed and decaying), UCB1, and Thompson sampling with Beta posteriors; EXP3 and sliding-window UCB for drifting payouts | None: each agent learns only from the rewards it sees, and Thompson's posteriors narrow as it pulls | Each agent's regret curve and best-machine share, Thompson's Beta posteriors and UCB bars per machine, and your own regret ranked against the agents on the same machines |
| **CartPole** | Policy gradients in pure NumPy: REINFORCE with a baseline, actor-critic with a learned value baseline trained on Monte Carlo returns, and a cross-entropy search over linear policies | None: the policies are learned from reward alone. The PD controller is hand-tuned, not learned | The pole's angle and the cart's position as the policy's probabilities and the critic's value change, and the learning curves across seeds |

**Live demo: https://kb-search-agents.fly.dev** (the first request after idle can take a few seconds while the server wakes up).

## Results

### N-Puzzle: learned guidance on random 15-puzzles

40 uniformly random 15-puzzles (mean optimal solution 52.6 moves), each solved optimally beforehand by IDA* + pattern database:

| Search | Solved | Median nodes expanded | Mean extra moves vs optimal | Optimal paths |
|---|---|---|---|---|
| IDA\* + 5-5-5 pattern database (optimal) | 40/40 | 598,989 | 0% | 40/40 |
| Batch weighted A\* + pattern database | 37/40 | 19,895 | 2.2% | 20/37 |
| **Batch weighted A\* + neural heuristic** | **40/40** | **5,009** | **3.3%** | **18/40** |

The learned heuristic expands about 120x fewer nodes than optimal search, at the cost of paths 3.3% longer on average. Deployed on a shared 1 GB machine, it solves a random 15-puzzle in under a second.

All algorithms on 50 uniformly random 8-puzzles (mean optimal solution 21.8 moves; IDS is omitted because it exceeds the time limit at this depth):

| Algorithm | Mean nodes expanded | Mean path length | Optimal paths |
|---|---|---|---|
| BFS | 73,756 | 21.8 | 50/50 |
| Uniform-cost search | 88,444 | 21.8 | 50/50 |
| DFS | 79,209 | 43,672 | 0/50 |
| Bidirectional BFS | 1,941 | 21.8 | 50/50 |
| Greedy best-first (Manhattan) | 267 | 40.7 | 5/50 |
| Weighted A\* (w = 2) | 313 | 23.6 | 20/50 |
| A\* (Manhattan) | 771 | 21.8 | 50/50 |
| A\* (linear conflict) | 404 | 21.8 | 50/50 |
| IDA\* (Manhattan) | 2,090 | 21.8 | 50/50 |

Heuristic strength for IDA\* on 15 random 60-move scrambles: Manhattan distance 12.8M nodes, linear conflict 2.8M, pattern database 416K. The neural heuristic's average error against true optimal costs is 2.8 moves, vs 12.2 for the pattern database alone.

### Checkers: what alpha-beta pruning saves

Both agents assume the opponent always answers with its best reply. On a position 8 moves into a game, searched to a fixed depth, they choose the same move with the same score:

| Depth (half-moves) | Minimax positions | Alpha-beta positions | Fewer by |
|---|---|---|---|
| 4 | 2,962 | 618 | 4.8x |
| 5 | 16,845 | 1,252 | 13x |
| 6 | 92,608 | 3,566 | 26x |

The move generator matches the published perft counts from the opening (7, 49, 302, 1,469, 7,361 positions at depths 1-5).

### Battleship: shots to sink a random fleet

100 random fleets on a 10x10 board with the standard fleet (5, 4, 3, 3, 2; 17 ship cells). Every agent plays
the same fleets, so the comparison is paired. From `python -m battleship benchmark --games 100 --seed 1`:

| Agent | Mean shots | Median | Worst | Time per game (Python) |
|---|---|---|---|---|
| Random | 95.4 | 97 | 100 | 0.005 s |
| Hunt/target (checkerboard, then neighbours) | 51.9 | 54 | 67 | 0.004 s |
| **Bayesian probability targeting** | **45.9** | **45.5** | **64** | 0.78 s |

Head to head on the same fleets, the Bayesian agent needs fewer shots than hunt/target on 67 of 100 fleets (3 ties), 6.0 shots fewer on average (95% interval 3.7 to 8.2).

The Bayesian agent pays per shot: it recounts consistent layouts (or samples 1000 of them) before every
shot, so its time per game is well over 100 times the baselines'.

### Blackjack: exact basic strategy and a learner that finds it

- **House edge** with perfect basic strategy, solved exactly (infinite deck, dealer stands on soft 17, dealer peek, 3:2 blackjack, double on any two cards, one split with doubling after it): **0.570%** of each bet.
- **Measured** through a 6-deck shoe over 200,000 rounds with the same table: **-0.66% ± 0.26%** per unit bet, consistent with the exact figure (a finite shoe is not the infinite-deck model).
- **Monte Carlo control** (exploring starts, every-visit averages, seed 1), agreement with the exact table on the 321 decisive cells: **73%** after 40,000 hands (mean value lost per cell 0.039) and **78%** after 100,000 hands (0.023). Near-tied cells need far more hands, so agreement rises slowly.

### 2048: TD-learned n-tuple network

500 games, no search, each move chosen by reward + learned afterstate value:

| Agent | Games | Mean score | Median score | Reached 2048 | Reached 4096 | Reached 8192 | Best tile |
|---|---|---|---|---|---|---|---|
| N-tuple network, greedy (no search) | 500 | 42,085 | 40,834 | 81% | 31% | 1% | 8192 |

With expectimax search on top (50 ms per move; 4 games per agent, so a small sample):

| Agent | Games | Mean score | Median score | Reached 2048 | Reached 4096 | Reached 8192 | Best tile |
|---|---|---|---|---|---|---|---|
| Expectimax, tuned eval + cache + probability cutoff (50 ms/move) | 4 | 8,277 | 9,446 | 0% | 0% | 0% | 512 |
| Expectimax, learned n-tuple eval (50 ms/move) | 4 | 78,849 | 64,462 | 100% | 50% | 25% | 8192 |

Tuning the hand-crafted evaluation's six weights with the cross-entropy method raised greedy play from a mean score of 5,221 to 7,081 (1,000 fresh games each).

### Sudoku: 200 generated puzzles with unique solutions

| Solver | Solved | Mean guesses | Max guesses |
|---|---|---|---|
| Plain backtracking | 198/200 (2M-guess limit) | 119,468 | 2,000,001 |
| MRV + forward checking | 200/200 | 364 | 4,058 |
| Constraint propagation + MRV | 200/200 | 7 | 89 |

### Connect Four: AlphaZero-style self-play

Trained only by self-play on a laptop CPU: 2,432 games (19 iterations). The served network with 200 PUCT simulations per move, 50 games per opponent, colours alternating and two random opening moves per game (`python -m connect4.evaluate`):

| Opponent | AlphaZero wins | Draws | Losses | Score |
|---|---|---|---|---|
| Alpha-beta minimax, depth 2 | 30 | 4 | 16 | 64% |
| Alpha-beta minimax, depth 4 | 36 | 5 | 9 | 77% |
| Alpha-beta minimax, depth 6 | 18 | 8 | 24 | 44% |

For reference, plain MCTS with 1,000 random-rollout simulations scores 32-2-16 against depth-4 minimax. AlphaZero implementations typically train on tens of thousands of self-play games for Connect Four; this one has seen 2,432. How it was trained, on what hardware and for how long: [docs/connect4-training.md](docs/connect4-training.md).


### Lights Out: exact linear algebra over GF(2)

Pressing a light is addition mod 2, so a board is a linear system A x = b with a 25x25 matrix of rank 23. That splits the boards: only 1 in 4 can be cleared at all, and every solvable 5x5 board has exactly 4 solutions. Over 3,000 random solvable boards, the lightest solution averages 9.9 presses, and the heaviest of the four is 5.2 presses heavier on average. Solving a 5x5 board takes about 3 ms in Python.

### Bandits: exploration vs exploitation

Five slot machines with hidden payout probabilities, 100 seeds per casino, 10,000 pulls. Cumulative regret is the expected reward lost against the best machine; the table gives the mean with a 95% band (the half-width after ±). `python -m bandits benchmark` regenerates `results/bandits_benchmark.md` and `.json`.

| Casino | Agent | Regret at T=1,000 | Regret at T=10,000 | % best machine, last 1,000 pulls |
|---|---|---:|---:|---:|
| Bernoulli | greedy | 70.3 ± 23.4 | 679.9 ± 236.0 | 63.0% |
| Bernoulli | epsilon-greedy 0.1 | 44.6 ± 4.7 | 309.9 ± 19.5 | 91.2% |
| Bernoulli | epsilon-greedy decaying | 26.4 ± 7.2 | 161.2 ± 75.3 | 86.0% |
| Bernoulli | UCB1 | 78.6 ± 2.2 | 197.2 ± 9.5 | 95.4% |
| Bernoulli | Thompson sampling | 23.1 ± 2.4 | 35.6 ± 4.1 | 99.6% |
| Gaussian (sigma 0.2) | greedy | 22.4 ± 8.8 | 202.5 ± 87.1 | 80.0% |
| Gaussian | epsilon-greedy 0.1 | 32.3 ± 2.0 | 282.5 ± 16.7 | 92.1% |
| Gaussian | epsilon-greedy decaying | 15.2 ± 4.1 | 58.9 ± 36.3 | 94.0% |
| Gaussian | UCB1 | 8.5 ± 0.6 | 13.0 ± 1.1 | 99.9% |
| Gaussian | Thompson sampling | 6.5 ± 0.8 | 9.0 ± 0.8 | 99.9% |
| Drifting (redrawn every 500 pulls) | greedy | 169.6 ± 25.4 | 2640.1 ± 96.4 | 26.8% |
| Drifting | epsilon-greedy 0.1 | 123.3 ± 14.0 | 2316.1 ± 70.2 | 24.9% |
| Drifting | epsilon-greedy decaying | 134.2 ± 18.9 | 2456.0 ± 93.1 | 26.7% |
| Drifting | UCB1 | 86.6 ± 5.3 | 978.3 ± 51.8 | 58.5% |
| Drifting | Thompson sampling | 91.6 ± 11.1 | 2058.8 ± 72.4 | 27.3% |
| Drifting | EXP3 | 241.2 ± 10.8 | 2558.0 ± 64.1 | 25.4% |
| Drifting | sliding-window UCB (W = 200) | 92.6 ± 2.2 | 950.5 ± 8.4 | 63.3% |

The Lai-Robbins floor (the asymptotic lower bound for fixed machines, averaged over the machine sets) is 7.34 ln t for Bernoulli (67.6 at T = 10,000) and 1.45 ln t for Gaussian (13.3 at T = 10,000). The bound is a limit statement, not a guarantee at a given T: Thompson and UCB1 on Gaussian machines, and Thompson on Bernoulli machines, finish below the curve at these horizons, which is consistent with the theorem. Thompson sampling is the most efficient on stationary casinos, and UCB1's forced early bonus costs it early regret (78.6 at T = 1,000 on Bernoulli). With drifting payouts, Thompson's posteriors never forget, so it falls to the level of greedy; the windowed and the optimistic agents follow the drift.

### CartPole: policy gradients in NumPy

Benchmark: 200 seeded episodes per agent (seeds 10000-10199, none used in training), 500-step cap. Learned policies act greedily.

| Agent | Mean steps | 95% CI | Reached 500 |
|---|---:|---:|---:|
| Random | 22.5 | ±1.6 | 0.0% |
| PD controller (hand-tuned) | 500.0 | ±0.0 | 100.0% |
| REINFORCE + baseline | 500.0 | ±0.0 | 100.0% |
| Actor-critic (learned value baseline, Monte Carlo returns) | 500.0 | ±0.0 | 100.0% |
| Cross-entropy search (linear policy) | 500.0 | ±0.0 | 100.0% |

The benchmark saturates: every learned agent balances all 200 episodes, so it does not separate them. Learning speed does. Training used seeds 0-4 for 1,500 episodes each. The trailing-100 mean first reaches 475 at these episodes:

| Agent | Seed 0 | Seed 1 | Seed 2 | Seed 3 | Seed 4 |
|---|---:|---:|---:|---:|---:|
| REINFORCE + baseline | 376 | 384 | 408 | 478 | 377 |
| Actor-critic | 344 | 381 | 370 | 347 | 382 |
| Cross-entropy search | 391 | 388 | 528 | 299 | 329 |

REINFORCE reaches 475 in 376 to 478 episodes, actor-critic in 344 to 382, and cross-entropy in 299 to 528. The shipped policy for each agent is the seed with the best last 50 episodes (`results/cartpole_learning.md`).

### Endgame tablebases: exact distance to mate

Built by `python -m endgame build` (every entry re-checked against the rules afterwards), counted by `python -m endgame stats`:

| | KQK | KRK |
|---|---:|---:|
| Legal positions | 368,452 | 399,112 |
| Longest forced mate | 10 moves (19 plies) | 16 moves (31 plies) |
| Black to move: lost / drawn | 200,896 / 23,048 | 201,700 / 22,244 |
| Solve time (quiet laptop / shared laptop) | 4.9 s / 25.0 s | 4.4 s / 34.3 s |

Every position with White to move is a win in both endgames. The longest mates are the known values for these endgames (10 and 16 moves).

### Sokoban: pushes, deadlocks, and bounds

Twelve levels, from one push to 19 pushes in the optimal solution. Levels 1 to 7 are hand-drawn and 8 to 12 are generated (random reverse pushes from the goals, kept when the solver finds the most pushes). Each run gets 20,000 expanded nodes and 10 s.

| Search | Deadlock pruning | Solved | Optimal | Nodes expanded, all 12 levels | Time |
|---|---|---:|---:|---:|---:|
| BFS | off | 8/12 | 8/12 | 77,355 | 41.8 s |
| BFS | on | 10/12 | 10/12 | 38,292 | 32.4 s |
| Greedy (matching) | off | 12/12 | 11/12 | 6,039 | 5.9 s |
| Greedy (matching) | on | 12/12 | 11/12 | 153 | 0.3 s |
| A\* (simple bound) | off | 10/12 | 10/12 | 28,743 | 29.6 s |
| A\* (simple bound) | on | 12/12 | 12/12 | 10,907 | 11.3 s |
| **A\* (matching)** | off | 12/12 | 12/12 | 269 | 0.3 s |
| **A\* (matching)** | **on** | **12/12** | **12/12** | **154** | **0.2 s** |

Unsolved runs ran out of budget. "Optimal" means the solved plan has the optimal push count. Greedy's plan on the Aisle level is the only non-optimal one. The per-level node counts are in `results/sokoban_benchmark.md`.

- Pruning helps every rule. Greedy drops from 6,039 to 153 nodes, the simple bound from 28,743 to 10,907 (and solves 12 levels instead of 10), and BFS solves 10 levels instead of 8 within the budget.
- The matching bound does most of the work: without pruning it expands 269 nodes for all 12 levels, against 28,743 for the simple bound, and pruning trims that further to 154.

### Pac-Man: approximate Q-learning vs baselines

Seeds 10000-10199, 200 games per maze on two hand-drawn mazes (400 games per agent).

| Agent | Win rate | Mean score | Mean turns | Won / lost / timeout |
|---|---|---|---|---|
| **Approximate Q-learning** | **64.2%** | **1350.3** | 193.9 | 257 / 47 / 96 |
| Reflex (greedy nearest pellet, avoids ghosts) | 0.5% | 157.5 | 22.8 | 2 / 395 / 3 |
| Random | 0.0% | 45.5 | 8.9 | 0 / 400 / 0 |

Per maze: the Q-agent wins 43.5% on Neon Lanes (mean 1218) and 85.0% on Vault (mean 1483).
Reflex: 0.5% and 0.5% (means 178 and 137). Random: 0% on both (means 51 and 40).

For comparison, the 12-feature agent this replaced (no survival feature) won 5.0% with a mean of 667.5 on the same seeds. The timeouts are games that reached the 300-turn limit with pellets left: the agent survives them but clears the maze too slowly.

Training: 3000 self-play games, seed 1, alternating the two mazes; step size 0.01 falling
to 0.001, discount 0.9, exploration 0.3 falling to 0.02. Over the final 25 training games
(exploration 0.02) it averaged 852 points and won 5 of them.

### Warehouse robots: optimal cost without collisions

Bottleneck map, 6 robots, 10 random instances (`python -m warehouse benchmark --layout bottleneck --robots 6 --instances 10`):

| Planner | Solved | Mean sum of costs | Mean makespan | Mean collisions | Mean CBS nodes | Mean seconds |
|---|---|---|---|---|---|---|
| **Conflict-Based Search** | **10/10** | **52.9** | **15.5** | **0.00** | 693.6 | 0.19 |
| Prioritized planning | 10/10 | 55.9 | 15.7 | 0.00 | - | 0.0008 |
| Independent A\* | 0/10 | 47.7 | 14.3 | 3.80 | - | 0.0002 |

Prioritized planning was suboptimal on 5 of the 10 instances (mean extra cost 3.0). Independent A* is cheapest only because its paths collide: its cost is not a valid plan.

Shelf aisles, 8 robots, 10 instances (`--layout aisles --robots 8 --instances 10`):

| Planner | Solved | Mean sum of costs | Mean makespan | Mean collisions | Mean CBS nodes | Mean seconds |
|---|---|---|---|---|---|---|
| **Conflict-Based Search** | **10/10** | **75.7** | 17.4 | **0.00** | 135.0 | 0.04 |
| Prioritized planning | 10/10 | 79.5 | 17.1 | 0.00 | - | 0.0004 |
| Independent A\* | 0/10 | 72.9 | 17.0 | 3.50 | - | 0.003 |

Prioritized planning was suboptimal on 9 of 10 (mean extra cost 3.8).

### Wordle: information theory vs minimax vs random

All 3,568 answers (SCOWL common-word tier), with 6,748 allowed guesses for each strategy. Full numbers:
`results/wordle_benchmark.md` and `results/wordle_benchmark.json`.

| Strategy | Mean guesses | 1 | 2 | 3 | 4 | 5 | 6 | Failed (over 6) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **Max expected information (entropy)** | **3.627** | 0 | 59 | 1,513 | 1,710 | 272 | 14 | **0** |
| Minimax (smallest worst bucket) | 3.739 | 0 | 50 | 1,223 | 1,927 | 344 | 24 | 0 |
| Random consistent word | 4.412 | 1 | 94 | 699 | 1,274 | 886 | 368 | 246 (6.89%) |

Best opening word by expected information: TARES (6.23 bits; worst bucket 251, expected 78.7 words left).
These answers come from SCOWL's common tier, not the official list, so the numbers are not directly comparable
with published Wordle figures.

### Poker: counterfactual regret minimization on Leduc hold'em

CFR+ with alternating updates, 5000 iterations, on the six-card game with a public card (288 information sets). The committed average strategy is exploitable by 0.05 milli-big-blinds per hand (the best responses are exact, so this is not a sampled estimate), and the game value for the first player is -0.0856 chips, which matches the published value for this game. Vanilla CFR reaches 0.0036 chips after the same 5000 iterations, about 240 times the exploitability, so the CFR+ curve drops much faster (`results/poker_train_log.jsonl`). On Kuhn poker the value is -0.0555556 against the exact -1/18, an error of 4e-9 for CFR+ after 10000 iterations.

Head-to-head, the CFR+ bot against each opponent over 100,000 seeded deals, each played in both seats (200,000 hands; milli-big-blinds per hand, big blind = 2 chips, 95% CI from the spread over deals):

| Opponent | mbb/hand (bot's side) | 95% CI |
|---|---:|---:|
| random | +358.0 | [+350.2, +365.7] |
| always-call | +317.7 | [+313.1, +322.2] |
| hand-strength (rules on own card and the public card) | +68.2 | [+63.5, +72.9] |
| the bot against itself | +3.1 | [-1.4, +7.6] |

The self-play row is zero within its interval, as equilibrium requires.

### Minesweeper: proofs first, then the lowest exact odds

Seeds 10000 onward, the same boards for every agent. Win rate has a 95% Wilson interval.
The first click is free and is not counted as a guess.

| size | agent | win rate (95% CI) | guesses per game | guesses per win | wins with no guess |
|---|---|---|---|---|---|
| beginner 9x9, 10 mines (1000 games) | random | 0.0% (0.0%-0.4%) | 3.36 | - | 0 |
| | rules | 79.8% (77.2%-82.2%) | 0.87 | 0.58 | 579 of 798 |
| | csp | 96.1% (94.7%-97.1%) | 0.12 | 0.08 | 902 of 961 |
| | probability | 97.5% (96.3%-98.3%) | 0.14 | 0.12 | 902 of 975 |
| intermediate 16x16, 40 mines (500 games) | random | 0.0% (0.0%-0.8%) | 4.22 | - | 0 |
| | rules | 49.2% (44.8%-53.6%) | 2.03 | 1.39 | 111 of 246 |
| | csp | 85.4% (82.0%-88.2%) | 0.44 | 0.23 | 363 of 427 |
| | probability | 87.8% (84.6%-90.4%) | 0.43 | 0.27 | 363 of 439 |
| expert 30x16, 99 mines (200 games) | random | 0.0% (0.0%-1.9%) | 4.12 | - | 0 |
| | rules | 1.5% (0.5%-4.3%) | 3.98 | 1.67 | 2 of 3 |
| | csp | 39.0% (32.5%-45.9%) | 2.29 | 1.58 | 33 of 78 |
| | probability | 48.5% (41.7%-55.4%) | 2.88 | 2.22 | 33 of 97 |

The probability agent wins more expert boards but guesses more per game, because it keeps playing
longer. The random baseline uses a uniformly random covered cell and no reasoning at all.

### Hex: RAVE vs UCT, and how the gap moves with budget

Hex 7x7, 400 simulations per move, 40 seeded games per matchup (colours alternate). Win rate is for the first agent, with a 95% Wilson interval. From `results/hexgame_benchmark.md`:

| matchup | first agent win rate (95% CI) | wins as DOWN | wins as ACROSS | mean moves |
|---|---|---|---|---|
| RAVE vs UCT | 100.0% (91.2%-100.0%) | 20/20 | 20/20 | 15.9 |
| RAVE vs shortest-path | 97.5% (87.1%-99.6%) | 19/20 | 20/20 | 16.8 |
| UCT vs shortest-path | 32.5% (20.1%-48.0%) | 6/20 | 7/20 | 18.7 |
| RAVE vs random | 100.0% (91.2%-100.0%) | 20/20 | 20/20 | 14.8 |
| UCT vs random | 100.0% (91.2%-100.0%) | 20/20 | 20/20 | 25.5 |

Speed on an empty 7x7 board, single core, pure Python: RAVE 5811 simulations/s, UCT 5783 simulations/s, random fill playouts 10739/s.

RAVE beats UCT 40 of 40 games at 400 simulations per move, but the gap narrows with budget. A budget check, run with `python -m hexgame benchmark --sims 1000 --games 20` (not written to `results/`, about 3.5 minutes on one core):

| matchup | first agent win rate (95% CI) | wins as DOWN | wins as ACROSS | mean moves |
|---|---|---|---|---|
| RAVE vs UCT | 90.0% (69.9%-97.2%) | 10/10 | 8/10 | 15.6 |
| RAVE vs shortest-path | 90.0% (69.9%-97.2%) | 9/10 | 9/10 | 17.0 |
| UCT vs shortest-path | 55.0% (34.2%-74.2%) | 7/10 | 4/10 | 19.2 |
| RAVE vs random | 100.0% (83.9%-100.0%) | 10/10 | 10/10 | 14.0 |
| UCT vs random | 100.0% (83.9%-100.0%) | 10/10 | 10/10 | 19.1 |

At 1,000 simulations RAVE wins 18 of 20 against UCT. With 20 games the interval is wide (69.9%-97.2%), so this shows the advantage shrinking, not a measured reversal.

## How it works

### N-Puzzle

- **Search** (`npuzzle/search.py`, `npuzzle/iterative.py`). One frontier interface covers FIFO, LIFO, and a priority queue with decrease-key (lazy deletion, with live states tracked separately so a heap of stale entries never looks non-empty). IDA\* runs on one mutable board and updates Manhattan distance and pattern database values incrementally: a move changes one tile, so it changes one lookup.
- **Pattern database** (`npuzzle/pdb.py`). The 15 tiles split into three groups of five. A 0-1 BFS from the goal records, for every placement of a group, the fewest moves of that group's tiles; other tiles move for free, so the three values add up to an admissible heuristic. Building all three takes about 40 seconds.
- **Neural heuristic** (`npuzzle/train.py`, `npuzzle/neural.py`). h(s) = PDB(s) + softplus(MLP(one-hot(s))): the network learns only how far the pattern database underestimates. Training uses approximate value iteration (as in DeepCubeA): sample boards by random walks from the goal, set the target to 1 + min over children of a frozen copy of h, fit, and copy. No solver labels are needed. The learned h is not admissible, so it is used with batch weighted A\*, which scores hundreds of states per network call.
- **Test set** (`npuzzle/testset.py`). 100 uniformly random 15-puzzles solved optimally by IDA\* + PDB. The mean optimal cost is 52.6 moves, which matches the known average for random 15-puzzles.

### Connect Four

- **Board** (`connect4/board.py`). Two 64-bit integers (all stones, and the stones of the player to move), with a spare bit per column so line checks never wrap. Four in a row is four shift-and-AND operations.
- **AlphaZero** (`connect4/puct.py`, `connect4/train_az.py`). PUCT search guided by a residual policy-value network. Self-play runs 128 games at once and batches every game's leaf evaluation into one network call per simulation round. Positions are stored with mirror images; the loss is value MSE plus policy cross-entropy.
- **Serving without torch** (`connect4/net.py`). Batch norm is folded into the convolutions at export, so inference is NumPy convolutions and matrix multiplies.

### 2048

- **Board** (`game2048/board.py`). One 64-bit integer, four bits per tile. Every row's slide is precomputed for all 65,536 rows, so a move is four lookups plus a bit-trick transpose.
- **Expectimax** (`game2048/expectimax.py`). Six hand-crafted features (empty cells, monotonicity, smoothness, max tile in a corner, mergeable neighbours, max tile) come from per-row lookup tables. Their weights are found automatically by the cross-entropy method over batches of simulated games (`game2048/tune.py`). A transposition table and a cutoff for unlikely tile spawns let the search go deeper in the same time.
- **N-tuple network** (`game2048/ntuple.py`, `game2048/train_td.py`). Six 5-cell patterns, each applied in all 8 board symmetries with a shared table: 48 lookups per evaluation. Trained by TD(0) on afterstates with 1,000 games in lockstep in NumPy. Batched TD updates are averaged per weight; summing them made shared weights take batch-sized steps and diverge.

### Checkers

- **Rules** (`checkers/board.py`). The 32 dark squares as a tuple; mandatory captures, multi-jumps (a captured piece stays on the board until the move ends, so it cannot be jumped twice), and crowning, which ends the move. Verified against published perft counts.
- **Search** (`checkers/search.py`). Negamax, so each side maximizes its own score and the opponent's reply is assumed to be the one worst for it. Alpha-beta skips a move as soon as one reply proves it worse than an alternative already found; iterative deepening, a transposition table, and trying the previous best move first make those cutoffs come early. Positions with a capture pending are searched one level deeper rather than scored mid-exchange.

### Blackjack

- **Dealer** (`blackjack/solver.py`). A recursion on the dealer's next card gives, for every hard total and ace flag, the probability of each final total (17 to 21, or bust). The dealer peeks for blackjack under an ace or a ten, so the player's decisions are made on the hole cards conditioned on "no blackjack".
- **Player** (`blackjack/solver.py`). Each hand is a state: hard total, whether an ace is in it, and whether it is still two cards. Hitting always adds a card, so each state depends only on states with larger totals, and memoised recursion is exact backward induction. A pair splits into two independent hands under the infinite deck, so the split is twice the average over the second card of one hand's best play.
- **Learner** (`blackjack/learner.py`). Monte Carlo control with no model: exploring starts from every table cell, every-visit sample averages, and epsilon-greedy choices. Stand, hit, and double share statistics by hard total and ace flag, since their future does not depend on card count. It is compared with the exact table, not trained on it.
- **Game** (`blackjack/game.py`, `python -m blackjack`). A 6-deck shoe with reshuffles at 75%, the table's rules, and the same basic strategy for `simulate`. The advice on the page and in the terminal is the exact table, not the learner's.

### Battleship

- **Rules** (`battleship/board.py`). Cells are 0-99 as bitmasks, so set operations on the board are single integer operations. Ships are straight, on the board, and may touch but not overlap. A shooter learns a sunk ship's full cell list, which is the only information the agents see. Fleets are drawn uniformly over legal layouts (rejection sampling), which matches the uniform prior the probability model uses.
- **Probability model** (`battleship/probability.py`). For each cell, P(ship) is the share of consistent fleet layouts that use it, where a layout is consistent when it avoids misses and sunk ships and covers every hit that has not sunk its ship. Layouts are counted exactly by a depth-first search with pruning when the placements are few enough (about 60,000 candidate checks at most). Otherwise 1000 layouts are sampled by importance sampling: each ship is drawn uniformly from the placements that fit, and each sample is weighted by the number of options at each step, which makes the weighted samples an unbiased stand-in for a uniform draw over consistent layouts. Sampling is vectorised over all samples with numpy.
- **Agents** (`battleship/agents.py`). The Bayesian agent fires at the unknown cell with the highest probability, breaking ties at random. The hunt/target baseline fires on a checkerboard (every ship covers an even cell) until a hit, then along a line once two hits align, and otherwise next to a hit.
- **Server** (`server/battleship_api.py`). The agent's fleet is random and stored server-side under an unguessable id, with a bounded store and a 30-minute expiry, so the page cannot read it before the game ends. The player's own fleet never reaches the server: the page sends the agent's shots at it, and the server returns the odds for those observations.
- **Page** (`web/battleship.html`). Place a fleet, fire at the agent's fleet, and watch the agent's odds glow over your grid while its reticle settles on its next cell. Watch mode lets the agent hunt a random fleet at a chosen speed.

### Route planner

- **Solvers** (`routes/tsp.py`). All share one move, 2-opt: reverse a segment of the route, which removes a crossing whenever two roads cross. Its effect on length depends on four edges only, so it costs O(1) to evaluate.
  - *Greedy + 2-opt*: nearest-neighbour construction, then improving 2-opt moves until none is left (a local optimum).
  - *Simulated annealing*: random 2-opt moves from a random route; worse routes are accepted with probability exp(-Δ/T), and T cools geometrically from a start chosen so about 80% of uphill moves are accepted at first. A final 2-opt pass removes any crossing left by random sampling.
  - *Genetic algorithm*: tournament selection, order crossover (keeps a slice of one parent, fills the rest in the other parent's order, so the child is always a valid route), inversion mutation, and elitism.
  - *Held-Karp*: exact dynamic programming over subsets, O(n²·2ⁿ), used for maps of 12 cities or fewer to report how far each heuristic is from optimal.

### Lights Out

- **Rules** (`lightsout/board.py`). Cells are numbered row by row. A press toggles the cell and its orthogonal neighbours. Boards and press sets are integers, so a press is one XOR of neighbourhood masks.
- **Solver** (`lightsout/solver.py`). Row i of A is the equation for light i. Gauss-Jordan elimination mod 2 reduces [A | b], with each row operation a single integer XOR. A row with no left-hand side and a 1 on the right proves the board unreachable. Otherwise the solutions are one particular solution plus every combination of the null-space basis, and the lightest of the 2^nullity solutions is returned. For 3x3, 6x6, 7x7 and 8x8 the matrix is invertible, so every board is solvable; 4x4 has nullity 4 and 9x9 has nullity 8.
- **Checks** (`tests/test_lightsout.py`). Exhaustive brute force over all 512 boards of 3x3, sampled agreement on the minimum press count for 4x4, and solvability checked against the null-space orthogonality test.

### Endgame tablebases

- **Rules** (`endgame/rules.py`). Squares 0..63, a1 = 0. A position is (white king, white piece, black king, side to move). Legal moves, unmoves (predecessors), attack lines between squares precomputed once, and FEN. The black king taking the piece leaves king against king, so it is a draw and never in the table.
- **Solver** (`endgame/retro.py`). Checkmates are seeded at distance 0. A FIFO queue processes decided positions in order of distance: a predecessor where the side to move can move into a loss is a win one ply deeper; a predecessor whose every move loses is a loss at the last-resolved move's distance. Whatever is never decided is a draw.
- **Tables** (`endgame/tablebase.py`). One uint16 per index (side to move included): an odd code is a win in that many plies, an even code a loss, and two sentinels mark draws and illegal tuples. The agent plays the fastest win, the longest defence, or a drawing move.
- **Checks** (`tests/test_endgame.py`, `python -m endgame build`). The move and unmove generators are exact inverses. Every entry is checked against the definition of distance to mate, and the tests verify the known maxima and optimal-play games.

### Sokoban

- **Rules** (`sokoban/board.py`). A level is a padded grid with flat indices and four offsets. Walking is a BFS around the boxes. A push moves a box one cell when the far side is free floor.
- **Search** (`sokoban/search.py`). A state is the player's smallest reachable cell plus the sorted box cells, so each state is a box layout and a region. Successors are pushes only. BFS, greedy best-first, and A\* share one loop and differ only in the priority. A\* with matching returns the fewest pushes; the plan's walks are rebuilt by BFS from the parent links.
- **Dead squares** (`sokoban/tables.py`). A backward BFS from each goal runs the push relation in reverse: a box at y could have come from y - d if the player could stand at y - 2d. Cells no goal reaches are dead.
- **Frozen boxes** (`sokoban/deadlock.py`). A box with a wall or a frozen box on one side of each axis can never move. The frozen set is the greatest fixed point of that rule, which covers 2x2 blocks. It is sound: it never flags a solvable position, and the tests check this exhaustively on small levels.
- **Bounds** (`sokoban/heuristics.py`, `sokoban/matching.py`). The simple bound sums each box's push distance to its nearest goal. The matching bound is the cheapest one-to-one pairing of boxes and goals, found with the Hungarian algorithm. Both are admissible because the push distance ignores the other boxes. Pairs with no push path use a Manhattan fallback so the bound stays finite without pruning; that can break consistency, so the search reopens a state reached by a cheaper path.
- **Checks** (`tests/test_sokoban.py`). The matching is checked against brute force over permutations, the bounds against the remaining pushes of every optimal plan, and every built-in level is solved with its optimal push count and replayed.

### Sudoku

`sudoku/solver.py` has three solvers behind one interface, each recording a trace of guesses and backtracks for the visualizer. Propagation keeps candidate sets as 9-bit masks and, after every guess, fills cells with one candidate and digits with one possible place in a row, column, or box, until nothing changes.

### Pac-Man

The engine (`pacman/engine.py`) is a turn-based game: Pac-Man steps first, then each ghost moves one cell. Collisions are checked before and after the ghosts move, so a ghost cannot pass through him. Power pellets scare every ghost for eight turns, during which the ghosts flee from him.

Ghosts plan with A* (`pacman/search.py`, Manhattan heuristic, unit steps). The chaser targets Pac-Man's cell, the ambusher targets the cell four steps ahead of him, and the scatter ghost chases until it comes within four cells, then retreats to a corner.

The learned agent scores each legal move as `Q(s, a) = w · f(s, a)`, where `f` holds thirteen hand-built features of the position after the move: pellet and power-pellet eating, distances to the nearest pellet and power pellet (linear, so far targets still produce a gradient), active ghosts within six cells and one step away, a fatal move, scared ghosts within eight cells, eating a scared ghost, the openness of the destination, an interaction term for a ghost closing in on a dead end, and the survival feature. The survival feature (`pacman/lookahead.py`) plays the game forward from the move, letting the ghosts answer with the same AI the game uses, and records how many of the next six turns Pac-Man stays alive (up to six), with each position memoised. This is a search inside the feature, and it is exact for the ghost AI: the agent effectively has a model of how the ghosts move. The learned part is the weights. Without this feature the same learning reached about 5% wins. Each turn it applies the Q-learning update `w += α (r + γ max Q(s', ·) − Q(s, a)) f(s, a)`, with a per-turn cost and a large penalty for being caught. The weights come from self-play (`python -m pacman train`). The server runs the same Python code, and the page shows each move's Q-value and each feature's contribution.

### Warehouse robots

Each robot moves one cell or waits per time step, and must not share a cell with another robot or swap places with it. Independent A* plans each robot alone and so ignores these rules; the paths it returns can collide.

Prioritized planning plans robots in turn. Each new robot runs space-time A* around the paths of the robots already planned, including their goal cells, which stay occupied. Every plan it returns is collision-free, but early robots can take routes that cost later ones, and a later robot can be boxed in.

Conflict-Based Search (Sharon et al., 2012) is optimal for sum of costs. The root plans every robot alone. If the plan collides, the first collision between robots i and j splits the search into two children: one forbids i from that cell (or that move) at that time, the other forbids j. Only the constrained robot is replanned. Nodes are expanded cheapest first, so the first collision-free plan has the minimum total arrival time. The low level is space-time A* over (cell, time) states with the static walking distance as heuristic; it is optimal for one robot under constraints. The search is capped by node and time budgets and reports why it stopped.

On small random maps, a brute-force search over joint positions finds the same optimal cost as CBS; the tests check this for two robots.

### Wordle

A guess returns five tiles (gray, yellow, green), so there are 3^5 = 243 possible feedback patterns. The solver
keeps the answers consistent with the feedback so far, and for each allowed guess it counts how those answers
would split across the patterns. The expected information of a guess is the entropy of that split,
H = -sum p log2 p, in bits. The entropy strategy plays the guess with the most bits. Minimax plays the guess with
the smallest largest bucket. The random baseline picks any consistent answer. The feedback table (every guess
against every answer, one byte each) is precomputed, so scoring a guess is a gather and a bincount.

### Poker

- **Games** (`poker/kuhn.py`, `poker/leduc.py`, `poker/betting.py`). Kuhn is three cards and one bet; Leduc is six cards, a private card each, two betting rounds with bets of 2 and 4 chips, a public card between them, and at most two bets or raises per round. A state is immutable, and each player's information set is its card, the public card once shown, and the betting, so the key is the same for every history the player cannot tell apart.
- **Tree and exact values** (`poker/tree.py`). The game is built once as explicit arrays (9,451 nodes for Leduc). Expected value is one recursion. Exploitability is exact: a best response is one choice per information set, so it takes two passes (opponent and chance reach forward, then counterfactual values per information set deepest first), and the code is checked against brute force over every pure strategy on Kuhn.
- **CFR and CFR+** (`poker/cfr.py`). Each seat in turn walks the tree, weighting regrets by the opponent's and chance's reach and the average strategy by the seat's own reach. Regret matching turns regrets into a strategy. CFR+ floors regrets at zero and averages with weight t. The updates alternate between seats, which is the form CFR+ is usually run in: with simultaneous updates Kuhn stalls near 1e-3 instead of 1e-5 at 10000 iterations.
- **Bots and benchmark** (`poker/bots.py`, `poker/benchmark.py`). The bot samples its move from the average strategy at every decision. Baselines are random, always-call, and a hand-strength rule that reads its own card and the public card only after round 1. Every deal is played in both seats.
- **Serving** (`server/poker_api.py`). The server holds each hand, so the bot's card reaches the browser only at showdown. A bot move is a lookup in `poker/data/leduc_strategy.json`; nothing trains on the server.

### Minesweeper

- **Rules** (`minesweeper/board.py`). Mines are placed after the first click, which keeps the clicked cell and its neighbours clear. A zero floods outward. Agents see only a `View` of the revealed numbers and the mine count.
- **Single-cell rules** (`minesweeper/inference.py`, `rule_pass`). A number with as many mines still to place as covered neighbours makes all of them mines. A number whose known mines already satisfy it makes the rest safe. Repeated to a fixpoint.
- **Components** (`minesweeper/inference.py`). Each number is a constraint "exactly `need` of these covered cells are mines". Constraints sharing a covered cell form a component. Components are independent once the total is fixed, so each is counted on its own.
- **Counting** (`_count_layouts`). Backtracking assigns cells in row-major order and prunes any constraint whose residual need is negative or larger than its unassigned cells. The memo key is (cell index, residual needs of open constraints), so each state is solved once. The result is the number of layouts with k mines, for each k.
- **Exact probabilities.** The interior (covered cells touching no number) takes the leftover mines in C(n, k) ways. Combining components by convolution and weighting by C(n, remaining - t) gives the weight of every full layout. A cell's probability is the weight of layouts where it is a mine, over the total. Everything is integer arithmetic until the final division.
- **Agents** (`minesweeper/agents.py`). Random: a uniformly random covered cell. Rules: level 1, guesses at random when nothing is proven. CSP: level 2 proves the cells whose weight is zero (safe) or total (mine), and guesses at random when nothing is proven. Probability: level 3 guesses the covered cell with the lowest exact probability, breaking ties toward the most covered neighbours.
- **Checks** (`tests/test_minesweeper.py`). Counts and probabilities match brute-force enumeration of every layout on 60 random small boards. Every proof is checked against the hidden layout on real games. The probabilities sum to the remaining mine count exactly.

### Hex

- **Rules** (`hexgame/board.py`). An n x n rhombus (7x7 by default, 5 to 11 allowed). DOWN connects top to bottom and ACROSS connects left to right; Hex has no draws, so a full board has exactly one winner. The optional swap rule lets the second player take the first move.
- **Playouts** (`hexgame/mcts.py`). A playout fills the empty cells in a random order and checks the winner once, with one connectivity search at the end. That is far cheaper than playing to a result move by move. The fill also credits moves played after the game was already decided, which is the standard approximation for fill playouts.
- **Search** (`hexgame/mcts.py`). UCT with one node added per iteration, using win rate plus an exploration term with c = 0.6. RAVE keeps all-moves-as-first counts in each node and blends them with the UCT value by beta = sqrt(k / (3n + k)), with k = 300. Both constants were chosen, not tuned.
- **Baselines** (`hexgame/agents.py`, `hexgame/heuristic.py`). Random picks a legal cell. The shortest-path agent is a one-ply race heuristic, not a full two-distance model.
- **Checks** (`tests/test_hexgame.py`). Rules, search, baselines, the CLI, the benchmark writer, and the router, with the two hexgame test files passing.

### Bandits

- **Casino** (`bandits/env.py`). Machine means come from a seeded generator, redrawn until the best leads the second best by 0.05. Outcomes are rolled once per seed for every pull and machine, so all agents face the same luck. Regret is expected: the best mean minus the mean of the machine pulled.
- **Agents** (`bandits/agents.py`). Greedy and epsilon-greedy use the empirical means; UCB1 adds sqrt(2 ln t / n) to each mean; Thompson samples Beta(1 + wins, 1 + losses) per machine and pulls the largest sample; EXP3 keeps exponential weights with importance-weighted rewards and mixing gamma; sliding-window UCB uses the last 200 pulls only, with xi = 0.6 in its bonus.
- **Lai-Robbins** (`bandits/env.py`). For fixed Bernoulli machines the bound is sum over suboptimal arms of gap / KL(p_i || p*) times ln t; for Gaussian machines with known variance it is sum of 2 sigma^2 / gap.
- **Portable randomness** (`bandits/rng.py`). mulberry32 uniforms, Marsaglia polar normals, and Marsaglia-Tsang gammas, so the page's JavaScript port reproduces the Python arm sequences (checked for all 15 agent and casino pairs at seeds 0 to 4).
- **Checks** (`tests/test_bandits.py`, `tests/test_server_bandits.py`). RNG moments, the Lai-Robbins constant against a hand calculation, agent invariants (forced first round, distribution floors, the window forgetting old rewards), regret consistency between a run and scoring its arms, the benchmark shape, and the CLI.

### CartPole

- **Physics** (`cartpole/env.py`). The standard cart-pole equations, integrated by explicit Euler at 0.02 s. The episode ends when |theta| > 12 degrees or |x| > 2.4 m, and is truncated at 500 steps. Reward is 1 per step, so the return equals the balance length.
- **Policies** (`cartpole/nets.py`, `cartpole/train.py`). A one-hidden-layer tanh MLP with a softmax over left and right, trained with hand-written backprop and Adam. REINFORCE uses the batch-mean return as its baseline. The actor-critic uses a learned value network as its baseline, trained on Monte Carlo returns. A TD(0) critic that bootstraps from its own estimate did not learn reliably in the pilot runs, so the shipped actor-critic uses Monte Carlo returns.
- **Cross-entropy search** (`cartpole/train.py`). Samples five-number linear policies, keeps the best quarter, and refits the sampling distribution.
- **Checks** (`tests/test_cartpole.py`). Backprop is compared with finite differences for the policy and the critic, and the physics against the equations worked by hand. The browser port is checked step by step against the Python physics (`/api/cartpole/rollout`).

## Architecture

```
web/            static HTML/CSS/JS, no build step
server/         FastAPI: REST for game moves, a WebSocket that streams N-Puzzle searches
npuzzle/ connect4/ checkers/ routes/ game2048/ sudoku/ lightsout/ blackjack/ battleship/ pacman/ warehouse/ endgame/ sokoban/ wordle/ poker/ minesweeper/ hexgame/ bandits/ cartpole/   search code, training scripts, data files
```

- Searches run in worker threads. A semaphore caps concurrent searches, each client is rate limited, and every request has node and time limits. Searches that keep every state in memory use about 1 KB per expanded node, so they stop at 250,000 nodes; IDS and IDA\* use memory linear in depth and may run longer.
- Trained models ship as `.npz` weights and run in NumPy, so the server image has no torch (about 350 MB) and loads every model in about 120 MB of RAM.

## Running locally

```bash
uv venv && uv pip install -e ".[dev]"      # add ".[train]" for torch and matplotlib
uvicorn server.app:app --reload            # then open http://localhost:8000
python -m pytest -q
```

Command-line tools:

```bash
python -m connect4                                                           # play Connect Four against AlphaZero
python -m game2048                                                           # play 2048 with w/a/s/d
python -m checkers --agent minimax --level 3                                 # play checkers against alpha-beta or minimax
python -m lightsout --solve 110/011/101                                      # the fewest presses that clear a 3x3 board
python -m blackjack                                                          # play blackjack against the dealer
python -m blackjack simulate --hands 100000                                  # measured return of basic strategy through a 6-deck shoe
python -m battleship                                                         # play Battleship in the terminal
python -m battleship benchmark --games 100 --seed 1                          # shots to sink a random fleet, per agent
python -m pacman                                                             # play Pac-Man in the terminal
python -m pacman benchmark --games 200                                       # win rate and score for each agent
python -m warehouse                                                          # plan robot routes in the terminal
python -m warehouse benchmark --layout bottleneck --robots 6 --instances 10  # compare the three planners
python -m endgame                                                            # KQK: a random winning position, you are the lone king
python -m endgame --piece R --as strong                                      # KRK: you have the rook; the agent defends and grades your moves
python -m endgame analyze "8/8/8/5k2/8/8/1Q6/K7 w - - 0 1"                   # every move's distance to mate and the agent's choice
python -m endgame stats                                                      # counts by result, distance-to-mate histograms, build time
python -m endgame build                                                      # re-solve, verify, and write endgame/data
python -m sokoban                                                            # play Sokoban in the terminal (w a s d move and push; u undo, h hint, x solve)
python -m sokoban watch --level 9                                            # the solver's plan for a level, animated
python -m sokoban solve --level 12 --algorithm astar --heuristic matching    # one search: nodes expanded and the plan
python -m sokoban benchmark --out results/sokoban_benchmark                  # every search rule on every level
python -m wordle                                                             # you guess; h = hint with the solver's top guesses and their bits
python -m wordle solve                                                       # the solver guesses; you type the feedback (gy..g)
python -m wordle watch --answer crane                                        # the solver plays a secret word and explains each guess
python -m wordle benchmark                                                   # all answers, all strategies; writes results/wordle_benchmark.*
python -m poker                                                              # play Leduc hold'em against the CFR+ bot (h = hint)
python -m poker train --save                                                 # train CFR and CFR+ on Leduc and write the committed files
python -m poker train --game kuhn                                            # train Kuhn and check the game value against -1/18
python -m poker exploit                                                      # best responses and exploitability of the committed strategy
python -m poker benchmark --deals 100000                                     # the bot against each baseline; writes results/poker_benchmark.*
python -m minesweeper                                                        # beginner 9x9: you play (click reveals, flag mode flags, h hint)
python -m minesweeper -p expert --seed 7                                     # 30 columns x 16 rows, 99 mines, repeatable
python -m minesweeper watch --agent probability -p intermediate --seed 3 --delay 0.2
python -m minesweeper benchmark                                              # rewrites results/minesweeper_benchmark.json and .md (about 10 minutes)
python -m hexgame                                                            # 7x7, you are DOWN, against RAVE-MCTS
python -m hexgame play -n 9 --you across --swap                              # swap rule on, you are ACROSS
python -m hexgame watch --red rave --blue uct                                # two agents, one game
python -m hexgame benchmark --write                                          # the full run in results/hexgame_benchmark.*
python -m bandits play --pulls 200                                           # pull the slot machines yourself, then rank against the agents
python -m bandits watch --agent thompson --pulls 60 --delay 0.2              # watch one agent learn, with its reason for each pull
python -m bandits benchmark                                                  # every agent on every casino, 100 seeds; writes results/bandits_benchmark.*
python -m cartpole watch --agent reinforce --seed 3                          # ASCII animation of one episode
python -m cartpole play                                                      # you push: a (left) and d (right), then Enter
python -m cartpole train --algo reinforce --seeds 0 1 2 3 4 --episodes 1500
python -m cartpole benchmark --episodes 200
python -m routes --compare --cities 12                                       # compare the TSP solvers on a random map
python -m npuzzle astar 7,2,4,5,0,6,8,3,1                                    # any algorithm by name
python -m npuzzle.benchmark 8puzzle                                          # results/npuzzle_8puzzle.md
python -m sudoku.generate --count 200                                        # unique-solution puzzles
python -m sudoku solve 009000000160004023000009...                           # or: python -m sudoku benchmark
python -m game2048.benchmark greedy --games 1000
```

Training (CPU is enough; times are for a laptop i7):

```bash
python -m npuzzle.pdb                              # pattern database, ~40 s
python -m npuzzle.train --minutes 45               # neural heuristic
python -m connect4.train_az --hours 3              # AlphaZero self-play (--resume continues; --device cuda --amp on a GPU)
python -m connect4.evaluate                        # AlphaZero vs minimax at depths 2, 4, 6
python -m game2048.train_td --minutes 90           # n-tuple network
```

## Deploying

`Dockerfile` builds the serving image and `fly.toml` runs it on one always-on Fly.io machine (shared CPU, 1 GB):

```bash
fly launch --no-deploy   # first time: create the app
fly deploy
```
