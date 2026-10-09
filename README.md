# search-agents

Twenty-six puzzles and games, each played or solved by a classic AI algorithm: search guided by hand-built heuristics, pattern databases, or models trained from self-generated data; exact solvers, including retrograde analysis for endgame tablebases; reinforcement learning; and multi-agent pathfinding. Every algorithm runs in Python, in the terminal and behind a browser frontend.

| Domain | Classic search | Learned guidance | Live demo shows |
|---|---|---|---|
| **N-Puzzle** | BFS, DFS, IDS, UCS, bidirectional BFS, greedy, A\*, weighted A\*, IDA\* | Neural cost-to-go heuristic trained by approximate value iteration on top of a 5-5-5 pattern database | Every expansion as it happens, depth vs heuristic plots, side-by-side algorithm comparison |
| **Connect Four** | Alpha-beta negamax, UCT Monte Carlo tree search | AlphaZero-style PUCT with a policy-value ResNet trained only by self-play | The network's prior vs search visits per column, and its win probability |
| **Checkers** | Plain minimax, alpha-beta with iterative deepening, a transposition table, and move ordering; Monte Carlo tree search (UCT) with capped, seeded rollouts; greedy and random baselines | Hand-built evaluation (material, advancement, home row, centre), also used to score MCTS rollouts | How many positions it searched and how many branches pruning cut, with every candidate move's score; MCTS win rates; W/D/L matches between any two agents |
| **Battleship** | Bayesian probability targeting: counts every fleet layout that fits what it has seen, exactly when that is cheap and by importance sampling otherwise | Hunt/target (checkerboard parity, then the neighbours of wounded ships) and random firing | Its odds of every cell of your fleet glowing live, and the cell it fires at next |
| **Blackjack** | Exact dynamic programming over the dealer's and player's hands (a Markov decision process), plus Monte Carlo control | None needed: the values are exact for the infinite-deck model | The full basic-strategy table, the expected value of each move for your hand, and a learner converging on the same table |
| **Route planner** (traveling salesman) | Nearest neighbour + 2-opt; exact Held-Karp dynamic programming up to 12 cities | Simulated annealing and a genetic algorithm (order crossover, inversion mutation, elitism) | Each route untangling live, length and temperature charts, a three-way race, and drawing your own route to compare |
| **2048** | Expectimax with six hand-crafted features, weights tuned by the cross-entropy method | N-tuple network trained by TD(0) on afterstates, used greedily or inside expectimax | Expected value of each move, search depth |
| **Sudoku** | Backtracking, MRV + forward checking | Constraint propagation (naked and hidden singles) | Every guess, forced fill, and backtrack, replayed |
| **Lights Out** | Gaussian elimination over GF(2), exact (no search): the null space gives every solution and the lightest one is the answer | None: the answer is exact, so there is nothing to learn | The augmented matrix reducing one pivot at a time, the rank, and which boards can never be cleared |
| **Endgame tablebases** (king and queen or rook against a lone king) | Retrograde analysis: every position's exact distance to mate, built backwards from checkmate. No search at play time | None: the tables are exact, so there is nothing to learn | Each legal move's distance to mate, the agent's choice against the alternatives, the lone king's heat map, and the distance-to-mate histogram over all 368,452 KQK positions |
| **Sokoban** | Search over pushes (player walks are free, so states are box layouts and the player's region): BFS, greedy best-first, and A\* with a sum-of-nearest-goal bound or a minimum-cost box-to-goal matching (Hungarian), each with or without deadlock pruning (dead squares, frozen boxes) | None needed: the search is exact, and both bounds are admissible (they never overestimate) | The next push the solver suggests, the matching lines from each box to its goal, dead squares tinted red, and the nodes expanded with and without deadlock pruning, side by side |
| **Pac-Man** | A* routes for ghosts with chaser, ambusher and scatter personalities, or chance ghosts that roll fixed odds; random and greedy reflex baselines | Approximate Q-learning over 13 hand-built features (one is a six-turn survival search over the ghosts' real moves), trained by epsilon-greedy self-play | Each ghost's A* route as glowing lines, every move's Q-value, and the feature contributions behind it |
| **Warehouse robots** (multi-agent pathfinding) | Independent A\*, prioritized planning (robots planned one at a time, earlier paths as moving obstacles) | Conflict-Based Search: optimal sum of costs, branching on each collision and replanning only the constrained robot with space-time A\* | Each robot's path step by step with glowing trails, collisions flashing, the constraint tree growing as CBS branches, and a race between the three planners |
| **Wordle** | Information theory: pick the guess with the most expected bits (entropy of the feedback patterns over the candidates); minimax (smallest worst-case bucket); random consistent word as the baseline | None needed: a guess is scored exactly against every answer, so the search is exhaustive over the allowed guesses | Every candidate's feedback and bucket size, the bits each guess is expected to give against the bits it actually gave, and the remaining words after each guess |
| **Poker** (Leduc hold'em, imperfect information) | Exact best response and exploitability on the game tree (no search); the Kuhn value is checked against -1/18 | Counterfactual regret minimization by self-play: vanilla CFR and CFR+, with the average strategy as the answer | The bot's actions, its probability bars for every decision once the showdown shows them, a hint with the equilibrium mix for your card, why its bets are bluffs, and exploitability falling over iterations |
| **Minesweeper** | Constraint satisfaction on the frontier, then exact probabilities: each number is a constraint, components are counted by memoized backtracking, and the global mine count weights every layout | None: the counts are exact, so the agent proves what it can and prices the rest | Which cells are proven safe or mines, the constraint components, and the exact mine probability of every covered cell |
| **Hex** | Monte Carlo tree search with RAVE: UCT statistics blended with all-moves-as-first statistics, beta = sqrt(k / (3n + k)); shortest-path and random baselines | None: every playout is a random fill, so nothing is learned | The agent's visit heat map and principal variation, a hint with visits and win rates, and agent-versus-agent games |
| **Bandits** (slot machines) | Greedy, epsilon-greedy (fixed and decaying), UCB1, and Thompson sampling with Beta posteriors; EXP3 and sliding-window UCB for drifting payouts | None: each agent learns only from the rewards it sees, and Thompson's posteriors narrow as it pulls | Each agent's regret curve and best-machine share, Thompson's Beta posteriors and UCB bars per machine, and your own regret ranked against the agents on the same machines |
| **CartPole** | Policy gradients in pure NumPy: REINFORCE with a baseline, actor-critic with a learned value baseline trained on Monte Carlo returns, and a cross-entropy search over linear policies | None: the policies are learned from reward alone. The PD controller is hand-tuned, not learned | The pole's angle and the cart's position as the policy's probabilities and the critic's value change, and the learning curves across seeds |
| **N-Queens** | Backtracking with column and diagonal bitmasks (exhaustive, so it proves infeasibility), steepest-ascent hill climbing with restarts, simulated annealing on the conflict count, min-conflicts from a greedy start | None: local search over one-queen-per-row boards, with no learned component | Four agents on one board: backtracking blowing up past 32 queens, hill climbing stalling on plateaus, and min-conflicts solving a live pixel board of a thousand queens |
| **Snake** | Cross-entropy method tuning eight weights of an evaluation function over hand-built features of each move (no gradients), 90.1 mean apples on held-out boards; a genetic algorithm evolving a 339-weight net is the second learned agent | BFS path planner to the food with a tail-chasing safety check, as the classic baseline | Each move's weighted terms and scores for the evaluation function; the net's senses, hidden units, and output probabilities |
| **Rover** (D* Lite on a map learned from a sensor) | D* Lite incremental replanning, which repairs only the cells a new wall affects, against A* replanned from scratch | None: the rover learns the map from its sensor, not from training | The learned map, the planned route and its replans, and the expansion counts of both planners |
| **Ghost Hunt** (invisible ghosts from noisy sonar) | Hidden Markov model: exact forward filtering (predict with a known motion model, weigh by the sonar likelihood), a particle filter approximation, and Viterbi for the most likely path after a bust | None: the motion model and the sonar noise are known, so the belief is computed, not learned | The belief fog for each ghost, the particle cloud, sonar rings, the autopilot's bust decisions, and a Viterbi replay of the path against the true one |
| **Lost Robot** (Bayes filtering on a floor plan) | A histogram Bayes filter (exact on its bins) against Monte Carlo localisation with systematic resampling and augmented-MCL kidnap recovery | None: the robot never learns the map; the filters use the known floor plan | The belief over where the robot is, the particle cloud, the range beams, and the grid heat map |
| **Tetris** (placement search) | Placement search over every rotation and column, scored by nine board features; a cross-entropy search tunes the weights | None for play: the weights are tuned, not learned end to end. The tuned v3 weights beat the hand-picked set on held-out 10-row games (one run, 6 iterations) | The chosen placement as a ghost, the nine feature values, and the tuning history |
| **Cluster lab** (a lab, not a game) | k-means (Lloyd's algorithm with k-means++ starts), DBSCAN, and EM for a Gaussian mixture | None: unsupervised, so the lab shows how each method finds groups in points you paint | The step-by-step cluster memberships and centroid moves, DBSCAN's core and border points, the fitted Gaussian ellipses, and inertia, silhouette and elbow charts |
| **Neural net lab** (a lab, not a game) | A NumPy multilayer perceptron with hand-written backprop, mini-batch Adam or SGD, L2, and five seeded datasets; the browser runs the same maths, checked against the Python code by a parity test | None: the network learns from the labelled points you place | The forward pass, the loss, backprop, and each neuron's own heat map, updated live as the network trains |
| **Pathfinding arena** (a lab, not a game) | Breadth-first, depth-first, uniform cost (Dijkstra), greedy best-first, A* (Manhattan, Euclidean, octile), weighted A*, and bidirectional BFS, on seeded mazes with weighted terrain and optional diagonal moves | None: classic search, checked against a Bellman-Ford reference and the weighted-A* bound | Seven searches race one map, with expansions, frontier size, path cost, and whether each path is optimal |
| **Gridworld MDP** (a lab, not a game) | Value iteration, policy iteration, and tabular Q-learning or SARSA on a slippery grid with rewards, pits and walls | None: the planners are exact; the learner uses only the rewards it sees | The value heat map and greedy arrows per sweep, policy iteration's evaluate-improve rounds, and the learner's returns against the optimal policy |
| **Optimizer race** (a lab, not a game) | Gradient descent, momentum, Nesterov, RMSProp, Adam and AdaGrad from one start point on six loss landscapes and a paint-your-own one | None: the update rules are hand-written, and the race is a comparison of their step sizes and geometry | The trails over the loss heat map, the loss chart, and each optimizer's settle or divergence badge |
| **Tree lab** (a lab, not a game) | CART decision trees (Gini or entropy, exhaustive midpoint splits, cost-complexity pruning) and bagged random forests with out-of-bag scoring | None: greedy splits and bagging, with no learned features | The growing tree and its regions over the plane, the pruning view, and the forest's out-of-bag and test accuracy |
| **Markov text** (a lab, not a game) | Order-n Markov chain counted from public-domain text, with add-alpha smoothed next-token odds, temperature, and held-out perplexity per order | None: the counts are the model, with no gradient training | The next-token odds, the context window, and a copy meter that shows how much of the output is verbatim from the source |
| **Regression** (a lab, not a game) | Least squares by normal equations and by Householder QR (compared), ridge penalties, gradient descent, and logistic regression by gradient descent | None: closed-form and gradient fits, with no learned features | The fitted curve and its squared residuals, the train and test error by degree, the descent path, and the logistic boundary |
| **Evolving walkers** (a lab, not a game) | Genetic algorithm over soft-body creatures (point masses, springs, pulsing muscles) with NEAT-style speciation, index-aligned crossover, and graph mutation | None: the gaits are not coded, and selection sees only distance | A population of creatures racing right, the champion gaits, species, a hall of fame, and a creature lab to drop your own design in |
| **404 page** (tic-tac-toe, a page, not a game) | Alpha-beta minimax, solved from the empty board and checked against a full Python reference | None: the game is solved, so the agent plays perfectly | The agent's live search tree on the empty board, with its pruned branches |
| **Nonogram** | Line solving by automaton reachability (the clue as a finite automaton, forward and backward sweeps over the cells), then DPLL SAT or a guess-and-propagate search where line solving stops | None: the answer is exact once the search finishes, and the solver counts solutions to prove uniqueness | Each row and column's forced cells, the DP over positions, the SAT search counters, and the hint's reason for each cell |

**Live demo: https://kb-search-agents.fly.dev** (the first request after idle can take a few seconds while the server wakes up).

Labs are playgrounds where you change the inputs and watch an algorithm respond. They are listed in their own menu and are not counted as games. The labs so far are the Cluster lab, the Neural net lab, the Pathfinding arena, the Gridworld MDP lab, the Optimizer race, the Tree lab, the Markov text lab, the Regression lab, and the Evolving walkers lab.

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

Both agents assume the opponent always answers with its best reply. On a position 8 random legal plies into a game (seed 1 of 10 such positions; the search counts are in `results/checkers_search_nodes.md`), searched to a fixed depth, they choose the same move with the same score. The alpha-beta counts are the iterative-deepening search summed over depths 1 to d, as the agent runs it, so they include the shallower passes. Across all 10 positions, the scores match at every depth and the moves match on 7 of 8 non-forced roots, with the mismatches being ties:

| Depth (half-moves) | Minimax positions | Alpha-beta positions | Fewer by |
|---|---|---|---|
| 4 | 2,962 | 618 | 4.8x |
| 5 | 16,845 | 1,252 | 13x |
| 6 | 92,608 | 3,566 | 26x |

The move generator matches the published perft counts from the opening (7, 49, 302, 1,469, 7,361 positions at depths 1-5; `results/checkers_search_nodes.md`).

### Checkers match results (8x8, level 1)

`python -m checkers match FIRST SECOND --games 10 --level 1`, run for every pair of the five search and baseline agents (`results/checkers_match_8x8_level1.txt`), and then alpha-beta against chance (`results/checkers_match_alphabeta_chance_8x8_level1.txt`; the chance row was added after the round robin, so its totals are not in the 40-game line below). Games come in pairs that start from the same seeded random 4-ply opening, with the colours swapped, so no agent is always first or always red. Alpha-beta uses its node budget here, so the file replays exactly. Forced captures. Ten games per pair is a small sample, so read these as a rough ordering at this strength.

| First | Second | First W-D-L | Second W-D-L |
|---|---|---|---|
| alpha-beta | minimax | 7-3-0 | 0-3-7 |
| alpha-beta | MCTS | 5-5-0 | 0-5-5 |
| alpha-beta | greedy | 10-0-0 | 0-0-10 |
| alpha-beta | random | 10-0-0 | 0-0-10 |
| minimax | MCTS | 1-6-3 | 3-6-1 |
| minimax | greedy | 10-0-0 | 0-0-10 |
| minimax | random | 10-0-0 | 0-0-10 |
| MCTS | greedy | 9-1-0 | 0-1-9 |
| MCTS | random | 10-0-0 | 0-0-10 |
| greedy | random | 7-3-0 | 0-3-7 |
| alpha-beta | chance | 10-0-0 | 0-0-10 |

Totals over 40 games each: alpha-beta 32-8-0, MCTS 22-12-6, minimax 21-9-10, greedy 7-4-29, random 0-3-37.

### Battleship: shots to sink a random fleet

100 random fleets on a 10x10 board with the standard fleet (5, 4, 3, 3, 2; 17 ship cells). Every agent plays
the same fleets, so the comparison is paired. From `python -m battleship benchmark --games 100 --seed 1` (`results/battleship_benchmark.md`):

| Agent | Mean shots | Median | Worst | Time per game (Python, under load) |
|---|---|---|---|---|
| Random | 95.4 | 97 | 100 | 0.006 s |
| Hunt/target (checkerboard, then neighbours) | 51.9 | 54 | 67 | 0.007 s |
| **Bayesian probability targeting** | **45.9** | **45.5** | **64** | 1.1 s |

Head to head on the same fleets, the Bayesian agent needs fewer shots than hunt/target on 67 of 100 fleets (3 ties), 6.0 shots fewer on average (95% t interval 3.7 to 8.2; `results/battleship_head_to_head.json`).

**Chance (fixed odds) as an opponent.** Chance fires at a random unknown cell, weighted by a fixed table: a cell's weight is `profile[row] * profile[col]` with profile `(1, 2, 3, 4, 4, 4, 4, 3, 2, 1)`, so the middle of the board is favoured 16 to 1 over a corner. It ignores hits and misses and does no search. It is the Battleship version of the 2048 spawner's fixed split. `python -m battleship match --games 100 --seed 1` races the Bayesian AI against chance, each with its own random fleet, shots alternating; the first fleet sunk wins. The AI won 100 of 100 (chance 0; `results/battleship_match.md`). Hunt/target vs chance: 100 of 100 for hunt/target. Races cannot be drawn.

The Bayesian agent pays per shot: it recounts consistent layouts (or samples 1000 of them) before every
shot, so its time per game is well over 100 times the baselines'.

### Blackjack: exact basic strategy and a learner that finds it

- **House edge** with perfect basic strategy, solved exactly (infinite deck, dealer stands on soft 17, dealer peek, 3:2 blackjack, double on any two cards, one split with doubling after it): **0.570%** of each bet.
- **Measured** through a 6-deck shoe over 200,000 rounds with the same table (`results/blackjack_simulate.md`): **-0.66%** per unit bet, standard error 0.26%, consistent with the exact figure (a finite shoe is not the infinite-deck model).
- **Monte Carlo control** (exploring starts, every-visit averages, seed 1), agreement with the exact table on the 321 decisive cells: **73%** after 40,000 hands (mean value lost per cell 0.039) and **78%** after 100,000 hands (0.023). Near-tied cells need far more hands, so agreement rises slowly.

### 2048: TD-learned n-tuple network

The network has four 5-cell patterns and two 6-cell patterns (`PATTERNS` in `game2048/ntuple.py`; the shapes are in `web/2048.html`), each read under all 8 board symmetries. Trained with `python -m game2048.train_td --minutes 70 --games 1000 --alpha 0.1 --seed 0`. The trainer logs once a minute; the current network's run predates the final record it now writes at exit, so the counts cited are the last logged records: the current network at 45,514 games and 69.04 minutes (`results/game2048_train_log_v2.jsonl`), and the earlier six-5-cell network at 42,980 games and 64.1 minutes (`results/game2048_train_log_v1.jsonl`). Held-out games use the seed 500000 for their batch, which neither the trainer nor the tuner used. Intervals are 95% (normal for the mean, Wilson for the rates).

Greedy, no search, each move chosen by reward + learned afterstate value (`results/game2048_ntuple_v1.md`, `results/game2048_ntuple_v2.md`):

| Network | Games | Mean score (95% CI) | Median score | Reached 2048 (95% CI) | Reached 4096 (95% CI) | Reached 8192 (95% CI) |
|---|---|---|---|---|---|---|
| Earlier, six 5-cell patterns (`results/game2048_ntuple_v1.npz`) | 3,000 | 43,970 (43,168 to 44,773) | 41,482 | 83.5% (82.1 to 84.8) | 32.1% (30.5 to 33.8) | 2.1% (1.6 to 2.7) |
| Current, two 6-cell patterns (`game2048/data/ntuple_2048.npz`) | 3,000 | 54,835 (53,791 to 55,879) | 46,340 | 88.1% (86.9 to 89.2) | 48.7% (46.9 to 50.5) | 6.7% (5.9 to 7.7) |

With expectimax search on top (50 ms per move, learned value as the leaf score; 12 games per network, so the intervals are wide; `results/game2048_ntuple_search_v1.md`, `results/game2048_ntuple_search_v2.md`):

| Network | Games | Mean score (95% CI) | Median score | Reached 2048 (95% CI) | Reached 4096 (95% CI) | Reached 8192 (95% CI) |
|---|---|---|---|---|---|---|
| Earlier | 12 | 46,459 (32,404 to 60,513) | 44,128 | 75.0% (46.8 to 91.1) | 41.7% (19.3 to 68.0) | 0.0% (0.0 to 24.3) |
| Current | 12 | 104,144 (82,370 to 125,919) | 90,894 | 100.0% (75.7 to 100.0) | 91.7% (64.6 to 98.5) | 41.7% (19.3 to 68.0) |

The current network's tables take 72 MiB in memory as float16 (the earlier one took 12 MiB as float16, or 24 MiB as float32 in training). The 6-cell tables are stored dense, so each costs 32 MiB in float16 whether or not training visited its entries. The file on disk is 9.8 MB. A scalar evaluation takes about 65 µs per board (best of five runs over 2,000 random boards, on a shared laptop under load; the median run was 120 µs).

The hand-crafted evaluation's six weights (`game2048/data/heuristic_weights.json`, identical to `results/game2048_heuristic_weights_v1.json`) came from an unbounded tuning run with no recorded settings. The two bounded runs, `results/game2048_heuristic_weights_v2.json` and `_v3.json`, both used `python -m game2048.tune --generations 20 --population 32 --elite 8 --games 300 --seed 0`. The weights were checked on held-out games. Intervals are 95%.

Greedy one-move lookahead on the features (the tuner's objective; 3,000 games per row, one shared seed 500000 for the batch; `python -m game2048.tune --evaluate game2048/data/heuristic_weights.json results/game2048_heuristic_weights_v2.json results/game2048_heuristic_weights_v3.json --games 3000 --seed 500000 --out results/game2048_heuristic_greedy_heldout.json`, which also writes the `.md`):

| Weights | Smoothness weight | Mean score (95% CI) |
|---|---|---|
| Shipped, unbounded run (same weights as `results/game2048_heuristic_weights_v1.json`) | −0.129 | 7,074 (6,930 to 7,219) |
| Bounded at zero (`results/game2048_heuristic_weights_v2.json`) | 0 | 5,823 (5,709 to 5,936) |
| Bounded at zero, second run (`results/game2048_heuristic_weights_v3.json`) | 0 | 5,768 (5,657 to 5,880) |

Expectimax search with the features (50 ms per move; 60 games per row, seeds 500000 to 500059; `results/game2048_heuristic_v1_gap_baseline.md`, `results/game2048_heuristic_combined.md`):

| Weights and smoothness | Mean score (95% CI) | Reached 2048 (95% CI) |
|---|---|---|
| Shipped weights, smoothness skips empty cells (current code) | 16,954 (15,259 to 18,650) | 11.7% (5.8 to 22.2) |
| Bounded weights, second run (`_v3`, smoothness weight 0) | 18,639 (16,461 to 20,817) | 20.0% (11.8 to 31.8) |
| Shipped weights, earlier smoothness definition (touching tiles only; not in the current code) | 16,988 (15,131 to 18,844) | 10.0% (4.7 to 20.1) |

The sign question: a negative weight on the penalty term rewards rougher rows and columns, and the greedy objective prefers that by about 1,250 points. Bounding the weights at zero (the tuner now does this) therefore costs greedy play. The bounded run also scored about 1,700 points higher in expectimax, but the intervals overlap (the difference is not significant). The shipped weights were kept, as the bounded set was not measurably better. The shipped smoothness weight stays negative and the README says so; the smoothness reading "touching tiles only" gave the same expectimax result as the shipped version, so the code keeps the shipped definition.

### Sudoku: 200 generated puzzles with unique solutions

| Solver | Solved | Mean guesses | Max guesses |
|---|---|---|---|
| Plain backtracking | 198/200 (2M-guess limit) | 119,468 | 2,000,001 |
| MRV + forward checking | 200/200 | 364 | 4,058 |
| Constraint propagation + MRV | 200/200 | 7 | 89 |

### Connect Four: AlphaZero-style self-play

Trained only by self-play: 37 iterations on a laptop CPU, then 59 on a Kaggle notebook's CPU (4 threads; no GPU was attached). The served network is the export from iteration 96, after 34,944 self-play games (the file records `iteration` 96 and `games` 34944). Its strength with 200 PUCT simulations per move, 100 games per opponent, colours alternating and two random opening moves per game (`python -m connect4.evaluate --games 100 --depths D`, one run per depth; `results/connect4_eval.json`), next to the iteration-19 network served before it (2,432 games), on the same openings:

| Opponent | AlphaZero wins | Draws | Losses | Score | Iteration 19 |
|---|---|---|---|---|---|
| Alpha-beta minimax, depth 2 | 67 | 8 | 25 | 71% | 61-5-34 (63.5%) |
| Alpha-beta minimax, depth 4 | 79 | 4 | 17 | 81% | 53-10-37 (58%) |
| Alpha-beta minimax, depth 6 | 77 | 8 | 15 | 81% | 29-20-51 (39%) |

For reference, plain MCTS with 1,000 random-rollout simulations scores 32-2-16 against depth-4 minimax (50 games). How it was trained, on what hardware and for how long: [docs/connect4-training.md](docs/connect4-training.md).


### Lights Out: exact linear algebra over GF(2)

Pressing a light is addition mod 2, so a board is a linear system A x = b with a 25x25 matrix of rank 23. That splits the boards: only 1 in 4 can be cleared at all (736 of 3,000 uniform random boards), and every solvable 5x5 board has exactly 4 solutions. Over 3,000 random solvable boards, the lightest solution averages 9.9 presses, and the heaviest of the four is 5.2 presses heavier on average. Solving a 5x5 board takes about 0.7 ms in Python (3,000 solves on a shared laptop under load; `results/lightsout_stats.md`).

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

REINFORCE reaches 475 in 376 to 478 episodes, actor-critic in 344 to 382, and cross-entropy in 299 to 528. The shipped policy for each agent is the seed with the best last 50 episodes (`results/cartpole_learning.md`). Ties go to the seed listed first: several seeds reach the 500-step cap, so their last-50 means tie at 500.

### N-Queens: local search repairs what backtracking cannot finish

Min-conflicts solves every sampled board from 8 to a million queens, and its work barely grows: a median of 24 repairs at 8 queens, 56 at 10,000, and 47 at a million (one run). Backtracking solves 8 queens with 113 squares scored and 16 queens in about 10,000, then hits its 2,000,000-step cap at 32 queens and above. Hill climbing and annealing solve 32 queens, but none of their runs finish at 128 or above within their time caps.

| Agent | N | Runs | Solved | Steps (median, solved) | Squares scored (median, solved) | Time (median) | Unsolved runs ended by |
|---|---:|---:|---:|---:|---:|---:|---|
| Backtrack | 8 | 1 | 1 | 218 | 113 | 0.0001 s | — |
| Backtrack | 16 | 1 | 1 | 20,088 | 10,052 | 0.0263 s | — |
| Backtrack | 32 | 1 | 0 | — | — | 2.21 s | step cap |
| Backtrack | 128 | 1 | 0 | — | — | 2.28 s | step cap |
| Backtrack | 1,000 | 1 | 0 | — | — | 3.25 s | step cap |
| Backtrack | 10,000 | 1 | 0 | — | — | 5.77 s | step cap |
| Hill climbing | 8 | 200 | 200 | 22 | 1,440 | 0.0002 s | — |
| Hill climbing | 16 | 100 | 100 | 170 | 43,392 | 0.0061 s | — |
| Hill climbing | 32 | 30 | 30 | 1,024 | 1,049,088 | 0.142 s | — |
| Hill climbing | 128 | 3 | 0 | — | — | 15 s | time cap |
| Hill climbing | 1,000 | 2 | 0 | — | — | 20.1 s | time cap |
| Annealing | 8 | 200 | 200 | 778 | 778 | 0.0016 s | — |
| Annealing | 16 | 100 | 100 | 13,420 | 13,420 | 0.0381 s | — |
| Annealing | 32 | 30 | 30 | 712,202 | 712,202 | 1.76 s | — |
| Annealing | 128 | 5 | 0 | — | — | 20 s | time cap |
| Annealing | 1,000 | 3 | 0 | — | — | 30 s | time cap |
| Annealing | 10,000 | 2 | 0 | — | — | 55.6 s | step cap |
| Min-conflicts | 8 | 200 | 200 | 24 | 4,826 | 0.0026 s | — |
| Min-conflicts | 16 | 200 | 200 | 44 | 9,976 | 0.0043 s | — |
| Min-conflicts | 32 | 100 | 100 | 44 | 13,554 | 0.0047 s | — |
| Min-conflicts | 128 | 100 | 100 | 52 | 26,432 | 0.0088 s | — |
| Min-conflicts | 1,000 | 50 | 50 | 54 | 85,032 | 0.0209 s | — |
| Min-conflicts | 10,000 | 20 | 20 | 56 | 624,649 | 0.149 s | — |
| Min-conflicts | 100,000 | 3 | 3 | 116 | 11,941,980 | 2.05 s | — |
| Min-conflicts | 1,000,000 | 1 | 1 | 47 | 50,094,321 | 13.6 s | — |

Seeds 1000 onward, the same seeds for every agent. Each agent's step counts its own unit (defined in `queens/agents.py`), so the squares-scored column is the comparable work measure. Hill climbing at N = 10,000 is not run: one step scores 10^8 squares. The full table is in `results/queens_benchmark.md`.

### Snake: learned agents against a BFS planner

Two agents are learned here, and neither uses gradients. Both are shown on the page, and the one that scores best on held-out boards is the default.

**The evaluation function (v4, the default).** Each move that does not end the game is scored by a weighted sum of eight features of the position it leads to: apple, food near, food reachable, area, room for body, tail reachable, exits, and length. The move with the largest sum is taken. The features and the rule are hand-built (`snake/evaluator.py`); only the eight weights are learned, by the cross-entropy method (`snake/cem.py`). Each generation has 48 candidate weight vectors, each scored on 6 fresh seeded boards that the whole generation shares. Starving counts as a death. The best 12 refit a diagonal Gaussian, and its spread is floored at 0.05. `python -m snake cem` runs 30 generations (about 55 minutes on two loaded cores). The champion is the generation-30 candidate that scored best on 32 fixed validation boards, seeds 40000 to 40031, which no training generation draws. Its weights are committed at four decimals in `snake/data/evaluator.json`:

| Feature | Weight |
|---|---:|
| apple | 0.7266 |
| food near (1 - BFS distance / 24) | 0.8261 |
| food reachable | -0.6941 |
| area (reachable cells / empty cells) | -0.2007 |
| room for body (reachable / length, capped at 1) | 0.3796 |
| tail reachable | 1.5404 |
| exits (empty neighbours of the head / 4) | 0.1065 |
| length (over the number of cells) | 0.7549 |

**The neural net (v2).** A 339-weight network (17 senses, 16 tanh hidden units, 3 turn outputs) is evolved by a genetic algorithm: population 60, 8 fresh training boards per generation, 80 generations, elites 4, tournament 3, mutation 0.1 at sigma 0.15, seed 1. `python -m snake evolve` takes about 10 minutes on two laptop cores. The champion is the generation-80 genome that scored best on 16 fixed validation boards, seeds 20000 to 20015. Those seeds fall inside the training range (1000 to 39999), so that score is a same-distribution check, not a held-out one; the benchmark seeds (50000 and up) are held out.

Benchmark: 200 seeded 12x12 games per agent, seeds 50000 onward, the same boards for every agent. The training seeds are 1000 to 39999, so no benchmark board was used for training, validation, or choosing the default.

| Agent | Mean apples | Median | Max | Death rate | Starved | Full | Mean steps |
|---|---:|---:|---:|---:|---:|---:|---:|
| Random | 0.16 | 0 | 2 | 100.0% | 0 | 0 | 26.9 |
| Greedy (safe move closest to the food) | 20.12 | 19 | 47 | 100.0% | 0 | 0 | 190.9 |
| Planner (BFS to the food, tail-chasing check) | 56.13 | 49.5 | 137 | 0.0% | 200 | 0 | 1241.3 |
| Neural net (v2 champion) | 13.69 | 13 | 26 | 38.5% | 123 | 0 | 457.9 |
| **Evaluation function (v4 champion)** | **90.13** | **86** | **133** | **0.0%** | **200** | **0** | **2436.4** |

**The evaluation function beats greedy (20.12) and the BFS planner (56.13) on mean apples**, with the same boards for every agent. Be clear about what this shows. Every game of the evaluation function and of the planner ends by starving: two laps with no apple. Neither ever dies, but both stall, and the evaluation function eats more before it stalls. No game filled the board. The features and the one-move lookahead are hand-built, so the learned part is the eight weights, not the strategy as a whole. The training curve (`results/snake_eval_train_log.jsonl`) rose from 5.28 mean apples in generation 1 to about 100 by generation 25 and then flattened. The validation score was 96.3 and the benchmark 90.1, so the gap between training and held-out boards is small.

The earlier neural runs did much worse, and their rows are kept for comparison:

| Version | Evolved mean | Median | Max | Death rate | Starved | Mean steps | Files |
|---|---:|---:|---:|---:|---:|---:|---|
| v1: 3 fixed training boards, 14 senses, 291 weights, 100 generations | 2.05 | 1 | 11 | 5.0% | 190 | 308.0 | `results/snake_benchmark_v1.*`, `results/snake_train_log_v1.jsonl` |
| v2: fresh boards per generation, 8 games, starvation penalty, room senses, 17 senses, 339 weights, 80 generations | 13.69 | 13 | 26 | 38.5% | 123 | 457.9 | `results/snake_benchmark_v2.*`, `results/snake_train_log.jsonl` |
| v3a: as v2 but population 100, 200 generations, starvation penalty 2.0 (scratch run, not committed) | 0.93 | 1 | 3 | 100.0% | 0 | 54.5 | not in `results/` |
| v3b: as v3a but starvation penalty 0.5 (scratch run, not committed) | 13.22 | 13 | 30 | 53.5% | 93 | 405.5 | not in `results/` |
| v4: evaluation function, 8 weights, cross-entropy, 30 generations of 48 candidates | 90.13 | 86 | 133 | 0.0% | 200 | 2436.4 | `results/snake_benchmark.*`, `snake/data/evaluator.json`, `results/snake_eval_train_log.jsonl` |
The baselines are the same in every row; only the learned agent changed. Two longer neural runs were tried after v2. v3a, with a starvation penalty of 2.0, made looping cost more than dying early, so the population selected agents that died around step 50 (0.93 apples). v3b, with a penalty of 0.5, reached 13.22 apples, close to v2's 13.69, so v2 stays the neural champion. Neither longer run is committed as a champion. The neural net's training metric was 12.7 apples per game (`train_apples` in `snake/data/champion.json`), close to its 13.7 on the benchmark boards. It stays selectable on the page, and it is still far from the evaluation function.

**The neural net almost never turns right.** Counting its moves over the first 40 benchmark boards (seeds 50000 to 50039) gives 19,829 moves: 17,033 straight (85.9%), 2,793 left (14.1%), and 3 right (0.02%). The benchmark table cannot show this, because the mean score hides which way the snake turns. The counts come from `python -m snake actions --write` and are committed in `results/snake_net_actions.json`. Read the neural rows with this in mind: the net's play is nearly one-sided, not a balanced turning policy.

### Rover: D* Lite vs A* replanned from scratch

Sensor radius 2, five seeded maps per row, 4-connected grid, start top-left, goal bottom-right. Both planners drive the same routes. The columns are node expansions (the work measure); "total" is everything the planner expanded over the whole run, and the ratio is D* Lite over A*.

| Size | Walls | Steps | Replans | D* initial | D* replans | A* initial | A* replans | D* total / A* total | D* ms | A* ms |
|---|---|---|---|---|---|---|---|---|---|---|
| 21x21 | 10% | 45 | 15.6 | 441 | 32 | 41 | 384 | 473 / 425 (1.11x) | 14 | 2 |
| 21x21 | 20% | 47 | 24.4 | 441 | 57 | 41 | 551 | 498 / 592 (0.84x) | 7 | 2 |
| 21x21 | 30% | 46 | 30.2 | 441 | 59 | 41 | 718 | 500 / 759 (0.66x) | 6 | 2 |
| 41x41 | 10% | 87 | 29.6 | 1681 | 55 | 81 | 1279 | 1736 / 1360 (1.28x) | 46 | 9 |
| 41x41 | 20% | 100 | 56.4 | 1681 | 169 | 81 | 2353 | 1850 / 2434 (0.76x) | 41 | 8 |
| 41x41 | 30% | 127 | 86.4 | 1681 | 467 | 81 | 3920 | 2148 / 4001 (0.54x) | 63 | 22 |
| 61x61 | 10% | 135 | 47.0 | 3721 | 100 | 121 | 3030 | 3821 / 3151 (1.21x) | 88 | 25 |
| 61x61 | 20% | 147 | 82.2 | 3721 | 241 | 121 | 5076 | 3962 / 5197 (0.76x) | 93 | 23 |
| 61x61 | 30% | 196 | 131.2 | 3721 | 696 | 121 | 8154 | 4417 / 8275 (0.53x) | 182 | 68 |

**D* Lite's initial plan costs more than A*'s, so it wins on total expansions only from about 20% walls.** On an open map its first search runs from the goal over the whole start-goal box (441 expansions at 21x21 against 41 for A*; 3,721 against 121 at 61x61). Its replans are much cheaper, about 8 to 30 times cheaper than A*'s in every row. At 10% walls D* Lite spends 1.11x to 1.28x as many expansions as A*. At 20% and 30% walls it spends 0.84x down to 0.53x. The timings are one-off Python numbers on one core.

Cost check: the two planners agreed on the cost from the rover's cell at every replan (0 disagreements over 2,515 replans on 45 maps).

### Ghost Hunt: exact belief vs particles, and Viterbi after a bust

Maze 15x15, two ghosts, the autopilot plays (bust when a peak holds over half the ghost's probability and is next to you, capped at 300 turns). Twelve seeded games per row.

| Motion | Sonar (sigma) | Filter | Turns (mean) | All busted | True-cell mass |
|---|---|---|---:|---:|---:|
| random | medium (1.0) | exact | 95.1 | 100% | 0.279 |
| random | medium (1.0) | particles 10 / 50 / 200 | 160.3 / 123.2 / 102.5 | 92% / 83% / 92% | 0.175 / 0.245 / 0.287 |
| random | high (2.0) | exact | 295.4 | 8% | 0.176 |
| lurker | medium (1.0) | exact | 34.8 | 100% | 0.376 |
| patrol | medium (1.0) | exact | 63.1 | 100% | 0.585 |
| patrol | medium (1.0) | particles 10 / 200 | 267.1 / 79.7 | 17% / 92% | 0.121 / 0.357 |

Particle convergence on recorded pings (random motion, medium noise), KL(exact to particles) over cells: 4.91 at 10 particles, 0.42 at 100, 0.059 at 300, 0.016 at 1000. The true-cell mass is 0.281 for the exact belief and 0.285 for 1000 particles.

Viterbi path accuracy, against the filtered MAP cell (the cell with the highest belief from the readings so far), over the turns of busted ghosts: patrol at medium noise 0.893 against 0.680; random at low noise 0.816 against 0.682. For the lurker at medium noise it is 0.491 against 0.511, so there the smoothed path does not beat the filtered cell.

**Few particles lose the ghost.** With 10 to 50 particles on patrol the crowd often collapses onto one corridor, so fewer games finish; at 200 particles it is still slower: on patrol at low noise it takes 137.7 turns against the exact filter's 82.3, and busts 75% of the ghosts against 100%. High sonar noise is hard for every filter, and the autopilot is a simple rule, so the turn counts measure the belief and the policy together. Pings give only distances, so cells at the same distance are indistinguishable and exact-cell Viterbi accuracy stays below 1.

### Lost Robot: where am I on a known floor plan?

The robot knows the floor plan but not where it is. Its range beams see walls at a few distances, and its wheels drift. Two estimators work from the same sensor and motion models: an exact histogram Bayes filter over x, y and heading (16 headings, 0.25 m bins), and Monte Carlo localisation with particles and augmented-MCL kidnap recovery. The benchmark is 16 seeded trials (2 floors, 4 floor seeds, 2 robot seeds), 100 steps each, in Python on one core.

| Particles | Median steps to within 0.5 m | Failures | ms per step |
|---|---:|---:|---:|
| 100 | 12 | 12/16 | 1.0 |
| 500 | 2 | 6/16 | 2.0 |
| 2000 | 2 | 2/16 | 6.3 |

The grid filter fails 0 of 16 runs with a median of 6 steps, at about 14 ms per step. Timings move with machine load; the committed file is from a quiet run.

Kidnapped at step 40 (60-step window):

| Filter | Median steps to recover | Failures |
|---|---:|---:|
| Augmented MCL (replaces a random subset of particles when the fast likelihood average falls) | 11 | 2/16 |
| Plain MCL (no injection) | 18 | 14/16 |
| Grid filter (exact Bayes) | 33 | 1/16 |

Augmented MCL injects 1.3% of particles per step on average when nothing is wrong, and 13% per step after the kidnap. Noise sweeps (sensor sigma 0.15 to 0.6 cells, odometry scale 0.5 to 2) move the failure counts by a few trials, which at 16 trials is not a trend.

**Honest limits**

- **100 particles fail 12 of 16 runs.** The cloud is too thin to keep the possible places alive until one of them wins; 2000 is the page default.
- **Kidnap recovery is slow in look-alike rooms.** Twin Halls has six rooms that look alike from inside, so after a kidnap the beams match several places and the belief splits. Recovery takes a median of 11 steps for augmented MCL and 33 for the grid filter.
- **The page's default seed was chosen to show a visible split.** Seed 4 on Twin Halls starts the belief in three places before it locks. It was picked by search for that picture and is not a benchmark result or a typical run.

```
python -m localize                       # autopilot drives; the particle cloud is drawn on the floor
python -m localize --drive               # drive with w/a/s/d in a terminal (q quits)
python -m localize --kidnap-at 40        # teleport the robot at step 40 and watch the cloud re-spread
python -m localize benchmark             # the seeded benchmark; writes results/localize_benchmark.md
```

### Tetris: tuned v3 weights beat the hand-picked weights on held-out games

The AI places each piece by trying every rotation and column, dropping the piece straight down, and scoring the result with nine board features (landing height, eroded piece cells, row and column transitions, holes, cumulative wells, aggregate height, bumpiness, completed lines). The score is a weighted sum. Tuning sets the nine weights: a genetic algorithm (v2) first, then a noisy cross-entropy search (v3). The default AI uses the tuned v3 weights; the hand-picked set is one button away on the page.

**On the harder benchmark (10-row board, 30 held-out seeds, games run to game over), the tuned v3 weights beat the hand-picked weights: 6,702 lines against 2,128.5, and their 95% intervals do not overlap.** The v2 GA weights (388.9) did not beat the hand-picked set.

`results/tetris_benchmark_hard_v3.md`: 10-row board, 30 held-out seeds (50000-50029), piece cap 100,000. No game reached the cap. The v2 GA row is from `results/tetris_benchmark_hard.md`, under the same protocol.

| Strategy | Mean lines | 95% CI | Median | Min-max | Mean pieces |
|---|---:|---:|---:|---:|---:|
| Random placement | 0.0 | ± 0.1 | 0 | 0-1 | 14 |
| Hand-picked weights | 2,128.5 | ± 895.6 | 867 | 80-8,743 | 5,342 |
| GA v2 weights | 388.9 | ± 109.0 | 311 | 38-1,009 | 993 |
| Cross-entropy v3 weights (default AI) | 6,702.0 | ± 2,194.9 | 6,305 | 84-24,870 | 16,776 |

Caveats, stated plainly:

- The cross-entropy run was stopped after 6 iterations, for time (the `config` in `tetris/tuned.json` records 6).
- It is one training run, so the spread across training runs is unknown.
- It was tuned on 10-row games, while the page plays the 20-row game. The v3 weights have not been benchmarked on the 20-row game; the saturated 10x20 table below predates them.
- Training used fresh seeds per iteration (from 200,000), disjoint from the benchmark seeds.

For the record, the v2 GA was trained on 10-row games capped at 2,500 pieces, with 6 games per genome, population 16, and 16 generations. Its best genome scored 804.5 mean lines on its generation's seeds, an optimistic training figure. The v3 run's final elite mean was 6,869.4 (`train_fitness` in `tetris/tuned.json`), also on training seeds.

**Note: the 10x20 table is saturated and does not separate the strategies (kept for the record).** `results/tetris_benchmark.md`: 10x20 board, 30 held-out seeds, piece cap 2,000.

| Strategy | Mean lines | Hit cap | Topped out |
|---|---:|---:|---:|
| Random placement | 0.2 | 0 / 30 | 30 |
| Hand-picked weights | 797.6 | 30 / 30 | 0 |
| GA-tuned weights (v1, trained on 300-piece games) | 797.2 | 30 / 30 | 0 |
| GA-tuned + 1-piece lookahead (v1) | 798.3 | 30 / 30 | 0 |

At 2,000 pieces every surviving strategy sits at the ceiling: a steady stack clears 4 cells per piece over 10 cells per line, so 2,000 pieces allow at most 800 lines. The table cannot tell them apart. Supplementary run, not committed: greedy only, 20,000-piece cap, 12 games, hand-picked 7,998.0 mean lines (12 of 12 hit the cap); v1 GA-tuned 7,413.6 (11 of 12 hit the cap; one game topped out at piece 2,536).

The page's slider panel shows each live weight beside its tuned value, and the HAND-PICKED button switches the AI to the hand-picked set.

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

Bottleneck map, 6 robots, 10 random instances (`python -m warehouse benchmark --layout bottleneck --robots 6 --instances 10`; `results/warehouse_benchmark_bottleneck.md`). Seconds are measured on a shared laptop under load:

| Planner | Solved | Mean sum of costs | Mean makespan | Mean collisions | Mean CBS nodes | Mean seconds |
|---|---|---|---|---|---|---|
| **Conflict-Based Search** | **10/10** | **52.9** | **15.5** | **0.00** | 693.6 | 0.35 |
| Prioritized planning | 10/10 | 55.9 | 15.7 | 0.00 | - | 0.0020 |
| Independent A\* | 0/10 | 47.7 | 14.3 | 3.80 | - | 0.0006 |

Prioritized planning was suboptimal on 5 of the 10 instances (mean extra cost 3.0). Independent A* is cheapest only because its paths collide: its cost is not a valid plan.

Shelf aisles, 8 robots, 10 instances (`python -m warehouse benchmark --layout aisles --robots 8 --instances 10`; `results/warehouse_benchmark_aisles.md`):

| Planner | Solved | Mean sum of costs | Mean makespan | Mean collisions | Mean CBS nodes | Mean seconds |
|---|---|---|---|---|---|---|
| **Conflict-Based Search** | **10/10** | **75.7** | 17.4 | **0.00** | 135.0 | 0.06 |
| Prioritized planning | 10/10 | 79.5 | 17.1 | 0.00 | - | 0.0007 |
| Independent A\* | 0/10 | 72.9 | 17.0 | 3.50 | - | 0.0006 |

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

CFR+ with alternating updates, 5000 iterations, on the six-card game with a public card (288 information sets). The committed average strategy is exploitable by 0.05 milli-big-blinds per hand (the best responses are exact, so this is not a sampled estimate), and the game value for the first player is -0.0856 chips, which matches the published value for this game. Vanilla CFR reaches 0.0036 chips after the same 5000 iterations. That is about 240 times the exploitability of the unrounded CFR+ run (1.5e-5 chips, the basis here, from `results/poker_train_log.jsonl`), so the CFR+ curve drops much faster. The committed table is rounded to four decimals, which puts it at 1.1e-4 chips, about 34 times below vanilla CFR. On Kuhn poker the value is -0.0555556 against the exact -1/18, an error of 4e-9 for CFR+ after 10000 iterations.

Head-to-head, the CFR+ bot against each opponent over 100,000 seeded deals, each played in both seats (200,000 hands; milli-big-blinds per hand, big blind = 2 chips, 95% CI from the spread over deals):

| Opponent | mbb/hand (bot's side) | 95% CI |
|---|---:|---:|
| random | +358.0 | [+350.2, +365.7] |
| always-call | +317.7 | [+313.1, +322.2] |
| hand-strength (rules on own card and the public card) | +68.2 | [+63.5, +72.9] |
| the bot against itself | +3.1 | [-1.4, +7.6] |

The self-play row is zero within its interval, as equilibrium requires.

The **chance** opponent (`poker/chance.py`) is the poker version of the 2048 tile spawner: it draws each action from fixed odds, fold 15%, call or check 55%, bet or raise 30%, renormalised over the legal actions. It reads no card and runs no search, so its mix is the same at every decision. Against the CFR+ bot, over 200,000 seeded hands:

```
PYTHONPATH=. python -m poker match --opponent chance --hands 200000 --seed 1
```

The CFR+ side wins +358.0 mbb/hand, 95% CI [+350.4, +365.6] (`results/poker_match.md`). The same opponent is playable on the page as "You vs Chance" and "Watch AI vs Chance".

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

### Cluster lab: k-means, DBSCAN and EM on points you paint

Paint points or load a preset (blobs, stretched blobs, rings, moons, uniform noise, a smiley), then watch k-means alternate assign and update steps, DBSCAN grow clusters from core points, or a Gaussian mixture fit ellipses by EM. The page runs in the browser; the Python package `clusters/` runs the same algorithms with the same seeds, and a node parity test checks the page's math against it (labels must match exactly, centres to 1e-9).

This lab has no benchmark table: it is a playground, and its checks are correctness checks. `tests/test_clusters.py`, `tests/test_clusters_parity.py` (needs node) and `tests/test_server_clusters.py`: 61 passed.

### Neural net lab: a small network learns a boundary you draw

Place points in two classes, choose the layers, activation, optimiser and regularisation, then train and watch the decision boundary bend. The page shows the forward pass, the loss, the backpropagated gradients, and each hidden neuron's own heat map. Its How it works section explains the defaults and why they are there. The browser runs the same maths as `nnlab/`, and `tests/test_nnlab_parity.py` (needs node) checks the two against each other.

This lab has no benchmark table: it is a playground, and its checks are correctness checks. `tests/test_nnlab.py`, `tests/test_nnlab_parity.py` and `tests/test_server_nnlab.py`: 20 passed.

### Pathfinding arena: seven searches on one map

Seven searches race the same seeded maze. Every run is deterministic, and the cost column is checked against a Dijkstra reference. This is one seed printed by the CLI, not a benchmark: the lab is a playground, and its checks are correctness checks. `race --size 21 --maze prim --seed 3 --no-render` prints:

```
map 21x21  maze=prim  seed=3  swamp=0%  diagonal=off  heuristic=octile  w=2
ALGO     FOUND   EXPANDED  GENERATED   PATH       COST  OPTIMAL
---------------------------------------------------------------
bfs      yes          180        182     37      36.00  yes
dfs      yes          153        157     37      36.00  yes
ucs      yes          180        182     37      36.00  yes
greedy   yes          105        112     37      36.00  yes
astar    yes           94        101     37      36.00  yes
wastar   yes           68         77     37      36.00  yes
bidi     yes           72         81     37      36.00  yes
optimal cost (Dijkstra reference): 36.00
```

On this map every search finds an optimal path, and they differ only in how much they expand. Greedy and weighted A* expand fewest here, and A* expands 94 cells against 180 for breadth-first. On a map with diagonals, Manhattan is inadmissible by design, and the page warns about it: A* with Manhattan on 8-way moves can return a longer path, and the leaderboard marks it.

### Gridworld MDP lab: three solvers, one world

A grid with walls, rewards, pits, and slippery moves: each move slides sideways with probability `slip`. Three solvers work the same world. Value iteration spreads values like heat, one sweep at a time. Policy iteration alternates evaluation and improvement until the policy is stable. Q-learning or SARSA walks the grid and learns Q from its own episodes. The CLI output for the cliff preset, with the flags `--gamma 0.95 --slip 0.1` (the preset's own defaults are gamma 0.99 and slip 0):

```
Cliff walk  12x4  states=37  gamma=0.95 slip=0.1 living=-1

VALUE ITERATION  145 sweeps to residual < 1e-06 (last residual 9.10e-07)
...
POLICY ITERATION  5 improvement rounds, 871 evaluation sweeps, policy stable
agreement with value iteration: 100.0%
```

The Python package and the browser page compute the same numbers for the same seed: `tests/test_mdplab_parity.py` (needs node) checks values to 1e-9 and the Q-learning trajectories exactly. The cliff preset uses discount 0.99 on the page, not Sutton and Barto's 1, because with 1 a deterministic starting policy can loop forever and policy iteration never settles.

This lab has no benchmark table: it is a playground, and its checks are correctness checks.

### Optimizer race: six update rules, one landscape

Six optimizers race from the same start point down the same landscape. The default race is the ravine, a steep narrow valley where plain gradient descent zigzags. Run with no noise from (3.2, 0.9), the optimizers settle at these steps (`race --surface ravine --steps 300 --no-map`, verified on this branch):

```
optimizer        lr      final x      final y     final loss   to tol  status
-----------------------------------------------------------------------------
SGD            0.05    6.641e-07  2.1689e-181    2.20512e-13      158  reached minimum
Momentum       0.02  -3.9769e-07  -1.1438e-07    2.42603e-13      150  reached minimum
Nesterov       0.02   2.2686e-09  -7.7092e-53    2.57337e-18       86  reached minimum
RMSProp        0.05    -0.024977        0.025     0.00812461        -  still moving
Adam            0.1  -3.0369e-07   1.3076e-07    2.59842e-13      155  reached minimum
AdaGrad         0.5   1.0392e-09   9.3505e-90    5.39918e-19      108  reached minimum
```

Nesterov reaches the tolerance first. RMSProp, at the lr shown, is still moving after 300 steps. On Rosenbrock's banana from (-1.5, 2), the default run shows SGD, momentum and Nesterov diverging at their chosen step sizes (`race --surface rosenbrock --start -1.5,2 --steps 500`), which is why the step sizes are part of the comparison, not just the update rules. The results depend on the step sizes chosen: the race uses one step size per optimizer, and `--lr` changes them.

This lab has no benchmark table: it is a playground, and its checks are correctness checks.

### Tree lab: CART and random forests

Paint points or pick a preset, grow a CART tree one split at a time, then train a random forest of bootstrap trees and compare it with the single tree. The CLI prints the same accuracies the page shows.

```
preset xor  seed 1  train 168  test 72  criterion gini  depth<=4  min_leaf 3
leaves 7
train accuracy 0.964
test accuracy  0.931
```

```
preset spiral  seed 1  trees 50  train 168  test 72  criterion gini  depth<=5  min_leaf 3
OOB accuracy   0.845
test accuracy  0.917
single tree (depth 12, min_leaf 1) test accuracy  0.986
```

The forest does not beat a deep single tree on the spiral, and the lab shows that instead of hiding it: a depth-12 tree with one-sample leaves is more accurate on this split (0.986) than the 50-tree forest (0.917). The forest's out-of-bag estimate (0.845) is a different number from its test accuracy, and both are shown. This lab has no benchmark table: it is a playground, and its checks are correctness checks.

### Markov text: a chain that copies, and perplexity that turns back up

Four public-domain texts are counted into an order-n chain on words or characters. The chain then writes one token at a time from those counts, with a temperature and a seed. The page shows the next-token probabilities, the context window, and a copy meter: at order 5 most generated phrases are copied verbatim from the source.

Sources, exact text with the Project Gutenberg header and licence removed: Lewis Carroll, *Alice's Adventures in Wonderland*, chapter 1 (Project Gutenberg eBook #11); Jane Austen, *Pride and Prejudice*, chapter 1 (eBook #1342); Shakespeare's Sonnets I to XV (eBook #1041); and the Preamble and Article I of the U.S. Constitution from the National Archives transcript.

Held-out perplexity (last 10% of words, add-alpha smoothing with alpha 0.01), orders 1 to 5:

| Corpus | Order 1 | Order 2 | Order 3 | Order 4 | Order 5 | Lowest |
|---|---:|---:|---:|---:|---:|---|
| Alice | 494 | 209 | 291 | 349 | 380 | order 2 |
| Pride and Prejudice | 1299 | 128 | 115 | 134 | 155 | order 3 |
| Sonnets I to XV | 1744 | 707 | 693 | 701 | 700 | order 3 |
| Constitution | 459 | 297 | 349 | 416 | 468 | order 2 |

The honest finding: held-out perplexity is lowest at order 2 or 3 on every corpus, then climbs as the model overfits. Training perplexity keeps falling until order 3 or 4 and then flattens (Alice: 7.37, 7.03, 7.06 at orders 3 to 5). The climb is clear on Alice (1.8x from order 2 to order 5) and the Constitution (1.6x), and small on Pride and Prejudice (155 against 115 at order 3, 1.35x). The sonnets barely climb (693 to 700 from order 3 to 5): with about 1,800 training words there are almost no repeated contexts past order 3, so longer contexts add little beyond the smoothing.

```
python -m markov generate --corpus alice --order 3 --words 60 --seed 1
python -m markov generate --corpus sonnets --level char --order 4 --temperature 0.7 --seed 3
python -m markov perplexity --corpus alice --order 1..5
```

### Regression lab: least squares, gradient descent, and logistic regression

Drag points and watch the best-fit curve move. The same fit comes out of two solvers, normal equations and Householder QR, which agree to rounding. Gradient descent reaches the same answer only with enough steps, and the lab shows how far it is at a given budget.

Noisy sine, 30 points (23 train, 7 test), seed 7, degree 5:

| Method | Train MSE | Test MSE |
|---|---:|---:|
| QR (normal equations give the same weights to 6 digits) | 0.0472 | 0.0852 |
| Gradient descent, 500 steps, learning rate 0.1 | 0.2845 | 0.2699 |
| Gradient descent, 20,000 steps, learning rate 0.1 | 0.0538 | 0.0702 |

At 500 steps the descent has not converged: its weights are still far from the least-squares ones (w3 is -1.21 against -18.0), and its train error is six times the exact fit's. Degree is the other knob. Train error falls with every degree, but on this sample the test error bottoms out at degree 7 (0.072) and rises again at degrees 9 and 10 (0.097 and 0.092), where the curve is fitting the noise. The test set has only 7 points, so the exact location of that minimum moves with the seed.

Logistic regression on the moons preset (60 points, 45 train, 15 test) reaches 97.8% train and 80% test accuracy with a straight boundary, which is the honest limit of a linear classifier on that shape.

```
python -m regression fit --data noisy-sine --degree 5 --method qr
python -m regression fit --data noisy-sine --degree 5 --method gd --steps 500 --lr 0.1
python -m regression sweep --data noisy-sine --max-degree 10
python -m regression logistic --data moons --steps 500
```

### Evolving walkers: soft bodies learn to travel

Soft-bodied creatures (point masses, springs, pulsing muscles) are bred by a genetic algorithm with NEAT-style speciation. Fitness is the centre-of-mass distance in 10 simulated seconds, less a spin penalty, plus a bonus for speed in the second half. Three seeds on flat ground and one on hills, 300 generations each, population 80 (`results/walkers_evolution.md`):

| Run | Gen 100 best | Gen 200 best | Gen 300 best | Champion |
|---|---:|---:|---:|---|
| seed 1, flat | 2.13 | 5.74 | 14.32 | 13.72 m in 10 s: a one-hop skipper at 1.4 m/s |
| seed 2, flat | 1.67 | 7.17 | 11.70 | a 10-node crawler, 12.9 m |
| seed 3, flat | 2.20 | 4.82 | 14.07 | 15.30 m in 10 s: an inchworm, never above 0.31 m |
| seed 1, hills | 1.26 | 2.55 | 3.14 | 3.83 m over uneven ground |

Scores are in metres. No creature exploded. The honest limits:

- **Not converged.** The best score rose about 3 m between generations 250 and 300 in the flat runs, so longer runs would likely go further.
- **No hopper.** No body hops three times. The best flat walkers are a one-hop skipper and an inchworm that never leaves the ground by more than 0.31 m.
- **Nodes pass through each other.** There is no contact between nodes, so a body can fold through itself. When a spring's length passes through zero its force direction flips, and with no drag that adds energy: in a check with one undamped spring (k 200, rest 0.4 m, released 1.0 m apart), the energy rose from 35.8 J to 59.6 J. Drag damps this but does not prevent it. The physics is unchanged because the champions depend on it; a minimum-separation contact is the future fix, with a new evolution run.
- **A tumbling exploit was found and penalised.** An intermediate run with a weaker spin penalty found a two-node body that tumbled end over end: 16 full turns and 99 m, score 79, with the distance coming from tumbling rather than walking. The spin penalty is now 1.0 m per radian beyond the first radian, and a body with more than one full turn of spin is labelled a tumbler. That run is not committed as a result.

```
python -m walkers evolve --generations 50 --pop 80 --seed 1   # one line per generation: best, mean, median, species
python -m walkers replay "<creature string>"                  # ASCII side view of one run
python -m walkers study --reuse results/walkers_evolution.json # the committed study's presets
```

### Tic-tac-toe: the solved game behind the 404 page

Unknown URLs get a 404 page with a tic-tac-toe game against the agent, and the page shows the agent's live search tree. The Python reference solves the whole game (`python -m tictactoe stats`):

```
positions:       5,478 distinct boards reachable (4,520 still in play)
game-tree nodes: 549,946 (every move sequence, no pruning)
perfect play: draw
alpha-beta from the empty board: searched 18,297, pruned 6,930
```

With perfect play the game is a draw, so the agent never loses. Alpha-beta searches 18,297 nodes from the empty board, pruning 6,930 more, against 549,946 nodes for the full game tree. The browser port (`web/js/tictactoe-core.js`) is tested against the same reference by `tests/test_tictactoe.py` and `tests/web/tictactoe.test.mjs`.

This page is an easter egg, not a lab: it is checked by its tests, and it has no benchmark table.

## How it works

### N-Puzzle

- **Search** (`npuzzle/search.py`, `npuzzle/iterative.py`). One frontier interface covers FIFO, LIFO, and a priority queue with decrease-key (lazy deletion, with live states tracked separately so a heap of stale entries never looks non-empty). IDA\* runs on one mutable board and updates Manhattan distance and pattern database values incrementally: a move changes one tile, so it changes one lookup.
- **Pattern database** (`npuzzle/pdb.py`). The 15 tiles split into three groups of five. A 0-1 BFS from the goal records, for every placement of a group, the fewest moves of that group's tiles; other tiles move for free, so the three values add up to an admissible heuristic. Building all three takes about 40 seconds.
- **Neural heuristic** (`npuzzle/train.py`, `npuzzle/neural.py`). h(s) = PDB(s) + softplus(MLP(one-hot(s))): the network learns only how far the pattern database underestimates. Training uses approximate value iteration (as in DeepCubeA): sample boards by random walks from the goal, set the target to 1 + min over children of a frozen copy of h, fit, and copy. No solver labels are needed. The learned h is not admissible, so it is used with batch weighted A\*, which scores hundreds of states per network call.
- **Test set** (`npuzzle/testset.py`). 100 uniformly random 15-puzzles solved optimally by IDA\* + PDB. The mean optimal cost is 52.6 moves, which matches the known average for random 15-puzzles.

### Connect Four

- **Board** (`connect4/board.py`). Two 64-bit integers (all stones, and the stones of the player to move), with a spare bit per column so line checks never wrap. Four in a row is four shift-and-AND operations.
- **AlphaZero** (`connect4/puct.py`, `connect4/train_az.py`). PUCT search guided by a residual policy-value network. Self-play runs a whole iteration's games at once (128 per iteration on the laptop, 512 on Kaggle) and batches every game's leaf evaluation into one network call per simulation round. Positions are stored with mirror images; the loss is value MSE plus policy cross-entropy.
- **Serving without torch** (`connect4/net.py`). Batch norm is folded into the convolutions at export, so inference is NumPy convolutions and matrix multiplies.
- **Chance baseline** (`connect4/agents.py`). A no-skill opponent with no search or evaluation: it draws a column from fixed weights 1,2,3,4,3,2,1 (odds 6%/12%/19%/25%/19%/12%/6% on an open board), renormalised over the columns that still have room. The page and `python -m connect4` both offer it. A 20-game match against AlphaZero at level 1, with alternating colours and seed 0, is in `results/connect4_chance_match.md`, from `python -m connect4 --match 20 --agent alphazero --level 1 --seed 0`: AlphaZero won all 20 games.

### 2048

- **Board** (`game2048/board.py`). One 64-bit integer, four bits per tile. Every row's slide is precomputed for all 65,536 rows, so a move is four lookups plus a bit-trick transpose.
- **Expectimax** (`game2048/expectimax.py`). Six hand-crafted features (empty cells, monotonicity, smoothness, max tile in a corner, mergeable neighbours, max tile) come from per-row lookup tables. Their weights are found automatically by the cross-entropy method over batches of simulated games (`game2048/tune.py`). A transposition table and a cutoff for unlikely tile spawns let the search go deeper in the same time.
- **N-tuple network** (`game2048/ntuple.py`, `game2048/train_td.py`). Six patterns (four 5-cell, two 6-cell), each applied in all 8 board symmetries with a shared table: 48 lookups per evaluation. Trained by TD(0) on afterstates with 1,000 games in lockstep in NumPy. Batched TD updates are averaged per weight; summing them made shared weights take batch-sized steps and diverge. Tables are stored as float16 (72 MiB in memory).

### Checkers

- **Rules** (`checkers/board.py`). The dark squares as a tuple: 32 on 8x8, 50 on 10x10 and 72 on 12x12, with 3, 4 or 5 rows of men per side. Multi-jumps (a captured piece stays on the board until the move ends, so it cannot be jumped twice) and crowning, which ends the move. Verified against published perft counts on 8x8 (7, 49, 302, 1,469, 7,361) and pinned counts on the bigger boards (9, 81, 658 and 11, 121, 1,222 at depths 1-3; depth 1 is counted by hand from the front rows). Bigger boards keep American rules: men move and capture forward only, kings move one step, and there are no flying kings.
- **Captures.** Forced by default, as in standard rules. With `--no-forced-capture` (or the page's toggle) a simple move is legal even when a capture exists. Either way a capture chain is played to its end: a player who starts a multi-jump cannot stop partway through it. Stopping partway through a chain is not allowed in standard rules either, so the optional rule changes only whether a capture is compulsory.
- **Search** (`checkers/search.py`). Negamax, so each side maximizes its own score and the opponent's reply is assumed to be the one worst for it. Alpha-beta skips a move as soon as one reply proves it worse than an alternative already found; iterative deepening, a transposition table, and trying the previous best move first make those cutoffs come early. Positions with a capture pending are searched further, through captures only, rather than scored mid-exchange; with optional captures the side may also decline to capture and keep the static score (stand pat). Plain minimax has no time limit, so its depth is fixed per board size and capture rule (`checkers/agents.py`).
- **Monte Carlo tree search** (`checkers/mcts.py`). UCT: selection by the upper confidence bound, one expansion, a rollout, and backpropagation, for a fixed number of iterations (300, 1,000 or 2,000 on 8x8). Rollouts prefer captures, stop after 40 plies, and score the position with the same evaluation as the tree searches. The random generator is seeded from the position, so the same position always gets the same answer.
- **Chance** (`checkers/chance.py`). Fixed odds with no search: each legal move is weighted by its type (captures 5, a man moving into the centre 3, other man moves 2, king moves 1), the weights are renormalised over the legal moves, and one is sampled. Seeded from the position. It is the checkers counterpart of the 2048 tile spawner.
- **Greedy and random** (`checkers/search.py`). Greedy picks the move whose resulting position scores best after one ply, and counts a move that leaves no legal reply as a win. Random picks uniformly from the legal moves with a fixed seed.
- **Budgets.** Alpha-beta is time-boxed (0.2, 0.8 and 2.0 s), so the depth it reaches depends on the machine. Match mode counts nodes instead (3,000, 12,000 and 40,000), so a match replays exactly. Worst case on this laptop over three midgame positions per board and capture rule: alpha-beta 2.05 s at level 3 (its time box), MCTS 1.7 s at level 3, minimax 0.9 s at its deepest setting, greedy and random under 0.01 s. Every request stays under about 2 s of search, so one request cannot hold the 1 GB server for long.

### Blackjack

- **Dealer** (`blackjack/solver.py`). A recursion on the dealer's next card gives, for every hard total and ace flag, the probability of each final total (17 to 21, or bust). The dealer peeks for blackjack under an ace or a ten, so the player's decisions are made on the hole cards conditioned on "no blackjack".
- **Player** (`blackjack/solver.py`). Each hand is a state: hard total, whether an ace is in it, and whether it is still two cards. Hitting always adds a card, so each state depends only on states with larger totals, and memoised recursion is exact backward induction. A pair splits into two independent hands under the infinite deck, so the split is twice the average over the second card of one hand's best play.
- **Learner** (`blackjack/learner.py`). Monte Carlo control with no model: exploring starts from every table cell, every-visit sample averages, and epsilon-greedy choices. Stand, hit, and double share statistics by hard total and ace flag, since their future does not depend on card count. It is compared with the exact table, not trained on it.
- **Game** (`blackjack/game.py`, `python -m blackjack`). A 6-deck shoe with reshuffles at 75%, the table's rules, and the same basic strategy for `simulate`. The advice on the page and in the terminal is the exact table, not the learner's.

### Battleship

- **Rules** (`battleship/board.py`). Cells are 0-99 as bitmasks, so set operations on the board are single integer operations. Ships are straight, on the board, and may touch but not overlap. A shooter learns a sunk ship's full cell list, which is the only information the agents see. Fleets are drawn uniformly over legal layouts (rejection sampling), which matches the uniform prior the probability model uses.
- **Probability model** (`battleship/probability.py`). For each cell, P(ship) is the share of consistent fleet layouts that use it, where a layout is consistent when it avoids misses and sunk ships and covers every hit that has not sunk its ship. Layouts are counted exactly by a depth-first search with pruning when the placements are few enough (about 60,000 candidate checks at most). Otherwise 1000 layouts are sampled by importance sampling: each ship is drawn uniformly from the placements that fit, and each sample is weighted by the number of options at each step, which makes the weighted samples an unbiased stand-in for a uniform draw over consistent layouts. Sampling is vectorised over all samples with numpy.
- **Agents** (`battleship/agents.py`). The Bayesian agent fires at the unknown cell with the highest probability, breaking ties at random. The hunt/target baseline fires on a checkerboard (every ship covers an even cell) until a hit, then along a line once two hits align, and otherwise next to a hit. The chance agent draws an unknown cell with probability proportional to the fixed `CHANCE_WEIGHTS` table, renormalised over the unknown cells; it is seeded, so the same seed and history give the same shot.
- **Race** (`battleship/match.py`). Two fleets, two shooters, alternating shots; the first fleet sunk decides. `match` plays N races with the first shooter alternating by game index.
- **Server** (`server/battleship_api.py`). The agent's fleet is random and stored server-side under an unguessable id, with a bounded store and a 30-minute expiry, so the page cannot read it before the game ends. The player's own fleet never reaches the server: the page sends the agent's shots at it, and the server returns the odds for those observations. `/agent-shot` takes `agent` (`probability` or `chance`) and a `seed` for chance; `/meta` returns the chance profile and rule.
- **Page** (`web/battleship.html`). Place a fleet, fire at the opponent's fleet, and watch the opponent's odds glow over your grid. The opponent selector picks Probability (Bayesian) or Chance (fixed odds). Watch mode races the AI against the chosen opponent, with pause, step, a speed control, a running AI W/L score and auto-restart; the chance table is drawn on the page from `/meta`.

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
- **Chance opponent** (`endgame/chance.py`, `web/js/endgame-core.js`). The page and the CLI can pick the agent as *Chance (fixed odds)* instead of the tablebase. It never searches or reads the table. Each legal move gets a category, checked in this order: capture (the lone king takes the piece) weight 4, check weight 2, a king step toward the centre weight 2, any other move weight 1. A move's chance is its weight over the total weight of the legal moves, the same renormalisation as the 2048 spawner's 90/10 split. The page shows each move's chance in the agent panel, and watch mode (MODE: *Watch: tablebase vs chance*) plays the two against each other, with pause, step, speed, auto-restart and a score. The watch has no level: the tablebase is exact, so there is nothing to tune.
- **Match** (`python -m endgame match`). Plays the tablebase against the chance opponent over random winning positions, with no prompts, in both seats. The committed run is `python -m endgame match --games 200 --piece both --seed 0` (results in `results/endgame_chance_match.md` and `.json`): the tablebase won all 400 games with the piece, and as the lone king it drew 399 of 400 (lost one in KQK), because the chance side's random moves often leave the piece to be taken.
- **Parity** (`tests/test_endgame_parity.py`). The JavaScript odds and picks match the Python reference on the same positions and the same uniform numbers, and the table literal in the JavaScript matches `CHANCE_WEIGHTS`.

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

### Pac-Man: chance ghosts (fixed odds)

The ghosts can also move by a fixed table instead of A* routes (`pacman/ghosts.py`, `CHANCE_ODDS`). Each turn a chance ghost keeps going with probability 0.60, turns left or right with 0.15 each, and reverses with 0.10. The shares are renormalised over the directions that are open, and "keep going" means the direction of the ghost's last move. The ghost does no search and does not look at Pac-Man or the pellets. The page's GHOSTS menu switches between the two policies, and `python -m pacman match` plays each agent against both on the same seeds.

Result from `python -m pacman match --games 200`, written to `results/pacman_chance_match.json` and `.md`. Q-learning agent, seeds 10000-10199, 200 games per maze:

| Ghosts | Win rate | Mean score | Mean turns | Won / lost / timeout |
|---|---|---|---|---|
| AI ghosts (A* routes) | 64.2% | 1350.3 | 193.9 | 257 / 47 / 96 |
| Chance (fixed odds) | 46.2% | 1513.7 | 110.8 | 185 / 215 / 0 |

The AI row matches the benchmark above. Against chance ghosts the Q-agent wins less often and its games end sooner, with no timeouts, while its mean score is higher. The score difference is not explained yet. The match cannot separate ghost difficulty from the Q-agent's model: its survival feature replays the game with the A* ghosts, so it predicts the wrong moves when the ghosts are chance ghosts.

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
- **Serving** (`server/poker_api.py`). The server holds each hand. Until the showdown a response carries only the bot's actions and, for each one, its "spot" (the bot's information set with its card replaced by `?`). The bot's card and the mix it drew from reach the browser only in `bot_reveal` once the hand is over, because the mix depends on the card. A bot move is a lookup in `poker/data/leduc_strategy.json`; nothing trains on the server.

### Minesweeper

- **Rules** (`minesweeper/board.py`). Mines are placed after the first click, which keeps the clicked cell and its neighbours clear. A zero floods outward. Agents see only a `View` of the revealed numbers and the mine count.
- **Single-cell rules** (`minesweeper/inference.py`, `rule_pass`). A number with as many mines still to place as covered neighbours makes all of them mines. A number whose known mines already satisfy it makes the rest safe. Repeated to a fixpoint.
- **Components** (`minesweeper/inference.py`). Each number is a constraint "exactly `need` of these covered cells are mines". Constraints sharing a covered cell form a component. Components are independent once the total is fixed, so each is counted on its own.
- **Counting** (`_count_layouts`). Backtracking assigns cells in row-major order and prunes any constraint whose residual need is negative or larger than its unassigned cells. The memo key is (cell index, residual needs of open constraints), so each state is solved once. The result is the number of layouts with k mines, for each k. The count has a work budget (`WORK_BUDGET`, 2,000,000 units per analysis). A board over it keeps its single-cell proofs, which are still exact, and gets estimated odds instead: the API says so with `"exact": false`, and the page labels the odds as estimates. On real expert play about 1.6% of positions go over, and the crafted wide-frontier boards in the tests stop at about a second.
- **Exact probabilities.** The interior (covered cells touching no number) takes the leftover mines in C(n, k) ways. Combining components by convolution and weighting by C(n, remaining - t) gives the weight of every full layout. A cell's probability is the weight of layouts where it is a mine, over the total. Everything is integer arithmetic until the final division.
- **Agents** (`minesweeper/agents.py`). Random: a uniformly random covered cell. Rules: level 1, guesses at random when nothing is proven. CSP: level 2 proves the cells whose weight is zero (safe) or total (mine), and guesses at random when nothing is proven. Probability: level 3 guesses the covered cell with the lowest exact probability, breaking ties toward the most covered neighbours.
- **Checks** (`tests/test_minesweeper.py`). Counts and probabilities match brute-force enumeration of every layout on the random small boards that are still open after their reveals (54 of the 60 seeds; the other six end the game first and are skipped). Every proof is checked against the hidden layout on real games. The probabilities sum to the remaining mine count exactly.

### Hex

- **Rules** (`hexgame/board.py`). An n x n rhombus (7x7 by default, 5 to 11 allowed). DOWN connects top to bottom and ACROSS connects left to right; Hex has no draws, so a full board has exactly one winner. The optional swap rule lets the second player take the first move.
- **Playouts** (`hexgame/mcts.py`). A playout fills the empty cells in a random order and checks the winner once, with one connectivity search at the end. That is far cheaper than playing to a result move by move. The fill also credits moves played after the game was already decided, which is the standard approximation for fill playouts.
- **Search** (`hexgame/mcts.py`). UCT with one node added per iteration, using win rate plus an exploration term with c = 0.6. RAVE keeps all-moves-as-first counts in each node and blends them with the UCT value by beta = sqrt(k / (3n + k)), with k = 300. Both constants were chosen, not tuned.
- **Baselines** (`hexgame/agents.py`, `hexgame/heuristic.py`). Random picks a legal cell. The shortest-path agent is a one-ply race heuristic, not a full two-distance model.
- **Chance baseline** (`hexgame/agents.py`). Picks an empty cell with no search: each cell is weighted by its hex distance from the centre (rings 0, 1, 2 and 3+ get 4, 3, 2 and 1), renormalised over the empty cells, and it never swaps. A 20-game match against RAVE at level 1 (300 simulations per move), with alternating colours, is in `results/hexgame_chance_match.md`, from `python -m hexgame match --agent rave --level 1 --games 20`: RAVE won all 20 games.
- **Checks** (`tests/test_hexgame.py`). Rules, search, baselines, the CLI, the benchmark writer, and the router, with the two hexgame test files passing.

### Bandits

- **Casino** (`bandits/env.py`). Machine means come from a seeded generator, redrawn until the best leads the second best by 0.05. Outcomes are rolled once per seed for every pull and machine, so all agents face the same luck. Regret is expected: the best mean minus the mean of the machine pulled.
- **Agents** (`bandits/agents.py`). Greedy and epsilon-greedy use the empirical means; UCB1 adds sqrt(2 ln t / n) to each mean; Thompson samples Beta(1 + wins, 1 + losses) per machine and pulls the largest sample; EXP3 keeps exponential weights with importance-weighted rewards and mixing gamma; sliding-window UCB uses the last 200 pulls only, with xi = 0.6 in its bonus.
- **Lai-Robbins** (`bandits/env.py`). For fixed Bernoulli machines the bound is sum over suboptimal arms of gap / KL(p_i || p*) times ln t; for Gaussian machines with known variance it is sum of 2 sigma^2 / gap.
- **Portable randomness** (`bandits/rng.py`). mulberry32 uniforms, Marsaglia polar normals, and Marsaglia-Tsang gammas, so the page's JavaScript port reproduces the Python arm sequences (checked for all 17 agent and casino pairs at seeds 0 to 4, in `tests/test_bandits_parity.py`).
- **Checks** (`tests/test_bandits.py`, `tests/test_server_bandits.py`). RNG moments, the Lai-Robbins constant against a hand calculation, agent invariants (forced first round, distribution floors, the window forgetting old rewards), regret consistency between a run and scoring its arms, the benchmark shape, and the CLI.

### CartPole

- **Physics** (`cartpole/env.py`). The standard cart-pole equations, integrated by explicit Euler at 0.02 s. The episode ends when |theta| > 12 degrees or |x| > 2.4 m, and is truncated at 500 steps. Reward is 1 per step, so the return equals the balance length.
- **Policies** (`cartpole/nets.py`, `cartpole/train.py`). A one-hidden-layer tanh MLP with a softmax over left and right, trained with hand-written backprop and Adam. REINFORCE uses the batch-mean return as its baseline. The actor-critic uses a learned value network as its baseline, trained on Monte Carlo returns. A TD(0) critic that bootstraps from its own estimate did not learn reliably in the pilot runs, so the shipped actor-critic uses Monte Carlo returns.
- **Cross-entropy search** (`cartpole/train.py`). Samples five-number linear policies, keeps the best quarter, and refits the sampling distribution.
- **Checks** (`tests/test_cartpole.py`). Backprop is compared with finite differences for the policy and the critic, and the physics against the equations worked by hand. The browser port is checked step by step against the Python physics (`/api/cartpole/rollout`).

### N-Queens

- **Board** (`queens/board.py`). One queen per row, so only columns and the two diagonal families can conflict. Three count arrays (column, r+c, r-c+n-1) give the attacks on any square in O(1), and place/remove keep the conflict count exact. Brute-force checks in the tests cover both.
- **Backtracking** (`queens/agents.py`). Row by row, with three bitmasks: columns taken, and the two diagonal masks shifted one bit per row. The lowest open column is tried first. It is iterative, so 10,000 rows do not overflow the stack. It proves infeasibility for n = 2 and 3.
- **Hill climbing and annealing.** Steepest ascent scores all n^2 moves per step and restarts at a local minimum. Annealing proposes one move at a time, with an O(1) cost change d and acceptance exp(-d/T).
- **Min-conflicts.** A greedy start (random unused columns, up to 4,096 tries per row) leaves about 10 conflicts at 1,000 to 10,000 queens (the only sizes checked for the count). A repair picks a conflicted queen at random, lifts it, scores all n columns of its row in one numpy pass, and moves it to a least-attacked column. The run restarts if it stalls on a plateau, which happens at small n.
- **Checks** (`tests/test_queens.py`). Incremental counts against brute force under random moves; every agent's returned placement is validated; the min-conflicts trace never rises; the backtracking node counts are pinned; the scripted play and CLI paths are exercised.

### Snake

- **Evaluation function** (`snake/evaluator.py`). For each safe move the snake imagines the position after it and reads eight features: whether it eats, how near the food is by BFS, whether the food is reachable, the reachable area, whether that area can hold the body, whether the tail is reachable (the escape route), the head's exits, and the length. The move with the largest weighted sum is taken. The feature code is one breadth-first search per move, so it is cheap enough to train on.
- **Cross-entropy training** (`snake/cem.py`). A diagonal Gaussian over the eight weights, refit each generation to the best 12 of 48 candidates. Every candidate in a generation plays the same fresh boards, so the comparison is fair; the next generation sees new boards. The champion is chosen on fixed validation boards that are never used for training.
- **Neural net** (`snake/net.py`, `snake/evolve.py`). The snake senses its surroundings in its own frame: danger, free run, food and tail offsets, its heading, and how much room it has to move ahead, left and right (a capped flood fill). A one-hidden-layer tanh network maps the 17 senses to left, straight, or right. Its 339 weights are bred by a genetic algorithm: each generation plays on fresh boards, the top genomes are kept, parents are chosen by tournament, children mix weights uniformly, and about one weight in ten is nudged by a small Gaussian. Starving is penalised in fitness.
- **Planner baseline** (`snake/agents.py`). Paths to the food with BFS and takes that path only if the snake could still reach its tail after eating.
- **Browser** (`web/js/snake-core.js`, `web/js/snake.js`). Both learned agents run in the page with the same arithmetic. The page's evaluation-function panel shows each move's terms and scores, and the net view shows the network's units.
- **Checks** (`tests/test_snake_eval.py`, `tests/test_snake_eval_parity.py`, `tests/test_snake.py`, `tests/test_server_snake.py`). Features on hand-made boards (an open board, an eating move, a sealed pocket with the tail cut off, fatal moves excluded), determinism of the choice and of training, a worker-count independence test, and a node parity test of the JavaScript evaluator against Python on recorded games.

### Rover

- **Map and sensor** (`rover/world.py`, `rover/explorer.py`). A seeded grid with random walls, start top-left, goal bottom-right. The rover senses a square window (radius 2 by default) and learns those cells as it sees them. Unknown cells are planned as free, so a route can be wrong until the rover senses the wall.
- **D\* Lite** (`rover/dstar.py`). Searches backwards from the goal. Each cell keeps g (its believed cost to the goal) and rhs (a one-step lookahead). A new wall makes some rhs values wrong; only those cells are repaired, instead of replanning the whole map.
- **A\*** (`rover/astar.py`). The baseline runs a fresh A* search from the rover's cell each time the map changes.
- **Checks** (`tests/test_rover.py`, `tests/test_rover_parity.py`, `tests/test_server_rover.py`). The parity test runs the page's JavaScript planner (`web/js/rover_core.js`) under node and compares it with the Python planner.

### Ghost Hunt

- **Hidden state and motion** (`ghosthunt/motion.py`). A ghost's hidden state is its open cell and heading. Random walk, lurker (drifts toward the player, so its transitions depend on the observed player position) and patrol (keeps to corridors) are known transition distributions.
- **Sonar** (`ghosthunt/sonar.py`). The reading is the maze distance plus discrete Gaussian noise, normalised over the readings the sonar can show.
- **Forward algorithm** (`ghosthunt/hmm.py`). Each turn: predict with the motion model, then multiply by the sonar likelihood and normalise. Exact over every hidden state.
- **Particle filter** (`ghosthunt/particles.py`). N weighted samples, systematic resampling when the effective sample size drops below N/2.
- **Viterbi** (`ghosthunt/hmm.py`). The same recursion with max in log space and back-pointers; it recovers the most likely path after a bust, which the page replays.
- **Checks** (`tests/test_ghosthunt.py`, `tests/test_ghosthunt_parity.py`, `tests/test_server_ghosthunt.py`). Forward and Viterbi against enumeration of every state sequence on tiny mazes, normalisation, determinism, particle convergence, and the JavaScript port (`web/js/ghosthunt-core.js`) against Python on six scenarios.

### Lost Robot

- **Floor plans and sensors** (`localize/world.py`). Walls are rectangles on a grid. Twin Halls has six rooms that look alike from inside (only one differs), and Vault is an irregular plan with no symmetry. Eight range beams are ray-cast against the walls, with Gaussian noise on each distance. Odometry noise grows with the size of the turn and the drive.
- **Histogram Bayes filter** (`localize/grid.py`). A belief over every (x, y, heading) bin. Each step predicts with the motion model, then multiplies by the beam likelihoods and normalises. Exact on its discretisation, and the baseline for the others.
- **Monte Carlo localisation** (`localize/particles.py`). N weighted particles, each moved with noise and weighed by its beams, with systematic (low-variance) resampling when the effective sample size drops. Augmented MCL keeps a fast and a slow running average of the measurement likelihood and, when the fast one falls below the slow one, replaces a uniform random subset of the particles with poses over the floor, which is how it recovers from a kidnap.
- **Autopilot and mode count** (`localize/sim.py`). The true robot drives along breadth-first routes to random destinations. The filters see only the beams and the odometry readings, never the true pose. The page counts the separate places the belief occupies.
- **Browser core** (`web/js/localize-core.js`). The same floors, sensor and motion models, grid filter and particle filter. `tests/test_localize_parity.py` runs it in node on a fixed scenario and compares the output with Python.
- **Checks** (`tests/test_localize.py`, `tests/test_localize_parity.py`, `tests/test_server_localize.py`). Filter behaviour on simple floors, the ray caster, resampling, kidnap injection, the CLI, the JavaScript parity test, and the router metadata.

### Tetris

- **Board and pieces** (`tetris/board.py`, `tetris/pieces.py`, `tetris/game.py`). Board rows are bitmasks, so a line clear is a mask test and a drop is a few shifts. Each piece has every distinct rotation as row masks.
- **Placement search** (`tetris/search.py`). For each rotation and column the piece drops until it lands, and the result is scored after the lock and line clear. Optional one-piece lookahead uses the preview.
- **Features and weights** (`tetris/features.py`, `tetris/tuned.py`). The nine features are listed under the results above. The default AI uses the tuned v3 weights in `tetris/tuned.json`, which is a copy of `results/tetris_cem_weights.json`; the hand-picked set is kept as a preset.
- **Tuning** (`tetris/cem.py`, `tetris/evolve.py`). The v3 weights come from a noisy cross-entropy search (`tetris/cem.py`): a Gaussian over the nine weights is sampled, the elite samples refit it, and fresh seeds are drawn each iteration. The v2 weights came from a genetic algorithm (`tetris/evolve.py`, CLI defaults 16 genomes, 6 games, 16 generations), with fresh seeds shared by each generation.
- **Checks** (`tests/test_tetris.py`, `tests/test_server_tetris.py`). Features, line clears and placements, and a node parity test of the JavaScript engine against the Python search.

### Nonogram

- **Clue automaton** (`nonogram/automaton.py`). A clue such as (3, 1) is a regular language over empty and filled cells, and a small deterministic automaton recognises the lines that satisfy it.
- **Line solving** (`nonogram/lines.py`). For one line, a forward sweep marks the automaton states reachable after each prefix and a backward sweep marks the states from which the rest can still complete. A cell's value is forced when only one value survives both sweeps; the answer is exact without listing the arrangements.
- **Hybrid search** (`nonogram/solvers.py`). Line passes until nothing changes, then a guess on the most constrained line's first unknown cell, with backtracking. Uniqueness is checked by counting solutions up to a limit.
- **SAT** (`nonogram/cnf.py`, `nonogram/dpll.py`). The whole puzzle becomes a CNF formula over the cells and the automaton states, solved by DPLL with unit propagation, phase saving and chronological backtracking. No clause learning.
- **Checks** (`tests/test_nonogram.py`, `tests/test_server_nonogram.py`). Solver and encoding agreement, library uniqueness, the CLI, and the router.

### Cluster lab

- **k-means** (`clusters/kmeans.py`). Assign each point to its nearest centroid, then move each centroid to the mean of its points. Starts are k-means++ or random, drawn from the same mulberry32 stream as the page, so Python and the page agree on seeds.
- **DBSCAN** (`clusters/dbscan.py`). Core points have at least `min_pts` neighbours within `eps`; clusters grow through core points. A border point joins the cluster of its lowest-index core neighbour, a fixed rule. sklearn picks by visiting order, so labels can differ from sklearn on border points.
- **EM for a Gaussian mixture** (`clusters/gmm.py`). E-step: soft responsibilities. M-step: weighted means and covariances. A regularisation of 1e-6 is added to each covariance diagonal, and the log-likelihood check uses a 1e-7 per-point tolerance for rounding.
- **Metrics** (`clusters/metrics.py`). Inertia, silhouette, and the elbow sweep over k.
- **Landing sim.** The cabinet on the landing page runs its own small k-means, so the landing page does not load the lab's module.

### Neural net lab

- **Network** (`nnlab/mlp.py`). A multilayer perceptron in NumPy with hand-written backprop: the forward pass stores each layer's activations, and the backward pass applies the chain rule layer by layer.
- **Training** (`nnlab/train.py`). Mini-batch Adam or SGD, with L2 regularisation and a held-out test fraction. Data come from five seeded generators (`nnlab/data.py`): blobs, xor, circles, spiral and moons.
- **Portable maths** (`web/js/nnlab-core.js`). The same forward pass, loss and gradients in JavaScript, so the page does not need the server.
- **Checks** (`tests/test_nnlab.py`, `tests/test_nnlab_parity.py`, `tests/test_server_nnlab.py`). Gradients, training behaviour on the seeded datasets, and the JavaScript parity against Python.

### Pathfinding arena

- **Grid and movement** (`pathfind/grid.py`). A cell is a wall (0) or a terrain cost of at least 1 (swamp is 5, mud is 3). Entering a cell costs its terrain, times sqrt(2) on a diagonal step. Diagonals may not cut a wall corner. The three heuristics are Manhattan, Euclidean and octile; they are admissible for unit cost, and Manhattan overestimates diagonals.
- **Searches** (`pathfind/search.py`). Each expansion records the cells it discovered and their g values. Ties break by push order and neighbours are visited in a fixed order, so every run is deterministic. Weighted A* uses priority g + w·h, with w = 2 by default.
- **Mazes** (`pathfind/mazes.py`). Recursive backtracker and Prim's carve perfect mazes on the odd-coordinate lattice; scatter and rooms are not perfect. Maps come from a 32-bit mulberry32 generator, so the same seed gives the same map in Python and in the browser.
- **Browser port** (`web/js/pathfind-core.js`). A line-for-line port with no DOM. `tests/test_pathfind_parity.py` (needs node) runs it on fixed seeds and compares maze cells, random numbers, expansion order, g values, paths and costs with Python, step for step.
- **Checks** (`tests/test_pathfind.py`). Uniform cost and A* match a Bellman-Ford reference on 120 random grids, with and without diagonals. Weighted A* stays within w times the optimum. BFS is optimal on unit-cost grids. Mazes are perfect, and searches are deterministic.
- **Painted swamps and mud** overwrite walls and open them, which makes loops in a perfect maze. That is why depth-first can return a longer path than Dijkstra on a painted map. This is intended, and the board marks it.

### Gridworld MDP lab

- **Model** (`mdplab/`). States, actions, transitions with slip, and rewards. Walls, pits and goals are terminal or blocked cells; the living cost is a per-step reward.
- **Planners.** Value iteration sweeps the Bellman optimality backup until the residual is small. Policy iteration evaluates the current policy to convergence, then improves it greedily, and stops when the policy no longer changes.
- **Learners.** Tabular Q-learning (off-policy) and SARSA (on-policy) on the same world, from the same seeded random stream.
- **Shared stream.** The random numbers come from `bandits/rng.py`, the mulberry32 generator also used by the JavaScript port, so the page and the Python package agree.
- **Checks** (`tests/test_mdplab.py`). Hand-computed values, Bellman optimality, agreement of value and policy iteration, Q-learning convergence on a tiny grid, seed determinism, and the CLI. `tests/test_mdplab_parity.py` runs the browser port under node, and `tests/test_server_mdplab.py` covers the read-only API.

### Optimizer race

- **Landscapes and updates** (`optlab/surfaces.py`, `optlab/optimizers.py`). Six landscapes, including the ravine and Rosenbrock, and a paint-your-own one. Each optimizer is one update rule with its own state: momentum, Nesterov's look-ahead, RMSProp's and Adam's running averages with bias correction, and AdaGrad's accumulated squares.
- **Race loop** (`optlab/race.py`). Every optimizer starts from the same point, the same landscape and the same noise stream, and the loop records the first step each one reaches the tolerance, or the step where it diverges.
- **Browser port** (`web/js/optlab-core.js`). The same maths without the DOM. `tests/test_optlab_parity.py` (needs node) runs both on seven scenarios and requires equal settle and divergence steps and trajectories to 1e-9.
- **Checks** (`tests/test_optlab.py`, `tests/test_server_optlab.py`). Update rules against hand-computed steps and the race's bookkeeping; the router is not registered because the page does not call it.

### Tree lab

- **CART** (`treelab/`). Gini or entropy impurity, exhaustive midpoint thresholds over each split's samples, depth and leaf-size limits, and cost-complexity pruning.
- **Forests.** Bagged bootstrap trees with feature subsampling at each split. Thresholds are exhaustive over the sampled features (Breiman style), not random (ExtraTrees style). Each tree is scored on the rows its bootstrap left out, giving the out-of-bag accuracy.
- **Presets.** Seeded two-dimensional datasets: blobs, checkerboard, diagonal, nested, spiral and xor.
- **Browser core** (`web/js/treelab-core.js`). The same trees, thresholds, counts, pruned trees, forest OOB and importances in JavaScript. `tests/test_treelab_parity.py` (needs node) checks them against Python.
- **Landing sim.** A fixed depth-2 illustration on the XOR pattern, not a live CART run.
- **Checks** (`tests/test_treelab.py`, `tests/test_treelab_cli.py`, `tests/test_treelab_parity.py`, `tests/test_server_treelab.py`). Tree construction, the CLI, the parity test, and the router.

### Markov text

- **Model** (`markov/model.py`). Counts per (order-1)-token context. Sampling raises each count to 1/T and draws one uniform per token from `bandits/rng.py`, so a seed reproduces the text.
- **Corpora** (`markov/data/*.txt`, mirrored in `web/data/markov/`). Each file has a header with the source, the Project Gutenberg ebook number and URL, and the public-domain note. `tests/test_markov.py` checks that the web copies match the package copies byte for byte.
- **Perplexity** (`markov/model.py`). The last 10% of tokens are held out, with add-alpha smoothing (alpha 0.01) so unseen words get a small probability instead of zero.
- **Browser core** (`web/js/markov-core.js`). The same tokeniser, counts, sampling and copy statistics as the Python module. `tests/test_markov_parity.py` runs the JavaScript on 15 corpus, level, order, temperature and seed cases, including a restart, and compares it with Python token for token.
- **Checks** (`tests/test_markov.py`, `tests/test_markov_parity.py`). Counts on tiny texts, probabilities that sum to 1, temperature, seeded determinism, copy statistics, held-out perplexity behaviour, and the CLI.

### Regression lab

- **Design matrix and solvers** (`regression/core.py`). Polynomial features in one or two variables. Normal equations solved by Gauss elimination with partial pivoting, and a Householder QR solver for the same least-squares problem. Ridge adds a penalty to every weight except the intercept.
- **Gradient descent** (`regression/core.py`). The squared-error gradient, or the log-loss gradient for the classifier, with a fixed step size. The run reports divergence when the loss stops being finite instead of returning NaN weights.
- **Data** (`regression/data.py`). Presets are generated with the mulberry32 generator from `bandits/rng.py`, so the page and the terminal share the same points for the same seed and preset.
- **Browser core** (`web/js/regression-core.js`). The same features, solvers and descent. `tests/test_regression.py` runs it in node on the same inputs and compares every number with Python to 1e-9; the test is skipped when node is missing.
- **Checks** (`tests/test_regression.py`). Exact fits on collinear and polynomial points, normal equations against QR and `lstsq`, ridge limits, singular systems, gradient descent convergence and divergence, finite-difference gradient checks, reproducible presets, the CLI output, the degree sweep, and the JavaScript parity test.

### Evolving walkers

- **Creatures** (`walkers/core.py`). A genome is a graph of nodes, springs and muscles. Muscles pulse with a phase, so the body moves without any coded gait.
- **Physics** (`walkers/core.py`). Point masses with gravity, springs, friction and restitution against flat or hilly ground, integrated in fixed 1/240 s steps. The same steps run in the browser. Nodes have no contact with each other, so they can pass through each other (see the limits above and `tests/test_walkers.py`).
- **Selection** (`walkers/core.py`). Fitness is the centre-of-mass travel in x, minus a spin penalty, plus a bonus for speed in the second half. Species share fitness, so a new body plan is not wiped out before it can improve.
- **Operators** (`walkers/core.py`). Index-aligned crossover of two genomes, and graph mutation that adds or removes nodes and springs and toggles muscles.
- **Browser core and worker** (`web/js/walkers-core.js`, `web/js/walkers-worker.js`). The same physics and operators, run in a module worker so the page stays responsive. `tests/test_walkers_parity.py` compares the JavaScript with Python on the same creatures.
- **Checks** (`tests/test_walkers.py`, `tests/test_walkers_parity.py`, `tests/test_server_walkers.py`). Physics sanity and momentum, determinism, the operators, a known creature, and the CLI.

### Tic-tac-toe and the 404 page

- **Reference** (`tictactoe/core.py`). Board states, legal moves, the win check, and minimax with alpha-beta pruning, with the number of nodes searched and pruned counted for each call.
- **Browser port** (`web/js/tictactoe-core.js`). The same search in JavaScript, so the 404 page needs no server. `tests/web/tictactoe.test.mjs` checks it against the Python reference.
- **404 page** (`web/404.html`, `web/js/notfound.js`). Unknown GET pages get this page with status 404. Unknown paths under `/api/` stay JSON 404s, so the API's error format is unchanged (`server/app.py`, `WebFiles`).
- **Checks** (`tests/test_tictactoe.py`, `tests/test_server_404.py`, `tests/web/tictactoe.test.mjs`). Reference and port agreement, the game's rules, and the 404 status for pages and the JSON 404 for the API.

## Architecture

```
web/            static HTML/CSS/JS, no build step
server/         FastAPI: REST for game moves, a WebSocket that streams N-Puzzle searches
npuzzle/ connect4/ checkers/ routes/ game2048/ sudoku/ lightsout/ blackjack/ battleship/ pacman/ warehouse/ endgame/ sokoban/ wordle/ poker/ minesweeper/ hexgame/ bandits/ cartpole/ queens/ snake/ rover/ ghosthunt/ tetris/ nonogram/ localize/ clusters/ nnlab/ pathfind/ mdplab/ optlab/ treelab/ markov/ regression/ walkers/   search code, training scripts, data files
```

- Searches run in worker threads. A semaphore caps concurrent searches, each client is rate limited, and every request has node and time limits. Searches that keep every state in memory use about 1 KB per expanded node, so they stop at 250,000 nodes; IDS and IDA\* use memory linear in depth and may run longer.
- Trained models ship as `.npz` weights and run in NumPy, so the server image has no torch (about 350 MB) and loads every model in about 120 MB of RAM.

## Sound

The site has optional synth sound effects, made with the Web Audio API (`web/js/sfx.js`, no audio files). They are off by default. To turn them on or off, press the speaker icon at the right end of the nav bar (keyboard: Tab to it, then Enter or Space). The choice is kept in this browser's localStorage and applies to every page. Sound starts only after you interact with the page, pauses while the tab is hidden, and is rate limited so long searches and watch modes stay quiet.

Tests for the module run with `node --test tests/web/*.test.mjs`.

## Easter eggs

- **AGENT OVERDRIVE.** Press the Konami code (up, up, down, down, left, right, left, right, b, a), or tap the nav logo seven times quickly. The agents tick ten times faster, the hero search gets a BFS race beside it, and a badge counts down from 30 seconds. Escape, the code again, the badge, or the 30 seconds end it.
- **Search wave.** Each page arrival plays a BFS wave from the last click, which is the search reveal on every page (`web/js/reveal.js`, loaded by `web/js/nav.js`).
- **Unknown pages.** Any unknown URL gets the 404 page, with a tic-tac-toe game against the solved agent.

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
python -m checkers --size 10 --no-forced-capture --red mcts --white alphabeta # watch MCTS and alpha-beta on 10x10
python -m checkers match mcts alphabeta --games 10 --level 1                 # win/draw/loss table between two agents
python -m lightsout --solve 110/011/101                                      # the fewest presses that clear a 3x3 board
python -m blackjack                                                          # play blackjack against the dealer
python -m blackjack simulate --hands 100000                                  # measured return of basic strategy through a 6-deck shoe
python -m battleship                                                         # play Battleship in the terminal
python -m battleship benchmark --games 100 --seed 1                          # shots to sink a random fleet, per agent
python -m battleship match --games 100 --seed 1                              # AI vs chance (fixed odds) races: wins and mean shots
python -m pacman                                                             # play Pac-Man in the terminal
python -m pacman benchmark --games 200                                       # win rate and score for each agent
python -m pacman match --games 200                                           # the agent against the A* ghosts and against chance ghosts
python -m warehouse                                                          # plan robot routes in the terminal
python -m warehouse benchmark --layout bottleneck --robots 6 --instances 10  # compare the three planners
python -m endgame                                                            # KQK: a random winning position, you are the lone king
python -m endgame --piece R --as strong                                      # KRK: you have the rook; the agent defends and grades your moves
python -m endgame analyze "8/8/8/5k2/8/8/1Q6/K7 w - - 0 1"                   # every move's distance to mate and the agent's choice
python -m endgame --opponent chance                                             # you are the lone king; the agent draws its moves from fixed odds
python -m endgame match --games 200 --piece both --seed 0                       # tablebase vs chance, both seats, no prompts (results/endgame_chance_match.md)
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
python -m poker --opponent chance                                            # play against chance (fixed odds, ignores its card)
python -m poker match --opponent chance --hands 200000 --seed 1             # CFR+ against chance over seeded hands, printed as a result
python -m poker train --save                                                 # train CFR and CFR+ on Leduc and write the committed files
python -m poker train --game kuhn                                            # train Kuhn and check the game value against -1/18
python -m poker exploit                                                      # best responses and exploitability of the committed strategy
python -m poker benchmark --deals 100000                                     # the bot against each baseline; writes results/poker_benchmark.*
python -m minesweeper                                                        # beginner 9x9: you play (click reveals, flag mode flags, h hint)
python -m minesweeper -p expert --seed 7                                     # 30 columns x 16 rows, 99 mines, repeatable
python -m minesweeper watch --agent probability -p intermediate --seed 3 --delay 0.2
python -m minesweeper benchmark --out results                                # rewrites results/minesweeper_benchmark.json and .md (about 10 minutes)
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
python -m queens play                                                        # place 8 queens yourself; h = min-conflicts hint, s = a solution
python -m queens solve -n 8 --agent backtrack --board                        # one agent on one board, printed
python -m queens solve -n 1000000 --agent minconf --seed 1                   # a million queens, summary only
python -m queens benchmark --out results                                     # the committed run: results/queens_benchmark.{json,md}
python -m snake                                  # play yourself, turn by turn (w a s d)
python -m snake play --delay 0.15                # timed play
python -m snake watch --agent planner --seed 3   # watch an agent (random, greedy, planner, evolved, evolved-eval)
python -m snake evolve                           # the neural net's committed run (about 10 minutes, 2 workers)
python -m snake cem                              # the evaluation function's weights (about 55 minutes, 2 workers)
python -m snake benchmark --games 200 --write    # results/snake_benchmark.{json,md}
python -m snake actions --games 40 --write        # results/snake_net_actions.json: the net's left, straight, right counts
python -m rover                                  # 21x21, 20% walls, D* Lite drives, animated
python -m rover --driver astar --density 0.3     # steer with A* replanned from scratch
python -m rover benchmark                        # the seeded comparison in results/rover_benchmark.md
python -m ghosthunt                                # play in the terminal (w/a/s/d, bw/ba/bs/bd to bust)
python -m ghosthunt --autopilot --filter particles # watch the autopilot with particles
python -m ghosthunt benchmark                      # the seeded benchmark in results/ghosthunt_benchmark.md
python -m localize                       # autopilot drives; the particle cloud is drawn on the floor
python -m localize --drive               # drive with w/a/s/d in a terminal (q quits)
python -m localize --kidnap-at 40        # teleport the robot at step 40 and watch the cloud re-spread
python -m localize benchmark             # the seeded benchmark; writes results/localize_benchmark.md
python -m tetris play [--seed N]                                             # turn-based; h shows the agent's placement
python -m tetris watch --pieces 300                                          # the agent plays a seeded game
python -m tetris cem --iters 20 --samples 20                                 # the noisy cross-entropy tuner (v3); writes results/tetris_cem_weights.json and its log
python -m tetris evolve                                                      # the v2 genetic algorithm on the 10-row board; writes results/tetris_ga_weights.json and results/tetris_train_log.jsonl
python -m tetris benchmark --height 10 --cap 100000 --no-lookahead --weights results/tetris_cem_weights.json --out results/tetris_benchmark_hard_v3
python -m nonogram                                  # play the first library picture (rows and columns counted from 1)
python -m nonogram play --puzzle rocket             # a library picture: f r c fills, x r c crosses, h hints, s shows the solution
python -m nonogram play --random 12x12 --seed 7     # a random picture with a unique solution
python -m nonogram solve --puzzle cat --method sat --stats   # line, hybrid, or sat; prints the search counters
python -m nonogram random 10x10 --seed 3 --solve    # clues of a random unique picture
python -m nonogram benchmark --write                # the full run in results/nonogram_benchmark.*
python -m clusters run --algo kmeans --k 4 --data blobs --seed 2
python -m clusters run --algo dbscan --eps 0.09 --min-pts 5 --data rings
python -m clusters run --algo gmm --k 3 --data aniso --seed 1
python -m clusters elbow --data blobs --seed 2 --k-max 10
python -m nnlab train --data spiral --layers 8,8 --act tanh --epochs 500
python -m nnlab train --data xor --layers 4 --epochs 400 --test-frac 0.2
python -m pathfind race --size 41 --maze prim --seed 3
python -m pathfind show --algo astar --size 41 --maze prim --seed 3 --trace 6
python -m mdplab solve --preset cliff --gamma 0.95 --slip 0.1   # value iteration and policy iteration; prints V and the arrow policy
python -m mdplab learn --preset cliff --episodes 2000 --algo q  # Q-learning (or --algo sarsa); prints returns and agreement with the optimal policy
python -m optlab surfaces
python -m optlab race --surface rosenbrock --start -1.5,2 --steps 500
python -m optlab race --surface ravine --steps 300 --noise 0.3 --seed 7 --lr sgd=0.05,adam=0.1 --no-map
python -m treelab grow --data xor --depth 4                  # rules, accuracy, ASCII region map
python -m treelab grow --data nested --criterion entropy --prune 0.02
python -m treelab forest --data spiral --trees 50            # OOB and test accuracy, importances
python -m tictactoe                # play tic-tac-toe in the terminal against the minimax agent
python -m tictactoe stats          # reachable positions, the game value, and the alpha-beta node counts
python -m markov generate --corpus alice --order 3 --words 60 --seed 1   # sample text from the chain
python -m markov generate --corpus sonnets --level char --order 4 --temperature 0.7 --seed 3
python -m markov perplexity --corpus alice --order 1..5                  # train and held-out perplexity per order
python -m regression fit --data noisy-sine --degree 5 --method qr        # least squares by QR
python -m regression fit --data noisy-sine --degree 5 --method gd --steps 500 --lr 0.1
python -m regression sweep --data noisy-sine --max-degree 10            # train and test MSE by degree
python -m regression logistic --data moons --steps 500                  # sigmoid classifier by gradient descent
python -m walkers evolve --generations 50 --pop 80 --seed 1   # one line per generation: best, mean, median, species
python -m walkers replay "<creature string>"                  # ASCII side view of one run
python -m walkers benchmark                                   # simulation throughput and the presets' distances
python -m tictactoe positions      # the alpha-beta search tree from the empty board, as JSON
python -m routes --compare --cities 12                                       # compare the TSP solvers on a random map
python -m npuzzle astar 7,2,4,5,0,6,8,3,1                                    # any algorithm by name
python -m npuzzle.benchmark 8puzzle                                          # results/npuzzle_8puzzle.md
python -m sudoku.generate --count 200                                        # unique-solution puzzles
python -m sudoku solve 009000000160004023000009...                           # or: python -m sudoku benchmark
python -m game2048.benchmark greedy --games 3000 --seed 500000   # n-tuple, no search
python -m game2048.benchmark search --games 60 --workers 8 --budget 0.05 --seed 500000 --agents expectimax
```

Training (CPU is enough; times are for a laptop i7):

```bash
python -m npuzzle.pdb                              # pattern database, ~40 s
python -m npuzzle.train --minutes 45               # neural heuristic
python -m connect4.train_az --hours 3              # AlphaZero self-play (--resume continues; --device cuda --amp on a GPU)
python -m connect4.evaluate                        # AlphaZero vs minimax at depths 2, 4, 6
python -m game2048.train_td --minutes 70 --games 1000 --alpha 0.1 --seed 0   # n-tuple network
python -m game2048.tune --generations 20 --population 32 --elite 8 --games 300 --seed 0   # heuristic weights
```

## Deploying

`Dockerfile` builds the serving image and `fly.toml` runs it on one always-on Fly.io machine (shared CPU, 1 GB):

```bash
fly launch --no-deploy   # first time: create the app
fly deploy
```
