# search-agents

Eleven puzzles and games, each played or solved by a classic AI algorithm: search guided by hand-built heuristics, pattern databases, or models trained from self-generated data; exact solvers; reinforcement learning; and multi-agent pathfinding. Every algorithm runs in Python, in the terminal and behind a browser frontend.

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
| **Pac-Man** | A* routes for ghosts with chaser, ambusher and scatter personalities; random and greedy reflex baselines | Approximate Q-learning over 13 hand-built features (one is a six-turn survival search over the ghosts' real moves), trained by epsilon-greedy self-play | Each ghost's A* route as glowing lines, every move's Q-value, and the feature contributions behind it |
| **Warehouse robots** (multi-agent pathfinding) | Independent A\*, prioritized planning (robots planned one at a time, earlier paths as moving obstacles) | Conflict-Based Search: optimal sum of costs, branching on each collision and replanning only the constrained robot with space-time A\* | Each robot's path step by step with glowing trails, collisions flashing, the constraint tree growing as CBS branches, and a race between the three planners |

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

For reference, plain MCTS with 1,000 random-rollout simulations scores 32-2-16 against depth-4 minimax. AlphaZero implementations typically train on tens of thousands of self-play games for Connect Four; this one has seen 2,432.


### Lights Out: exact linear algebra over GF(2)

Pressing a light is addition mod 2, so a board is a linear system A x = b with a 25x25 matrix of rank 23. That splits the boards: only 1 in 4 can be cleared at all, and every solvable 5x5 board has exactly 4 solutions. Over 3,000 random solvable boards, the lightest solution averages 9.9 presses, and the heaviest of the four is 5.2 presses heavier on average. Solving a 5x5 board takes about 3 ms in Python.

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

## Architecture

```
web/            static HTML/CSS/JS, no build step
server/         FastAPI: REST for game moves, a WebSocket that streams N-Puzzle searches
npuzzle/ connect4/ checkers/ routes/ game2048/ sudoku/ lightsout/ blackjack/ battleship/ pacman/ warehouse/   search code, training scripts, data files
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
python -m connect4.train_az --hours 3              # AlphaZero self-play
python -m connect4.evaluate                        # AlphaZero vs minimax at depths 2, 4, 6
python -m game2048.train_td --minutes 90           # n-tuple network
```

## Deploying

`Dockerfile` builds the serving image and `fly.toml` runs it on one always-on Fly.io machine (shared CPU, 1 GB):

```bash
fly launch --no-deploy   # first time: create the app
fly deploy
```
