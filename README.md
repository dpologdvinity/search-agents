# search-agents

Agents that solve puzzles and play games by search, each paired with a stronger guide than the textbook version: hand-built heuristics, precomputed pattern databases, and models trained from self-generated data. Every search runs in Python and streams to a browser frontend.

| Domain | Classic search | Learned guidance | Live demo shows |
|---|---|---|---|
| **N-Puzzle** | BFS, DFS, IDS, UCS, bidirectional BFS, greedy, A\*, weighted A\*, IDA\* | Neural cost-to-go heuristic trained by approximate value iteration on top of a 5-5-5 pattern database | Every expansion as it happens, depth vs heuristic plots, side-by-side algorithm comparison |
| **Connect Four** | Alpha-beta negamax, UCT Monte Carlo tree search | AlphaZero-style PUCT with a policy-value ResNet trained only by self-play | The network's prior vs search visits per column, and its win probability |
| **Checkers** | Plain minimax, alpha-beta with iterative deepening, a transposition table, and move ordering | Hand-built evaluation (material, advancement, home row, centre) | How many positions it searched and how many branches pruning cut, with every candidate move's score |
| **Route planner** (traveling salesman) | Nearest neighbour + 2-opt; exact Held-Karp dynamic programming up to 12 cities | Simulated annealing and a genetic algorithm (order crossover, inversion mutation, elitism) | Each route untangling live, length and temperature charts, a three-way race, and drawing your own route to compare |
| **2048** | Expectimax with six hand-crafted features, weights tuned by the cross-entropy method | N-tuple network trained by TD(0) on afterstates, used greedily or inside expectimax | Expected value of each move, search depth |
| **Sudoku** | Backtracking, MRV + forward checking | Constraint propagation (naked and hidden singles) | Every guess, forced fill, and backtrack, replayed |
| **Lights Out** | Gaussian elimination over GF(2), exact (no search): the null space gives every solution and the lightest one is the answer | None: the answer is exact, so there is nothing to learn | The augmented matrix reducing one pivot at a time, the rank, and which boards can never be cleared |

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

Training is in progress on a laptop CPU. After 1,664 self-play games (iteration 13), with 200 simulations per move over 20 games per opponent:

| Opponent | AlphaZero wins | Draws | Losses |
|---|---|---|---|
| Alpha-beta minimax, depth 2 | 7 | 0 | 13 |
| Alpha-beta minimax, depth 4 | 9 | 2 | 9 |
| Alpha-beta minimax, depth 6 | 2 | 6 | 12 |

For reference, plain MCTS with 1,000 random-rollout simulations goes 5-5 against depth-4 minimax. AlphaZero implementations typically need tens of thousands of self-play games on Connect Four; these numbers will be updated as training continues.


### Lights Out: exact linear algebra over GF(2)

Pressing a light is addition mod 2, so a board is a linear system A x = b with a 25x25 matrix of rank 23. That splits the boards: only 1 in 4 can be cleared at all, and every solvable 5x5 board has exactly 4 solutions. Over 3,000 random solvable boards, the lightest solution averages 9.9 presses, and the heaviest of the four is 5.2 presses heavier on average. Solving a 5x5 board takes about 3 ms in Python.

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

## Architecture

```
web/            static HTML/CSS/JS, no build step
server/         FastAPI: REST for game moves, a WebSocket that streams N-Puzzle searches
npuzzle/ connect4/ checkers/ routes/ game2048/ sudoku/ lightsout/   search code, training scripts, data files
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
python -m connect4                                 # play Connect Four against AlphaZero
python -m game2048                                 # play 2048 with w/a/s/d
python -m checkers --agent minimax --level 3       # play checkers against alpha-beta or minimax
python -m lightsout --solve 110/011/101            # the fewest presses that clear a 3x3 board
python -m routes --compare --cities 12             # compare the TSP solvers on a random map
python -m npuzzle astar 7,2,4,5,0,6,8,3,1          # any algorithm by name
python -m npuzzle.benchmark 8puzzle               # results/npuzzle_8puzzle.md
python -m sudoku.generate --count 200            # unique-solution puzzles
python -m sudoku solve 009000000160004023000009...  # or: python -m sudoku benchmark
python -m game2048.benchmark greedy --games 1000
```

Training (CPU is enough; times are for a laptop i7):

```bash
python -m npuzzle.pdb                              # pattern database, ~40 s
python -m npuzzle.train --minutes 45               # neural heuristic
python -m connect4.train_az --hours 3              # AlphaZero self-play
python -m game2048.train_td --minutes 90           # n-tuple network
```

## Deploying

`Dockerfile` builds the serving image and `fly.toml` runs it on one always-on Fly.io machine (shared CPU, 1 GB):

```bash
fly launch --no-deploy   # first time: create the app
fly deploy
```
