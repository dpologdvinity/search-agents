# Training the Connect Four AlphaZero agent

How the Connect Four network was trained, on what hardware, for how long, and how strong it got at each stage. The network learns only from games against itself: no human games, no opening book, and no positions labelled by another engine.

## Method

- **Network**: a residual policy-value network, 4 residual blocks of 32 channels (`connect4/train_az.py`). The policy head scores the 7 columns and the value head predicts the result from the side to move.
- **Self-play**: 128 games are played at once per iteration, each move chosen by PUCT search with 64 simulations. Every game's leaf position goes to the network in one batch per simulation round. Positions are stored with their left-right mirror images in a replay buffer.
- **Training**: 100 steps per iteration on minibatches of 256 from the buffer. The loss is value mean squared error plus policy cross-entropy against the search's visit counts.
- **Serving**: batch norm is folded into the convolutions and the weights are exported to NumPy (`connect4/net.py`), so the website runs the network without PyTorch.
- **Evaluation**: `python -m connect4.evaluate` plays the exported network (200 simulations per move) against alpha-beta minimax at fixed depths. Colours alternate and each game opens with two random moves so games differ.

## Hardware

One laptop: Intel Core i7-1265U (10 cores, 12 threads), no GPU. The laptop was shared with other jobs throughout, so training ran at the lowest scheduling priority and was pinned to a few cores.

## Phase 1: laptop CPU, 4 threads (iterations 1–10)

- 1,280 self-play games.
- 41.5 minutes of compute: 31.6 min self-play, 9.9 min training (about 4.2 min per iteration).
- Check at iteration 10 (20 games per opponent, 100 simulations), score where a win is 1 and a draw is ½: 48% against depth 2, 45% against depth 4, 28% against depth 6.

## Phase 2: laptop CPU, 3 threads on 2 shared cores (iterations 11–19)

- 2,432 self-play games in total.
- 85.1 minutes of compute for 9 iterations (about 9.5 min per iteration, over twice as slow as phase 1).
- **What went wrong**: 3 PyTorch threads pinned to 2 cores that other jobs were also using. The threads spent most of their time waiting on each other. It was worst during evaluation, which runs the network on one position at a time: the evaluation after iteration 19 had not finished after about 2.5 hours, so the run was stopped and that iteration's update was lost. The same evaluation, single-threaded, takes under a minute.
- **Strength at 2,432 games** (50 games per opponent, 200 simulations per move):

| Opponent | Wins | Draws | Losses | Score |
|---|---|---|---|---|
| Alpha-beta minimax, depth 2 | 30 | 4 | 16 | 64% |
| Alpha-beta minimax, depth 4 | 36 | 5 | 9 | 77% |
| Alpha-beta minimax, depth 6 | 18 | 8 | 24 | 44% |

For comparison, plain MCTS with 1,000 random-rollout simulations scores 32-2-16 against depth 4.

## Phase 3: laptop CPU, 1 thread on 1 core (iterations 20–25)

- 3,200 self-play games in total.
- About 1.65 min per iteration, nearly 6 times faster than phase 2 on fewer cores. The lesson: never run more PyTorch threads than dedicated cores.
- Check at iteration 20: 55% against depth 2, 53% against depth 4, 43% against depth 6.

## Phase 4: laptop CPU, 3 threads on 3–4 dedicated cores (iterations 26 onward)

- 4,480 self-play games by iteration 35.
- About 2.3 min per iteration, slower on iterations where other jobs competed for the same cores.
- Check at iteration 30: 65% against depth 2, 43% against depth 4, 48% against depth 6. These in-training checks use only 20 games per opponent, so they move by several points from noise alone.

## Totals on the laptop (as of iteration 35)

- 35 iterations, 4,480 self-play games.
- About 2 hours 40 minutes of training compute (130.6 min self-play, 28.7 min training).
- Wall-clock time was longer: restarts, the stalled evaluation, and other jobs sharing the machine.

## Phase 5: Kaggle GPU (next)

Training moves to a Kaggle notebook with a GPU, resuming from the laptop's last checkpoint (`checkpoints/az_connect4.pt`: weights, optimizer state, and iteration count), so no progress is lost. The goal is to win almost every game against depth-6 minimax. This section will be updated with the hardware, the hours used, and the results.

The full per-iteration log is `results/connect4_train_log.jsonl`.
