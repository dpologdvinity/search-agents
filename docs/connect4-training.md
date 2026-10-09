# Training the Connect Four AlphaZero agent

How the Connect Four network was trained, on what hardware, for how long, and how strong it got at each stage. The network learns only from games against itself: no human games, no opening book, and no positions labelled by another engine.

## Which network is served

The served weights (`connect4/data/az_connect4.npz`) are the export from **iteration 96**, after **34,944 self-play games**: 37 iterations on the laptop, then 59 on a Kaggle notebook (phase 5). The file records this in its `iteration` and `games` keys. Until the Kaggle run, the served network was the laptop's iteration 19 (2,432 games); its numbers are kept below for comparison. The per-iteration log (`results/connect4_train_log.jsonl`) covers all 96 iterations.

## Method

- **Network**: a residual policy-value network, 4 residual blocks of 32 channels (`connect4/train_az.py`). The policy head scores the 7 columns and the value head predicts the result from the side to move.
- **Self-play**: each iteration plays `--games` games at once, and each move is chosen by PUCT search with `--simulations` simulations. Every game's leaf position goes to the network in one batch per simulation round. Positions are stored with their left-right mirror images in a replay buffer.
- **Training**: `--steps` training steps per iteration on minibatches of 256 from the buffer. The loss is value mean squared error plus policy cross-entropy against the search's visit counts.
- **Serving**: batch norm is folded into the convolutions and the weights are exported to NumPy (`connect4/net.py`), so the website runs the network without PyTorch.
- **Evaluation**: `python -m connect4.evaluate` plays the exported network (200 simulations per move) against alpha-beta minimax at fixed depths. Colours alternate and each game opens with two random moves so games differ.

## Hardware

One laptop: Intel Core i7-1265U (10 cores, 12 threads), no GPU. The laptop was shared with other jobs throughout, so training ran at the lowest scheduling priority and was pinned to a few cores.

## Run flags by phase

Every laptop phase used `--games 128 --simulations 64 --steps 100`. They differ in `--threads`, which was matched to the cores each phase could use:

| Phase | Iterations | `--threads` | Cores | Self-play games at the end of the phase |
|---|---|---|---|---|
| 1 | 1–10 | 4 | not pinned to a set (first run) | 1,280 |
| 2 | 11–19 | 3 | 2, shared with other jobs | 2,432 |
| 3 | 20–25 | 1 | 1 | 3,200 |
| 4 | 26–35 | 3 | 3–4, dedicated | 4,480 |

Each restart (before iterations 11, 20 and 26) resumed from the checkpoint without the replay buffer, which was not saved with the checkpoint, so the buffer was refilled from the run's own games. In-training checks ran at iterations 10, 20 and 30, with 20 games per opponent and 100 simulations per move.

## Phase 1: laptop CPU, 4 threads (iterations 1–10)

- 1,280 self-play games.
- 41.5 minutes of compute: 31.6 min self-play, 9.9 min training (about 4.2 min per iteration).
- Check at iteration 10 (20 games per opponent, 100 simulations), score where a win is 1 and a draw is ½: 48% against depth 2, 45% against depth 4, 28% against depth 6.

## Phase 2: laptop CPU, 3 threads on 2 shared cores (iterations 11–19)

- 2,432 self-play games in total, after iteration 19.
- 85.1 minutes of compute for 9 iterations (about 9.5 min per iteration, over twice as slow as phase 1).
- **What went wrong**: 3 PyTorch threads pinned to 2 cores that other jobs were also using. The threads spent most of their time waiting on each other. It was worst during evaluation, which runs the network on one position at a time. The evaluation after iteration 20 had not finished after about 2.5 hours, so the run was stopped. Iteration 20's self-play and training were never checkpointed, so the next run resumed from iteration 19 (the served network).
- **Strength at 2,432 games** (the network served until phase 5, 50 games per opponent, 200 simulations per move; an earlier version of `results/connect4_eval.json`):

| Opponent | Wins | Draws | Losses | Score |
|---|---|---|---|---|
| Alpha-beta minimax, depth 2 | 30 | 4 | 16 | 64% |
| Alpha-beta minimax, depth 4 | 36 | 5 | 9 | 77% |
| Alpha-beta minimax, depth 6 | 18 | 8 | 24 | 44% |

For comparison, plain MCTS with 1,000 random-rollout simulations scores 32-2-16 against depth 4 (same file). Phase 5 re-ran this network over 100 games per opponent with the same protocol as the new one; see the table there.

## Phase 3: laptop CPU, 1 thread on 1 core (iterations 20–25)

- 3,200 self-play games in total.
- About 1.65 min per iteration, nearly 6 times faster than phase 2 on fewer cores. The lesson: never run more PyTorch threads than dedicated cores.
- Check at iteration 20: 55% against depth 2, 53% against depth 4, 43% against depth 6. This is the in-training check of the iteration-20 network, not the served one.

## Phase 4: laptop CPU, 3 threads on 3–4 dedicated cores (iterations 26–35)

- 4,480 self-play games by iteration 35.
- About 2.3 min per iteration, slower on iterations where other jobs competed for the same cores.
- Check at iteration 30: 65% against depth 2, 43% against depth 4, 48% against depth 6. These in-training checks use only 20 games per opponent, so they move by several points from noise alone.

## Totals on the laptop (as of iteration 35)

- 35 iterations, 4,480 self-play games.
- About 2 hours 40 minutes of training compute (130.6 min self-play, 28.7 min training).
- Wall-clock time was longer: restarts, the stalled evaluation, and other jobs sharing the machine.

Two more laptop iterations (36 and 37, 4,736 games) ran before the checkpoint was moved to Kaggle. They played 128 games per iteration with 1 thread, like phase 3.

## Phase 5: Kaggle notebook, 4 CPU threads (iterations 38–96, served)

The Kaggle kernel (`kaggle/train_connect4.py`) resumed from the laptop's iteration-37 checkpoint (`az_connect4.pt`: weights, optimizer state, and iteration count), so no progress was lost. It was set up for a GPU, but no GPU was attached to the session. `check_cuda` found no CUDA device, so the run used the CPU with 4 threads.

- Flags: `--hours 8 --device cpu --games 512 --simulations 100 --batch 512 --steps 400 --buffer 400000 --threads 4 --eval-every 5 --save-buffer --resume`. The buffer was empty at the restart and was full (400,000 positions) from iteration 49.
- 59 iterations in 8 hours, from 4,736 to 34,944 self-play games (512 per iteration).
- 480.7 minutes of compute: 364.7 min self-play and 81.4 min training (about 8.1 min per iteration).
- In-training checks every 5 iterations (20 games per opponent, 100 simulations). The score against depth 6 went from 83% at iteration 40 to 88% at iteration 95, but swung between 53% and 90% along the way. At iteration 95 it was 95% against depth 2 and 100% against depth 4.

**Strength of the served network (iteration 96)** compared with the iteration-19 network it replaced. Both networks played 100 games per opponent with 200 simulations per move and the same seeded openings (`python -m connect4.evaluate --games 100 --depths D`, one run per depth; `results/connect4_eval.json`):

| Opponent | Iteration 19 (2,432 games) | Iteration 96 (34,944 games) |
|---|---|---|
| Alpha-beta minimax, depth 2 | 61-5-34 (63.5%) | 67-8-25 (71%) |
| Alpha-beta minimax, depth 4 | 53-10-37 (58%) | 79-4-17 (81%) |
| Alpha-beta minimax, depth 6 | 29-20-51 (39%) | 77-8-15 (81%) |

Results are wins-draws-losses, and the score counts a draw as half a win. Over 100 games the iteration-19 network scores lower than in its 50-game run (77% against depth 4 there), which shows how much 50 games move from noise. The goal of winning almost every game against depth 6 was not reached: the network still loses 15 of 100 games.

The full per-iteration log is `results/connect4_train_log.jsonl`.
