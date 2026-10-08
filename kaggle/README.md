# Training Connect Four AlphaZero on Kaggle

`train_connect4.py` runs `connect4.train_az` as a Kaggle GPU script kernel. It resumes from a checkpoint attached as a
dataset, trains for a set number of hours, and leaves the outputs in `/kaggle/working`.

| File | Purpose |
|---|---|
| `train_connect4.py` | The kernel script: locates the code and checkpoint, checks the GPU, runs training |
| `kernel-metadata.json` | Kernel template for `kaggle kernels push` (GPU on, internet on, checkpoint dataset attached) |

Kaggle images already include torch and numpy, so nothing is installed. The kernel needs internet only to clone the
public repo (or it can read a packaged copy of the code from an attached dataset, see step 2).

Kaggle names used below: user `kaitlynbassford`, checkpoint dataset `kaitlynbassford/connect4-az-checkpoint`, kernel
`kaitlynbassford/connect4-alphazero-training`.

## One-time setup (on the laptop)

The Kaggle CLI is installed at `~/.local/bin/kaggle` and authenticated. Commands below run from the repo root unless a
path is given.

## 1. Create the checkpoint dataset

Stop any local training run first, so the copied checkpoint is not written while it is copied. Then:

```bash
DS=/tmp/az-checkpoint-ds
mkdir -p $DS
cp checkpoints/az_connect4.pt $DS/
cp checkpoints/az_connect4.buffer.npz $DS/     # optional: only if the run used --save-buffer
cp results/connect4_train_log.jsonl $DS/       # optional: the kernel continues this log
kaggle datasets init -p $DS                    # writes $DS/dataset-metadata.json
```

Edit `$DS/dataset-metadata.json` so that `id` is `kaitlynbassford/connect4-az-checkpoint` and `title` is
`connect4 az checkpoint`. Then create the dataset (private by default):

```bash
kaggle datasets create -p $DS
```

After each training session, download the new outputs and publish a new version of the same dataset:

```bash
kaggle kernels output kaitlynbassford/connect4-alphazero-training -p /tmp/kernel-output
cp /tmp/kernel-output/az_connect4.pt /tmp/kernel-output/az_connect4.buffer.npz \
   /tmp/kernel-output/connect4_train_log.jsonl $DS/
kaggle datasets version -p $DS -m "connect4 az, after session N"
```

Kaggle attaches the newest version each time the kernel runs. Upload a checkpoint only from a finished session.

## 2. Make the code available to the kernel

The kernel looks for the code in this order: `--repo`, a packaged copy in an attached dataset, the copy next to the
script, and finally `git clone https://github.com/dpologdvinity/search-agents`.

The public repo gets snapshot commits only (`/publish`). Until the trainer is published there, the clone is too old and
the script stops with "predates --device". Two ways around that:

- Publish the current trainer to the public repo first, so the clone works as is.
- Package the code yourself as a dataset. Export `connect4/` from the branch, then create the dataset with
  `kaitlynbassford/connect4-trainer-src` as its id:

  ```bash
  SRC=/tmp/trainer-src
  rm -rf $SRC && mkdir -p $SRC
  git archive HEAD connect4 | tar -x -C $SRC      # run from the branch that has the --device trainer
  kaggle datasets init -p $SRC                    # then set "id" to kaitlynbassford/connect4-trainer-src
  kaggle datasets create -p $SRC
  ```

  Then add `"kaitlynbassford/connect4-trainer-src"` to `dataset_sources` in `kernel-metadata.json`. The script finds
  `connect4/train_az.py` inside the attached dataset and copies it to `/tmp/search-agents`, since inputs are read-only.

## 3. Push the kernel

Kaggle pushes the folder that holds the script and its metadata, so stage those two files in their own directory:

```bash
RUN=/tmp/kernel-run
rm -rf $RUN && mkdir -p $RUN
cp kaggle/train_connect4.py kaggle/kernel-metadata.json $RUN/
kaggle kernels push -p $RUN
```

`kernel-metadata.json` sets `enable_gpu` and `enable_internet` to `true`. It pins `machine_shape` to `NvidiaTeslaT4`.
Remove that line to let Kaggle pick whichever GPU is free. Kaggle's current accelerator list does not include the P100,
so the T4 is the expected GPU.

The defaults in `train_connect4.py` are sized for the GPU: `--hours 8`, 512 self-play games per iteration, 100
simulations per move, 512-row minibatches, 400 steps, a 400k-row replay buffer, `--amp` (fp16) on CUDA, 4 CPU
threads, and a minimax evaluation every 5 iterations. Extra flags go to `connect4.train_az` after `--`:

```bash
python kaggle/train_connect4.py --output /tmp/az-smoke --fresh -- --iterations 1 --games 2 --simulations 4 --steps 1
```

The local smoke run uses the CPU (no GPU here), a tiny game count, and `--fresh` because no checkpoint is attached.
Its outputs go to `/tmp/az-smoke`, and that directory can be deleted afterwards.

## 4. Check the status

```bash
kaggle kernels status kaitlynbassford/connect4-alphazero-training
```

Stream the training log with the kernel's Log tab on kaggle.com, or read `train_stdout.log` in the output once it
finishes. Each iteration prints one JSON line with `device`, `amp`, `selfplay_s`, `train_s`, the losses, and
`vs_minimax` on evaluation iterations.

## 5. Download the outputs

```bash
kaggle kernels output kaitlynbassford/connect4-alphazero-training -p /tmp/kernel-output
```

The output directory holds:

- `az_connect4.pt`: the checkpoint (model, optimizer, iteration, games, scaler). Resumes on CPU or CUDA.
- `az_connect4.buffer.npz`: the replay buffer, so the next session continues with the same data.
- `az_connect4.npz`: the exported NumPy weights that the server and `python -m connect4` use.
- `connect4_train_log.jsonl`: one line per iteration, with the earlier sessions included.
- `train_stdout.log`: everything the trainer printed.

To use the weights locally, copy `az_connect4.npz` to `connect4/data/az_connect4.npz`, and the checkpoint and buffer to
`checkpoints/`. Then `python -m connect4.train_az --resume --device cpu` continues the run on the laptop.

## How the script maps to the trainer

| Kernel reads | Kernel writes (`/kaggle/working`) |
|---|---|
| `/kaggle/input/**/az_connect4.pt` (read-only) | `az_connect4.pt` (`--checkpoint-out`) |
| `/kaggle/input/**/az_connect4.buffer.npz` (loaded automatically on resume) | `az_connect4.buffer.npz` (`--save-buffer`) |
| `/kaggle/input/**/connect4_train_log.jsonl` (copied first, so the log continues) | `connect4_train_log.jsonl` (`--log`) |
| | `az_connect4.npz` (`--weights-out`) |
| | `train_stdout.log` |

With no checkpoint attached, the script stops instead of starting over, so a missing dataset cannot silently discard
progress. Use `--fresh` for a new run.

## Limits to know about

- Kaggle caps GPU sessions at about 12 hours (check the current limit on the Kaggle site). The loop stops after the
  iteration that passes `--hours`, so one iteration can run past it. The default of 8 hours leaves room for that.
- Self-play is mostly Python tree search, one simulation round at a time. The GPU speeds up the network calls, so
  larger `--games` helps throughput more than a faster GPU does. The 4-core Kaggle CPU limits the Python side.
- Each minimax evaluation plays single positions, so each network call has batch size 1. On the GPU that is
  latency-bound, so `--eval-every` trades training time for measurement.
