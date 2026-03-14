# AlphaZero Training

Self-play AlphaZero training for Ludax games. Uses Gumbel MuZero policy (via `mctx`) and a ResNet policy/value network. Originally based on the [PGX AlphaZero example](https://github.com/sotetsuk/pgx/blob/main/examples/alphazero/train.py).

## Files

| File | Purpose |
|---|---|
| `train.py` | Self-play training loop; saves checkpoints periodically |
| `tournament.py` | Offline Elo tournament across saved checkpoint directories |
| `plot_tournament.py` | Plots Elo curves from tournament output |
| `network.py` | ResNet-based policy/value network (`AZNet`) |
| `az_config.py` | Shared `Config` dataclass and device setup |

## Training (`train.py`)

Each iteration:
1. **Self-play** — runs `selfplay_batch_size` games in parallel using Gumbel MuZero MCTS
2. **Training** — shuffles self-play data into minibatches and updates the network
3. **Evaluation** (every `eval_interval` iterations) — plays `eval_num_games` against the current best model (sampling only, no MCTS) and updates the best model if the current one wins the majority
4. **Checkpoint** — saves a `.ckpt` file to `ckpt_dir`

Config is passed via CLI. All keys are defined in `az_config.py`.

```bash
# Basic run (defaults)
python examples/03-alpha-zero/train.py

# Override specific keys
python examples/03-alpha-zero/train.py env_id=hex env_type=ldx num_simulations=64
```

Checkpoints are written to `./checkpoints/{env_id}_{env_type}_{timestamp}/` by default, or to `ckpt_dir` if set.

## Tournament (`tournament.py`)

Loads all `.ckpt` files from one or more training run directories and runs a continuous round-robin Elo tournament between them. Matches use sampling (no MCTS) for speed. An optional PGX baseline model can be injected as a fixed reference point.

The tournament runs until interrupted with `Ctrl-C`, periodically saving an Elo snapshot to a `.pkl` file.

```bash
python examples/03-alpha-zero/tournament.py \
    --env_id reversi \
    --env_type ldx \
    --dirs checkpoints/reversi_ldx_run1 checkpoints/reversi_ldx_run2 \
    --games_per_pair 64 \
    --log_interval 100 \
    --output_path examples/03-alpha-zero/data/elo_reversi.pkl \
    --baseline othello_v0   # optional PGX baseline
```

If `--env_type` is omitted, it will be auto-detected from the first checkpoint.

## Plotting (`plot_tournament.py`)

Loads a tournament `.pkl` and produces two figures:

- **Individual runs** — one Elo curve per training run, colored by env type (Ludax vs PGX)
- **Mean ± 1σ** — smoothed mean and variance across runs, one band per env type

Figures are saved as both PDF and PNG.

```bash
python examples/03-alpha-zero/plot_tournament.py \
    --game reversi \
    --output_dir examples/03-alpha-zero/data
```

| Argument | Default | Description |
|---|---|---|
| `--game` | `reversi` | Game name; used to derive the default input filename |
| `--input_path` | `{output_dir}/elo_{game}.pkl` | Explicit path to tournament pkl |
| `--output_dir` | `examples/03-alpha-zero/data` | Directory for output figures |
| `--games_per_batch` | `4096` | Frames per training iteration (for x-axis scaling) |
| `--smooth_window` | `5` | Rolling window for mean/variance plot |

---

## Suggested Configurations

### Reversi (8×8)

Also know as Othello. It's a simple symmetric game. The default trainig parameters are tuned for it.e

```bash
python examples/03-alpha-zero/train.py env_id=reversi
```

### Wolf and Sheep (8×8)

Asymmetric game: Wolf (P1) tries to reach the far edge; Sheep (P2) tries to block. Games are short and the strategy is simple enough that a smaller network trains faster.

```bash
python examples/03-alpha-zero/train.py env_id=wolf_and_sheep
```

### Hex (11×11)

Larger board with longer games. More simulations help since Hex is highly tactical and the value function takes longer to learn. Increase `max_num_steps` to accommodate the full board (up to 121 plies).

```bash
python examples/03-alpha-zero/train.py \
    env_id=hex \
    max_num_steps=256 \
    max_num_iters=400
```
