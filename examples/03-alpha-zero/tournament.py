"""
python examples/03-alpha-zero/tournament.py --env_id reversi --env_type pgx --games_per_pair 64 --log_interval 100 --output_path examples/05-paper-figures/data/rl_runs/elo_reversi.pkl --baseline=othello_v0 --dirs /users/alexpadula/store/ludax/checkpoints/reversi2/*
python examples/03-alpha-zero/tournament.py --env_id hex --env_type pgx --games_per_pair 64 --log_interval 100 --output_path examples/05-paper-figures/data/rl_runs/elo_hex.pkl --dirs /users/alexpadula/store/ludax/checkpoints/hex2/*
"""

"""
Tournament Elo evaluation for AlphaZero checkpoints.

Takes a list of checkpoint directories (each from a different training run)
and evaluates relative Elo ratings across all checkpoints in a shared tournament.
Now supports injecting a Pgx baseline model into the tournament if one exists.

Key design decisions:
  - No MCTS during evaluation — moves are sampled directly from the policy head.
  - The tournament continuously iterates over shuffled pairings.
  - Matches dynamically compile based on the opponent type (AZNet vs AZNet,
    AZNet vs Baseline, etc.) and cache the compiled JAX functions.
"""

import argparse
import glob
import itertools
import math
import os
import pickle
from functools import partial
from typing import Any, Dict, List, NamedTuple, Tuple

import haiku as hk
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

import pgx
from pgx.experimental import auto_reset

# Lazy imports that depend on the training code
from network import AZNet
from pydantic import BaseModel
from ludax.config import ACTION_DTYPE


class Config(BaseModel):
    env_id: str = "reversi"
    env_type: str = "ldx"
    seed: int = 0
    max_num_iters: int = 1000
    num_channels: int = 128
    num_layers: int = 6
    resnet_v2: bool = True
    selfplay_batch_size: int = 1024
    num_simulations: int = 32
    max_num_steps: int = 256
    training_batch_size: int = 4096
    learning_rate: float = 0.001
    eval_interval: int = 5
    eval_num_games: int = 128
    ckpt_dir: str = ""

    class Config:
        extra = "forbid"


devices = jax.local_devices()
num_devices = len(devices)


# ---------------------------------------------------------------------------
# Elo helpers
# ---------------------------------------------------------------------------

K_FACTOR = 32.0
INITIAL_ELO = 1500.0


def expected_score(elo_a: float, elo_b: float) -> float:
    return 1.0 / (1.0 + 10.0 ** ((elo_b - elo_a) / 400.0))


def update_elo(
    elo_a: float, elo_b: float, wins_a: int, draws: int, wins_b: int
) -> Tuple[float, float]:
    """Update Elo ratings given match results. Returns (new_elo_a, new_elo_b)."""
    total = wins_a + draws + wins_b
    if total == 0:
        return elo_a, elo_b
    score_a = (wins_a + 0.5 * draws) / total
    score_b = 1.0 - score_a
    ea = expected_score(elo_a, elo_b)
    eb = 1.0 - ea
    k = K_FACTOR * math.sqrt(total)
    new_a = elo_a + k * (score_a - ea)
    new_b = elo_b + k * (score_b - eb)
    return new_a, new_b


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------

class Checkpoint(NamedTuple):
    run_name: str       # directory basename (run identifier)
    iteration: int      # training iteration
    path: str           # full path to .ckpt file
    model: Any          # (params, state) tuple or dummy array for baselines
    label: str          # short display label


def load_checkpoints(dirs: List[str]) -> List[Checkpoint]:
    """Load all checkpoints from the given directories, sorted by run then iteration."""
    checkpoints = []
    for d in dirs:
        run_name = os.path.basename(os.path.normpath(d))
        ckpt_files = sorted(glob.glob(os.path.join(d, "*.ckpt")))
        if not ckpt_files:
            print(f"Warning: no .ckpt files found in {d}, skipping.")
            continue
        for path in ckpt_files:
            with open(path, "rb") as f:
                data = pickle.load(f)
            iteration = data.get("iteration", 0)
            model = data["model"]
            label = f"{run_name}/{iteration:06d}"
            checkpoints.append(Checkpoint(
                run_name=run_name,
                iteration=iteration,
                path=path,
                model=model,
                label=label,
            ))
    checkpoints.sort(key=lambda c: (c.run_name, c.iteration))
    return checkpoints


# ---------------------------------------------------------------------------
# Game-playing infrastructure (Heterogeneous)
# ---------------------------------------------------------------------------

def build_forward(env, config_from_ckpt):
    """Build the forward function from a checkpoint's config."""
    def forward_fn(x, is_eval=False):
        net = AZNet(
            num_actions=env.num_actions,
            num_channels=config_from_ckpt.num_channels,
            num_blocks=config_from_ckpt.num_layers,
            resnet_v2=config_from_ckpt.resnet_v2,
        )
        policy_out, value_out = net(x, is_training=not is_eval, test_local_stats=False)
        return policy_out, value_out
    return hk.without_apply_rng(hk.transform_with_state(forward_fn))


def aznet_get_action(forward_fn):
    """Action extraction for Haiku AZNet models."""
    def _get_action(model_pytree, obs, legal_mask, key):
        params, state = model_pytree
        (logits, _), _ = forward_fn.apply(params, state, obs, is_eval=True)
        logits = jnp.where(legal_mask, logits, jnp.finfo(logits.dtype).min)
        return jax.random.categorical(key, logits, axis=-1)
    return _get_action


def baseline_get_action(baseline_fn):
    """Action extraction for Pgx Baseline models."""
    def _get_action(dummy_pytree, obs, legal_mask, key):
        # Baseline model encapsulates its own weights, so we ignore dummy_pytree
        logits, _ = baseline_fn(obs)
        logits = jnp.where(legal_mask, logits, jnp.finfo(logits.dtype).min)
        return jax.random.categorical(key, logits, axis=-1)
    return _get_action


def make_heterogeneous_play_fn(env, env_type, get_action_a, get_action_b):
    """Create the jit-compiled match-playing function between two arbitrary action functions."""
    def _observe(state):
        if env_type == "pgx":
            return state.observation
        else:
            return env.observe(state, state.current_player)

    def _step(state, action):
        return env.step(state, action)

    def _init(key):
        return env.init(key)

    @partial(jax.pmap, static_broadcasted_argnums=[4])
    def play_batch(rng_key, model_a_pytree, model_b_pytree, init_keys, player_a_side: int):
        batch_size = init_keys.shape[0]
        state = jax.vmap(_init)(init_keys)

        def body_fn(val):
            key, st, R = val
            obs = _observe(st)
            is_a_turn = (st.current_player == player_a_side).reshape((-1, 1))
            legal = st.legal_action_mask

            key, k1, k2 = jax.random.split(key, 3)
            action_a = get_action_a(model_a_pytree, obs, legal, k1)
            action_b = get_action_b(model_b_pytree, obs, legal, k2)
            action = jnp.where(is_a_turn.squeeze(-1), action_a, action_b)

            if env_type != "pgx":
                action = action.astype(ACTION_DTYPE)

            st = jax.vmap(_step)(st, action)
            rewards = st.rewards
            R = R + rewards[jnp.arange(batch_size), player_a_side]
            return (key, st, R)

        _, _, R = jax.lax.while_loop(
            lambda x: ~(x[1].terminated.all()),
            body_fn,
            (rng_key, state, jnp.zeros(batch_size)),
        )
        return R

    return play_batch


def play_match(
    rng_key, model_a, model_b, num_games: int, play_batch_fn
) -> Tuple[int, int, int]:
    """Play num_games as each side. Returns (wins_a, draws, wins_b)."""
    games_per_device = max(1, num_games // num_devices)
    total_wins_a, total_draws, total_wins_b = 0, 0, 0

    rep_a = jax.device_put_replicated(model_a, devices)
    rep_b = jax.device_put_replicated(model_b, devices)

    for side in [0, 1]:
        rng_key, subkey = jax.random.split(rng_key)
        all_keys = jax.random.split(subkey, num_devices * games_per_device)
        init_keys = all_keys.reshape(num_devices, games_per_device, 2)
        rng_keys = jax.random.split(rng_key, num_devices)

        R = play_batch_fn(rng_keys, rep_a, rep_b, init_keys, side)
        R = R.reshape(-1)
        total_wins_a += int((R > 0.5).sum())
        total_draws += int((jnp.abs(R) < 0.5).sum())
        total_wins_b += int((R < -0.5).sum())

    return total_wins_a, total_draws, total_wins_b


# ---------------------------------------------------------------------------
# Snapshots
# ---------------------------------------------------------------------------

def save_elo_snapshot(elos: Dict[str, float], checkpoints: List[Checkpoint], step: int, sweep: int, output_path: str):
    """Pickle the current Elo state to disk."""
    snapshot_path = os.path.splitext(output_path)[0] + ".pkl"
    payload = {
        "step": step,
        "sweep": sweep,
        "elos": dict(elos),
        "checkpoints": [
            {"label": c.label, "run_name": c.run_name, "iteration": c.iteration, "path": c.path}
            for c in checkpoints
        ],
    }
    with open(snapshot_path, "wb") as f:
        pickle.dump(payload, f)
    print(f"  Elo snapshot saved to {snapshot_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Tournament Elo evaluation for AZ checkpoints")
    parser.add_argument("--dirs", nargs="+", required=True, help="Checkpoint directories")
    parser.add_argument("--env_id", type=str, default=None, help="Game id (auto-detected if omitted)")
    parser.add_argument("--env_type", type=str, default=None, help="pgx or ldx (auto-detected if omitted)")
    parser.add_argument("--baseline", type=str, default=None, help="Optional Pgx baseline model id (e.g. 'othello_v0')")
    parser.add_argument("--games_per_pair", type=int, default=32, help="Games per pair per side")
    parser.add_argument("--log_interval", type=int, default=None, help="Log every N pairings (default: num_pairs)")
    parser.add_argument("--output_path", type=str, default="elo_ratings.pkl", help="Path to save the Elo plot")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # -- Load checkpoints -------------------------------------------------
    print("Loading checkpoints...")
    checkpoints = load_checkpoints(args.dirs)
    
    if len(checkpoints) == 0:
        print("No checkpoints found. Exiting.")
        return

    # -- Detect env config from first checkpoint --------------------------
    with open(checkpoints[0].path, "rb") as f:
        first_data = pickle.load(f)
    first_config = first_data["config"]

    env_id = args.env_id or getattr(first_config, "env_id", "reversi")
    env_type = args.env_type or getattr(first_config, "env_type", "ldx")

    if env_id == "othello":
        env_id = "reversi"

    print(f"Environment: {env_id} ({env_type})")

    # -- Setup environment and network ------------------------------------
    if env_type == "pgx":
        env = pgx.make(env_id if env_id != "reversi" else "othello")
    else:
        from ludax import LudaxEnvironment, games as ludax_games
        env = LudaxEnvironment(game_str=getattr(ludax_games, env_id))

    forward = build_forward(env, first_config)

    # -- Attempt to load Pgx Baseline -------------------------------------
    has_baseline = False
    baseline_fn = None
    try:
        if args.baseline:
            baseline_id = args.baseline
            # baseline_id = f"{env_id}_v0" if env_id != "reversi" else "othello_v0"
            baseline_fn = pgx.make_baseline_model(baseline_id)
            
            baseline_ckpt = Checkpoint(
                run_name="pgx_baseline",
                iteration=0,
                path="network_download",
                model=jnp.empty(0), # Dummy PyTree array for jax.device_put_replicated
                label=f"baseline/{baseline_id}"
            )
            checkpoints.append(baseline_ckpt)
            has_baseline = True
            print(f"Successfully loaded Pgx baseline: {baseline_id}")
    except Exception as e:
        print(f"Note: Could not load Pgx baseline for '{env_id}' ({e}). Proceeding without it.")

    n = len(checkpoints)
    if n < 2:
        print(f"Need at least 2 agents for a tournament, found {n}.")
        return

    # -- Setup Action Functions & Cache -----------------------------------
    action_fns = {
        "aznet": aznet_get_action(forward)
    }
    if has_baseline:
        action_fns["baseline"] = baseline_get_action(baseline_fn)
        
    compiled_play_fns = {} # Cache for compiled matchup functions

    # -- Init Elo ratings -------------------------------------------------
    elos: Dict[str, float] = {c.label: INITIAL_ELO for c in checkpoints}

    # -- All pairs --------------------------------------------------------
    pairs = list(itertools.combinations(range(n), 2))
    num_pairs = len(pairs)
    log_interval = args.log_interval or num_pairs
    print(f"Total pairs: {num_pairs} — logging every {log_interval} pairings.")

    # -- Continuous tournament loop ----------------------------------------
    rng_key = jax.random.PRNGKey(args.seed)
    np_rng = np.random.default_rng(args.seed)
    step = 0
    sweep = 0
    snapshot_elos = dict(elos)  

    try:
        while True:
            sweep += 1
            order = np_rng.permutation(num_pairs)

            pbar = tqdm(
                order,
                desc=f"Sweep {sweep}",
                unit="pair",
                leave=False,
            )

            for idx in pbar:
                i, j = pairs[idx]
                ca, cb = checkpoints[i], checkpoints[j]

                # Identify matchup type and fetch compiled fn
                type_a = "baseline" if ca.run_name == "pgx_baseline" else "aznet"
                type_b = "baseline" if cb.run_name == "pgx_baseline" else "aznet"
                match_type = (type_a, type_b)

                if match_type not in compiled_play_fns:
                    compiled_play_fns[match_type] = make_heterogeneous_play_fn(
                        env, env_type, action_fns[type_a], action_fns[type_b]
                    )
                
                play_batch_fn = compiled_play_fns[match_type]

                rng_key, subkey = jax.random.split(rng_key)
                wins_a, draws, wins_b = play_match(
                    subkey, ca.model, cb.model,
                    args.games_per_pair, play_batch_fn,
                )

                elos[ca.label], elos[cb.label] = update_elo(
                    elos[ca.label], elos[cb.label], wins_a, draws, wins_b,
                )

                step += 1

                pbar.set_postfix_str(
                    f"{ca.label} vs {cb.label}: "
                    f"W={wins_a} D={draws} L={wins_b}"
                )

                # Periodic logging
                if step % log_interval == 0:
                    max_delta = max(abs(elos[c.label] - snapshot_elos[c.label]) for c in checkpoints)
                    mean_delta = np.mean([abs(elos[c.label] - snapshot_elos[c.label]) for c in checkpoints])

                    save_elo_snapshot(elos, checkpoints, step, sweep, args.output_path)

                    print(f"\n=== Step {step} (sweep {sweep}) | max Δ={max_delta:.1f} | mean Δ={mean_delta:.1f} ===")
                    print(f"\nPress Ctrl-C to stop.\n")

                    snapshot_elos = dict(elos)

            pbar.close()

    except KeyboardInterrupt:
        print("\n\nTournament interrupted by user.")

    # -- Final summary ----------------------------------------------------
    print("\n=== Final Elo Ratings ===")
    ranked = sorted(checkpoints, key=lambda c: elos[c.label], reverse=True)
    for rank, c in enumerate(ranked, 1):
        print(f"  {rank:3d}. {c.label:40s}  Elo={elos[c.label]:7.1f}")

    save_elo_snapshot(elos, checkpoints, step, sweep, args.output_path)
    print("Done.")


if __name__ == "__main__":
    main()