"""
Post-training Elo tournament.

Loads checkpoints from one or more training run directories and runs a full
round-robin (or sampled) MCTS-based tournament to produce Elo ratings.

Usage:
    python tournament.py \\
        --ckpt_dirs checkpoints/reversi_ldx_20260225025721 checkpoints/reversi_pgx_20260225040246 \\
        --num_games 8 \\
        --num_simulations 32 \\
        --max_opponents 0 \\
        --wandb_project pgx-az-tournament

    # max_opponents=0 means full round-robin (every pair plays).
    # max_opponents=N means each checkpoint plays at most N opponents
    #   (sampled uniformly), which is useful for very large pools.



python examples/03-alpha-zero/tournament.py --ckpt_dirs checkpoints/reversi_ldx_20260225025721 checkpoints/reversi_pgx_20260225040246  --num_games 4 --num_simulations 8 --max_opponents 8 --wandb_project pgx-az-tournament
"""

import argparse
import glob
import itertools
import os
import pickle
import sys
from functools import partial
from typing import Any, Dict, List, NamedTuple, Optional, Tuple

import haiku as hk
import jax
import jax.numpy as jnp
import mctx
import numpy as np
import optax
import wandb
from pydantic import BaseModel
from tqdm import tqdm

from ludax import LudaxEnvironment, games

import pgx
from pgx.experimental import auto_reset


devices = jax.local_devices()
num_devices = len(devices)


# ---------------------------------------------------------------------------
# Local copy of Config so pickle can deserialise checkpoints saved by train.py
# without importing train.py (which runs CLI parsing at module level).
# Fields must be a superset of what train.py defines.
# ---------------------------------------------------------------------------

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
        extra = "allow"  # tolerate fields from older/newer train.py versions


class _PickleHelper(pickle.Unpickler):
    """Unpickler that resolves Config from __main__ to our local copy."""

    def find_class(self, module, name):
        if name == "Config" and module == "__main__":
            return Config
        return super().find_class(module, name)


# ---------------------------------------------------------------------------
# Argparse
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Post-training Elo tournament")
    p.add_argument("--ckpt_dirs", nargs="+", required=True,
                    help="One or more checkpoint directories to include")
    p.add_argument("--num_games", type=int, default=64,
                    help="Games per side per pairing (total per pair = 2x this)")
    p.add_argument("--num_simulations", type=int, default=32,
                    help="MCTS simulations per move during tournament")
    p.add_argument("--max_opponents", type=int, default=0,
                    help="Max opponents per checkpoint (0 = full round-robin)")
    p.add_argument("--initial_elo", type=float, default=1000.0,
                    help="Starting Elo for all checkpoints")
    p.add_argument("--elo_k", type=float, default=32.0,
                    help="K-factor for Elo updates")
    p.add_argument("--num_elo_passes", type=int, default=3,
                    help="Number of passes over all match results to converge ratings")
    p.add_argument("--wandb_project", type=str, default="pgx-az-tournament",
                    help="wandb project name")
    p.add_argument("--wandb_name", type=str, default="",
                    help="wandb run name (auto-generated if empty)")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------

class Checkpoint(NamedTuple):
    run_dir: str       # which training run this came from
    iteration: int
    model: Any         # (params, state) on CPU
    config: Any        # original training config
    label: str         # human-readable label e.g. "run1/iter_100"


def load_checkpoints(ckpt_dirs: List[str]) -> List[Checkpoint]:
    """Load all .ckpt files from the given directories."""
    checkpoints = []
    for d in ckpt_dirs:
        run_name = os.path.basename(d.rstrip("/"))
        files = sorted(glob.glob(os.path.join(d, "*.ckpt")))
        if not files:
            print(f"Warning: no .ckpt files found in {d}")
            continue
        for f in files:
            with open(f, "rb") as fh:
                data = _PickleHelper(fh).load()
            label = f"{run_name}/iter_{data['iteration']}"
            checkpoints.append(Checkpoint(
                run_dir=d,
                iteration=data["iteration"],
                model=data["model"],
                config=data.get("config"),
                label=label,
            ))
        print(f"Loaded {len(files)} checkpoints from {d}")
    print(f"Total checkpoints: {len(checkpoints)}")
    return checkpoints


# ---------------------------------------------------------------------------
# Environment & network setup (rebuilt from checkpoint config)
# ---------------------------------------------------------------------------

# These are set up lazily in main() once we know the config.
env = None
forward = None


def setup_env_and_network(cfg):
    """Initialize the environment and network from a training config."""
    global env, forward

    env_id = cfg.env_id
    env_type = cfg.env_type

    if env_type == "pgx" and env_id == "reversi":
        env_id = "othello"
    if env_type == "ldx" and env_id == "othello":
        env_id = "reversi"

    env = (
        pgx.make(env_id)
        if env_type == "pgx"
        else LudaxEnvironment(game_str=getattr(games, env_id))
    )

    def forward_fn(x, is_eval=False):
        from network import AZNet
        net = AZNet(
            num_actions=env.num_actions,
            num_channels=cfg.num_channels,
            num_blocks=cfg.num_layers,
            resnet_v2=cfg.resnet_v2,
        )
        policy_out, value_out = net(x, is_training=not is_eval, test_local_stats=False)
        return policy_out, value_out

    forward = hk.without_apply_rng(hk.transform_with_state(forward_fn))
    return env, forward


@partial(jax.jit, static_argnums=0)
def observe(env_obj, state):
    # We need env_type from the config; for simplicity we check for observation attr
    if hasattr(state, "observation") and not hasattr(env_obj, "observe"):
        return state.observation
    else:
        return env_obj.observe(state, state.current_player)


# ---------------------------------------------------------------------------
# MCTS recurrent function for tournament play
# ---------------------------------------------------------------------------

def make_recurrent_fn(env_obj, forward_apply, env_type):
    """Create a recurrent_fn closed over the environment."""
    def recurrent_fn(model, rng_key, action, state):
        del rng_key
        model_params, model_state = model

        if env_type != "pgx":
            action = action.astype(jnp.int8)
        state = jax.vmap(env_obj.step)(state, action)

        obs = observe(env_obj, state)
        (logits, value), _ = forward_apply(
            model_params, model_state, obs, is_eval=True
        )
        logits = logits - jnp.max(logits, axis=-1, keepdims=True)
        logits = jnp.where(state.legal_action_mask, logits, jnp.finfo(logits.dtype).min)

        rewards = state.rewards
        reward = rewards[jnp.arange(rewards.shape[0]), state.current_player]
        value = jnp.where(state.terminated, 0.0, value)
        discount = -1.0 * jnp.ones_like(value)
        discount = jnp.where(state.terminated, 0.0, discount)

        return mctx.RecurrentFnOutput(
            reward=reward, discount=discount, prior_logits=logits, value=value
        ), state

    return recurrent_fn


# ---------------------------------------------------------------------------
# Match runner (MCTS-based)
# ---------------------------------------------------------------------------

def _replicate_model(model):
    return jax.device_put_replicated(model, devices)


def build_play_fn(env_obj, forward_apply, recurrent_fn_impl, num_simulations, env_type):
    """Build a jax.pmap'd match function for the given MCTS config."""

    @partial(jax.pmap, static_broadcasted_argnums=[4])
    def _play_match_batch(rng_key, model_a, model_b, init_keys, player_a_side: int):
        params_a, state_a = model_a
        params_b, state_b = model_b
        batch_size = init_keys.shape[0]
        state = jax.vmap(env_obj.init)(init_keys)

        def _get_action_mcts(model_params, model_state, obs, legal_mask, key, st):
            (logits, value), _ = forward_apply(
                model_params, model_state, obs, is_eval=True
            )
            model = (model_params, model_state)
            root = mctx.RootFnOutput(prior_logits=logits, value=value, embedding=st)
            policy_output = mctx.gumbel_muzero_policy(
                params=model,
                rng_key=key,
                root=root,
                recurrent_fn=recurrent_fn_impl,
                num_simulations=num_simulations,
                invalid_actions=~legal_mask,
                qtransform=mctx.qtransform_completed_by_mix_value,
                gumbel_scale=1.0,
            )
            return policy_output.action

        def body_fn(val):
            key, st, R = val
            obs = observe(env_obj, st)
            is_a_turn = (st.current_player == player_a_side).reshape((-1, 1))
            legal = st.legal_action_mask

            key, k1, k2 = jax.random.split(key, 3)
            action_a = _get_action_mcts(params_a, state_a, obs, legal, k1, st)
            action_b = _get_action_mcts(params_b, state_b, obs, legal, k2, st)
            action = jnp.where(is_a_turn.squeeze(-1), action_a, action_b)

            if env_type != "pgx":
                action = action.astype(jnp.int8)

            st = jax.vmap(env_obj.step)(st, action)
            rewards = st.rewards
            R = R + rewards[jnp.arange(batch_size), player_a_side]
            return (key, st, R)

        _, _, R = jax.lax.while_loop(
            lambda x: ~(x[1].terminated.all()),
            body_fn,
            (rng_key, state, jnp.zeros(batch_size)),
        )
        return R

    return _play_match_batch


def play_match(
    play_fn, rng_key, model_a, model_b, num_games: int
) -> Tuple[int, int, int]:
    """Play num_games as each side. Returns (wins_a, draws, wins_b)."""
    games_per_device = max(1, num_games // num_devices)
    total_wins_a, total_draws, total_wins_b = 0, 0, 0

    rep_a = _replicate_model(model_a)
    rep_b = _replicate_model(model_b)

    for side in [0, 1]:
        rng_key, subkey = jax.random.split(rng_key)
        all_keys = jax.random.split(subkey, num_devices * games_per_device)
        init_keys = all_keys.reshape(num_devices, games_per_device, 2)
        rng_keys = jax.random.split(rng_key, num_devices)

        R = play_fn(rng_keys, rep_a, rep_b, init_keys, side)
        R = R.reshape(-1)
        total_wins_a += int((R > 0.5).sum())
        total_draws += int((jnp.abs(R) < 0.5).sum())
        total_wins_b += int((R < -0.5).sum())

    return total_wins_a, total_draws, total_wins_b


# ---------------------------------------------------------------------------
# Elo computation
# ---------------------------------------------------------------------------

def expected_score(ra: float, rb: float) -> float:
    return 1.0 / (1.0 + 10.0 ** ((rb - ra) / 400.0))


def compute_elo_ratings(
    match_results: List[Tuple[int, int, int, int, int]],
    num_players: int,
    initial_elo: float = 1000.0,
    k_factor: float = 32.0,
    num_passes: int = 3,
) -> np.ndarray:
    """Iterative Elo from match results.

    Each entry in match_results: (idx_a, idx_b, wins_a, draws, wins_b).
    Multiple passes over the data help ratings converge.
    """
    ratings = np.full(num_players, initial_elo)
    for _ in range(num_passes):
        for idx_a, idx_b, wins_a, draws, wins_b in match_results:
            total = wins_a + draws + wins_b
            if total == 0:
                continue
            score_a = (wins_a + 0.5 * draws) / total
            ea = expected_score(ratings[idx_a], ratings[idx_b])
            ratings[idx_a] += k_factor * (score_a - ea)
            ratings[idx_b] += k_factor * ((1.0 - score_a) - (1.0 - ea))
    return ratings


# ===========================================================================
# Main
# ===========================================================================

def main():
    args = parse_args()

    # -- Load checkpoints -------------------------------------------------
    checkpoints = load_checkpoints(args.ckpt_dirs)
    if len(checkpoints) < 2:
        print("Need at least 2 checkpoints to run a tournament.")
        return

    # -- Setup env & network from first checkpoint's config ---------------
    cfg = checkpoints[0].config
    if cfg is None:
        print("Error: checkpoint missing 'config' field.")
        return
    env_obj, fwd = setup_env_and_network(cfg)
    env_type = cfg.env_type
    recurrent_fn_impl = make_recurrent_fn(env_obj, fwd.apply, env_type)
    play_fn = build_play_fn(
        env_obj, fwd.apply, recurrent_fn_impl, args.num_simulations, env_type
    )

    # -- Determine pairings -----------------------------------------------
    n = len(checkpoints)
    if args.max_opponents <= 0:
        # Full round-robin
        pairings = list(itertools.combinations(range(n), 2))
    else:
        # Sampled: each player plays at most max_opponents others
        rng = np.random.default_rng(args.seed)
        pairings_set = set()
        for i in range(n):
            others = [j for j in range(n) if j != i]
            k = min(args.max_opponents, len(others))
            chosen = rng.choice(others, size=k, replace=False)
            for j in chosen:
                pair = (min(i, j), max(i, j))
                pairings_set.add(pair)
        pairings = sorted(pairings_set)

    print(f"Tournament: {n} checkpoints, {len(pairings)} pairings, "
          f"{args.num_games} games/side, {args.num_simulations} MCTS sims")

    # -- wandb init -------------------------------------------------------
    run_name = args.wandb_name or f"tournament_{n}ckpts"
    wandb.init(
        project=args.wandb_project,
        name=run_name,
        config={
            "num_checkpoints": n,
            "num_pairings": len(pairings),
            "num_games_per_side": args.num_games,
            "num_simulations": args.num_simulations,
            "ckpt_dirs": args.ckpt_dirs,
            "labels": [c.label for c in checkpoints],
        },
    )

    # -- Play matches -----------------------------------------------------
    rng_key = jax.random.PRNGKey(args.seed)
    match_results = []  # (idx_a, idx_b, wins_a, draws, wins_b)

    for idx_a, idx_b in tqdm(pairings, desc="Playing matches"):
        rng_key, subkey = jax.random.split(rng_key)
        wins_a, draws, wins_b = play_match(
            play_fn, subkey,
            checkpoints[idx_a].model,
            checkpoints[idx_b].model,
            args.num_games,
        )
        match_results.append((idx_a, idx_b, wins_a, draws, wins_b))
        total = wins_a + draws + wins_b
        tqdm.write(
            f"  {checkpoints[idx_a].label} vs {checkpoints[idx_b].label}: "
            f"W={wins_a} D={draws} L={wins_b}  "
            f"({wins_a/total:.1%} / {draws/total:.1%} / {wins_b/total:.1%})"
        )

    # -- Compute Elo ratings ----------------------------------------------
    print(f"\nComputing Elo ratings ({args.num_elo_passes} passes)...")
    ratings = compute_elo_ratings(
        match_results, n,
        initial_elo=args.initial_elo,
        k_factor=args.elo_k,
        num_passes=args.num_elo_passes,
    )

    # -- Print results sorted by Elo -------------------------------------
    ranked = sorted(range(n), key=lambda i: ratings[i], reverse=True)
    print("\n" + "=" * 60)
    print(f"{'Rank':<6}{'Label':<35}{'Elo':>8}")
    print("-" * 60)
    for rank, i in enumerate(ranked, 1):
        print(f"{rank:<6}{checkpoints[i].label:<35}{ratings[i]:>8.1f}")
    print("=" * 60)

    # -- Log to wandb -----------------------------------------------------
    # 1) Elo ratings table
    elo_table = wandb.Table(
        columns=["rank", "label", "run_dir", "iteration", "elo"]
    )
    for rank, i in enumerate(ranked, 1):
        elo_table.add_data(
            rank, checkpoints[i].label, checkpoints[i].run_dir,
            checkpoints[i].iteration, round(ratings[i], 1),
        )
    wandb.log({"tournament/elo_ratings": elo_table})

    # 2) Match results table
    match_table = wandb.Table(
        columns=["player_a", "player_b", "wins_a", "draws", "wins_b", "win_rate_a"]
    )
    for idx_a, idx_b, wa, d, wb in match_results:
        total = wa + d + wb
        match_table.add_data(
            checkpoints[idx_a].label, checkpoints[idx_b].label,
            wa, d, wb, round(wa / total, 3) if total else 0.0,
        )
    wandb.log({"tournament/match_results": match_table})

    # 3) Elo vs iteration plot (one line per run)
    #    Log as individual points so wandb can plot them.
    run_dirs_seen = {}
    for i in ranked:
        cp = checkpoints[i]
        run_name_short = os.path.basename(cp.run_dir.rstrip("/"))
        if run_name_short not in run_dirs_seen:
            run_dirs_seen[run_name_short] = []
        run_dirs_seen[run_name_short].append((cp.iteration, ratings[i]))

    # Log as a wandb Table for flexible plotting
    elo_curve_table = wandb.Table(columns=["run", "iteration", "elo"])
    for run_label, points in run_dirs_seen.items():
        for it, elo in sorted(points):
            elo_curve_table.add_data(run_label, it, round(elo, 1))
    wandb.log({"tournament/elo_curves": elo_curve_table})

    # -- Save results to disk ---------------------------------------------
    results_path = os.path.join(args.ckpt_dirs[0], "tournament_results.pkl")
    with open(results_path, "wb") as f:
        pickle.dump({
            "labels": [c.label for c in checkpoints],
            "ratings": ratings.tolist(),
            "match_results": match_results,
            "args": vars(args),
        }, f)
    print(f"\nResults saved to {results_path}")

    wandb.finish()
    print("Done.")


if __name__ == "__main__":
    main()