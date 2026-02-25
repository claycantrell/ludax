# Training script with simplified evaluation (win/draw rate vs best model).
# Checkpoints are saved every `eval_interval` steps for later tournament evaluation.
#
# Based on the PGX AlphaZero training script:
# https://github.com/sotetsuk/pgx/blob/18799f81a03651e7de8fb9dc79daee9090e2e695/examples/alphazero/train.py

import datetime
import os
import pickle
import time
from functools import partial
from typing import Any, Dict, NamedTuple, Tuple

import haiku as hk
import jax
import jax.numpy as jnp
import mctx
import optax
import wandb
from omegaconf import OmegaConf
from pydantic import BaseModel
from tqdm import tqdm

from ludax import LudaxEnvironment, games

import pgx
from pgx.experimental import auto_reset

from network import AZNet


devices = jax.local_devices()
num_devices = len(devices)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
# python examples/03-alpha-zero/train.py env_id=hex     env_type=pgx     seed=0     max_num_iters=400     num_channels=64     num_layers=4     resnet_v2=True     selfplay_batch_size=4096     num_simulations=32     max_num_steps=128     training_batch_size=4096     learning_rate=0.001     eval_interval=10     eval_num_games=128
class Config(BaseModel):
    env_id: str = "reversi"
    env_type: str = "ldx"
    seed: int = 0
    max_num_iters: int = 1000
    # network params
    num_channels: int = 128
    num_layers: int = 6
    resnet_v2: bool = True
    # selfplay params
    selfplay_batch_size: int = 1024
    num_simulations: int = 32
    max_num_steps: int = 256
    # training params
    training_batch_size: int = 4096
    learning_rate: float = 0.001
    # eval params
    eval_interval: int = 5
    eval_num_games: int = 128  # games per side vs best model (total = 2x this)
    # checkpoint params
    ckpt_dir: str = ""  # auto-generated if empty

    class Config:
        extra = "forbid"

conf_dict = OmegaConf.from_cli()
config: Config = Config(**conf_dict)
print(config)


# ---------------------------------------------------------------------------
# Environment helpers
# ---------------------------------------------------------------------------

@partial(jax.jit, static_argnums=0)
def observe(env, state: pgx.State) -> jnp.ndarray:
    if config.env_type == "pgx":
        return state.observation
    else:
        return env.observe(state, state.current_player)


# ---------------------------------------------------------------------------
# Network
# ---------------------------------------------------------------------------

def forward_fn(x, is_eval=False):
    net = AZNet(
        num_actions=env.num_actions,
        num_channels=config.num_channels,
        num_blocks=config.num_layers,
        resnet_v2=config.resnet_v2,
    )
    policy_out, value_out = net(x, is_training=not is_eval, test_local_stats=False)
    return policy_out, value_out


forward = hk.without_apply_rng(hk.transform_with_state(forward_fn))
optimizer = optax.adam(learning_rate=config.learning_rate)


# ---------------------------------------------------------------------------
# MCTS recurrent function
# ---------------------------------------------------------------------------

def recurrent_fn(model, rng_key, action, state):
    del rng_key
    model_params, model_state = model

    if config.env_type != "pgx":
        action = action.astype(jnp.int8)
    state = jax.vmap(env.step)(state, action)

    (logits, value), _ = forward.apply(
        model_params, model_state, observe(env, state), is_eval=True
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


# ---------------------------------------------------------------------------
# Self-play
# ---------------------------------------------------------------------------

class SelfplayOutput(NamedTuple):
    obs: jnp.ndarray
    reward: jnp.ndarray
    terminated: jnp.ndarray
    action_weights: jnp.ndarray
    discount: jnp.ndarray


@jax.pmap
def selfplay(model, rng_key: jnp.ndarray) -> SelfplayOutput:
    model_params, model_state = model
    batch_size = config.selfplay_batch_size // num_devices

    def step_fn(state, key):
        key1, key2 = jax.random.split(key)
        observation = observe(env, state)

        (logits, value), _ = forward.apply(
            model_params, model_state, observation, is_eval=True
        )
        root = mctx.RootFnOutput(prior_logits=logits, value=value, embedding=state)

        policy_output = mctx.gumbel_muzero_policy(
            params=model,
            rng_key=key1,
            root=root,
            recurrent_fn=recurrent_fn,
            num_simulations=config.num_simulations,
            invalid_actions=~state.legal_action_mask,
            qtransform=mctx.qtransform_completed_by_mix_value,
            gumbel_scale=1.0,
        )

        actor = state.current_player
        keys = jax.random.split(key2, batch_size)
        action = policy_output.action
        if config.env_type != "pgx":
            action = action.astype(jnp.int8)
        state = jax.vmap(auto_reset(env.step, env.init))(state, action, keys)
        discount = -1.0 * jnp.ones_like(value)
        discount = jnp.where(state.terminated, 0.0, discount)

        rewards = state.rewards
        return state, SelfplayOutput(
            obs=observation,
            action_weights=policy_output.action_weights,
            reward=rewards[jnp.arange(rewards.shape[0]), actor],
            terminated=state.terminated,
            discount=discount,
        )

    rng_key, sub_key = jax.random.split(rng_key)
    keys = jax.random.split(sub_key, batch_size)
    state = jax.vmap(env.init)(keys)
    key_seq = jax.random.split(rng_key, config.max_num_steps)
    _, data = jax.lax.scan(step_fn, state, key_seq)
    return data


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

class Sample(NamedTuple):
    obs: jnp.ndarray
    policy_tgt: jnp.ndarray
    value_tgt: jnp.ndarray
    mask: jnp.ndarray


@jax.pmap
def compute_loss_input(data: SelfplayOutput) -> Sample:
    batch_size = config.selfplay_batch_size // num_devices
    value_mask = jnp.cumsum(data.terminated[::-1, :], axis=0)[::-1, :] >= 1

    def body_fn(carry, i):
        ix = config.max_num_steps - i - 1
        v = data.reward[ix] + data.discount[ix] * carry
        return v, v

    _, value_tgt = jax.lax.scan(
        body_fn, jnp.zeros(batch_size), jnp.arange(config.max_num_steps)
    )
    value_tgt = value_tgt[::-1, :]

    return Sample(
        obs=data.obs, policy_tgt=data.action_weights,
        value_tgt=value_tgt, mask=value_mask,
    )


def loss_fn(model_params, model_state, samples: Sample):
    (logits, value), model_state = forward.apply(
        model_params, model_state, samples.obs, is_eval=False
    )
    policy_loss = jnp.mean(optax.softmax_cross_entropy(logits, samples.policy_tgt))
    value_loss = jnp.mean(optax.l2_loss(value, samples.value_tgt) * samples.mask)
    return policy_loss + value_loss, (model_state, policy_loss, value_loss)


@partial(jax.pmap, axis_name="i")
def train(model, opt_state, data: Sample):
    model_params, model_state = model
    grads, (model_state, policy_loss, value_loss) = jax.grad(loss_fn, has_aux=True)(
        model_params, model_state, data
    )
    grads = jax.lax.pmean(grads, axis_name="i")
    updates, opt_state = optimizer.update(grads, opt_state)
    model_params = optax.apply_updates(model_params, updates)
    return (model_params, model_state), opt_state, policy_loss, value_loss


# ---------------------------------------------------------------------------
# Quick evaluation: current model vs best model (sampling-based, fast)
# ---------------------------------------------------------------------------

def _replicate_model(model):
    return jax.device_put_replicated(model, devices)


@partial(jax.pmap, static_broadcasted_argnums=[4])
def _play_match_batch(rng_key, model_a, model_b, init_keys, player_a_side: int):
    """Play a batch of games with sampling (no MCTS) for speed.
    Returns rewards from model_a's perspective.
    """
    params_a, state_a = model_a
    params_b, state_b = model_b
    batch_size = init_keys.shape[0]
    state = jax.vmap(env.init)(init_keys)

    def _get_action(model_params, model_state, obs, legal_mask, key):
        (logits, _), _ = forward.apply(model_params, model_state, obs, is_eval=True)
        logits = jnp.where(legal_mask, logits, jnp.finfo(logits.dtype).min)
        return jax.random.categorical(key, logits, axis=-1)

    def body_fn(val):
        key, st, R = val
        obs = observe(env, st)
        is_a_turn = (st.current_player == player_a_side).reshape((-1, 1))
        legal = st.legal_action_mask

        key, k1, k2 = jax.random.split(key, 3)
        action_a = _get_action(params_a, state_a, obs, legal, k1)
        action_b = _get_action(params_b, state_b, obs, legal, k2)
        action = jnp.where(is_a_turn.squeeze(-1), action_a, action_b)

        if config.env_type != "pgx":
            action = action.astype(jnp.int8)

        st = jax.vmap(env.step)(st, action)
        rewards = st.rewards
        R = R + rewards[jnp.arange(batch_size), player_a_side]
        return (key, st, R)

    _, _, R = jax.lax.while_loop(
        lambda x: ~(x[1].terminated.all()),
        body_fn,
        (rng_key, state, jnp.zeros(batch_size)),
    )
    return R


def play_match(rng_key, model_a, model_b, num_games: int) -> Tuple[int, int, int]:
    """Play `num_games` as each side. Returns (wins_a, draws, wins_b)."""
    games_per_device = num_games // num_devices
    total_wins_a, total_draws, total_wins_b = 0, 0, 0

    rep_a = _replicate_model(model_a)
    rep_b = _replicate_model(model_b)

    for side in [0, 1]:
        rng_key, subkey = jax.random.split(rng_key)
        all_keys = jax.random.split(subkey, num_devices * games_per_device)
        init_keys = all_keys.reshape(num_devices, games_per_device, 2)
        rng_keys = jax.random.split(rng_key, num_devices)

        R = _play_match_batch(rng_keys, rep_a, rep_b, init_keys, side)
        R = R.reshape(-1)
        total_wins_a += int((R > 0.5).sum())
        total_draws += int((jnp.abs(R) < 0.5).sum())
        total_wins_b += int((R < -0.5).sum())

    return total_wins_a, total_draws, total_wins_b


# ===========================================================================
# Main
# ===========================================================================

if __name__ == "__main__":

    # -- Environment setup ------------------------------------------------
    if config.env_id == "othello":
        config.env_id = "reversi"

    env = (
        pgx.make(config.env_id if config.env_id != "reversi" else "othello")
        if config.env_type == "pgx"
        else LudaxEnvironment(game_str=getattr(games, config.env_id))
    )

    # -- Model init -------------------------------------------------------
    wandb.init(project="pgx-az", config=config.model_dump())

    dummy_state = jax.vmap(env.init)(jax.random.split(jax.random.PRNGKey(0), 2))
    dummy_input = observe(env, dummy_state)
    model = forward.init(jax.random.PRNGKey(0), dummy_input)
    opt_state = optimizer.init(params=model[0])
    model, opt_state = jax.device_put_replicated((model, opt_state), devices)

    # -- Checkpoint dir ---------------------------------------------------
    if config.ckpt_dir:
        ckpt_dir = config.ckpt_dir
    else:
        now = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9)))
        now = now.strftime("%Y%m%d%H%M%S")
        ckpt_dir = os.path.join("checkpoints", f"{config.env_id}_{config.env_type}_{now}")
    os.makedirs(ckpt_dir, exist_ok=True)
    print(f"Checkpoints will be saved to: {ckpt_dir}")

    # -- Save initial (random) checkpoint --------------------------------
    model_0_cpu = jax.tree_util.tree_map(lambda x: jax.device_get(x[0]), model)
    with open(os.path.join(ckpt_dir, "000000.ckpt"), "wb") as f:
        pickle.dump({
            "config": config, "model": model_0_cpu,
            "iteration": 0, "frames": 0, "hours": 0.0,
        }, f)

    # -- Best model tracking (for quick eval) -----------------------------
    best_model_cpu = model_0_cpu
    best_iteration = 0

    # -- Training loop ----------------------------------------------------
    iteration = 0
    hours = 0.0
    frames = 0
    rng_key = jax.random.PRNGKey(config.seed)

    pbar = tqdm(total=config.max_num_iters, desc="Training", unit="iter")

    while iteration < config.max_num_iters:
        iteration += 1
        log: Dict[str, Any] = {"iteration": iteration}
        st = time.time()

        # ---- Selfplay ----
        rng_key, subkey = jax.random.split(rng_key)
        keys = jax.random.split(subkey, num_devices)
        data: SelfplayOutput = selfplay(model, keys)
        samples: Sample = compute_loss_input(data)

        # ---- Shuffle & minibatch ----
        samples = jax.device_get(samples)
        frames += samples.obs.shape[0] * samples.obs.shape[1] * samples.obs.shape[2]
        samples = jax.tree_util.tree_map(lambda x: x.reshape((-1, *x.shape[3:])), samples)
        rng_key, subkey = jax.random.split(rng_key)
        ixs = jax.random.permutation(subkey, jnp.arange(samples.obs.shape[0]))
        samples = jax.tree_util.tree_map(lambda x: x[ixs], samples)
        num_updates = samples.obs.shape[0] // config.training_batch_size
        minibatches = jax.tree_util.tree_map(
            lambda x: x.reshape((num_updates, num_devices, -1) + x.shape[1:]), samples
        )

        # ---- Training ----
        policy_losses, value_losses = [], []
        for i in range(num_updates):
            minibatch: Sample = jax.tree_util.tree_map(lambda x: x[i], minibatches)
            model, opt_state, policy_loss, value_loss = train(model, opt_state, minibatch)
            policy_losses.append(policy_loss.mean().item())
            value_losses.append(value_loss.mean().item())
        avg_policy_loss = sum(policy_losses) / len(policy_losses)
        avg_value_loss = sum(value_losses) / len(value_losses)

        et = time.time()
        hours += (et - st) / 3600
        log.update({
            "train/policy_loss": avg_policy_loss,
            "train/value_loss": avg_value_loss,
            "hours": hours,
            "frames": frames,
        })

        # ---- Eval & Checkpoint every eval_interval ----
        if iteration % config.eval_interval == 0:
            model_cpu = jax.tree_util.tree_map(lambda x: jax.device_get(x[0]), model)

            # Quick eval: current vs best
            rng_key, subkey = jax.random.split(rng_key)
            wins, draws, losses = play_match(
                subkey, model_cpu, best_model_cpu, config.eval_num_games
            )
            total = wins + draws + losses
            loss_rate = losses / total if total else 0.0
            win_rate = wins / total if total else 0.0
            draw_rate = draws / total if total else 0.0

            log["eval/vs_best_loss_rate"] = round(loss_rate, 3)
            log["eval/vs_best_win_rate"] = round(win_rate, 3)
            log["eval/vs_best_draw_rate"] = round(draw_rate, 3)
            log["eval/best_iteration"] = best_iteration

            # Update best if current wins majority
            if wins > losses:
                best_model_cpu = model_cpu
                best_iteration = iteration
                log["eval/new_best"] = 1
            else:
                log["eval/new_best"] = 0

            # Save checkpoint
            ckpt_path = os.path.join(ckpt_dir, f"{iteration:06d}.ckpt")
            opt_state_0 = jax.tree_util.tree_map(lambda x: x[0], opt_state)
            with open(ckpt_path, "wb") as f:
                pickle.dump({
                    "config": config,
                    "rng_key": rng_key,
                    "model": model_cpu,
                    "opt_state": jax.device_get(opt_state_0),
                    "iteration": iteration,
                    "frames": frames,
                    "hours": hours,
                    "pgx.__version__": pgx.__version__,
                    "env_id": config.env_id,
                }, f)

            tqdm.write(
                f"[iter {iteration}] vs best (iter {best_iteration}): "
                f"W={wins} D={draws} L={losses}  win_rate={win_rate:.3f}"
            )

        wandb.log(log)
        pbar.update(1)
        pbar.set_postfix(ploss=f"{avg_policy_loss:.4f}", vloss=f"{avg_value_loss:.4f}")

    pbar.close()
    print(f"\nTraining complete. Checkpoints saved to: {ckpt_dir}")
    wandb.finish()