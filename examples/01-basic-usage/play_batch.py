import jax
import jax.numpy as jnp

from ludax.games import tic_tac_toe, english_draughts
from ludax import LudaxEnvironment
from ludax.config import ACTION_DTYPE, REWARD_DTYPE


BATCH_SIZE = 1024

env = LudaxEnvironment(game_str=english_draughts)
init = jax.jit(jax.vmap(env.init))
step = jax.jit(jax.vmap(env.step))


def _run_batch(state, key):
    def cond_fn(args):
        state, _, _ = args
        return ~(state.terminated | state.truncated).all()

    def body_fn(args):
        state, key, step_count = args
        key, subkey = jax.random.split(key)
        logits = jnp.log(state.legal_action_mask.astype(REWARD_DTYPE))
        action = jax.random.categorical(key, logits=logits, axis=1).astype(ACTION_DTYPE)
        new_state = step(state, action)
        # Only increment step_count for games not yet done
        done = state.terminated | state.truncated
        step_count = step_count + (~done).astype(ACTION_DTYPE)
        return new_state, key, step_count

    step_count = jnp.zeros(BATCH_SIZE, dtype=ACTION_DTYPE)
    state, key, step_count = jax.lax.while_loop(cond_fn, body_fn, (state, key, step_count))

    return state, key, step_count


run_batch = jax.jit(_run_batch)

key = jax.random.PRNGKey(0)
key, subkey = jax.random.split(key)
keys = jax.random.split(subkey, BATCH_SIZE)

state = init(keys)
state, key, step_counts = run_batch(state, key)

# Rewards: state.rewards shape is (BATCH_SIZE, 2)
rewards = state.rewards
mean_reward_p0 = float(rewards[:, 0].mean())
mean_reward_p1 = float(rewards[:, 1].mean())
mean_game_length = float(step_counts.mean())

print(f"Mean reward P0: {mean_reward_p0:.4f}")
print(f"Mean reward P1: {mean_reward_p1:.4f}")
print(f"Mean game length: {mean_game_length:.1f} steps")