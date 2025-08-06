import jax
import jax.numpy as jnp

from ludax import LudaxEnvironment


GAME_PATH = "games/tic_tac_toe.ldx"
BATCH_SIZE = 1024

env = LudaxEnvironment(GAME_PATH)
init = jax.jit(jax.vmap(env.init))
step = jax.jit(jax.vmap(env.step))


def _run_batch(state, key):
    def cond_fn(args):
        state, _ = args
        return ~(state.terminated | state.truncated).all()

    def body_fn(args):
        state, key = args
        key, subkey = jax.random.split(key)
        logits = jnp.log(state.legal_action_mask.astype(jnp.float32))
        action = jax.random.categorical(key, logits=logits, axis=1).astype(jnp.int16)
        state = step(state, action)
        return state, key

    state, key = jax.lax.while_loop(cond_fn, body_fn, (state, key))

    return state, key


run_batch = jax.jit(_run_batch)

key = jax.random.PRNGKey(42)
key, subkey = jax.random.split(key)
keys = jax.random.split(subkey, BATCH_SIZE)

state = init(keys)
state, key = run_batch(state, key)
print(f"Winner (0: first player, 1: second player, -1: draw): {state.winner}")