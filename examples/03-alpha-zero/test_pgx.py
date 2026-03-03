import random

import jax
import jax.numpy as jnp
from ludax import LudaxEnvironment, games

# Import the PGX environment
import pgx
from pgx.experimental import auto_reset

game = "tic_tac_toe"
BATCH_SIZE = 2

env_pgx = pgx.make(game)
env_ldx = LudaxEnvironment(game_str=getattr(games, game))

seed = random.randint(0, 1000)

for env, type in zip((env_pgx, env_ldx), ("PGX", "Ludax")):
    print(f"\n\n\n------- Testing {type} environment -------")

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
            action = jax.random.categorical(key, logits=logits, axis=1).astype(jnp.int8)
            state = step(state, action)
            return state, key

        state, key = jax.lax.while_loop(cond_fn, body_fn, (state, key))

        return state, key


    run_batch = jax.jit(_run_batch)

    key = jax.random.PRNGKey(0)
    key, subkey = jax.random.split(key)
    keys = jax.random.split(subkey, BATCH_SIZE)

    state = init(keys)
    state, key = run_batch(state, key)

    jax.debug.print(f"\nCurrent player: {state.current_player}")
    jax.debug.print(f"Legal action mask: {state.legal_action_mask}")
    jax.debug.print(f"Observation:\n{state.observation if type == 'PGX' else env.observe(state, state.current_player)}")

    print("Game over!")
