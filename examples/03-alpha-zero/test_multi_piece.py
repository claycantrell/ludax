import jax
import jax.numpy as jnp

from ludax.games import tic_tac_toe
from ludax import LudaxEnvironment, games

BATCH_SIZE = 1

# test_games = games.__all__
test_games = ["tic_tac_toe", "hex", "english_draughts", "reversi"]
for game in test_games:
    jax.debug.print(f"\nGame: {game}")
    
    try:
        env = LudaxEnvironment(game_str=getattr(games, game))
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

        key = jax.random.PRNGKey(0)
        key, subkey = jax.random.split(key)
        keys = jax.random.split(subkey, BATCH_SIZE)

        state = init(keys)
        state, key = run_batch(state, key)
    
        jax.debug.print(f"Legal action mask: {state.legal_action_mask.shape}")
        jax.debug.print(f"Game Board Shape: {state.game_state.board.shape}")
        jax.debug.print(f"Game Info Observation Shape: {env.game_info.observation_shape}")
        jax.debug.print(f"Real Observation Shape:\n{env.observe(state, state.current_player).shape}")

    except Exception as e:
        jax.debug.print(f"Error testing game {game}: {e}")

