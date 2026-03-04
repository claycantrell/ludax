import jax
import jax.numpy as jnp
from ludax import LudaxEnvironment, games

BATCH_SIZE = 16

test_games = games.__all__
# test_games = ["tic_tac_toe", "hex", "english_draughts", "reversi"]
for game in test_games:
    jax.debug.print(f"\nGame: {game}")
    
    try:
        env = LudaxEnvironment(game_str=getattr(games, game))
        init = jax.jit(jax.vmap(env.init))
        step = jax.jit(jax.vmap(env.step))

        def _run_batch(state, key):
            def cond_fn(args):
                state, _, _ = args
                return ~(state.terminated | state.truncated).all()

            def body_fn(args):
                state, key, step_count = args
                key, subkey = jax.random.split(key)
                logits = jnp.log(state.legal_action_mask.astype(jnp.float32))
                action = jax.random.categorical(key, logits=logits, axis=1).astype(jnp.int16)
                new_state = step(state, action)
                # Only increment step_count for games not yet done
                done = state.terminated | state.truncated
                step_count = step_count + (~done).astype(jnp.int32)
                return new_state, key, step_count

            step_count = jnp.zeros(BATCH_SIZE, dtype=jnp.int32)
            state, key, step_count = jax.lax.while_loop(cond_fn, body_fn, (state, key, step_count))

            return state, key, step_count


        run_batch = jax.jit(_run_batch)

        key = jax.random.PRNGKey(0)
        key, subkey = jax.random.split(key)
        keys = jax.random.split(subkey, BATCH_SIZE)

        state = init(keys)
        state, key, step_counts = run_batch(state, key)

        rewards = state.rewards
        mean_reward_p0 = float(rewards[:, 0].mean())
        mean_reward_p1 = float(rewards[:, 1].mean())
        mean_game_length = float(step_counts.mean())

        jax.debug.print(f"Mean reward P0: {mean_reward_p0:.4f}")
        jax.debug.print(f"Mean reward P1: {mean_reward_p1:.4f}")
        jax.debug.print(f"Mean game length: {mean_game_length:.1f} steps")
    
        jax.debug.print(f"Legal action mask: {state.legal_action_mask.shape}")
        # jax.debug.print(f"Game Board Shape: {state.game_state.board.shape}")
        # jax.debug.print(f"Game Info Observation Shape: {env.game_info.observation_shape}")
        jax.debug.print(f"Real Observation Shape:\n{env.observe(state, state.current_player).shape}")

    except Exception as e:
        jax.debug.print(f"Error testing game {game}: {e}")

