from functools import partial

from ludax.environment import LudaxEnvironment
import jax
import jax.numpy as jnp


@partial(jax.jit, static_argnames=["seed", "batch_size"])
def play_batch(seed, batch_size):
    """Play a batch of games in the Ludax environment."""

    # Initialize the environment
    env = LudaxEnvironment(game_path="games/tic_tac_toe.ldx")
    init_b = jax.jit(jax.vmap(env.init))
    step_b = jax.jit(jax.vmap(env.step))
    key = jax.random.PRNGKey(seed)
    key, init_key = jax.random.split(key)
    state_b = init_b(jax.random.split(init_key, batch_size))

    def cond_fn(args):
        """Condition function for while loop."""
        state_b, _ = args
        return ~(state_b.terminated | state_b.truncated).all()

    def body_fn(args):
        """Body function for while loop."""
        state_b, key = args

        # Sample an action
        key, subkey = jax.random.split(key)
        logits = jnp.log(state_b.legal_action_mask.astype(jnp.float32))
        action = jax.random.categorical(subkey, logits=logits, axis=1).astype(jnp.int16)

        # Step the environment
        state_b = step_b(state_b, action)

        return state_b, key

    # Run the batch until all games are terminated or truncated
    state_b, key = jax.lax.while_loop(cond_fn, body_fn, (state_b, key))

    return state_b


if __name__ == "__main__":
    state_b = play_batch(seed=0, batch_size=16)
    print("Winner (0: first player, 1: second player, -1: draw):", state_b.winner)
