import jax.numpy as jnp


def bad_heuristic(state_b):
    """
    A bad heuristic that returns the opposite of the reward for each state. Otherwise, 0.0.
    """

    return jnp.where(state_b.terminated, -100* state_b.mover_reward, 0.0)


