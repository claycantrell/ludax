import jax.numpy as jnp


def bad_heuristic(state_b):
    """
    A bad heuristic that returns the opposite of the reward for each state. Otherwise, 0.0.
    """

    return jnp.where(state_b.terminated, -100* state_b.mover_reward, 0.0)


def zero_heuristic(state_b):
    """
    A zero heuristic that returns 0.0 for all states.
    """

    return jnp.zeros(state_b.legal_action_mask.shape[0], dtype=jnp.float32)