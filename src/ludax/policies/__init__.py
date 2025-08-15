import jax.numpy as jnp

def zero_heuristic(state_b):
    """
    A zero heuristic that returns 0.0 for all states.
    """

    return jnp.zeros(state_b.legal_action_mask.shape[0], dtype=jnp.float32)

BIG = 10**6
SMALL = 10**-3



from .beam import beam_search_policy
from .negamax import negamax_policy
from .simple import one_ply_policy, random_policy
from .mcts import mcts_policy, gumbel_policy


__all__ = [
    "beam_search_policy",
    "negamax_policy",
    "one_ply_policy",
    "random_policy",
    "mcts_policy",
    "gumbel_policy",
    "zero_heuristic",
    "BIG",
    "SMALL"
]