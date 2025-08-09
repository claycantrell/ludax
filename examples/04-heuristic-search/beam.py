import jax
import jax.numpy as jnp
from heuristics.test import zero_heuristic
from heuristics import BIG, SMALL

def beam_search_policy(step_b, heuristic=zero_heuristic, topk=200, iterations=10):

    def beam_search_f(state_b, key):
        """Expand the top-k actions at each iteration. Rank the best action based on the heuristic."""


    def beam_step_f(top_states_b_flat, key, ):
        """
        Perform one step of the beam search.
        :param top_states_b_flat: Current top-k descendant states for each root_state in the batch.
        :param key: JAX PRNG key for random number generation.
        :return: New top-k descendant states after one step.
        """


    return jax.jit(beam_search_f)