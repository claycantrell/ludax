import jax
import jax.numpy as jnp
from heuristics.test import zero_heuristic
from heuristics import BIG, SMALL

def beam_search_policy(step_b, heuristic=zero_heuristic, topk=200, iterations=10):

    def beam_search_f(state, key):
        """Expand the most important actions at each iteration. Then run minimax on the most important states."""

        # top_action_sequences



    def beam_step_f_single(root_player, top_states, top_values, key):
        """
        Perform one step of the beam search.
        :param top_states_flat: Current top-k descendant states, as measured by abs(top_values)
        :param top_values: Vales of the top-k states, from the prospective of the root player.
        :param key: JAX PRNG key for random number generation.
        :return: New top-k descendant states after one step.
        """

        _topk, num_actions = top_states.legal_action_mask.shape

        # (batch, num_actions) repeat each action for each state in the batch
        all_actions = jnp.broadcast_to(jnp.arange(num_actions), (topk, num_actions))
        actions_flat = all_actions.reshape(-1)

        # Repeat every state `num_actions` times
        state_flat = jax.tree_util.tree_map(lambda x: jnp.repeat(x, num_actions, axis=0), top_states)

        # Step every (state, action) pair in one call
        flat_next_state = step_b(state_flat, actions_flat.astype(jnp.int16))

        # Polarity of the heuristic, if the next state is played by the opponent to the mover, then the heuristic is negative.
        # We'll make the action values from the perspective of the root player later.
        polarity = jnp.where(flat_next_state.game_state.current_player == state_flat.game_state.current_player, 1.0,
                             -1.0)

        # Add some noise to the heuristic to break ties non-deterministically
        noise = jax.random.uniform(key, shape=flat_next_state.mover_reward.shape, minval=-SMALL, maxval=SMALL)

        action_values_flat = BIG * flat_next_state.mover_reward + polarity * heuristic(flat_next_state) + noise

        # Mask out illegal actions without reshaping the action values
        action_values_flat = jnp.where(state_flat.legal_action_mask, action_values_flat, -jnp.inf)

        # Get the top-k from the combined previous top-k vales and the new action values by:
        # 1. Concatenating the previous top-k values with the new action values
        # 2. Taking the top-k indices from the concatenated values












    return jax.jit(jax.vmap(beam_search_f, in_axes=(0, None), out_axes=0))