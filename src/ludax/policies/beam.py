import jax
import jax.numpy as jnp
from jax import lax

from . import BIG, SMALL, zero_heuristic
from ludax.config import ACTION_DTYPE

from .simple import random_policy

def beam_search_policy(step_b, heuristic=zero_heuristic, cache_size=1000, topk=100, iterations=10):
    # Temporary for debugging
    random_policy_f = random_policy()

    def beam_search_f(state, key):
        """Expand the most important actions at each iteration. Then run minimax on the most important states."""
        """
        Temporary beam search loop with debug output using jax.debug.print.
        Calls beam_step_f_single, prints debug info, and picks a random valid action at the end.
        """
        root_player = state.game_state.current_player

        # Start with the initial state repeated topk times
        frontier_states = jax.tree_util.tree_map(
            lambda x: jnp.repeat(jnp.expand_dims(x, axis=0), topk, axis=0),
            state
        )

        root_value = heuristic(state) # TODO figure out if it's okay to expect heuristic to support both single and batched states
        frontier_values = jnp.full((topk,), -jnp.inf, dtype=root_value.dtype)
        frontier_values = frontier_values.at[0].set(root_value)

        jax.debug.print("[Init] frontier_values = {x}", x=frontier_values)

        for i in range(iterations):
            key, subkey = jax.random.split(key)
            top_states, top_response_values, new_frontier_states, new_frontier_values = beam_step_f_single(
                root_player, frontier_states, frontier_values, subkey
            )
            jax.debug.print("[Iter {i}] frontier_values = {x}", i=i + 1, x=frontier_values)
            jax.debug.print("           top_response_values = {x}", i=i + 1, x=top_response_values)

        return random_policy_f(state, key)


    def beam_step_f_single(root_player, frontier_states, frontier_values, key):
        """
        Perform one step of the beam search, expanding the top-k states and evaluating their actions.
        Top-k is defined from the perspective of the mover sice we want to identify
        """

        # Pop the top-k frontier states and leave the rest in cache.
        # Assumes values_b is from the perspective of the mover to the state. This is not good for the minmax tree since
        # state doesn't have a reference to who was the previous player. But values_b is only used to select which
        # states to expand, as such it's only trying to prioritize exploring good actions for the previous player.
        # The top_response_values are the values are instead used for minimax.
        top_values, top_idx = lax.top_k(frontier_values, topk)
        top_states = jax.tree_util.tree_map(
            lambda x: jnp.take(x, top_idx, axis=0), frontier_states
        )

        # Expand the frontier states
        _topk, num_actions = top_states.legal_action_mask.shape

        all_actions = jnp.broadcast_to(jnp.arange(num_actions), (topk, num_actions))
        actions_flat = all_actions.reshape(-1)

        state_flat = jax.tree_util.tree_map(
            lambda x: jnp.repeat(x, num_actions, axis=0), top_states
        )

        flat_next_state = step_b(state_flat, actions_flat.astype(ACTION_DTYPE))

        # Small noise to break ties
        noise = jax.random.uniform(key, shape=flat_next_state.mover_reward.shape)

        # Signs per term
        heur_sign = jnp.where(flat_next_state.current_player == state_flat.current_player, 1.0, -1.0)

        # Combine from current player's perspective
        action_values_flat = (
                BIG * flat_next_state.mover_reward + heur_sign * heuristic(flat_next_state) + SMALL * noise
        )

        # Mask out dummy children by setting their values to the parent's value if the parent's value is <= -BIG.
        top_values_flat = jnp.repeat(top_values, num_actions)
        action_values_flat = jnp.where(top_values_flat == -jnp.inf, top_values_flat, action_values_flat)

        # Mask illegal actions as -inf
        legality_flat = state_flat.legal_action_mask[
            jnp.arange(actions_flat.shape[0]), actions_flat
        ]
        action_values_flat = jnp.where(legality_flat, action_values_flat, -jnp.inf)


        # Identify the value of the best response to each explored action.
        # This will be used to in the minmax tree, but won't determine which nodes will be expanded next.
        # top_response_values = jnp.zeros_like(top_states.legal_action_mask.shape) # temporary placeholder
        top_response_values = jnp.max(
            action_values_flat.reshape((topk, num_actions)), axis=1
        )
        # Make response values from the perspective of the root not the mover
        root_sign = jnp.where(state_flat.current_player == root_player, 1.0, -1.0)
        top_response_values = root_sign * top_response_values

        # jax.debug.print(
        #     "[beam_step] action_values_flat = {x}",
        #     x=action_values_flat,
        #     ordered=True
        # )

        # Cache states that are not in the top-k
        n = frontier_values.shape[0]
        is_top = jnp.zeros(n, dtype=bool).at[top_idx].set(True)  # mark tops
        rest_idx = jnp.nonzero(~is_top, size=n - topk, fill_value=0)[0]

        cache_values = jnp.take(frontier_values, rest_idx, axis=0)
        cache_states = jax.tree_util.tree_map(
            lambda x: jnp.take(x, rest_idx, axis=0), frontier_states
        )

        # Build combined pool to select the top-k from.
        combined_states = jax.tree_util.tree_map(
            lambda old, new: jnp.concatenate([old, new], axis=0),
            cache_states, flat_next_state
        )
        combined_values = jnp.concatenate([cache_values, action_values_flat], axis=0)

        # Select new top states and values for the cache.
        _top_val, top_indices = lax.top_k(combined_values, cache_size)

        new_frontier_values = jnp.take(combined_values, top_indices, axis=0)
        new_frontier_states = jax.tree_util.tree_map(
            lambda x: jnp.take(x, top_indices, axis=0), combined_states
        )

        # Debugging
        jax.debug.print(
            "[beam_step] legal_pairs={lp}/{tot}, kept_min={mn}, kept_max={mx}",
            lp=jnp.sum(legality_flat), tot=legality_flat.size,
            mn=jnp.min(new_frontier_values), mx=jnp.max(new_frontier_values)
        )

        return top_states, top_response_values, new_frontier_states, new_frontier_values


    return jax.jit(jax.vmap(beam_search_f, in_axes=(0, None), out_axes=0))