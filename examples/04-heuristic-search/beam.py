import jax
import jax.numpy as jnp
from jax import lax

from heuristics.test import zero_heuristic
from heuristics import BIG, SMALL

def beam_search_policy(step_b, heuristic=zero_heuristic, topk=200, iterations=10):

    def beam_search_f(state, key):
        """Expand the most important actions at each iteration. Then run minimax on the most important states."""
        """
            Temporary beam search loop with debug output using jax.debug.print.
            Calls beam_step_f_single, prints debug info, and picks a random valid action at the end.
            """
        root_player = state.game_state.current_player

        # Start with the initial state repeated topk times
        top_states = jax.tree_util.tree_map(
            lambda x: jnp.repeat(jnp.expand_dims(x, axis=0), topk, axis=0),
            state
        )

        # Initial values — heuristic from root perspective
        init_polarity = jnp.where(
            top_states.game_state.current_player == root_player, 1.0, -1.0
        )
        top_values = init_polarity * heuristic(top_states)

        jax.debug.print("[Init] top_values = {x}", x=top_values)

        for i in range(iterations):
            key, subkey = jax.random.split(key)
            top_states, top_values = beam_step_f_single(
                root_player, top_states, top_values, subkey
            )
            jax.debug.print("[Iter {i}] top_values = {x}", i=i + 1, x=top_values)

        # Pick random legal action from the best state (no dynamic shapes)
        best_idx = jnp.argmax(top_values)
        best_state = jax.tree_util.tree_map(lambda x: x[best_idx], top_states)

        # legal_action_mask is shape [num_actions]. Build logits with -inf for illegal actions.
        legal_mask = best_state.legal_action_mask
        logits = jnp.where(legal_mask, 0.0, -jnp.inf)  # equal weight among legal actions

        key, subkey = jax.random.split(key)
        action = jax.random.categorical(subkey, logits)

        jax.debug.print(
            "[Final choice] Best state index: {idx}, num_legal: {n}, action: {a}",
            idx=best_idx, n=jnp.sum(legal_mask), a=action
        )
        return action

    def beam_step_f_single(root_player, top_states, top_values, key):
        _topk, num_actions = top_states.legal_action_mask.shape

        all_actions = jnp.broadcast_to(jnp.arange(num_actions), (_topk, num_actions))
        actions_flat = all_actions.reshape(-1)

        state_flat = jax.tree_util.tree_map(
            lambda x: jnp.repeat(x, num_actions, axis=0), top_states
        )

        flat_next_state = step_b(state_flat, actions_flat.astype(jnp.int16))

        # Small noise to break ties
        noise = jax.random.uniform(key, shape=flat_next_state.mover_reward.shape, minval=-SMALL, maxval=SMALL)

        # Signs per term
        # Root perspective for the HEURISTIC: + if root is the next player to move
        heur_sign = jnp.where(flat_next_state.game_state.current_player == root_player, 1.0, -1.0)

        # Root perspective for the IMMEDIATE REWARD: + if root was the mover that just acted
        # (the mover in the *previous* state, i.e., state_flat)
        reward_sign = jnp.where(state_flat.game_state.current_player == root_player, 1.0, -1.0)

        # Combine into root player's perspective
        action_values_flat = (
                BIG * reward_sign * flat_next_state.mover_reward
                + heur_sign * heuristic(flat_next_state)
                + noise
        )



        # legality for each (state, action) pair
        legality_flat = state_flat.legal_action_mask[
            jnp.arange(actions_flat.shape[0]), actions_flat
        ]

        # keep illegal as -inf in values (safe for later concatenation)
        action_values_flat = jnp.where(legality_flat, action_values_flat, -jnp.inf)

        # jax.debug.print(
        #     "[beam_step] action_values_flat = {x}",
        #     x=action_values_flat,
        #     ordered=True
        # )

        # Build combined pool
        combined_values = jnp.concatenate([top_values, action_values_flat], axis=0)
        combined_states = jax.tree_util.tree_map(
            lambda old, new: jnp.concatenate([old, new], axis=0),
            top_states, flat_next_state
        )

        # Legality mask for the combined pool:
        # - previous top_k states are valid candidates (True)
        # - new action candidates follow legality_flat
        combined_legal = jnp.concatenate(
            [jnp.ones_like(top_values, dtype=bool), legality_flat], axis=0
        )

        # Select new top-k by absolute value, but make illegal magnitudes -inf so they can never win
        abs_vals = jnp.abs(combined_values)
        abs_vals = jnp.where(combined_legal, abs_vals, -jnp.inf)

        _top_abs, top_indices = lax.top_k(abs_vals, _topk)

        new_top_values = jnp.take(combined_values, top_indices, axis=0)
        new_top_states = jax.tree_util.tree_map(
            lambda x: jnp.take(x, top_indices, axis=0), combined_states
        )

        # Debugging
        jax.debug.print(
            "[beam_step] legal_pairs={lp}/{tot}, kept_abs_min={mn}, kept_abs_max={mx}",
            lp=jnp.sum(legality_flat), tot=legality_flat.size,
            mn=jnp.min(jnp.abs(new_top_values)), mx=jnp.max(jnp.abs(new_top_values))
        )

        return new_top_states, new_top_values


    return jax.jit(jax.vmap(beam_search_f, in_axes=(0, None), out_axes=0))