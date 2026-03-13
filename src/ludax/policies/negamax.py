import functools
import jax
import jax.numpy as jnp
from . import BIG, SMALL, zero_heuristic
from ludax.config import ACTION_DTYPE

def negamax_policy(step_b, depth: int, heuristic=zero_heuristic):
    """
    Depth-limited search with correct handling of mover_reward and mover switch.
    - Matches one_ply when depth == 1.
    - V(state, d) is from the *current mover's* perspective.
    - Edge scoring: BIG * next.mover_reward + polarity * V(next, d-1),
      where polarity = +1 if next.current_player == state.current_player else -1.
    """

    def _value_impl(state_b, d: int):
        # Base evaluation from the current mover's perspective
        base_val = heuristic(state_b)  # (batch,)

        if d == 0:
            return base_val

        batch_size, num_actions = state_b.legal_action_mask.shape

        # Repeat states and enumerate actions
        all_actions = jnp.broadcast_to(jnp.arange(num_actions, dtype=ACTION_DTYPE),
                                       (batch_size, num_actions))
        actions_flat = all_actions.reshape(-1)
        state_flat = jax.tree_util.tree_map(lambda x: jnp.repeat(x, num_actions, axis=0), state_b)

        # Step to all children
        next_state_flat = step_b(state_flat, actions_flat)

        # Recursively evaluate children (current mover = next_state.current_player)
        child_vals_flat = value_batched(next_state_flat, d - 1)  # (batch*num_actions,)
        child_vals = child_vals_flat.reshape(batch_size, num_actions)

        # Polarity: +1 if mover didn't switch (e.g., terminal), else -1
        same_mover_flat = (
            next_state_flat.game_state.current_player == state_flat.game_state.current_player
        )
        polarity = jnp.where(same_mover_flat, 1.0, -1.0).reshape(batch_size, num_actions)

        # Immediate reward belongs to the parent mover (good for parent), so ADD it
        edge_reward = next_state_flat.mover_reward.reshape(batch_size, num_actions)

        # Edge score and masking
        scores = BIG * edge_reward + polarity * child_vals
        scores = jnp.where(state_b.legal_action_mask, scores, -jnp.inf)

        # If no legal moves, fall back to base heuristic
        has_legal = jnp.any(state_b.legal_action_mask, axis=1)
        best = jnp.max(scores, axis=1)
        return jnp.where(has_legal, best, base_val)

    # Recursion unrolled at trace-time via static depth
    value_batched = jax.jit(_value_impl, static_argnums=(1,))

    @jax.jit
    def policy_f(state_b, key):
        batch_size, num_actions = state_b.legal_action_mask.shape

        # One-step expansion at root
        all_actions = jnp.broadcast_to(jnp.arange(num_actions, dtype=ACTION_DTYPE),
                                       (batch_size, num_actions))
        actions_flat = all_actions.reshape(-1)
        state_flat = jax.tree_util.tree_map(lambda x: jnp.repeat(x, num_actions, axis=0), state_b)
        next_state_flat = step_b(state_flat, actions_flat)

        # Evaluate children at depth-1
        d_child = max(depth - 1, 0)
        child_vals_flat = value_batched(next_state_flat, d_child)
        child_vals = child_vals_flat.reshape(batch_size, num_actions)

        # Root polarity and edge reward
        same_mover_flat = (
            next_state_flat.game_state.current_player == state_flat.game_state.current_player
        )
        polarity = jnp.where(same_mover_flat, 1.0, -1.0).reshape(batch_size, num_actions)
        edge_reward = next_state_flat.mover_reward.reshape(batch_size, num_actions)

        # Final root scores (matches one_ply when depth == 1)
        scores = BIG * edge_reward + polarity * child_vals

        # Mask & tie-break noise
        scores = jnp.where(state_b.legal_action_mask, scores, -jnp.inf)
        noise = jax.random.uniform(key, shape=(batch_size, num_actions), minval=-SMALL, maxval=SMALL)
        return jnp.argmax(scores + noise, axis=1).astype(ACTION_DTYPE)

    return policy_f
