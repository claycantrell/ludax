from functools import partial

import jax.numpy as jnp
import jax

@jax.jit
def zero_heuristic(state_b):
    """
    A zero heuristic that returns 0.0 for all states.
    """

    return jnp.zeros(state_b.legal_action_mask.shape[0], dtype=jnp.float32)


def random_playout_heuristic_constructor(step_b, num_playouts=10):
    """
    A heuristic that performs random playouts from the current state and returns the average outcome from the
    perspective of the current player.
    """
    def random_playout_heuristic(state_b, key):

        def cond_fn(carry):
            state, _key = carry
            # continue while at least one playout is not terminated
            return jnp.any(~state.terminated)

        def body_fn(carry):
            state, _key = carry
            _key, subkey = jax.random.split(_key)

            legal_action_mask = state.legal_action_mask  # [N, A] after repeat
            # build logits: 0 for legal moves, -inf for illegal
            logits = jnp.where(legal_action_mask.astype(bool), 0.0, -jnp.inf)  # [N, A]

            # sample one action per state (remove the last axis → shape [N])
            action = jax.random.categorical(subkey, logits=logits, axis=-1).astype(jnp.int16)

            state = step_b(state, action)
            return state, _key

        # Repeat each element of the batch num_playouts times along axis 0
        state_repeated = jax.tree_util.tree_map(
            lambda x: jnp.repeat(x, num_playouts, axis=0), state_b
        )

        final_state_repeated, _ = jax.lax.while_loop(cond_fn, body_fn, (state_repeated, key))

        # Average rewards over playouts, from each original state's current player perspective
        rewards = final_state_repeated.rewards                     # [N, num_players]
        current_players = state_repeated.game_state.current_player # [N]
        batch_size = state_b.legal_action_mask.shape[0]
        N = rewards.shape[0]  # batch_size * num_playouts

        rewards_from_perspective = rewards[jnp.arange(N), current_players]  # [N]
        avg_rewards = rewards_from_perspective.reshape(batch_size, num_playouts).mean(axis=1)

        return avg_rewards

    return jax.jit(random_playout_heuristic)



