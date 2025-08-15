import jax
import jax.numpy as jnp
from . import BIG, SMALL, zero_heuristic


def random_policy():
    def random_policy_f(state_b, key):
        """
        Randomly select an action from the legal actions.
        :param state_b: Current state of the game.
        :param step_b: Function to step the game forward.
        :param key: JAX PRNG key for random number generation.
        :return: Selected action.
        """
        legal_action_mask = state_b.legal_action_mask
        logits = jnp.log(legal_action_mask.astype(jnp.float32))
        return jax.random.categorical(key, logits=logits, axis=1).astype(jnp.int16)

    return jax.jit(random_policy_f)


def one_ply_policy(step_b, heuristic=zero_heuristic):
    def one_step_lookahead_f(state_b, key):
        """
        Pick the move with the best one‑ply value for the *current* mover (i.e. maximise next_state.mover_reward).
        All else equal, consider heuristic.
        """
        batch_size, num_actions = state_b.legal_action_mask.shape

        # (batch, num_actions) repeat each action for each state in the batch
        all_actions = jnp.broadcast_to(jnp.arange(num_actions), (batch_size, num_actions))
        actions_flat = all_actions.reshape(-1)

        # Repeat every state `num_actions` times
        state_flat = jax.tree_util.tree_map(lambda x: jnp.repeat(x, num_actions, axis=0), state_b)

        # Step every (state, action) pair in one call
        flat_next_state = step_b(state_flat, actions_flat.astype(jnp.int16))

        # Sum the next_state.mover_reward with the heuristic
        # Note: even in games where the players always alternate, this is necessary since the terminal state doesn't
        # switch the current player, and thus the heuristic will be with respect to wrong player.
        # In fact `polarity = jnp.where(next_state.terminated, 1, -1)` is equivalent in Hex
        polarity = jnp.where(flat_next_state.game_state.current_player == state_flat.game_state.current_player, 1.0, -1.0)
        # debug_polarity = jnp.where(flat_next_state.terminated, 0.0, polarity)
        # jax.debug.print("polarity: {polarity}", polarity=debug_polarity, ordered=True)

        # Add some noise to the heuristic to break ties non-deterministically
        noise = jax.random.uniform(key, shape=flat_next_state.mover_reward.shape, minval=-SMALL, maxval=SMALL)

        action_values_flat = BIG * flat_next_state.mover_reward + polarity * heuristic(flat_next_state) + noise

        # Mask out illegal actions. Som
        action_values = action_values_flat.reshape(batch_size, num_actions)
        action_values = jnp.where(state_b.legal_action_mask, action_values, -jnp.inf)

        return jnp.argmax(action_values, axis=1).astype(jnp.int16)

    return jax.jit(one_step_lookahead_f)