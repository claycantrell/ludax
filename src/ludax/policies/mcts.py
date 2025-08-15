import jax
import jax.numpy as jnp
from . import zero_heuristic
import mctx

def ludax_recurrent(root_player, step_b, heuristic):
    def recurrent_fn(params, rng_key, action, state):

        next_state = step_b(state, action.astype(jnp.int16))

        # Reward from the root player's perspective
        # If your env stores per-player rewards, pick the root_player's component.
        r = next_state.rewards[jnp.arange(next_state.rewards.shape[0]), root_player]

        # Heuristic for the side to move, flipped to root perspective
        to_play = next_state.game_state.current_player
        sign = jnp.where(to_play == root_player, 1.0, -1.0)
        v = sign * heuristic(next_state)

        # Terminal handling
        v = jnp.where(next_state.terminated, 0.0, v)
        d = jnp.where(next_state.terminated, 0.0, 1.0)  # 1.0 non-terminal, 0.0 terminal

        # Uniform logits over legal moves
        logits = jnp.where(next_state.legal_action_mask, 0.0, -jnp.inf)

        out = mctx.RecurrentFnOutput(
            reward=r,
            discount=d,
            prior_logits=logits,
            value=v,
        )

        return out, next_state
    return jax.jit(recurrent_fn)


def mcts_policy(step_b, heuristic=zero_heuristic, num_simulations=100):
    """
    MCTX-based implementation
    """

    def mcts_policy_f(state_b, key):
        """
        MCTS policy function that uses the MCTX library to select an action based on the current state.
        :param state_b: Current state of the game.
        :param key: JAX PRNG key for random number generation.
        :return: Selected action.
        """
        root_player = state_b.game_state.current_player
        root_logits = jnp.where(state_b.legal_action_mask, 0.0, -jnp.inf)

        root = mctx.RootFnOutput(
            prior_logits=root_logits,
            value=jnp.where(state_b.game_state.current_player == root_player, 1.0, -1.0) * heuristic(state_b),
            embedding=state_b,
        )

        # Initialize MCTX model
        policy_output = mctx.muzero_policy(
            params=None,
            rng_key=key,
            root=root,
            recurrent_fn=ludax_recurrent(root_player, step_b, heuristic),
            num_simulations=num_simulations,
            # dirichlet_fraction=0.0,
            invalid_actions=~state_b.legal_action_mask
        )

        return policy_output.action.astype(jnp.int16)

    return jax.jit(mcts_policy_f)


def gumbel_policy(step_b, heuristic=zero_heuristic, num_simulations=100):
    def policy_f(state_b, key):
        root_player = state_b.game_state.current_player  # shape [B]
        root_logits = jnp.where(state_b.legal_action_mask, 0.0, -jnp.inf)

        root = mctx.RootFnOutput(
            prior_logits=root_logits,
            value=jnp.where(state_b.game_state.current_player == root_player, 1.0, -1.0) * heuristic(state_b),
            embedding=state_b,
        )

        num_actions = state_b.legal_action_mask.shape[1]

        # Option A: Gumbel MuZero – ensures each legal root action is expanded once
        policy_output = mctx.gumbel_muzero_policy(
            params=None,
            rng_key=key,
            root=root,
            recurrent_fn=ludax_recurrent(root_player, step_b, heuristic),
            num_simulations=num_simulations,
            max_num_considered_actions=num_actions,
            invalid_actions=~state_b.legal_action_mask,
            gumbel_scale=0.0  # Perfect information game
        )

        return policy_output.action.astype(jnp.int16)

    return jax.jit(policy_f)
