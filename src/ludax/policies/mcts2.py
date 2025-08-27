from functools import partial

import jax
import jax.numpy as jnp
from . import random_playout_heuristic_constructor
import mctx

# 1 No difference: support one-ply logits
# 2 TODO: support one-ply logits with heuristic
# 3 10% improvement: default to a random rollout value fn if no heuristic is given
# 4 TODO: integrate random rollouts with heuristics

WINNING_LOGIT = 300.0
LOSING_LOGIT = 200.0
LEGAL_LOGIT = 100.0

@partial(jax.jit, static_argnames=['step_b'])
def one_ply_logits(step_b, state_b, root_player_b):
    """
    Return the logits for exploring each action based on a one-ply lookahead.
    This is intended to be used as the prior logits in MCTS.
    """
    batch_size, num_actions = state_b.legal_action_mask.shape

    # (batch, num_actions) repeat each action for each state in the batch
    all_actions = jnp.broadcast_to(jnp.arange(num_actions), (batch_size, num_actions))
    actions_flat = all_actions.reshape(-1)

    # Repeat every state `num_actions` times
    state_flat = jax.tree_util.tree_map(lambda x: jnp.repeat(x, num_actions, axis=0), state_b)

    # Repeat the root player index across actions so shapes align
    root_player_flat = jnp.repeat(root_player_b, num_actions, axis=0)

    # Step every (state, action) pair in one call
    next_state_flat = step_b(state_flat, actions_flat.astype(jnp.int16))

    # Reward from the root player's perspective after taking the action
    root_rewards_flat = next_state_flat.rewards[
        jnp.arange(next_state_flat.rewards.shape[0]),
        root_player_flat
    ]

    # Start all actions at 100 (we'll mask illegals to 0 at the end)
    logits_flat = jnp.full(root_rewards_flat.shape, LEGAL_LOGIT, dtype=jnp.float32)

    # Upgrade legal logits: 200 if opponent wins, 300 if root wins
    logits_flat = jnp.where(root_rewards_flat < 0, LOSING_LOGIT, logits_flat)
    logits_flat = jnp.where(root_rewards_flat > 0, WINNING_LOGIT, logits_flat)

    # Mask out illegal actions by setting their logits to 0
    legal_flat = state_b.legal_action_mask.reshape(-1)
    logits_flat = jnp.where(legal_flat, logits_flat, 0.0)

    # Reshape back to (batch, num_actions)
    return logits_flat.reshape(batch_size, num_actions).astype(jnp.float32)


def ludax_recurrent(root_player_b, step_b, heuristic):
    def recurrent_fn(params, rng_key, action, state):

        key, sub_key = jax.random.split(rng_key)

        next_state = step_b(state, action.astype(jnp.int16))

        # Reward from the root player's perspective
        r = next_state.rewards[jnp.arange(next_state.rewards.shape[0]), root_player_b]
        l = one_ply_logits(step_b, next_state, root_player_b)

        # Heuristic for the side to move, flipped to root perspective
        to_play = next_state.game_state.current_player
        sign = jnp.where(to_play == root_player_b, 1.0, -1.0)
        v = sign * heuristic(next_state, sub_key)

        # Terminal handling
        v = jnp.where(next_state.terminated, 0.0, v)
        d = jnp.where(next_state.terminated, 0.0, 1.0)  # 1.0 non-terminal, 0.0 terminal

        out = mctx.RecurrentFnOutput(
            reward=r,
            discount=d,
            prior_logits=l,
            value=v,
        )

        return out, next_state
    return jax.jit(recurrent_fn)


def mcts_policy(step_b, heuristic=None, num_simulations=100):
    """
    MCTX-based implementation
    """
    if heuristic is None:
        heuristic = random_playout_heuristic_constructor(step_b)

    def mcts_policy_f(state_b, key):
        """
        MCTS policy function that uses the MCTX library to select an action based on the current state.
        :param state_b: Current state of the game.
        :param key: JAX PRNG key for random number generation.
        :return: Selected action.
        """
        root_player_b = state_b.game_state.current_player
        root_logits = one_ply_logits(step_b, state_b, root_player_b)

        root = mctx.RootFnOutput(
            prior_logits=root_logits,
            value=jnp.where(state_b.game_state.current_player == root_player_b, 1.0, -1.0) * heuristic(state_b),
            embedding=state_b,
        )

        # Initialize MCTX model
        policy_output = mctx.muzero_policy(
            params=None,
            rng_key=key,
            root=root,
            recurrent_fn=ludax_recurrent(root_player_b, step_b, heuristic),
            num_simulations=num_simulations,
            # dirichlet_fraction=0.0,
            invalid_actions=~state_b.legal_action_mask
        )

        return policy_output.action.astype(jnp.int16)

    return jax.jit(mcts_policy_f)


def gumbel_policy(step_b, heuristic=None, num_simulations=100):
    if heuristic is None:
        heuristic = random_playout_heuristic_constructor(step_b)

    def policy_f(state_b, key):
        root_player_b = state_b.game_state.current_player  # shape [B]
        root_logits = one_ply_logits(step_b, state_b, root_player_b)

        key, subkey = jax.random.split(key)

        root = mctx.RootFnOutput(
            prior_logits=root_logits,
            value=jnp.where(state_b.game_state.current_player == root_player_b, 1.0, -1.0) * heuristic(state_b, subkey),
            embedding=state_b,
        )

        num_actions = state_b.legal_action_mask.shape[1]

        # Option A: Gumbel MuZero – ensures each legal root action is expanded once
        policy_output = mctx.gumbel_muzero_policy(
            params=None,
            rng_key=key,
            root=root,
            recurrent_fn=ludax_recurrent(root_player_b, step_b, heuristic),
            num_simulations=num_simulations,
            max_num_considered_actions=num_actions,
            invalid_actions=~state_b.legal_action_mask,
            gumbel_scale=0.0  # Perfect information game
        )

        return policy_output.action.astype(jnp.int16)

    return jax.jit(policy_f)
