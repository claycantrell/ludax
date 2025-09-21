'''
An extremely vanilla implementation of UCB Monte Carlo Tree Search (MCTS) in JAX / Ludax, emphasizing clarity over performance.

Largely inspired by: https://github.com/chrisgrimm/muzero/blob/main/networks/mcts.py
'''


from typing import NamedTuple, Tuple

import jax
import jax.ops
import jax.numpy as jnp
import numpy as np

from ludax import LudaxEnvironment
from ludax.config import State


REWARD_SCALE = 1
class MCTSParams(NamedTuple):
    '''
    The main data structure that stores the current state of the MCTS tree

    Params:
    - node_num (int): the current number of nodes in the tree / index of the next node to be added
    - transitions (array of shape [num_sims, num_actions]): for each node, the index of the child node for each action, or -1 if that action has not been taken yet
    - states (array of shape [num_sims, state_dim]): the game state at each node, used for expansion via the environment's step function
    - legal_actions (array of shape [num_sims, num_actions]): the legal action mask at each node
    - visits (array of shape [num_sims, num_actions]): the visit count N(s,a) for each node / action
    - rewards (array of shape [num_sims, num_actions]): the total reward W(s,a) for each node / action
    - to_play (array of shape [num_sims,]): the player to play at each node (used to flip the sign of rewards for the opponent)
    '''
    player_idx: int
    node_num: int
    transitions: jnp.ndarray
    states: jnp.ndarray
    legal_actions: jnp.ndarray
    visits: jnp.ndarray
    rewards: jnp.ndarray
    to_play: jnp.ndarray

class MCTSTraversal(NamedTuple):
    '''
    Stores the information from a single rollout from the root to a leaf node

    Params:
    - nodes (array of shape [max_depth,]): the indices of the nodes visited during the rollout
    - actions (array of shape [max_depth,]): the actions taken at each node during the rollout
    - valid (array of shape [max_depth,]): whether each step in the rollout was valid (i.e. did not reach a terminal / leaf state)
    '''
    nodes: jnp.ndarray
    actions: jnp.ndarray
    valid: jnp.ndarray

def initialize(environment: LudaxEnvironment, root_state: State, num_sims: int, max_depth: int, key) -> Tuple[MCTSParams, jnp.ndarray]:
    '''
    Initialize the MCTS tree with the root node

    Params:
    - environment (LudaxEnvironment): the environment to use for expansion
    - root_state (State): the initial state of the game at the root node
    - num_sims (int): the maximum number of nodes in the tree
    - max_depth (int): the maximum depth of the tree (used to size the rollout arrays). For safety, this can be set to num_sims
    - seed (int): random seed for initialization

    Returns:
    - params (MCTSParams): the initialized MCTS parameters
    - key (jax.random.PRNGKey): the updated random key
    '''
    transitions = -jnp.ones((num_sims, environment.num_actions), dtype=jnp.int32)
    states = jax.vmap(lambda i: root_state)(jnp.arange(num_sims))
    legal_actions = jnp.ones((num_sims, environment.num_actions), dtype=jnp.int32)
    visits = jnp.zeros((num_sims, environment.num_actions), dtype=jnp.int32)
    rewards = jnp.zeros((num_sims, environment.num_actions), dtype=jnp.float32)
    to_play = jnp.zeros((num_sims,), dtype=jnp.int32)

    legal_actions = legal_actions.at[0].set(root_state.legal_action_mask)
    to_play = to_play.at[0].set(root_state.current_player)
    
    params = MCTSParams(
        player_idx=root_state.current_player,
        node_num=0,
        transitions=transitions,
        states=states,
        legal_actions=legal_actions,
        visits=visits,
        rewards=rewards,
        to_play=to_play
    )
    
    return params, key

def next_state(mcts_params: MCTSParams, node_idx: int, action: int) -> int:
    '''
    Returns the node index of the child node resulting from taking the given action at the given node
    '''
    return mcts_params.transitions[node_idx, action]

def is_valid_node(mcts_params: MCTSParams, node_idx: int, action: int) -> bool:
    '''
    Returns whether the given action is valid at the given node (i.e. whether the child node has already been expanded)
    '''
    # return mcts_params.transitions[node_idx, action] != -1 & mcts_params.legal_actions[node_idx, action]
    return mcts_params.transitions[node_idx, action] != -1

def select_action_ucb(mcts_params: MCTSParams, node_idx: int, key: jax.random.PRNGKey, c: float = 1.41) -> int:
    '''
    Selects an action at the given node using the UCB formula

    Params:
    - mcts_params (MCTSParams): the current MCTS parameters
    - node_idx (int): the index of the node to select an action from (the parent)
    - c (float): the exploration constant

    Returns:
    - action (int): the selected action
    '''
    visits = mcts_params.visits[node_idx]
    legal_actions = mcts_params.legal_actions[node_idx]

    # The rewards need to be flipped if the player to play at this node is not the root player
    # jax.debug.print("Current player is root? {}", mcts_params.to_play[node_idx] == mcts_params.player_idx)
    rewards = jax.lax.select(
        mcts_params.to_play[node_idx] == mcts_params.player_idx,
        mcts_params.rewards[node_idx],
        -mcts_params.rewards[node_idx]
    )
    '''
    rewards = rewards if (player := root_player) else -rewards
    '''
    # rewards = mcts_params.rewards[node_idx]
    
    total_visits = jnp.sum(visits)
    
    # UCB formula
    ucb_values = jnp.where(
        visits > 0,
        rewards / visits + c * jnp.sqrt(jnp.log(total_visits) / visits),
        jnp.inf
    )

    # ucb_values = rewards / (visits + 1e-6) + c * jnp.sqrt(jnp.log(total_visits + 1) / (visits + 1e-6))

    ucb_values = jnp.where(legal_actions, ucb_values, -jnp.inf)
    # max_action = jnp.argmax(ucb_values)

    logits = jnp.where(ucb_values == ucb_values.max(), 0.0, -jnp.inf)
    max_action = jax.random.categorical(key, logits=logits, axis=0).astype(jnp.int16)

    # jax.debug.print("Selecting action {} at node {}, to_play {}, root_player {}", max_action, node_idx, mcts_params.to_play[node_idx], mcts_params.player_idx)

    return max_action, ucb_values

def traverse_to_leaf(mcts_params: MCTSParams, max_depth: int, key: jax.random.PRNGKey) -> MCTSTraversal:
    '''
    Proceeds from the root node to a leaf node by selecting actions using UCB

    Params:
    - mcts_params (MCTSParams): the current MCTS parameters
    - max_depth (int): the maximum depth of the tree (used to size the rollout arrays)

    Returns:
    - rollout (MCTSRollout): the rollout information
    '''

    def body_fn(carry, i):
        node_idx, action, valid, key = carry

        key, subkey = jax.random.split(key)

        # If we've reached an unexpanded node, then all subsequent steps are invalid
        new_valid = valid & is_valid_node(mcts_params, node_idx, action)

        # Advance to the next node and select the next action. We don't need to worry about
        # the behavior here if the current node is invalid, since later all that information
        # will get discarded
        next_node_idx = next_state(mcts_params, node_idx, action)
        next_action, _ = select_action_ucb(mcts_params, next_node_idx, key)

        new_carry = (next_node_idx, next_action, new_valid, key)
        output = (node_idx, action, valid)

        return new_carry, output

    # Initialize the scan with the root node and an initial action
    node_idx, (action, _) = 0, select_action_ucb(mcts_params, 0, key)
    # jax.debug.print("\nStarting rollout from node {}, action {}", node_idx, action)
    init_carry = (node_idx, action, True, key)
    (_, _, _, key), (nodes, actions, valid) = jax.lax.scan(body_fn, init_carry, jnp.arange(max_depth))

    rollout = MCTSTraversal(nodes=nodes, actions=actions, valid=valid)

    return rollout, key

def expand_leaf(mcts_params: MCTSParams, traversal: MCTSTraversal, environment: LudaxEnvironment, step_fn: callable, key: jax.random.PRNGKey) -> MCTSParams:
    '''
    Expands the leaf node reached by the given rollout

    Params:
    - mcts_params (MCTSParams): the current MCTS parameters
    - rollout (MCTSRollout): the rollout information
    - environment (LudaxEnvironment): the environment to use for expansion

    Returns:
    - mcts_params (MCTSParams): the updated MCTS parameters
    '''
    
    num_steps = len(traversal.nodes)
    last_valid_step = jnp.argmax(jnp.where(traversal.valid, jnp.arange(num_steps), 0))
    node_idx, action = traversal.nodes[last_valid_step], traversal.actions[last_valid_step]

    # jax.debug.print(" - Expanding leaf at depth {}, node {}, action {}", last_valid_step, node_idx, action)
    # jax.debug.print(" - Expanding leaf at depth {}, node {}, actions {}", last_valid_step, node_idx, rollout.actions)
    leaf_state = jax.tree_util.tree_map(lambda x: x[node_idx], mcts_params.states)
    next_state = step_fn(leaf_state, action.astype(jnp.int16))
    # jax.debug.print(" - Next state board: {}", next_state.game_state.board)
    reward, key = evaluate_state(mcts_params, next_state, step_fn, key)

    # If the leaf was terminal, then we use the actual reward instead of a rollout reward
    # leaf_reward = (leaf_state.rewards[mcts_params.player_idx] + 1) / 2
    leaf_reward = leaf_state.rewards[mcts_params.player_idx] * REWARD_SCALE
    reward = jax.lax.select(leaf_state.terminated | leaf_state.truncated, leaf_reward, reward)

    # jax.debug.callback(logging_callback, "leaf idx: {} -- leaf term. {} -- leaf reward: {:.3f} -- next term. {} -- next reward: {:.3f} -- reward: {}", node_idx, leaf_state.terminated, leaf_reward, next_state.terminated, next_state.rewards[mcts_params.player_idx], reward)

    # jax.debug.print(" - Leaf state was terminated: {}, truncated: {}, leaf reward: {}", leaf_state.terminated, leaf_state.truncated, leaf_reward)
    # jax.debug.print(" - Evaluated leaf state with reward {}", reward)

    expansion_node_idx = mcts_params.node_num + 1

    # The state needs to be updated using a tree map since it's a pytree (struct of arrays)
    updated_states = jax.tree_util.tree_map(lambda arr, new: arr.at[expansion_node_idx].set(new), mcts_params.states, next_state)

    num_sims = len(mcts_params.transitions)
    valid_node_idxs = jnp.where(traversal.valid, traversal.nodes, num_sims + 1)
    valid_actions = jnp.where(traversal.valid, traversal.actions, environment.num_actions + 1)

    # Increment the visit counts and rewards along the path (including the newly expanded node)
    updated_visits = mcts_params.visits.at[valid_node_idxs, valid_actions].add(1)
    updated_rewards = mcts_params.rewards.at[valid_node_idxs, valid_actions].add(reward)

    # Update 9/11/2025 -- this seems to be duplicating the last valid node idx
    # updated_visits = updated_visits.at[node_idx, action].add(1)
    # updated_rewards = updated_rewards.at[node_idx, action].add(reward)

    # jax.debug.print("Valid node idxs: {}, node idx: {}", valid_node_idxs, node_idx)

    mcts_params = mcts_params._replace(
        visits=updated_visits,
        rewards=updated_rewards,
    )

    # Only update the transitions if the leaf node is not terminal
    mcts_params = jax.lax.cond(
        ~(leaf_state.terminated | leaf_state.truncated),
        lambda params: params._replace(
            node_num=expansion_node_idx,
            states=updated_states,
            legal_actions=params.legal_actions.at[expansion_node_idx].set(next_state.legal_action_mask),
            transitions=params.transitions.at[node_idx, action].set(expansion_node_idx),
            to_play=params.to_play.at[expansion_node_idx].set(next_state.current_player),

            # Update 9/11/2025 -- update the visit and reward for the expansion node as well
            visits=params.visits.at[expansion_node_idx].set(1),
            rewards=params.rewards.at[expansion_node_idx].set(reward),
        ),
        lambda params: params,
        mcts_params
    )

    # jax.debug.callback(logging_callback, "expanding? {} -- exp idx: {} -- terminated at exp: {}", ~(leaf_state.terminated | leaf_state.truncated), expansion_node_idx, next_state.terminated)

    return mcts_params, key

def evaluate_state(mcts_params: MCTSParams, state: State, step_fn: callable, key: jax.random.PRNGKey) -> float:
    '''
    Evaluates the given state by running a random rollout to a terminal state and returning the final reward
    '''

    def cond_fn(args):
        state, _ = args
        return ~(state.terminated | state.truncated).all()

    def body_fn(args):
        state, key = args
        key, subkey = jax.random.split(key)
        logits = jnp.log(state.legal_action_mask.astype(jnp.float32))
        action = jax.random.categorical(key, logits=logits, axis=0).astype(jnp.int16)
        state = step_fn(state, action)
        return state, key

    final_state, key = jax.lax.while_loop(cond_fn, body_fn, (state, key))

    # Return the reward for the player to play at the root, r \in [-1, 0, 1]
    reward = final_state.rewards[mcts_params.player_idx]

    # Scale [-1, 1] to [0, 1]
    # reward = (reward + 1) / 2

    reward *= REWARD_SCALE

    return reward, key


def uct_mcts_policy(environment, max_depth=20, num_simulations=100):
    step_fn = jax.jit(environment.step)

    def policy_single(state, key):
        params, key = initialize(environment, state, num_simulations, max_depth, key)

        def body_fn(i, carry):
            params, key = carry
            key, subkey = jax.random.split(key)
            rollout, key = traverse_to_leaf(params, max_depth, key)
            params, key = expand_leaf(params, rollout, environment, step_fn, key)

            return params, key

        params, key = jax.lax.fori_loop(0, num_simulations, body_fn, (params, key))
        return jnp.argmax(params.visits[0])

    def policy_f(state_b, key):
        return jax.vmap(policy_single)(state_b, jax.random.split(key, state_b.rewards.shape[0]))

    return jax.jit(policy_f)
