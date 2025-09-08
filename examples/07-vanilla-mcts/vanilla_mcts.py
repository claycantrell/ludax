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
class MCTSParams(NamedTuple):
    '''
    The main data structure that stores the current state of the MCTS tree

    Params:
    - node_num (int): the current number of nodes in the tree / index of the next node to be added
    - transitions (array of shape [num_sims, num_actions]): for each node, the index of the child node for each action, or -1 if that action has not been taken yet
    - states (array of shape [num_sims, state_dim]): the game state at each node, used for expansion via the environment's step function
    - visits (array of shape [num_sims, num_actions]): the visit count N(s,a) for each node / action
    - rewards (array of shape [num_sims, num_actions]): the total reward W(s,a) for each node / action
    - to_play (array of shape [num_sims,]): the player to play at each node (used to flip the sign of rewards for the opponent)
    '''
    player_idx: int
    node_num: int
    transitions: jnp.ndarray
    states: jnp.ndarray
    visits: jnp.ndarray
    rewards: jnp.ndarray
    to_play: jnp.ndarray

class MCTSRollout(NamedTuple):
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

def initialize(environment: LudaxEnvironment, root_state: State, num_sims: int, max_depth: int, seed: int = 0) -> Tuple[MCTSParams, jnp.ndarray]:
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
    key = jax.random.PRNGKey(seed)

    transitions = -jnp.ones((num_sims, environment.num_actions), dtype=jnp.int32)
    states = jax.vmap(lambda i: root_state)(jnp.arange(num_sims))
    visits = jnp.zeros((num_sims, environment.num_actions), dtype=jnp.int32)
    rewards = jnp.zeros((num_sims, environment.num_actions), dtype=jnp.float32)
    to_play = jnp.zeros((num_sims,), dtype=jnp.int32)
    
    params = MCTSParams(
        player_idx=root_state.current_player,
        node_num=0,
        transitions=transitions,
        states=states,
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
    return mcts_params.transitions[node_idx, action] != -1

def select_action_ucb(mcts_params: MCTSParams, node_idx: int, c: float = 1.44) -> int:
    '''
    Selects an action at the given node using the UCB formula

    Params:
    - mcts_params (MCTSParams): the current MCTS parameters
    - node_idx (int): the index of the node to select an action from
    - c (float): the exploration constant

    Returns:
    - action (int): the selected action
    '''
    visits = mcts_params.visits[node_idx]
    rewards = mcts_params.rewards[node_idx]
    
    total_visits = jnp.sum(visits)
    
    # UCB formula
    ucb_values = jnp.where(
        visits > 0,
        rewards / visits + c * jnp.sqrt(jnp.log(total_visits + 1) / visits),
        jnp.inf
    )
    
    max_action = jnp.argmax(ucb_values)
    return max_action

def rollout_to_leaf(mcts_params: MCTSParams, max_depth: int) -> MCTSRollout:
    '''
    Proceeds from the root node to a leaf node by selecting actions using UCB

    Params:
    - mcts_params (MCTSParams): the current MCTS parameters
    - max_depth (int): the maximum depth of the tree (used to size the rollout arrays)

    Returns:
    - rollout (MCTSRollout): the rollout information
    '''

    def body_fn(carry, i):
        node_idx, action, valid = carry

        # If we've reached an unexpanded node, then all subsequent steps are invalid
        new_valid = valid & is_valid_node(mcts_params, node_idx, action)

        # Advance to the next node and select the next action. We don't need to worry about
        # the behavior here if the current node is invalid, since later all that information
        # will get discarded
        next_node_idx = next_state(mcts_params, node_idx, action)
        next_action = select_action_ucb(mcts_params, next_node_idx)

        new_carry = (next_node_idx, next_action, new_valid)
        output = (node_idx, action, valid)

        return new_carry, output

    # Initialize the scan with the root node and an initial action
    node_idx, action = 0, select_action_ucb(mcts_params, 0)
    init_carry = (node_idx, action, True)
    _, (nodes, actions, valid) = jax.lax.scan(body_fn, init_carry, jnp.arange(max_depth))

    rollout = MCTSRollout(nodes=nodes, actions=actions, valid=valid)

    return rollout

def expand_leaf(mcts_params: MCTSParams, rollout: MCTSRollout, environment: LudaxEnvironment, step_fn: callable) -> MCTSParams:
    '''
    Expands the leaf node reached by the given rollout

    Params:
    - mcts_params (MCTSParams): the current MCTS parameters
    - rollout (MCTSRollout): the rollout information
    - environment (LudaxEnvironment): the environment to use for expansion

    Returns:
    - mcts_params (MCTSParams): the updated MCTS parameters
    '''
    
    num_steps = len(rollout.nodes)
    last_valid_step = jnp.argmax(jnp.where(rollout.valid, jnp.arange(num_steps), 0))
    node_idx, action = rollout.nodes[last_valid_step], rollout.actions[last_valid_step]

    # print(f"Rollout to depth {last_valid_step}, expanding node {node_idx} with action {action}")

    leaf_state = jax.tree_util.tree_map(lambda x: x[node_idx], mcts_params.states)
    next_state = step_fn(leaf_state, action.astype(jnp.int16))
    reward = evaluate_state(mcts_params, next_state, step_fn, jax.random.PRNGKey(0))

    expansion_node_idx = mcts_params.node_num + 1

    # The state needs to be updated using a tree map since it's a pytree (struct of arrays)
    updated_states = jax.tree_util.tree_map(lambda arr, new: arr.at[expansion_node_idx].set(new), mcts_params.states, next_state)

    num_sims = len(mcts_params.transitions)
    valid_node_idxs = jnp.where(rollout.valid, rollout.nodes, num_sims + 1)
    valid_actions = jnp.where(rollout.valid, rollout.actions, environment.num_actions + 1)

    # Increment the visit counts and rewards along the path (including the newly expanded node)
    updated_visits = mcts_params.visits.at[valid_node_idxs, valid_actions].add(1)
    updated_rewards = mcts_params.rewards.at[valid_node_idxs, valid_actions].add(reward)

    updated_visits = updated_visits.at[node_idx, action].add(1)
    updated_rewards = updated_rewards.at[node_idx, action].add(reward)

    mcts_params = mcts_params._replace(
        node_num=expansion_node_idx,
        transitions=mcts_params.transitions.at[node_idx, action].set(expansion_node_idx),
        states=updated_states,
        visits=updated_visits,
        rewards=updated_rewards,
    )

    return mcts_params

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
        action = jax.random.categorical(key, logits=logits, axis=1).astype(jnp.int16)
        state = step_fn(state, action)
        return state, key

    state, key = jax.lax.while_loop(cond_fn, body_fn, (state, key))

    # Return the reward for the player to play at the root
    reward = state.rewards[mcts_params.player_idx]
    return reward

if __name__ == "__main__":
    import time
    from ludax.games import tic_tac_toe
    from ludax.config import BoardShapes

    def display_board(state, env):
        if env.game_info.board_shape != BoardShapes.HEXAGON:
            shaped_board = state.game_state.board.reshape(env.obs_shape[:2])
            for row in shaped_board:
                pretty_row = ' '.join(str(cell) for cell in row + 1)
                print(pretty_row.replace('0', '.').replace('1', 'X').replace('2', 'O'))
            print()
            if hasattr(state.game_state, "connected_components"):
                shaped_components = state.game_state.connected_components.reshape(env.obs_shape[:2])
                print(shaped_components)
        else:
            print(f"Observation shape: {env.obs_shape}")
            print(f"Board: {state.game_state.board}")
            if hasattr(state.game_state, "connected_components"):
                print(f"Components: {state.game_state.connected_components}")

    key = jax.random.PRNGKey(0) 

    environment = LudaxEnvironment(game_str=tic_tac_toe)
    step_fn = jax.jit(environment.step)

    num_sims = 1000
    max_depth = 15

    root_state = environment.init(key)
    while not root_state.terminated and not root_state.truncated:
        print("\nCurrent board:")
        display_board(root_state, environment)

        params, key = initialize(environment, root_state, num_sims, max_depth)
        print("Initialized MCTSParams!")

        def body_fn(i, params):
            rollout = rollout_to_leaf(params, max_depth)
            params = expand_leaf(params, rollout, environment, step_fn)

            return params
        
        print(f"Performing MCTS from the perspective of player {params.player_idx}...")
        params = jax.lax.fori_loop(0, 1000, body_fn, params)
        action = jnp.argmax(params.visits[0])

        print(f"Player {root_state.current_player} selecting action {action} with {params.visits[0, action]} visits")
        root_state = step_fn(root_state, action.astype(jnp.int16))