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
    # states = jnp.zeros((num_sims,) + state.shape, dtype=state.dtype)
    # states = states.at[0].set(state)

    visits = jnp.zeros((num_sims, environment.num_actions), dtype=jnp.int32)
    rewards = jnp.zeros((num_sims, environment.num_actions), dtype=jnp.float32)
    to_play = jnp.zeros((num_sims,), dtype=jnp.int32)
    
    params = MCTSParams(
        node_num=1,
        transitions=transitions,
        states=root_state,
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
    return int(max_action)

def proceed_to_leaf(mcts_params: MCTSParams, max_depth: int) -> MCTSRollout:
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
        valid = valid & is_valid_node(mcts_params, node_idx, action)

        # Advance to the next node and select the next action. We don't need to worry about
        # the behavior here if the current node is invalid, since later all that information
        # will get discarded
        next_node_idx = next_state(mcts_params, node_idx, action)
        next_action = select_action_ucb(mcts_params, next_node_idx)

        new_carry = (next_node_idx, next_action, valid)
        output = (node_idx, action, valid)

        return new_carry, output

    # Initialize the scan with the root node and an initial action
    node_idx, action = 0, select_action_ucb(mcts_params, 0)
    init_carry = (node_idx, action, True)
    _, (nodes, actions, valid) = jax.lax.scan(body_fn, init_carry, jnp.arange(max_depth))

    rollout = MCTSRollout(nodes=nodes, actions=actions, valid=valid)

    return rollout

if __name__ == "__main__":
    import time
    from ludax.games import tic_tac_toe

    key = jax.random.PRNGKey(0) 

    environment = LudaxEnvironment(game_str=tic_tac_toe)
    root_state = environment.init(key)
    num_sims = 1000
    max_depth = 100
    params, key = initialize(environment, root_state, num_sims, max_depth)

    print("Initialized MCTSParams!")
    
    action = select_action_ucb(params, 0)
    print(f"Selected action {action} at root node")