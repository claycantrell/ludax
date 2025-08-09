from functools import partial
import time
import jax
import jax.numpy as jnp
import numpy as np

from ludax import LudaxEnvironment

from ludax.games import hex, connect_four, reversi, tic_tac_toe, complexity_demo

from heuristics.hex import distance_heuristic, connectivity_heuristic
from heuristics.test import bad_heuristic, zero_heuristic

from mcts import mcts_policy, gumbel_policy
from simple import one_ply_policy, random_policy
from beam import beam_search_policy


jax.numpy.set_printoptions(threshold=np.inf, linewidth=np.inf)


def initialize(env: LudaxEnvironment, batch_size: int = 10 ** 3, seed: int = 0) -> tuple:
    """
    Initialize the game state for a batch of games.
    :param env: The environment to initialize.
    :param batch_size: Number of games to initialize.
    :param seed: Random seed for initialization.
    :return: (state_b, step_b, key)
    """
    init_b = jax.jit(jax.vmap(env.init))
    step_b = jax.jit(jax.vmap(env.step))
    key = jax.random.PRNGKey(seed)
    key, init_key = jax.random.split(key)
    state_b = init_b(jax.random.split(init_key, batch_size))
    return state_b, step_b, key


@partial(jax.jit, static_argnames=['policy_p1', 'policy_p2', 'step_b'])
def evaluate_policy(policy_p1, policy_p2, state_b, step_b, key) -> tuple:
    """
    Unfairly compare two different agents playing a two player game. The first agent will always be P1. Return the number of wins, draws, and losses for the first agent.
    :param policy_p1: a function that takes a state, a step function, and a key, then returns an action
    :param policy_p2: a function that takes a state, a step function, and a key, then returns an action
    :return: (wins, draws, losses) for the first agent, and the updated key
    """
    #jax.debug.print("\n\n" + "-" * 1000 + "\nevaluate_policy", ordered=True)
    def cond_fn(args):
        state, _ = args
        return ~(state.terminated.all() | state.truncated.all())

    def body_fn(args):
        state, key = args
        key, subkey = jax.random.split(key)

        # ToDo: Find a more efficient way. Right now, we're calling both policies every step.
        # Get the action from the policy of the current player
        action1 = policy_p1(state, subkey)
        action2 = policy_p2(state, subkey)
        action = jnp.where(state.game_state.current_player == 0, action1, action2)
        state = step_b(state, action)
        return state, key

    state_b, key = jax.lax.while_loop(cond_fn, body_fn, (state_b, key))

    # Count the results
    wins = jnp.sum(state_b.winner == 0)
    draws = jnp.sum(state_b.winner == -1)
    losses = jnp.sum(state_b.winner == 1)
    return (wins, draws, losses), key



def main():
    env = LudaxEnvironment(
        # game_str=complexity_demo,
        # game_str=hex,
        # game_str=connect_four,
        # game_str=reversi,
        game_str=tic_tac_toe,
    )

    # Initialize the environment and state
    state_b, step_b, key = initialize(env, batch_size=100, seed=42)

    # AGENT1 = random_policy()
    # AGENT1 = one_ply_policy(step_b)
    # AGENT1 = one_ply_policy(step_b)
    # AGENT1 = one_ply_policy(step_b, connectivity_heuristic)
    # AGENT1 = gumbel_policy(step_b, heuristic=distance_heuristic, num_simulations=200)
    AGENT1 = beam_search_policy(step_b, topk=9, iterations=10)

    AGENT2 = random_policy()
    # AGENT2 = mcts_policy(step_b, heuristic=distance_heuristic, num_simulations=10)
    # AGENT2 = mcts_policy(step_b, heuristic=distance_heuristic, num_simulations=10)
    # AGENT2 = gumbel_policy(step_b, heuristic=distance_heuristic, num_simulations=200)
    # AGENT2 = one_ply_policy(step_b)

    start_time = time.time()
    (w1, d1, l1), key = evaluate_policy(AGENT1, AGENT2, state_b, step_b, key)
    (w2, d2, l2), key = evaluate_policy(AGENT2, AGENT1, state_b, step_b, key)

    compile_time = time.time()

    print(f"Evaluating {AGENT1.__name__} vs {AGENT2.__name__} on {env.game_info}")
    print(f"Agent 1 - Rate:{(w1 + l2) / (w1 + l2 + l1 + w2) :.4f}, Wins: {w1}+{l2}, Draws: {d1}+{d2}")
    print(f"Agent 2 - Rate:{(l1 + w2) / (w1 + l2 + l1 + w2) :.4f}, Wins: {l1}+{w2}, Draws: {d1}+{d2}")
    end_time = time.time()
    print(f"Total time: {end_time - start_time:.2f}s (compile: {compile_time - start_time:.2f}s, run: {end_time - compile_time:.2f}s)")


if __name__ == "__main__":
    main()