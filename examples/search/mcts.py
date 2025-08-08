from functools import partial
import time
import jax
import jax.numpy as jnp
import numpy as np

from ludax import LudaxEnvironment

from heuristics.hex import distance_heuristic, connectivity_heuristic
from heuristics.test import bad_heuristic, zero_heuristic
import mctx

jax.numpy.set_printoptions(threshold=np.inf, linewidth=np.inf)

BIG = 10**6
SMALL = 10**-3

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


def beam_search_policy(step_b, heuristic=zero_heuristic, topk=200, iterations=10):

    def beam_search_f(state_b, key):
        """Expand the top-k actions at each iteration. Rank the best action based on the heuristic."""


    def beam_step_f(top_states_b, key, ):
        """
        Perform one step of the beam search.
        :param top_states_b: Current top-k descendant states for each root_state in the batch.
        :param key: JAX PRNG key for random number generation.
        :return: New top-k descendant states after one step.
        """


    return jax.jit(beam_search_f)



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
        )

        return policy_output.action.astype(jnp.int16)

    return jax.jit(policy_f)


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
    # GAME_PATH = "games/tic_tac_toe.ldx"
    GAME_PATH = "games/hex.ldx"
    # GAME_PATH = "games/connect_four.ldx"
    # GAME_PATH = "games/reversi.ldx"
    # GAME_PATH = "games/complexity_demo.ldx"

    env = LudaxEnvironment(GAME_PATH)

    # Initialize the environment and state
    state_b, step_b, key = initialize(env, batch_size=200, seed=42)

    # AGENT1 = random_policy()
    # AGENT1 = one_ply_policy(step_b)
    AGENT1 = one_ply_policy(step_b)
    # AGENT1 = one_ply_policy(step_b, connectivity_heuristic)
    # AGENT1 = gumbel_policy(step_b, heuristic=distance_heuristic, num_simulations=200)

    # AGENT2 = random_policy()
    # AGENT2 = mcts_policy(step_b, heuristic=distance_heuristic, num_simulations=10)
    # AGENT2 = mcts_policy(step_b, heuristic=distance_heuristic, num_simulations=10)
    # AGENT2 = gumbel_policy(step_b, heuristic=distance_heuristic, num_simulations=200)
    AGENT2 = one_ply_policy(step_b, bad_heuristic)

    start_time = time.time()
    (w1, d1, l1), key = evaluate_policy(AGENT1, AGENT2, state_b, step_b, key)
    (w2, d2, l2), key = evaluate_policy(AGENT2, AGENT1, state_b, step_b, key)

    compile_time = time.time()

    print(f"Evaluating {GAME_PATH}:")
    print(f"Agent 1 - Rate:{(w1 + l2) / (w1 + l2 + l1 + w2) :.4f}, Wins: {w1}+{l2}, Draws: {d1}+{d2}")
    print(f"Agent 2 - Rate:{(l1 + w2) / (w1 + l2 + l1 + w2) :.4f}, Wins: {l1}+{w2}, Draws: {d1}+{d2}")
    end_time = time.time()
    print(f"Total time: {end_time - start_time:.2f}s (compile: {compile_time - start_time:.2f}s, run: {end_time - compile_time:.2f}s)")


if __name__ == "__main__":
    main()
