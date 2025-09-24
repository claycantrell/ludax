import time
from functools import partial

import jax
import jax.numpy as jnp

from ludax import LudaxEnvironment
from ludax.games import *
from ludax.policies import simple_mctx_policy, lookahead_mctx_policy, uct_mcts_policy, random_policy


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


@partial(jax.jit, static_argnames=['policy', 'step_b'])
def gavel_metrics(policy, state_b, step_b, key, truncate=200) -> tuple:
    """
    :param policy: a function that takes a state, a step function, and a key, then returns an action
    :return: (
        1. *balance*: the largest difference in winrates between any pair of players,
        2. *decisiveness*: the proportion of games that do not end in a draw,
        3. *completion*: the proportion of games that reach an end state,
        4. *agency*: the proportion of turns for which the player to move has more than one legal move,
        5. *coverage*: the proportion of board sites (e.g. squares on a chessboard) that get occupied by a game piece at least once in a playout
        6. Raw game statistics: (wins, draws, losses, truncated, total_games)
        )
    """

    B = state_b.winners.shape[0]
    occ0 = state_b.game_state.board != -1
    agency_num0 = jnp.zeros((B,), jnp.int32)  # turns with >1 legal move
    agency_den0 = jnp.zeros((B,), jnp.int32)  # total turns taken (per game)

    carry0 = (state_b, key, 0, agency_num0, agency_den0, occ0)

    def cond_fn(carry):
        state, _, t, *rest = carry
        return jnp.logical_and(jnp.any(~state.terminated), t < truncate)

    def body_fn(carry):
        state, key, t, a_num, a_den, covered = carry
        key, subkey = jax.random.split(key)
        t = t + 1

        active = ~state.terminated  # [B]
        legal_n = jnp.sum(state.legal_action_mask, axis=-1)
        occ_now = state.game_state.board != -1

        a_num = a_num + (active & (legal_n > 1)).astype(jnp.int32)
        a_den = a_den + active.astype(jnp.int32)
        covered = jnp.where(active[:, None], covered | occ_now, covered)

        action = policy(state, subkey)
        state = step_b(state, action)

        return (state, key, t, a_num, a_den, covered)

    state_b, key, t, a_num, a_den, covered = jax.lax.while_loop(cond_fn, body_fn, carry0)

    # Existing metrics
    wins = jnp.sum(state_b.winners == 1, axis=0)[0]
    losses = jnp.sum(state_b.winners == 0, axis=0)[0]
    draws = jnp.sum(state_b.winners == -1, axis=0)[0]
    truncated = jnp.sum(~state_b.terminated)
    total_games = state_b.winners.shape[0]
    jax.debug.print("wins: {wins}, draws: {draws}, losses: {losses}, truncated: {truncated}, total_games: {total_games}",
                    wins=wins, draws=draws, losses=losses, truncated=truncated, total_games=total_games)
    balance = 1 - (jnp.abs(wins - losses) / total_games)
    decisiveness = (total_games - draws) / total_games
    completion = (total_games - truncated) / total_games

    # New metrics (per spec)
    # agency: mean over games of (fraction of that game's turns with >1 legal move)
    agency_per_game = a_num / jnp.maximum(a_den, 1)
    agency = jnp.mean(agency_per_game)

    # coverage: mean over games of (fraction of sites ever occupied)
    coverage_per_game = jnp.mean(covered, axis=1)  # covered is [B, S] bool
    coverage = jnp.mean(coverage_per_game)

    return (balance, decisiveness, completion, agency, coverage, (wins, draws, losses, truncated, total_games)), key


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
        key, k1, k2 = jax.random.split(key, 3)

        # ToDo: Find a more efficient way. Right now, we're calling both policies every step.
        # Get the action from the policy of the current player
        action1 = policy_p1(state, k1)
        action2 = policy_p2(state, k2)
        action = jnp.where(state.game_state.current_player == 0, action1, action2)
        state = step_b(state, action)
        return state, key

    state_b, key = jax.lax.while_loop(cond_fn, body_fn, (state_b, key))

    # Count the results
    wins = jnp.sum(state_b.winners == 1, axis=0)[0]
    losses = jnp.sum(state_b.winners == 0, axis=0)[0]
    draws = jnp.sum(state_b.winners == -1, axis=0)[0]
    return (wins, draws, losses), key


def evaluate_game(game_str):
    """Evaluate the game string to check if it is valid."""
    try:
        env = LudaxEnvironment(game_str=game_str)
        state_b, step_b, key = initialize(env, batch_size=50, seed=42)
    except Exception as e:
        return -3, None

    # r_policy = random_policy()
    # g_policy = simple_mctx_policy(step_b, num_simulations=100)
    # g_policy = lookahead_mctx_policy(step_b, num_simulations=10)
    g_policy = uct_mcts_policy(env, num_simulations=100, max_depth=25)
    g2_policy = uct_mcts_policy(env, num_simulations=50, max_depth=25)

    # try:
    #     (r_balance, _, _, r_agency, _, (wins, draws, losses, truncated, total)), key = gavel_metrics(r_policy, state_b, step_b, key)
    # except Exception as e:
    #     print("Error during first evaluation:", e)
    #     return -2, None
    #
    # if r_balance < 0.5 or r_agency < 0.5 or wins + losses + draws + truncated < total:
    #     return -1, None
    #
    # print("Passed random policy sanity check.")

    try:
        # ToDO why is P1 always winning instead of drawing?
        (balance, decisiveness, completion, agency, coverage, _), key = gavel_metrics(g_policy, state_b, step_b, key)

        # Evaluate strategic depth
        (w1, d1, l1), key = evaluate_policy(g_policy, g2_policy, state_b, step_b, key)
        (w2, d2, l2), key = evaluate_policy(g2_policy, g_policy, state_b, step_b, key)

        strategic_depth = (w1 + l2) / (w1 + l1 + w2 + l2) if (w1 + l1 + w2 + l2) > 0 else 0

        print(f"Balance: {balance}, Decisiveness: {decisiveness}, Completion: {completion}, "
              f"Agency: {agency}, Coverage: {coverage}, Strategic Depth: {strategic_depth}")

        # Harmonic mean
        if balance == 0 or decisiveness == 0 or completion == 0 or agency == 0 or coverage == 0 or strategic_depth == 0:
            return 0, (balance, decisiveness, completion, agency, coverage, strategic_depth)

        mean = 6 / (
            1 / balance + 1 / decisiveness + 1 / completion + 1 / agency + 1 / coverage + 1 / strategic_depth
        )

        return mean, (balance, decisiveness, completion, agency, coverage, strategic_depth)

    except Exception as e:
        print("Unexpected error during second evaluation:", e)
        return -2, None


if __name__ == "__main__":

    # games = [
    #     '( game "RandomGame" ( players 2 ) ( equipment ( board ( rectangle 10 5 ) ) ) ( rules ( start ( place P1 ( 11 ) ) ( place P2 ( 2 ) ) ( place P2 ( 2 ) ) ( place P1 ( 11 8 ) ) ( place P1 ( 5 10 2 11 1 ) ) ( place P1 ( 3 ) ) ) ( play ( once_through ( P1 P1 ) ( place mover ( destination empty ) ) ( force_pass ) ) ) ( end ( if 0 ( draw ) ) ( if 10 ( draw ) ) ( if 6 ( draw ) ) ( if 1 ( opponent lose ) ) ) ) ( rendering ( color P2 black ) ) )',
    #     '( game "RandomGame" ( players 2 ) ( equipment ( board ( square 6 ) ) ) ( rules ( start ( place P1 ( 9 ) ) ( place P1 ( 1 ) ) ( place P2 ( 3 ) ) ) ( play ( repeat ( P2 ) ( place mover ( destination corners ) ( effects ( capture corners ) ) ) ) ) ( end ( if 2 ( draw ) ) ( if 8 ( draw ) ) ( if 2 ( draw ) ) ) ) ( rendering ( color P1 black ) ( color P1 black ) ( color P1 white ) ( color P1 white ) ) )',
    #     '( game "RandomGame" ( players 2 ) ( equipment ( board ( hexagon 11 ) ) ) ( rules ( start ( place P1 ( 2 11 ) ) ) ( play ( repeat ( P2 P1 ) ( place ( destination empty ) ) ( force_pass ) ) ) ( end ( if 3 ( draw ) ) ) ) ( rendering ( color P2 black ) ) )',
    #     '( game "RandomGame" ( players 2 ) ( equipment ( board ( hex_rectangle 11 10 ) ) ) ( rules ( play ( repeat ( P2 P1 P2 ) ( place ( destination empty ) ( effects ( flip empty ) ) ) ( force_pass ) ) ) ( end ( if 8 ( draw ) ) ( if ( not ( not 9 ) ) ( mover win ) ) ) ) )'
    #     '( game "RandomGame" ( players 2 ) ( equipment ( board ( square 9 ) ) ) ( rules ( start ( place P1 ( 4 4 ) ) ) ( play ( once_through ( P1 P1 P1 P2 ) ( place mover ( destination empty ) ( result 7 ) ) ) ) ( end ( if 7 ( draw ) ) ) ) ( rendering ( color P2 black ) ( color P1 white ) ( color P2 black ) ( color P2 black ) ) )'
    # ]
    #
    # for game in games:
    #     print("\n\n" + "=" * 100)
    #     print("Evaluating game:\n", game)
    #     score, breakdown = evaluate_game(game)
    #     print(f"Final score: {score}, breakdown: {breakdown}")

    start_time = time.time()
    print(evaluate_game(hex))

    print(f"Evaluation took {time.time() - start_time} seconds.")