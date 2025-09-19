from functools import partial

import jax
import jax.numpy as jnp
from jax import disable_jit
from lark import Lark, Token

from ludax import LudaxEnvironment
from ludax.games import tic_tac_toe
from ludax.policies import random_policy, gumbel_policy

from gavel_eval import gavel_metrics, initialize

from incremental_parser import tree_to_source, print_partial_tree

import random


with open("./grammar2.lark", "r") as f:
    grammar = f.read()


def generate_random_game(max_depth=5, seed=None):
    """Generate a random game description."""
    parser = Lark(
        grammar,
        start="game",  # same start rule as before
        parser="lalr",  # use LALR instead of Earley
        maybe_placeholders=False
    )

    partial_description = '(game "RandomGame" (players 2)'

    interactive = parser.parse_interactive(partial_description)
    interactive.exhaust_lexer()

    if seed is None:
        seed = random.randint(0, 2**32 - 1)

    random.seed(seed)

    depth = 0
    while True:
        tokens = list(sorted(interactive.accepts()))
        if not tokens:
            break

        if depth >= max_depth and len(tokens) > 1:
            tokens = [token for token in tokens if token != "LPAR"]

        # Try removing sources of crashes
        tokens = [token for token in tokens if "score" not in token.lower()]

        # if len(tokens) == 0:
        #     break

        token = random.choice(tokens)
        value = token.lower()

        if token == "$END":
            break
        elif token == "POS_INT":
            value = str(random.randint(1, 11))
        elif token == "NONNEG_INT":
            value = str(random.randint(0, 10))
        elif token == "LPAR":
            value = "("
            depth += 1
        elif token == "RPAR":
            value = ")"
            depth -= 1
        elif token == "P1" or token == "P2":
            value = token

        interactive.feed_token(Token(type=token, value=value))

        # print("\n\n\n")
        # print_partial_tree(interactive.parser_state.value_stack)

    tree = interactive.resume_parse()
    game_str = tree_to_source(tree)

    return game_str, seed


def simple_eval(game_str):
    """Evaluate the game string to check if it is valid."""
    try:
        env = LudaxEnvironment(game_str=game_str)
        state_b, step_b, key = initialize(env, batch_size=100, seed=42)
    except Exception as e:
        print(f"Error initializing game: {e}")
        return "Uninitializable"

    r_policy = random_policy()
    # g_policy = gumbel_policy(step_b, num_simulations=100)

    try:
        (r_balance, r_decisiveness, _, r_agency, _, (wins, draws, losses, truncated, total)), key = gavel_metrics(r_policy, state_b, step_b, key)
    except Exception as e:
        print(f"Error during first evaluation: {e}")
        return "Unplayable"

    print(r_balance, r_decisiveness, r_agency, wins, draws, losses, truncated, total)

    if wins + losses + draws + truncated < total:
        return "Contradictory"

    if r_balance < 0.2 or r_agency < 0.2 or r_decisiveness < 0.2:
        return "Uninteresting"

    return "Playable"

if __name__ == "__main__":
    print(simple_eval(tic_tac_toe))

    total = 100
    fitness = []
    for _ in range(total):
        # try:
            game_str, seed = generate_random_game(max_depth=3)
            print("\n\n", game_str)
            fitness.append(simple_eval(game_str))
            print(fitness[-1])
        # except Exception as e:
        #     print(f"Error generating game: {e}")
        #
        #     fitness.append(-4)

    print(f"Generated {total} games with the following fitness scores:")
    print(f"Uninitializable: {fitness.count('Uninitializable')}")
    print(f"Unplayable: {fitness.count('Unplayable')}")
    print(f"Contradictory: {fitness.count('Contradictory')}")
    print(f"Uninteresting: {fitness.count('Uninteresting')}")
    print(f"Playable: {fitness.count('Playable')}")












