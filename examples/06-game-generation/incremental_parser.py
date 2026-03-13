from importlib.resources import files

import jax
import jax.numpy as jnp
from lark import Lark, UnexpectedEOF, Token, UnexpectedToken, Tree

from ludax import LudaxEnvironment
from ludax.config import ACTION_DTYPE, REWARD_DTYPE
from ludax.games import tic_tac_toe


def print_partial_tree(value_stack, indent=0):
    """Recursively print Tree/Token objects in value_stack."""
    for item in value_stack:
        if isinstance(item, Tree):
            print("  " * indent + f"Tree({item.data!r})")
            print_partial_tree(item.children, indent + 1)
        elif isinstance(item, Token):
            print("  " * indent + f"Token({item.type!r}, {item.value!r})")
        else:
            # Might be an intermediate value from a transformer
            print("  " * indent + repr(item))

def tree_to_source(obj):
    """Recursively rebuild the original source text from a Lark Tree or Token."""
    if isinstance(obj, Tree):
        return " ".join(tree_to_source(c) for c in obj.children)
    elif isinstance(obj, Token):
        return obj.value
    else:
        return str(obj)


if __name__ == "__main__":
    tic_tac_toe_partial = """
    (game "Tic-Tac-Toe" 
        (players 2)
        (equipment 
            (board (square 3))
        ) 

        (rules 
            (play
                (repeat (P1 P2)
                    (place (destination empty))
                )
            )

            (end 
                (if (line 3) 
    """

    tic_tac_toe_complete = """
    (game "Tic-Tac-Toe" 
        (players 2)
        (equipment 
            (board (square 3))
        ) 

        (rules 
            (play
                (repeat (P1 P2)
                    (place (destination empty))
                )
            )

            (end 
                (if (line 3) (mover win))
                (if (full_board) (draw)) 
    """

    # Note: Had to redefine the grammar to name all literals which eliminates __ANON_* tokens
    # with files("ludax").joinpath('grammar.lark').open('r') as f:
    #     grammar = f.read()

    with open("./grammar2.lark", "r") as f:
        grammar = f.read()

    parser = Lark(
        grammar,
        start="game",            # same start rule as before
        parser="lalr",           # use LALR instead of Earley
        maybe_placeholders=False
    )

    # interactive = parser.parse_interactive('(game "demo" ')
    interactive = parser.parse_interactive(tic_tac_toe_partial)
    interactive.exhaust_lexer()


    while True:
        tokens = list(sorted(interactive.accepts()))

        if not tokens:
            print("No more tokens to choose from. Exiting.")
            break

        print("Tokens:", [f"{i}: {t}" for i, t in enumerate(tokens)])
        # print("Tokens from .choices():", list(sorted(interactive.choices())))
        # print("Tokens from .accepts():", list(interactive.accepts()))
        # breakpoint()

        token_idx = 0
        if len(tokens) > 1:
            token_idx = input("Enter choice: ")

            try:
                token_idx = int(token_idx)
                if token_idx < 0 or token_idx >= len(tokens):
                    print("Invalid index, please try again.")
                    continue
            except ValueError:
                print("Invalid input, please enter a number.")
                continue

        token_type = tokens[token_idx]
        value = token_type.lower()
        if token_type == "POS_INT":
            while True:
                try:
                    value = input("Enter a positive integer: ")
                    if int(value) <= 0:
                        raise ValueError("Value must be positive.")
                    break
                except ValueError as e:
                    print(f"Invalid input: {e}. Please try again.")
        elif token_type == "NONNEG_INT":
            while True:
                try:
                    value = input("Enter a positive integer: ")
                    if int(value) < 0:
                        raise ValueError("Value must be nonnegative.")
                    break
                except ValueError as e:
                    print(f"Invalid input: {e}. Please try again.")
        elif token_type == "LPAR":
            value = "("
        elif token_type == "RPAR":
            value = ")"

        # Note: for some reason if we add $END token, it will crash when we resume_parse()
        if tokens[token_idx] == "$END":
            print("End of input reached. Exiting.")
            break


        token = Token(type=tokens[token_idx], value=value)
        # print(token)
        interactive.feed_token(token)
        interactive.exhaust_lexer()

        print_partial_tree(interactive.parser_state.value_stack)

    # Print the current state of the parser
    tree = interactive.resume_parse()
    game_str = tree_to_source(tree)

    print("\n\nFinal game string:")
    print(game_str)


    env = LudaxEnvironment(game_str=game_str)
    init = jax.jit(jax.vmap(env.init))
    step = jax.jit(jax.vmap(env.step))


    def _run_batch(state, key):
        def cond_fn(args):
            state, _ = args
            return ~(state.terminated | state.truncated).all()

        def body_fn(args):
            state, key = args
            key, subkey = jax.random.split(key)
            logits = jnp.log(state.legal_action_mask.astype(REWARD_DTYPE))
            action = jax.random.categorical(key, logits=logits, axis=1).astype(ACTION_DTYPE)
            state = step(state, action)
            return state, key

        state, key = jax.lax.while_loop(cond_fn, body_fn, (state, key))

        return state, key


    run_batch = jax.jit(_run_batch)

    key = jax.random.PRNGKey(42)
    key, subkey = jax.random.split(key)
    keys = jax.random.split(subkey, 100)

    state = init(keys)
    state, key = run_batch(state, key)
    print(f"Winner (0: first player, 1: second player, -1: draw): {state.winners}")