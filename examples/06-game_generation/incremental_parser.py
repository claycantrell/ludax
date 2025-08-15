from importlib.resources import files

from lark import Lark, UnexpectedEOF, Token, UnexpectedToken

from ludax.games import tic_tac_toe


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

print(tic_tac_toe_partial)

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
    tokens = list(sorted(interactive.choices()))

    if not tokens:
        print("No more tokens to choose from. Exiting.")
        break

    # NOTE: for some reason, when I choose a lowercase token and skip LPAR, everything breaks
    tokens = [t for t in tokens if not t.islower()]

    # NOTE: sometimes choices still return invalid tokens. Super weired bug, probably in lexer itself. To reproduce
    # start from tick_tac_toe_partial. choices() will include POS_INT, and RPAR tokens but feed_token only accepts LPAR.
    safe_tokens = []
    for token in tokens:
        try:
            interactive2 = interactive.copy()
            interactive2.feed_token(Token(type=token, value=token.lower()))
            safe_tokens.append(token)
        except UnexpectedToken as e:
            print(f"Unexpected token encountered: {e.token} --- expected {e.expected}")

    tokens = safe_tokens


    print("Tokens:", [f"{i}: {t}" for i, t in enumerate(tokens)])

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
                if int(value) < 0:
                    raise ValueError("Value must be positive.")
                break
            except ValueError as e:
                print(f"Invalid input: {e}. Please try again.")
    elif token_type == "SIGNED_NUMBER":
        while True:
            try:
                value = input("Enter a signed int (e.g., -3): ")
                int(value)
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


# Print the current state of the parser
tree = interactive.resume_parse()
print(tree.pretty())