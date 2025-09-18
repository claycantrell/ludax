
import jax
import jax.numpy as jnp
from ludax import LudaxEnvironment
from ludax.config import BoardShapes
from ludax.games import connect_four

from vanilla_mcts import initialize, traverse_to_leaf, expand_leaf

tic_tac_big = '''
(game "Tic-Tac-Toe" 
    (players 2)
    (equipment 
        (board (square 7))
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
        )
    )
)'''

letters = 'abcdefghijklmnopqrstuvwxyz'
numbers = '123456789'

def display_board(state, env):
    shaped_board = state.game_state.board.reshape(env.obs_shape[:2])
    for i, row in enumerate(shaped_board):
        pretty_row = ' '.join(str(cell) for cell in row + 1)
        print(f"{letters[i]}: {pretty_row.replace('0', '.').replace('1', 'X').replace('2', 'O')}")
    print("   " + ' '.join(numbers[:shaped_board.shape[1]]))
    print()


environment = LudaxEnvironment(game_str=tic_tac_big)
step_fn = jax.jit(environment.step)

NUM_SIMS = 100000
MAX_DEPTH = 25
SEED = 0

root_state = environment.init(jax.random.PRNGKey(SEED))
# root_state = step_fn(root_state, 0)
turn_idx = 0
while not root_state.terminated and not root_state.truncated:
    print("\nCurrent board:")
    display_board(root_state, environment)
    
    if turn_idx % 2 == 0:
        action_str = input(f"Player {root_state.current_player}, enter your move (letter-number): ")
        num_cols = environment.obs_shape[1]
        action = (letters.index(action_str[0].lower()) * num_cols + (int(action_str[1:]) - 1))
        if action < 0 or action >= environment.num_actions or root_state.legal_action_mask[action] == 0:
            raise ValueError(f"Invalid action: {action_str}")
        root_state = step_fn(root_state, action)
        turn_idx += 1
        continue

    params, key = initialize(environment, root_state, NUM_SIMS, MAX_DEPTH, SEED)
    print("Initialized MCTSParams!")

    def body_fn(i, carry):
        params, key = carry
        key, subkey = jax.random.split(key)
        rollout, key = traverse_to_leaf(params, MAX_DEPTH, key)
        params, key = expand_leaf(params, rollout, environment, step_fn, key)

        return params, key

    print(f"Performing MCTS from the perspective of player {params.player_idx}...")
    params, key = jax.lax.fori_loop(0, NUM_SIMS, body_fn, (params, key))
    action = jnp.argmax(params.visits[0])

    print(f"Player {root_state.current_player} selecting action {action} with {params.visits[0, action]} visits")
    root_state = step_fn(root_state, action.astype(jnp.int16))

    turn_idx += 1

display_board(root_state, environment)