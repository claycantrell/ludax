"""TODO update to work with policy API changes"""

# import time
# import logging
# import os
#
# import jax
# import jax.numpy as jnp
#
# from ludax import LudaxEnvironment
# from ludax.games import tic_tac_toe
# from ludax.config import BoardShapes
#
#
# # Configure logging to write to a file
# os.remove("mcts_outputs.log") if os.path.exists("mcts_outputs.log") else None
# logging.basicConfig(
#     filename="mcts_outputs.log",
#     filemode="a",
#     format="%(asctime)s - %(levelname)s - %(message)s",
#     level=logging.INFO
# )
#
# def logging_callback(msg, *args):
#     if args:
#         msg = msg.format(*args)
#
#     logging.info(msg)
#
#
# tic_tac_big = '''(game "Tic-Tac-Toe"
#         (players 2)
#         (equipment
#             (board (square 7))
#         )
#
#         (rules
#             (play
#                 (repeat (P1 P2)
#                     (place (destination empty))
#                 )
#             )
#
#             (end
#                 (if (line 3) (mover win))
#                 (if (full_board) (draw))
#             )
#         )
#     )'''
#
#
# def display_board(state, env):
#     if env.game_info.board_shape != BoardShapes.HEXAGON:
#         shaped_board = state.game_state.board.reshape(env.obs_shape[:2])
#         for row in shaped_board:
#             pretty_row = ' '.join(str(cell) for cell in row + 1)
#             print(pretty_row.replace('0', '.').replace('1', 'X').replace('2', 'O'))
#         print()
#         if hasattr(state.game_state, "connected_components"):
#             shaped_components = state.game_state.connected_components.reshape(env.obs_shape[:2])
#             print(shaped_components)
#     else:
#         print(f"Observation shape: {env.obs_shape}")
#         print(f"Board: {state.game_state.board}")
#         if hasattr(state.game_state, "connected_components"):
#             print(f"Components: {state.game_state.connected_components}")
#
#
# seed = 0
#
# environment = LudaxEnvironment(game_str=tic_tac_big)
# step_fn = jax.jit(environment.step)
#
# num_sims = 10000
# max_depth = 25
#
# # Debugging!
# root_state = environment.init(jax.random.PRNGKey(seed))
# root_state = environment.step(root_state, 0)
# # root_state = environment.step(root_state, 36)
# # root_state = environment.step(root_state, 31)
# # root_state = environment.step(root_state, 20)
# # root_state = environment.step(root_state, 32)
#
# # print("Root state rewards:", root_state.rewards)
# # print("Root state eval: ", evaluate_state(MCTSParams(0,0,None,None,None,None,None,None), root_state, step_fn, jax.random.PRNGKey(seed))[0])
# # display_board(root_state, environment)
# # breakpoint()
#
# # root_state = environment.step(root_state, 6)
# # root_state = environment.step(root_state, 2)
# # root_state = environment.step(root_state, 7)
# # root_state = environment.step(root_state, 16)
# # root_state = environment.step(root_state, 9)
#
# params, key = initialize(environment, root_state, num_sims, max_depth, seed)
#
#
# def body_fn(i, carry):
#     params, key = carry
#     key, subkey = jax.random.split(key)
#
#     # jax.debug.print("\nIteration {}: rewards[0] = {}", i, params.rewards[0])
#     # jax.debug.print(" - visits[0] = {} ({})", params.visits[0], jnp.argmax(params.visits[0]))
#
#     rewards = params.rewards[0]
#     visits = params.visits[0]
#     best_action = jnp.argmax(params.visits[0])
#     prop_best_action = visits[best_action] / (jnp.sum(visits) + 1e-6)
#     avg_rewards = rewards / (visits + 1e-6)
#     jax.debug.callback(logging_callback,
#                        "[Iter {}] avg. reward = {:.3f} --  prop. best action = {:.3f} -- best action {}", i,
#                        jnp.mean(avg_rewards), prop_best_action, best_action)
#
#     rollout, key = traverse_to_leaf(params, max_depth, key)
#     params, key = expand_leaf(params, rollout, environment, step_fn, key)
#
#     return params, key
#
#
# params, _ = jax.lax.fori_loop(0, num_sims, body_fn, (params, key))
# display_board(root_state, environment)
#
# node_from_8 = params.transitions[0, 8]
# rewards_from_8 = params.rewards[node_from_8]
# visits_from_8 = params.visits[node_from_8]
#
# node_from_12 = params.transitions[0, 12]
# rewards_from_12 = params.rewards[node_from_12]
# visits_from_12 = params.visits[node_from_12]
#
# node_from_15 = params.transitions[0, 15]
# rewards_from_15 = params.rewards[node_from_15]
# visits_from_15 = params.visits[node_from_15]
#
# node_from_24 = params.transitions[0, 24]
# rewards_from_24 = params.rewards[node_from_24]
# visits_from_24 = params.visits[node_from_24]
#
# node_from_24_17 = params.transitions[node_from_24, 17]
# rewards_from_24_17 = params.rewards[node_from_24_17]
# visits_from_24_17 = params.visits[node_from_24_17]
#
# action = jnp.argmax(params.visits[0])
# print(f"Player {root_state.current_player} selecting action {action} with {params.visits[0, action]} visits")
# root_state = step_fn(root_state, action.astype(jnp.int16))
# display_board(root_state, environment)
#
# root_state = environment.init(jax.random.PRNGKey(seed))
# while not root_state.terminated and not root_state.truncated:
#     print("\nCurrent board:")
#     display_board(root_state, environment)
#
#     params, key = initialize(environment, root_state, num_sims, max_depth)
#     print("Initialized MCTSParams!")
#
#
#     def body_fn(i, carry):
#         params, key = carry
#         key, subkey = jax.random.split(key)
#         rollout, key = traverse_to_leaf(params, max_depth, key)
#         params, key = expand_leaf(params, rollout, environment, step_fn, key)
#
#         return params, key
#
#
#     print(f"Performing MCTS from the perspective of player {params.player_idx}...")
#     params, key = jax.lax.fori_loop(0, 10000, body_fn, (params, key))
#     action = jnp.argmax(params.visits[0])
#
#     print(f"Player {root_state.current_player} selecting action {action} with {params.visits[0, action]} visits")
#     root_state = step_fn(root_state, action.astype(jnp.int16))
#
# display_board(root_state, environment)