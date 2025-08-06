import math
import os
import time

import jax
import jax.numpy as jnp
from flask import render_template, request, jsonify
from markupsafe import Markup

from .. import environment
from ..config import BoardShapes, RENDER_CONFIG

from . import app
from .render import InteractiveBoardHandler

ENV, HANDLER, STATE = None, None, None

def cube_round(q, r, s):
    q_round, r_round, s_round = round(q), round(r), round(s)
    q_diff, r_diff, s_diff = abs(q_round - q), abs(r_round - r), abs(s_round - s)

    if q_diff > r_diff and q_diff > s_diff:
        q_round = -r_round - s_round

    elif r_diff > s_diff:
        r_round = -q_round - s_round

    else:
        s_round = -q_round - r_round

    return q_round, r_round, s_round


def display_board(state, env):

    # Display the board
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

def count(state):
    board = state.game_state.board  # ints, shape (...cells...)
    labels = state.game_state.connected_components  # same shape as board, int labels
    player = state.game_state.current_player

    # Keep labels only on the current player's stones; 0 elsewhere.
    # ToDo why !=
    flat = jnp.where(board != player, labels, 0).ravel().astype(jnp.int32)

    # Count presence of each label. Use a static length: #cells + 1 (for label 0).
    n_cells = flat.shape[0]  # static at trace time
    counts = jnp.bincount(flat, length=n_cells + 1)

    # Number of non‑empty component labels, excluding label 0.
    num_components = jnp.count_nonzero(counts[1:] > 0)
    return num_components

def debug_state(state: environment.State, env: environment.LudaxEnvironment):
    """Debug the current state of the game."""
    print(f"state.global_step_count: {state.global_step_count}")
    print(f"state.first_player: {state.first_player}")
    print(f"state.winner: {state.winner}")
    print(f"state.terminated: {state.terminated}")
    print(f"state.truncated: {state.truncated}")
    print(f"state.mover_reward: {state.mover_reward}")
    print(f"state.legal_action_mask: {state.legal_action_mask}")
    print(f"state.observation: \n{state.observation}")

    print(f"state.game_state: \n{state.game_state}")


    if hasattr(state.game_state, "connected_components"):
        print(f"Connected components: {jnp.unique(jnp.where(state.game_state.board != state.game_state.current_player, state.game_state.connected_components, 0)).size - 1}")
        print(count(state))
    #     print(f"Connected components: {state.game_state.connected_components}")

    # display_board(state, env)

@app.route('/') 
def index():
    game_names = [file_name.replace(".ldx", "") for file_name in os.listdir("./games") if file_name.endswith(".ldx")]
    return render_template('index.html', games=game_names)

@app.route('/game/<id>') 
def render_game(id):

    global ENV
    global HANDLER
    global STATE

    ENV = environment.LudaxEnvironment(f"games/{id}.ldx")
    HANDLER = InteractiveBoardHandler(ENV.game_info)

    STATE = ENV.init(jax.random.PRNGKey(42))
    
    HANDLER.render(STATE)
    time.sleep(0.1)

    return render_template('game.html', game_svg=Markup(HANDLER.rendered_svg))

@app.route('/step', methods=['POST'])
def step():
    global ENV
    global HANDLER
    global STATE
    if ENV is None:
        return "No game loaded"
    
    # Get x and y from the request
    data = request.get_json()
    x = float(data['x'])
    y = float(data['y'])

    action_idx = HANDLER.pixel_to_action((x, y))

    # Check if the action is legal
    legal_action_mask = STATE.legal_action_mask

    # Temporary workaround: if there is only one legal action, then
    # we always take it
    if legal_action_mask.sum() == 1:
        action_idx = int(legal_action_mask.argmax())
        print(f"Only one legal action available, taking action {action_idx}!")
    else:
        # Check if the selected action is legal
        if action_idx >= len(legal_action_mask) or not legal_action_mask[action_idx]:
            print(f"Illegal action selected: {action_idx}")
            # Return current state with an error message
            HANDLER.render(STATE)
            time.sleep(0.1)

            if hasattr(STATE.game_state, "scores"):
                scores = list(map(float, STATE.game_state.scores))
            else:
                scores = [0.0, 0.0]

            return jsonify({
                "svg": HANDLER.rendered_svg,
                "terminated": bool(STATE.terminated),
                "winner": int(STATE.winner),
                "current_player": int(STATE.game_state.current_player),
                "scores": scores,
                "error": "Illegal move! Please select a valid action."
            })

        action = HANDLER.action_indices[action_idx]

    STATE = ENV.step(STATE, action_idx)

    HANDLER.render(STATE)
    time.sleep(0.1)

    terminated = bool(STATE.terminated)
    winner = int(STATE.winner)
    if hasattr(STATE.game_state, "scores"):
        scores = list(map(float, STATE.game_state.scores))
    else:
        scores = [0.0, 0.0]

    print("\n" + "-" * 40)
    print(f"Current player: {STATE.game_state.current_player}")
    print(f"Scores: {scores}\n")

    debug_state(STATE, ENV)

    return {"svg": HANDLER.rendered_svg, "terminated": terminated, "winner": winner, "current_player": int(STATE.game_state.current_player),
            "scores": scores}

@app.route('/reset', methods=['POST'])
def reset():
    global ENV
    global HANDLER
    global STATE
    if ENV is None:
        return "No game loaded"
    
    STATE = ENV.init(jax.random.PRNGKey(42))
    HANDLER.render(STATE)
    time.sleep(0.1)

    svg_data = ""

    return {"svg": svg_data}