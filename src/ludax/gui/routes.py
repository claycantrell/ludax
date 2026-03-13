import math
import time

import jax
import jax.numpy as jnp
from flask import render_template, request, jsonify
from markupsafe import Markup
import numpy as np

from .. import environment, utils
from ..config import ActionTypes, Shapes, RENDER_CONFIG

from . import app
from .render import InteractiveBoardHandler

ENV, HANDLER, STATE = None, None, None
SLIDE_LOOKUP = None
MOVE_INFO = {
    "stage": "selecting_piece",
    "select_idx": None
}

from ludax import games

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
    if env.game_info.board_shape != Shapes.HEXAGON:
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
    print(f"state.winners: {state.winners}")
    print(f"state.terminated: {state.terminated}")
    print(f"state.truncated: {state.truncated}")
    print(f"state.mover_reward: {state.mover_reward}")
    print(f"state.legal_action_mask: {state.legal_action_mask}")

    print(f"state.game_state: \n{state.game_state}")


    if hasattr(state.game_state, "connected_components"):
        print(f"Connected components: {jnp.unique(jnp.where(state.game_state.board != state.game_state.current_player, state.game_state.connected_components, 0)).size - 1}")
        print(count(state))
    #     print(f"Connected components: {state.game_state.connected_components}")

    # display_board(state, env)

@app.route('/') 
def index():
    return render_template('index.html', games=games.__all__)

@app.route('/game/<id>') 
def render_game(id):

    global ENV
    global HANDLER
    global STATE
    global SLIDE_LOOKUP
    global MOVE_INFO

    print(f"Loading the following game:\n{getattr(games, id)}")
    ENV = environment.LudaxEnvironment(game_str=getattr(games, id))
    HANDLER = InteractiveBoardHandler(ENV.game_info, ENV.rendering_info)
    MOVE_INFO = {
        "stage": "selecting_piece",
        "select_idx": None
    }
    if ENV.game_info.uses_slide_logic:
        SLIDE_LOOKUP = utils._get_slide_lookup(ENV.game_info)

    # NOTE: there's currently a rendering bug where "rendering info" is not properly cleared when switching games
    # in the brower.
    print(f"Rendering info: {ENV.rendering_info.color_mapping}, {ENV.rendering_info.piece_shape_mapping}")

    STATE = ENV.init(jax.random.PRNGKey(42))

    # Placement games (board_size,)
    if HANDLER.game_info.action_type == ActionTypes.TO:
        HANDLER.render(STATE)

    # Step / hop games (board_size, num_directions)
    elif HANDLER.game_info.action_type == ActionTypes.FROM_DIR:
        legal_selections = STATE.legal_action_mask.reshape((ENV.board_size, HANDLER.game_info.num_directions)).any(axis=1)
        HANDLER.render(STATE, legal_actions=legal_selections)

    # Sliding / moving games (board_size, board_size)
    elif HANDLER.game_info.action_type == ActionTypes.FROM_TO:
        legal_selections = STATE.legal_action_mask.reshape((ENV.board_size, ENV.board_size)).any(axis=1)
        HANDLER.render(STATE, legal_actions=legal_selections)

    else:
        raise ValueError(f"Unknown action type: {HANDLER.game_info.action_type}")
        
    time.sleep(0.1)

    # Generate region legend
    region_legend = HANDLER.render_legend()

    return render_template('game.html', game_svg=Markup(HANDLER.rendered_svg), region_legend=Markup(region_legend))

@app.route('/step', methods=['POST'])
def step():
    global ENV
    global HANDLER
    global STATE
    global SLIDE_LOOKUP
    global MOVE_INFO

    if ENV is None:
        return "No game loaded"
    
    # Get x and y from the request
    data = request.get_json()
    x = float(data['x'])
    y = float(data['y'])

    action_idx = HANDLER.pixel_to_action((x, y))

    # Placement games (board_size,)
    if HANDLER.game_info.action_type == ActionTypes.TO:
        legal_action_mask = STATE.legal_action_mask

    # Step / hop games (board_size, num_directions)
    elif HANDLER.game_info.action_type == ActionTypes.FROM_DIR:
        if MOVE_INFO['stage'] == "selecting_piece":
            legal_action_mask = STATE.legal_action_mask.reshape((ENV.board_size, HANDLER.game_info.num_directions)).any(axis=1)

        # When selecting a destination, we need to convert from direction indices into actual board positions
        elif MOVE_INFO['stage'] == "selecting_destination":
            direction_mask = STATE.legal_action_mask.reshape((ENV.board_size, HANDLER.game_info.num_directions))[MOVE_INFO["select_idx"]]
            valid_directions = jnp.argwhere(direction_mask).flatten()
            valid_ends = SLIDE_LOOKUP[valid_directions, MOVE_INFO["select_idx"],  1] # TODO: handle distances > 1
            legal_action_mask = jnp.zeros(ENV.board_size)
            legal_action_mask = legal_action_mask.at[valid_ends].set(1)

    # Sliding / moving games (board_size, board_size)
    elif HANDLER.game_info.action_type == ActionTypes.FROM_TO:
        if MOVE_INFO['stage'] == "selecting_piece":
            legal_action_mask = STATE.legal_action_mask.reshape((ENV.board_size, ENV.board_size)).any(axis=1)
        elif MOVE_INFO['stage'] == "selecting_destination":
            legal_action_mask = STATE.legal_action_mask.reshape((ENV.board_size, ENV.board_size))[MOVE_INFO["select_idx"]]

    else:
        raise ValueError(f"Unknown action type: {HANDLER.game_info.action_type}")

    # Temporary workaround: if there is only one legal action, then
    # we always take it
    if legal_action_mask.sum() == 1:
        action_idx = int(legal_action_mask.argmax())
        print(f"Only one legal action available, taking action {action_idx}!")
    else:
        # Check if the selected action is legal
        if action_idx >= len(legal_action_mask) or not legal_action_mask[action_idx]:
            print(f"Illegal action selected: {action_idx}")
            print("Legal action mask:\n", legal_action_mask)
            print("Legal action indices:\n", jnp.where(legal_action_mask)[0])
            
            # Return current state with an error message
            HANDLER.render(STATE, legal_actions=legal_action_mask)
            time.sleep(0.1)

            if hasattr(STATE.game_state, "scores"):
                scores = list(map(float, STATE.game_state.scores))
            else:
                scores = [0.0, 0.0]

            return jsonify({
                "svg": HANDLER.rendered_svg,
                "terminated": bool(STATE.terminated),
                "rewards": list(map(int, STATE.rewards)),
                "current_player": int(STATE.game_state.current_player),
                "scores": scores,
                "error": "Illegal move! Please select a valid action."
            })


    # Placement games (board_size,)
    if HANDLER.game_info.action_type == ActionTypes.TO:
        STATE = ENV.step(STATE, action_idx)
        HANDLER.render(STATE)

    # Step / hop games (board_size, num_directions)
    elif HANDLER.game_info.action_type == ActionTypes.FROM_DIR:
        if MOVE_INFO['stage'] == "selecting_piece":
            shaped_mask = STATE.legal_action_mask.reshape((ENV.board_size, HANDLER.game_info.num_directions))
            direction_mask = shaped_mask[action_idx]
            valid_directions = jnp.argwhere(direction_mask).flatten()
            valid_ends = SLIDE_LOOKUP[valid_directions, action_idx,  1] # TODO: handle distances > 1
            legal_moves = jnp.zeros(ENV.board_size)
            legal_moves = legal_moves.at[valid_ends].set(1)

            MOVE_INFO["select_idx"] = action_idx
            HANDLER.render(STATE, legal_actions=legal_moves)
            MOVE_INFO['stage'] = "selecting_destination"

        elif MOVE_INFO['stage'] == "selecting_destination":
            slide_ends = SLIDE_LOOKUP[:, MOVE_INFO["select_idx"], 1]  # TODO: handle distances > 1
            direction_idx = jnp.argwhere(slide_ends == action_idx).flatten()[0]
            final_action_idx = np.ravel_multi_index((MOVE_INFO["select_idx"], direction_idx), (ENV.board_size, HANDLER.game_info.num_directions))

            STATE = ENV.step(STATE, final_action_idx)
            legal_selections = STATE.legal_action_mask.reshape((ENV.board_size, HANDLER.game_info.num_directions)).any(axis=1)
            HANDLER.render(STATE, legal_actions=legal_selections)
            MOVE_INFO['stage'] = "selecting_piece"

    # Sliding / moving games (board_size, board_size)
    elif HANDLER.game_info.action_type == ActionTypes.FROM_TO:
        if MOVE_INFO['stage'] == "selecting_piece":
            legal_moves = STATE.legal_action_mask.reshape((ENV.board_size, ENV.board_size))[action_idx]
            MOVE_INFO["select_idx"] = action_idx
            HANDLER.render(STATE, legal_actions=legal_moves)
            MOVE_INFO['stage'] = "selecting_destination"

            # Remove the "last action" class from the SVG so it doesn't highlight twice
            HANDLER.rendered_svg = HANDLER.rendered_svg.replace(HANDLER.animation_snippet, "")

        elif MOVE_INFO['stage'] == "selecting_destination":
            final_action_idx = np.ravel_multi_index((MOVE_INFO["select_idx"], action_idx), (ENV.board_size, ENV.board_size))
            STATE = ENV.step(STATE, final_action_idx)
            legal_selections = STATE.legal_action_mask.reshape((ENV.board_size, ENV.board_size)).any(axis=1)
            HANDLER.render(STATE, legal_actions=legal_selections)
            MOVE_INFO['stage'] = "selecting_piece"

    else:
        raise ValueError(f"Unknown action type: {HANDLER.game_info.action_type}")

    time.sleep(0.1)

    terminated = bool(STATE.terminated)
    rewards = list(map(int, STATE.rewards))
    if hasattr(STATE.game_state, "scores"):
        scores = list(map(float, STATE.game_state.scores))
    else:
        scores = [0.0, 0.0]

    print("\n" + "-" * 40)
    print(f"Current player: {STATE.game_state.current_player}")
    print(f"Scores: {scores}")
    print("-" * 40)

    # debug_state(STATE, ENV)

    return {"svg": HANDLER.rendered_svg, "terminated": terminated, "rewards": rewards, "current_player": int(STATE.game_state.current_player),
            "scores": scores}

@app.route('/reset', methods=['POST'])
def reset():
    global ENV
    global HANDLER
    global STATE
    global MOVE_INFO

    if ENV is None:
        return "No game loaded"
    
    STATE = ENV.init(jax.random.PRNGKey(42))

    # Placement games (board_size,)
    if HANDLER.game_info.action_type == ActionTypes.TO:
        HANDLER.render(STATE)

    # Step / hop games (board_size, num_directions)
    elif HANDLER.game_info.action_type == ActionTypes.FROM_DIR:
        legal_selections = STATE.legal_action_mask.reshape((ENV.board_size, HANDLER.game_info.num_directions)).any(axis=1)
        HANDLER.render(STATE, legal_actions=legal_selections)

    # Sliding / moving games (board_size, board_size)
    elif HANDLER.game_info.action_type == ActionTypes.FROM_TO:
        legal_selections = STATE.legal_action_mask.reshape((ENV.board_size, ENV.board_size)).any(axis=1)
        HANDLER.render(STATE, legal_actions=legal_selections)

    else:
        raise ValueError(f"Unknown action type: {HANDLER.game_info.action_type}")

    MOVE_INFO = {
        "stage": "selecting_piece",
        "select_idx": None
    }

    time.sleep(0.1)

    return {"svg": HANDLER.rendered_svg}