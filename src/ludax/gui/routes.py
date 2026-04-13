import math

import jax
import jax.numpy as jnp
from flask import render_template, request, jsonify
from markupsafe import Markup
import numpy as np

from .. import environment, utils
from ..config import ActionTypes, Shapes, RENDER_CONFIG

from . import app
from .render import InteractiveBoardHandler
from .rules import generate_rules

ENV, HANDLER, STATE = None, None, None
SLIDE_LOOKUP = None
AVAILABLE_POLICIES = {}  # name -> callable | None (None = human)
P1_POLICY = None
P2_POLICY = None
RNG_KEY = jax.random.PRNGKey(0)

from ludax import games


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_scores():
    if hasattr(STATE.game_state, "scores"):
        return list(map(float, STATE.game_state.scores))
    return [0.0, 0.0]


def _is_ai_turn():
    """True if the current player has an AI policy and the game is not over."""
    if bool(STATE.terminated):
        return False
    current_player = int(STATE.game_state.current_player)
    policy = P1_POLICY if current_player == 0 else P2_POLICY
    return policy is not None


def _render_selection_state():
    """Re-render HANDLER for the start of a new turn (piece-selection view).
    Returns (stage, select_idx) always reset to ("selecting_piece", None)."""
    if HANDLER.game_info.action_type == ActionTypes.TO:
        HANDLER.render(STATE)
    elif HANDLER.game_info.action_type == ActionTypes.FROM_DIR:
        legal_selections = STATE.legal_action_mask.reshape(
            (ENV.board_size, HANDLER.game_info.num_directions)).any(axis=1)
        HANDLER.render(STATE, legal_actions=legal_selections)
    elif HANDLER.game_info.action_type == ActionTypes.FROM_TO:
        legal_selections = STATE.legal_action_mask.reshape(
            (ENV.board_size, ENV.board_size)).any(axis=1)
        HANDLER.render(STATE, legal_actions=legal_selections)
    else:
        raise ValueError(f"Unknown action type: {HANDLER.game_info.action_type}")
    return "selecting_piece", None


def _build_step_response(stage, select_idx):
    return {
        "svg": HANDLER.rendered_svg,
        "terminated": bool(STATE.terminated),
        "rewards": list(map(int, STATE.rewards)),
        "current_player": int(STATE.game_state.current_player),
        "scores": _get_scores(),
        "stage": stage,
        "select_idx": select_idx,
        "ai_turn": _is_ai_turn(),
    }


# ---------------------------------------------------------------------------
# Debug helpers (unused in production paths)
# ---------------------------------------------------------------------------

def display_board(state, env):
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
    board = state.game_state.board
    labels = state.game_state.connected_components
    player = state.game_state.current_player
    flat = jnp.where(board != player, labels, 0).ravel().astype(jnp.int32)
    n_cells = flat.shape[0]
    counts = jnp.bincount(flat, length=n_cells + 1)
    num_components = jnp.count_nonzero(counts[1:] > 0)
    return num_components

def debug_state(state: environment.State, env: environment.LudaxEnvironment):
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


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route('/')
def index():
    return render_template('index.html', games=games.__all__)


@app.route('/game/<id>')
def render_game(id):
    global ENV, HANDLER, STATE, SLIDE_LOOKUP
    global AVAILABLE_POLICIES, P1_POLICY, P2_POLICY, RNG_KEY

    print(f"Loading the following game:\n{getattr(games, id)}")
    ENV = environment.LudaxEnvironment(game_str=getattr(games, id))
    HANDLER = InteractiveBoardHandler(ENV.game_info, ENV.rendering_info)
    if ENV.game_info.uses_slide_logic:
        SLIDE_LOOKUP = utils._get_slide_lookup(ENV.game_info)

    print(f"Rendering info: {ENV.rendering_info.color_mapping}, {ENV.rendering_info.piece_shape_mapping}")

    AVAILABLE_POLICIES = dict(app.config.get('POLICIES', {}))
    # print(f"Loaded policies from config: {list(AVAILABLE_POLICIES.keys())}")

    # Add built-in non-random policies that depend on the loaded environment
    try:
        from ludax.policies.simple import one_ply_policy, random_policy
        from ludax.policies.mctx_v2 import MCTSPolicy

        if 'random' not in AVAILABLE_POLICIES:
            AVAILABLE_POLICIES['random'] = random_policy()

        AVAILABLE_POLICIES['one_ply'] = one_ply_policy(jax.jit(ENV.step))
        AVAILABLE_POLICIES['mctx_easy'] = MCTSPolicy(ENV, num_simulations=1000, max_depth=25)
        AVAILABLE_POLICIES['mctx_hard'] = MCTSPolicy(ENV, num_simulations=15000, max_depth=50)

    except Exception as e:
        print(f"Failed to load built-in policies: {e}")

    P1_POLICY = None
    P2_POLICY = None
    RNG_KEY = jax.random.PRNGKey(0)

    # Build AI policies that need the environment/step function
    step_b = jax.jit(ENV.step)
    try:
        from ludax.policies.simple import one_ply_policy
        AVAILABLE_POLICIES['one_ply'] = one_ply_policy(step_b)
        print("Loaded one-ply lookahead policy")
    except Exception as e:
        print(f"Failed to load one_ply policy: {e}")

    try:
        from ludax.policies.mcts import uct_mcts_policy
        AVAILABLE_POLICIES['mcts_50'] = uct_mcts_policy(ENV, num_simulations=50, max_depth=15)
        print("Loaded MCTS policy (50 simulations)")
    except Exception as e:
        print(f"Failed to load MCTS policy: {e}")

    # Load AlphaZero checkpoint for this game if ludax_agents is installed
    try:
        from ludax_agents import az_checkpoint_policy, get_checkpoint_path
        ckpt_path = get_checkpoint_path(id)
        if ckpt_path is not None:
            AVAILABLE_POLICIES['alphazero'] = az_checkpoint_policy(ENV, ckpt_path)
            print(f"Loaded AlphaZero checkpoint: {ckpt_path}")
    except ImportError:
        pass  # ludax_agents not installed
    except Exception as e:
        print(f"Failed to load AlphaZero checkpoint for '{id}': {e}")

    STATE = ENV.init(jax.random.PRNGKey(42))
    _render_selection_state()

    # Warm up JAX JIT compilation so the first user click is fast
    _ = ENV.step(STATE, 0)

    region_legend = HANDLER.render_legend()
    policy_names = ['human'] + list(AVAILABLE_POLICIES.keys())

    game_str = getattr(games, id)
    rules_html = generate_rules(game_str)

    return render_template('game.html',
                           game_svg=Markup(HANDLER.rendered_svg),
                           region_legend=Markup(region_legend),
                           rules_html=Markup(rules_html),
                           policy_names=policy_names)


@app.route('/set_policy', methods=['POST'])
def set_policy():
    global P1_POLICY, P2_POLICY

    if ENV is None:
        return jsonify({"error": "No game loaded"}), 400

    data = request.get_json()
    player = int(data['player'])
    name = data['policy_name']

    if name == 'human':
        policy_fn = None
    elif name in AVAILABLE_POLICIES:
        policy_fn = AVAILABLE_POLICIES[name]
    else:
        return jsonify({"error": f"Unknown policy: {name}"}), 400

    if player == 0:
        P1_POLICY = policy_fn
    else:
        P2_POLICY = policy_fn

    stage, select_idx = _render_selection_state()
    return jsonify(_build_step_response(stage, select_idx))


@app.route('/step_ai', methods=['POST'])
def step_ai():
    """Advance the game by exactly one AI move. The client loops this to animate AI play."""
    global STATE, RNG_KEY

    if ENV is None:
        return jsonify({"error": "No game loaded"}), 400

    if bool(STATE.terminated) or not _is_ai_turn():
        stage, select_idx = _render_selection_state()
        return jsonify(_build_step_response(stage, select_idx))

    current_player = int(STATE.game_state.current_player)
    policy = P1_POLICY if current_player == 0 else P2_POLICY

    state_b = jax.tree_util.tree_map(lambda x: x[None], STATE)
    RNG_KEY, subkey = jax.random.split(RNG_KEY)
    action_b = policy(state_b, subkey)
    STATE = ENV.step(STATE, int(action_b[0]))

    stage, select_idx = _render_selection_state()
    return jsonify(_build_step_response(stage, select_idx))


@app.route('/step', methods=['POST'])
def step():
    global ENV, HANDLER, STATE, SLIDE_LOOKUP

    if ENV is None:
        return "No game loaded"

    data = request.get_json()
    x = float(data['x'])
    y = float(data['y'])
    stage = data.get('stage', 'selecting_piece')
    select_idx = data.get('select_idx', None)

    action_idx = HANDLER.pixel_to_action((x, y))

    # Deselect if the user clicks the already-selected piece (STATE does not advance)
    if stage == "selecting_destination" and action_idx == select_idx:
        if HANDLER.game_info.action_type == ActionTypes.FROM_DIR:
            legal_selections = STATE.legal_action_mask.reshape((ENV.board_size, HANDLER.game_info.num_directions)).any(axis=1)
        elif HANDLER.game_info.action_type == ActionTypes.FROM_TO:
            legal_selections = STATE.legal_action_mask.reshape((ENV.board_size, ENV.board_size)).any(axis=1)
        else:
            legal_selections = STATE.legal_action_mask
        HANDLER.render(STATE, legal_actions=legal_selections)
        return jsonify(_build_step_response("selecting_piece", None))

    # Compute legal action mask for legality check
    if HANDLER.game_info.action_type == ActionTypes.TO:
        legal_action_mask = STATE.legal_action_mask

    elif HANDLER.game_info.action_type == ActionTypes.FROM_DIR:
        if stage == "selecting_piece":
            legal_action_mask = STATE.legal_action_mask.reshape((ENV.board_size, HANDLER.game_info.num_directions)).any(axis=1)
        elif stage == "selecting_destination":
            direction_mask = STATE.legal_action_mask.reshape((ENV.board_size, HANDLER.game_info.num_directions))[select_idx]
            valid_directions = jnp.argwhere(direction_mask).flatten()
            valid_ends = SLIDE_LOOKUP[valid_directions, select_idx, 1]  # TODO: handle distances > 1
            legal_action_mask = jnp.zeros(ENV.board_size)
            legal_action_mask = legal_action_mask.at[valid_ends].set(1)

    elif HANDLER.game_info.action_type == ActionTypes.FROM_TO:
        if stage == "selecting_piece":
            legal_action_mask = STATE.legal_action_mask.reshape((ENV.board_size, ENV.board_size)).any(axis=1)
        elif stage == "selecting_destination":
            legal_action_mask = STATE.legal_action_mask.reshape((ENV.board_size, ENV.board_size))[select_idx]

    else:
        raise ValueError(f"Unknown action type: {HANDLER.game_info.action_type}")

    # If only one legal action, take it automatically
    if legal_action_mask.sum() == 1:
        action_idx = int(legal_action_mask.argmax())
        print(f"Only one legal action available, taking action {action_idx}!")
    else:
        if action_idx >= len(legal_action_mask) or not legal_action_mask[action_idx]:
            print(f"Illegal action selected: {action_idx}")
            print("Legal action mask:\n", legal_action_mask)
            print("Legal action indices:\n", jnp.where(legal_action_mask)[0])
            HANDLER.render(STATE, legal_actions=legal_action_mask)
            return jsonify({
                **_build_step_response(stage, select_idx),
                "error": "Illegal move! Please select a valid action.",
            })

    new_stage = "selecting_piece"
    new_select_idx = None

    # Placement games (board_size,) — STATE always advances
    if HANDLER.game_info.action_type == ActionTypes.TO:
        STATE = ENV.step(STATE, action_idx)
        new_stage, new_select_idx = _render_selection_state()

    # Step / hop games (board_size, num_directions)
    elif HANDLER.game_info.action_type == ActionTypes.FROM_DIR:
        if stage == "selecting_piece":
            # First click: show valid destinations, do not advance STATE
            shaped_mask = STATE.legal_action_mask.reshape((ENV.board_size, HANDLER.game_info.num_directions))
            direction_mask = shaped_mask[action_idx]
            valid_directions = jnp.argwhere(direction_mask).flatten()
            valid_ends = SLIDE_LOOKUP[valid_directions, action_idx, 1]  # TODO: handle distances > 1
            legal_moves = jnp.zeros(ENV.board_size)
            legal_moves = legal_moves.at[valid_ends].set(1)
            new_select_idx = action_idx
            new_stage = "selecting_destination"
            HANDLER.render(STATE, legal_actions=legal_moves, selected_action=action_idx)

        elif stage == "selecting_destination":
            # Second click: advance STATE
            slide_ends = SLIDE_LOOKUP[:, select_idx, 1]  # TODO: handle distances > 1
            direction_idx = jnp.argwhere(slide_ends == action_idx).flatten()[0]
            final_action_idx = np.ravel_multi_index((select_idx, direction_idx), (ENV.board_size, HANDLER.game_info.num_directions))
            STATE = ENV.step(STATE, final_action_idx)
            new_stage, new_select_idx = _render_selection_state()

    # Sliding / moving games (board_size, board_size)
    elif HANDLER.game_info.action_type == ActionTypes.FROM_TO:
        if stage == "selecting_piece":
            # First click: show valid destinations, do not advance STATE
            legal_moves = STATE.legal_action_mask.reshape((ENV.board_size, ENV.board_size))[action_idx]
            new_select_idx = action_idx
            new_stage = "selecting_destination"
            HANDLER.render(STATE, legal_actions=legal_moves, selected_action=action_idx)
            # Strip last-action animation so it doesn't pulse on the selected piece
            HANDLER.rendered_svg = HANDLER.rendered_svg.replace(HANDLER.animation_snippet, "")

        elif stage == "selecting_destination":
            # Second click: advance STATE
            final_action_idx = np.ravel_multi_index((select_idx, action_idx), (ENV.board_size, ENV.board_size))
            STATE = ENV.step(STATE, final_action_idx)
            new_stage, new_select_idx = _render_selection_state()

    else:
        raise ValueError(f"Unknown action type: {HANDLER.game_info.action_type}")

    print("\n" + "-" * 40)
    print(f"Current player: {STATE.game_state.current_player}")
    print(f"Scores: {_get_scores()}")
    print("-" * 40)

    # debug_state(STATE, ENV)

    return jsonify(_build_step_response(new_stage, new_select_idx))


@app.route('/reset', methods=['POST'])
def reset():
    global STATE, RNG_KEY

    if ENV is None:
        return "No game loaded"

    STATE = ENV.init(jax.random.PRNGKey(42))
    RNG_KEY = jax.random.PRNGKey(0)

    stage, select_idx = _render_selection_state()
    return jsonify(_build_step_response(stage, select_idx))
