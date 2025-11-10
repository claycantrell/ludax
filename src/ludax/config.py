from typing import Any
try:
    from enum import StrEnum  # Python ≥ 3.11
except ImportError: # Python ≤ 3.10
    from enum import Enum
    class StrEnum(str, Enum):
        """Minimal stand-in for the 3.11 enum.StrEnum."""
        def _generate_next_value_(name, start, count, last_values):
            return name.lower()

import jax.numpy as jnp

from .struct import dataclass

Array = Any
PRNGKey = Any

TRUE = jnp.bool_(True)
FALSE = jnp.bool_(False)

INVALID = jnp.int16(-2)  # used to represent invalid cells when projecting hex grid onto rectangular array
EMPTY = jnp.int16(-1)
P1 = jnp.int16(0)
P2 = jnp.int16(1)

MAX_STEP_COUNT = 200

@dataclass
class State():
    '''
    The current state of the environment

    Attributes:
    - game_state: a custom class that contains all the information necessary to advance the game state (e.g. board, current player, ...)
    - current_player: the identity of the current player (copied from game_state for convenience)
    - legal_action_mask: a mask indicating which actions are legal
    - winner: the player who won the game
    - rewards: the rewards for each player
    - mover_reward: the reward for the player who made the last move
    - terminated: whether the game has ended
    - truncated: whether the game has been truncated for lasting too many steps
    - global_step_count: the number of steps taken in the game
    '''
    game_state: type
    current_player: Array
    legal_action_mask: Array
    winners: Array = EMPTY * jnp.ones(2, jnp.int16)
    rewards: Array = jnp.float32([0.0, 0.0])
    mover_reward: Array = jnp.float32(0.0)
    terminated: Array = FALSE
    truncated: Array = FALSE
    global_step_count: Array = jnp.int16(0)

class BoardShapes(StrEnum):
    SQUARE = 'board_square'
    RECTANGLE = 'board_rectangle'
    HEXAGON = 'board_hexagon'
    HEX_RECTANGLE = 'board_hex_rectangle'

class EdgeTypes(StrEnum):
    TOP = 'top'
    BOTTOM = 'bottom'
    LEFT = 'left'
    RIGHT = 'right'

    # Hexagonal edges
    TOP_LEFT = 'top_left'
    TOP_RIGHT = 'top_right'
    BOTTOM_LEFT = 'bottom_left'
    BOTTOM_RIGHT = 'bottom_right'

class Directions(StrEnum):
    ANY = 'any'
    BACK_DIAGONAL = 'back_diagonal'
    DOWN = 'down'
    DOWN_LEFT = 'down_left'
    DOWN_RIGHT = 'down_right'
    FORWARD_DIAGONAL = 'forward_diagonal'
    DIAGONAL = 'diagonal'
    HORIZONTAL = 'horizontal'
    LEFT = 'left'
    ORTHOGONAL = 'orthogonal'
    RIGHT = 'right'
    UP = 'up'
    UP_LEFT = 'up_left'
    UP_RIGHT = 'up_right'
    VERTICAL = 'vertical'

class Orientations(StrEnum):
    ANY = 'any'
    BACK_DIAGONAL = 'back_diagonal'
    FORWARD_DIAGONAL = 'forward_diagonal'
    DIAGONAL = 'diagonal'
    HORIZONTAL = 'horizontal'
    ORTHOGONAL = 'orthogonal'
    VERTICAL = 'vertical'

class PlayPhases(StrEnum):
    SPECIFIC_PLAYER = 'phase_specific_player'
    ALTERNATE = 'phase_alternate'
    ALTERNATURE_UNTIL = 'phase_alternate_until'

class PlayTypes(StrEnum):
    MOVE = 'play_move'
    PLACE = 'play_place'
    REMOVE = 'play_remove'

class PlayerAndMoverRefs(StrEnum):
    MOVER = 'mover'
    OPPONENT = 'opponent'
    P1 = 'P1'
    P2 = 'P2'
    BOTH = 'both'

class GameResult(StrEnum):
    WIN = 'result_win'
    LOSS = 'result_lose'
    DRAW = 'result_draw'
    BY_SCORE = 'result_by_score'

class Functions(StrEnum):
    CONNECTED = 'function_connected'
    LINE = 'function_line'

class Masks(StrEnum):
    ADJACENT = 'mask_adjacent'
    CORNER_CUSTODIAL = 'mask_corner_custodial'
    CUSTODIAL = 'mask_custodial'
    EMPTY = 'mask_empty'
    LOOP = 'mask_loop'
    OCCUPIED = 'mask_occupied'
    PATTERN = 'mask_pattern'

class MoveTypes(StrEnum):
    HOP = 'move_hop'
    SLIDE = 'move_slide'
    PLACE = 'move_place'

class PlayEffects(StrEnum):
    CAPTURE = 'effect_capture'
    FLIP = 'effect_flip'

class Predicates(StrEnum):
    EXISTS = 'predicate_exists'

class OptionalArgs(StrEnum):
    DIRECTION = 'direction_arg'
    DISTANCE = 'distance_arg'
    EXACT = 'exact_arg'
    EXCLUDE = 'exclude_arg'
    INCREMENT_SCORE = 'increment_score_arg'
    ORIENTATION = 'orientation_arg'
    MOVER = 'mover_arg'
    MULTI_MASK = 'multi_mask_arg'
    PATTERN = 'pattern_arg'
    PLAYER = 'player_arg'
    ROTATE = 'rotate_arg'

DEFAULT_ARGUMENTS = {
    Functions.CONNECTED: {OptionalArgs.MOVER: 'mover', OptionalArgs.DIRECTION: 'any'},
    Functions.LINE: {OptionalArgs.ORIENTATION: 'any', OptionalArgs.EXACT: False, OptionalArgs.PLAYER: 'mover', OptionalArgs.EXCLUDE: None},

    Masks.ADJACENT: {OptionalArgs.DIRECTION: 'any'},
    Masks.CORNER_CUSTODIAL: {OptionalArgs.MOVER: 'mover'},
    Masks.CUSTODIAL: {OptionalArgs.MOVER: 'mover', OptionalArgs.ORIENTATION: 'any'},
    Masks.LOOP: {OptionalArgs.MOVER: 'mover'},
    Masks.OCCUPIED: {OptionalArgs.MOVER: 'mover'},
    Masks.PATTERN: {OptionalArgs.ROTATE: False},

    MoveTypes.HOP: {OptionalArgs.DIRECTION: 'any'},
    MoveTypes.SLIDE: {OptionalArgs.DIRECTION: 'any', OptionalArgs.DISTANCE: None},

    PlayEffects.CAPTURE: {OptionalArgs.MOVER: 'opponent', OptionalArgs.INCREMENT_SCORE: False},
    PlayEffects.FLIP: {OptionalArgs.MOVER: 'opponent'},
}


RENDER_CONFIG = {
    "cell_size": 50,
    "piece_radius": 35,
    "legal_radius": 15,
    "hexagon_orientation": "pointy",

    # Colors
    "light_blue": "#d2e6ff",
    "light_grey": "#c3cdd8",
    "white": "#fafafa",
    "black": "#323232",
    "dark_grey": "#a6a6a6",
    "purple": "#8a7af0",

    # SVG Style information
    "svg_style": """
        <style>
        .fade-in-out {
            animation: fadeOutIn 2s ease-in-out;
        }
        @keyframes fadeOutIn {
            0% { opacity: 1; }
            50% { opacity: 0; }
            100% { opacity: 1; }
        }
        </style>
    """
}