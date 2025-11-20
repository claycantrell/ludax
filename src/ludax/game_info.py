from collections import namedtuple
from dataclasses import dataclass
import typing

import jax.numpy as jnp
from lark import Lark, Token, Tree
from lark.visitors import Visitor

from .config import TRUE, FALSE, BoardShapes, PieceShapes, PlayerAndMoverRefs

@dataclass
class GameInfo:
    num_players: int = 2
    board_shape: str = None
    observation_shape: tuple = None
    board_dims: tuple = None
    board_size: int = None
    hex_diameter: int = None
    game_state_class: type = None
    game_state_attributes: list = None
    move_type: str = None

    num_piece_types: int = None
    piece_names: tuple[str] = ()
    piece_owners: tuple[str] = ()

    forward_directions: tuple[str] = ()

    def __repr__(self):
        return f"GameInfo(board_shape={self.board_shape}, observation_shape={self.observation_shape}, board_size={self.board_size}, hex_diameter={self.hex_diameter})"

class RenderingInfo:
    color_mapping: dict = {"P1": "white", "P2": "black"}
    piece_shape_mapping: dict = {}

class GameInfoExtractor(Visitor):
    def __init__(self):
        self.game_info = GameInfo()

        # These attributes are shared across all games
        self.game_state_attributes = [
            "board",
            "legal_action_mask",
            "current_player",
            "phase_idx",
            "phase_step_count",

            # The action position of the previous action (i.e. destination for piece movement) for each player.
            # The final position stores the last action taken (regardless of player) which is useful for cases
            # of double-moves
            "previous_actions"
        ]

        self.defaults = []

        self.rendering_info = RenderingInfo()
    
    def __call__(self, tree):
        self.visit_topdown(tree)
        
        # We dynamically create a class for the game state that includes only the necessary attributes
        defaults = self.defaults if self.defaults else None
        game_state_class = namedtuple("GameState", self.game_state_attributes, defaults=defaults)
        self.game_info.game_state_class = game_state_class
        self.game_info.game_state_attributes = self.game_state_attributes

        # Check to see if any pieces don't have an assigned shape and assign them the first unused shape
        assigned_shapes = set(self.rendering_info.piece_shape_mapping.values())
        unused_shapes = [shape for shape in PieceShapes if shape not in assigned_shapes]

        for piece_name in self.game_info.piece_names:
            if piece_name not in self.rendering_info.piece_shape_mapping:
                self.rendering_info.piece_shape_mapping[piece_name] = unused_shapes.pop(0)

        return self.game_info, self.rendering_info

    def _nav(self, tree: Tree, children_indices: typing.Union[int, list[int]]):
        if isinstance(children_indices, int):
            children_indices = [children_indices]

        while len(children_indices) > 0:
            tree = tree.children[children_indices.pop(0)]

        if isinstance(tree, Token):
            return tree.value
        
        return tree

    def board(self, tree):

        shape_tree = tree.children[0]
        board_shape = str(shape_tree.data)
        self.game_info.board_shape = board_shape

        if board_shape == BoardShapes.SQUARE:
            size = int(shape_tree.children[0])
            self.game_info.observation_shape = (size, size, 2)
            self.game_info.board_dims = (size, size)
            self.game_info.board_size = size ** 2        

        elif board_shape == BoardShapes.RECTANGLE:
            width, height = map(int, shape_tree.children)
            self.game_info.observation_shape = (width, height, 2)
            self.game_info.board_dims = (width, height)
            self.game_info.board_size = width * height

        elif board_shape == BoardShapes.HEXAGON:
            diameter = int(shape_tree.children[0])
            assert diameter % 2 == 1, "Hexagon board diameter must be odd!"

            self.game_info.observation_shape = (diameter, 2*diameter-1, 2)
            self.game_info.board_dims = (diameter, 2*diameter-1)
            self.game_info.hex_diameter = diameter

            # Consider rings of tiles moving outward from the center tile.
            # The total number of tiles in a hex-hex board is: 1 + [(i* 6) for i in range(1, diameter-1)]
            self.game_info.board_size = 1 + sum([(i * 6) for i in range(1, diameter//2 + 1)])

        elif board_shape == BoardShapes.HEX_RECTANGLE:
            width, height = map(int, shape_tree.children)
            self.game_info.observation_shape = (width, height, 2)
            self.game_info.board_dims = (width, height)
            self.game_info.board_size = width * height    

        else:
            raise NotImplementedError(f"Board shape {board_shape} not implemented yet!")
        
    def pieces(self, tree):
        piece_infos = list(map(lambda x: (self._nav(x, 0), self._nav(x, 1)), tree.children))
        piece_names, piece_owners = zip(*piece_infos)

        if len(set(piece_names)) != len(piece_names):
            raise SyntaxError(f"Piece names must be unique: {piece_names}")

        if len(piece_names) > len(PieceShapes):
            raise ValueError(f"Number of piece types exceeds available shapes: {len(piece_names)} > {len(PieceShapes)}")
        
        self.game_info.num_piece_types = len(piece_names)
        self.game_info.piece_names = tuple(piece_names)
        self.game_info.piece_owners = tuple(piece_owners)
    
    def force_pass(self, tree):
        if "passed" not in self.game_state_attributes:
            self.game_state_attributes.append("passed")
            self.defaults.append(jnp.zeros(2, dtype=jnp.bool_))

    def function_connected(self, tree):
        '''
        Checking if two regions are connected requires computing the connected components of the board
        '''
        if "connected_components" not in self.game_state_attributes:
            self.game_state_attributes.append("connected_components")
            self.defaults.append(jnp.zeros((self.game_info.num_piece_types, self.game_info.board_size), dtype=jnp.int8))

    def play_effects(self, tree):
        '''
        Play effects might require referencing the score. Currently, effects are the only way
        to change the score so we don't need to separately look for things like the "score" function
        '''
        if "scores" not in self.game_state_attributes:
            self.game_state_attributes.append("scores")
            self.defaults.append(jnp.zeros(2, dtype=jnp.float32))

    def play_mechanic(self, tree):
        child = tree.children[0]
        
        if child.data == "play_place":
            self.game_info.move_type = "place"
        elif child.data == "play_move" or child.data == "play_multi_move":
            self.game_info.move_type = "move"
        else:
            raise NotImplementedError(f"Play mechanic {child.data} not implemented yet!")

    def effect_capture(self, tree):
        '''
        Track which board positions had a piece captured in the last move
        '''
        if "captured" not in self.game_state_attributes:
            self.game_state_attributes.append("captured")
            self.defaults.append(jnp.zeros(self.game_info.board_size, dtype=jnp.bool_))

    def effect_extra_turn(self, tree):
        '''
        Track whether the current player is about to take an extra turn as well as the index
        of the "extra turn function" that granted it
        '''
        if "extra_turn_fn_idx" not in self.game_state_attributes:
            self.game_state_attributes.append("extra_turn_fn_idx")
            self.defaults.append(-1)

    def effect_promote(self, tree):
        '''
        Track which pieces were promoted in the last move
        '''
        if "promoted" not in self.game_state_attributes:
            self.game_state_attributes.append("promoted")
            self.defaults.append(jnp.zeros(self.game_info.board_size, dtype=jnp.bool_))

    def mask_captured(self, tree):
        '''
        Track which board positions had a piece captured in the last move
        '''
        if "captured" not in self.game_state_attributes:
            self.game_state_attributes.append("captured")
            self.defaults.append(jnp.zeros(self.game_info.board_size, dtype=jnp.bool_))

    def mask_hopped(self, tree):
        '''
        Hopping movement requires tracking which pieces are hopped over
        '''
        if "hopped" not in self.game_state_attributes:
            self.game_state_attributes.append("hopped")
            self.defaults.append(jnp.zeros(self.game_info.board_size, dtype=jnp.bool_))

    '''
    Custom assignments for relative directions (i.e. "forward")
    '''
    def forward_assignments(self, tree):
        p1_dir = self._nav(tree, [0, 1])
        p2_dir = self._nav(tree, [1, 1])
        self.game_info.forward_directions = (p1_dir, p2_dir)

    '''
    Rendering and graphics related functions
    '''
    def color_assignment(self, tree):
        player, color = map(str, tree.children)
        self.rendering_info.color_mapping[player] = color

    def piece_shape_assignment(self, tree):
        piece, shape = map(str, tree.children)
        self.rendering_info.piece_shape_mapping[piece] = shape

if __name__ == '__main__':
    grammar = open('grammar.lark').read()
    parser = Lark(grammar, start='game')

    game = open('games/tic_tac_toe.ldx').read()
    tree = parser.parse(game)

    info, rendering_info = GameInfoExtractor()(tree)

    print(info)