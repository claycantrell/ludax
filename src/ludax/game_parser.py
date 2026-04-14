import inspect
from itertools import groupby
import jax
import jax.numpy as jnp
from lark.visitors import Transformer

from .config import EMPTY, P1, P2, TRUE, FALSE, DEFAULT_ARGUMENTS, ActionTypes, Directions, EdgeTypes, MoveTypes, PieceRefs, PlayerAndMoverRefs, OptionalArgs, Shapes, ACTION_DTYPE, BOARD_DTYPE
from .game_info import GameInfo
from . import utils

class GameRuleParser(Transformer):
    def __init__(self, game_info: GameInfo):
        self.game_info = game_info
        self.end_rule_info = []
        self.adjacency_lookup = utils._get_adjacency_lookup(self.game_info)
        if self.game_info.uses_slide_logic:
            self.slide_lookup = utils._get_slide_lookup(self.game_info)
            self.hop_between_lookup = utils._get_hop_between_lookup(self.game_info, self.slide_lookup)

        self.num_directions = 8 if self.game_info.board_shape in [Shapes.SQUARE, Shapes.RECTANGLE] else 6
        self.num_players = game_info.num_players
        self.action_space_shape = None
        self.stacking = game_info.max_stack_height > 1

        # Optional attributes
        if "extra_turn_fn_idx" in self.game_info.game_state_attributes:
            self.extra_turn_fns = []

    def _make_apply_fn(self, piece, side_effect_fn=None):
        """Build an apply_action_fn for any FROM_TO movement type.

        Every move type does: pop source, optional side effect, push destination.
        The side_effect_fn (if any) takes (state, board, src, dst) and returns board.
        """
        board_size = self.game_info.board_size
        num_players = self.num_players

        if side_effect_fn is None:
            def apply_action_fn(state, action):
                src, dst = action // board_size, action % board_size
                board = state.board.at[piece, src].set(EMPTY)
                board = board.at[piece, dst].set(state.current_player)
                dst_idx = ACTION_DTYPE(dst)
                pa = state.previous_actions.at[state.current_player].set(dst_idx).at[num_players].set(dst_idx)
                return state._replace(board=board, previous_actions=pa)
        else:
            def apply_action_fn(state, action):
                src, dst = action // board_size, action % board_size
                board = state.board.at[piece, src].set(EMPTY)
                board = side_effect_fn(state, board, src, dst)
                board = board.at[piece, dst].set(state.current_player)
                dst_idx = ACTION_DTYPE(dst)
                pa = state.previous_actions.at[state.current_player].set(dst_idx).at[num_players].set(dst_idx)
                return state._replace(board=board, previous_actions=pa)

        return apply_action_fn

    def _make_legal_fn(self, piece, reachability_fn):
        """Build a legal_action_fn for any FROM_TO movement type.

        reachability_fn(state) returns (board_size, board_size) mask of
        reachable (src, dst) pairs regardless of piece ownership.
        This function filters to only the current player's pieces.
        """
        def legal_action_fn(state):
            reachable = reachability_fn(state)
            piece_mask = (state.board[piece] == state.current_player).astype(BOARD_DTYPE)
            return jnp.where(piece_mask[:, jnp.newaxis], reachable, jnp.zeros_like(reachable))

        return legal_action_fn

    def __default__(self, data, children, meta):
        if len(children) == 1:
            to_return = children[0]
        else:
            to_return = children

        # In order to handle optional arguments, they need to return their name along with their value
        if data.endswith("_arg"):
            return str(data), to_return
        
        # Special case for shapes (return the shape information along with the dimensions)
        if data.endswith("_shape"):
            return str(data), to_return
        
        return to_return
    
    def __default_token__(self, token):
        '''
        Attempt to parse tokens into the data types they represent,
        but fall back to a string if you can't
        '''
        if token == 'true':
            return True
        elif token == 'false':
            return False

        try:
            return int(token)
        except ValueError:
            return str(token)
    
    def game(self, children):

        # Case 1: no optional rendering rules
        if len(children) == 4:
            _, _, _, game_rule_dicts = children

        # Case 2: optional rendering rules are present
        else:
            _, _, _, game_rule_dicts, _ = children

        # Handle the case where there are no start rules
        if len(game_rule_dicts) == 2:
            play_rule_dict, end_rule_dict = game_rule_dicts
            info_dict = {'start_rules': lambda state: state, **play_rule_dict, **end_rule_dict}
        
        else:
            start_rule_dict, play_rule_dict, end_rule_dict = game_rule_dicts
            info_dict = {**start_rule_dict, **play_rule_dict, **end_rule_dict}

        # Handle any optional game state attributes
        addl_info_functions = []
        for attribute in self.game_info.game_state_attributes:

            # Compute the connected components of the board after each move
            if attribute == "connected_components":
                addl_info_functions.append(utils._get_connected_components_fn(self.game_info, self.adjacency_lookup))

            # Set the "extra turn" flag to the sentinel -1 after each move (it's set to some non-negative index by effect_extra_turn)
            # This means it will only ever be non-negative during the "compute legal actions" step, but that's relevant for the same_piece
            # condition in extra turn effects
            if attribute == "extra_turn_fn_idx":
                addl_info_functions.append(lambda state, action: state._replace(extra_turn_fn_idx=-1))

            if attribute == "promoted":
                addl_info_functions.append(lambda state, action: state._replace(promoted=jnp.zeros(self.game_info.board_size, dtype=jnp.bool_)))

        # Mancala: store initial seeds and pit ownership setup in the info dict
        if hasattr(self, 'initial_seeds'):
            initial_seeds = self.initial_seeds
            board_size = self.game_info.board_size
            half = board_size // 2

            # Original start_rules
            original_start = info_dict.get('start_rules', lambda state: state)

            def mancala_start(state):
                state = original_start(state)
                # Set all pits to initial_seeds
                seed_counts = jnp.full(board_size, BOARD_DTYPE(initial_seeds))
                # Pit ownership: first half = P1, second half = P2
                pit_owner = jnp.concatenate([
                    jnp.zeros(half, dtype=BOARD_DTYPE),
                    jnp.ones(board_size - half, dtype=BOARD_DTYPE)
                ])
                return state._replace(seed_counts=seed_counts, pit_owner=pit_owner)

            info_dict['start_rules'] = mancala_start

        if len(addl_info_functions) > 0:
            n = len(addl_info_functions)
            def addl_info_fn(state, action):
                def body_fn(i, args):
                    state, action = args
                    state = jax.lax.switch(i, addl_info_functions, state, action)
                    return state, action
    
                result, _ = jax.lax.fori_loop(0, n, body_fn, (state, action))
                return result
            
            info_dict['addl_info_fn'] = addl_info_fn

        else:
            info_dict['addl_info_fn'] = lambda state, action: state

        return info_dict
    
    '''
    =========Start rules=========
    '''
    def start_rules(self, children):
        '''
        Start rules are optional, but we assume that if they are present then
        there's at least one
        '''
        rules = children
        n_rules = len(rules)
        
        def combined_start_rules(state):
            body_fn = lambda i, state: jax.lax.switch(i, rules, state)
            result = jax.lax.fori_loop(0, n_rules, body_fn, state)
            return result

        return {'start_rules': combined_start_rules}

    def start_place(self, children):
        '''
        Begin the game by placing one or more pieces belong to a specific
        player at specified locations on the board
        '''
        piece, player_ref, (place_arg_type, place_arg_info) = children

        if player_ref == PlayerAndMoverRefs.P1:
            player = P1
        elif player_ref == PlayerAndMoverRefs.P2:
            player = P2

        if place_arg_type == OptionalArgs.INDICES:
            pattern = jnp.array(place_arg_info, dtype=ACTION_DTYPE)

            def start_fn(state):
                board = state.board.at[piece, pattern].set(player)
                return state._replace(board=board)
            
        elif place_arg_type == OptionalArgs.MULTI_MASK:

            # Case where there's only one mask function (it will be a tuple of (fn, info))
            if isinstance(place_arg_info, tuple):
                place_mask_fns = [place_arg_info[0]]
            else:
                place_mask_fns, _ = list(zip(*place_arg_info))
            
            collect_values = utils._get_collect_values_fn(place_mask_fns)

            def start_fn(state):
                all_masks = collect_values(state)
                result = all_masks.any(axis=0).astype(BOARD_DTYPE)
                sub_board = jnp.where(result, player, state.board[piece])
                board = state.board.at[piece].set(sub_board)
                return state._replace(board=board)

        else:
            raise NotImplementedError(f"Start place argument type {place_arg_type} not implemented yet!")
        
        return start_fn

    '''
    =========Play rules=========
    '''
    def play_rules(self, children):
        '''
        The children of play_rules are play phases, so this function is responsible for collecting
        each phase and constructing the general play rule dictionary.
        '''

        (action_sizes, apply_action_fns, legal_action_mask_fns, 
         apply_effects_fns, next_phase_fns, next_player_fns) = zip(*children)
        

        # For single-phase games, we can just use the functions directly
        if len(action_sizes) == 1:
            apply_action_fn = apply_action_fns[0]
            legal_action_mask_fn = legal_action_mask_fns[0]
            apply_effects_fn = apply_effects_fns[0]
            next_phase_fn = next_phase_fns[0]
            next_player_fn = next_player_fns[0]

        else:
            max_action_size = max(action_sizes)

            # Pad each phase's legal mask and apply_action to the max action size
            # so jax.lax.switch gets uniform output shapes
            padded_legal_fns = []
            padded_apply_fns = []
            for i, (asize, legal_fn, apply_fn) in enumerate(zip(action_sizes, legal_action_mask_fns, apply_action_fns)):
                if asize < max_action_size:
                    pad_size = max_action_size - asize  # compile-time constant
                    def _make_padded_legal(fn=legal_fn, ps=pad_size):
                        def padded(state):
                            base = fn(state)
                            return jnp.concatenate([base, jnp.zeros(ps, dtype=BOARD_DTYPE)])
                        return padded
                    padded_legal_fns.append(_make_padded_legal())

                    def _make_padded_apply(fn=apply_fn, s=asize):
                        def padded(state, action):
                            # Clamp action to valid range for this phase
                            action = jnp.minimum(action, s - 1)
                            return fn(state, action)
                        return padded
                    padded_apply_fns.append(_make_padded_apply())
                else:
                    padded_legal_fns.append(legal_fn)
                    padded_apply_fns.append(apply_fn)

            def apply_action_fn(state, action):
                return jax.lax.switch(state.phase_idx, padded_apply_fns, state, action)

            def legal_action_mask_fn(state):
                return jax.lax.switch(state.phase_idx, padded_legal_fns, state)

            def apply_effects_fn(state, original_player):
                return jax.lax.switch(state.phase_idx, apply_effects_fns, state, original_player)

            def next_phase_fn(state):
                return jax.lax.switch(state.phase_idx, next_phase_fns, state)

            def next_player_fn(state):
                return jax.lax.switch(state.phase_idx, next_player_fns, state)

        play_rule_dict = {
            'action_size': max(action_sizes),
            'apply_action_fn': apply_action_fn,
            'legal_action_mask_fn': legal_action_mask_fn,
            'apply_effects_fn': apply_effects_fn,
            'next_phase_fn': next_phase_fn,
            'next_player_fn': next_player_fn
        }

        return play_rule_dict

    '''
    =========Play phases=========
    '''
    
    def phase_once_through(self, children):
        '''
        Proceed once through the specified order of players and then advance the phase
        '''
        (next_player_fn, num_moves), play_mechanic = children
        action_size, apply_action_fn, legal_action_mask_fn, apply_effects_fn = play_mechanic

        # The phase advances after all the moves in the sequence have been made
        next_phase_fn = lambda state: (state.phase_idx + (state.phase_step_count == num_moves-1), (state.phase_step_count != num_moves-1) * (state.phase_step_count + 1))

        return action_size, apply_action_fn, legal_action_mask_fn, apply_effects_fn, next_phase_fn, next_player_fn
    
    def phase_repeat(self, children):
        '''
        Repeat the specified order of players until the game is over
        '''
        (next_player_fn, _), play_mechanic = children
        action_size, apply_action_fn, legal_action_mask_fn, apply_effects_fn = play_mechanic

        # The phase never advances in this case
        next_phase_fn = lambda state: (state.phase_idx, state.phase_step_count + 1)

        return action_size, apply_action_fn, legal_action_mask_fn, apply_effects_fn, next_phase_fn, next_player_fn
    
    def play_mover_order(self, children):
        '''
        Specifies a finite sequence of P1 and P2 that determines the player order
        '''
        sequence = jnp.array([P1 if child == PlayerAndMoverRefs.P1 else P2 for child in children])
        n = len(sequence)
        apt = self.game_info.actions_per_turn

        if apt > 1:
            def next_player_fn(state):
                # Each player takes apt actions before advancing
                turn_idx = state.phase_step_count // apt
                return sequence[turn_idx % n]
        else:
            def next_player_fn(state):
                return sequence[state.phase_step_count % n]

        return next_player_fn, n

    '''
    =========Play mechanics=========
    '''

    def play_super_mechanic(self, children):
        '''
        A play mechanic is responsible for collecting the information about how to apply actions,
        determine legal actions, and apply effects before making any high-level modifications
        (e.g. applying a "force pass") rule
        '''
        (action_size, apply_action_fn, legal_action_mask_fn, apply_effects_fn), *optional = children
        force_pass = any(v is True for v in optional)

        # Determine if this phase is placement-only (action_size == board_size)
        # vs movement (action_size == board_size^2). Extra turn logic only applies to movement.
        is_placement_phase = (action_size == self.game_info.board_size)

        if force_pass:
            action_size += 1

            # This relies on the fgact that the "passed" property has been added to the state
            # by the GameInfo class
            def new_apply_action_fn(state, action):
                is_pass = (action == action_size - 1)

                pass_fn = lambda state, action: state
                state = jax.lax.cond(is_pass, pass_fn, apply_action_fn, state, action)
                passed = state.passed.at[state.current_player].set(is_pass)

                return state._replace(passed=passed)
            
            def new_legal_action_mask_fn(state):
                base_mask = legal_action_mask_fn(state)
                can_pass = ~base_mask.any()
                mask = jnp.concatenate((base_mask, jnp.array([can_pass], dtype=BOARD_DTYPE)))
                return mask
            
            return action_size, new_apply_action_fn, new_legal_action_mask_fn, apply_effects_fn
        
        elif "extra_turn_fn_idx" in self.game_info.game_state_attributes and not is_placement_phase:
            def new_legal_action_mask_fn(state):
                base_mask = legal_action_mask_fn(state)
                extra_turn_idx = state.extra_turn_fn_idx

                new_mask = jax.lax.switch(extra_turn_idx, self.extra_turn_fns, state, base_mask)
                new_mask = jax.lax.select(extra_turn_idx >= 0, new_mask, base_mask)

                return new_mask
            
            return action_size, apply_action_fn, new_legal_action_mask_fn, apply_effects_fn

        else:
            return action_size, apply_action_fn, legal_action_mask_fn, apply_effects_fn

    '''
    Pieces
    '''
    def piece_reference(self, children):
        '''
        Throwing a syntax error here makes the language not context-free, but it's necessary
        to ensure the game is actually playable
        '''
        piece_ref = children[0]

        if piece_ref not in self.game_info.piece_names:
            raise SyntaxError(f"Piece '{piece_ref}' not defined in game pieces: {self.game_info.piece_names}")
        
        return ACTION_DTYPE(self.game_info.piece_names.index(piece_ref))

    '''
    Regions
    '''
    def region_definition(self, children):
        name, (region_arg_type, region_arg_info) = children
        region_idx = self.game_info.region_names.index(name)

        if region_arg_type == OptionalArgs.INDICES:
            pattern = jnp.array(region_arg_info, dtype=ACTION_DTYPE)
            def region_fn(state):
                mask = jnp.zeros(self.game_info.board_size, dtype=BOARD_DTYPE)
                mask = mask.at[pattern].set(1)
                return mask
            
        elif region_arg_type == OptionalArgs.MULTI_MASK:

            # Case where there's only one mask function (it will be a tuple of (fn, info))
            if isinstance(region_arg_info, tuple):
                region_mask_fns = [region_arg_info[0]]
            else:
                region_mask_fns, _ = list(zip(*region_arg_info))
            
            collect_values = utils._get_collect_values_fn(region_mask_fns)

            def region_fn(state):
                all_masks = collect_values(state)
                mask = all_masks.any(axis=0).astype(BOARD_DTYPE)
                return mask

        else:
            raise NotImplementedError(f"Region argument type {region_arg_type} not implemented yet!")
        
        self.game_info.region_mask_fns[region_idx] = region_fn

    def region_reference(self, children):
        '''
        Throwing a syntax error here makes the language not context-free, but it's necessary
        to ensure the game is actually playable
        '''
        region_ref = children[0]

        if region_ref not in self.game_info.region_names:
            raise SyntaxError(f"Region '{region_ref}' not defined in game regions: {self.game_info.region_names}")
        
        return self.game_info.region_names.index(region_ref)

    '''
    Place rules
    '''
    def play_place(self, children):
        '''
        The default behavior of games currently. Players put a piece onto one position
        on the board. This has one required argument which specifies the mask of legal
        destinations (typically empty squares) and then optional arguments which specify:
        - who the placed piece belongs to (default: the current player)
        - any constraints over the *result* of the action (e.g. in Reversi, where actions
          must be custodial)
        - any additional effects of the action (e.g. incrementing a score)
        '''

        # Case 1: the 'mover' argument is specified
        if isinstance(children[1], str):
            piece, mover_ref, (destination_constraint_fn, _), *optional_args = children
            if mover_ref == PlayerAndMoverRefs.MOVER:
                offset = 0
            elif mover_ref == PlayerAndMoverRefs.OPPONENT:
                offset = 1

        # Case 2 (default): the mover is the current player 
        else:
            piece, (destination_constraint_fn, _), *optional_args = children
            offset = 0

        # Case 1: no optional arguments -- legal actions determined by the destination constraint
        # and there are no effects
        if len(optional_args) == 0:
            legal_action_mask_fn = destination_constraint_fn
            apply_effects_fn = lambda state, original_player: state

        # Case 2: one optional argument is specified -- either a result constraint or an effect
        elif len(optional_args) == 1:
            arg = optional_args[0]
            if arg.__name__ == "result_constraint_fn" or arg.__name__ == "lookahead_mask_fn":
                legal_action_mask_fn = lambda state: destination_constraint_fn(state) & arg(state)
                apply_effects_fn = lambda state, original_player: state

            elif arg.__name__ == "apply_effects_fn":
                legal_action_mask_fn = destination_constraint_fn
                apply_effects_fn = arg

        # Case 3: two optional arguments are specified -- result constraints and effects, always
        # in that order
        else:
            result_constraint_fn, apply_effects_fn = optional_args
            legal_action_mask_fn = lambda state: destination_constraint_fn(state) & result_constraint_fn(state)

        action_size = self.game_info.board_size
        stacking = self.game_info.max_stack_height > 1

        if stacking:
            max_h = self.game_info.max_stack_height

            # Override legal mask: allow placement where stack isn't full
            base_legal = legal_action_mask_fn
            def legal_action_mask_fn(state):
                base = base_legal(state)
                not_full = (state.stack_heights < max_h).astype(BOARD_DTYPE)
                return base | not_full  # can place on empty OR non-full stacks

            def apply_action_fn(state, action):
                owner = (state.current_player + offset) % self.num_players
                state = utils._push_stack(state, action, BOARD_DTYPE(piece), owner)
                previous_actions = state.previous_actions.at[state.current_player].set(ACTION_DTYPE(action)).at[self.num_players].set(ACTION_DTYPE(action))
                return state._replace(previous_actions=previous_actions)
        else:
            def apply_action_fn(state, action):
                board = state.board.at[piece, action].set((state.current_player + offset) % self.num_players)
                previous_actions = state.previous_actions.at[state.current_player].set(ACTION_DTYPE(action)).at[self.num_players].set(ACTION_DTYPE(action))
                return state._replace(board=board, previous_actions=previous_actions)

        return action_size, apply_action_fn, legal_action_mask_fn, apply_effects_fn
    
    def play_drop(self, children):
        '''
        Drop a piece from the player's hand onto the board.
        Only legal if the player has that piece type in hand and destination is valid.
        '''
        piece, (destination_constraint_fn, _) = children

        board_size = self.game_info.board_size

        def legal_action_mask_fn(state):
            base_mask = destination_constraint_fn(state)
            # Only legal if player has this piece in hand
            has_piece = state.hand_pieces[state.current_player, piece] > 0
            return jnp.where(has_piece, base_mask, jnp.zeros_like(base_mask))

        def apply_action_fn(state, action):
            board = state.board.at[piece, action].set(state.current_player)
            hand_pieces = state.hand_pieces.at[state.current_player, piece].add(-1)
            previous_actions = state.previous_actions.at[state.current_player].set(ACTION_DTYPE(action)).at[self.num_players].set(ACTION_DTYPE(action))
            return state._replace(board=board, hand_pieces=hand_pieces, previous_actions=previous_actions)

        apply_effects_fn = lambda state, original_player: state

        return board_size, apply_action_fn, legal_action_mask_fn, apply_effects_fn

    def play_sow(self, children):
        '''
        Mancala sow mechanic. Action = pick a pit owned by current player with seeds > 0.
        Seeds are distributed one-by-one counter-clockwise around the board.
        '''
        # Parse args: seeds:N and optional capture rule
        initial_seeds = 4
        capture_rule = None
        for child in children:
            if isinstance(child, tuple) and child[0] == 'sow_seeds_arg':
                initial_seeds = int(child[1])
            elif isinstance(child, str):
                capture_rule = child

        board_size = self.game_info.board_size
        num_players = self.num_players

        # Store initial seeds for environment init
        self.initial_seeds = initial_seeds

        # Build the sow track for a standard 2-row mancala board
        # Assumes rectangle W H where row 0 = P1 pits, row 1 = P2 pits
        # Track goes: row0 left-to-right, then row1 right-to-left (counter-clockwise)
        if "rectangle" in self.game_info.board_shape:
            w, h = self.game_info.board_dims
            if h == 2:
                # Standard 2-row: P1 = row 0 (left→right), P2 = row 1 (right→left)
                track = list(range(w)) + list(range(2 * w - 1, w - 1, -1))
            else:
                # Multi-row: just go cell by cell
                track = list(range(board_size))
        else:
            track = list(range(board_size))

        track_arr = jnp.array(track, dtype=ACTION_DTYPE)
        track_len = len(track)

        # Pit ownership: first half = P1, second half = P2
        half = board_size // 2
        pit_owner_init = jnp.concatenate([
            jnp.zeros(half, dtype=BOARD_DTYPE),
            jnp.ones(board_size - half, dtype=BOARD_DTYPE)
        ])

        # Build track position lookup: for each cell, its position in the track
        track_pos = jnp.full(board_size, -1, dtype=ACTION_DTYPE)
        for i, cell in enumerate(track):
            track_pos = track_pos.at[cell].set(i)

        def legal_action_mask_fn(state):
            # Can sow from pits owned by current player that have seeds > 0
            owned = (state.pit_owner == state.current_player)
            has_seeds = (state.seed_counts > 0)
            return (owned & has_seeds).astype(BOARD_DTYPE)

        def apply_action_fn(state, action):
            pit = action
            seeds = state.seed_counts[pit]
            seed_counts = state.seed_counts.at[pit].set(0)

            # Find starting position in track
            start_pos = track_pos[pit]

            # Distribute seeds one-by-one along the track using lax.fori_loop
            def sow_one(i, counts):
                pos = (start_pos + i + 1) % track_len
                cell = track_arr[pos]
                return counts.at[cell].add(1)

            seed_counts = jax.lax.fori_loop(0, seeds, sow_one, seed_counts)

            # Find where the last seed landed
            last_pos = (start_pos + seeds) % track_len
            last_cell = track_arr[last_pos]

            # Capture rule: capture_opposite
            # If last seed lands in an empty pit on your side, capture opposite pit's seeds
            if capture_rule == "capture_opposite":
                opposite_cell = board_size - 1 - last_cell
                landed_in_own_empty = (
                    (state.pit_owner[last_cell] == state.current_player) &
                    (state.seed_counts[last_cell] == 0)  # was empty before sowing
                )
                captured = jax.lax.select(
                    landed_in_own_empty,
                    seed_counts[opposite_cell] + 1,  # +1 for the seed that landed
                    BOARD_DTYPE(0)
                )
                seed_counts = jax.lax.cond(
                    landed_in_own_empty,
                    lambda sc: sc.at[last_cell].set(0).at[opposite_cell].set(0),
                    lambda sc: sc,
                    seed_counts
                )
                # Add captured seeds to player's score
                if "scores" in self.game_info.game_state_attributes:
                    scores = state.scores.at[state.current_player].add(captured.astype(REWARD_DTYPE))
                    state = state._replace(scores=scores)

            pa = state.previous_actions.at[state.current_player].set(ACTION_DTYPE(pit)).at[num_players].set(ACTION_DTYPE(last_cell))
            return state._replace(seed_counts=seed_counts, previous_actions=pa)

        apply_effects_fn = lambda state, original_player: state

        return board_size, apply_action_fn, legal_action_mask_fn, apply_effects_fn

    def place_result_constraint(self, children):
        '''
        Result constraints refer to things that must be true about the board
        immediately following an action. Each kind of action has its own
        'result_constraint' function that knows how to apply that action

        TODO: this needs to consider the piece which is being placed, which is only defined
        above it in the syntax tree...
        '''
        (predicate_fn, predicate_fn_info) = children[0]

        # The predicate function defines a condition that must be satisfied over the resulting board
        # in order for a move to be legal. However, certain constraints (specifically, custodial and
        # line constraints) at the current moment can be defined in terms of the *current* board state.
        # The predicate_fn_info dictionary will contain as 'lookahead_mask_fn' if and only if the game's
        # result constraints can be expressed in this way, in which case we use it to optimize the
        # constraint checking. Otherwise, we map the predicate function over all possible resulting
        # board states
        if predicate_fn_info.get('lookahead_mask_fn') is None:
            def apply_action(state, action):
                pred_val = predicate_fn(state._replace(
                    board=state.board.at[action].set(state.current_player),
                    previous_actions=state.previous_actions.at[state.current_player].set(ACTION_DTYPE(action)).at[self.num_players].set(ACTION_DTYPE(action))
                ))
                return pred_val
            
            apply = jax.jit(apply_action)

            # TODO: maybe this logic should be moved to play_place and actually use
            # the 'apply_action_fn' defined there instead of having a separate one
            def result_constraint_fn(state):
                resulting_mask = jax.vmap(apply, in_axes=(None, 0))(state, jnp.arange(self.game_info.board_size, dtype=BOARD_DTYPE))
                return resulting_mask.astype(BOARD_DTYPE)

        else:
            result_constraint_fn = predicate_fn_info['lookahead_mask_fn']

        return result_constraint_fn

    '''
    Move rules
    '''

    def _group_by_index(self, items, index):
        '''
        Group a list of tuples by the specified index
        '''
        grouped = groupby(
            sorted(items, key=lambda x: x[index]),
            key=lambda x: x[index]
        )
        groups = [list(items) for key, items in grouped]
        return groups

    def _combine_move_fns(self, move_types, legal_action_mask_fns, apply_action_fns):
        '''
        Combine multiple movement types into a single legal mask and apply function.

        All movement types produce (board_size, board_size) masks in the unified FROM_TO
        action model. Combining is always: OR the masks, dispatch by checking which mask matched.
        '''
        board_size = self.game_info.board_size

        if len(legal_action_mask_fns) == 1:
            def legal_action_mask_fn(state):
                return legal_action_mask_fns[0](state).flatten().astype(BOARD_DTYPE)
            return legal_action_mask_fn, apply_action_fns[0]

        collect_legal_masks = utils._get_collect_values_fn(legal_action_mask_fns)

        def legal_action_mask_fn(state):
            all_masks = collect_legal_masks(state)
            return all_masks.any(axis=0).flatten().astype(BOARD_DTYPE)

        def apply_action_fn(state, action):
            all_masks = collect_legal_masks(state)
            src = action // board_size
            dst = action % board_size
            move_idx = jnp.argmax(all_masks[:, src, dst])
            return jax.lax.switch(move_idx, apply_action_fns, state, action)

        return legal_action_mask_fn, apply_action_fn


    def play_move(self, children):
        '''
        Compile movement rules. All movement types produce (board_size, board_size) masks
        in the unified FROM_TO action model. action = source * board_size + destination.
        '''
        move_infos, *optional_args = children

        if not isinstance(move_infos, list):
            move_infos = [move_infos]

        board_size = self.game_info.board_size
        action_size = board_size * board_size
        self.action_space_shape = (board_size, board_size)

        legal_fns_by_prio, apply_fns_by_prio = [], []
        for group_by_prio in self._group_by_index(move_infos, -1):

            legal_fns_by_piece, apply_fns_by_piece = [], []
            for group_by_piece in self._group_by_index(group_by_prio, 0):

                _, move_types, legal_fns, apply_fns, _, _ = zip(*list(group_by_piece))
                _legal_action_mask_fn, _apply_action_fn = self._combine_move_fns(
                    move_types, legal_fns, apply_fns
                )

                legal_fns_by_piece.append(_legal_action_mask_fn)
                apply_fns_by_piece.append(_apply_action_fn)

            if len(legal_fns_by_piece) == 1:
                piece_legal_action_mask_fn = legal_fns_by_piece[0]
                piece_apply_action_fn = apply_fns_by_piece[0]
            else:
                # Multiple piece types: OR their masks, dispatch by which piece is at source
                def build_legal(legal_fns):
                    collect_legal_masks = utils._get_collect_values_fn(legal_fns)
                    def piece_legal_action_mask_fn(state):
                        all_masks = collect_legal_masks(state)
                        return all_masks.any(axis=0).astype(BOARD_DTYPE)
                    return piece_legal_action_mask_fn

                def build_apply(apply_fns, bs=board_size):
                    def piece_apply_action_fn(state, action):
                        src = action // bs
                        piece = state.board[:, src].argmax()
                        return jax.lax.switch(piece, apply_fns, state, action)
                    return piece_apply_action_fn

                piece_legal_action_mask_fn = build_legal(legal_fns_by_piece)
                piece_apply_action_fn = build_apply(apply_fns_by_piece)
            
            legal_fns_by_prio.append(piece_legal_action_mask_fn)
            apply_fns_by_prio.append(piece_apply_action_fn)
        

        if len(legal_fns_by_prio) == 1:
            legal_action_mask_fn = legal_fns_by_prio[0]
            apply_action_fn = apply_fns_by_prio[0]
        
        # For legal actions with different priorities, we return the mask of the highest priority
        # with at least one legal action
        else:
            collect_general = utils._get_collect_values_fn(legal_fns_by_prio)
        
            def legal_action_mask_fn(state):
                all_masks = collect_general(state)

                any_legal = all_masks.any()
                first_active = all_masks.any(axis=1).argmax()

                mask = jax.lax.select(any_legal, all_masks[first_active], jnp.zeros(action_size, dtype=BOARD_DTYPE))

                return mask

            def apply_action_fn(state, action):
                # Determine which priority level the action belongs to based on the legal action masks,
                # since we can assume that the action is legal if and only if it is legal under the highest
                # priority mask that contains it
                all_masks = collect_general(state)
                prio_idx = all_masks[:, action].argmax()

                return jax.lax.switch(prio_idx, apply_fns_by_prio, state, action)

        if len(optional_args) == 0:
            apply_effects_fn = lambda state, original_player: state
        else:
            apply_effects_fn = optional_args[0]

        # If necessary, compute the "can_move_again" result and add it to the end of the 
        # apply_action_fn by scanning over the can_move_again functions returned by each move
        # NOTE: if there are multiple move definitions for the same piece / move type, this
        # will apply the *last* can_move_again function (overwriting previous calls)
        if "can_move_again" in self.game_info.game_state_attributes:
            _, _, _, _, can_move_again_fns, _ = zip(*move_infos)

            def new_apply_action_fn(state, action):
                new_state = apply_action_fn(state, action)

                for fn in can_move_again_fns:
                    new_state = fn(new_state)
                
                return new_state

            apply_action_fn_to_return = new_apply_action_fn

        else:
            apply_action_fn_to_return = apply_action_fn

        return action_size, apply_action_fn_to_return, legal_action_mask_fn, apply_effects_fn

    def move_type(self, children):
        '''
        This is the direct grammatical parent of the specific move types (defined below). It's responsible
        for adding optional tracking information to the apply_action_fn that gets used by later predicates,
        like "action_was" in English Draughts
        '''

        piece, move_type, legal_action_fn, apply_action_fn, can_move_again_fn, priority = children[0]
        move_type_idx = list(MoveTypes).index(move_type)

        if "action_was" in self.game_info.game_state_attributes:
            def action_was_fn(state):
                return state._replace(action_was=state.action_was.at[state.current_player].set(move_type_idx))
            
        else:
            action_was_fn = lambda state: state

        def new_apply_action_fn(state, action):
            new_state = apply_action_fn(state, action)
            new_state = action_was_fn(new_state)

            return new_state
        
        return piece, move_type, legal_action_fn, new_apply_action_fn, can_move_again_fn, priority


    def move_leap(self, children):
        '''Leap to non-adjacent cell via offset patterns (knight, camel, etc.).'''
        piece, *optional_args = children
        optional_args = self._parse_optional_args(optional_args)
        capture = optional_args[OptionalArgs.CAPTURE]
        priority = optional_args[OptionalArgs.PRIORITY]
        leap_offsets_raw = optional_args[OptionalArgs.LEAP_OFFSETS]

        offsets = utils.LEAP_PATTERNS.get(leap_offsets_raw, utils.LEAP_PATTERNS['knight']) if isinstance(leap_offsets_raw, str) else leap_offsets_raw
        leap_lookup = utils._get_leap_lookup(self.game_info, offsets)
        board_size = self.game_info.board_size

        def reachability_fn(state):
            friendly = (state.board == state.current_player).any(axis=0).astype(BOARD_DTYPE)
            mask = jnp.zeros((board_size, board_size), dtype=BOARD_DTYPE)
            for oi in range(len(offsets)):
                dests = leap_lookup[oi]
                valid = (dests < board_size).astype(BOARD_DTYPE)
                valid = valid & ~friendly.at[dests.clip(0, board_size - 1)].get()
                mask = mask.at[jnp.arange(board_size), dests.clip(0, board_size - 1)].add(valid)
            return (mask > 0).astype(BOARD_DTYPE)

        legal_action_fn = self._make_legal_fn(piece, reachability_fn)

        if capture:
            def capture_side_effect(state, board, src, dst):
                return board.at[:, dst].set(EMPTY)
            apply_action_fn = self._make_apply_fn(piece, side_effect_fn=capture_side_effect)
        else:
            apply_action_fn = self._make_apply_fn(piece)

        def can_move_again_fn(state):
            return state

        return piece, MoveTypes.LEAP, legal_action_fn, apply_action_fn, can_move_again_fn, priority

    def move_hop(self, children):
        '''
        Hop over a piece to land beyond it. Always produces (board_size, board_size) FROM_TO mask.
        Uses hop_between_lookup to find the cell being hopped over for captures.
        '''
        piece, *optional_args = children
        optional_args = self._parse_optional_args(optional_args)

        direction = optional_args[OptionalArgs.DIRECTION]
        hop_over_piece = optional_args[OptionalArgs.PIECE]
        hop_over_player = optional_args[OptionalArgs.HOP_OVER]
        capture = optional_args[OptionalArgs.CAPTURE]
        priority = optional_args[OptionalArgs.PRIORITY]

        hop_over_mask_fn = utils._get_occupied_mask_fn(hop_over_piece, hop_over_player)
        hop_over_friendly = (hop_over_player == PlayerAndMoverRefs.MOVER)

        board_size = self.game_info.board_size
        hop_between = self.hop_between_lookup
        p1_direction_indices, p2_direction_indices = utils._get_direction_indices(self.game_info, direction)
        all_direction_indices = jnp.array([p1_direction_indices, p2_direction_indices], dtype=BOARD_DTYPE)
        num_dirs = len(p1_direction_indices)
        arange_bs = jnp.arange(board_size, dtype=jnp.int32)

        def legal_action_fn(state):
            dirs = all_direction_indices[state.current_player]
            owned = (state.board[piece] == state.current_player)  # (board_size,)
            hop_over_mask = hop_over_mask_fn(state)

            # Between cells and hop destinations: (num_dirs, board_size)
            between = self.slide_lookup[dirs, :, 1]  # (num_dirs, board_size)
            dests = self.slide_lookup[dirs, :, 2]    # (num_dirs, board_size)

            on_board = dests < board_size
            has_hop_piece = hop_over_mask.at[between.clip(0, board_size - 1)].get()  # (num_dirs, board_size)
            valid = owned[jnp.newaxis, :] & on_board & has_hop_piece.astype(jnp.bool_)

            if hop_over_friendly:
                enemy = ((state.board != EMPTY) & (state.board != state.current_player)).any(axis=0)
                dest_has_enemy = enemy.at[dests.clip(0, board_size - 1)].get()
                valid = valid & dest_has_enemy
            else:
                occupied = (state.board != EMPTY).any(axis=0)
                dest_empty = ~occupied.at[dests.clip(0, board_size - 1)].get()
                valid = valid & dest_empty

            flat_idx = arange_bs[jnp.newaxis, :] * board_size + dests.clip(0, board_size - 1)
            mask = jnp.zeros(board_size * board_size, dtype=BOARD_DTYPE)
            mask = mask.at[flat_idx.flatten()].set(valid.flatten().astype(BOARD_DTYPE))
            return mask.reshape(board_size, board_size)

        if capture:
            def apply_action_fn(state, action):
                src, dst = action // board_size, action % board_size
                between = hop_between[src, dst]  # precomputed between cell

                board = state.board.at[piece, src].set(EMPTY)
                if hop_over_friendly:
                    board = board.at[:, dst].set(EMPTY)
                    board = board.at[piece, dst].set(state.current_player)
                    captured_mask = jnp.zeros_like(state.captured).at[dst].set(True)
                else:
                    board = board.at[piece, dst].set(state.current_player)
                    board = board.at[:, between].set(EMPTY)
                    captured_mask = jnp.zeros_like(state.captured).at[between].set(True)

                previous_actions = state.previous_actions.at[state.current_player].set(ACTION_DTYPE(dst)).at[self.num_players].set(ACTION_DTYPE(dst))
                return state._replace(board=board, previous_actions=previous_actions, captured=captured_mask)
        else:
            def apply_action_fn(state, action):
                src, dst = action // board_size, action % board_size
                board = state.board.at[piece, dst].set(state.current_player)
                board = board.at[piece, src].set(EMPTY)
                previous_actions = state.previous_actions.at[state.current_player].set(ACTION_DTYPE(dst)).at[self.num_players].set(ACTION_DTYPE(dst))
                return state._replace(board=board, previous_actions=previous_actions)

        move_type_idx = list(MoveTypes).index(MoveTypes.HOP)
        def can_move_again_fn(state):
            direction_indices = all_direction_indices[state.current_player]
            end_idx = state.previous_actions[state.current_player]

            hop_over_mask = hop_over_mask_fn(state)
            empty_mask = (state.board == EMPTY).all(axis=0).astype(BOARD_DTYPE)

            step_indices = self.slide_lookup[direction_indices, end_idx, 1].T
            hop_indices = self.slide_lookup[direction_indices, end_idx, 2].T

            step_occ = hop_over_mask.at[step_indices].get(mode='fill', fill_value=0).astype(jnp.bool_)
            hop_empty = empty_mask.at[hop_indices].get(mode='fill', fill_value=0).astype(jnp.bool_)
            can_move_again = (step_occ & hop_empty).any()

            return state._replace(can_move_again=state.can_move_again.at[state.current_player, piece, move_type_idx].set(can_move_again))
        
        return piece, MoveTypes.HOP, legal_action_fn, apply_action_fn, can_move_again_fn, priority

    def move_slide(self, children):
        '''
        Slide a piece in one of the specified directions (default to any) any number of spaces,
        limited by the board boundaries and the non-empty cell encountered
        '''
        
        piece, *optional_args = children
        optional_args = self._parse_optional_args(optional_args)

        priority = optional_args[OptionalArgs.PRIORITY]

        distance = optional_args[OptionalArgs.DISTANCE]
        if distance is None:
            distance = max(self.game_info.observation_shape[:2])

        direction = optional_args[OptionalArgs.DIRECTION]
        p1_direction_indices, p2_direction_indices = utils._get_direction_indices(self.game_info, direction)

        # TODO
        all_direction_indices = jnp.array([p1_direction_indices, p2_direction_indices], dtype=BOARD_DTYPE)
        
        if self.game_info.action_type != ActionTypes.FROM_TO:
            raise ValueError("Slide moves require FROM_TO action type!")


        num_directions = len(p1_direction_indices)
        indices = jnp.arange(self.game_info.board_size, dtype=ACTION_DTYPE)      
        general_indices = jnp.indices((self.game_info.board_size, num_directions, distance), dtype=ACTION_DTYPE)[2]
        occupied_pad = jnp.ones((self.game_info.board_size, num_directions, 1), dtype=ACTION_DTYPE)
        
        def legal_action_fn(state):
            direction_indices = all_direction_indices[state.current_player]

            occupied_mask = (state.board != EMPTY).any(axis=0).astype(BOARD_DTYPE)
            slide_indices = self.slide_lookup[direction_indices, :, :distance].transpose(1, 0, 2) # shape (board_size, num_directions, distance)
            occupied_at_slide = occupied_mask.at[slide_indices].get(mode="fill", fill_value=1)

            # Temporarily set the source square to unoccupied so it doesn't trigger the argmax below
            occupied_at_slide = occupied_at_slide.at[:, :, 0].set(0)

            # Pad the edge of the board with "occupied" so pieces can slide fully across the board
            occupied_at_slide = jnp.concatenate([occupied_at_slide, occupied_pad], axis=2)
            slide_until_idx = jnp.argmax(occupied_at_slide, axis=2)

            # Compute the valid destinations from the slide indices and reshape to (board_size, num_directions*distance)
            valid_destinations = jnp.where(
                general_indices < slide_until_idx[:, :, jnp.newaxis],
                slide_indices, self.game_info.board_size + 1
            ).reshape(self.game_info.board_size, -1)
            
            mask = jnp.zeros((self.game_info.board_size, self.game_info.board_size), dtype=BOARD_DTYPE)
            mask = mask.at[indices[:, jnp.newaxis], valid_destinations].set(1)

            # Pieces can't move onto their own positions
            mask = mask.at[indices, indices].set(0)

            # Filter to only the rows corresponding to pieces belonging to the current player
            piece_mask = (state.board[piece] == state.current_player).astype(BOARD_DTYPE)
            mask = jnp.where(
                piece_mask[:, jnp.newaxis],
                mask, jnp.zeros_like(mask)
            )

            return mask
        
        # Slide's legal_action_fn already filters by piece ownership
        apply_action_fn = self._make_apply_fn(piece)

        move_type_idx = list(MoveTypes).index(MoveTypes.SLIDE)
        def can_move_again_fn(state):
            return state._replace(can_move_again=state.can_move_again.at[state.current_player, piece, move_type_idx].set(1))

        return piece, MoveTypes.SLIDE, legal_action_fn, apply_action_fn, can_move_again_fn, priority
    
    def move_step(self, children):
        '''Step a piece exactly `distance` cells in a straight line.'''
        piece, *optional_args = children
        optional_args = self._parse_optional_args(optional_args)
        priority = optional_args[OptionalArgs.PRIORITY]
        direction = optional_args[OptionalArgs.DIRECTION]
        distance = optional_args[OptionalArgs.DISTANCE] or 1

        board_size = self.game_info.board_size
        p1_dir, p2_dir = utils._get_direction_indices(self.game_info, direction)
        all_dirs = jnp.array([p1_dir, p2_dir], dtype=BOARD_DTYPE)
        num_dirs = len(p1_dir)
        # Precompute source indices for all (src, dir) pairs: shape (num_dirs, board_size)
        arange_bs = jnp.arange(board_size, dtype=jnp.int32)

        def legal_action_fn(state):
            dirs = all_dirs[state.current_player]
            owned = (state.board[piece] == state.current_player)  # (board_size,)
            empty = (state.board == EMPTY).all(axis=0)  # (board_size,)

            # All destinations: (num_dirs, board_size)
            all_dests = self.slide_lookup[dirs, :, distance]  # (num_dirs, board_size)
            on_board = all_dests < board_size
            dest_empty = empty.at[all_dests.clip(0, board_size - 1)].get()  # (num_dirs, board_size)

            valid = owned[jnp.newaxis, :] & dest_empty & on_board  # (num_dirs, board_size)

            if distance > 1:
                for d in range(1, distance):
                    inter = self.slide_lookup[dirs, :, d]
                    inter_empty = empty.at[inter.clip(0, board_size - 1)].get()
                    valid = valid & (inter_empty | (inter >= board_size))

            # Scatter into flat mask: action = src * board_size + dst
            flat_idx = arange_bs[jnp.newaxis, :] * board_size + all_dests.clip(0, board_size - 1)
            mask = jnp.zeros(board_size * board_size, dtype=BOARD_DTYPE)
            mask = mask.at[flat_idx.flatten()].set(valid.flatten().astype(BOARD_DTYPE))
            return mask.reshape(board_size, board_size)
        apply_action_fn = self._make_apply_fn(piece)

        move_type_idx = list(MoveTypes).index(MoveTypes.STEP)
        def can_move_again_fn(state):
            dirs = all_dirs[state.current_player]
            end_idx = state.previous_actions[state.current_player]
            empty = (state.board == EMPTY).all(axis=0).astype(BOARD_DTYPE)
            steps = self.slide_lookup[dirs, end_idx, 1].T
            can_step = (empty.at[steps].get(mode='fill', fill_value=0) == 1).any()
            return state._replace(can_move_again=state.can_move_again.at[state.current_player, piece, move_type_idx].set(can_step))

        return piece, MoveTypes.STEP, legal_action_fn, apply_action_fn, can_move_again_fn, priority

    '''
    Play effects
    '''
    def play_effects(self, children):
        '''
        Each effect is a function that takes a state and the player
        who began the turn, and then returns a new state
        '''

        n = len(children)
        def apply_effects_fn(state, original_player):
            def body_fn(i, args):
                cur_state, original_player = args
                cur_state = jax.lax.switch(i, children, cur_state, original_player)
                return cur_state, original_player
            
            result, _ = jax.lax.fori_loop(0, n, body_fn, (state, original_player))
            return result
        
        return apply_effects_fn
    
    def play_if_effect(self, children):
        '''
        Apply an effect only if a given predicate is true
        '''
        (predicate_fn, _), effect_fn = children

        def dummy_effect(state, original_player):
            return state
        
        def apply_effects_fn(state, original_player):
            updated_state = state._replace(current_player=original_player)
            pred_val = predicate_fn(updated_state)
            return jax.lax.cond(
                pred_val,
                effect_fn,
                dummy_effect,
                state,
                original_player
            )

        return apply_effects_fn
    
    def play_if_else_effect(self, children):
        '''
        Apply one of two effects depending on the value of a given predicate
        '''
        (predicate_fn, _), effect_if_fn, effect_else_fn = children

        def apply_effects_fn(state, original_player):
            updated_state = state._replace(current_player=original_player)
            pred_val = predicate_fn(updated_state)
            return jax.lax.cond(
                pred_val,
                effect_if_fn,
                effect_else_fn,
                state,
                original_player
            )

        return apply_effects_fn

    def effect_capture(self, children):
        '''
        Remove pieces from the board according to a mask

        TODO: specify which piece is captured?
        '''
        (child_mask_fn, _), *optional_args = children
        optional_args = self._parse_optional_args(optional_args)

        mover_ref = optional_args[OptionalArgs.MOVER]
        if mover_ref == PlayerAndMoverRefs.MOVER:
            offset = 0
        else:
            offset = 1

        increment_score = optional_args[OptionalArgs.INCREMENT_SCORE]

        # TODO: if some actions capture and others don't, we need to force an update
        # to the "captured" field to set it back to empty -- maybe this happens in 
        # the "additional info" function?
        def apply_effects_fn(state, original_player):
            updated_state = state._replace(current_player=original_player)
            child_mask = child_mask_fn(updated_state)
            occupied_mask = (updated_state.board == (original_player + offset) % self.num_players).any(axis=0)

            to_capture = (occupied_mask * child_mask).astype(jnp.bool_)
            new_board = jnp.where(to_capture, EMPTY, updated_state.board)

            # If specified, increment the mover's score by the amount of pieces captured
            score_update = jax.lax.select(increment_score, to_capture.sum(), 0)
            scores = state.scores.at[original_player].set(state.scores[original_player] + score_update)

            return state._replace(board=new_board, scores=scores, captured=to_capture)
        
        return apply_effects_fn
    
    def effect_extra_turn(self, children):
        '''
        Cause the specified player to take an extra turn immediately by decrementing
        the phase step count before returning to the normal turn exchange

        NOTE: if two players both trigger an extra turn effect for their opponent in
        succession, it will appear like neither is getting an extra turn, since the bonus
        turns happen immediately after the current turn ends
        '''
        mover_ref, *optional_args = children
        optional_args = self._parse_optional_args(optional_args)

        if mover_ref == PlayerAndMoverRefs.MOVER:
            offset = 0
        else:
            offset = 1

        extra_turn_fn_idx = len(self.extra_turn_fns)

        # If the "same piece" flag is set and the game is a piece-moving game, then we restrict
        # legal actions in the extra turn to only those that move the same piece as in the previous turn,
        # relying on the fact that the rows of the legal action mask correspond to the piece start positions
        if optional_args[OptionalArgs.SAME_PIECE] and self.game_info.move_type == "move":
            action_shape = (self.game_info.board_size, self.game_info.board_size)

            def extra_turn_condition(state, legal_action_mask):
                last_action = state.previous_actions[self.num_players]
                start_mask = jnp.zeros(self.game_info.board_size, dtype=BOARD_DTYPE).at[last_action].set(1)[:, jnp.newaxis]
                base_mask = legal_action_mask.reshape(action_shape)
                new_legal_action_mask = jnp.where(start_mask, base_mask, 0).flatten()
                return new_legal_action_mask
        else:
            def extra_turn_condition(state, legal_action_mask):
                return legal_action_mask
            
        self.extra_turn_fns.append(extra_turn_condition)

        def apply_effects_fn(state, original_player):
            return state._replace(phase_step_count=jnp.maximum(state.phase_step_count - 1, 0), current_player=(original_player + offset) % self.num_players,
                                  extra_turn_fn_idx=extra_turn_fn_idx)
        
        return apply_effects_fn

    def effect_capture_to_hand(self, children):
        '''
        Capture pieces matching a mask and add them to the capturing player's hand.
        Like effect_capture but pieces go to hand instead of being removed.
        '''
        (child_mask_fn, _), *optional_args = children
        optional_args = self._parse_optional_args(optional_args)
        mover_ref = optional_args[OptionalArgs.MOVER]
        offset = 0 if mover_ref == PlayerAndMoverRefs.MOVER else 1

        def apply_effects_fn(state, original_player):
            updated_state = state._replace(current_player=original_player)
            capture_mask = child_mask_fn(updated_state)
            captor = (original_player + offset) % self.num_players

            # For each piece type, count captured pieces and add to hand
            hand_pieces = state.hand_pieces
            board = state.board
            for pt in range(self.game_info.num_piece_types):
                # Pieces of this type that match the capture mask and belong to opponent
                target_player = (captor + 1) % self.num_players
                occupied = (board[pt] == target_player).astype(BOARD_DTYPE)
                to_capture = occupied * capture_mask
                count = to_capture.sum()
                hand_pieces = hand_pieces.at[captor, pt].add(count)
                # Remove captured pieces from board
                board = jnp.where(
                    to_capture[jnp.newaxis, :],
                    jnp.full_like(board[:, :1], EMPTY).broadcast_to(board.shape[0], 1) * jnp.ones_like(board),
                    board
                )

            # Clear all board cells where capture_mask is True
            board = jnp.where(capture_mask[jnp.newaxis, :], EMPTY, state.board)

            # Count what was captured per piece type
            for pt in range(self.game_info.num_piece_types):
                target_player = (captor + 1) % self.num_players
                was_occupied = (state.board[pt] == target_player).astype(BOARD_DTYPE)
                captured_count = (was_occupied * capture_mask).sum()
                hand_pieces = hand_pieces.at[captor, pt].add(captured_count)

            return state._replace(board=board, hand_pieces=hand_pieces)

        return apply_effects_fn

    def effect_flip(self, children):
        '''
        Flip pieces on the board belonging to an opponent according to a mask

        TODO: specify which piece is flipped?
        '''
        (child_mask_fn, _), *optional_args = children
        optional_args = self._parse_optional_args(optional_args)

        mover_ref = optional_args[OptionalArgs.MOVER]
        if mover_ref == PlayerAndMoverRefs.MOVER:
            offset = 0
        else:
            offset = 1

        def apply_effects_fn(state, original_player):
            updated_state = state._replace(current_player=original_player)
            child_mask = child_mask_fn(updated_state)
            occupied_mask = (updated_state.board == (original_player + offset) % self.num_players)

            to_flip = (occupied_mask * child_mask).astype(BOARD_DTYPE)
            new_board = jnp.where(to_flip, (updated_state.board + 1) % self.num_players, updated_state.board)

            return state._replace(board=new_board)
        
        return apply_effects_fn

    def effect_increment_score(self, children):
        '''
        Increment the score for a specified player by the result of a function
        '''
        mover_ref, (amount_fn, _) = children
        if mover_ref == PlayerAndMoverRefs.MOVER:
            offset = 0
        else:
            offset = 1

        # We need to keep track of the player that started the turn
        # in order to correctly increment the score
        def apply_effects_fn(state, original_player):
            updated_state = state._replace(current_player=original_player)
            idx = (updated_state.current_player + offset) % self.num_players
            amount = amount_fn(updated_state)

            scores = state.scores.at[idx].set(state.scores[idx] + amount)
            return state._replace(scores=scores)
        
        return apply_effects_fn
    
    def effect_promote(self, children):
        '''
        Promote pieces on the board according to a mask and a resulting piece type
        '''
        promotee, piece, (child_mask_fn, _), *optional_args = children
        optional_args = self._parse_optional_args(optional_args)

        mover_ref = optional_args[OptionalArgs.MOVER]
        if mover_ref == PlayerAndMoverRefs.MOVER:
            offset = 0
        else:
            offset = 1

        def apply_effects_fn(state, original_player):
            updated_state = state._replace(current_player=original_player)
            child_mask = child_mask_fn(updated_state)
            occupied_mask = (updated_state.board[promotee] == (original_player + offset) % self.num_players)

            promote_mask = (occupied_mask * child_mask).astype(jnp.bool_)
            owner = (original_player + offset) % self.num_players
            # Clear old piece type at promote positions, set new piece type
            new_board = jnp.where(promote_mask[jnp.newaxis, :], EMPTY, updated_state.board)
            new_board = jnp.where(promote_mask[jnp.newaxis, :] & (jnp.arange(self.game_info.num_piece_types)[:, jnp.newaxis] == piece), owner, new_board)

            return state._replace(board=new_board, promoted=promote_mask)

        return apply_effects_fn

    def effect_set_score(self, children):
        '''
        Set the score for a specified player to the result of a function
        '''
        mover_ref, (amount_fn, _) = children
        if mover_ref == PlayerAndMoverRefs.MOVER:
            offset = 0
        else:
            offset = 1

        # We need to keep track of the player that started the turn
        # in order to correctly increment the score
        def apply_effects_fn(state, original_player):
            updated_state = state._replace(current_player=original_player)
            idx = (updated_state.current_player + offset) % self.num_players
            amount = amount_fn(updated_state)

            scores = state.scores.at[idx].set(amount)
            return state._replace(scores=scores)
        
        return apply_effects_fn

    def effect_swap(self, children):
        '''
        Swap the piece at the last-moved position with adjacent pieces matching a mask.
        The mover's piece and the target piece exchange board positions.
        '''
        (child_mask_fn, _) = children[0]

        def apply_effects_fn(state, original_player):
            updated_state = state._replace(current_player=original_player)
            last_pos = state.previous_actions[self.num_players]  # last action position
            swap_mask = child_mask_fn(updated_state)

            # Find adjacent cells to last_pos that match the mask
            adj_mask = self.adjacency_lookup[last_pos]
            targets = swap_mask * adj_mask

            # Swap with the first matching target
            target_idx = jnp.argmax(targets)
            should_swap = targets.sum() > 0

            # Read pieces at both positions
            src_pieces = state.board[:, last_pos]
            dst_pieces = state.board[:, target_idx]

            # Exchange
            board = jax.lax.select(
                should_swap,
                state.board.at[:, last_pos].set(dst_pieces).at[:, target_idx].set(src_pieces),
                state.board
            )

            return state._replace(board=board)

        return apply_effects_fn

    '''
    =========End rules=========
    '''
    def end_rules(self, children):
        '''
        Compile information about the end rules of the game
        '''
        rules = children
        n_rules = len(rules)
        
        # Multiple end rules should be applied in order -- if any of them flag
        # the game is terminated. Rewards are determined by the first rule that
        # flags the game as terminated
        def combined_end_rules(state):
            index = jnp.arange(n_rules)
            winners_by_rule, ends = jax.vmap(lambda i: jax.lax.switch(i, rules, state))(index)

            # Take winner(s) determined by the first active end rule
            winners = jax.lax.select(ends.any(), winners_by_rule[jnp.argmax(ends)], EMPTY * jnp.ones(self.game_info.num_players, jnp.int8))
            end = ends.any()
            
            return winners, end

        return {'end_rules': combined_end_rules}
    
    def end_rule(self, children):
        '''
        Parse out a given ending rule, which takes the form of a
        binary-valued predicate and a game result
        '''
        (predicate_fn, predicate_info), (get_winner, winner_info) = children
        
        # Record the joint information about the predicate and the winner
        joint_info = {**predicate_info, **winner_info}
        self.end_rule_info.append(joint_info)

        def end_rule_fn(state):
            pred_val = predicate_fn(state)
            winner = jax.lax.select(pred_val, get_winner(state), EMPTY * jnp.ones(self.game_info.num_players, jnp.int8))
            termination = jax.lax.select(pred_val, TRUE, FALSE)

            return winner, termination

        return end_rule_fn
    
    def result_win(self, children):
        '''
        The specified player wins the game
        '''
        mover_ref = children[0]

        # In this case, both players win so we can effectively just ignore
        # the offset by setting the "base" to be all ones
        if mover_ref == PlayerAndMoverRefs.BOTH:
            base = jnp.ones(self.game_info.num_players, jnp.int8)
            offset = 0

        else:
            base = jnp.zeros(2, jnp.int8)
            if mover_ref == PlayerAndMoverRefs.MOVER:
                offset = 0
            else:
                offset = 1

        def get_winner(state):
            winners = base.at[(state.current_player + offset) % self.num_players].set(1)
            return winners
        
        info = {}

        return get_winner, info
    
    def result_lose(self, children):
        '''
        The specified player loses the game
        '''
        mover_ref = children[0]

        # The inverse of the above: we can set the base to be all zeros
        # and ignore the offset
        if mover_ref == PlayerAndMoverRefs.BOTH:
            base = jnp.zeros(2, jnp.int8)
            offset = 0

        else:
            base = jnp.ones(self.game_info.num_players, jnp.int8)
            if mover_ref == PlayerAndMoverRefs.MOVER:
                offset = 0
            else:
                offset = 1

        def get_winner(state):
            winners = base.at[(state.current_player + offset) % self.num_players].set(0)
            return winners
        
        info = {}

        return get_winner, info
    
    def result_draw(self, children):
        '''
        The game ends in a draw
        '''
        def get_winner(state):
            winners = EMPTY * jnp.ones(self.game_info.num_players, jnp.int8)
            return winners
        
        info = {}

        return get_winner, info
    
    def result_by_score(self, children):
        '''
        The game is won by the player with the highest score (draw if scores are equal)
        '''

        draw_result = EMPTY * jnp.ones(self.game_info.num_players, jnp.int8)
        p1_win_result = jnp.array([1, 0], dtype=BOARD_DTYPE)
        p2_win_result = jnp.array([0, 1], dtype=BOARD_DTYPE)

        def get_winner(state):
            is_draw = state.scores[0] == state.scores[1]
            p1_wins = state.scores[0] > state.scores[1]

            return jax.lax.select(is_draw, draw_result, jax.lax.select(p1_wins, p1_win_result, p2_win_result))
        
        info = {}

        return get_winner, info

    '''
    =========Masks=========
    '''
    def _parse_optional_args(self, children):
        '''
        Returns the setting of the optional arguments for a given predicate
        by using the name of the calling function as a key
        '''
        predicate_name = inspect.stack()[1][3]
        default_args = DEFAULT_ARGUMENTS.get(predicate_name, {})
        optional_args = {rule: value for rule, value in children}

        return {**default_args, **optional_args}
    
    def super_mask_and(self, children):
        '''
        Apply a location-wise boolean AND operation to the masks of all children
        '''
        children_mask_fns, children_infos = zip(*children)

        collect_values = utils._get_collect_values_fn(children_mask_fns)
        def mask_fn(state):
            all_masks = collect_values(state)
            result = all_masks.all(axis=0).astype(BOARD_DTYPE)

            return result
        
        info = {}

        return mask_fn, info

    def super_mask_or(self, children):
        '''
        Apply a location-wise boolean OR operation to the masks of all children
        '''
        children_mask_fns, children_infos = zip(*children)

        collect_values = utils._get_collect_values_fn(children_mask_fns)
        def mask_fn(state):
            all_masks = collect_values(state)
            result = all_masks.any(axis=0).astype(BOARD_DTYPE)

            return result
        
        info = {"mask_or_children": children_infos}

        return mask_fn, info
    
    def super_mask_not(self, children):
        '''
        Apply a location-wise boolean NOT operation to the mask of the child
        '''
        child_mask_fn, child_info = children[0]

        def mask_fn(state):
            return (child_mask_fn(state) == 0).astype(BOARD_DTYPE)
        
        info = {}

        return mask_fn, info
    
    def mask_like_function(self, children):
        '''
        Returns a mask defined by a special kind of function (currently only "line")
        which defines a mask_fn
        '''
        _, child_info = children[0]
        return child_info['mask_fn'], {}

    
    def mask_adjacent(self, children):
        '''
        Return a mask of all the positions on the board which are adjacent in the
        specified direction to any of the active positions in the child mask
        '''

        (child_mask_fn, child_info), *optional_args = children
        optional_args = self._parse_optional_args(optional_args)

        # TODO: handle relative direction? Odd since we don't specify a player here...
        direction = optional_args[OptionalArgs.DIRECTION]
        direction_indices, _ = utils._get_direction_indices(self.game_info, direction)
        local_lookup = self.adjacency_lookup[direction_indices]

        def mask_fn(state):
            child_mask = child_mask_fn(state)
            adjacent_mask = (child_mask * local_lookup).any(axis=(0, 2)).astype(BOARD_DTYPE)

            return adjacent_mask

        return mask_fn, child_info
    
    def mask_captured(self, children):
        '''
        Return a mask of all the positions on the board which were captured
        in the last turn
        '''
        def mask_fn(state):
            return state.captured.astype(BOARD_DTYPE)
        
        return mask_fn, {}

    def mask_center(self, children):
        '''
        Return the center of the board (defined as halfway through the board array). For
        boards with an even number of positions, this will be empty
        '''
        center_idx = self.game_info.board_size // 2 if self.game_info.board_size % 2 == 1 else self.game_info.board_size + 1
        mask = jnp.zeros(self.game_info.board_size, dtype=BOARD_DTYPE).at[center_idx].set(1)

        def mask_fn(state):
            return mask

        return mask_fn, {}
    
    def mask_column(self, children):
        '''
        Return a mask that's true for all positions in the specified column
        '''
        column_idx = int(children[0])
        column_indices = utils._get_column_indices(self.game_info, column_idx)
        mask = jnp.zeros(self.game_info.board_size, dtype=BOARD_DTYPE).at[column_indices].set(1)

        def mask_fn(state):
            return mask
        
        return mask_fn, {}

    def mask_corners(self, children):
        '''
        Return the corners of the board (the number and location of which depends on the shape)
        '''
        corner_indices = utils._get_corner_indices(self.game_info)
        mask = jnp.zeros(self.game_info.board_size, dtype=BOARD_DTYPE)
        mask = mask.at[corner_indices].set(1)

        def mask_fn(state):
            return mask
        
        return mask_fn, {}
    
    def mask_corner_custodial(self, children):
        '''
        A special case of the custodial mask (below) that tracks whether an opponent's
        piece in the corner is flanked by the mover's pieces along both edges
        '''

        piece, *optional_args = children

        optional_args = self._parse_optional_args(optional_args)
        mover_ref = optional_args[OptionalArgs.MOVER]
        if mover_ref == PlayerAndMoverRefs.MOVER:
            offset = 0
        elif mover_ref == PlayerAndMoverRefs.OPPONENT:
            offset = 1

        corner_indices = utils._get_corner_indices(self.game_info)
        direction_indices, _ = utils._get_direction_indices(self.game_info, Directions.ORTHOGONAL)

        local_lookup = self.adjacency_lookup[direction_indices].any(axis=0)
        outer_indices = []
        for corner in corner_indices:
            outer_indices.append(jnp.argwhere(local_lookup[corner] == 1).flatten())
        
        outer_indices = jnp.array(outer_indices, dtype=ACTION_DTYPE)
        inner_indices = corner_indices[:, jnp.newaxis]
        full_indices = jnp.concatenate([outer_indices[:, 0:1], inner_indices, outer_indices[:, 1:]], axis=1)
        full_match_width = 3

        def mask_fn(state):
            outer_player = (state.current_player + offset) % self.num_players
            inner_player = (outer_player + 1) % self.num_players
            outer_mask = (state.board[piece] == outer_player)
            inner_mask = (state.board[piece] == inner_player)

            # Only keep the custodial arrangements which include the last move
            # among the outer indices. TODO: make this an argument?
            last_move = state.previous_actions[outer_player]
            valid_outer = (outer_indices == last_move).any(axis=1)[:, jnp.newaxis]

            outer_match = (outer_mask[outer_indices] == 1).all(axis=1)[:, jnp.newaxis]
            inner_match = (inner_mask[inner_indices] == 1).all(axis=1)[:, jnp.newaxis]
            full_match = jnp.tile(outer_match & inner_match & valid_outer, full_match_width)

            # Ensure invalid indices are set to one larger than the board size so that they're not indexed
            matched_indices = jnp.where(full_match, full_indices, self.game_info.board_size+1).flatten()
            mask = jnp.zeros(self.game_info.board_size, dtype=BOARD_DTYPE).at[matched_indices].set(1)

            return mask

        return mask_fn, {}

    def mask_custodial(self, children):
        '''
        Returns a mask of all of the board positions which are part of a 'custodial'
        arrangement -- i.e. a line of pieces of the same color with pieces of the 
        opposite color on each end. This is a mask (instead of just a function)
        because we need to be able to count / capture / flip the specific pieces
        that appear in the arrangement

        NOTE: mask_custodial is one of the few masks / functions that has a
        lookahead_mask_fn defined
        '''
        piece, (_, n), *optional_args = children

        optional_args = self._parse_optional_args(optional_args)
        orientation = optional_args[OptionalArgs.ORIENTATION]

        mover_ref = optional_args[OptionalArgs.MOVER]
        if mover_ref == PlayerAndMoverRefs.MOVER:
            offset = 0
        elif mover_ref == PlayerAndMoverRefs.OPPONENT:
            offset = 1
        
        # Concatenate the custodial checks for all valid lengths into one lookup
        if n == "any":
            max_dimension = max(self.game_info.board_dims) - 2

            line_indices = []
            inner_indices, outer_indices = [], []

            for i in range(1, max_dimension+1):
                local_line_indices = utils._get_line_indices(self.game_info, i+2, orientation)
                local_inner_indices, local_outer_indices = utils._get_custodial_indices(self.game_info, i, orientation)
                local_line_indices = jnp.pad(local_line_indices, ((0, 0), (0, max_dimension - i)), mode='maximum')
                local_inner_indices = jnp.pad(local_inner_indices, ((0, 0), (0, max_dimension - i)), mode='maximum')

                line_indices.append(local_line_indices)
                inner_indices.append(local_inner_indices)
                outer_indices.append(local_outer_indices)

            if len(line_indices) > 0:
                line_indices = jnp.concatenate(line_indices, axis=0)
                inner_indices = jnp.concatenate(inner_indices, axis=0)
                outer_indices = jnp.concatenate(outer_indices, axis=0)
            else:
                line_indices = jnp.array([], dtype=ACTION_DTYPE)

        else:
            n = int(n)
            line_indices = utils._get_line_indices(self.game_info, n+2, orientation)
            inner_indices, outer_indices = utils._get_custodial_indices(self.game_info, n, orientation)

        # If there aren't any valid custodial arrangements, we can just return an empty mask
        if line_indices.shape[0] == 0:
            def mask_fn(state):
                return jnp.zeros(self.game_info.board_size, dtype=BOARD_DTYPE)
            
            def lookahead_mask_fn(state):
                return jnp.zeros(self.game_info.board_size, dtype=BOARD_DTYPE)
            
            return mask_fn, {"lookahead_mask_fn": lookahead_mask_fn}
        
        full_match_width = line_indices.shape[1]
        
        def mask_fn(state):
            outer_player = (state.current_player + offset) % self.num_players
            inner_player = (outer_player + 1) % self.num_players
            outer_mask = (state.board[piece] == outer_player)
            inner_mask = (state.board[piece] == inner_player)

            # Only keep the custodial arrangements which include the last move
            # among the outer indices. TODO: make this an argument?
            last_move = state.previous_actions[outer_player]
            valid_outer = (outer_indices == last_move).any(axis=1)[:, jnp.newaxis]

            outer_match = (outer_mask[outer_indices] == 1).all(axis=1)[:, jnp.newaxis]
            inner_match = (inner_mask[inner_indices] == 1).all(axis=1)[:, jnp.newaxis]
            full_match = jnp.tile(outer_match & inner_match & valid_outer, full_match_width)

            # Ensure invalid indices are set to one larger than the board size so that they're not indexed
            matched_indices = jnp.where(full_match, line_indices, self.game_info.board_size+1).flatten()
            mask = jnp.zeros(self.game_info.board_size, dtype=BOARD_DTYPE).at[matched_indices].set(1)

            return mask
        
        # The lookahead mask version looks for lines that have all of the inner indices set and
        # exactly one of the outer indices set -- "left" matches have the first outer index set 
        # and the second empty, and "right" matches are the reverse
        left_indices = outer_indices[:, 0]
        right_indices = outer_indices[:, 1]

        def lookahead_mask_fn(state):
            outer_player = (state.current_player + offset) % self.num_players
            inner_player = (outer_player + 1) % self.num_players

            outer_mask = (state.board[piece] == outer_player)
            inner_mask = (state.board[piece] == inner_player)
            
            inner_match = (inner_mask[inner_indices] == 1).all(axis=1)
            left_match = (outer_mask[left_indices] == 1) & (outer_mask[right_indices] == 0) & inner_match
            right_match = (outer_mask[right_indices]  == 1) & (outer_mask[left_indices] == 0) & inner_match

            left_match_indices = jnp.where(left_match, right_indices, self.game_info.board_size+1)
            right_match_indices = jnp.where(right_match, left_indices, self.game_info.board_size+1)

            mask = jnp.zeros(self.game_info.board_size, dtype=BOARD_DTYPE)
            mask = mask.at[left_match_indices].set(1)
            mask = mask.at[right_match_indices].set(1)

            return mask
        
        return mask_fn, {"lookahead_mask_fn": lookahead_mask_fn}
    
    def mask_edge(self, children):
        '''
        Masks the edges of the board (which depends on the shape of the board). In addition
        to the standard edge types (e.g. top, bottom, ...), this mask also supports "forward"
        and "backward" edges which depend on the "forward assignment" for each player
        '''
        edge_type = children[0]

        # If the edge type is absolute, we can just get the indices directly
        if edge_type in [item.value for item in EdgeTypes]:
            indices = utils._get_edge_indices(self.game_info, edge_type)

            def mask_fn(state):
                mask = jnp.zeros(self.game_info.board_size).astype(jnp.bool_)
                mask = mask.at[indices].set(True)
                return mask.astype(BOARD_DTYPE)

        # Otherwise, we compute them relative to each player's "forward assignment" 
        else:
            p1_edge_type = utils._get_relative_edge_type(self.game_info, edge_type, P1)
            p2_edge_type = utils._get_relative_edge_type(self.game_info, edge_type, P2)

            p1_indices = utils._get_edge_indices(self.game_info, p1_edge_type)
            p2_indices = utils._get_edge_indices(self.game_info, p2_edge_type)

            p1_mask = jnp.zeros(self.game_info.board_size).astype(jnp.bool_).at[p1_indices].set(True)
            p2_mask = jnp.zeros(self.game_info.board_size).astype(jnp.bool_).at[p2_indices].set(True)

            def mask_fn(state):
                mask = jax.lax.select(
                    state.current_player == P1,
                    p1_mask,
                    p2_mask
                )
                return mask.astype(BOARD_DTYPE)
        
        return mask_fn, {}

    def mask_empty(self, children):
        '''
        Return all the positions on the board which are empty of all piece types
        '''
        def mask_fn(state):
            return (state.board == EMPTY).all(axis=0).astype(BOARD_DTYPE)
        
        return mask_fn, {}

    def mask_hopped(self, children):
        '''
        Return the positions on the board which were hopped over
        in the last move
        '''
        
        def mask_fn(state):
            return state.hopped.astype(BOARD_DTYPE)
        
        return mask_fn, {}

    def mask_occupied(self, children):
        '''
        Return the positions on the board which are occupied
        by the specified player. If no player is specified,
        then the mask checks for either player
        '''

        if len(children) == 0:
            def mask_fn(state):
                return (state.board != EMPTY).any(axis=0).astype(BOARD_DTYPE)
            
            return mask_fn, {}
        
        player_or_mover = children[0]
        mask_fn = utils._get_occupied_mask_fn(PieceRefs.ANY, player_or_mover)
         
        return mask_fn, {}

    def mask_prev_move(self, children):
        '''
        Returns a mask that is only active at the position of the current player's
        last move. If it's a player's first move, then the mask is empty (for a game
        with a condition like 'your move must be adjacent to your previous move', you
        may need to express it as multiple phases, where the condition is only applied
        in the second phase after both players have made their first move)
        '''
        mover_ref = children[0]

        if mover_ref == PlayerAndMoverRefs.MOVER:
            offset = 0
        elif mover_ref == PlayerAndMoverRefs.OPPONENT:
            offset = 1

        def mask_fn(state):
            mask = jnp.zeros(self.game_info.board_size).astype(jnp.bool_)
            prev_move = state.previous_actions[(state.current_player + offset) % self.num_players]
            mask = jax.lax.select(prev_move != -1, mask.at[prev_move].set(True), mask)

            return mask.astype(BOARD_DTYPE)
        
        return mask_fn, {}
    
    def mask_promoted(self, children):
        '''
        Return a mask of all the positions on the board which were promoted
        in the last turn
        '''
        def mask_fn(state):
            return state.promoted.astype(BOARD_DTYPE)
        
        return mask_fn, {}

    def mask_row(self, children):
        '''
        Return a mask that's true for all positions in the specified row
        '''
        row_idx = int(children[0])
        row_indices = utils._get_row_indices(self.game_info, row_idx)
        mask = jnp.zeros(self.game_info.board_size, dtype=BOARD_DTYPE).at[row_indices].set(1)

        def mask_fn(state):
            return mask
        
        return mask_fn, {}
    
    def mask_region(self, children):
        '''
        Return a mask that's true for all positions in the specified region
        '''
        region_idx = children[0]
        region_mask_fn = self.game_info.region_mask_fns[region_idx]

        return region_mask_fn, {}

    '''
    ==========Multi-masks==========
    '''
    def _get_static_mask(self, indices):
        mask = jnp.zeros(self.game_info.board_size, dtype=BOARD_DTYPE)
        mask = mask.at[indices].set(1)
        return mask

    def multi_mask_corners(self, children):
        '''
        Returns a list of masks where each mask corresponds to one corner
        '''
        corner_indices = utils._get_corner_indices(self.game_info)
        
        # This mapping over a build function appears necessary to stop the
        # local variable 'idx' from being retroactively bound to the last
        # value for every mask function
        def build(idx):
            mask = self._get_static_mask(idx)
            return lambda state: mask
        
        mask_fns = list(map(build, corner_indices))
        mask_infos = [{} for _ in corner_indices]

        return list(zip(mask_fns, mask_infos))
    
    def multi_mask_edges(self, children):
        '''
        Returns a list of masks where each mask corresponds to one edge
        '''
        edge_types = utils._get_valid_edge_types(self.game_info)

        def build(edge_type):
            mask = self._get_static_mask(utils._get_edge_indices(self.game_info, edge_type))
            return lambda state: mask
        
        mask_fns = list(map(build, edge_types))
        mask_infos = [{} for _ in edge_types]

        return list(zip(mask_fns, mask_infos))
    
    def multi_mask_edges_no_corners(self, children):
        '''
        Returns a list of masks where each mask corresponds to one edge but
        corners are removed
        '''
        edge_types = utils._get_valid_edge_types(self.game_info)
        corner_indices = utils._get_corner_indices(self.game_info)

        def build(edge_type):
            mask = self._get_static_mask(utils._get_edge_indices(self.game_info, edge_type))
            mask = mask.at[corner_indices].set(0)
            return lambda state: mask
        
        mask_fns = list(map(build, edge_types))
        mask_infos = [{} for _ in edge_types]
        
        return list(zip(mask_fns, mask_infos))

    '''
    ==========Predicates==========
    '''
    def super_predicate_and(self, children):
        '''
        Apply a boolean AND operation over all the child predicates
        '''
        children_pred_fns, children_info = zip(*children)
        collect_values = utils._get_collect_values_fn(children_pred_fns)

        def predicate_fn(state):
            all_values = collect_values(state)
            result = all_values.all()
            return result
        
        info = {}

        # If all of the children have a 'lookahead_mask_fn' then we can combine them
        if all([info.get('lookahead_mask_fn', False) for info in children_info]):
            lookahead_mask_fns = [info['lookahead_mask_fn'] for info in children_info]
            collect_lookahead_values = utils._get_collect_values_fn(lookahead_mask_fns)   

            def lookahead_mask_fn(state):
                all_masks = collect_lookahead_values(state)
                result = (all_masks > 0).all(axis=0)
                return result
            
            info['lookahead_mask_fn'] = lookahead_mask_fn

        return predicate_fn, info
    
    def super_predicate_or(self, children):
        '''
        Apply a boolean OR operation over all the child predicates
        '''
        children_pred_fns, children_info = zip(*children)
        collect_values = utils._get_collect_values_fn(children_pred_fns)

        def predicate_fn(state):
            all_values = collect_values(state)
            result = all_values.any()
            return result
        
        info = {}

        # If all of the children have a 'lookahead_mask_fn' then we can combine them
        if all([info.get('lookahead_mask_fn', False) for info in children_info]):
            lookahead_mask_fns = [info['lookahead_mask_fn'] for info in children_info]
            collect_lookahead_values = utils._get_collect_values_fn(lookahead_mask_fns)

            def lookahead_mask_fn(state):
                all_masks = collect_lookahead_values(state)
                result = (all_masks > 0).any(axis=0)
                return result
            
            info['lookahead_mask_fn'] = lookahead_mask_fn

        return predicate_fn, info
    
    def super_predicate_not(self, children):
        '''
        Apply a boolean NOT operation to the child predicate
        '''
        child_pred_fn, child_info = children[0]

        def predicate_fn(state):
            return ~child_pred_fn(state)
        
        info = {}

        if child_info.get('lookahead_mask_fn', False):
            child_lookahead_mask_fn = child_info['lookahead_mask_fn']

            def lookahead_mask_fn(state):
                mask = child_lookahead_mask_fn(state)
                return (mask == 0).astype(BOARD_DTYPE)

            info['lookahead_mask_fn'] = lookahead_mask_fn

        return predicate_fn, info

    def predicate_action_was(self, children):
        '''
        Returns whether the last action taken by the specified player
        was of the specified move type
        '''
        mover_ref, move_type = children
        move_type_idx = list(MoveTypes).index(f"move_{move_type}")

        if mover_ref == PlayerAndMoverRefs.MOVER:
            offset = 0
        elif mover_ref == PlayerAndMoverRefs.OPPONENT:
            offset = 1

        def predicate_fn(state):
            player = (state.current_player + offset) % self.num_players
            return state.action_was[player] == move_type_idx
        
        return predicate_fn, {}

    def predicate_can_move_again(self, children):
        '''
        Returns whether the current player can make another move with the same piece they
        previously moved
        '''

        move_type = children[0]
        move_type_idx = list(MoveTypes).index(f"move_{move_type}")

        def predicate_fn(state):
            piece = state.board[:, state.previous_actions[state.current_player]].argmax()
            return state.can_move_again[state.current_player, piece, move_type_idx]
        
        return predicate_fn, {}

    def predicate_equals(self, children):
        '''
        Return whether all of the child function values are equal
        '''
        children_functions, children_info = zip(*children)
        collect_values = utils._get_collect_values_fn(children_functions)

        def predicate_fn(state):
            all_values = collect_values(state)
            result = (all_values == all_values[0]).all(axis=0)
            return result
        
        info = {}

        # If all of the children have a 'lookahead_mask_fn' then we can combine them
        if all([info.get('lookahead_mask_fn', False) for info in children_info]):
            lookahead_mask_fns = [info['lookahead_mask_fn'] for info in children_info]
            collect_lookahead_values = utils._get_collect_values_fn(lookahead_mask_fns)

            def lookahead_mask_fn(state):
                all_masks = collect_lookahead_values(state)
                result = (all_masks == all_masks[0]).all(axis=0)
                return result

            info['lookahead_mask_fn'] = lookahead_mask_fn

        return predicate_fn, info

    def predicate_captured_all(self, children):
        '''
        Return whether a player has no pieces left on the board.
        (captured_all opponent) = opponent has 0 pieces
        (captured_all mover) = mover has 0 pieces
        '''
        mover_ref = children[0]
        if mover_ref == PlayerAndMoverRefs.MOVER:
            offset = 0
        else:
            offset = 1

        def predicate_fn(state):
            target = (state.current_player + offset) % self.num_players
            return ~(state.board == target).any()

        return predicate_fn, {}

    def predicate_exists(self, children):
        '''
        Return whether the given mask is active anywhere on the board
        '''

        child_mask_fn, child_info = children[0]

        def predicate_fn(state):
            mask = child_mask_fn(state)
            return mask.any()
        
        info = {}
        if child_info.get('lookahead_mask_fn', False):
            child_lookahead_mask_fn = child_info['lookahead_mask_fn']

            def lookahead_mask_fn(state):
                mask = child_lookahead_mask_fn(state)
                return (mask >= 1).astype(BOARD_DTYPE)

            info['lookahead_mask_fn'] = lookahead_mask_fn
        
        return predicate_fn, info
    
    def predicate_full_board(self, children):
        '''
        Return whether the board is completely full of pieces
        '''
        def predicate_fn(state):
            return (state.board != EMPTY).any(axis=0).all()
        
        return predicate_fn, {}
    
    def predicate_function(self, children):
        '''
        Special syntax: returns whether a function has a value greater than 0,
        which is equivalent to (>= function 1)
        '''
        child_fn, child_info = children[0]

        def predicate_fn(state):
            return child_fn(state) > 0
        
        info = {}
        if child_info.get('lookahead_mask_fn', False):
            child_lookahead_mask_fn = child_info['lookahead_mask_fn']

            def lookahead_mask_fn(state):
                mask = child_lookahead_mask_fn(state)
                return (mask >= 1).astype(BOARD_DTYPE)

            info['lookahead_mask_fn'] = lookahead_mask_fn
        
        return predicate_fn, child_info
    
    def predicate_greater_equals(self, children):
        '''
        Returns whether the first child function vlaue is greater than or equal to the second
        '''
        (child_fn_left, child_fn_right), (child_info_left, child_info_right) = zip(*children)

        def predicate_fn(state):
            return child_fn_left(state) >= child_fn_right(state)
        
        info = {}
        if child_info_left.get('lookahead_mask_fn', False) and child_info_right.get('lookahead_mask_fn', False):
            child_lookahead_mask_fn_left = child_info_left['lookahead_mask_fn']
            child_lookahead_mask_fn_right = child_info_right['lookahead_mask_fn']

            def lookahead_mask_fn(state):
                mask_left = child_lookahead_mask_fn_left(state)
                mask_right = child_lookahead_mask_fn_right(state)
                return (mask_left >= mask_right).astype(BOARD_DTYPE)

            info['lookahead_mask_fn'] = lookahead_mask_fn

        return predicate_fn, info
    
    def predicate_last_move_in(self, children):
        '''
        Returns whether the current player's last move was in the specified mask
        '''
        child_mask_fn, child_info = children[0]

        def predicate_fn(state):
            last_move = state.previous_actions[state.current_player]
            mask = child_mask_fn(state)
            return jax.lax.select(last_move != -1, mask[last_move] == 1, FALSE)
        
        info = {}

        return predicate_fn, info

    def predicate_less_equals(self, children):
        '''
        Returns whether the first child function vlaue is less than or equal to the second
        '''
        (child_fn_left, child_fn_right), (child_info_left, child_info_right) = zip(*children)

        def predicate_fn(state):
            return child_fn_left(state) <= child_fn_right(state)
        
        info = {}
        if child_info_left.get('lookahead_mask_fn', False) and child_info_right.get('lookahead_mask_fn', False):
            child_lookahead_mask_fn_left = child_info_left['lookahead_mask_fn']
            child_lookahead_mask_fn_right = child_info_right['lookahead_mask_fn']

            def lookahead_mask_fn(state):
                mask_left = child_lookahead_mask_fn_left(state)
                mask_right = child_lookahead_mask_fn_right(state)
                return (mask_left <= mask_right).astype(BOARD_DTYPE)

            info['lookahead_mask_fn'] = lookahead_mask_fn

        return predicate_fn, info
    
    def predicate_mover_is(self, children):
        '''
        Returns whether the state's current player is the specified player
        '''
        player_ref = children[0]
        
        if player_ref == PlayerAndMoverRefs.P1:
            player = P1
        elif player_ref == PlayerAndMoverRefs.P2:
            player = P2

        def predicate_fn(state):
            return state.current_player == player
        
        info = {}

        return predicate_fn, info
    
    def predicate_no_legal_actions(self, children):
        '''
        Returns whether the acting player has no legal moves available
        '''
        
        def predicate_fn(state):
            return ~state.legal_action_mask.any()

        return predicate_fn, {}
    
    def predicate_passed(self, children):
        '''
        Returns whether one or both players passed their last turn
        '''
        player_ref = children[0]
        
        if player_ref == PlayerAndMoverRefs.BOTH:
            def predicate_fn(state):
                return state.passed.all()

        else:
            if player_ref == PlayerAndMoverRefs.P1:
                idx = 0
            elif player_ref == PlayerAndMoverRefs.P2:
                idx = 1

            def predicate_fn(state):
                return state.passed[idx]

        return predicate_fn, {}
    
    '''
    ==========Functions==========
    '''

    def function_add(self, children):
        '''
        Return the sum of each of the child function values
        '''
        children_fns, children_info = zip(*children)
        collect_values = utils._get_collect_values_fn(children_fns)

        def function_fn(state):
            all_values = collect_values(state)
            result = all_values.sum(axis=0)
            return result
            
        return function_fn, children_info

    def function_constant(self, children):
        '''
        Return a constant specified value

        NOTE: this is one of a few functions / masks which returns a "lookahead_mask_fn"
        '''
        value = children[0]

        def lookahead_mask_fn(state):
            return jnp.ones(self.game_info.board_size, dtype=BOARD_DTYPE) * value
        
        info = {"lookahead_mask_fn": lookahead_mask_fn}

        return lambda state: value, info
    
    def function_connected(self, children):
        '''
        Return the number of the specified masks which are connected to each other
        via the last move made by the current player.

        NOTE: to simplify syntax, it's possible to specify "multi-masks" (see above)
        instead of manually enumerating each of the target masks

        TODO: in games where pieces can be removed, the connected components might
        change in unexpected ways
        '''
        piece, (_, target_masks_infos), *optional_args = children
        optional_args = self._parse_optional_args(optional_args)

        # Parse out the target mask functions
        target_mask_fns, _ = zip(*target_masks_infos)
        collect_masks = utils._get_collect_values_fn(target_mask_fns)

        if optional_args[OptionalArgs.MOVER] == PlayerAndMoverRefs.MOVER:
            offset = 0
        else:
            offset = 1

        def function_fn(state):
            mover = (state.current_player + offset) % self.num_players
            target_masks = collect_masks(state)

            # The connected components are necessarily computed in the update_additional_info call
            set_val = state.previous_actions[mover] + 1
            fill_mask = (state.connected_components[piece] == set_val).astype(ACTION_DTYPE)
            
            target_intersections = jnp.sum(target_masks & fill_mask, axis=1)
            return jnp.sum(target_intersections > 0)

        return function_fn, {}

    def function_count(self, children):
        '''
        Return the number of active positions in the child mask
        '''
        child_mask_fn, child_info = children[0]

        def function_fn(state):
            mask = child_mask_fn(state)
            return mask.sum()
        
        return function_fn, child_info
    
    def function_line(self, children):
        '''
        Returns the number of distinct lines of a particular length present on the board

        NOTE: 'line' is one of a few functions / masks which returns a "lookahead_mask_fn"
        that can be used to optimize certain games with "result constraints"

        NOTE: 'line' is also the only "mask-like" function, which means it also returns a
        "mask_fn" in its info dict. This is mostly useful for game rules that refer to an
        action that forms a line (e.g. "go again if you form a line of 4"), since that can't
        be read from only the function value, but we can check that the last move is part of
        the relevant mask
        '''
        piece, n, *optional_args = children

        n = int(n)
        optional_args = self._parse_optional_args(optional_args)
        orientation = optional_args[OptionalArgs.ORIENTATION]
        player = optional_args[OptionalArgs.PLAYER]

        # Get the basic "occupied" mask function for the specified piece / player, which might
        # be composed with an "exclude" mask to remove certain positions from consideration
        base_mask_fn = utils._get_occupied_mask_fn(piece, player)
        
        if optional_args[OptionalArgs.EXCLUDE] is not None:
            _, exclude_mask_info = optional_args[OptionalArgs.EXCLUDE]

            # Case where there's only one mask function (it will be a tuple of (fn, info))
            if isinstance(exclude_mask_info, tuple):
                exclude_mask_fns = [exclude_mask_info[0]]
            else:
                exclude_mask_fns, _ = list(zip(*exclude_mask_info))
            
            collect_values = utils._get_collect_values_fn(exclude_mask_fns)

            def get_occupied_mask(state):
                occupied_mask = base_mask_fn(state)
                exclude_mask = collect_values(state).any(axis=0).astype(BOARD_DTYPE)
                return occupied_mask & ~exclude_mask
            
        else:
            def get_occupied_mask(state):
                return base_mask_fn(state)


        if optional_args[OptionalArgs.EXACT] and n < max(self.game_info.board_dims):
            line_indices = utils._get_line_indices(self.game_info, n, orientation)

            # If there are no valid lines of this length / orientation, then always return 0
            if line_indices.size == 0:
                return lambda state: 0, {"lookahead_mask_fn": lambda state: jnp.zeros(self.game_info.board_size, dtype=BOARD_DTYPE),
                                         "mask_fn": lambda state: jnp.zeros(self.game_info.board_size, dtype=BOARD_DTYPE)}

            overshoot_line_indices = utils._get_line_indices(self.game_info, n+1, orientation)

            # This array contains the "left" and "right" overshoot positions for each line and is set to
            # an overflow value if there is no overshoot in that direction
            overshoot_indices = jnp.ones((line_indices.shape[0], 2), dtype=ACTION_DTYPE) * (self.game_info.board_size + 1)

            for i in range(line_indices.shape[0]):
                line = line_indices[i]
                left_match = (overshoot_line_indices[:, :-1] == line).all(axis=1)
                right_match = (overshoot_line_indices[:, 1:] == line).all(axis=1)

                if left_match.any():
                    match_idx = jnp.argmax(left_match)
                    overshoot_indices = overshoot_indices.at[i, 0].set(overshoot_line_indices[match_idx, -1])
                if right_match.any():
                    match_idx = jnp.argmax(right_match)
                    overshoot_indices = overshoot_indices.at[i, 1].set(overshoot_line_indices[match_idx, 0])

            def function_fn(state):
                occupied_mask = get_occupied_mask(state)
                line_matches = (occupied_mask[line_indices] == 1).all(axis=1)
                no_overshoot = (occupied_mask.at[overshoot_indices].get(fill_value=0) == 0).all(axis=1)

                no_overshoot_line_matches = line_matches & no_overshoot
                return no_overshoot_line_matches.sum()
            
            def mask_fn(state):
                occupied_mask = get_occupied_mask(state)
                line_matches = (occupied_mask[line_indices] == 1).all(axis=1)
                no_overshoot = (occupied_mask.at[overshoot_indices].get(fill_value=0) == 0).all(axis=1)

                no_overshoot_line_matches = line_matches & no_overshoot

                matched_indices = jnp.where(no_overshoot_line_matches[:, jnp.newaxis], line_indices, self.game_info.board_size+1).flatten()
                mask = jnp.zeros(self.game_info.board_size, dtype=BOARD_DTYPE).at[matched_indices].set(1)

                return mask

            num_lines = line_indices.shape[0]
            def lookahead_mask_fn(state):
                occupied_mask = get_occupied_mask(state)

                line_mask = (occupied_mask[line_indices] == 1)
                almost_line_mask = (line_mask.sum(axis=1) == n-1)
                no_overshoot = (occupied_mask.at[overshoot_indices].get(fill_value=0) == 0).all(axis=1)

                arange = jnp.arange(num_lines).astype(ACTION_DTYPE)
                missing_line_indices = jnp.argmin(line_mask, axis=1).astype(ACTION_DTYPE)
                missing_positions = jnp.where(almost_line_mask & no_overshoot, line_indices[arange, missing_line_indices], self.game_info.board_size+1)
                unique_positions, counts = jnp.unique_counts(missing_positions, size=num_lines, fill_value=self.game_info.board_size+1)

                unique_positions = unique_positions.astype(ACTION_DTYPE)
                counts = counts.astype(BOARD_DTYPE)

                mask = jnp.zeros(self.game_info.board_size, dtype=BOARD_DTYPE)
                mask = mask.at[unique_positions].set(counts)

                return mask

            info = {"mask_fn": mask_fn, "lookahead_mask_fn": lookahead_mask_fn}

        else:
            line_indices = utils._get_line_indices(self.game_info, n, orientation)

            # If there are no valid lines of this length / orientation, then always return 0
            if line_indices.size == 0:
                return lambda state: 0, {"lookahead_mask_fn": lambda state: jnp.zeros(self.game_info.board_size, dtype=BOARD_DTYPE),
                                         "mask_fn": lambda state: jnp.zeros(self.game_info.board_size, dtype=BOARD_DTYPE)}

            def function_fn(state):
                occupied_mask = get_occupied_mask(state)
                line_matches = (occupied_mask[line_indices] == 1).all(axis=1)
                return line_matches.sum()
            
            def mask_fn(state):
                occupied_mask = get_occupied_mask(state)
                line_matches = (occupied_mask[line_indices] == 1).all(axis=1)

                matched_indices = jnp.where(line_matches[:, jnp.newaxis], line_indices, self.game_info.board_size+1).flatten()
                mask = jnp.zeros(self.game_info.board_size, dtype=BOARD_DTYPE).at[matched_indices].set(1)

                return mask
            
            num_lines = line_indices.shape[0]
            def lookahead_mask_fn(state):
                occupied_mask = get_occupied_mask(state)

                line_mask = (occupied_mask[line_indices] == 1)
                almost_line_mask = (line_mask.sum(axis=1) == n-1)

                arange = jnp.arange(num_lines).astype(ACTION_DTYPE)
                missing_line_indices = jnp.argmin(line_mask, axis=1).astype(ACTION_DTYPE)
                missing_positions = jnp.where(almost_line_mask, line_indices[arange, missing_line_indices], self.game_info.board_size+1)
                unique_positions, counts = jnp.unique_counts(missing_positions, size=num_lines, fill_value=self.game_info.board_size+1)

                unique_positions = unique_positions.astype(ACTION_DTYPE)
                counts = counts.astype(BOARD_DTYPE)

                mask = jnp.zeros(self.game_info.board_size, dtype=BOARD_DTYPE)
                mask = mask.at[unique_positions].set(counts)

                return mask

            info = {"mask_fn": mask_fn, "lookahead_mask_fn": lookahead_mask_fn}
        
        return function_fn, info

    def function_multiply(self, children):
        '''
        Return the product of each of the child function values
        '''
        children_fns, children_info = zip(*children)
        collect_values = utils._get_collect_values_fn(children_fns)

        def function_fn(state):
            all_values = collect_values(state)
            result = all_values.prod(axis=0)
            return result
            
        return function_fn, {}
    
    def function_pattern(self, children):
        piece, (pattern_arg_type, pattern), *optional_args = children
        optional_args = self._parse_optional_args(optional_args)

        player = optional_args[OptionalArgs.PLAYER]
        exclude = optional_args[OptionalArgs.EXCLUDE]
        rotate = optional_args[OptionalArgs.ROTATE]

        # Get the basic "occupied" mask function for the specified piece / player, which might
        # be composed with an "exclude" mask to remove certain positions from consideration
        base_mask_fn = utils._get_occupied_mask_fn(piece, player)
        
        if exclude is not None:
            _, exclude_mask_info = exclude

            # Case where there's only one mask function (it will be a tuple of (fn, info))
            if isinstance(exclude_mask_info, tuple):
                exclude_mask_fns = [exclude_mask_info[0]]
            else:
                exclude_mask_fns, _ = list(zip(*exclude_mask_info))
            
            collect_values = utils._get_collect_values_fn(exclude_mask_fns)

            def get_occupied_mask(state):
                occupied_mask = base_mask_fn(state)
                exclude_mask = collect_values(state).any(axis=0).astype(BOARD_DTYPE)
                return occupied_mask & ~exclude_mask
            
        else:
            def get_occupied_mask(state):
                return base_mask_fn(state)
            
        # Precompute the pattern indices
        pattern_indices = utils._get_pattern_indices(self.game_info, pattern_arg_type, pattern, rotate)

        # If there are no pattern placements, then always return 0
        if pattern_indices.size == 0:
            return lambda state: 0, {"lookahead_mask_fn": lambda state: jnp.zeros(self.game_info.board_size, dtype=BOARD_DTYPE),
                                     "mask_fn": lambda state: jnp.zeros(self.game_info.board_size, dtype=BOARD_DTYPE)}

        def function_fn(state):
            occupied_mask = get_occupied_mask(state)
            line_matches = (occupied_mask[pattern_indices] == 1).all(axis=1)
            return line_matches.sum()
        
        # TODO: mask_fn and lookahead_mask_fn

        return function_fn, {}

    def function_score(self, children):
        '''
        Return the score of the specified player
        '''
        mover_ref = children[0]
        
        if mover_ref == PlayerAndMoverRefs.MOVER:
            offset = 0
        elif mover_ref == PlayerAndMoverRefs.OPPONENT:
            offset = 1

        def function_fn(state):
            score = state.scores[(state.current_player + offset) % self.num_players]
            return score
        
        return function_fn, {}

    def function_subtract(self, children):
        '''
        Return the difference between the first and second child function values
        '''
        (child_fn1, _), (child_fn2, _) = children

        def function_fn(state):
            return child_fn1(state) - child_fn2(state)
        
        return function_fn, {}