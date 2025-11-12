import inspect
from itertools import groupby
import jax
import jax.numpy as jnp
from lark.visitors import Transformer

from .config import EMPTY, P1, P2, TRUE, FALSE, DEFAULT_ARGUMENTS, Directions, EdgeTypes, PieceRefs, PlayerAndMoverRefs, OptionalArgs
from .game_info import GameInfo
from . import utils

class GameRuleParser(Transformer):
    def __init__(self, game_info: GameInfo):
        self.game_info = game_info
        self.end_rule_info = []
        self.adjacency_lookup = utils._get_adjacency_lookup(self.game_info)

        # Optional attributes
        if "extra_turn_fn_idx" in self.game_info.game_state_attributes:
            self.extra_turn_fns = []

    def __default__(self, data, children, meta):
        if len(children) == 1:
            to_return = children[0]
        else:
            to_return = children

        # In order to handle optional arguments, they need
        # to return their name along with their value
        if data.endswith("_arg"):
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

            # Set the "extra turn" flag to FALSE after each move (it's set to TRUE by the effect). This means it will
            # only ever be set to TRUE during the "compute legal actions" step, but that's relevant for the same_piece
            # condition in extra turn effects
            if attribute == "extra_turn_fn_idx":
                addl_info_functions.append(lambda state, action: state._replace(extra_turn_fn_idx=-1))

            if attribute == "promoted":
                addl_info_functions.append(lambda state, action: state._replace(promoted=jnp.zeros(self.game_info.board_size, dtype=jnp.bool_)))

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

        if place_arg_type == OptionalArgs.PATTERN:
            pattern = jnp.array(place_arg_info, dtype=jnp.int16)

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
                result = all_masks.any(axis=0).astype(jnp.int16)
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

        def apply_action_fn(state, action):
            return jax.lax.switch(state.phase_idx, apply_action_fns, state, action)

        def legal_action_mask_fn(state):
            return jax.lax.switch(state.phase_idx, legal_action_mask_fns, state)
        
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
        (action_size, apply_action_fn, legal_action_mask_fn, apply_effects_fn), *force_pass = children
        force_pass = True if len(force_pass) > 0 else False
        
        # The "force_pass" rule adds an additional action which passes the turn to the next player
        # that can only be taken (and must be taken) when no other legal moves are available
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
                mask = jnp.concatenate((base_mask, jnp.array([can_pass], dtype=jnp.int16)))
                return mask
            
            return action_size, new_apply_action_fn, new_legal_action_mask_fn, apply_effects_fn
        
        # TODO: Is this the right place to put this?
        elif "extra_turn_fn_idx" in self.game_info.game_state_attributes:
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
        
        return self.game_info.piece_names.index(piece_ref)

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
            piece_id, mover_ref, (destination_constraint_fn, _), *optional_args = children
            if mover_ref == PlayerAndMoverRefs.MOVER:
                offset = 0
            elif mover_ref == PlayerAndMoverRefs.OPPONENT:
                offset = 1

        # Case 2 (default): the mover is the current player 
        else:
            piece_id, (destination_constraint_fn, _), *optional_args = children
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
        def apply_action_fn(state, action):
            board = state.board.at[action].set((state.current_player + offset) % 2)
            previous_actions = state.previous_actions.at[jnp.array([state.current_player, 2])].set(action)
            return state._replace(board=board, previous_actions=previous_actions)

        return action_size, apply_action_fn, legal_action_mask_fn, apply_effects_fn
    
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
                    previous_actions=state.previous_actions.at[jnp.array([state.current_player, 2])].set(action)
                ))
                return pred_val
            
            apply = jax.jit(apply_action)

            # TODO: maybe this logic should be moved to play_place and actually use
            # the 'apply_action_fn' defined there instead of having a separate one
            def result_constraint_fn(state):
                resulting_mask = jax.vmap(apply, in_axes=(None, 0))(state, jnp.arange(self.game_info.board_size, dtype=jnp.int16))
                return resulting_mask.astype(jnp.int16)

        else:
            result_constraint_fn = predicate_fn_info['lookahead_mask_fn']

        return result_constraint_fn

    '''
    Move rules
    '''
    def play_move(self, children):
        '''
        Combine the move rules for each piece / move type based on their priorities
        '''
        action_size, apply_action_fn, legal_action_infos, apply_effects_fn = children[0]
        grouped_legal_action_infos = groupby(sorted(legal_action_infos, key=lambda x: x[1]), key=lambda x: x[1])

        groups = [list(items) for key, items in grouped_legal_action_infos]

        def build(idx):
            items = groups[idx]
            fns, _ = zip(*list(items))

            collect_same_prio = utils._get_collect_values_fn(fns)
            def single_legal_action_mask_fn(state):
                return collect_same_prio(state).any(axis=0).astype(jnp.int16)
            
            return single_legal_action_mask_fn

        legal_action_mask_fns = list(map(build, range(len(groups))))

        collect_general = utils._get_collect_values_fn(legal_action_mask_fns)
        
        # For legal actions with different priorities, we return the mask of the highest priority
        # with at least one legal action
        def legal_action_mask_fn(state):
            all_masks = collect_general(state)

            any_legal = all_masks.any()
            first_active = all_masks.any(axis=1).argmax()

            mask = jax.lax.select(any_legal, all_masks[first_active], jnp.zeros(action_size, dtype=jnp.int16))

            return mask

        return action_size, apply_action_fn, legal_action_mask_fn, apply_effects_fn

    def play_single_move(self, children):
        '''
        Players move a single kind of piece from one position on the board to another according to some
        rule (e.g. sliding, jumping, ...)
        '''

        piece, base_legal_mask_infos, (source_constraint_fn, _), (destination_constraint_fn, _), *optional_args = children


        if not isinstance(base_legal_mask_infos, list):
            base_legal_mask_infos = [base_legal_mask_infos]

        base_legal_mask_fns, priorities = zip(*base_legal_mask_infos)
        
        # TODO: work in progress detecting hopped pieces. The difficulty is that a move might support
        # multiple move types for a single piece, only some of which involve hopping. How can we determine
        # that a particular action involved a hop?
        if hasattr(self.game_info.game_state_class, 'hopped'):

            # Somewhat hacky, but a hop always moves a piece one of these lengths based on whether
            # it's horizontal, vertical, back-diagonal, or forward-diagonal
            # TODO: confirm this logic always holds
            potential_hop_lengths = jnp.array([
                2, 2 * self.game_info.board_dims[0], 2 * (self.game_info.board_dims[0] + 1),
                2 * (self.game_info.board_dims[0] - 1)
            ], dtype=jnp.int16)


            def move_piece_fn(state, action):
                start_idx, end_idx = action // self.game_info.board_size, action % self.game_info.board_size
                midpoint = (start_idx + end_idx) // 2

                board = state.board.at[piece, end_idx].set(state.current_player)
                board = board.at[piece, start_idx].set(EMPTY)
                previous_actions = state.previous_actions.at[jnp.array([state.current_player, 2])].set(end_idx)

                action_was_hop = jnp.isin(jnp.abs(end_idx - start_idx), potential_hop_lengths) & (state.board[:, midpoint] != EMPTY).any()
                hopped = jax.lax.select(
                    action_was_hop,
                    jnp.zeros_like(state.hopped).at[midpoint].set(True),
                    jnp.zeros_like(state.hopped)
                )

                return state._replace(board=board, previous_actions=previous_actions, hopped=hopped)

        else:
            def move_piece_fn(state, action):
                start_idx, end_idx = action // self.game_info.board_size, action % self.game_info.board_size
                board = state.board.at[piece, end_idx].set(state.current_player)
                board = board.at[piece, start_idx].set(EMPTY)
                previous_actions = state.previous_actions.at[jnp.array([state.current_player, 2])].set(end_idx)
                return state._replace(board=board, previous_actions=previous_actions)

        # Case 1: no optional arguments -- legal actions determined by the destination constraint
        # and there are no effects
        if len(optional_args) == 0:
            result_constraint_fn = lambda state: jnp.ones(self.game_info.board_size, dtype=jnp.int16)
            apply_effects_fn = lambda state, original_player: state

        # Case 2: one optional argument is specified -- either a result constraint or an effect
        elif len(optional_args) == 1:
            arg = optional_args[0]
            if arg.__name__ == "result_constraint_fn" or arg.__name__ == "lookahead_mask_fn":
                result_constraint_fn = lambda state: arg(state)
                apply_effects_fn = lambda state, original_player: state

            elif arg.__name__ == "apply_effects_fn":
                result_constraint_fn = lambda state: jnp.ones(self.game_info.board_size, dtype=jnp.int16)
                apply_effects_fn = arg

        # Case 3: two optional arguments are specified -- result constraints and effects, always
        # in that order
        else:
            result_constraint_fn, apply_effects_fn = optional_args
        
        # We model the action space as "take a piece from any board position and move it to
        # any other board position" so the total number of actions is board_size^2
        action_size = self.game_info.board_size ** 2

        # We define a separate "legal action mask" for seach sub-type of move (e.g. slide, hop) which
        # will ultimately be combined to form the overall legal action mask based on the priorities of
        # each move and piece type
        legal_action_mask_infos = []

        def build(idx):
            base_legal_mask_fn = base_legal_mask_fns[idx]
            priority = priorities[idx]

            def legal_action_mask_fn(state):
                # Construct the meta mask of all legal moves starting at any board position
                base_mask = base_legal_mask_fn(state)

                # Keep only the rows corresponding to valid positions under the source constraint (but keep the shape) and
                # additionally impose constraints based on the piece being moved
                source_mask = source_constraint_fn(state)
                piece_mask = (state.board[piece] == state.current_player).astype(jnp.int16)
                source_mask = source_mask & piece_mask

                base_mask = jnp.where(
                    source_mask[:, jnp.newaxis],
                    base_mask, jnp.zeros_like(base_mask)
                )

                # Keep only the columns corresponding to valid positions under the destination and
                # result constraints (but keep the shape)
                destination_mask = destination_constraint_fn(state) & result_constraint_fn(state)
                base_mask = jnp.where(
                    destination_mask[jnp.newaxis, :],
                    base_mask, jnp.zeros_like(base_mask)
                )

                # Finally, flatten the mask to get the legal action mask
                legal_mask = base_mask.flatten()

                return legal_mask.astype(jnp.int16)
            
            return legal_action_mask_fn, priority
            
        legal_action_mask_infos = list(map(build, range(len(base_legal_mask_fns))))
        
        return action_size, move_piece_fn, legal_action_mask_infos, apply_effects_fn
    
    def play_multi_move(self, children):
        '''
        Players can move different kinds of pieces according to different rules. By construction, any board position
        contains only one piece type which allows us to (for instance) compute the overall legal action mask by taking
        the union of the legal action masks for each piece type.
        '''
        action_sizes, apply_action_fns, legal_action_infos, apply_effects_fns = zip(*children)

        # Combine the legal action information from each piece type
        legal_action_infos = sum(legal_action_infos, [])

        action_size = max(action_sizes)

        def apply_action_fn(state, action):
            start_pos = action // self.game_info.board_size
            piece_idx = state.board[:, start_pos].argmax()
            return jax.lax.switch(piece_idx, apply_action_fns, state, action)
        
        # For move rules, the "previous action" contains the destination position
        def apply_effects_fn(state, original_player):
            piece_idx = state.board[:, state.previous_actions[original_player]].argmax()
            return jax.lax.switch(piece_idx, apply_effects_fns, state, original_player)
        
        return action_size, apply_action_fn, legal_action_infos, apply_effects_fn


    def move_hop(self, children):
        '''
        Move a piece from one position to another by jumping over a piece. Optional arguments can specify a direction,
        a specific piece to hop over, and which player's pieces can be hopped over (by default: "any" for both args)
        '''

        optional_args = self._parse_optional_args(children)

        priority = optional_args[OptionalArgs.PRIORITY]

        piece = optional_args[OptionalArgs.PIECE]
        if piece == PieceRefs.ANY:
            filter_to_piece = lambda occ_mask: occ_mask.any(axis=0)
        else:
            filter_to_piece = lambda occ_mask: occ_mask[piece]

        mover = optional_args[OptionalArgs.MOVER]
        if mover == PlayerAndMoverRefs.BOTH:
            def piece_match_fn(state):
                occupied_mask = (state.board != EMPTY).astype(jnp.int16)
                return filter_to_piece(occupied_mask)
        elif mover == PlayerAndMoverRefs.MOVER:
            def piece_match_fn(state):
                occupied_mask = (state.board == state.current_player).astype(jnp.int16)
                return filter_to_piece(occupied_mask)
        elif mover == PlayerAndMoverRefs.OPPONENT:
            def piece_match_fn(state):
                occupied_mask = (state.board == (state.current_player + 1) % 2).astype(jnp.int16)
                return filter_to_piece(occupied_mask)
        

        direction = optional_args[OptionalArgs.DIRECTION]
        p1_direction_indices, p2_direction_indices = utils._get_direction_indices(self.game_info, direction)
        
        # Hopping is kind of like sliding a distance of 2 except that the middle position needs to be occupied
        # by a specific piece and the end position needs to be empty
        slide_lookup = utils._get_slide_lookup(self.game_info)
        slide_lookup = slide_lookup[:, :, :3]

        def legal_hop_mask_fn(state):
            center_piece_mask = piece_match_fn(state).astype(jnp.int16)
            occupied_mask = (state.board != EMPTY).any(axis=0).astype(jnp.int16)

            direction_indices = jax.lax.select(
                state.current_player == P1,
                p1_direction_indices,
                p2_direction_indices
            )
            slide_indices = slide_lookup[direction_indices, :, :]

            start_indices = slide_indices[:, :, 0]
            middle_indices = slide_indices[:, :, 1]
            dest_indices = slide_indices[:, :, 2]

            inner_match = (center_piece_mask[middle_indices] == 1)
            dest_free = (occupied_mask[dest_indices] == 0)

            legal_hop = inner_match & dest_free
            start_positions = jnp.where(legal_hop, start_indices, self.game_info.board_size+1)
            dest_positions = jnp.where(legal_hop, dest_indices, self.game_info.board_size+1)
            mask = jnp.zeros((self.game_info.board_size, self.game_info.board_size), dtype=jnp.int16)
            mask = mask.at[start_positions, dest_positions].set(1)

            return mask            
        
        return legal_hop_mask_fn, priority

    def move_slide(self, children):
        '''
        Slide a piece in one of the specified directions (default to any) any number of spaces,
        limited by the board boundaries and the non-empty cell encountered
        '''

        optional_args = self._parse_optional_args(children)

        priority = optional_args[OptionalArgs.PRIORITY]

        distance = optional_args[OptionalArgs.DISTANCE]
        if distance is None:
            distance = max(self.game_info.observation_shape[:2])

        direction = optional_args[OptionalArgs.DIRECTION]
        p1_direction_indices, p2_direction_indices = utils._get_direction_indices(self.game_info, direction)
        
        # Restrict the slide lookup to only the specified distance
        slide_lookup = utils._get_slide_lookup(self.game_info)
        slide_lookup = slide_lookup[:, :, :distance+1]

        # Precompute the static indices used in the sliding logic
        actions = jnp.arange(self.game_info.board_size, dtype=jnp.int16)
        general_indices = jnp.indices(slide_lookup[p1_direction_indices, :, :].shape, dtype=jnp.int16)[2]
        ones_array = jnp.ones((len(p1_direction_indices), self.game_info.board_size, 1), dtype=jnp.int16)

        def legal_slide_mask_fn(state):
            occupied_mask = (state.board != EMPTY).any(axis=0).astype(jnp.int16)

            direction_indices = jax.lax.select(
                state.current_player == P1,
                p1_direction_indices,
                p2_direction_indices
            )
            slide_indices = slide_lookup[direction_indices, :, :]

            # Get the occupied mask at the slide indices and pad it with 'occupied' to
            # represent the edge of the board, then find the index of the first occupied cell
            occupied_at_slide = occupied_mask.at[slide_indices].get(mode="fill", fill_value=1)
            occupied_at_slide = occupied_at_slide.at[:, :, 0].set(0)
            
            occupied_at_slide = jnp.concatenate([occupied_at_slide, ones_array], axis=2)
            slide_until_idx = jnp.argmax(occupied_at_slide, axis=2)

            # Extract the board indices corresponding to legal slides, replacing the other
            # indices with a pad value that's larger than the board size
            final_indices = jnp.where(
                general_indices < slide_until_idx[:, :, jnp.newaxis],
                slide_indices, self.game_info.board_size+1
            )

            final_indices = final_indices.transpose((1, 0, 2)).reshape(self.game_info.board_size, -1)

            mask = jnp.zeros((self.game_info.board_size, self.game_info.board_size), dtype=jnp.int16)
            mask = mask.at[actions[:, jnp.newaxis], final_indices].set(1)
            mask = mask.at[actions, actions].set(0)

            return mask

        return legal_slide_mask_fn, priority

    def move_result_constraint(self, children):
        raise NotImplementedError("Move result constraints not implemented yet!")

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
            occupied_mask = (updated_state.board == (original_player + offset) % 2).any(axis=0)

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
        succession, it will seem like neither is getting an extra turn, since the bonus
        turns happen **immediately**
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
        # relying on the fact that the rows of t
        if optional_args[OptionalArgs.SAME_PIECE] and self.game_info.move_type == "move":
            def extra_turn_condition(state, legal_action_mask):
                last_action = state.previous_actions[-1]
                start_mask = jnp.zeros(self.game_info.board_size, dtype=jnp.int16).at[last_action].set(1)[:, jnp.newaxis]
                base_mask = legal_action_mask.reshape(self.game_info.board_size, self.game_info.board_size)
                new_legal_action_mask = jnp.where(start_mask, base_mask, 0).flatten()

                return new_legal_action_mask
        else:
            def extra_turn_condition(state, legal_action_mask):
                return legal_action_mask
            
        self.extra_turn_fns.append(extra_turn_condition)

        def apply_effects_fn(state, original_player):
            return state._replace(phase_step_count=jnp.maximum(state.phase_step_count - 1, 0), current_player=(original_player + offset) % 2,
                                  extra_turn_fn_idx=extra_turn_fn_idx)
        
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
            occupied_mask = (updated_state.board == (original_player + offset) % 2)

            to_flip = (occupied_mask * child_mask).astype(jnp.int16)
            new_board = jnp.where(to_flip, (updated_state.board + 1) % 2, updated_state.board)

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
            idx = (updated_state.current_player + offset) % 2
            amount = amount_fn(updated_state)

            scores = state.scores.at[idx].set(state.scores[idx] + amount)
            return state._replace(scores=scores)
        
        return apply_effects_fn
    
    def effect_promote(self, children):
        '''
        Promote pieces on the board according to a mask and a resulting piece type
        '''
        piece, (child_mask_fn, _), *optional_args = children
        optional_args = self._parse_optional_args(optional_args)

        mover_ref = optional_args[OptionalArgs.MOVER]
        if mover_ref == PlayerAndMoverRefs.MOVER:
            offset = 0
        else:
            offset = 1

        def apply_effects_fn(state, original_player):
            updated_state = state._replace(current_player=original_player)
            child_mask = child_mask_fn(updated_state)
            occupied_mask = (updated_state.board == (original_player + offset) % 2).any(axis=0)

            to_promote = jnp.argwhere(occupied_mask * child_mask, size=self.game_info.board_size, fill_value=self.game_info.board_size+1).flatten()
            new_board = updated_state.board.at[:, to_promote].set(EMPTY)
            new_board = new_board.at[piece, to_promote].set((original_player + offset) % 2)

            return state._replace(board=new_board, promoted=(occupied_mask * child_mask).astype(jnp.bool_))

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
            idx = (updated_state.current_player + offset) % 2
            amount = amount_fn(updated_state)

            scores = state.scores.at[idx].set(amount)
            return state._replace(scores=scores)
        
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
            winners = jax.lax.select(ends.any(), winners_by_rule[jnp.argmax(ends)], EMPTY * jnp.ones(2, jnp.int16))
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
            winner = jax.lax.select(pred_val, get_winner(state), EMPTY * jnp.ones(2, jnp.int16))
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
            base = jnp.ones(2, jnp.int16)
            offset = 0

        else:
            base = jnp.zeros(2, jnp.int16)
            if mover_ref == PlayerAndMoverRefs.MOVER:
                offset = 0
            else:
                offset = 1

        def get_winner(state):
            winners = base.at[(state.current_player + offset) % 2].set(1)
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
            base = jnp.zeros(2, jnp.int16)
            offset = 0

        else:
            base = jnp.ones(2, jnp.int16)
            if mover_ref == PlayerAndMoverRefs.MOVER:
                offset = 0
            else:
                offset = 1

        def get_winner(state):
            winners = base.at[(state.current_player + offset) % 2].set(0)
            return winners
        
        info = {}

        return get_winner, info
    
    def result_draw(self, children):
        '''
        The game ends in a draw
        '''
        def get_winner(state):
            winners = EMPTY * jnp.ones(2, jnp.int16)
            return winners
        
        info = {}

        return get_winner, info
    
    def result_by_score(self, children):
        '''
        The game is won by the player with the highest score (draw if scores are equal)
        '''

        draw_result = EMPTY * jnp.ones(2, jnp.int16)
        p1_win_result = jnp.array([1, 0], dtype=jnp.int16)
        p2_win_result = jnp.array([0, 1], dtype=jnp.int16)

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
            result = all_masks.all(axis=0).astype(jnp.int16)

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
            result = all_masks.any(axis=0).astype(jnp.int16)

            return result
        
        info = {"mask_or_children": children_infos}

        return mask_fn, info
    
    def super_mask_not(self, children):
        '''
        Apply a location-wise boolean NOT operation to the mask of the child
        '''
        child_mask_fn, child_info = children[0]

        def mask_fn(state):
            return (child_mask_fn(state) == 0).astype(jnp.int16)
        
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
            adjacent_mask = (child_mask * local_lookup).any(axis=(0, 2)).astype(jnp.int16)

            return adjacent_mask

        return mask_fn, child_info
    
    def mask_captured(self, children):
        '''
        Return a mask of all the positions on the board which were captured
        in the last turn
        '''
        def mask_fn(state):
            return state.captured.astype(jnp.int16)
        
        return mask_fn, {}

    def mask_center(self, children):
        '''
        Return the center of the board (defined as halfway through the board array). For
        boards with an even number of positions, this will be empty
        '''
        center_idx = self.game_info.board_size // 2 if self.game_info.board_size % 2 == 1 else self.game_info.board_size + 1
        mask = jnp.zeros(self.game_info.board_size, dtype=jnp.int16).at[center_idx].set(1)

        def mask_fn(state):
            return mask

        return mask_fn, {}
    
    def mask_column(self, children):
        '''
        Return a mask that's true for all positions in the specified column
        '''
        column_idx = int(children[0])
        column_indices = utils._get_column_indices(self.game_info, column_idx)
        mask = jnp.zeros(self.game_info.board_size, dtype=jnp.int16).at[column_indices].set(1)

        def mask_fn(state):
            return mask
        
        return mask_fn, {}

    def mask_corners(self, children):
        '''
        Return the corners of the board (the number and location of which depends on the shape)
        '''
        corner_indices = utils._get_corner_indices(self.game_info)
        mask = jnp.zeros(self.game_info.board_size, dtype=jnp.int16)
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
        direction_indices = utils._get_direction_indices(self.game_info, Directions.ORTHOGONAL)

        local_lookup = self.adjacency_lookup[direction_indices].any(axis=0)
        outer_indices = []
        for corner in corner_indices:
            outer_indices.append(jnp.argwhere(local_lookup[corner] == 1).flatten())
        
        outer_indices = jnp.array(outer_indices)
        inner_indices = corner_indices[:, jnp.newaxis]
        full_indices = jnp.concatenate([outer_indices[:, 0:1], inner_indices, outer_indices[:, 1:]], axis=1)
        full_match_width = 3

        def mask_fn(state):
            outer_player = (state.current_player + offset) % 2
            inner_player = (outer_player + 1) % 2
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
            mask = jnp.zeros(self.game_info.board_size, dtype=jnp.int16).at[matched_indices].set(1)

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
                line_indices = jnp.array([], dtype=jnp.int16)

        else:
            n = int(n)
            line_indices = utils._get_line_indices(self.game_info, n+2, orientation)
            inner_indices, outer_indices = utils._get_custodial_indices(self.game_info, n, orientation)

        # If there aren't any valid custodial arrangements, we can just return an empty mask
        if line_indices.shape[0] == 0:
            def mask_fn(state):
                return jnp.zeros(self.game_info.board_size, dtype=jnp.int16)
            
            def lookahead_mask_fn(state):
                return jnp.zeros(self.game_info.board_size, dtype=jnp.int16)
            
            return mask_fn, {"lookahead_mask_fn": lookahead_mask_fn}
        
        full_match_width = line_indices.shape[1]
        
        def mask_fn(state):
            outer_player = (state.current_player + offset) % 2
            inner_player = (outer_player + 1) % 2
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
            mask = jnp.zeros(self.game_info.board_size, dtype=jnp.int16).at[matched_indices].set(1)

            return mask
        
        # The lookahead mask version looks for lines that have all of the inner indices set and
        # exactly one of the outer indices set -- "left" matches have the first outer index set 
        # and the second empty, and "right" matches are the reverse
        left_indices = outer_indices[:, 0]
        right_indices = outer_indices[:, 1]

        def lookahead_mask_fn(state):
            outer_player = (state.current_player + offset) % 2
            inner_player = (outer_player + 1) % 2

            outer_mask = (state.board[piece] == outer_player)
            inner_mask = (state.board[piece] == inner_player)
            
            inner_match = (inner_mask[inner_indices] == 1).all(axis=1)
            left_match = (outer_mask[left_indices] == 1) & (outer_mask[right_indices] == 0) & inner_match
            right_match = (outer_mask[right_indices]  == 1) & (outer_mask[left_indices] == 0) & inner_match

            left_match_indices = jnp.where(left_match, right_indices, self.game_info.board_size+1)
            right_match_indices = jnp.where(right_match, left_indices, self.game_info.board_size+1)

            mask = jnp.zeros(self.game_info.board_size, dtype=jnp.int16)
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
                return mask.astype(jnp.int16)

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
                return mask.astype(jnp.int16)
        
        return mask_fn, {}

    def mask_empty(self, children):
        '''
        Return all the positions on the board which are empty of all piece types
        '''
        def mask_fn(state):
            return (state.board == EMPTY).all(axis=0).astype(jnp.int16)
        
        return mask_fn, {}

    def mask_hopped(self, children):
        '''
        Return the positions on the board which were hopped over
        in the last move
        '''
        
        def mask_fn(state):
            return state.hopped.astype(jnp.int16)
        
        return mask_fn, {}

    def mask_occupied(self, children):
        '''
        Return the positions on the board which are occupied
        by the specified player. If no player is specified,
        then the mask checks for either player
        '''

        if len(children) == 0:
            def mask_fn(state):
                return (state.board != EMPTY).any(axis=0).astype(jnp.int16)
            
            return mask_fn, {}
        
        mover_ref = children[0]
        
        if mover_ref == PlayerAndMoverRefs.MOVER:
            offset = 0
        elif mover_ref == PlayerAndMoverRefs.OPPONENT:
            offset = 1

        def mask_fn(state):
            return (state.board == ((state.current_player + offset)%2)).any(axis=0).astype(jnp.int16)
        
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
            prev_move = state.previous_actions[(state.current_player + offset) % 2]
            mask = jax.lax.select(prev_move != -1, mask.at[prev_move].set(True), mask)

            return mask.astype(jnp.int16)
        
        return mask_fn, {}
    
    def mask_promoted(self, children):
        '''
        Return a mask of all the positions on the board which were promoted
        in the last turn
        '''
        def mask_fn(state):
            return state.promoted.astype(jnp.int16)
        
        return mask_fn, {}

    def mask_row(self, children):
        '''
        Return a mask that's true for all positions in the specified row
        '''
        row_idx = int(children[0])
        row_indices = utils._get_row_indices(self.game_info, row_idx)
        mask = jnp.zeros(self.game_info.board_size, dtype=jnp.int16).at[row_indices].set(1)

        def mask_fn(state):
            return mask
        
        return mask_fn, {}

    '''
    ==========Multi-masks==========
    '''
    def _get_static_mask(self, indices):
        mask = jnp.zeros(self.game_info.board_size, dtype=jnp.int16)
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
                return (mask == 0).astype(jnp.int16)

            info['lookahead_mask_fn'] = lookahead_mask_fn

        return predicate_fn, info

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
    
    def predicate_can_hop(self, children):
        '''
        This is somewhat horrific, but the intent is to be able to check
        whether there are any legal hop moves available starting from the
        given mask (e.g. for games like checkers where an extra turn is only
        granted if the capturing piece can continue hopping)

        TODO: clean this up...
        '''
        (child_mask_fn, _), *optional_args = children
        optional_args = self._parse_optional_args(optional_args)

        piece = optional_args[OptionalArgs.PIECE]
        if piece == PieceRefs.ANY:
            filter_to_piece = lambda occ_mask: occ_mask.any(axis=0)
        else:
            filter_to_piece = lambda occ_mask: occ_mask[piece]

        mover = optional_args[OptionalArgs.MOVER]
        if mover == PlayerAndMoverRefs.BOTH:
            def piece_match_fn(state):
                occupied_mask = (state.board != EMPTY).astype(jnp.int16)
                return filter_to_piece(occupied_mask)
        elif mover == PlayerAndMoverRefs.MOVER:
            def piece_match_fn(state):
                occupied_mask = (state.board == state.current_player).astype(jnp.int16)
                return filter_to_piece(occupied_mask)
        elif mover == PlayerAndMoverRefs.OPPONENT:
            def piece_match_fn(state):
                occupied_mask = (state.board == (state.current_player + 1) % 2).astype(jnp.int16)
                return filter_to_piece(occupied_mask)
        

        direction = optional_args[OptionalArgs.DIRECTION]
        p1_direction_indices, p2_direction_indices = utils._get_direction_indices(self.game_info, direction)
        
        # Hopping is kind of like sliding a distance of 2 except that the middle position needs to be occupied
        # by a specific piece and the end position needs to be empty
        slide_lookup = utils._get_slide_lookup(self.game_info)
        slide_lookup = slide_lookup[:, :, :3]

        def legal_hop_mask_fn(state):
            center_piece_mask = piece_match_fn(state).astype(jnp.int16)
            occupied_mask = (state.board != EMPTY).any(axis=0).astype(jnp.int16)

            direction_indices = jax.lax.select(
                state.current_player == P1,
                p1_direction_indices,
                p2_direction_indices
            )
            slide_indices = slide_lookup[direction_indices, :, :]

            start_indices = slide_indices[:, :, 0]
            middle_indices = slide_indices[:, :, 1]
            dest_indices = slide_indices[:, :, 2]

            inner_match = (center_piece_mask[middle_indices] == 1)
            dest_free = (occupied_mask[dest_indices] == 0)

            legal_hop = inner_match & dest_free
            start_positions = jnp.where(legal_hop, start_indices, self.game_info.board_size+1)
            dest_positions = jnp.where(legal_hop, dest_indices, self.game_info.board_size+1)
            mask = jnp.zeros((self.game_info.board_size, self.game_info.board_size), dtype=jnp.int16)
            mask = mask.at[start_positions, dest_positions].set(1)

            return mask
        
        def predicate_fn(state):
            child_mask = child_mask_fn(state)
            legal_hop_mask = legal_hop_mask_fn(state).any(axis=1)

            # Check whether there are any legal hops starting from the given mask
            masked_legal_hops = legal_hop_mask * child_mask
            return masked_legal_hops.any()
        
        info = {}

        return predicate_fn, info

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
                return (mask >= 1).astype(jnp.int16)

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
                return (mask >= 1).astype(jnp.int16)

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
                return (mask_left >= mask_right).astype(jnp.int16)

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
                return (mask_left <= mask_right).astype(jnp.int16)

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
            return jnp.ones(self.game_info.board_size, dtype=jnp.int16) * value
        
        info = {"lookahead_mask_fn": lookahead_mask_fn}

        return lambda state: value, info
    
    def function_connected(self, children):
        '''
        Return the number of the specified masks which are connected to each other
        via the last move made by the current player.

        NOTE: to simplify syntax, it's possible to specify "multi-masks" (see above)
        instead of manually enumerating each of the target masks
        '''
        piece, (_, target_masks_infos), *optional_args = children
        optional_args = self._parse_optional_args(optional_args)

        # Parse out the target mask functions
        target_mask_fns, _ = zip(*target_masks_infos)
        num_targets = len(target_mask_fns)

        if optional_args[OptionalArgs.MOVER] == PlayerAndMoverRefs.MOVER:
            offset = 0
        else:
            offset = 1

        def function_fn(state):
            mover = (state.current_player + offset) % 2
            target_masks = jax.vmap(lambda i: jax.lax.switch(i, target_mask_fns, state))(jnp.arange(num_targets))

            # The connected components are necessarily computed in the update_additional_info call
            set_val = state.previous_actions[mover] + 1
            fill_mask = (state.connected_components[piece] == set_val).astype(jnp.int16)
            
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
        be read from only the function value
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
                exclude_mask = collect_values(state).any(axis=0).astype(jnp.int16)
                return occupied_mask & ~exclude_mask
            
        else:
            def get_occupied_mask(state):
                return base_mask_fn(state)


        if optional_args[OptionalArgs.EXACT] and n < max(self.game_info.board_dims):
            line_indices = utils._get_line_indices(self.game_info, n, orientation)

            # If there are no valid lines of this length / orientation, then always return 0
            if line_indices.size == 0:
                return lambda state: 0, {}

            overshoot_line_indices = utils._get_line_indices(self.game_info, n+1, orientation)

            def function_fn(state):
                occupied_mask = get_occupied_mask(state)
                line_matches = (occupied_mask[line_indices] == 1).all(axis=1)
                num_lines = line_matches.sum()
                num_overshoot_lines = jax.lax.cond(num_lines, lambda: 0, lambda: (occupied_mask[overshoot_line_indices] == 1).all(axis=1).sum())

                return num_lines - 2 * num_overshoot_lines
            
            def mask_fn(state):
                occupied_mask = get_occupied_mask(state)
                line_matches = (occupied_mask[line_indices] == 1).all(axis=1)
                overshoot_line_matches = (occupied_mask[overshoot_line_indices] == 1).all(axis=1)

                matched_indices = jnp.where(line_matches[:, jnp.newaxis], line_indices, self.game_info.board_size+1).flatten()
                overshoot_matched_indices = jnp.where(overshoot_line_matches[:, jnp.newaxis], overshoot_line_indices, self.game_info.board_size+1).flatten()
                
                mask = jnp.zeros(self.game_info.board_size, dtype=jnp.int16).at[matched_indices].set(1)
                mask = mask.at[overshoot_matched_indices].set(0)

                return mask

            info = {"mask_fn": mask_fn}

        else:
            line_indices = utils._get_line_indices(self.game_info, n, orientation)

            # If there are no valid lines of this length / orientation, then always return 0
            if line_indices.size == 0:
                return lambda state: 0, {"lookahead_mask_fn": lambda state: jnp.zeros(self.game_info.board_size, dtype=jnp.int16)}

            def function_fn(state):
                occupied_mask = get_occupied_mask(state)
                line_matches = (occupied_mask[line_indices] == 1).all(axis=1)
                return line_matches.sum()
            
            def mask_fn(state):
                occupied_mask = get_occupied_mask(state)
                line_matches = (occupied_mask[line_indices] == 1).all(axis=1)

                matched_indices = jnp.where(line_matches[:, jnp.newaxis], line_indices, self.game_info.board_size+1).flatten()
                mask = jnp.zeros(self.game_info.board_size, dtype=jnp.int16).at[matched_indices].set(1)

                return mask
            
            num_lines = line_indices.shape[0]
            def lookahead_mask_fn(state):
                occupied_mask = get_occupied_mask(state)

                line_mask = (occupied_mask[line_indices] == 1)
                almost_line_mask = (line_mask.sum(axis=1) == n-1)

                arange = jnp.arange(num_lines).astype(jnp.int16)
                missing_line_indices = jnp.argmin(line_mask, axis=1).astype(jnp.int16)
                missing_positions = jnp.where(almost_line_mask, line_indices[arange, missing_line_indices], self.game_info.board_size+1)
                unique_positions, counts = jnp.unique_counts(missing_positions, size=num_lines, fill_value=self.game_info.board_size+1)

                unique_positions = unique_positions.astype(jnp.int16)
                counts = counts.astype(jnp.int16)

                mask = jnp.zeros(self.game_info.board_size, dtype=jnp.int16)
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
            score = state.scores[(state.current_player + offset) % 2]
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