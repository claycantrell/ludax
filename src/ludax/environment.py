from typing import Optional

import jax
import jax.numpy as jnp
from lark import Lark

import ludax
from .config import Array, PRNGKey, State, EMPTY, TRUE, MAX_STEP_COUNT
from .game_info import GameInfoExtractor
from .game_parser import GameRuleParser


class LudaxEnvironment():
    def __init__(self,
                 game_path: str = None,
                 game_str: str = None):
        super().__init__()

        assert game_path is not None or game_str is not None, "Must provide either a game path or a game string!"
        assert game_path is None or game_str is None, "Cannot provide both a game path and a game string!"
        
        if game_path is not None:
            with open(game_path, 'r') as f:
                game_str = f.read()

        parser = Lark(ludax.grammar, start='game')

        game_tree = parser.parse(game_str)
        self.game_info, self.rendering_info = GameInfoExtractor()(game_tree)
        game_rules = GameRuleParser(self.game_info).transform(game_tree)

        self.game_state_cls = self.game_info.game_state_class

        self.obs_shape = self.game_info.observation_shape
        self.board_size = self.game_info.board_size
        self.num_piece_types = self.game_info.num_piece_types

        self.num_actions = game_rules['action_size']
        self._initialize_board = game_rules['start_rules']
        self._apply_action = game_rules['apply_action_fn']
        self._update_info = game_rules['addl_info_fn']
        self._get_legal_action_mask = game_rules['legal_action_mask_fn']
        self._apply_effects = game_rules['apply_effects_fn']
        self._get_phase_idx = game_rules['next_phase_fn']
        self._get_next_player = game_rules['next_player_fn']

        self._get_winner = game_rules['end_rules']
        self._get_terminal = lambda board, player: (board != EMPTY).all()

    def init(self, rng: PRNGKey) -> State:

        # Temporarily hard-coding the init of the game state
        temp_current_player = jnp.int8(0)
        game_state = self.game_state_cls(
            board=jnp.ones((self.num_piece_types, self.board_size), dtype=jnp.int8) * EMPTY,
            legal_action_mask=jnp.ones((self.num_actions,), dtype=jnp.bool_),
            current_player=temp_current_player,
            phase_idx=jnp.int8(0),
            phase_step_count=jnp.int8(0),
            previous_actions=jnp.int8([-1, -1, -1]),
        )

        # Initialize the board using the game rules
        game_state = self._initialize_board(game_state)

        # Compute the actual starting player
        current_player = self._get_next_player(game_state)
        game_state = game_state._replace(current_player=current_player)

        game_state = self._update_info(game_state, -1)

        legal_action_mask = self._get_legal_action_mask(game_state).astype(jnp.bool_)
        game_state = game_state._replace(legal_action_mask=legal_action_mask)

        state = State(
            game_state=game_state,
            legal_action_mask=legal_action_mask,
            current_player=current_player
        )

        return state

    def step(self, state: State, action: Array, key: Optional[Array] = None) -> State:
        """
        The basic environment step function, largely taken from the PGX implementation
        (https://github.com/sotetsuk/pgx/blob/main/pgx/core.py) and used under the
        Apache-2.0 license.
        """
        is_illegal = ~state.legal_action_mask[action]
        current_player = state.current_player

        # If the state is already terminated or truncated, environment does not take usual step,
        # but return the same state with zero-rewards for all players
        state = jax.lax.cond(
            (state.terminated | state.truncated),
            lambda: state.replace(rewards=jnp.zeros_like(state.rewards)),  # type: ignore
            lambda: self._step(state, action, key),  # type: ignore
        )

        # Taking illegal action leads to immediate game terminal with negative reward
        state = jax.lax.cond(
            is_illegal,
            lambda: self._step_with_illegal_action(state, current_player),
            lambda: state,
        )

        # All legal_action_mask elements are **TRUE** at terminal state
        # This is to avoid zero-division error when normalizing action probability
        # Taking any action at terminal state does not give any effect to the state
        state = jax.lax.cond(
            state.terminated,
            lambda: state.replace(legal_action_mask=jnp.ones_like(state.legal_action_mask)),  # type: ignore
            lambda: state,
        )

        # NOTE: unlike PGX, we don't compute the observation inside the step function, instead leaving it
        # to, for example, the RL training loop using env._observe(state, player_id)

        return state

    def _step(self, state: State, action: Array, key) -> State:
        # Currently, Ludax games don't have any stochastic elements, so we don't use the key
        del key

        # The game state is separate from the "environment state" and contains
        # only the information necessary to progress the game
        game_state = state.game_state

        # Track the player who started the turn
        original_player = game_state.current_player

        # Compute the new board state
        game_state = self._apply_action(game_state, action)

        # Compute any 'additional information' required by later steps. For instance, connected
        # components
        game_state = self._update_info(game_state, action)

        # Compute the new phase index
        new_phase_idx, phase_step_count = self._get_phase_idx(game_state)
        game_state = game_state._replace(phase_idx=new_phase_idx, phase_step_count=phase_step_count)

        # Use the new phase and the global player offset to determine the next player
        next_player = self._get_next_player(game_state)
        game_state = game_state._replace(current_player=next_player)

        # Apply any effects that occur at the end of the turn. The effects functions take in the
        # original player so they can correctly handle "extra turns" given to the mover / opponent
        game_state = self._apply_effects(game_state, original_player)

        # Compute the legal action mask for the upcoming player (which is used in some end conditions)
        # TODO: should this be stored separately for each player?
        new_legal_action_mask = self._get_legal_action_mask(game_state).astype(jnp.bool_)
        game_state = game_state._replace(legal_action_mask=new_legal_action_mask)
        state = state.replace(legal_action_mask=new_legal_action_mask)

        # Use the new board to compute the winner, terminal, and rewards -- but consider the
        # current player to be the "original" player so they get credit for winning on their turn
        winners, terminated = self._get_winner(game_state._replace(current_player=original_player))
        terminal_rewards = jnp.where((winners == EMPTY).all(), jnp.zeros_like(winners), jnp.where(winners, 1, -1)).astype(jnp.float32)
        rewards = jax.lax.select(terminated, terminal_rewards, jnp.zeros_like(terminal_rewards))

        mover_reward = rewards[original_player]
        state = state.replace(winners=winners, rewards=rewards, terminated=terminated, mover_reward=mover_reward)

        # If the terminated, then reset the current player back to the original player so that
        # "mover reward" refers to the correct player
        game_state = jax.lax.cond(
            terminated,
            lambda: game_state._replace(current_player=original_player),
            lambda: game_state
        )

        # Update the game state in the state
        state = state.replace(game_state=game_state, current_player=game_state.current_player)

        # Increment the global step count
        state = state.replace(global_step_count=state.global_step_count + 1, truncated=state.global_step_count >= MAX_STEP_COUNT)

        return state
    
    def observe(self, state: State, player_id: Array) -> Array:
        """
        Convert a flat board with values in {-1, 0, 1} symbolizing the current piece type into a boolean (rows, cols, 2) tensor
        board[i] == -1  → empty square
        board[i] == 0   → square occupied by white (current player or not)
        board[i] == 1   → square occupied by black (current player or not)
        observation[:, :, 0] == True  → squares occupied by the current player (black or white)
        observation[:, :, 1] == True  → squares occupied by the other player (black or white)
        """
        board = state.game_state.board
        observation_shape = self.game_info.observation_shape

        board2d = board.reshape(*board.shape[:-1], observation_shape[-3], observation_shape[-2])
        player_id = player_id[..., None, None]

        obs = jnp.stack((board2d == player_id, board2d == jnp.abs(1 - player_id)), axis=-1)

        return jax.lax.stop_gradient(obs)  # Taken from PGX

    @property
    def _illegal_action_penalty(self) -> float:
        """
        Negative reward given when illegal action is selected
        """
        return -1.0
    
    def _step_with_illegal_action(self, state: State, loser: Array) -> State:
        penalty = self._illegal_action_penalty
        reward = jnp.ones_like(state.rewards) * (-1 * penalty) * (self.game_info.num_players - 1)
        reward = reward.at[loser].set(penalty)
        return state.replace(rewards=reward, terminated=TRUE)  # type: ignore