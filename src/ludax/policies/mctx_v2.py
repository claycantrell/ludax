from functools import partial

import jax
import jax.numpy as jnp
import mctx

from ludax.environment import LudaxEnvironment, ACTION_DTYPE, REWARD_DTYPE

class MCTSPolicy():
    '''
    A simple, no-heuristic MCTS policy for Ludax based on https://github.com/bcorfman/pgx-connectfour/blob/main/src/model.py
    '''
    def __init__(self, env: LudaxEnvironment, batch_size: int = 1, num_simulations: int = 1000, max_depth: int = 10):
        self.env = env
        self.batch_size = batch_size
        self.num_simulations = num_simulations
        self.max_depth = max_depth

        self.env_step = jax.jit(env.step)
        self.env_init = jax.jit(env.init)

    def policy_fn(self, state):
        '''
        Return the logits of a random policy
        '''
        logits = jnp.where(state.legal_action_mask, 0.0, -jnp.inf)
        return logits
    
    def value_fn(self, state, key: jax.random.PRNGKey):
        '''
        Returns the value of a state by performing a random rollout from the perspective of the current player
        '''
        def cond_fn(carry):
            state, key = carry
            return ~(state.terminated | state.truncated).all()
        
        def body_fn(carry):
            state, key = carry

            key, subkey = jax.random.split(key)
            logits = jnp.log(state.legal_action_mask.astype(REWARD_DTYPE))
            action = jax.random.categorical(subkey, logits=logits, axis=1).astype(ACTION_DTYPE)
            next_state = self.env_step(state, action)
            return next_state, key
        
        current_player = state.game_state.current_player
        state, _ = jax.lax.while_loop(cond_fn, body_fn, (state, key))
        return state.rewards[current_player]

    def recurrent_fn(self, params, key, action, state):
        del params

        current_player = state.game_state.current_player
        next_state = self.env_step(state, action)
        logits = self.policy_fn(next_state)
        reward = next_state.rewards[current_player]
        value = jax.lax.select(next_state.terminated, 0.0, self.value_fn(next_state, key))
        discount = jax.lax.select(next_state.terminated, 0.0, -1.0)

        recurrent_fn_output = mctx.RecurrentFnOutput(
            reward=reward,
            discount=discount,
            prior_logits=logits,
            value=value,
        )

        return recurrent_fn_output, next_state
    
    def run(self, state, key):

        key, subkey = jax.random.split(key)
        keys = jax.random.split(subkey, self.batch_size)
        key, subkey = jax.random.split(key)

        root = mctx.RootFnOutput(
            prior_logits=jax.vmap(self.policy_fn)(state),
            value=jax.vmap(self.value_fn)(state, keys),
            embedding=state
        )

        policy_output = mctx.muzero_policy(
            params=None,
            rng_key=subkey,
            root=root,
            invalid_actions=~state.legal_action_mask,
            recurrent_fn=jax.vmap(self.recurrent_fn, in_axes=(None, None, 0, 0)),
            num_simulations=self.num_simulations,
            max_depth=self.max_depth,
            qtransform=partial(mctx.qtransform_by_min_max, min_value=-1.0, max_value=1.0),
            dirichlet_fraction=0.25
        )

        return policy_output.action
    
    def __call__(self, state, key):
        return self.run(state, key)

if __name__ == "__main__":
    from ludax.games import tic_tac_toe, english_draughts
    env = LudaxEnvironment(game_str=tic_tac_toe.replace("square 3", "square 4"))
    init = jax.jit(env.init)

    policy = MCTSPolicy(env)
    key = jax.random.PRNGKey(0)
    key, subkey = jax.random.split(key)
    state = jax.vmap(policy.env_init)(jax.random.split(subkey, policy.batch_size))


    while ~(state.terminated | state.truncated).all():
        print(state.game_state.board[0].reshape(4, 4))
        policy_output = policy(state, key)
        print("Selected action:", policy_output)
        state = jax.vmap(policy.env_step)(state, policy_output)
        # print("Rewards:", state.rewards)