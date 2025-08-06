from ludax import LudaxEnvironment
import jax
import jax.numpy as jnp
import random

env = LudaxEnvironment(game_path="games/tic_tac_toe.ldx")
init = jax.jit(env.init)
step = jax.jit(env.step)

seed = random.randint(0, 1000)
key = jax.random.PRNGKey(seed)
key, init_key = jax.random.split(key)
state = init(init_key)

print(f"Playing with seed {seed}...")

i = 0
while not bool(state.terminated):
    i += 1
    print(f"Step {i}")

    # Sample an action from legal moves
    key, subkey = jax.random.split(key)
    logits = jnp.log(state.legal_action_mask.astype(jnp.float32))  # -inf for illegal
    action = jax.random.categorical(subkey, logits=logits, axis=0).astype(jnp.int16)

    # Step the environment
    state = step(state, action)

print("Game over!")

print(f"Winner (0: first player, 1: second player, -1: draw): {state.winner}")
