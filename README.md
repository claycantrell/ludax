# Ludax
Ludax is a domain specific language for board games that compiles into hardware-accelerated learning environments using [JAX](https://github.com/jax-ml/jax). Ludax draws inspiration from the [Ludii](https://ludii.games/index.php) game description language as well as [PGX](https://github.com/sotetsuk/pgx), a library of JAX implementations for classic board games and video games. Ludax supports a variety of two-player perfect-information board games and can run at tens of millions of steps per second on modern GPUs.

![Throughput of Ludax environments compared to PGX and Ludii implementations](/assets/throughput_comparison.png)

## Installation
> [!IMPORTANT]
> Ludax requires a Python version of at least `3.9`.
> We recommend first installing the JAX library (see [here](https://docs.jax.dev/en/latest/installation.html) for instructions) and then installing Ludax, otherwise JAX will run on the CPU instead of your accelerator.

### Package Installation
To install Ludax as a pip package, run
```bash
pip install ludax[gui]
```

> [!TIP]
> This will install the Ludax package along with the optional GUI dependencies. If you do not need the GUI, you can install it without the `[gui]` option.

### Development Installation
To try out the example scripts in this repository or to contribute to the Ludax codebase, you can clone the repository and install the dependencies using:
```
pip install -r requirements-dev.txt
```

## Basic Usage
To instantiate an environment in Ludax, you pass in the path to grammatically-valid `.ldx` file (see `grammar.lark` for syntax details). The general environment API is very similar to PGX and gymnax:
```python
import jax
import jax.numpy as jnp

from ludax import LudaxEnvironment


GAME_PATH = "games/tic_tac_toe.ldx"
BATCH_SIZE = 1024

env = LudaxEnvironment(GAME_PATH)
init = jax.jit(jax.vmap(env.init))
step = jax.jit(jax.vmap(env.step))


def _run_batch(state, key):
    def cond_fn(args):
        state, _ = args
        return ~(state.terminated | state.truncated).all()

    def body_fn(args):
        state, key = args
        key, subkey = jax.random.split(key)
        logits = jnp.log(state.legal_action_mask.astype(jnp.float32))
        action = jax.random.categorical(key, logits=logits, axis=1)
        state = step(state, action)
        return state, key

    state, key = jax.lax.while_loop(cond_fn, body_fn, (state, key))

    return state, key


run_batch = jax.jit(_run_batch)

key = jax.random.PRNGKey(42)
key, subkey = jax.random.split(key)
keys = jax.random.split(subkey, BATCH_SIZE)

state = init(keys)
state, key = run_batch(state, key)
print(f"Winner (0: first player, 1: second player, -1: draw): {state.winner}")
```

## Comparisons
To generate comparisons against Ludii and PGX, run `compare_implementations.py` with the appropriate command-line arguments. For instance, to compare on Tic-Tac-Toe on batch sizes of 1 to 1024, you would run
```
python examples/figures/compare_implementations.py --game tic_tac_toe --batch_size_step 2 --num_batch_sizes 11
```

## Reinforcement Learning
We provide a demonstration of using the PGX AlphaZero implementation to train agents in the Ludax implementation in `pgx_alphazero/train.py`.

## Interactive Mode
To play a game interactively, run `python examples/ludax_gui/interactive.py`. This will launch an app on your local host on port 8080. After running the command, navigate to [http://127.0.0.1:8080](http://127.0.0.1:8080) and you will see the list games currently in the `games/` directory. Navigating to any of the links will let you playtest the game in the browser by clicking on a square to make your move.