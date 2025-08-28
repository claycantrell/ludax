# ttt_selfplay_save_trees.py
# ---------------------------------------------------------------
# Self-play tic-tac-toe with MCTX/Gumbel MuZero and tree logging.
# Saves a search tree artifact at EVERY move.
#
# Requires:
#   - jax, jaxlib
#   - mctx (DeepMind's mctx)
#   - ludax (your environment)
# Optional:
#   - pygraphviz + Graphviz "dot" binary (for SVG output)
#     Without it, falls back to saving .npz tensors.
# ---------------------------------------------------------------

from functools import partial
import os
import itertools
from typing import Optional, Sequence

import jax
import jax.numpy as jnp
import numpy as np
import mctx

from ludax import LudaxEnvironment
from ludax.games import tic_tac_toe

# ---------- Pretty printing ----------
np.set_printoptions(threshold=np.inf, linewidth=np.inf)
jax.numpy.set_printoptions(threshold=np.inf, linewidth=np.inf)

# ---------- Optional Graphviz output ----------
try:
    import pygraphviz  # type: ignore
    _HAS_PYGRAPHVIZ = True
except Exception:
    _HAS_PYGRAPHVIZ = False


# ---------- Helpers: make a batch env ----------
def initialize(env: LudaxEnvironment, batch_size: int = 1, seed: int = 0):
    """Initialize batched env state and step function."""
    init_b = jax.jit(jax.vmap(env.init))
    step_b = jax.jit(jax.vmap(env.step))
    key = jax.random.PRNGKey(seed)
    key, init_key = jax.random.split(key)
    state_b = init_b(jax.random.split(init_key, batch_size))
    return state_b, step_b, key


# ---------- Heuristic & priors ----------
WINNING_LOGIT = 300.0
LOSING_LOGIT = 200.0
LEGAL_LOGIT = 100.0
NEG_INF = jnp.array(-jnp.inf, dtype=jnp.float32)


def _zero_heuristic_builder(_step_b):
    def h(state, key=None):
        del key
        # shape: [batch]
        return jnp.zeros(state.legal_action_mask.shape[0], dtype=jnp.float32)
    return h


@partial(jax.jit, static_argnames=['step_b'])
def one_ply_logits(step_b, state_b, root_player_b):
    """
    One-ply lookahead logits (root perspective).
    """
    batch_size, num_actions = state_b.legal_action_mask.shape

    all_actions = jnp.broadcast_to(jnp.arange(num_actions), (batch_size, num_actions))
    actions_flat = all_actions.reshape(-1)

    state_flat = jax.tree_util.tree_map(lambda x: jnp.repeat(x, num_actions, axis=0), state_b)
    root_player_flat = jnp.repeat(root_player_b, num_actions, axis=0)

    next_state_flat = step_b(state_flat, actions_flat.astype(jnp.int16))

    root_rewards_flat = next_state_flat.rewards[
        jnp.arange(next_state_flat.rewards.shape[0]),
        root_player_flat
    ]

    logits_flat = jnp.full(root_rewards_flat.shape, LEGAL_LOGIT, dtype=jnp.float32)
    logits_flat = jnp.where(root_rewards_flat < 0, LOSING_LOGIT, logits_flat)
    logits_flat = jnp.where(root_rewards_flat > 0, WINNING_LOGIT, logits_flat)

    legal = state_b.legal_action_mask & (~state_b.terminated[:, None])
    legal_flat = legal.reshape(-1)
    logits_flat = jnp.where(legal_flat, logits_flat, NEG_INF)

    return logits_flat.reshape(batch_size, num_actions).astype(jnp.float32)


def ludax_recurrent(root_player_b, step_b, heuristic):
    def recurrent_fn(params, rng_key, action, state):
        key, sub_key = jax.random.split(rng_key)
        next_state = step_b(state, action.astype(jnp.int16))

        r = next_state.rewards[jnp.arange(next_state.rewards.shape[0]), root_player_b]
        l = one_ply_logits(step_b, next_state, root_player_b)
        l = jnp.where(next_state.terminated[:, None], jnp.full_like(l, NEG_INF), l)


        to_play = next_state.game_state.current_player
        sign = jnp.where(to_play == root_player_b, 1.0, -1.0)
        v = sign * heuristic(next_state, sub_key)

        v = jnp.where(next_state.terminated, 0, v)
        d = jnp.where(next_state.terminated, 0.0, 1.0)

        out = mctx.RecurrentFnOutput(
            reward=r,
            discount=d,
            prior_logits=l,
            value=v,
        )
        return out, next_state
    return jax.jit(recurrent_fn)


# ---------- Tree → Graph / Saver ----------
def convert_tree_to_graph(
    tree: mctx.Tree,
    action_labels: Optional[Sequence[str]] = None,
    batch_index: int = 0,
):
    """Converts a search tree into a Graphviz graph (requires pygraphviz)."""
    if not _HAS_PYGRAPHVIZ:
        raise RuntimeError("pygraphviz not available; cannot build graph.")

    batch_size = tree.node_values.shape[0]
    if action_labels is None:
        action_labels = list(range(tree.num_actions))
    elif len(action_labels) != tree.num_actions:
        raise ValueError(
            f"action_labels has {len(action_labels)} entries; expected {tree.num_actions}"
        )

    def node_to_str(node_i, reward, discount):
        return (f"{node_i}\n"
                f"Reward: {reward}\n"
                f"Discount: {discount}\n"
                f"Value: {tree.node_values[batch_index, node_i]}\n"
                f"Visits: {tree.node_visits[batch_index, node_i]}\n")

    def edge_to_str(node_i, a_i):
        node_index = jnp.full([batch_size], node_i)
        probs = jax.nn.softmax(tree.children_prior_logits[batch_index, node_i])
        return (f"{action_labels[a_i]}\n"
                f"Q: {tree.qvalues(node_index)[batch_index, a_i]}\n"
                f"p: {probs[a_i]}\n")

    graph = pygraphviz.AGraph(directed=True)
    graph.add_node(0, label=node_to_str(0, 0,0), color="green")

    for node_i in range(tree.num_simulations):
        for a_i in range(tree.num_actions):
            child_i = int(tree.children_index[batch_index, node_i, a_i])
            if child_i >= 0:
                discount = float(tree.children_discounts[batch_index, node_i, a_i])
                is_terminal = discount == 0.0

                graph.add_node(
                    child_i,
                    label=node_to_str(
                        child_i,
                        reward=float(tree.children_rewards[batch_index, node_i, a_i]),
                        discount=discount,
                    ),
                    color="blue" if is_terminal else "red",
                )
                graph.add_edge(node_i, child_i, label=edge_to_str(node_i, a_i))
    return graph


def save_search_tree(
    tree: mctx.Tree,
    out_path_noext: str,
    action_labels: Optional[Sequence[str]] = None,
    batch_index: int = 0,
):
    """Save tree to SVG if pygraphviz available; else save raw arrays to .npz."""
    os.makedirs(os.path.dirname(out_path_noext), exist_ok=True)
    if _HAS_PYGRAPHVIZ:
        g = convert_tree_to_graph(tree, action_labels=action_labels, batch_index=batch_index)
        svg_path = out_path_noext + ".svg"
        g.draw(svg_path, prog="dot", format="svg")
        return svg_path
    else:
        import numpy as _np
        arr = lambda x: jax.device_get(x)
        npz_path = out_path_noext + ".npz"
        _np.savez(
            npz_path,
            children_index=arr(tree.children_index)[batch_index],
            children_rewards=arr(tree.children_rewards)[batch_index],
            children_discounts=arr(tree.children_discounts)[batch_index],
            children_prior_logits=arr(tree.children_prior_logits)[batch_index],
            node_values=arr(tree.node_values)[batch_index],
            node_visits=arr(tree.node_visits)[batch_index],
            num_actions=tree.num_actions,
            num_simulations=tree.num_simulations,
        )
        return npz_path



def gumbel_policy(step_b, heuristic=None, num_simulations=100,
                  save_tree: bool = False,
                  tree_out_dir: str = "./search_trees",
                  file_prefix: str = "gumbel",
                  action_labels: Optional[Sequence[str]] = None,
                  save_every: int = 1):
    """Gumbel MuZero policy (perfect info: gumbel_scale=0), optionally logging trees."""
    if heuristic is None:
        heuristic = _zero_heuristic_builder(step_b)

    def _compiled(state_b, key):
        root_player_b = state_b.game_state.current_player
        root_logits = one_ply_logits(step_b, state_b, root_player_b)

        key, subkey = jax.random.split(key)

        root = mctx.RootFnOutput(
            prior_logits=root_logits,
            value=jnp.where(state_b.game_state.current_player == root_player_b, 1.0, -1.0) * heuristic(state_b, subkey),
            embedding=state_b,
        )

        num_actions = state_b.legal_action_mask.shape[1]

        po = mctx.gumbel_muzero_policy(
            params=None,
            rng_key=key,
            root=root,
            recurrent_fn=ludax_recurrent(root_player_b, step_b, heuristic),
            num_simulations=num_simulations,
            max_num_considered_actions=num_actions,
            invalid_actions= ~(state_b.legal_action_mask & (~state_b.terminated[:, None])),
            max_depth=9,
            # gumbel_scale=0.0  # deterministic for perfect-information games
        )
        return po.action.astype(jnp.int16), po.search_tree

    _compiled = jax.jit(_compiled)

    if not save_tree:
        return jax.jit(lambda state_b, key: _compiled(state_b, key)[0])

    counter = itertools.count(1)
    os.makedirs(tree_out_dir, exist_ok=True)

    def _policy_with_logging(state_b, key):
        n = next(counter)
        action, tree = _compiled(state_b, key)
        if (n % save_every) == 0:
            batch_size = int(tree.node_values.shape[0])
            for b in range(batch_size):
                out_noext = os.path.join(tree_out_dir, f"{file_prefix}_step{n:02d}_b{b}")
                path = save_search_tree(tree, out_noext, action_labels=action_labels, batch_index=b)
                print(f"[Tree saved] {path}")
        return action

    return _policy_with_logging


# ---------- Self-play loop (saves tree each move via the policy wrapper) ----------
def self_play_and_log(env: LudaxEnvironment,
                      step_b,
                      key,
                      policy_fn,
                      action_labels: Sequence[str],
                      out_dir: str = "./ttt_trees"):
    """
    Plays ONE game of self-play (batch_size=1) and logs a tree on EVERY move.
    """
    # State already batched (size 1). Loop until terminal/truncated.
    def cond(carry):
        state, *_ = carry
        return ~(state.terminated.all() | state.truncated.all())

    def body(carry):
        state, key = carry
        key, sub = jax.random.split(key)
        # Single policy for both players; it logs internally every call.
        action = policy_fn(state, sub)
        next_state = step_b(state, action)
        # Minimal log to console:
        cur = int(jax.device_get(state.game_state.current_player[0]))
        act = int(jax.device_get(action[0]))
        print(f"Player {cur} played action {act}")
        return next_state, key

    # Run the while loop on host (Python) to interleave prints and file saving.
    state_b, step_b, key_local = state_batched, step_b, key
    move = 0
    while True:
        if (state_b.terminated.all() | state_b.truncated.all()):
            break
        key_local, k = jax.random.split(key_local)
        action = policy_fn(state_b, k)  # saves the tree internally
        move += 1
        print(f"[Move {move}] current_player={int(jax.device_get(state_b.game_state.current_player[0]))}, "
              f"action={int(jax.device_get(action[0]))}")
        state_b = step_b(state_b, action)

    # Report outcome
    winner = int(jax.device_get(state_b.winner[0]))
    if winner == -1:
        print("Game over: DRAW.")
    else:
        print(f"Game over: Player {winner} wins.")


# ---------- Main ----------
def main():
    # Build environment (tic-tac-toe)
    env = LudaxEnvironment(game_str=tic_tac_toe)

    # Initialize state (single game) and step function
    global state_batched  # used in self_play_and_log's host loop
    state_batched, step_b, key = initialize(env, batch_size=1, seed=42)

    # Edge labels for TTT actions (row,col)
    action_labels = [f"r{r}c{c}" for r in range(3) for c in range(3)]

    # Choose a policy; both support save_tree=True
    policy = gumbel_policy(
        step_b,
        heuristic=None,              # zero heuristic (value from search)
        num_simulations=500,          # tweak as you like
        save_tree=True,              # <-- save tree every call
        tree_out_dir="./ttt_trees",  # directory for artifacts
        file_prefix="game0_move",    # filenames like game0_move_step01_b0.png
        action_labels=action_labels,
        save_every=1                 # save every move
    )

    print("Starting self-play (tic-tac-toe)...")
    self_play_and_log(env, step_b, key, policy, action_labels, out_dir="./ttt_trees")


if __name__ == "__main__":
    main()
