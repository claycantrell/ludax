"""
A modified version of policies/mctx.py that saves the search tree to an SVG file,
with a few extra changes to try, in vane, to prevent the exploration of illegal actions.
"""


import html
from functools import partial
import os
import itertools
from typing import Optional, Sequence
import re

import jax
import jax.numpy as jnp
import numpy as np
import mctx

from ludax import LudaxEnvironment
from ludax.games import tic_tac_toe
from ludax.gui import InteractiveBoardHandler
import pygraphviz

from ludax.policies import construct_playout_heuristic


# ---------- Pretty printing ----------
np.set_printoptions(threshold=np.inf, linewidth=np.inf)
jax.numpy.set_printoptions(threshold=np.inf, linewidth=np.inf)


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
# WINNING_LOGIT = 0.3
# LOSING_LOGIT = 0.2
# LEGAL_LOGIT = 0.1
# WINNING_LOGIT = 6.0
# LEGAL_LOGIT = 0.0
# LOSING_LOGIT  = -6.0
# NEG_INF = jnp.array(-jnp.inf, dtype=jnp.float32)
NEG_BIG = jnp.array(-1e9, dtype=jnp.float32)
# NEG_INF = jnp.array(-1e3, dtype=jnp.float32)


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

    legal = state_b.legal_action_mask \
            & (~state_b.terminated[:, None])
    legal_flat = legal.reshape(-1)
    logits_flat = jnp.where(legal_flat, logits_flat, NEG_BIG)

    return logits_flat.reshape(batch_size, num_actions).astype(jnp.float32)


def ludax_recurrent(root_player_b, step_b, heuristic):
    def recurrent_fn(params, rng_key, action, state):
        key, sub_key = jax.random.split(rng_key)
        next_state = step_b(state, action.astype(jnp.int16))

        r = next_state.rewards[jnp.arange(next_state.rewards.shape[0]), root_player_b]
        l = one_ply_logits(step_b, next_state, root_player_b)
        # l = jnp.where(next_state.terminated[:, None], jnp.full_like(l, NEG_BIG), l)


        to_play = next_state.game_state.current_player
        sign = jnp.where(to_play == root_player_b, 1.0, -1.0)
        v = sign * heuristic(next_state, sub_key)

        v = jnp.where(next_state.terminated, 0, v)
        # d = jnp.ones_like(v)
        d = jnp.where(next_state.terminated, 0.0, 1.0)

        # r = jnp.where(state.terminated , -1e9, r)

        # jax.debug.print("r: {r} d: {d} v: {v}", r=r, d=d, v=v, ordered=True)

        out = mctx.RecurrentFnOutput(
            reward=r,
            discount=d,
            prior_logits=l,
            value=v,
        )
        return out, next_state
    return jax.jit(recurrent_fn)



def convert_tree_to_graph(
    tree: mctx.Tree,
    action_labels: Optional[Sequence[str]] = None,
    batch_index: int = 0,
):
    batch_size = tree.node_values.shape[0]
    if action_labels is None:
        action_labels = list(range(tree.num_actions))
    elif len(action_labels) != tree.num_actions:
        raise ValueError(
            f"action_labels has {len(action_labels)} entries; expected {tree.num_actions}"
        )

    # We stored Ludax states in embeddings; pull them back node-by-node.
    embeddings = getattr(tree, "embeddings", None)
    if embeddings is None:
        raise RuntimeError("Tree has no embeddings; pass `embedding=state_b` at the root.")

    def take_node_field(fn):
        # Helper to map a field out of the embedding PyTree at [batch, node]
        return lambda node_i: jax.device_get(
            jax.tree_util.tree_map(lambda x: x[batch_index, node_i], embeddings)
        )

    get_emb = take_node_field(lambda x: x)

    def board_label_for(node_i: int) -> str:
        emb = get_emb(node_i)
        board_np = np.asarray(emb.game_state.board)
        board_str = np.array2string(board_np, separator=" ")
        return f"Board: {board_str}\n"

    def node_flags(node_i: int):
        emb = get_emb(node_i)
        term = bool(np.asarray(emb.terminated))
        # Blocked: all child priors are -inf (no legal actions from here)
        child_logits = np.asarray(tree.children_prior_logits[batch_index, node_i])
        blocked = bool(np.all(child_logits <= NEG_BIG))
        # Expanded: any child created?
        expanded = bool(np.any(np.asarray(tree.children_index[batch_index, node_i]) >= 0))
        cp = int(np.asarray(emb.game_state.current_player))
        return term, blocked, expanded, cp

    def node_to_str(node_i: int, reward) -> str:
        term, blocked, expanded, cp = node_flags(node_i)
        val = float(tree.node_values[batch_index, node_i])
        visits = int(tree.node_visits[batch_index, node_i])
        # Invariant checks
        warn = ""
        if term and not blocked:
            warn += " ⚠term-but-not-blocked"
        if term and val != 0.0:
            # Allow tiny num noise
            if abs(val) > 1e-5:
                warn += f" ⚠term-value={val:.3g}"
        # Child prior stats (useful to see masking mistakes)
        child_logits = np.asarray(tree.children_prior_logits[batch_index, node_i])
        logits_max = float(np.max(child_logits))
        logits_lse = float(jax.scipy.special.logsumexp(child_logits))

        return (f"{node_i}  (P{cp}){warn}\n"
                f"Visits: {visits}\n"
                f"Value: {val:.6g}  Reward:{reward}\n"
                f"Terminal: {term}  Blocked: {blocked}  Expanded: {expanded}\n"
                f"logits[max]={logits_max:.3g} logsumexp={logits_lse:.3g}\n"
                f"{board_label_for(node_i)}\n\n\n\n\n")

    def edge_to_str(parent_i: int, a_i: int) -> str:
        # Edge stats parent->child for action a_i
        child_i = int(tree.children_index[batch_index, parent_i, a_i])
        r = float(tree.children_rewards[batch_index, parent_i, a_i])
        d = float(tree.children_discounts[batch_index, parent_i, a_i])
        child_term = (d == 0.0)

        parent_term, _, _, _ = node_flags(parent_i)
        post_term = parent_term  # i.e., "from a terminal parent"

        # Q and P for this edge
        parent_index = jnp.full([tree.node_values.shape[0]], parent_i)
        q = float(tree.qvalues(parent_index)[batch_index, a_i])
        probs = jax.nn.softmax(tree.children_prior_logits[batch_index, parent_i])
        p = float(probs[a_i])

        # Sanity warnings
        warn = ""
        if post_term and (r != 0.0 and not np.isneginf(r)):
            warn += " ⚠post-term-reward"
        if child_term and p > 0:
            warn += " ⚠child-term-has-prob"

        return (f"{action_labels[a_i]}\n"
                f"Q:{q:.6g}  r:{r:.6g}  d:{d:.3g}\nP:{p:.6f}\nchild_term:{child_term}  from_term:{post_term}\n{warn}"
                )

    g = pygraphviz.AGraph(directed=True)
    g.add_node(0, label=node_to_str(0, 0), color="green")

    for node_i in range(tree.num_simulations):
        for a_i in range(tree.num_actions):
            child_i = int(tree.children_index[batch_index, node_i, a_i])
            if child_i >= 0:
                d = float(tree.children_discounts[batch_index, node_i, a_i])
                is_terminal_child = (d == 0.0)
                g.add_node(
                    child_i,
                    label=node_to_str(child_i, tree.children_rewards[batch_index, node_i, a_i]),
                    color=("blue" if is_terminal_child else "red"),
                )
                g.add_edge(node_i, child_i, label=edge_to_str(node_i, a_i))
    return g





BOARD_TEXT_RE = re.compile(
    r'(?P<open><text\b(?P<attrs>[^>]*)>)\s*Board:\s*\[(?P<board>[^\]]+)\]\s*(?P<close></text>)',
    re.IGNORECASE | re.DOTALL
)

def _attr(attrs: str, name: str):
    m = re.search(fr'\b{name}\s*=\s*["\']([^"\']+)["\']', attrs)
    if not m:
        print(f"Failed to find {name} in {attrs}")
        return 0

    return float(m.group(1))

def strip_outer_svg(svg_markup: str) -> str:
    # remove optional XML header
    s = re.sub(r'^\s*<\?xml[^>]*>\s*', '', svg_markup, flags=re.IGNORECASE)
    m = re.search(r'<svg\b[^>]*>(?P<inner>.*)</svg\s*>', s, flags=re.IGNORECASE|re.DOTALL)
    return (m.group('inner') if m else s).strip()

def render_states(svg_path, renderer: InteractiveBoardHandler):
    with open(svg_path, "r") as f:
        svg_text = f.read()

    def process_board_text(board_fragment: str) -> str:
        board = [int(x) for x in re.findall(r'-?\d+', html.unescape(board_fragment))]
        assert len(board) == renderer.game_info.board_size, f"Expected {renderer.game_info.board_size} cells, got {len(board)}"
        render = renderer.render_fn(board, add_button=False)
        return render

    def _replacer(m: re.Match) -> str:
        attrs = m.group('attrs')
        x = _attr(attrs, 'x')
        y = _attr(attrs, 'y')
        inner = strip_outer_svg(process_board_text(m.group('board')))

        x += -50
        y += 0
        w = h = 10
        tx, ty = x - w / 2, y - h / 2
        return f'<g transform="translate({tx:.2f},{ty:.2f}) scale(0.3)">{inner}</g>'

    svg_text = BOARD_TEXT_RE.sub(_replacer, svg_text)

    with open(svg_path, "w") as f:
        f.write(svg_text)



def save_search_tree(
    tree: mctx.Tree,
    renderer: InteractiveBoardHandler,
    out_path_noext: str,
    action_labels: Optional[Sequence[str]] = None,
    batch_index: int = 0,
):
    """Save tree to SVG if pygraphviz available; else save raw arrays to .npz."""
    os.makedirs(os.path.dirname(out_path_noext), exist_ok=True)
    g = convert_tree_to_graph(tree, action_labels=action_labels, batch_index=batch_index)
    svg_path = out_path_noext + ".svg"
    g.draw(svg_path, prog="dot", format="svg")
    render_states(svg_path, renderer)
    return svg_path


def qtransform_by_parent_and_siblings_with_discount_mask(
    tree,
    node_index,
    *,
    epsilon = 1e-8,
):
    # Run the original transform.
    q = mctx.qtransform_by_parent_and_siblings(tree, node_index, epsilon=epsilon)

    # Per-child discounts for this parent.
    discounts = tree.children_discounts[node_index]

    jax.debug.print("q: {q}", q=q, ordered=True)
    jax.debug.print("discounts: {d}", d=discounts, ordered=True)

    q_masked = jnp.where(discounts < 1, NEG_BIG * jnp.ones_like(q), q)
    jax.debug.print("q_masked: {qm}\n\n", qm=q_masked, ordered=True)
    return q_masked

def gumbel_policy(step_b, renderer, heuristic=None, num_simulations=100,
                  save_tree: bool = False,
                  tree_out_dir: str = "./search_trees",
                  file_prefix: str = "gumbel",
                  action_labels: Optional[Sequence[str]] = None,
                  save_every: int = 1):
    """Gumbel MuZero policy (perfect info: gumbel_scale=0), optionally logging trees."""
    if heuristic is None:
        heuristic = construct_playout_heuristic(step_b, num_playouts=100)

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
            invalid_actions= ~(state_b.legal_action_mask),
            max_depth=9,
            gumbel_scale=0.0,  # deterministic for perfect-information games
            # qtransform=qtransform_by_parent_and_siblings_with_discount_mask,
            # qtransform=mctx.qtransform_completed_by_mix_value
            # qtransform=lambda tree, node_index: mctx.qtransform_by_min_max(tree, node_index, min_value=0, max_value=300)
        )


        # po = mctx.muzero_policy(
        #         params=None,
        #         rng_key=key,
        #         root=root,
        #         recurrent_fn=ludax_recurrent(root_player_b, step_b, heuristic),
        #         num_simulations=num_simulations,
        #         invalid_actions= ~(state_b.legal_action_mask),
        #         max_depth=9,
        #         temperature=0.0,
        #         dirichlet_fraction=0.0,
        # )

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
                path = save_search_tree(tree, renderer, out_noext, action_labels=action_labels, batch_index=b)
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

    # Run the while loop on host (Python) to interleave prints and file saving.
    state_b, step_b, key_local = state_batched, step_b, key
    move = 0
    while True:
        if (state_b.terminated.all() | state_b.truncated.all()):
            break
        key_local, k = jax.random.split(key_local)
        action = policy_fn(state_b, k)  # saves the tree internally
        move += 1
        print("\n\n\n\n", "-" * 40)
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
    renderer = InteractiveBoardHandler(env.game_info, env.rendering_info)

    # Initialize state (single game) and step function
    global state_batched  # used in self_play_and_log's host loop
    state_batched, step_b, key = initialize(env, batch_size=1, seed=42)

    # Edge labels for TTT actions (row,col)
    action_labels = [f"r{r}c{c}" for r in range(3) for c in range(3)]

    # Choose a policy; both support save_tree=True
    policy = gumbel_policy(
        step_b,
        renderer,
        heuristic=None,
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
