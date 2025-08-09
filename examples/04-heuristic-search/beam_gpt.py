import math
import jax
import jax.numpy as jnp
from jax import tree_util as jtu

from heuristics.test import zero_heuristic
from heuristics import BIG, SMALL

from typing import NamedTuple


class Beam(NamedTuple):
    state: any                 # Ludax state PyTree, shape [B, K, ...]
    score: jnp.ndarray         # [B, K]
    first_action: jnp.ndarray  # [B, K], int16
    depth: jnp.ndarray         # [B, K], int32
    valid: jnp.ndarray         # [B, K], bool


def _node_score(state, root_player, heuristic):
    """
    Score like one-ply on arbitrary leading batch dims:
      BIG * mover_reward + sign * heuristic, with heuristic=0 on terminals.
    Works when state has shape [B, ...], [B, A, ...], or [B, K*A, ...].
    """
    # Leading batch shape is the shape of mover_reward.
    batch_shape = state.mover_reward.shape
    lead_ndims = len(batch_shape)
    flatN = math.prod(batch_shape)

    # Flatten all leading batch dims -> single axis
    state_flat = jtu.tree_map(
        lambda x: x.reshape((flatN,) + x.shape[lead_ndims:]),
        state,
    )

    # Broadcast root_player to batch_shape, then flatten
    rp = root_player
    for _ in range(lead_ndims - rp.ndim):
        rp = rp[..., None]
    rp_b = jnp.broadcast_to(rp, batch_shape).reshape(flatN,)

    to_play_flat = state_flat.game_state.current_player           # [flatN]
    sign_flat = jnp.where(to_play_flat == rp_b, 1.0, -1.0)

    # Zero heuristic on terminals to avoid double counting
    h_flat = heuristic(state_flat)                                # [flatN]
    h_flat = jnp.where(state_flat.terminated, 0.0, h_flat)

    score_flat = BIG * state_flat.mover_reward + sign_flat * h_flat  # [flatN]
    return score_flat.reshape(batch_shape)


def init_beam(state_b, key, step_b, heuristic, k):
    B, A = state_b.legal_action_mask.shape

    # Expand all root actions
    actions = jnp.broadcast_to(jnp.arange(A), (B, A)).reshape(-1).astype(jnp.int16)
    state_flat = jtu.tree_map(lambda x: jnp.repeat(x, A, axis=0), state_b)  # [B*A, ...]
    next_flat = step_b(state_flat, actions)                                  # [B*A, ...]
    next_BA = jtu.tree_map(lambda x: x.reshape(B, A, *x.shape[1:]), next_flat)

    # Score in root perspective (no accumulation)
    root = state_b.game_state.current_player          # [B]
    score_BA = _node_score(next_BA, root[:, None], heuristic)

    # Mask illegal root actions
    score_BA = jnp.where(state_b.legal_action_mask, score_BA, -jnp.inf)

    # Small noise to break ties
    noise = jax.random.uniform(key, shape=score_BA.shape, minval=-SMALL, maxval=SMALL)
    score_BA = score_BA + noise

    # Top-k per batch
    scores_top, idx = jax.lax.top_k(score_BA, k)      # idx: [B, K] in 0..A-1
    idx = idx.astype(jnp.int32)

    # Gather next states by idx
    def gather_state(xBA):
        # xBA: [B, A, ...], idx: [B, K]
        gather_idx = jnp.expand_dims(idx, tuple(range(2, xBA.ndim)))  # [B, K, 1, ...]
        return jnp.take_along_axis(xBA, gather_idx, axis=1)

    next_state_bk = jtu.tree_map(gather_state, next_BA)

    first_action_bk = idx.astype(jnp.int16)
    depth_bk = jnp.zeros((B, k), dtype=jnp.int32)
    valid_bk = jnp.isfinite(scores_top)

    return Beam(next_state_bk, scores_top, first_action_bk, depth_bk, valid_bk)


def beam_step(beam: Beam, key, step_b, heuristic, k):
    B, K = beam.score.shape
    A = beam.state.legal_action_mask.shape[-1]

    # Expand each beam slot over all actions
    actions = jnp.broadcast_to(jnp.arange(A), (B, K, A))
    actions_flat = actions.reshape(B * K * A).astype(jnp.int16)

    # Repeat states along action axis, then flatten to [B*K*A, ...]
    state_BKA = jtu.tree_map(lambda x: jnp.repeat(x, A, axis=1), beam.state)      # [B, K*A, ...]
    state_flat = jtu.tree_map(lambda x: x.reshape(B * K * A, *x.shape[2:]), state_BKA)

    next_flat = step_b(state_flat, actions_flat)                                   # [B*K*A, ...]
    next_BKA = jtu.tree_map(lambda x: x.reshape(B, K * A, *x.shape[1:]), next_flat)

    # Score children from root perspective (same scoring as one-ply)
    root = beam.state.game_state.current_player[:, :1]  # [B,1]
    score_BKA = _node_score(next_BKA, root, heuristic)  # [B, K*A]

    # Mask using the PARENT'S legal mask and validity
    parent_legal_BKA = beam.state.legal_action_mask.reshape(B, K * A)
    parent_valid_BKA = jnp.repeat(beam.valid[..., None], A, axis=2).reshape(B, K * A)
    score_BKA = jnp.where(parent_legal_BKA & parent_valid_BKA, score_BKA, -jnp.inf)

    # Tie-break noise
    noise = jax.random.uniform(key, shape=score_BKA.shape, minval=-SMALL, maxval=SMALL)
    score_BKA = score_BKA + noise

    # Select top-k from K*A candidates
    scores_top, idx = jax.lax.top_k(score_BKA, k)  # idx in [0, K*A)
    idx = idx.astype(jnp.int32)

    # Gather next states by idx
    def take1(xBKA):
        gather_idx = jnp.expand_dims(idx, tuple(range(2, xBKA.ndim)))  # [B, K, 1, ...]
        return jnp.take_along_axis(xBKA, gather_idx, axis=1)

    next_state_bk = jtu.tree_map(take1, next_BKA)

    # Propagate first_action from selected parent beam slot
    parent_id = (idx // A).astype(jnp.int32)  # which old [K] slot
    first_action_bk = jnp.take_along_axis(beam.first_action, parent_id, axis=1)

    depth_bk = jnp.take_along_axis(beam.depth, parent_id, axis=1) + 1
    valid_bk = jnp.isfinite(scores_top)

    return Beam(next_state_bk, scores_top, first_action_bk, depth_bk, valid_bk)


def beam_search_policy(step_b, heuristic=zero_heuristic, topk=200, iterations=10):
    def beam_search_f(state_b, key):
        beam = init_beam(state_b, key, step_b, heuristic, topk)

        def body(i, bm):
            key_i = jax.random.fold_in(key, i)
            return beam_step(bm, key_i, step_b, heuristic, topk)

        beam = jax.lax.fori_loop(0, iterations, body, beam)

        # Choose best slot per batch and return its root action
        best_idx = jnp.argmax(beam.score, axis=1)  # [B]
        best_action = jnp.take_along_axis(beam.first_action, best_idx[:, None], axis=1).squeeze(1)
        return best_action.astype(jnp.int16)

    return jax.jit(beam_search_f)
