import jax
import jax.numpy as jnp
from ludax.config import BOARD_DTYPE, REWARD_DTYPE

def connectivity_heuristic(state_b):

    def per_state(state):
        board = state.game_state.board  # ints, shape (...cells...)
        labels = state.game_state.connected_components  # same shape as board, int labels
        player = state.game_state.current_player

        # Keep labels only on the current player's stones; 0 elsewhere.
        flat = jnp.where(board != player, labels, 0).ravel().astype(BOARD_DTYPE)

        # Count presence of each label. Use a static length: #cells + 1 (for label 0).
        n_cells = flat.shape[0]  # static at trace time
        counts = jnp.bincount(flat, length=n_cells + 1)

        # Number of non‑empty component labels, excluding label 0.
        num_components = jnp.count_nonzero(counts[1:] > 0)
        return -num_components.astype(REWARD_DTYPE)

    return jax.vmap(per_state, in_axes=0, out_axes=0)(state_b)



_NEIGHBOR_OFFSETS = [(-1, 0), (-1, 1),
                     ( 0, -1), ( 0, 1),
                     ( 1, -1), ( 1, 0)]

def _shift_no_wrap(arr, dr, dc, big):
    """Roll `arr` by (dr,dc) but overwrite wrapped cells with `big`."""
    rolled = jnp.roll(arr, (dr, dc), axis=(0, 1))

    n = arr.shape[0]
    rows = jnp.arange(n)[:, None]      # shape (n,1)
    cols = jnp.arange(n)[None, :]      # shape (1,n)

    row_bad = (dr == -1) & (rows == n - 1) | (dr == 1) & (rows == 0)
    col_bad = (dc == -1) & (cols == n - 1) | (dc == 1) & (cols == 0)
    mask = row_bad | col_bad

    return jnp.where(mask, big, rolled)

def _min_path_length(cost_flat, vertical):
    n = int(cost_flat.shape[0] ** 0.5)
    cost = cost_flat.reshape((n, n))

    BIG = n * n + 1
    dist = jnp.full_like(cost, BIG)

    # seed distances on the start edge
    dist = jax.lax.cond(
        vertical,
        lambda d: d.at[0, :].set(cost[0, :]),   # top edge
        lambda d: d.at[:, 0].set(cost[:, 0]),   # left edge
        dist,
    )

    def relax(d):
        neigh_min = BIG
        for dr, dc in _NEIGHBOR_OFFSETS:
            neigh = _shift_no_wrap(d, dr, dc, BIG)
            neigh_min = jnp.minimum(neigh_min, neigh)
        return jnp.minimum(d, cost + neigh_min)

    dist = jax.lax.fori_loop(0, n * n, lambda i, d: relax(d), dist)

    target = jax.lax.cond(
        vertical, lambda d: d[-1, :],            # bottom edge
        lambda d: d[:, -1],                      # right edge
        dist,
    )
    return jnp.min(target)


def distance_heuristic(state_b):
    """
    Batched heuristic = opponent_dist - my_dist  (positive is good for current player).
    """

    def per_state(state):
        board_flat = state.game_state.board          # shape (n²,)
        player     = state.game_state.current_player # 0 or 1
        EMPTY      = -1

        # helper that builds the 0‑1‑∞ cost *flat* grid for a given player
        def cost_for(p):
            return jnp.where(
                board_flat == p, 0,
                jnp.where(board_flat == EMPTY, 1,
                          board_flat.size + 1)       # blocked
            )

        my_cost  = cost_for(player)
        opp_cost = cost_for(1 - player)

        my_dist  = _min_path_length(my_cost,  vertical=(player == 0))
        opp_dist = _min_path_length(opp_cost, vertical=(player == 1))

        return (opp_dist - my_dist).astype(REWARD_DTYPE)

    return jax.vmap(per_state, in_axes=0, out_axes=0)(state_b)