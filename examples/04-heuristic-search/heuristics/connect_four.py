import jax
import jax.numpy as jnp
from ludax.config import BOARD_DTYPE, REWARD_DTYPE

# ----- precompute all 4-in-a-row windows on a 6x7 board -----
def _make_windows_indices():
    R, C = 6, 7
    idxs = []
    # Horizontal
    for r in range(R):
        for c in range(C - 3):
            idxs.append([r*C + c + k for k in range(4)])
    # Vertical
    for r in range(R - 3):
        for c in range(C):
            idxs.append([(r + k)*C + c for k in range(4)])
    # Diagonal down-right
    for r in range(R - 3):
        for c in range(C - 3):
            idxs.append([(r + k)*C + (c + k) for k in range(4)])
    # Diagonal up-right
    for r in range(3, R):
        for c in range(C - 3):
            idxs.append([(r - k)*C + (c + k) for k in range(4)])
    return jnp.array(idxs, dtype=BOARD_DTYPE)  # (69, 4)

WINDOWS_IDX = _make_windows_indices()  # constant used by the heuristic

def connect_four_heuristic(state):
    """
    Jittable heuristic for connect four with encoding:
      -1 = empty, 0 = P1, 1 = P2
    Uses state.game_state.board (..., 42) and state.game_state.current_player (...,).
    Returns one scalar per leading batch element.
    """
    board = state.game_state.board  # (..., 42), int
    cur   = state.game_state.current_player  # (...,) in {0,1}
    opp   = 1 - cur

    # Gather windows: (..., 69, 4)
    # axis=-1 because board is flattened with 42 in the last dim
    cells = jnp.take(board, WINDOWS_IDX, axis=-1)

    # Broadcast player ids to windows: (..., 1, 1)
    cur_b = cur[..., None, None]
    opp_b = opp[..., None, None]

    # Masks
    empty = (cells == -1)
    mine  = (cells == cur_b)
    theirs = (cells == opp_b)

    # Counts per window: (..., 69)
    my_cnt   = jnp.sum(mine, axis=-1)
    opp_cnt  = jnp.sum(theirs, axis=-1)

    # "Open" windows = no opponent (for me) / no me (for opponent)
    my_open  = (opp_cnt == 0)
    opp_open = (my_cnt  == 0)

    # Map counts -> scores (0..4). We ignore 4-in-a-row here; terminal rewards can handle wins.
    # Tweak weights as you like.
    score_map = jnp.array([0.0, 0.1, 1.0, 5.0, 0.0], dtype=REWARD_DTYPE)

    my_line_score  = jnp.take(score_map, my_cnt)
    opp_line_score = jnp.take(score_map, opp_cnt)

    my_score  = jnp.sum(jnp.where(my_open,  my_line_score,  0.0), axis=-1)
    opp_score = jnp.sum(jnp.where(opp_open, opp_line_score, 0.0), axis=-1)

    return (my_score - opp_score).astype(REWARD_DTYPE)


def _center_bonus(board, cur):
    # center column is column 3 (0-based); rows 0..5 -> indices [0*7+3, 1*7+3, ...]
    center_idx = jnp.array([3, 10, 17, 24, 31, 38], dtype=BOARD_DTYPE)
    center_cells = jnp.take(board, center_idx, axis=-1)
    return 0.05 * jnp.sum(center_cells == cur[..., None], axis=-1)  # (...,)


