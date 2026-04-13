BIG = 10**6
SMALL = 10**-3


from .value import zero_heuristic, construct_playout_heuristic
from .beam import beam_search_policy
from .negamax import negamax_policy
from .simple import one_ply_policy, random_policy
from .mctx_original import simple_mctx_policy, lookahead_mctx_policy
from .mcts import uct_mcts_policy


__all__ = [
    "beam_search_policy",
    "negamax_policy",
    "one_ply_policy",
    "random_policy",
    "simple_mctx_policy",
    "lookahead_mctx_policy",
    "zero_heuristic",
    "BIG",
    "SMALL"
]