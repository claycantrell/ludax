BIG = 10**6
SMALL = 10**-3


from .value import zero_heuristic, random_playout_heuristic_constructor
from .beam import beam_search_policy
from .negamax import negamax_policy
from .simple import one_ply_policy, random_policy
from .mcts import mcts_policy, gumbel_policy


__all__ = [
    "beam_search_policy",
    "negamax_policy",
    "one_ply_policy",
    "random_policy",
    "mcts_policy",
    "gumbel_policy",
    "zero_heuristic",
    "BIG",
    "SMALL"
]