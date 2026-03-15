import pickle
from pathlib import Path

import jax
import jax.numpy as jnp
import haiku as hk

from ludax.config import ACTION_DTYPE
from .network import AZNet

CKPT_DIR = Path(__file__).parent / "checkpoints"


def get_checkpoint_path(game_id: str) -> Path | None:
    """Return the checkpoint path for *game_id*, or None if it doesn't exist."""
    path = CKPT_DIR / f"{game_id}.ckpt"
    return path if path.exists() else None


class _CheckpointUnpickler(pickle.Unpickler):
    """Redirect ``__main__.Config`` to ``ludax_agents.az_config.Config``.

    Checkpoints produced by the AlphaZero trainer pickle the ``Config``
    dataclass as ``__main__.Config`` because ``az_config.py`` was run directly.
    This unpickler transparently remaps that reference so checkpoints can be
    loaded from any calling context.
    """
    def find_class(self, module, name):
        if module == "__main__":
            module = "ludax_agents.az_config"
        return super().find_class(module, name)


def _find_num_actions(params) -> int:
    """Read num_actions from the policy-head linear weight in the checkpoint."""
    return params["az_net/linear"]["w"].shape[-1]


def az_checkpoint_policy(env, ckpt_path):
    """Load an AlphaZero checkpoint and return a GUI-compatible policy.

    The returned function has signature ``policy(state_b, key) -> action_b``,
    matching the interface expected by the Ludax GUI and all other policies in
    ``ludax.policies``.

    Args:
        env:       A ``LudaxEnvironment`` instance for the target game.
        ckpt_path: Path to a ``.ckpt`` file produced by the AlphaZero trainer.

    Returns:
        A JIT-compiled policy function.
    """
    with open(ckpt_path, "rb") as f:
        data = _CheckpointUnpickler(f).load()

    config = data["config"]
    params, model_state = data["model"]
    # Read num_actions from the policy head weight shape — the checkpoint is
    # authoritative and handles games with extra actions (e.g. Reversi's pass move).
    num_actions = _find_num_actions(params)

    def forward_fn(x, is_eval=False):
        net = AZNet(
            num_actions=num_actions,
            num_channels=config.num_channels,
            num_blocks=config.num_layers,
            resnet_v2=config.resnet_v2,
        )
        return net(x, is_training=not is_eval, test_local_stats=False)

    forward = hk.without_apply_rng(hk.transform_with_state(forward_fn))

    def policy_fn(state_b, key):
        # observe() supports arbitrary leading batch dims natively
        obs_b = env.observe(state_b, state_b.game_state.current_player)
        (logits, _value), _ = forward.apply(params, model_state, obs_b, is_eval=True)
        # mask illegal actions to -inf before sampling
        logits = jnp.where(state_b.legal_action_mask, logits, jnp.finfo(logits.dtype).min)
        return jax.random.categorical(key, logits, axis=1).astype(ACTION_DTYPE)

    return jax.jit(policy_fn)
