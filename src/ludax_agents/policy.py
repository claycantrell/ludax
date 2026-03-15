import pickle
from pathlib import Path

import jax
import jax.numpy as jnp
import haiku as hk
import mctx

from ludax.config import ACTION_DTYPE
from .network import AZNet

CKPT_DIR = Path(__file__).parent / "checkpoints"
NUM_SIMULATIONS = 256


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
    """Load an AlphaZero checkpoint and return a GUI-compatible MCTS policy.

    Runs Gumbel MuZero search (NUM_SIMULATIONS={NUM_SIMULATIONS}) at each
    move, identical to the search used during self-play training.

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
    model_tuple = (params, model_state)

    def recurrent_fn(model, rng_key, action, embedding):
        """One MCTS simulation step: apply action, evaluate resulting state."""
        del rng_key
        model_params, model_st = model
        state = jax.vmap(env.step)(embedding, action.astype(ACTION_DTYPE))
        obs = env.observe(state, state.current_player)
        (logits, value), _ = forward.apply(model_params, model_st, obs, is_eval=True)
        logits = logits - jnp.max(logits, axis=-1, keepdims=True)
        logits = jnp.where(state.legal_action_mask, logits, jnp.finfo(logits.dtype).min)

        rewards = state.rewards
        reward = rewards[jnp.arange(rewards.shape[0]), state.current_player]
        value = jnp.where(state.terminated, 0.0, value)
        discount = jnp.where(state.terminated, 0.0, -jnp.ones_like(value))

        return mctx.RecurrentFnOutput(
            reward=reward, discount=discount, prior_logits=logits, value=value
        ), state

    def policy_fn(state_b, key):
        obs_b = env.observe(state_b, state_b.current_player)
        (logits, value), _ = forward.apply(params, model_state, obs_b, is_eval=True)
        logits = logits - jnp.max(logits, axis=-1, keepdims=True)
        logits = jnp.where(state_b.legal_action_mask, logits, jnp.finfo(logits.dtype).min)

        root = mctx.RootFnOutput(prior_logits=logits, value=value, embedding=state_b)
        policy_output = mctx.gumbel_muzero_policy(
            params=model_tuple,
            rng_key=key,
            root=root,
            recurrent_fn=recurrent_fn,
            num_simulations=NUM_SIMULATIONS,
            invalid_actions=~state_b.legal_action_mask,
            qtransform=mctx.qtransform_completed_by_mix_value,
            gumbel_scale=1.0,
        )
        return policy_output.action.astype(ACTION_DTYPE)

    return jax.jit(policy_fn)
