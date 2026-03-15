import jax
from pydantic import BaseModel


devices = jax.local_devices()
num_devices = len(devices)


class Config(BaseModel):
    env_id: str = "reversi"
    env_type: str = "ldx"
    seed: int = -1
    max_num_iters: int = 200
    # network params
    num_channels: int = 128
    num_layers: int = 6
    resnet_v2: bool = True
    # selfplay params
    selfplay_batch_size: int = 1024
    num_simulations: int = 64
    max_num_steps: int = 64
    # training params
    training_batch_size: int = 512
    learning_rate: float = 0.001
    # eval params
    eval_interval: int = 5
    eval_num_games: int = 128  # games per side vs best model (total = 2x this)
    # checkpoint params
    ckpt_dir: str = "./checkpoints"

    class Config:
        extra = "forbid"
