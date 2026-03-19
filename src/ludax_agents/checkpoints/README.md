# AlphaZero Checkpoints

Place one checkpoint file per game here, named `<game_id>.ckpt`.

Game IDs match the keys in `ludax.games.__all__`. To list them:

```python
from ludax import games
print(games.__all__)
```

Each `.ckpt` file is a checkpoint produced by `examples/03-alpha-zero/train.py`.
Copy the best iteration file (e.g. `000200.ckpt`) and rename it:

```bash
cp path/to/checkpoints/reversi_ldx_20250101/000200.ckpt \
   src/ludax_agents/checkpoints/reversi.ckpt
```

The `alphazero` policy appears automatically in the GUI dropdowns for any game
that has a matching file here. No configuration needed — just install the package:

```bash
pip install -e .[agents]
```
