tmux new -s ludax
srun --time=06:05:00 -A infra01 --container-writable --environment=/users/alexpadula/projects/ludax/jax.toml --pty bash

tmux attach -t ludax

uv pip install --system --break-system-packages -c constraints.txt --pre -r requirements-dev.txt