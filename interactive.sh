tmux new -s ludax
srun --time=6:00:00 -A infra01 --container-writable --environment=/users/alexpadula/projects/ludax/jax.toml --pty bash
srun --time=6:00:00 -A infra01 --container-writable --environment=/users/alexpadula/projects/ludax/jax_old.toml --pty bash

tmux attach -t ludax

uv pip install --system --break-system-packages -c constraints.txt --pre -r requirements-dev.txt
uv pip install --system --break-system-packages -c constraints_old.txt --pre -r requirements-dev.txt

srun --jobid=1606488 --overlap --environment=/users/alexpadula/projects/ludax/jax.toml --pty bash


watch squeue --me
scancel --me
scancel 1606488