tmux new -s ludax
srun --time=6:00:00 -A infra01 --container-writable --environment=/users/alexpadula/projects/ludax/jax.toml --pty bash

tmux attach -t ludax

uv pip install --system --break-system-packages -c constraints.txt --pre -r requirements-dev.txt


srun --jobid=1600773 --overlap --environment=/users/alexpadula/projects/ludax/jax.toml --pty bash


watch squeue --me
scancel --me
scancel 1396277