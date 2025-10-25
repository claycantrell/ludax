GPUS=("0" "1" "2" "3")
NUM_GPUS=${#GPUS[@]}

CMN="--num_batch_sizes 15"

GAMES="tic_tac_toe
connect_four
hex
reversi
gomoku
pente
yavalath
yavalax
connect_six
complexity_demo"

run_one() {
  local game="$1"
  local slot="$2"                    # 1..NUM_GPUS
  local idx=$((slot - 1))            # 0-based index
  local gpu="${GPUS[$idx]}"

  echo "[slot $slot → GPU $gpu] $game"
  CUDA_VISIBLE_DEVICES="$gpu" \
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
  XLA_PYTHON_CLIENT_MEM_FRACTION=0.8 \
  python compare_implementations.py --game "$game" $CMN
}
export -f run_one
export -a GPUS

# Feed the games; keep exactly NUM_GPUS concurrent jobs.
# `{%}` is the GNU Parallel job-slot number (1..NUM_GPUS).
printf "%s\n" $GAMES | parallel -j "$NUM_GPUS" --lb --joblog jobs.tsv \
  'run_one {} {%}'
