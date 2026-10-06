#!/usr/bin/env bash
set -euo pipefail
source /home/zxc/Imitator/scripts/activate_imitator.sh
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTHONPATH=/home/zxc/Imitator/The-Imitator-Game
experiment_dir=/home/zxc/Imitator/experiments/act_placemugrack_smoke
checkpoint=${IMITATOR_CHECKPOINT:-/home/zxc/Imitator/The-Imitator-Game/runs/act_placemugrack_smoke-20261006/checkpoints/final_model.pt}
evaluation_dir=${IMITATOR_EVAL_DIR:-"$experiment_dir/evaluation_$(date +%Y%m%d_%H%M%S)"}
python -u "$experiment_dir/evaluate_checked.py" \
  --checkpoint "$checkpoint" --eval-config "$experiment_dir/configs/eval_envs.txt" \
  --human-root /var/tmp/imitator-game-zxc/data/imitator_human_v1 \
  --sim-root /var/tmp/imitator-game-zxc/data/imitator_sim_v1_zed2i \
  --human-config "$experiment_dir/configs/human_eval.json" \
  --sim-config "$experiment_dir/configs/sim_eval.json" \
  --num-episodes 2 --num-envs 1 --max-episode-steps 500 \
  --sim-backend physx_cpu --control-mode pd_joint_pos \
  --input-mode video_only --device cuda \
  --output-dir "$evaluation_dir" \
  "$@"
