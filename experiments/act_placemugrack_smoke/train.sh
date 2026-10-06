#!/usr/bin/env bash
set -euo pipefail
source /home/zxc/Imitator/scripts/activate_imitator.sh
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
experiment_dir=/home/zxc/Imitator/experiments/act_placemugrack_smoke
experiment_name=${IMITATOR_EXP_NAME:-act_placemugrack_smoke}
checkpoint_dir="runs/${experiment_name}-$(date +%Y%m%d)/checkpoints"
if [[ -f "$checkpoint_dir/final_model.pt" ]]; then
  echo 'This run already exists. Set IMITATOR_EXP_NAME to a new name for another run.' >&2
  exit 1
fi
python -u -m examples.baselines.act.train_act_imitator \
  --exp-name "$experiment_name" --seed 1 \
  --human-root /var/tmp/imitator-game-zxc/data/imitator_human_v1 \
  --sim-root /var/tmp/imitator-game-zxc/data/imitator_sim_v1_zed2i \
  --human-dataset-file "$experiment_dir/configs/human_train.json" \
  --sim-dataset-file "$experiment_dir/configs/sim_train.json" \
  --env-id TwoRobotPlaceMugRack-v1 \
  --batch-size 8 --total-epochs 3 --warmup-epochs 0 \
  --lr 0.0001 --lr-backbone 0.00001 \
  --num-dataload-workers 0 --log-freq 10 --save-epoch-freq 100 \
  --task-num-frames 10 --frozen-backbone-num-frames 10 \
  --pred-horizon 24 --obs-horizon 1 --input-mode video_only \
  --te-cache-root "$experiment_dir/te_cache" \
  "$@"
