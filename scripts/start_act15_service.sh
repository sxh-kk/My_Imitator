#!/usr/bin/env bash
# Start all three reproduction conditions outside the interactive terminal.
set -euo pipefail
workspace_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
imitator_python=${IMITATOR_PYTHON:-/home/zxc/miniconda3/envs/imitator/bin/python}
unit_name=imitator-act15

if systemctl --user is-active --quiet "$unit_name.service"; then
    echo 'The ACT 15-task experiment is already running.'
    systemctl --user show "$unit_name.service" --property=ActiveState --property=SubState --property=MainPID
    exit 0
fi

"$imitator_python" "$workspace_dir/experiments/act_dinov2_15task/execute.py" --check-only
systemctl --user reset-failed "$unit_name.service" 2>/dev/null || true
systemd-run --user --unit="$unit_name" \
    --description='ACT+DINOv2 15-task pretrain, seen/unseen, scratch and few-shot fine-tuning' \
    --property="WorkingDirectory=$workspace_dir" \
    --property=Nice=5 --property=Restart=no --property=KillMode=control-group \
    --setenv=PYTHONUNBUFFERED=1 --setenv=HF_HUB_OFFLINE=1 --setenv=HF_DATASETS_OFFLINE=1 \
    "$imitator_python" "$workspace_dir/experiments/act_dinov2_15task/execute.py"

systemctl --user show "$unit_name.service" --property=ActiveState --property=SubState --property=MainPID
