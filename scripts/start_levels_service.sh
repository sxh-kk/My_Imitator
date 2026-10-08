#!/usr/bin/env bash
# Keep the frozen experiment queue outside the interactive tool session.
set -euo pipefail
workspace_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
imitator_python=${IMITATOR_PYTHON:-/home/zxc/miniconda3/envs/imitator/bin/python}
unit_name=imitator-placemugrack-levels

if systemctl --user is-active --quiet "$unit_name.service"; then
    systemctl --user status "$unit_name.service" --no-pager
    exit 0
fi

# Interrupted formal runs must be archived and restarted with the same seed;
# execute.py/run_queue.py deliberately reject an approximate continuation.
systemd-run --user --unit="$unit_name" \
    --description='Frozen PlaceMugRack ACT L0/L1/L2 experiment queue' \
    --property="WorkingDirectory=$workspace_dir" \
    --property=Nice=5 \
    --property=Restart=no \
    --property=KillMode=control-group \
    --setenv=PYTHONUNBUFFERED=1 \
    "$imitator_python" "$workspace_dir/experiments/act_placemugrack_levels/execute.py"

systemctl --user show "$unit_name.service" \
    --property=ActiveState --property=SubState --property=MainPID
