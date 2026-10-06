#!/usr/bin/env bash
# Usage: source /home/zxc/Imitator/scripts/activate_imitator.sh
if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    echo "Please source this script: source /home/zxc/Imitator/scripts/activate_imitator.sh" >&2
    exit 1
fi
source /home/zxc/miniconda3/etc/profile.d/conda.sh
conda activate imitator
cd /home/zxc/Imitator/The-Imitator-Game || return 1
