"""Inspect saved progress together with the actual controller process."""
import datetime
import json
import time
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[1]
ROOT = WORKSPACE / 'experiments/act_placemugrack_levels'


def inspect():
    state = json.loads((ROOT / 'status.json').read_text())
    controller = json.loads((ROOT / 'controller.json').read_text())
    command = Path(f'/proc/{controller["pid"]}/cmdline')
    try:
        controller_alive = str(ROOT / 'execute.py').encode() in command.read_bytes()
    except FileNotFoundError:
        controller_alive = False
    telemetry = ROOT / 'gpu-during-study.csv'
    return {
        'checked_at': datetime.datetime.now().astimezone().isoformat(),
        'recorded_status': state,
        'controller_pid': controller['pid'],
        'controller_alive': controller_alive,
        'telemetry_age_seconds': round(time.time() - telemetry.stat().st_mtime, 1),
        'runtime_status': (
            'complete' if state['status'] == 'complete'
            else 'running' if controller_alive
            else 'interrupted'
        ),
    }


if __name__ == '__main__':
    print(json.dumps(inspect(), indent=2, ensure_ascii=False))
