"""Exercise the real official paired loader and record the selected episodes/shapes."""
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from examples.baselines.lerobot_dataset.lerobot_paired_dataset import (
    PairedDatasetConfig, HumanSimPairedDataset, get_collate_fn,
)

HERE = Path(__file__).resolve().parent
random.seed(1)
np.random.seed(1)
torch.manual_seed(1)
config = PairedDatasetConfig(
    human_root='/var/tmp/imitator-game-zxc/data/imitator_human_v1',
    sim_root='/var/tmp/imitator-game-zxc/data/imitator_sim_v1_zed2i',
    human_dataset_file=str(HERE/'configs/human_train.json'),
    sim_dataset_file=str(HERE/'configs/sim_train.json'),
    include_depth=False, cameras=['zed2i'], num_frames=10, horizon=24,
    obs_horizon=1, state_type='qpos', input_mode='video_only',
)
dataset = HumanSimPairedDataset(config)
batch = next(iter(DataLoader(dataset, batch_size=8, shuffle=False, collate_fn=get_collate_fn('video_only'))))
def describe(value):
    if isinstance(value, torch.Tensor):
        assert torch.isfinite(value).all()
        return dict(shape=list(value.shape), dtype=str(value.dtype), min=value.min().item(), max=value.max().item())
    if isinstance(value, dict):
        return {key: describe(val) for key, val in value.items()}
    return value
result = dict(dataset_size=len(dataset), batch=describe(batch))
assert len(dataset) == 453, len(dataset)
assert list(batch['robot_actions'].shape) == [8, 24, 16]
(HERE/'data-check.json').write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
