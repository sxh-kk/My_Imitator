"""Experiment utilities; upstream model/dataset/environment sources stay unchanged."""
import hashlib
import json
import os
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parents[1]
SOURCE = WORKSPACE / 'The-Imitator-Game'
DATA = Path('/var/tmp/imitator-game-zxc/data')
TASK = 'L0_TwoRobotPlaceMugRack-v1'
HUMAN = 'human_H57'
OLD = WORKSPACE / 'experiments/act_placemugrack_study'
LEVELS = ['L0', 'L1', 'L2']
CONDITIONS = {'B0': ['L0'], 'B01': ['L0', 'L1'], 'B012': LEVELS}
REVISION = '57fa861d911afe899da5c0f28d973151411d46d1'


def task(level):
    assert level in LEVELS
    return f'{level}_TwoRobotPlaceMugRack-v1'


def stats_path(condition, level='L0'):
    return ROOT/'prepared'/condition/'sim'/task(level)/'meta/stats.json'
sys.path.insert(0, str(SOURCE))
# Allow both direct invocation and subprocesses/workers without shell activation.
_env = json.loads(Path('/home/zxc/miniconda3/envs/imitator/conda-meta/state').read_text())['env_vars']
for _key, _value in _env.items():
    os.environ.setdefault(_key, _value)
for _key in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS']:
    os.environ[_key] = '4'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['HF_DATASETS_OFFLINE'] = '1'
os.environ['PYTHONPATH'] = str(SOURCE) + ':' + str(ROOT)
os.chdir(SOURCE)


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    temp.replace(path)


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def seed_all(seed):
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_num_threads(4)


def build_args(condition, seed, batch, workers, cache):
    from examples.baselines.act.train_act_imitator import TrainingArgs
    return TrainingArgs(
        exp_name=f'{condition}_s{seed}', seed=seed,
        human_root=str(DATA/'imitator_human_v1'),
        sim_root=str(ROOT/'prepared'/condition/'sim'),
        human_dataset_file=str(ROOT/'prepared'/'human_train.json'),
        sim_dataset_file=str(ROOT/'prepared'/condition/'sim_train.json'),
        env_id='TwoRobotPlaceMugRack-v1', batch_size=batch,
        num_dataload_workers=workers, task_num_frames=10,
        frozen_backbone_num_frames=10, te_cache_root=str(cache),
        pred_horizon=24, obs_horizon=1, input_mode='video_only',
        no_eval=True, lr=1e-4, lr_backbone=1e-5,
    )


def build_dataset(args):
    from examples.baselines.act.train_act_imitator import HumanSimPairedDataset, PairedDatasetConfig
    return HumanSimPairedDataset(PairedDatasetConfig(
        human_root=args.human_root, sim_root=args.sim_root,
        task_mapping_file=args.task_mapping_file,
        human_dataset_file=args.human_dataset_file, sim_dataset_file=args.sim_dataset_file,
        human_task_description_file=args.human_task_description_file,
        sim_task_description_file=args.sim_task_description_file,
        split='train', cameras=args.cameras, include_depth=args.include_depth,
        image_size=args.image_size, num_frames=args.task_num_frames,
        horizon=args.pred_horizon, obs_horizon=args.obs_horizon,
        state_type=args.state_type, single_arm=args.single_arm,
        fps=args.fps, video_backend=args.video_backend,
        input_mode=args.input_mode, include_first_frame=args.include_first_frame,
    ))


def seed_worker(worker_id):
    import torch
    import numpy as np
    seed = torch.initial_seed() % (2**32)
    np.random.seed(seed)
    random.seed(seed)
    paired = torch.utils.data.get_worker_info().dataset
    # Albumentations owns its own RNG in recent versions.
    for name in ['transform', 'rgb_transform', 'depth_transform']:
        transform = getattr(paired.sim_dataset, name, None)
        if hasattr(transform, 'set_random_seed'):
            transform.set_random_seed(seed)


def processor(condition, human_episode=1):
    from examples.baselines.lerobot_dataset.evaluate_processor import HumanVideoSimEvaluateProcessor, HumanVideoSimEvaluateProcessorConfig
    return HumanVideoSimEvaluateProcessor(HumanVideoSimEvaluateProcessorConfig(
        human_root=str(DATA/'imitator_human_v1'),
        human_dataset_file=str(ROOT/'prepared'/f'human_ep{human_episode}.json'),
        human_include_depth=False, human_num_frames=10,
        sim_root=str(ROOT/'prepared'/condition/'sim'),
        sim_dataset_file=str(ROOT/'prepared'/condition/'sim_eval.json'),
    ))
