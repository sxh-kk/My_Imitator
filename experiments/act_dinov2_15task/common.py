"""Outer orchestration for the released ACT baseline; no upstream edits."""
import hashlib
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parents[1]
SOURCE = WORKSPACE / 'The-Imitator-Game'
DATA = Path('/var/tmp/imitator-game-zxc/data')
ASSETS = Path('/var/tmp/imitator-game-zxc/maniskill/data')
CONFIG = SOURCE / 'examples/baselines/lerobot_dataset/config/exp_configs'
PYTHON = Path('/home/zxc/miniconda3/envs/imitator/bin/python')
DATA_REVISION = '57fa861d911afe899da5c0f28d973151411d46d1'
ASSET_REVISION = '2d7a339b27aff14a0780db4bd4d5e3a68c6358bc'

for key, value in json.loads((PYTHON.parent.parent/'conda-meta/state').read_text())['env_vars'].items():
    os.environ.setdefault(key, value)
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '4'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['HF_DATASETS_OFFLINE'] = '1'
os.environ['PYTHONPATH'] = str(SOURCE) + ':' + str(ROOT)
sys.path.insert(0, str(SOURCE))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    tmp.replace(path)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def set_status(stage, **extra):
    path = ROOT/'status.json'
    value = json.loads(path.read_text()) if path.exists() else {}
    value.update(stage=stage, updated_at=time.time(), **extra)
    write_json(path, value)


def offline():
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['HF_DATASETS_OFFLINE'] = '1'


def seed_all(seed):
    import random
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(4)


def build_args(condition, workers=8):
    from examples.baselines.act.train_act_imitator import TrainingArgs
    plan = json.loads((ROOT/'plan.json').read_text())
    pretrain = condition == 'pretrain15'
    return TrainingArgs(
        exp_name='act_dinov2_15task_'+condition, seed=plan['training']['seed'],
        human_root=str(DATA/'imitator_human_v1'), sim_root=str(ROOT/'prepared/sim_rgb'),
        human_dataset_file=str(ROOT/'prepared'/('human_train_15.json' if pretrain else 'human_fewshot.json')),
        sim_dataset_file=str(ROOT/'prepared'/('sim_train_15.json' if pretrain else 'sim_fewshot.json')),
        task_mapping_file=str(SOURCE/'examples/baselines/lerobot_dataset/task_mapping.json'),
        human_task_description_file=str(SOURCE/'examples/baselines/lerobot_dataset/task_desc/human_desc.json'),
        sim_task_description_file=str(SOURCE/'examples/baselines/lerobot_dataset/task_desc/sim_desc.json'),
        total_epochs=plan['training']['epochs'], batch_size=plan['training']['batch_size'],
        warmup_epochs=plan['training']['warmup_epochs'], num_dataload_workers=workers,
        frozen_backbone_num_frames=4, task_num_frames=10,
        te_cache_root=str(ROOT/'feature_cache'/('train15' if pretrain else 'fewshot5')),
        no_eval=True, include_depth=False,
    )


def build_dataset(args):
    from examples.baselines.act.train_act_imitator import HumanSimPairedDataset, PairedDatasetConfig
    return HumanSimPairedDataset(PairedDatasetConfig(
        human_root=args.human_root, sim_root=args.sim_root,
        human_dataset_file=args.human_dataset_file, sim_dataset_file=args.sim_dataset_file,
        task_mapping_file=args.task_mapping_file,
        human_task_description_file=args.human_task_description_file,
        sim_task_description_file=args.sim_task_description_file,
        split='train', cameras=args.cameras, include_depth=False, image_size=args.image_size,
        num_frames=args.task_num_frames, horizon=args.pred_horizon, obs_horizon=args.obs_horizon,
        state_type=args.state_type, single_arm=False, fps=args.fps,
        video_backend=args.video_backend, input_mode=args.input_mode, include_first_frame=args.include_first_frame,
    ))


def seed_worker(worker_id):
    import random
    import numpy as np
    import torch
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)
    ds = torch.utils.data.get_worker_info().dataset
    for name in ['transform','rgb_transform','depth_transform']:
        transform=getattr(ds.sim_dataset,name,None)
        if hasattr(transform,'set_random_seed'):transform.set_random_seed(seed)


def make_loader(dataset, args, workers):
    from torch.utils.data import DataLoader
    from examples.baselines.act.train_act_imitator import get_collate_fn
    kw=dict(dataset=dataset,batch_size=args.batch_size,shuffle=True,num_workers=workers,
            collate_fn=get_collate_fn(args.input_mode),pin_memory=True,worker_init_fn=seed_worker)
    if workers:kw.update(persistent_workers=True,prefetch_factor=4,multiprocessing_context='forkserver')
    return DataLoader(**kw)
