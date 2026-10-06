"""Run official evaluation with read-only checks around checkpoint loading and actions.

No model, action aggregation, normalization, physics, or scoring logic is replaced.
CLI arguments are passed to the official eval_act_imitator.main().
"""
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from examples.baselines.act import eval_act_imitator as official

HERE = Path(__file__).resolve().parent
random.seed(1)
np.random.seed(1)
torch.manual_seed(1)
report = dict(status='failed', process_seed=1, policy_queries=0, env_steps=0,
              all_actions_finite=True, episodes=[], started_at=time.time())
report['argv'] = sys.argv[1:]
output_dir = Path(official.parse_args().output_dir)
output_dir.mkdir(parents=True, exist_ok=True)
original_load = official.load_agent_from_checkpoint
original_make = official.make_eval_envs_with_level


def checked_load(path, args, device):
    agent, config = original_load(path, args, device)
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    saved = {k.replace('.video_encoder.original.', '.video_encoder.')
             .replace('.lang_encoder.original.', '.lang_encoder.')
             .replace('task_encoder.backbone.', 'task_encoder.'): v
             for k, v in ckpt['agent_state_dict'].items()}
    # The official loader loads the frozen pretrained DINO lazily. Materialize
    # it now, then verify that even those initially unexpected keys match.
    agent.task_encoder._load_backbone()
    agent.to(device).eval()
    restored = agent.state_dict()
    assert set(restored) == set(saved), (set(restored)-set(saved), set(saved)-set(restored))
    for key, value in restored.items():
        assert torch.isfinite(value).all(), key
        assert torch.equal(value.detach().cpu(), saved[key]), key
    states = ckpt['optimizer_state_dict']['state']
    steps = sorted({int(v['step']) for v in states.values() if 'step' in v})
    assert steps == [171], steps
    assert any(torch.count_nonzero(v['exp_avg']) for v in states.values() if 'exp_avg' in v)
    assert ckpt['iteration'] == 171
    report['checkpoint'] = dict(path=path, bytes=Path(path).stat().st_size,
        iteration=ckpt['iteration'], epoch=ckpt['epoch'],
        equal_state_tensors=len(restored), optimizer_step_values=steps,
        optimizer_state_count=len(states), scheduler_last_epoch=ckpt['lr_scheduler_state_dict']['last_epoch'])
    original_action = agent.get_action

    def checked_action(obs):
        action = original_action(obs)
        assert tuple(action.shape) == (1, 24, 16), action.shape
        assert torch.isfinite(action).all()
        report['policy_queries'] += 1
        report.setdefault('first_policy_query', dict(
            obs_shapes={k:list(v.shape) for k,v in obs.items()},
            output_shape=list(action.shape), min=action.min().item(), max=action.max().item()))
        return action

    agent.get_action = checked_action
    print('CHECKPOINT VERIFIED: all state tensors exactly equal; optimizer at step 171', flush=True)
    return agent, config


def checked_make(*args, **kwargs):
    env = original_make(*args, **kwargs)
    original_step = env.step

    def checked_step(action):
        for key, value in action.items():
            assert np.asarray(value).shape == (1, 8), (key, np.asarray(value).shape)
            assert np.isfinite(value).all(), key
        report.setdefault('first_robot_action', {k:np.asarray(v).tolist() for k,v in action.items()})
        result = original_step(action)
        report['env_steps'] += 1
        obs, reward, terminated, truncated, info = result
        assert all(np.isfinite(v).all() for v in obs.values())
        assert np.isfinite(reward).all()
        if np.asarray(truncated).any():
            for final in info['final_info']:
                report['episodes'].append({k:np.asarray(v).tolist() for k,v in final['episode'].items()})
        return result

    env.step = checked_step
    return env


official.load_agent_from_checkpoint = checked_load
official.make_eval_envs_with_level = checked_make
try:
    official.main()
    assert report['env_steps'] == 1000, report['env_steps']
    assert report['policy_queries'] == 250, report['policy_queries']
    assert len(report['episodes']) == 2, report['episodes']
    report['status'] = 'passed'
finally:
    report['elapsed_seconds'] = time.time() - report['started_at']
    (output_dir/'evaluation-check.json').write_text(json.dumps(report, indent=2))
