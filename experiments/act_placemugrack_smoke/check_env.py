"""Preflight the official task with its real assets, RGB rendering and joint control."""
import json
from pathlib import Path
import gymnasium as gym
import torch
import mani_skill.envs
from PIL import Image
from mani_skill.envs.tasks.tabletop.utils import L0_L3_utils

HERE = Path(__file__).resolve().parent
L0_L3_utils.set_lr_mirror_robot_pose_enabled(False)
env = gym.make('TwoRobotPlaceMugRack-v1', num_envs=1, sim_backend='physx_cpu',
    obs_mode='rgb', control_mode='pd_joint_pos', render_mode='rgb_array',
    max_episode_steps=500, sensor_configs={'shader_pack':'rt-fast'},
    human_render_camera_configs={'shader_pack':'rt-fast'})
try:
    obs, info = env.reset(seed=42)
    action = {f'panda_wristcam-{i}': torch.cat([agent.robot.qpos[0,:7], torch.tensor([1.0])]).numpy()
              for i, agent in enumerate(env.unwrapped.agent.agents)}
    for _ in range(3):
        obs, reward, terminated, truncated, info = env.step(action)
    def shape(x):
        if isinstance(x, dict):
            return {k:shape(v) for k,v in x.items()}
        return list(x.shape) if hasattr(x, 'shape') else str(type(x))
    result = dict(env_id='TwoRobotPlaceMugRack-v1', seed=42, steps=3,
                  obs_shapes=shape(obs), action_shapes=shape(action),
                  reward=reward.tolist(), info_shapes=shape(info))
    im = env.render()
    im = im.cpu().numpy() if hasattr(im, 'cpu') else im
    Image.fromarray(im.squeeze().astype('uint8')).save(HERE/'env-preview.png')
    (HERE/'env-check.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
finally:
    env.close()
