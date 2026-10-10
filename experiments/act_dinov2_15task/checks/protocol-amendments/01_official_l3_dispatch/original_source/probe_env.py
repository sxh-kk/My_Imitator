"""Create an official level scene, render RGB, step and check phase tracking."""
from common import *
import argparse
import numpy as np
import gymnasium as gym
import torch
from examples.baselines.act import eval_act_imitator as official
from mani_skill.envs.tasks.tabletop.utils import L0_L3_utils


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--env',required=True);cli=ap.parse_args()
    os.chdir(SOURCE);offline()
    level,base=cli.env.split('_',1)
    official.set_l_level(level)
    L0_L3_utils.set_lr_mirror_robot_pose_enabled(False)
    env=gym.make(base,num_envs=1,sim_backend='physx_cpu',obs_mode='rgb',control_mode='pd_joint_pos',
        reward_mode='dense',render_mode='rgb_array',max_episode_steps=500,
        sensor_configs={'shader_pack':'rt-fast'},human_render_camera_configs={'shader_pack':'rt-fast'})
    try:
        obs,info=env.reset(seed=5999)
        e=env.unwrapped
        action={f'panda_wristcam-{i}':np.concatenate([agent.robot.qpos[0,:7].cpu().numpy(),[1.]]).astype('float32') for i,agent in enumerate(e.agent.agents)}
        for i in range(3):obs,reward,terminated,truncated,info=env.step(action)
        tracker=e.reward_tracker
        peaks={k:v.cpu().tolist() for k,v in tracker.get_peak_dict().items()}
        assert peaks and all(np.isfinite(v).all() for v in peaks.values())
        image=env.render()
        assert image is not None and image.shape[-1]==3
        report={'status':'passed','env':cli.env,'level':level,'phase_names':tracker.phase_names,'phase_peaks':peaks,
                'rgb_render_shape':list(image.shape),'action_shapes':{k:list(v.shape) for k,v in action.items()},
                'success_available':'success' in info}
        assert report['success_available']
        write_json(ROOT/'checks/envs'/f'{cli.env}.json',report)
        print('SCENE VERIFIED',cli.env,flush=True)
    finally:env.close();official.clear_l_level()


if __name__=='__main__':main()
