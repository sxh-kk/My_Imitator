"""Call the official ACT rollout, adding read-only phase peaks and paired seeds."""
from common import *
import argparse
import numpy as np
import torch
import gymnasium as gym
from examples.baselines.act import eval_act_imitator as official
from examples.baselines.lerobot_dataset.evaluate_processor import HumanVideoSimEvaluateProcessor, HumanVideoSimEvaluateProcessorConfig
from mani_skill.utils.wrappers.flatten import FlattenRGBDObservationWrapper


class PhaseProbe(gym.Wrapper):
    def reset(self,**kwargs):
        obs,info=self.env.reset(**kwargs)
        info=dict(info)
        info['reproduction_robot_roots']=torch.cat([a.robot.pose.raw_pose for a in self.unwrapped.agent.agents],dim=-1).clone()
        return obs,info

    def step(self,action):
        obs,reward,terminated,truncated,info=self.env.step(action)
        info=dict(info)
        tracker=self.unwrapped.reward_tracker
        info.update(tracker.get_peak_dict())
        info['reproduction_peak_reward_mean']=tracker.total().clone()
        return obs,reward,terminated,truncated,info


def scalar(value):
    a=np.asarray(value)
    return a.item() if a.size==1 else a.tolist()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--condition',required=True,choices=['pretrain15','scratch5','finetune5'])
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--env',required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--episodes',type=int,default=10)
    parser.add_argument('--probe',action='store_true')
    cli=parser.parse_args()
    offline();os.chdir(SOURCE)
    plan=json.loads((ROOT/'plan.json').read_text())
    out=cli.out;out.mkdir(parents=True,exist_ok=True)
    if (out/'result.json').exists():return
    seed_all(20261008)
    argv=sys.argv
    sys.argv=['evaluate','--checkpoint',str(cli.checkpoint),'--eval-config','unused','--output-dir',str(out)]
    args=official.parse_args();sys.argv=argv
    args.device=torch.device('cuda')
    agent,mc=official.load_agent_from_checkpoint(str(cli.checkpoint),args,args.device)
    saved=torch.load(cli.checkpoint,map_location='cpu',weights_only=False)
    translated={k.replace('task_encoder.backbone.','task_encoder.'):v for k,v in saved['agent_state_dict'].items()}
    state=agent.state_dict()
    assert set(state)==set(translated),(set(state)-set(translated),set(translated)-set(state))
    for key,value in state.items():
        assert torch.equal(value.detach().cpu(),translated[key]) and torch.isfinite(value).all(),key
    del saved,translated,state
    proc=HumanVideoSimEvaluateProcessor(HumanVideoSimEvaluateProcessorConfig(
        human_root=str(DATA/'imitator_human_v1'),human_dataset_file=str(ROOT/'prepared/human_eval.json'),
        human_task_description_file=str(SOURCE/'examples/baselines/lerobot_dataset/task_desc/human_desc.json'),
        human_include_depth=False,human_num_frames=mc['num_video_frames'],
        sim_root=str(ROOT/'prepared/sim_rgb'),sim_dataset_file=str(ROOT/'prepared/sim_eval.json'),
        sim_task_description_file=str(SOURCE/'examples/baselines/lerobot_dataset/task_desc/sim_desc.json'),
        task_mapping_file=str(SOURCE/'examples/baselines/lerobot_dataset/task_mapping.json'),
    ))
    seed_all(20261008)
    video=proc.get_video(cli.env,1).detach().clone()
    torch.save(video,out/'human_video.pt')
    proc.get_video=lambda *unused:video.clone()
    agent.task_encoder._load_backbone()
    agent.to('cuda').eval()
    level,base=cli.env.split('_',1)
    env=official.make_eval_envs_with_level(base,level,1,'physx_cpu',
        dict(control_mode='pd_joint_pos',reward_mode='dense',obs_mode='rgb',render_mode='rgb_array',max_episode_steps=500,
             sensor_configs=dict(shader_pack='rt-fast'),human_render_camera_configs=dict(shader_pack='rt-fast')),
        dict(obs_horizon=1),str(out/'videos'),[PhaseProbe,FlattenRGBDObservationWrapper])
    env.envs[0].unwrapped.reward_tracker  # Require the released phase tracker.
    rows=[];active=None;resets=0;steps=0;last_reset=None
    original_reset,original_step=env.reset,env.step
    def reset(*unused,**kwargs):
        nonlocal active,resets,last_reset
        if resets>=cli.episodes:return last_reset
        seed=plan['evaluation']['reset_seed_start']+resets
        obs,info=original_reset(seed=[seed]);resets+=1
        active={'seed':seed,'initial_state':np.asarray(obs['state'][0]).tolist(),
                'initial_rgb_sha256':hashlib.sha256(np.asarray(obs['rgb'][0]).tobytes()).hexdigest(),
                'robot_roots':np.asarray(info['reproduction_robot_roots'][0]).tolist()}
        last_reset=(obs,info)
        return last_reset
    def step(action):
        nonlocal steps
        for value in action.values():assert np.asarray(value).shape==(1,8) and np.isfinite(value).all()
        output=original_step(action);obs,reward,terminated,truncated,info=output
        steps+=1
        assert np.isfinite(reward).all()
        if bool(truncated[0]):
            final=info['final_info'][0]
            active['metrics']={k:scalar(v) for k,v in final['episode'].items()}
            active['phase_peaks']={k.removeprefix('peak_r_'):scalar(v) for k,v in final.items() if k.startswith('peak_r_')}
            assert active['phase_peaks'],cli.env
            threshold=plan['evaluation']['Sub_SR_threshold']
            active['sub_sr_tau_095']=float(np.mean([v>threshold for v in active['phase_peaks'].values()]))
            active['peak_reward_mean']=scalar(final['reproduction_peak_reward_mean'])
            active['final_success']=scalar(final['success'])
            assert bool(active['final_success'])==bool(active['metrics']['success_at_end'])
            rows.append(active.copy())
            write_json(out/'progress.json',{'env':cli.env,'episodes':rows,'finished':len(rows),'planned':cli.episodes})
            # Keep only the first video; RecordEpisode still returns identical observations.
            env.set_attr('_save_video',[False])
        return output
    env.reset,env.step=reset,step
    began=time.time()
    try:
        metrics=official.evaluate_with_task_encoder(cli.episodes,agent,env,
            dict(env_id=cli.env,delta_control=False,pred_horizon=24,temporal_agg=False,light_temporal_agg=True,tagg_window=4,
                 max_timesteps=500,device=torch.device('cuda'),sim_backend='physx_cpu'),
            proc,official.InputMode.VIDEO_ONLY,progress_bar=True,dtw_provider=None,traj_metrics=None)
        assert len(rows)==cli.episodes and steps==cli.episodes*500
        for key in ['success_at_end','success_once']:
            assert abs(float(np.mean(metrics[key]))-np.mean([r['metrics'][key] for r in rows]))<1e-7
        result={'status':'passed','condition':cli.condition,'env':cli.env,'group':'seen' if cli.env in plan['seen_envs'] else 'unseen',
                'checkpoint':str(cli.checkpoint),'checkpoint_sha256':sha256(cli.checkpoint),'episodes':rows,
                'SR':float(np.mean(metrics['success_at_end'])),'success_once':float(np.mean(metrics['success_once'])),
                'Sub_SR_tau_095':float(np.mean([r['sub_sr_tau_095'] for r in rows])),
                'Sub_SR_threshold':0.95,'Sub_SR_protocol':'user-authorized provisional threshold; not confirmed official paper threshold',
                'human_video_sha256':sha256(out/'human_video.pt'),'seconds':time.time()-began,
                'normalization':'unchanged released target task-level metadata'}
        write_json(out/'result.json',result)
        print('EVALUATION COMPLETE',cli.condition,cli.env,result['SR'],result['Sub_SR_tau_095'],flush=True)
    finally:
        env.close();official.clear_l_level()


if __name__=='__main__':main()
