"""Instrument the official evaluator with fixed video, reset seeds and checks."""
from common import *
import argparse
import fcntl
import time
import numpy as np
import torch
import gymnasium as gym
from examples.baselines.act import eval_act_imitator as official
from mani_skill.utils.wrappers.flatten import FlattenRGBDObservationWrapper

STAGES=['is_grasped','is_mug_above_table','is_mug_contact_rack','is_obj_placed','success']


class SceneProbe(gym.Wrapper):
    """Read-only initial object poses, exposed through vector reset info."""
    def reset(self, **kwargs):
        obs,info=self.env.reset(**kwargs)
        info=dict(info)
        info['study_initial_scene']=torch.cat([self.unwrapped.mug.pose.raw_pose,self.unwrapped.rack.pose.raw_pose],dim=-1).clone()
        return obs,info


def as_scalar(v):
    arr=np.asarray(v)
    return arr.item() if arr.size==1 else arr.tolist()


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--checkpoint',required=True)
    ap.add_argument('--condition',choices=['A50','A10','D2'],required=True)
    ap.add_argument('--out',required=True)
    ap.add_argument('--seed-start',type=int,default=1000)
    ap.add_argument('--episodes',type=int,default=20)
    ap.add_argument('--num-envs',type=int,default=1)
    ap.add_argument('--human-episode',type=int,default=1)
    ap.add_argument('--no-video',action='store_true')
    ap.add_argument('--video-batches',type=int,default=2)
    cli=ap.parse_args()
    # Two training runs may reach a milestone together. Serialize evaluators;
    # the other run may continue training, keeping at most two active GPU jobs.
    eval_lock=open(ROOT/'evaluation.lock','a')
    fcntl.flock(eval_lock,fcntl.LOCK_EX)
    out=Path(cli.out); out.mkdir(parents=True,exist_ok=True)
    if (out/'result.json').exists(): raise RuntimeError('Evaluation output already complete')
    assert cli.episodes%cli.num_envs==0
    seed_all(20261006)
    # The independent upstream evaluator leaves TF32 disabled by default.
    # Training's TF32 switch must not leak through the shared seed helper.
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    begin=time.time()
    # Reuse official CLI defaults to reconstruct checkpoint exactly.
    saved_argv=sys.argv
    sys.argv=['eval','--checkpoint',cli.checkpoint,'--eval-config','unused','--output-dir',str(out)]
    args=official.parse_args(); sys.argv=saved_argv
    args.device=torch.device('cuda')
    agent,model_config=official.load_agent_from_checkpoint(cli.checkpoint,args,args.device)
    ckpt=torch.load(cli.checkpoint,map_location='cpu',weights_only=False)
    saved={k.replace('.video_encoder.original.','.video_encoder.').replace('.lang_encoder.original.','.lang_encoder.').replace('task_encoder.backbone.','task_encoder.'):v for k,v in ckpt['agent_state_dict'].items()}
    restored=agent.state_dict()
    # Check every loaded state tensor, including buffers; frozen DINO is lazy.
    assert set(restored)<=set(saved),(set(restored)-set(saved))
    for k,v in restored.items():
        assert torch.equal(v.detach().cpu(),saved[k]),k
        assert torch.isfinite(v).all(),k
    unexpected=set(saved)-set(restored)
    assert all(k.startswith('task_encoder._backbone.') for k in unexpected),unexpected
    trainable=[n for n,p in agent.named_parameters() if p.requires_grad]
    assert all(n in saved for n in trainable)
    steps=sorted({int(v['step']) for v in ckpt['optimizer_state_dict']['state'].values()})
    assert steps==[ckpt['iteration']],steps
    assert ckpt['lr_scheduler_state_dict']['last_epoch']==ckpt['iteration']
    expected_stats=sha256(ROOT/'prepared'/cli.condition/'sim'/TASK/'meta/stats.json')
    if 'experiment' in ckpt: assert ckpt['experiment']['stats_sha256']==expected_stats
    report={'status':'running','checkpoint':str(Path(cli.checkpoint).resolve()),'checkpoint_sha256':sha256(cli.checkpoint),'checkpoint_step':ckpt['iteration'],'condition':cli.condition,'num_envs':cli.num_envs,'human_episode':cli.human_episode,'human_video_sha256':sha256(ROOT/'prepared'/f'human_ep{cli.human_episode}.pt'),'stats_sha256':expected_stats,'all_loaded_tensors_equal':len(restored),'trainable_tensor_count':len(trainable),'optimizer_steps':steps,'episodes':[],'policy_queries':0,'vector_steps':0,'auxiliary_dtw':False,'video_recording':not cli.no_video}
    report['preselected_video_seeds']=[] if cli.no_video else list(range(cli.seed_start,cli.seed_start+min(cli.episodes,cli.video_batches*cli.num_envs),cli.num_envs))
    del ckpt,saved,restored
    proc=processor(cli.condition,cli.human_episode)
    video=torch.load(ROOT/'prepared'/f'human_ep{cli.human_episode}.pt',weights_only=True)
    proc.get_video=lambda task,num_envs: video.expand(num_envs,-1,-1,-1,-1).clone()
    # Load immutable pretrained DINO before timing rollouts.
    agent.task_encoder._load_backbone()
    agent.to('cuda').eval()
    original_action=agent.get_action
    def checked_action(obs):
        action=original_action(obs)
        assert tuple(action.shape)==(cli.num_envs,24,16),action.shape
        assert torch.isfinite(action).all()
        report['policy_queries']+=1
        if report['policy_queries']==1:
            report['first_policy_action']=action[:,0].detach().cpu().tolist()
        return action
    agent.get_action=checked_action
    env=official.make_eval_envs_with_level(
        base_env_name='TwoRobotPlaceMugRack-v1',level='L0',num_envs=cli.num_envs,sim_backend='physx_cpu',
        env_kwargs=dict(control_mode='pd_joint_pos',reward_mode='dense',obs_mode='rgb',render_mode='rgb_array',max_episode_steps=500,sensor_configs=dict(shader_pack='rt-fast'),human_render_camera_configs=dict(shader_pack='rt-fast')),
        other_kwargs=dict(obs_horizon=1),video_dir=None if cli.no_video else str(out/'videos'),
        wrappers=[FlattenRGBDObservationWrapper,SceneProbe])
    reset=env.reset; step=env.step
    active=[]; next_offset=0; cached_reset=None; episode_step=0
    def checked_reset(*unused,**kwargs):
        nonlocal active,next_offset,cached_reset,episode_step
        if next_offset>=cli.episodes:
            return cached_reset  # official loop performs one unused reset after its final episode
        seeds=list(range(cli.seed_start+next_offset,cli.seed_start+next_offset+cli.num_envs))
        obs,info=reset(seed=seeds)
        active=[]; episode_step=0
        poses=info['study_initial_scene']
        for i,seed in enumerate(seeds):
            active.append({'seed':seed,'initial_scene':np.asarray(poses[i]).tolist(),'initial_state':np.asarray(obs['state'][i]).tolist(),'initial_rgb_sha256':hashlib.sha256(np.asarray(obs['rgb'][i]).tobytes()).hexdigest(),'stage_first_step':{k:None for k in STAGES}})
        next_offset+=cli.num_envs
        cached_reset=(obs,info)
        return obs,info
    def checked_step(action):
        nonlocal episode_step
        for k,v in action.items():
            assert np.asarray(v).shape==(cli.num_envs,8),(k,np.asarray(v).shape)
            assert np.isfinite(v).all()
        result=step(action)
        obs,reward,terminated,truncated,info=result
        report['vector_steps']+=1; episode_step+=1
        assert np.isfinite(reward).all()
        assert all(np.isfinite(v).all() for v in obs.values())
        for i in range(cli.num_envs):
            if bool(truncated[i]):
                current=info['final_info'][i]
            else:
                current={k:np.asarray(info[k])[i] for k in STAGES if k in info}
            for k in STAGES:
                if k in current and bool(np.asarray(current[k]).any()) and active[i]['stage_first_step'][k] is None:
                    active[i]['stage_first_step'][k]=episode_step
            if bool(truncated[i]):
                assert episode_step==500,episode_step
                active[i]['metrics']={k:as_scalar(v) for k,v in current['episode'].items()}
                active[i]['final_flags']={k:as_scalar(current[k]) for k in STAGES if k in current}
                report['episodes'].append(active[i])
        if np.asarray(truncated).any():
            assert np.asarray(truncated).all()
            if not cli.no_video and len(report['episodes'])==cli.video_batches*cli.num_envs:
                # The vector autoreset has already flushed the last selected video.
                # Disable only further recording; observations/physics are unchanged.
                env.set_attr('_save_video',[False]*cli.num_envs)
            write_json(out/'progress.json',report)
        return result
    env.reset=checked_reset; env.step=checked_step
    rollout_start=time.time()
    try:
        metrics=official.evaluate_with_task_encoder(cli.episodes,agent,env,
            dict(env_id=TASK,delta_control=False,pred_horizon=24,temporal_agg=False,light_temporal_agg=True,tagg_window=4,max_timesteps=500,device=torch.device('cuda'),sim_backend='physx_cpu'),
            proc,official.InputMode.VIDEO_ONLY,progress_bar=True,dtw_provider=None,traj_metrics=None)
        assert len(report['episodes'])==cli.episodes
        assert report['vector_steps']==cli.episodes//cli.num_envs*500
        assert report['policy_queries']==cli.episodes//cli.num_envs*125
        assert sorted(r['seed'] for r in report['episodes'])==list(range(cli.seed_start,cli.seed_start+cli.episodes))
        report['summary']={k:{'mean':float(np.mean(v)),'std':float(np.std(v))} for k,v in metrics.items()}
        for key in ['success_once','success_at_end']:
            assert abs(report['summary'][key]['mean']-np.mean([r['metrics'][key] for r in report['episodes']]))<1e-7
        report['stage_counts']={k:sum(r['stage_first_step'][k] is not None for r in report['episodes']) for k in STAGES}
        report['rollout_seconds']=time.time()-rollout_start
        report['status']='passed'
    finally:
        env.close()
        official.clear_l_level()
        report['wall_seconds']=time.time()-begin
        write_json(out/('result.json' if report['status']=='passed' else 'failure.json'),report)
    print('EVALUATION PASSED',json.dumps({k:v for k,v in report.items() if k not in ['episodes']}),flush=True)

if __name__=='__main__':
    main()
