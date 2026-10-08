"""Verify formal counts, provenance, held-out freeze and paired reset states."""
from common import *
import subprocess
import time
import numpy as np
import torch


def main():
    inputs=json.loads((ROOT/'checks/input-provenance.json').read_text())
    for p,h in json.loads((ROOT/'checks/protocol-frozen.json').read_text())['files'].items():
        assert sha256(p)==h,p
    for x in json.loads((ROOT/'checks/pretrained-provenance.json').read_text())['files']:
        assert sha256(x['path'])==x['sha256'],x['path']
    for p,h in inputs['upstream_python'].items():assert sha256(SOURCE/p)==h,p
    for p,h in inputs['old_scripts'].items():assert sha256(OLD/p)==h,p
    assert subprocess.check_output(['git','status','--short'],cwd=SOURCE,text=True)==inputs['git_status_before']
    for p,h in json.loads((ROOT/'execution-source-hashes.json').read_text()).items():assert sha256(ROOT/p)==h,p
    for s in inputs['a50_selections']:assert sha256(s['checkpoint'])==s['checkpoint_sha256']
    for p,h in inputs['human'].items():assert sha256(OLD/'prepared'/p)==h,p
    freeze=json.loads((ROOT/'final-test-selection.json').read_text()); selected={s['run']:s for s in freeze['selections']}
    totals={'training_runs':0,'optimizer_updates':0,'sample_presentations':0,'checkpoints':0,'phase_a_episodes':0,'dev_episodes':0,'test_episodes':0,'videos':0}
    seed_hashes={};rows=[]
    for name,s in selected.items():
        out=ROOT/'runs'/name;cond,seed=name.split('_s');levels=CONDITIONS[cond]
        complete=json.loads((out/'complete.json').read_text());init=json.loads((out/'initialization.json').read_text())
        assert complete['steps']==18000 and complete['samples']==1152000
        assert complete['batch_size']==64 and complete['workers']==8
        expected={l:1152000//len(levels) for l in levels}
        assert complete['level_sample_counts']==expected
        assert init['trainable_sha256']==seed_hashes.setdefault(seed,init['trainable_sha256'])
        totals['training_runs']+=1;totals['optimizer_updates']+=complete['steps'];totals['sample_presentations']+=complete['samples']
        for step in [1800,5400,9000,12600,18000]:
            ck=out/'checkpoints'/f'step_{step:06d}.pt';ready=json.loads(ck.with_suffix('.ready.json').read_text())
            assert sha256(ck)==ready['sha256'] and ready['roundtrip_equal']
            payload=torch.load(ck,map_location='cpu',weights_only=False)
            assert payload['iteration']==step and payload['experiment']['samples']==step*64
            assert payload['lr_scheduler_state_dict']['last_epoch']==step
            assert payload['sampling']['completed_batches']==step
            assert payload['sampling']['extra_rotation']==step%len(levels)
            assert payload['sampling']['counts']==payload['experiment']['level_sample_counts']
            assert hashlib.sha256(payload['sampling']['permutation'].tobytes()).hexdigest()==init['sampling_plan_sha256']
            assert all(torch.isfinite(v).all() for v in payload['agent_state_dict'].values() if v.is_floating_point())
            del payload;totals['checkpoints']+=1
        for role,target in [('best_model.pt',f"step_{s['step']:06d}.pt"),('last.pt','step_018000.pt')]:
            p=out/'checkpoints'/role
            if not p.exists():p.symlink_to(target)
            assert p.is_symlink() and p.resolve()==(p.parent/target).resolve()
        rows.append({'run':name,'steps':complete['steps'],'level_counts':expected,'selected_step':s['step']})
    initial={}; all_outputs=[]
    for p in (ROOT/'phase_a').glob('A50_s*/L*/result.json'):all_outputs.append((p,'phase_a_episodes',3000,100,True))
    for name in selected:
        for step in [1800,5400,9000,12600,18000]:
            for l in LEVELS:all_outputs.append((ROOT/'runs'/name/'dev'/f'step_{step:06d}'/l/'result.json','dev_episodes',4000,20,False))
        for l in LEVELS:all_outputs.append((ROOT/'runs'/name/'test'/l/'result.json','test_episodes',5000,100,True))
    for p,key,first,n,video in all_outputs:
        r=json.loads(p.read_text());assert r['status']=='passed' and len(r['episodes'])==n
        assert r['vector_steps']*4==n*500 and r['policy_queries']*4==n*125
        assert sorted(e['seed'] for e in r['episodes'])==list(range(first,first+n))
        assert r['stats_sha256']==sha256(stats_path(r['condition'],r['level']))
        assert r['human_video_sha256']==inputs['human']['human_ep1.pt']
        if key=='test_episodes':
            name=p.parents[2].name;assert r['checkpoint_sha256']==selected[name]['checkpoint_sha256']
            assert p.stat().st_mtime>=freeze['frozen_at']
        for e in r['episodes']:
            assert e['state_observations']==500
            assert all(0<=x<=1 for x in e['state_clipped_fraction_per_dim'])
            signature={k:e[k] for k in ['initial_scene','initial_state','initial_rgb_sha256','robot_roots','scene_config']}
            index=(key,r['level'],e['seed'])
            assert signature==initial.setdefault(index,signature),(str(p),e['seed'])
        totals[key]+=n
        videos=list((p.parent/'videos').glob('*.mp4'))
        assert len(videos)==(2 if video else 0),(p,len(videos))
        for v in videos:
            meta=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-show_entries','stream=nb_frames','-of','json',str(v)],text=True))
            assert int(meta['streams'][0]['nb_frames'])==501,v
        totals['videos']+=len(videos)
    assert totals=={'training_runs':9,'optimizer_updates':162000,'sample_presentations':10368000,'checkpoints':45,'phase_a_episodes':900,'dev_episodes':2700,'test_episodes':2700,'videos':72},totals
    results=json.loads((ROOT/'results.json').read_text());assert results['final_initial_conditions_paired']
    report={'status':'passed','counts':totals,'rows':rows,'all_core_and_old_scripts_unchanged':True,'all_checkpoint_sha256_verified':True,'all_same_level_seeds_paired':True,'all_tests_after_freeze':True,'completed_at':time.time()}
    write_json(ROOT/'checks/final-audit.json',report)
    print('FINAL AUDIT PASSED',totals,flush=True)


if __name__=='__main__':main()
