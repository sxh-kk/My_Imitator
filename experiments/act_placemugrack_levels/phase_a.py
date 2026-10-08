"""Run frozen A50 transfer before any new policy training."""
from common import *
import subprocess
import time


def launch(checkpoint,level,out,seeds,episodes,no_video=False):
    if (out/'result.json').exists(): return json.loads((out/'result.json').read_text())
    out.parent.mkdir(parents=True,exist_ok=True)
    cmd=[sys.executable,str(ROOT/'evaluate.py'),'--checkpoint',checkpoint,'--condition','A50','--level',level,'--out',str(out),'--seed-start',str(seeds),'--episodes',str(episodes),'--num-envs','4']
    if no_video: cmd.append('--no-video')
    with open(str(out)+'.log','w') as log: subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True)
    return json.loads((out/'result.json').read_text())


if __name__=='__main__':
    provenance=json.loads((ROOT/'checks/input-provenance.json').read_text())
    state={'status':'running','stage':'phase_a_preflight','started_at':time.time(),'completed_phase_a':[]}
    write_json(ROOT/'status.json',state)
    selected=provenance['a50_selections']
    try:
        for level in LEVELS:
            launch(selected[0]['checkpoint'],level,ROOT/'checks'/f'level_smoke_{level}',2800,4,True)
        source={p.name:sha256(p) for p in [ROOT/'common.py',ROOT/'evaluate.py',ROOT/'evaluate_levels.py']}
        write_json(ROOT/'checks/phase-a-source-hashes.json',source)
        state['stage']='phase_a'; write_json(ROOT/'status.json',state)
        rows=[]; initial={l:{} for l in LEVELS}
        for s in selected:
            for level in LEVELS:
                label=f"{s['run']}/{level}";state['active_phase_a']=label;write_json(ROOT/'status.json',state)
                r=launch(s['checkpoint'],level,ROOT/'phase_a'/s['run']/level,3000,100)
                assert r['checkpoint_sha256']==s['checkpoint_sha256']
                assert r['stats_sha256']==provenance['a50_stats_sha256']
                assert r['human_video_sha256']==provenance['human']['human_ep1.pt']
                for ep in r['episodes']:
                    key=ep['seed']; value={k:ep[k] for k in ['initial_scene','initial_state','initial_rgb_sha256','robot_roots','scene_config']}
                    if key in initial[level]: assert initial[level][key]==value,(label,key)
                    else: initial[level][key]=value
                rows.append({'run':s['run'],'level':level,'step':s['step'],'summary':r['summary'],'stages':r['stage_counts'],'wall_seconds':r['wall_seconds'],'rollout_seconds':r['rollout_seconds']})
                state['completed_phase_a'].append(label);write_json(ROOT/'status.json',state)
                print('PHASE A DONE',label,r['summary']['success_at_end']['mean'],flush=True)
        assert len(rows)==9
        assert all(sha256(ROOT/n)==h for n,h in source.items())
        write_json(ROOT/'phase_a/summary.json',{'status':'complete','episodes':900,'rows':rows,'paired_initial_conditions':True,'finished_at':time.time()})
        state.update(status='phase_a_complete',stage='data_preflight',active_phase_a=None,finished_phase_a=time.time())
        write_json(ROOT/'status.json',state)
    except Exception as exc:
        state.update(status='needs_diagnosis',error=str(exc));write_json(ROOT/'status.json',state);raise
