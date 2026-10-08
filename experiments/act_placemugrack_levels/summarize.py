"""Predeclared checkpoint selection and independent test aggregation."""
from common import *
import collections
import time
import numpy as np


def select_run(name):
    out=ROOT/'runs'/name; rows=[]
    for step in [1800,5400,9000,12600,18000]:
        level_results={l:json.loads((out/'dev'/f'step_{step:06d}'/l/'result.json').read_text()) for l in LEVELS}
        rows.append({'step':step,'macro_success_at_end':float(np.mean([r['summary']['success_at_end']['mean'] for r in level_results.values()])),
                     'macro_success_once':float(np.mean([r['summary']['success_once']['mean'] for r in level_results.values()])),
                     'levels':{l:{k:r['summary'][k]['mean'] for k in ['success_at_end','success_once']} for l,r in level_results.items()}})
    best=max(rows,key=lambda x:(x['macro_success_at_end'],x['macro_success_once'],-x['step']))
    checkpoint=out/'checkpoints'/f"step_{best['step']:06d}.pt"
    ready=json.loads(checkpoint.with_suffix('.ready.json').read_text())
    assert sha256(checkpoint)==ready['sha256']
    result={'run':name,'checkpoint':str(checkpoint),'checkpoint_sha256':ready['sha256'],'step':best['step'],'macro_dev_success_at_end':best['macro_success_at_end'],'macro_dev_success_once':best['macro_success_once'],'development':rows}
    write_json(out/'selection.json',result)
    return result


def category(ep):
    if ep['metrics']['success_at_end']:return 'success_at_end'
    if ep['metrics']['success_once']:return 'transient_success_only'
    first=ep['stage_first_step']
    if first['is_grasped'] is None:return 'never_grasped'
    if first['is_mug_above_table'] is None:return 'grasped_without_lift'
    if first['is_mug_contact_rack'] is None:return 'lifted_without_rack_contact'
    return 'rack_contact_without_success'


def aggregate_runs(rows):
    summary={}
    for cond in ['A50','B0','B01','B012']:
        values=[r for r in rows if r['condition']==cond]
        if not values:continue
        summary[cond]={}
        for level in LEVELS:
            selected=[r for r in values if r['level']==level]
            summary[cond][level]={}
            for metric in ['success_at_end','success_once']:
                x=np.array([r[metric] for r in selected])
                summary[cond][level][metric]={'mean':float(x.mean()),'sample_std':float(x.std(ddof=1)),'per_training_seed':x.tolist()}
    return summary


def main():
    plan=json.loads((ROOT/'plan.json').read_text()); frozen=json.loads((ROOT/'final-test-selection.json').read_text())
    rows=[]; failures={}; initial={l:{} for l in LEVELS}
    for selected in frozen['selections']:
        name=selected['run'];cond,seed=name.split('_s')
        for l in LEVELS:
            r=json.loads((ROOT/'runs'/name/'test'/l/'result.json').read_text())
            counts=collections.Counter(category(e) for e in r['episodes'])
            assert r['checkpoint_sha256']==selected['checkpoint_sha256']
            assert r['episodes'] and len(r['episodes'])==100
            for ep in r['episodes']:
                value={k:ep[k] for k in ['initial_scene','initial_state','initial_rgb_sha256','robot_roots','scene_config']}
                if ep['seed'] in initial[l]:assert initial[l][ep['seed']]==value,(name,l,ep['seed'])
                else:initial[l][ep['seed']]=value
            rows.append({'run':name,'condition':cond,'training_seed':int(seed),'level':l,'selected_step':selected['step'],
                         'success_at_end':r['summary']['success_at_end']['mean'],'success_once':r['summary']['success_once']['mean'],
                         'success_at_end_count':sum(bool(e['metrics']['success_at_end']) for e in r['episodes']),
                         'success_once_count':sum(bool(e['metrics']['success_once']) for e in r['episodes']),
                         'state_clipped_fraction_per_dim':np.mean([e['state_clipped_fraction_per_dim'] for e in r['episodes']],axis=0).tolist(),
                         'arm_ever_grasped_counts':[sum(e['arm_first_grasp_step'][a] is not None for e in r['episodes']) for a in [0,1]],
                         'gripper_command_min':np.min([e['gripper_command_min'] for e in r['episodes']],axis=0).tolist(),
                         'gripper_command_max':np.max([e['gripper_command_max'] for e in r['episodes']],axis=0).tolist()})
            failures[f'{name}/{l}']={'counts':dict(counts),'seeds':{str(e['seed']):category(e) for e in r['episodes']}}
    pairs={}
    by={(r['condition'],r['training_seed'],r['level']):r for r in rows}
    for high,low in [('B01','B0'),('B012','B01'),('B012','B0')]:
        pairs[f'{high}-{low}']={l:{k:[by[(high,s,l)][k]-by[(low,s,l)][k] for s in [1,2,3]] for k in ['success_at_end','success_once']} for l in LEVELS}
    phase=[]
    for p in sorted((ROOT/'phase_a').glob('A50_s*/L*/result.json')):
        r=json.loads(p.read_text());name=p.parent.parent.name
        phase.append({'run':name,'condition':'A50','training_seed':int(name.split('_s')[1]),'level':r['level'],**{k:r['summary'][k]['mean'] for k in ['success_at_end','success_once']}})
        failures[f'phase_a/{name}/{r["level"]}']={'counts':dict(collections.Counter(category(e) for e in r['episodes'])),'seeds':{str(e['seed']):category(e) for e in r['episodes']}}
    output={'status':'complete','completed_at':time.time(),'primary_metric':'success_at_end','rows':rows,'conditions':aggregate_runs(rows),
            'paired_differences':pairs,'frozen_transfer':{'rows':phase,'conditions':aggregate_runs(phase)},'final_initial_conditions_paired':True,
            'main_episode_counts':{'phase_a':900,'development':2700,'test':2700,'total':6300}}
    write_json(ROOT/'results.json',output)
    write_json(ROOT/'failure_analysis.json',{'status':'complete','paired_initial_conditions':True,'cells':failures})
    print('SUMMARY COMPLETE',json.dumps(output['conditions']),flush=True)


if __name__=='__main__':main()
