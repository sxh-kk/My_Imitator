"""Aggregate completed fixed-protocol runs without selecting on final test."""
from common import *
import statistics


def select_run(run_dir):
    results=[]
    for path in sorted((run_dir/'dev').glob('step_*/result.json')):
        r=json.loads(path.read_text())
        assert r['status']=='passed' and len(r['episodes'])==20
        results.append(r)
    if len(results)!=5: raise RuntimeError(f'Expected five development checkpoints: {run_dir} ({len(results)})')
    best=max(results,key=lambda r:(r['summary']['success_once']['mean'],r['summary']['success_at_end']['mean'],-r['checkpoint_step']))
    return {'run':run_dir.name,'checkpoint':best['checkpoint'],'checkpoint_sha256':best['checkpoint_sha256'],'step':best['checkpoint_step'],'dev_success_once':best['summary']['success_once']['mean'],'dev_success_at_end':best['summary']['success_at_end']['mean'],'development_results':[{'step':r['checkpoint_step'],'success':r['summary']['success_once']['mean'],'stages':r['stage_counts']} for r in results]}


def main():
    runs=ROOT/'runs'
    plan=json.loads((ROOT/'study-plan.json').read_text())
    result={'status':'complete','runs':[],'conditions':{},'paired_differences':[]}
    for name in plan['run_queue']:
        selected=select_run(runs/name)
        test=json.loads((runs/name/'test/result.json').read_text())
        assert test['checkpoint_sha256']==selected['checkpoint_sha256']
        assert len(test['episodes'])==100 and test['status']=='passed'
        selected.update(test_successes=sum(bool(e['metrics']['success_once']) for e in test['episodes']),test_episodes=100,test_success_rate=test['summary']['success_once']['mean'],test_stage_counts=test['stage_counts'])
        result['runs'].append(selected)
    by_name={r['run']:r for r in result['runs']}
    for condition in ['A50','A10']:
        rates=[by_name[f'{condition}_s{s}']['test_success_rate'] for s in [1,2,3]]
        result['conditions'][condition]={'rates':rates,'mean':statistics.mean(rates),'sample_std':statistics.stdev(rates)}
    for seed in [1,2,3]:
        a=json.loads((runs/f'A50_s{seed}'/'initialization.json').read_text())
        b=json.loads((runs/f'A10_s{seed}'/'initialization.json').read_text())
        assert a['state_sha256']==b['state_sha256'],'Paired initializations differ'
        result['paired_differences'].append(by_name[f'A50_s{seed}']['test_success_rate']-by_name[f'A10_s{seed}']['test_success_rate'])
    write_json(ROOT/'results.json',result)
    print(json.dumps(result,indent=2))

if __name__=='__main__': main()
