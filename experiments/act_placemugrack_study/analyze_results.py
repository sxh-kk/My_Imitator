"""Terminal stability, paired rollouts and failure-stage accounting."""
from common import *
import statistics


def category(episode):
    metrics=episode['metrics']; stages=episode['stage_first_step']
    if metrics['success_at_end']: return 'success_at_end'
    if metrics['success_once']: return 'transient_success_only'
    if stages['is_grasped'] is None: return 'never_grasped'
    if stages['is_mug_above_table'] is None: return 'grasped_without_lift'
    if stages['is_mug_contact_rack'] is None: return 'lifted_without_rack_contact'
    return 'rack_contact_without_success'


def main():
    plan=json.loads((ROOT/'study-plan.json').read_text())
    report={'runs':{},'conditions':{},'paired':{}}
    all_initial={}
    for name in plan['run_queue']:
        r=json.loads((ROOT/'runs'/name/'test/result.json').read_text())
        counts={k:0 for k in ['success_at_end','transient_success_only','never_grasped','grasped_without_lift','lifted_without_rack_contact','rack_contact_without_success']}
        rows={e['seed']:e for e in r['episodes']}
        for seed,e in rows.items():
            if seed in all_initial:
                assert e['initial_scene']==all_initial[seed]['initial_scene']
                assert e['initial_state']==all_initial[seed]['initial_state']
                assert e['initial_rgb_sha256']==all_initial[seed]['initial_rgb_sha256']
            else: all_initial[seed]=e
            counts[category(e)]+=1
        assert sum(counts.values())==100
        report['runs'][name]={'success_once_count':sum(bool(e['metrics']['success_once']) for e in rows.values()),'success_at_end_count':sum(bool(e['metrics']['success_at_end']) for e in rows.values()),'failure_categories':counts,'seed_categories':{str(s):category(e) for s,e in rows.items()}}
    for condition in ['A10','A50']:
        runs=[report['runs'][f'{condition}_s{s}'] for s in [1,2,3]]
        report['conditions'][condition]={}
        for key in ['success_once_count','success_at_end_count']:
            values=[r[key]/100 for r in runs]
            report['conditions'][condition][key]={'per_seed_rates':values,'mean':statistics.mean(values),'sample_std':statistics.stdev(values)}
        report['conditions'][condition]['failure_category_counts_300_rollouts']={key:sum(r['failure_categories'][key] for r in runs) for key in runs[0]['failure_categories']}
    for seed in [1,2,3]:
        a=json.loads((ROOT/'runs'/f'A50_s{seed}'/'test/result.json').read_text())
        b=json.loads((ROOT/'runs'/f'A10_s{seed}'/'test/result.json').read_text())
        a={e['seed']:e['metrics'] for e in a['episodes']}; b={e['seed']:e['metrics'] for e in b['episodes']}
        report['paired'][str(seed)]={metric:{'both_success':sum(bool(a[s][metric]) and bool(b[s][metric]) for s in a),'only_A50_success':sum(bool(a[s][metric]) and not bool(b[s][metric]) for s in a),'only_A10_success':sum(not bool(a[s][metric]) and bool(b[s][metric]) for s in a),'both_fail':sum(not bool(a[s][metric]) and not bool(b[s][metric]) for s in a)} for metric in ['success_once','success_at_end']}
    report['all_test_initial_conditions_identical_across_runs']=True
    write_json(ROOT/'failure_analysis.json',report)
    print(json.dumps(report['conditions'],indent=2))

if __name__=='__main__': main()
