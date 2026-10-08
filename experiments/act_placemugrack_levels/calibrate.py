"""Small fixed-B64 calibration; excludes its weights from formal experiments."""
from common import *
import subprocess
import time


def job(name,condition='B012',seed=1):
    out=ROOT/'calibration'/name;out.mkdir(parents=True,exist_ok=True)
    if (out/'complete.json').exists():return None
    cmd=[sys.executable,str(ROOT/'train.py'),'--condition',condition,'--seed',str(seed),'--run',f'calibration/{name}','--steps','550','--benchmark','--workers','8','--batch','64']
    log=open(out/'run.log','w');p=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT);return p,log


def wait(j):
    if j is None:return
    p,log=j;code=p.wait();log.close();assert code==0,(p.pid,code)


if __name__=='__main__':
    assert (ROOT/'phase_a/summary.json').exists(),'Complete frozen transfer first'
    calibration_start=time.time()
    results={}
    start=time.time();wait(job('single'));results['single_wall_seconds']=time.time()-start
    results['single']=json.loads((ROOT/'calibration/single/complete.json').read_text())
    start=time.time();one=job('dual_1',seed=1);two=job('dual_2',seed=2);wait(one);wait(two)
    results['dual_wall_seconds']=time.time()-start
    results['dual']=[json.loads((ROOT/f'calibration/dual_{i}/complete.json').read_text()) for i in [1,2]]
    results['dual_steady_gain']=sum(x['samples_per_second'] for x in results['dual'])/results['single']['samples_per_second']-1
    # Check same-seed initialization in all conditions before formal expansion.
    for condition in ['B0','B01']:wait(job(f'init_{condition}',condition))
    initial={c:json.loads((ROOT/'calibration'/n/'initialization.json').read_text())['trainable_sha256'] for c,n in [('B012','single'),('B0','init_B0'),('B01','init_B01')]}
    assert len(set(initial.values()))==1,initial
    results['initial_trainable_hashes']=initial
    # Mixed overlap is measured at the same sample/rollout workload.
    checkpoint=json.loads((ROOT/'checks/input-provenance.json').read_text())['a50_selections'][0]['checkpoint']
    solo=[sys.executable,str(ROOT/'evaluate.py'),'--checkpoint',checkpoint,'--condition','A50','--level','L2','--out',str(ROOT/'calibration/solo_eval'),'--episodes','20','--seed-start','2900','--num-envs','4','--no-video']
    if not (ROOT/'calibration/solo_eval/result.json').exists():
        with open(ROOT/'calibration/solo_eval.log','w') as log:subprocess.run(solo,stdout=log,stderr=subprocess.STDOUT,check=True)
    results['solo_eval']=json.loads((ROOT/'calibration/solo_eval/result.json').read_text())
    cmd=[sys.executable,str(ROOT/'evaluate.py'),'--checkpoint',checkpoint,'--condition','A50','--level','L2','--out',str(ROOT/'calibration/mixed_eval'),'--episodes','20','--seed-start','2900','--num-envs','4','--no-video']
    start=time.time();training=job('mixed_train')
    with open(ROOT/'calibration/mixed_eval.log','w') as log:
        subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True)
    wait(training);results['mixed_wall_seconds']=time.time()-start
    results['mixed_train']=json.loads((ROOT/'calibration/mixed_train/complete.json').read_text())
    results['mixed_eval']=json.loads((ROOT/'calibration/mixed_eval/result.json').read_text())
    results['mixed_gain']=(results['single_wall_seconds']+results['solo_eval']['wall_seconds'])/results['mixed_wall_seconds']-1
    import csv,datetime
    memory=[]
    for row in csv.reader((ROOT/'gpu-during-study.csv').read_text().splitlines()):
        try:
            stamp=datetime.datetime.strptime(row[0].strip(),'%Y/%m/%d %H:%M:%S.%f').timestamp()
            if stamp>=calibration_start:memory.append(float(row[2])/1024)
        except (ValueError,IndexError):continue
    results['total_gpu_memory_peak_GiB']=max(memory) if memory else None
    assert memory,'GPU telemetry missing'
    results['status']='passed'
    write_json(ROOT/'calibration/results.json',results)
    config={'batch_size':64,'workers':8,'optimizer_updates':18000,'warmup_updates':900,'eval_envs':4,
            'parallel_training_runs':2 if results['dual_steady_gain']>=.15 and max(memory)<40 else 1,
            'async_eval':results['mixed_gain']>=.15 and max(memory)<40,'max_top_level_gpu_jobs':2}
    write_json(ROOT/'execution-config.json',config)
    print('CALIBRATION PASSED',json.dumps({'dual_gain':results['dual_steady_gain'],'mixed_gain':results['mixed_gain'],'config':config}),flush=True)
