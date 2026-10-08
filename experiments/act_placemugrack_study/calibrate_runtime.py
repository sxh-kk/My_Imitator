from common import *
import argparse
import atexit
import subprocess
import time
import numpy as np


def launch(command,name):
    out=ROOT/'calibration'/name
    log=open(str(out)+'.log','w')
    proc=subprocess.Popen([sys.executable]+command,stdout=log,stderr=subprocess.STDOUT)
    return proc,log


def wait_job(job):
    p,log=job
    code=p.wait(); log.close()
    if code: raise RuntimeError(f'Calibration process {p.pid} failed: {code}')


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--workers',type=int,required=True); ap.add_argument('--batch',type=int,default=64)
    cli=ap.parse_args()
    root=ROOT/'calibration'
    gpu=open(root/'runtime_gpu.csv','w')
    monitor=subprocess.Popen(['nvidia-smi','--query-gpu=timestamp,utilization.gpu,memory.used,power.draw,temperature.gpu','--format=csv,noheader,nounits','-l','1'],stdout=gpu,stderr=subprocess.DEVNULL)
    atexit.register(monitor.terminate)
    checkpoint=root/'b64_w4/checkpoints/step_000250.pt'
    results=json.loads((root/'runtime_results.json').read_text()) if (root/'runtime_results.json').exists() else {'workers':cli.workers,'batch':cli.batch}
    if 'dual' not in results:
        jobs=[]
        start=time.time()
        for i in [1,2]:
            name=f'dual_{i}'
            jobs.append(launch([str(ROOT/'train.py'),'--run',f'calibration/{name}','--batch',str(cli.batch),'--workers',str(cli.workers),'--steps','550','--benchmark'],name))
        for job in jobs: wait_job(job)
        results['dual_wall_seconds']=time.time()-start
        results['dual']=[json.loads((root/f'dual_{i}/complete.json').read_text()) for i in [1,2]]
        write_json(root/'runtime_results.json',results)
    if 'single' not in results:
        start=time.time()
        wait_job(launch([str(ROOT/'train.py'),'--run','calibration/single_long','--batch',str(cli.batch),'--workers',str(cli.workers),'--steps','550','--benchmark'],'single_long'))
        results['single_wall_seconds']=time.time()-start
        results['single']=json.loads((root/'single_long/complete.json').read_text())
        write_json(root/'runtime_results.json',results)
    evals=results.get('evaluations',{})
    for n in [1,2,4]:
        if str(n) in evals: continue
        name=f'eval_n{n}'
        start=time.time()
        wait_job(launch([str(ROOT/'evaluate.py'),'--checkpoint',str(checkpoint),'--condition','A50','--out',str(root/name),'--seed-start','900','--episodes','4','--num-envs',str(n),'--no-video'],name))
        report=json.loads((root/name/'result.json').read_text())
        evals[str(n)]={'wall_seconds':time.time()-start,'rollout_seconds':report['rollout_seconds'],'summary':report['summary']}
        if n>1:
            one=json.loads((root/'eval_n1/result.json').read_text())
            by_seed={r['seed']:r for r in one['episodes']}
            assert all(np.allclose(r['initial_scene'],by_seed[r['seed']]['initial_scene'],atol=1e-7,rtol=0) for r in report['episodes'])
            assert all(np.allclose(r['initial_state'],by_seed[r['seed']]['initial_state'],atol=1e-7,rtol=0) for r in report['episodes'])
            evals[str(n)]['initial_scene_and_state_match']=True
            evals[str(n)]['rgb_hashes_equal']=all(r['initial_rgb_sha256']==by_seed[r['seed']]['initial_rgb_sha256'] for r in report['episodes'])
            first=np.array(one['first_policy_action'][0]); candidate=np.array(report['first_policy_action'][0])
            evals[str(n)]['first_action_max_difference']=float(np.max(np.abs(first-candidate)))
            assert np.allclose(first,candidate,rtol=1e-4,atol=1e-5)
        results['evaluations']=evals
        write_json(root/'runtime_results.json',results)
    fastest=min(evals,key=lambda k:evals[k]['rollout_seconds'])
    results['chosen_eval_envs']=int(fastest)
    start=time.time()
    job_train=launch([str(ROOT/'train.py'),'--run','calibration/mixed_train','--batch',str(cli.batch),'--workers',str(cli.workers),'--steps','550','--benchmark'],'mixed_train')
    job_eval=launch([str(ROOT/'evaluate.py'),'--checkpoint',str(checkpoint),'--condition','A50','--out',str(root/'mixed_eval'),'--seed-start','900','--episodes','4','--num-envs',fastest,'--no-video'],'mixed_eval')
    wait_job(job_train); wait_job(job_eval)
    results['mixed_wall_seconds']=time.time()-start
    results['mixed_train']=json.loads((root/'mixed_train/complete.json').read_text())
    results['mixed_eval']=json.loads((root/'mixed_eval/result.json').read_text())['rollout_seconds']
    results['dual_steady_gain']=sum(r['samples_per_second'] for r in results['dual'])/results['single']['samples_per_second']-1
    results['dual_whole_job_gain']=2*results['single_wall_seconds']/results['dual_wall_seconds']-1
    results['mixed_whole_job_gain']=(results['single_wall_seconds']+evals[fastest]['wall_seconds'])/results['mixed_wall_seconds']-1
    write_json(root/'runtime_results.json',results)
    print(json.dumps(results,indent=2),flush=True)

if __name__=='__main__': main()
