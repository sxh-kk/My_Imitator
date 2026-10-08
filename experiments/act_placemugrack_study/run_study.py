"""Execute the frozen study queue; resume only at completed-run boundaries."""
from common import *
import subprocess
import time
from summarize import select_run


def main():
    config=json.loads((ROOT/'execution-config.json').read_text())
    plan=json.loads((ROOT/'study-plan.json').read_text())
    provenance={p.name:sha256(p) for p in ROOT.glob('*.py')}
    write_json(ROOT/'execution-source-hashes.json',provenance)
    state={'status':'running','started_at':time.time(),'completed_runs':[],'active_runs':[],'config':config}
    def status(): write_json(ROOT/'status.json',state)
    gpu=open(ROOT/'gpu-during-study.csv','a')
    monitor=subprocess.Popen(['nvidia-smi','--query-gpu=timestamp,utilization.gpu,memory.used,power.draw,temperature.gpu','--format=csv,noheader,nounits','-l','1'],stdout=gpu,stderr=subprocess.DEVNULL)
    jobs={}
    def launch(name,pilot=False):
        out=ROOT/'runs'/name
        out.mkdir(parents=True,exist_ok=True)
        if (out/'complete.json').exists():
            state['completed_runs'].append(name); return
        if (out/'train.jsonl').exists():
            raise RuntimeError(f'Partial run requires diagnosis/restart in a new directory: {name}')
        condition,seed=name.split('_s')
        command=[sys.executable,str(ROOT/'train.py'),'--run',f'runs/{name}','--condition',condition,'--seed',seed,'--batch',str(config['batch_size']),'--workers',str(config['workers']),'--steps',str(config['optimizer_steps']),'--warmup',str(config['warmup_steps']),'--evaluate','--eval-envs',str(config['eval_envs'])]
        if pilot: command+=['--pilot']
        if config['async_eval'] and (pilot or config['parallel_training_runs']==1): command+=['--async-eval']
        log=open(out/'run.log','w')
        process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT)
        jobs[name]=(process,log)
        state['active_runs']=list(jobs); status()
        print('START',name,'PID',process.pid,flush=True)
    def finish(name):
        process,log=jobs.pop(name)
        code=process.wait(); log.close()
        out=ROOT/'runs'/name
        if code or not (out/'complete.json').exists():
            state.update(status='needs_diagnosis',failed_run=name,exit_code=code)
            status()
            raise RuntimeError(f'Run {name} needs diagnosis, exit={code}')
        select_run(out)  # all five checkpoints actually evaluated
        state['completed_runs'].append(name)
        state['active_runs']=list(jobs); status()
        print('FINISH',name,flush=True)
    try:
        launch('A50_s1',pilot=True)
        if 'A50_s1' in jobs: finish('A50_s1')
        pending=[r for r in plan['run_queue'] if r!='A50_s1']
        while pending or jobs:
            while pending and len(jobs)<config['parallel_training_runs']:
                launch(pending.pop(0))
            done=[name for name,(p,_) in jobs.items() if p.poll() is not None]
            for name in done: finish(name)
            if jobs: time.sleep(2)
        # Freeze selections before any final-test outcomes can be observed.
        selections=[select_run(ROOT/'runs'/name) for name in plan['run_queue']]
        freeze=ROOT/'final-test-selection.json'
        if freeze.exists():
            assert json.loads(freeze.read_text())['selections']==selections
        else:
            write_json(freeze,{'frozen_at':time.time(),'selection_protocol':'max dev success_once, then success_at_end, then earlier step','selections':selections})
        state['status']='final_testing'; status()
        for selected in selections:
            name=selected['run']; out=ROOT/'runs'/name/'test'
            if (out/'result.json').exists(): continue
            state['active_test']=name; status()
            command=[sys.executable,str(ROOT/'evaluate.py'),'--checkpoint',selected['checkpoint'],'--condition',name.split('_s')[0],'--out',str(out),'--seed-start','2000','--episodes','100','--num-envs',str(config['eval_envs'])]
            with open(ROOT/'runs'/name/'test.log','w') as log:
                subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
            print('TEST FINISH',name,flush=True)
        subprocess.run([sys.executable,str(ROOT/'summarize.py')],check=True)
        state.update(status='complete',finished_at=time.time(),active_test=None)
        status()
    except Exception as exc:
        state.update(error=str(exc),finished_at=time.time())
        if state['status']!='needs_diagnosis': state['status']='failed'
        status()
        # Let a healthy independent running job finish, but never silently drop it.
        for name,(process,log) in list(jobs.items()):
            code=process.wait(); log.close()
            state.setdefault('remaining_job_exit_codes',{})[name]=code
        status()
        raise
    finally:
        monitor.terminate(); monitor.wait(); gpu.close()

if __name__=='__main__': main()
