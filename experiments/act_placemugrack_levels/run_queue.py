"""Frozen pilot-first ACT queue, then all selected models' held-out tests."""
from common import *
import subprocess
import time
from summarize import select_run


def main():
    config=json.loads((ROOT/'execution-config.json').read_text());plan=json.loads((ROOT/'plan.json').read_text())
    for p in ['data-boundaries.json','evaluator-equivalence.json']:
        assert json.loads((ROOT/'checks'/p).read_text())['status']=='passed'
    assert json.loads((ROOT/'checks/update_equivalence/update-equivalence.json').read_text())['status']=='passed'
    state=json.loads((ROOT/'status.json').read_text());state.update(status='running',stage='training',started_training=time.time(),active_runs=[],completed_runs=[])
    source={p.name:sha256(p) for p in ROOT.glob('*.py')}
    source_path=ROOT/'execution-source-hashes.json'
    if source_path.exists():assert json.loads(source_path.read_text())==source
    else:write_json(source_path,source)
    frozen_protocol=ROOT/'checks/protocol-frozen.json'
    protocol_paths=[ROOT/'plan.json',ROOT/'execution-config.json',ROOT/'prepared/manifest.json']
    protocol_paths += [stats_path(c,l) for c in ['A50',*CONDITIONS] for l in LEVELS]
    protocol_paths += [ROOT/'prepared/human_ep0.pt',ROOT/'prepared/human_ep1.pt',ROOT/'prepared/human_cache_ep0/backbone_dinov2_vitl14/raw_features.pt',ROOT/'prepared/human_cache_ep1/backbone_dinov2_vitl14/raw_features.pt']
    protocol={str(p):sha256(p) for p in protocol_paths}
    if frozen_protocol.exists():assert json.loads(frozen_protocol.read_text())['files']==protocol
    else:write_json(frozen_protocol,{'frozen_at':time.time(),'files':protocol})
    jobs={}
    def status():write_json(ROOT/'status.json',state)
    def launch(name,pilot=False):
        out=ROOT/'runs'/name;out.mkdir(parents=True,exist_ok=True)
        if (out/'complete.json').exists():state['completed_runs'].append(name);return
        if (out/'train.jsonl').exists():raise RuntimeError(f'Partial run needs archive and deterministic restart: {name}')
        cond,seed=name.split('_s')
        cmd=[sys.executable,str(ROOT/'train.py'),'--run',f'runs/{name}','--condition',cond,'--seed',seed,'--batch','64','--workers',str(config['workers']),'--steps','18000','--warmup','900','--evaluate','--eval-envs','4']
        if pilot:cmd.append('--pilot')
        if config['async_eval'] and (pilot or config['parallel_training_runs']==1):cmd.append('--async-eval')
        log=open(out/'run.log','w');p=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT)
        jobs[name]=(p,log);state['active_runs']=list(jobs);status();print('START',name,p.pid,flush=True)
    def finish(name):
        p,log=jobs.pop(name);code=p.wait();log.close()
        if code or not (ROOT/'runs'/name/'complete.json').exists():raise RuntimeError(f'Run failed: {name}, exit={code}')
        select_run(name);state['completed_runs'].append(name);state['active_runs']=list(jobs);status();print('FINISH',name,flush=True)
    try:
        status();launch('B012_s1',True)
        if 'B012_s1' in jobs:finish('B012_s1')
        pending=[x for x in plan['run_queue'] if x!='B012_s1']
        while pending or jobs:
            while pending and len(jobs)<config['parallel_training_runs']:launch(pending.pop(0))
            for name,(p,_) in list(jobs.items()):
                if p.poll() is not None:finish(name)
            if jobs:time.sleep(2)
        selections=[select_run(x) for x in plan['run_queue']]
        frozen={'frozen_at':time.time(),'selection_protocol':plan['selection'],'selections':selections}
        path=ROOT/'final-test-selection.json'
        if path.exists():assert json.loads(path.read_text())['selections']==selections
        else:write_json(path,frozen)
        state.update(stage='final_testing',active_test=None);status()
        for s in selections:
            name=s['run'];out=ROOT/'runs'/name/'test'
            state['active_test']=name;status()
            cmd=[sys.executable,str(ROOT/'evaluate_levels.py'),'--checkpoint',s['checkpoint'],'--condition',name.split('_s')[0],'--out',str(out),'--seed-start','5000','--episodes','100','--num-envs','4']
            with open(ROOT/'runs'/name/'test.log','w') as log:subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True)
            print('TEST DONE',name,flush=True)
        subprocess.run([sys.executable,str(ROOT/'summarize.py')],check=True)
        state.update(status='calculations_complete',stage='final_audit',finished_computation=time.time(),active_test=None);status()
    except Exception as exc:
        state.update(status='needs_diagnosis',error=str(exc));status()
        for name,(p,log) in jobs.items():
            code=p.wait();log.close();state.setdefault('remaining_job_exit_codes',{})[name]=code
        status();raise


if __name__=='__main__':main()
