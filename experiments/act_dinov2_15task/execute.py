"""Durable sequential runner for all three ACT-15 reproduction conditions."""
from common import *
import argparse
import fcntl
import subprocess


def verified(path):
    return path.exists() and json.loads(path.read_text()).get('status')=='passed'


def run(script,arguments,log_name):
    logs=ROOT/'logs';logs.mkdir(exist_ok=True)
    cmd=[str(PYTHON),str(ROOT/script),*map(str,arguments)]
    with (logs/log_name).open('a',buffering=1) as log:
        log.write('\nSTART '+time.strftime('%Y-%m-%d %H:%M:%S')+' '+repr(cmd)+'\n')
        child=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,cwd=SOURCE)
        set_status('running_stage',child_pid=child.pid,active_command=script,active_log=str(logs/log_name))
        code=child.wait()
    if code:raise RuntimeError(f'{script} exited {code}; see {logs/log_name}')


def check_ready():
    assert verified(ROOT/'checks/data-download.json'),'Official data not verified'
    assert verified(ROOT/'checks/data-boundaries.json'),'Episode/config audit not passed'
    calibration=ROOT/'calibration/pretrain15_w8/benchmark.json'
    assert verified(calibration),'Batch256 calibration not passed'
    ckpt=ROOT/'calibration/pretrain15_w8/checkpoints/benchmark.pt'
    assert ckpt.is_file() and ckpt.with_suffix('.ready.json').exists(),'Calibration checkpoint missing'
    plan=json.loads((ROOT/'plan.json').read_text())
    assert plan['training']['batch_size']==256 and plan['training']['epochs']==100
    assert set(plan['pretrain_envs']).isdisjoint(plan['unseen_envs'])
    assert len(plan['seen_envs'])==len(plan['unseen_envs'])==20
    for condition in plan['conditions']:
        folder=ROOT/'runs'/condition
        if (folder/'train.jsonl').exists() and not (folder/'complete.json').exists():
            raise RuntimeError(f'Partial training requires diagnosis first: {folder}')
    return plan


def wait_assets():
    required=[ROOT/'checks/assets-download.json',ROOT/'checks/ycb-download.json']
    while not all(verified(p) for p in required):
        set_status('waiting_for_assets',status='running')
        for unit,path in [('imitator-act15-assets.service',required[0]),('imitator-act15-ycb.service',required[1])]:
            if verified(path):continue
            active=subprocess.check_output(['systemctl','--user','show',unit,'--property=ActiveState','--value'],text=True).strip()
            if active not in ['active','activating']:
                raise RuntimeError(f'Missing verified assets and downloader {unit} is {active}; inspect journalctl --user -u {unit}')
        time.sleep(15)


def train(condition,workers):
    complete=ROOT/'runs'/condition/'complete.json'
    if complete.exists():
        result=json.loads(complete.read_text());assert result['status']=='complete'
        assert sha256(result['checkpoint'])==result['checkpoint_sha256']
        return result['checkpoint']
    set_status('training',status='running',condition=condition)
    args=['--condition',condition,'--workers',workers]
    if condition=='finetune5':args+=['--initialize',ROOT/'runs/pretrain15/checkpoints/final_model.pt']
    run('train.py',args,f'train_{condition}.log')
    result=json.loads(complete.read_text())
    assert result['epochs']==100 and result['status']=='complete'
    return result['checkpoint']


def evaluate(condition,checkpoint,envs):
    for index,env in enumerate(envs):
        output=ROOT/'evaluation'/condition/env
        if verified(output/'result.json'):continue
        set_status('evaluation',status='running',condition=condition,eval_env=env,eval_cell=index+1,eval_cells=len(envs))
        run('evaluate.py',['--condition',condition,'--checkpoint',checkpoint,'--env',env,'--out',output,'--episodes',10],f'eval_{condition}_{env}.log')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--check-only',action='store_true')
    cli=parser.parse_args()
    plan=check_ready()
    if cli.check_only:
        print('READY: data, episode boundaries, batch256 calibration and checkpoint verified; main pipeline has not been started.')
        return
    lock=(ROOT/'execution.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (ROOT/'results.json').exists() and json.loads((ROOT/'results.json').read_text()).get('status')=='complete':
        print('Already complete; no duplicate training.',flush=True);return
    previous=json.loads((ROOT/'controller.json').read_text()) if (ROOT/'controller.json').exists() else None
    write_json(ROOT/'controller.json',{'pid':os.getpid(),'started_at':time.time(),'previous':previous})
    telemetry=(ROOT/'gpu-during-study.csv').open('a')
    monitor=subprocess.Popen(['nvidia-smi','--query-gpu=timestamp,utilization.gpu,memory.used,power.draw,temperature.gpu',
        '--format=csv,noheader,nounits','-l','5'],stdout=telemetry,stderr=subprocess.DEVNULL)
    set_status('preflight',status='running',controller_pid=os.getpid(),monitor_pid=monitor.pid)
    try:
        # Exercise the official loader/rollout before the long formal training.
        probe=ROOT/'checks/checkpoint_rollout'
        if not verified(probe/'result.json'):
            run('evaluate.py',['--condition','pretrain15','--checkpoint',ROOT/'calibration/pretrain15_w8/checkpoints/benchmark.pt',
                '--env','L0_TwoRobotPlaceMugRack-v1','--out',probe,'--episodes',1,'--probe'],'checkpoint_rollout.log')
        hashes={p.name:sha256(p) for p in ROOT.glob('*.py')}
        frozen=ROOT/'execution-source-hashes.json'
        if frozen.exists():assert json.loads(frozen.read_text())==hashes,'Execution scripts changed after protocol freeze'
        else:write_json(frozen,hashes)
        provenance=json.loads((ROOT/'checks/data-boundaries.json').read_text())
        assert all(sha256(SOURCE/p)==h for p,h in provenance['upstream_python'].items()),'Upstream Python source changed'
        checkpoints=ROOT/'runs/pretrain15/checkpoints/final_model.pt'
        # Assets can finish downloading while the independent training runs.
        checkpoints=train('pretrain15',8)
        wait_assets()
        for env in plan['seen_envs']+plan['unseen_envs']:
            path=ROOT/'checks/envs'/f'{env}.json'
            if not verified(path):run('probe_env.py',['--env',env],f'scene_{env}.log')
        evaluate('pretrain15',checkpoints,plan['seen_envs']+plan['unseen_envs'])
        scratch=train('scratch5',8)
        evaluate('scratch5',scratch,plan['unseen_envs'])
        fine=train('finetune5',8)
        evaluate('finetune5',fine,plan['unseen_envs'])
        set_status('final_audit',status='running')
        run('summarize.py',[],'summarize.log')
        print('ALL THREE CONDITIONS AND 800 FORMAL ROLLOUTS COMPLETE',flush=True)
    except Exception as exc:
        set_status('needs_diagnosis',status='needs_diagnosis',error=str(exc),failed_at=time.time())
        raise
    finally:
        monitor.terminate();monitor.wait();telemetry.close()


if __name__=='__main__':main()
