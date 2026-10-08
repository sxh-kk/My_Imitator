"""Continue from live frozen transfer into verified calibration and formal runs."""
from common import *
import fcntl
import subprocess
import time


def run(script):
    with open(ROOT/f'{Path(script).stem}.log','a') as log:
        subprocess.run([sys.executable,str(ROOT/script)],stdout=log,stderr=subprocess.STDOUT,check=True)


if __name__=='__main__':
    lock=open(ROOT/'execution.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    gpu=open(ROOT/'gpu-during-study.csv','a')
    monitor=subprocess.Popen(['nvidia-smi','--query-gpu=timestamp,utilization.gpu,memory.used,power.draw,temperature.gpu','--format=csv,noheader,nounits','-l','1'],stdout=gpu,stderr=subprocess.DEVNULL)
    previous=json.loads((ROOT/'controller.json').read_text()) if (ROOT/'controller.json').exists() else None
    controller={'pid':os.getpid(),'monitor_pid':monitor.pid,'started_at':previous['started_at'] if previous else time.time(),'current_process_started_at':time.time()}
    if previous:controller['previous_processes']=previous.get('previous_processes',[])+[{'pid':previous['pid'],'monitor_pid':previous['monitor_pid'],'reason':'reload finalized preflight gates before formal training'}]
    write_json(ROOT/'controller.json',controller)
    try:
        while not (ROOT/'phase_a/summary.json').exists():
            state=json.loads((ROOT/'status.json').read_text())
            if state['status']=='needs_diagnosis':raise RuntimeError(state)
            time.sleep(2)
        for p in [ROOT/'checks/data-boundaries.json',ROOT/'checks/evaluator-equivalence.json',ROOT/'checks/update_equivalence/update-equivalence.json']:
            assert json.loads(p.read_text())['status']=='passed'
        for l in LEVELS:
            assert json.loads((ROOT/'checks/new_checkpoint_rollout'/l/'result.json').read_text())['status']=='passed'
        state=json.loads((ROOT/'status.json').read_text());state.update(status='running',stage='calibration');write_json(ROOT/'status.json',state)
        if not (ROOT/'execution-config.json').exists():run('calibrate.py')
        run('run_queue.py');run('final_audit.py');run('write_report.py')
        state=json.loads((ROOT/'status.json').read_text());state.update(status='complete',stage='complete',finished_at=time.time());write_json(ROOT/'status.json',state)
        print('ALL EXPERIMENTS COMPLETE',flush=True)
    except Exception as exc:
        state=json.loads((ROOT/'status.json').read_text());state.update(status='needs_diagnosis',error=str(exc));write_json(ROOT/'status.json',state);raise
    finally:
        monitor.terminate();monitor.wait();gpu.close()
