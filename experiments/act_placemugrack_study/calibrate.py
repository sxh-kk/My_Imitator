from common import *
import subprocess
import time


def main():
    base=ROOT/'calibration'
    base.mkdir(exist_ok=True)
    gpu=open(base/'gpu.csv','w')
    monitor=subprocess.Popen(['nvidia-smi','--query-gpu=timestamp,utilization.gpu,memory.used,power.draw,temperature.gpu','--format=csv,noheader,nounits','-l','1'],stdout=gpu,stderr=subprocess.DEVNULL)
    results=[]
    try:
        for batch,workers in [(32,4),(64,4),(128,4),(64,0),(64,8)]:
            run=f'calibration/b{batch}_w{workers}'
            out=ROOT/run
            if (out/'complete.json').exists():
                results.append(json.loads((out/'complete.json').read_text())); continue
            command=[sys.executable,str(ROOT/'train.py'),'--run',run,'--batch',str(batch),'--workers',str(workers),'--steps','250','--benchmark']
            if batch==64 and workers==4: command+=['--benchmark-save']
            print('START',run,flush=True)
            with open(base/f'b{batch}_w{workers}.log','w') as log:
                subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
            result=json.loads((out/'complete.json').read_text())
            results.append(result)
            write_json(base/'serial_results.json',results)
            print('FINISH',run,json.dumps(result),flush=True)
    finally:
        monitor.terminate(); monitor.wait(); gpu.close()

if __name__=='__main__': main()
