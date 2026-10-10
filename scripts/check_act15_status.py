"""Read process-aware ACT-15 status; historical progress is never 'running'."""
import json
import subprocess
import time
from pathlib import Path

workspace=Path(__file__).resolve().parents[1]
root=workspace/'experiments/act_dinov2_15task'
path=root/'status.json'
state=json.loads(path.read_text()) if path.exists() else {}
show=subprocess.run(['systemctl','--user','show','imitator-act15.service',
    '--property=ActiveState','--property=SubState','--property=MainPID'],capture_output=True,text=True)
unit=dict(line.split('=',1) for line in show.stdout.splitlines() if '=' in line)
mainpid=int(unit.get('MainPID','0'))
live=False
if mainpid:
    cmdline=Path(f'/proc/{mainpid}/cmdline')
    live=cmdline.exists() and str(root/'execute.py') in cmdline.read_bytes().decode(errors='replace')
complete=(root/'checks/completion-verification.json').exists()
actual='complete' if complete else 'running' if live else 'needs_diagnosis' if state.get('status')=='needs_diagnosis' else 'not_running'
report={'actual_status':actual,'service':unit,'last_progress':state,
        'progress_age_seconds':round(time.time()-state.get('updated_at',time.time()))}
report['downloads']={}
for name in ['assets','ycb']:
    marker=root/'checks'/f'{name}-download.json'
    if marker.exists():report['downloads'][name]='verified'
    else:
        proc=subprocess.run(['systemctl','--user','show',f'imitator-act15-{name}.service','--property=ActiveState','--value'],capture_output=True,text=True)
        report['downloads'][name]=proc.stdout.strip() or 'not_loaded'
if not live and not complete:
    report['note']='Main experiment has not started or has stopped; last_progress may describe an already finished calibration.'
report['training_runs']={}
for condition in ['pretrain15','scratch5','finetune5']:
    marker=root/'runs'/condition/'complete.json'
    if marker.exists():
        run=json.loads(marker.read_text())
        report['training_runs'][condition]={k:run[k] for k in ['status','epochs','iterations','elapsed_seconds']}
    elif live and state.get('active_command')=='train.py' and state.get('condition')==condition:
        report['training_runs'][condition]={'status':'running','epoch':state.get('epoch'),
            'iteration':state.get('iteration'),'total_iterations':state.get('total_iterations')}
    else:
        report['training_runs'][condition]={'status':'not_complete'}
report['formal_evaluation']={}
for condition,planned in [('pretrain15',40),('scratch5',20),('finetune5',20)]:
    results=[]
    for path in (root/'evaluation'/condition).glob('*/result.json'):
        cell=json.loads(path.read_text())
        if cell.get('status')!='passed':continue
        if cell['env'].startswith('L3_') and cell.get('registered_env')!=cell['env'].split('_',1)[1].replace('-v','L3-v'):continue
        results.append(cell)
    report['formal_evaluation'][condition]={'completed_cells':len(results),'planned_cells':planned,
        'completed_episodes':sum(len(cell['episodes']) for cell in results)}
repair=root/'checks/l3-repair.json'
if repair.exists():report['l3_repair']=json.loads(repair.read_text())
# These fields used to retain preflight values after the pipeline advanced.
report['last_progress']['main_training_started']=(root/'runs/pretrain15/train.jsonl').exists()
report['last_progress']['assets']=report['downloads']['assets']
if report['last_progress'].get('active_command')=='train.py':
    for field in ['eval_env','eval_cell','eval_cells']:report['last_progress'].pop(field,None)
print(json.dumps(report,ensure_ascii=False,indent=2))
