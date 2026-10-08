from common import *
import time
s=ROOT/'status.json'
if s.exists():
 status=json.loads(s.read_text())
 print('STATUS',status['status'],'completed',status.get('completed_runs',[]),'active',status.get('active_runs',[]),'test',status.get('active_test'))
for run in sorted((ROOT/'runs').glob('*')):
 f=run/'train.jsonl'
 if not f.exists(): continue
 last=json.loads(f.read_text().splitlines()[-1])
 dev=[]
 for p in sorted((run/'dev').glob('*/result.json')):
  r=json.loads(p.read_text());dev.append(f"{r['checkpoint_step']}:{r['summary']['success_once']['mean']:.0%}/{r['summary']['success_at_end']['mean']:.0%}")
 record={'run':run.name,'step':last['step'],'loss':round(last['loss'],5),'dev_once/end':dev,'complete':(run/'complete.json').exists()}
 if (run/'test/result.json').exists():
  r=json.loads((run/'test/result.json').read_text());record['test_once/end']=[r['summary']['success_once']['mean'],r['summary']['success_at_end']['mean']]
 elif (run/'test/progress.json').exists():
  r=json.loads((run/'test/progress.json').read_text());record['test_episodes']=len(r['episodes'])
 for p in (run/'dev').glob('*/progress.json'):
  if not (p.parent/'result.json').exists():record['eval_progress']=[p.parent.name,len(json.loads(p.read_text())['episodes'])]
 print(json.dumps(record))
if (ROOT/'gpu-during-study.csv').exists():
 print('GPU', (ROOT/'gpu-during-study.csv').read_text().splitlines()[-1])
