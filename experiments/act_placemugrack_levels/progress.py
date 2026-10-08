"""Read-only concise progress, safe to run while the study is active."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parent


def show():
    state=json.loads((ROOT/'status.json').read_text()) if (ROOT/'status.json').exists() else {}
    print('status',json.dumps(state,ensure_ascii=False))
    for name in ['phase_a','runs']:
        for p in sorted((ROOT/name).glob('**/progress.json')):
            if (p.parent/'result.json').exists():continue
            j=json.loads(p.read_text())
            print('evaluating',str(p.parent.relative_to(ROOT)),len(j['episodes']))
    for p in sorted((ROOT/'runs').glob('*/train.jsonl')):
        done=p.parent/'complete.json'
        lines=p.read_text().splitlines()
        if lines:
            r=json.loads(lines[-1]);print(p.parent.name,'complete' if done.exists() else 'training',r['step'],r['loss'],r.get('level_sample_counts'))
    counts={name:sum(len(json.loads(p.read_text())['episodes']) for p in (ROOT/name).glob('**/result.json')) for name in ['phase_a','runs']}
    print('completed_episodes',counts)


if __name__=='__main__':show()
