"""Index stable successes and failures; supplement missing recorded examples."""
from common import *
import subprocess


def main():
    assert json.loads((ROOT / 'status.json').read_text())['status'] == 'complete'
    plan = json.loads((ROOT / 'study-plan.json').read_text())
    manifest = {'selection': 'recorded test examples first; otherwise earliest eligible seed',
                'supplementary_replays_excluded_from_primary_results': True, 'runs': {}}
    for name in plan['run_queue']:
        source = ROOT / 'runs' / name / 'test' / 'result.json'
        result = json.loads(source.read_text())
        episodes = sorted(result['episodes'], key=lambda e: e['seed'])
        recorded = {seed: source.parent / 'videos' / f'{i}.mp4'
                    for i, seed in enumerate(result['preselected_video_seeds'])}
        groups = {
            'stable_success': [e for e in episodes if e['metrics']['success_at_end']],
            'failure': [e for e in episodes if not e['metrics']['success_once']],
            'transient_success': [e for e in episodes if e['metrics']['success_once'] and not e['metrics']['success_at_end']],
        }
        rows = {}
        # Supply one success and one failure. If all episodes ever succeeded,
        # use a transient success to illustrate loss of terminal stability.
        for kind in ['stable_success', 'failure']:
            actual_kind = kind
            candidates = groups[kind]
            if kind == 'failure' and not candidates:
                candidates = groups['transient_success']; actual_kind = 'transient_success'
            if not candidates:
                rows[kind] = {'status': 'no_episode_in_this_category'}
                continue
            chosen = next((e for e in candidates if e['seed'] in recorded), candidates[0])
            seed = chosen['seed']
            if seed in recorded:
                video = recorded[seed]
                supplementary = False
                metadata = None
            else:
                output = ROOT / 'supplementary_videos' / name / f'{actual_kind}_seed_{seed}'
                if not (output / 'result.json').exists():
                    output.mkdir(parents=True, exist_ok=True)
                    with open(output / 'replay.log', 'w') as log:
                        subprocess.run([sys.executable, str(ROOT / 'record_example.py'),
                                        '--source-result', str(source), '--seed', str(seed),
                                        '--out', str(output)], stdout=log, stderr=subprocess.STDOUT, check=True)
                replay = json.loads((output / 'result.json').read_text())
                metadata = replay['supplementary_recording']
                video = output / 'videos' / '0.mp4'
                supplementary = True
            assert video.is_file(), str(video)
            rows[kind] = {'status': 'available', 'category': actual_kind, 'seed': seed,
                          'original_metrics': {k: chosen['metrics'][k] for k in ['success_once', 'success_at_end']},
                          'video': str(video), 'supplementary': supplementary, 'replay_checks': metadata}
        manifest['runs'][name] = rows
        write_json(ROOT / 'video-examples.json', manifest)
        print('VIDEO EXAMPLES', name, json.dumps(rows), flush=True)
    print('ALL VIDEO EXAMPLES INDEXED', flush=True)


if __name__ == '__main__':
    main()
