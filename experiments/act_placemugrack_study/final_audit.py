"""Verify completed artifacts and expose checkpoint roles without copying weights."""
from common import *
import math
import subprocess


def main():
    state = json.loads((ROOT / 'status.json').read_text())
    assert state['status'] == 'complete', state['status']
    plan = json.loads((ROOT / 'study-plan.json').read_text())
    config = json.loads((ROOT / 'execution-config.json').read_text())
    frozen = json.loads((ROOT / 'final-test-selection.json').read_text())
    selected = {r['run']: r for r in frozen['selections']}
    provenance = json.loads((ROOT / 'execution-source-hashes.json').read_text())
    for filename, digest in provenance.items():
        assert sha256(ROOT / filename) == digest, filename
    revision = subprocess.check_output(['git', '-C', str(SOURCE), 'rev-parse', 'HEAD'], text=True).strip()
    assert revision == plan['source_commit']
    changed_python = subprocess.check_output(
        ['git', '-C', str(SOURCE), 'diff', 'HEAD', '--name-only', '--', '*.py'], text=True
    ).splitlines()
    assert not changed_python, changed_python
    counts = {'training_runs': 0, 'optimizer_updates': 0, 'sample_presentations': 0,
              'checkpoints': 0, 'development_episodes': 0, 'test_episodes': 0,
              'environment_steps': 0, 'policy_queries_individual_envs': 0, 'videos': 0}
    rows = []
    for name in plan['run_queue']:
        directory = ROOT / 'runs' / name
        complete = json.loads((directory / 'complete.json').read_text())
        run_config = json.loads((directory / 'config.json').read_text())
        assert complete['status'] == 'complete'
        assert complete['steps'] == config['optimizer_steps']
        assert complete['samples'] == config['optimizer_steps'] * config['batch_size']
        assert complete['batch_size'] == config['batch_size']
        assert complete['workers'] == config['workers']
        assert frozen['frozen_at'] >= (directory / 'complete.json').stat().st_mtime
        trace = [json.loads(line) for line in (directory / 'train.jsonl').read_text().splitlines()]
        assert trace[-1]['step'] == complete['steps']
        assert trace[-1]['samples'] == complete['samples']
        assert all(math.isfinite(r[key]) for r in trace for key in ['loss', 'l1', 'kl'])
        assert all(value == 0 for value in trace[-1]['lr'])
        counts['training_runs'] += 1
        counts['optimizer_updates'] += complete['steps']
        counts['sample_presentations'] += complete['samples']
        for iteration in config['checkpoint_steps']:
            checkpoint = directory / 'checkpoints' / f'step_{iteration:06d}.pt'
            ready = json.loads(checkpoint.with_suffix('.ready.json').read_text())
            assert ready['roundtrip_equal'] is True
            assert ready['iteration'] == iteration and ready['optimizer_steps'] == [iteration]
            assert ready['bytes'] == checkpoint.stat().st_size
            assert ready['sha256'] == sha256(checkpoint), str(checkpoint)
            counts['checkpoints'] += 1
        evaluations = [(directory / 'dev' / f'step_{s:06d}', 20, 1000, 'development_episodes')
                       for s in config['checkpoint_steps']]
        evaluations.append((directory / 'test', 100, 2000, 'test_episodes'))
        for output, episodes, first_seed, count_key in evaluations:
            result = json.loads((output / 'result.json').read_text())
            assert result['status'] == 'passed'
            assert result['condition'] == complete['condition']
            assert result['num_envs'] == config['eval_envs']
            assert result['stats_sha256'] == run_config['stats_sha256']
            assert result['human_episode'] == 1 and result['auxiliary_dtw'] is False
            assert result['all_loaded_tensors_equal'] > 0
            assert result['optimizer_steps'] == [result['checkpoint_step']]
            assert result['vector_steps'] * result['num_envs'] == episodes * 500
            assert result['policy_queries'] * result['num_envs'] == episodes * 125
            assert sorted(e['seed'] for e in result['episodes']) == list(range(first_seed, first_seed + episodes))
            ready = json.loads(Path(result['checkpoint']).with_suffix('.ready.json').read_text())
            assert ready['sha256'] == result['checkpoint_sha256']
            for metric in ['success_once', 'success_at_end']:
                actual = sum(bool(e['metrics'][metric]) for e in result['episodes']) / episodes
                assert abs(actual - result['summary'][metric]['mean']) < 1e-7
            if count_key == 'test_episodes':
                assert result['checkpoint_sha256'] == selected[name]['checkpoint_sha256']
                assert result['checkpoint_step'] == selected[name]['step']
                assert (output / 'result.json').stat().st_mtime >= frozen['frozen_at']
            assert result['preselected_video_seeds'] == [first_seed, first_seed + config['eval_envs']]
            videos = sorted((output / 'videos').glob('*.mp4'))
            assert len(videos) == config['video_batches'], videos
            for video in videos:
                metadata = json.loads(subprocess.check_output(
                    ['ffprobe', '-v', 'error', '-select_streams', 'v:0',
                     '-show_entries', 'stream=nb_frames', '-of', 'json', str(video)], text=True))
                assert int(metadata['streams'][0]['nb_frames']) == 501, str(video)
            counts['videos'] += len(videos)
            counts[count_key] += episodes
            counts['environment_steps'] += episodes * 500
            counts['policy_queries_individual_envs'] += episodes * 125
        roles = {'best_model.pt': f"step_{selected[name]['step']:06d}.pt",
                 'last.pt': 'step_018000.pt', 'final_model.pt': 'step_018000.pt'}
        for filename, target in roles.items():
            link = directory / 'checkpoints' / filename
            if link.exists() or link.is_symlink():
                assert link.is_symlink() and os.readlink(link) == target, str(link)
            else:
                link.symlink_to(target)
            assert link.resolve().is_file()
        rows.append({'run': name, 'steps': complete['steps'], 'samples': complete['samples'],
                     'selected_step': selected[name]['step'], 'checkpoint_roles': roles})
    assert counts['training_runs'] == 6
    assert counts['optimizer_updates'] == 108000
    assert counts['development_episodes'] == counts['test_episodes'] == 600
    assert counts['environment_steps'] == 600000
    assert counts['checkpoints'] == 30 and counts['videos'] == 72
    failure = json.loads((ROOT / 'failure_analysis.json').read_text())
    assert failure['all_test_initial_conditions_identical_across_runs'] is True
    report = {'status': 'passed', 'counts': counts, 'source_commit': revision,
              'original_execution_scripts_unchanged': True, 'upstream_tracked_python_unchanged': True,
              'checkpoint_sha256_all_verified': True, 'videos_all_501_frames': True,
              'test_selections_frozen_before_tests': True, 'runs': rows}
    write_json(ROOT / 'checks' / 'final-audit.json', report)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
