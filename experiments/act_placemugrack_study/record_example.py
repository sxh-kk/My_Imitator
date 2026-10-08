"""Replay an existing four-environment batch, recording the chosen environment."""
from common import *
import argparse
import inspect
import evaluate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-result', required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--out', required=True)
    cli = parser.parse_args()
    source_path = Path(cli.source_result)
    original = json.loads(source_path.read_text())
    assert original['status'] == 'passed' and original['num_envs'] == 4
    first_seed = min(e['seed'] for e in original['episodes'])
    batch_start = first_seed + (cli.seed - first_seed) // 4 * 4
    environment_index = cli.seed - batch_start
    expected = next(e for e in original['episodes'] if e['seed'] == cli.seed)
    # Preserve reset seeds, environment positions and batch inference. The only
    # factory change chooses which CPU environment has the video wrapper.
    factory = evaluate.official.make_eval_envs
    source = inspect.getsource(factory)
    needle = 'video_dir if seed == 0 else None'
    assert source.count(needle) == 1
    modified = source.replace(needle, f'video_dir if seed == {environment_index} else None')
    namespace = dict(factory.__globals__)
    exec(compile(modified, 'supplementary_video_factory', 'exec'), namespace)
    evaluate.official.make_eval_envs = namespace['make_eval_envs']
    sys.argv = ['evaluate', '--checkpoint', original['checkpoint'],
                '--condition', original['condition'], '--out', cli.out,
                '--seed-start', str(batch_start), '--episodes', '4',
                '--num-envs', '4', '--video-batches', '1']
    evaluate.main()
    out = Path(cli.out)
    result = json.loads((out / 'result.json').read_text())
    replay = next(e for e in result['episodes'] if e['seed'] == cli.seed)
    for key in ['initial_scene', 'initial_state', 'initial_rgb_sha256']:
        assert replay[key] == expected[key], key
    matching_metrics = all(bool(replay['metrics'][k]) == bool(expected['metrics'][k])
                           for k in ['success_once', 'success_at_end'])
    result['preselected_video_seeds'] = [cli.seed]
    result['supplementary_recording'] = {
        'source_result': str(source_path.resolve()), 'seed': cli.seed,
        'environment_index': environment_index, 'batch_start': batch_start,
        'selected_after_test_for_illustration_only': True,
        'excluded_from_primary_results': True,
        'factory_only_change': 'video recording environment index',
        'initial_conditions_match': True,
        'success_metrics_match_original': matching_metrics,
        'stage_first_steps_match_original': replay['stage_first_step'] == expected['stage_first_step'],
    }
    write_json(out / 'result.json', result)
    print('SUPPLEMENTARY VIDEO', json.dumps(result['supplementary_recording']))


if __name__ == '__main__':
    main()
