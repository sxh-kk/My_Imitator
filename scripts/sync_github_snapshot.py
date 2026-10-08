"""Refresh the inspected publishing checkout with source and small evidence.

Does not commit or push. Git history and unrelated files are preserved. Large
datasets, model weights, feature caches and interrupted runs stay local.
"""
import argparse
import datetime
import hashlib
import json
import os
import re
import subprocess
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[1]
SOURCE = WORKSPACE / 'The-Imitator-Game'
TARGET = WORKSPACE / '.github-publish/My_Imitator'
SMALL_SUFFIXES = {
    '.py', '.md', '.json', '.jsonl', '.csv', '.yaml', '.yml', '.toml',
    '.sh', '.txt', '.lock', '.png', '.pdf', '.svg', '.service', '.patch', '.log',
}
SKIP_PARTS = {
    '__pycache__', '.git', 'te_cache', '.cache',
    'interrupted_runs', '.local-interruptions', 'hf-cache', 'uv-cache',
}
SECRET_PATTERNS = [
    re.compile(rb'github_pat_[A-Za-z0-9_]{30,}'),
    re.compile(rb'gh[pousr]_[A-Za-z0-9]{30,}'),
    re.compile(rb'hf_[A-Za-z0-9]{25,}'),
    re.compile(rb'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----'),
]
IGNORE_TEXT = '''# Local datasets, trained weights and caches
.github-publish/
__pycache__/
*.py[cod]
.env
.env.*
!.env.example
*.pt
*.pth
*.ckpt
*.safetensors
*.parquet
*.npy
*.npz
experiments/**/videos/
experiments/**/data/
**/te_cache/
**/human_cache_*/
**/checkpoints/*
!**/checkpoints/*.ready.json
**/interrupted_runs/
**/.cache/
/data/
/demos/
/hf-cache/
/uv-cache/
.DS_Store
'''


def git(*args):
    return subprocess.check_output(['git', *args], cwd=TARGET, text=True).strip()


def candidates():
    tracked = subprocess.check_output(['git', 'ls-files', '-z'], cwd=SOURCE).decode().split('\0')
    files = {SOURCE / p for p in tracked if p}
    files.add(SOURCE / 'uv.lock')
    for directory in ['LearningDocs', 'environment', 'scripts', 'experiments']:
        for p in (WORKSPACE / directory).rglob('*'):
            parts = p.relative_to(WORKSPACE).parts
            if any(x in SKIP_PARTS or x.startswith('human_cache_') for x in parts):
                continue
            if 'checkpoints' in parts and not p.name.endswith('.ready.json'):
                continue
            if p.is_symlink() or not p.is_file() or p.suffix.lower() not in SMALL_SUFFIXES:
                continue
            # Telemetry is summarized separately after the final audit.
            if p.name == 'gpu-during-study.csv':
                continue
            files.add(p)
    for name in ['README.md', 'IMITATOR_RESEARCH_PLAN.md', 'download-manifest-15task-rgb.json']:
        files.add(WORKSPACE / name)
    return sorted(p for p in files if p.exists() and not p.is_symlink())


def content(source):
    data = source.read_bytes()
    if any(pattern.search(data) for pattern in SECRET_PATTERNS):
        raise RuntimeError(f'Credential-like content detected: {source.relative_to(WORKSPACE)}')
    if len(data) >= 95 * 1024 ** 2:
        raise RuntimeError(f'Oversized publishing file: {source.relative_to(WORKSPACE)}')
    if source.suffix == '.json':
        json.loads(data)
    elif source.suffix == '.jsonl':
        # A live writer may have an incomplete trailing line; copy whole records.
        lines = data.splitlines(keepends=True)
        if lines and not lines[-1].endswith(b'\n'):
            try:
                json.loads(lines[-1])
            except json.JSONDecodeError:
                lines.pop()
        data = b''.join(lines)
        for line in data.splitlines():
            if line.strip():
                json.loads(line)
    elif source.suffix == '.md':
        target = TARGET / source.relative_to(WORKSPACE)
        def local_link(match):
            local = match.group(1)
            line = re.search(r':(\d+)$', local)
            if line:
                local = local[:line.start()]
            destination = TARGET / Path(local).relative_to(WORKSPACE)
            suffix = '#L' + line.group(1) if line else ''
            return '](' + os.path.relpath(destination, target.parent) + suffix + ')'
        text = re.sub(r'\]\((/home/zxc/Imitator/[^)]+)\)', local_link, data.decode())
        def local_artifact(match):
            label, link = match.group(1), match.group(2)
            if '://' in link or link.startswith('#'):
                return match.group(0)
            artifact = link.split('#', 1)[0]
            destination = (target.parent / artifact).resolve()
            if TARGET not in destination.parents:
                return match.group(0)
            relative = destination.relative_to(TARGET)
            if (relative.parts[0] in {'experiments', 'environment'}
                    and destination.suffix.lower() in {'.pt', '.pth', '.ckpt', '.safetensors', '.mp4', '.parquet', '.npz'}):
                return f'{label}（本机文件：`{relative}`）'
            return match.group(0)
        text = re.sub(r'\[([^\[\]]+)\]\(([^)]+)\)', local_artifact, text)
        data = text.encode()
    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    assert (TARGET / '.git').is_dir(), 'Inspect and initialize the publishing checkout first.'
    assert git('remote', 'get-url', 'origin') == 'https://github.com/sxh-kk/My_Imitator.git'
    assert git('branch', '--show-current') == 'main'
    payload = {str(p.relative_to(WORKSPACE)): content(p) for p in candidates()}
    payload['LICENSE'] = (SOURCE / 'LICENSE').read_bytes()
    payload['.gitignore'] = IGNORE_TEXT.encode()
    from check_levels_status import inspect
    runtime = inspect()
    capture = {
        'captured_at': datetime.datetime.now().astimezone().isoformat(),
        'snapshot_is_live': False,
        'experiment': 'PlaceMugRack L0/L1/L2 coverage',
        'runtime_at_capture': runtime,
        'final_results_available': runtime['runtime_status'] == 'complete',
        'recovery_history': 'experiments/act_placemugrack_levels/recovery-history.json',
    }
    payload['PUBLICATION_STATUS.json'] = (json.dumps(capture, indent=2, ensure_ascii=False) + '\n').encode()
    manifest = {
        'target_repository': 'sxh-kk/My_Imitator', 'visibility': 'public',
        'upstream_repository': 'https://github.com/imitator-game/The-Imitator-Game',
        'upstream_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=SOURCE, text=True).strip(),
        'snapshot_layout': 'The-Imitator-Game is a regular directory, not a submodule.',
        'captured_at': capture['captured_at'],
        'file_count': len(payload), 'copied_bytes': sum(len(b) for b in payload.values()),
        'exclusions': ['datasets', 'downloaded task assets', 'trained checkpoints',
                       'feature caches', 'environment caches', 'videos', 'interrupted partial runs'],
        'publication_only_changes': ['Markdown navigation converted to GitHub relative links',
                                     'local-only artifact links rendered as file descriptions',
                                     'live JSONL restricted to complete records'],
        'files': {p: {'bytes': len(b), 'sha256': hashlib.sha256(b).hexdigest()}
                  for p, b in sorted(payload.items())},
    }
    old_manifest = json.loads((TARGET / 'SOURCE_SNAPSHOT.json').read_text())
    previous = set(old_manifest.get('files', {}))
    if not previous:
        # The first publication used a manifest without its owned file list.
        # Only its original commit's paths are managed; later unrelated files
        # are preserved when upgrading that manifest format.
        initial_commit = git('rev-list', '--max-parents=0', 'HEAD').splitlines()[0]
        previous = set(git('ls-tree', '-r', '--name-only', initial_commit).splitlines())
        previous.discard('SOURCE_SNAPSHOT.json')
    stale = previous - set(payload)
    print(json.dumps({'apply': args.apply, 'file_count': manifest['file_count'],
                      'copied_MiB': round(manifest['copied_bytes'] / 1024 ** 2, 2),
                      'removed_previously_managed': sorted(stale),
                      'runtime_status': runtime['runtime_status']}, indent=2))
    if not args.apply:
        return
    for relative in stale:
        path = TARGET / relative
        assert TARGET in path.resolve().parents
        if path.is_file():
            path.unlink()
    for relative, data in payload.items():
        path = TARGET / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    (TARGET / 'SOURCE_SNAPSHOT.json').write_text(json.dumps(manifest, indent=2) + '\n')


if __name__ == '__main__':
    main()
