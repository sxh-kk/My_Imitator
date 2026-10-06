"""Prepare a GitHub snapshot while retaining the local upstream checkout.

Only tracked source files (plus uv.lock) and selected reproduction artifacts
are copied. Dataset, model and feature caches never enter the publishing tree.
"""
import hashlib
import json
import re
import shutil
import subprocess
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[1]
SOURCE = WORKSPACE / 'The-Imitator-Game'
TARGET = WORKSPACE / '.github-publish' / 'My_Imitator'
TARGET.mkdir(parents=True, exist_ok=True)
if (TARGET / '.git').exists():
    raise SystemExit(f'{TARGET} already contains a Git repository; inspect it before preparing a new snapshot.')

files = [SOURCE / name for name in subprocess.check_output(
    ['git', 'ls-files', '-z'], cwd=SOURCE).decode().split('\0') if name]
files.append(SOURCE / 'uv.lock')
for directory in ['LearningDocs', 'environment', 'scripts', 'experiments']:
    for path in (WORKSPACE / directory).rglob('*'):
        if not path.is_file():
            continue
        parts = path.relative_to(WORKSPACE).parts
        if any(part in {'__pycache__', '.git', 'te_cache', '.cache', 'checkpoints', 'runs'} for part in parts):
            continue
        if path.suffix.lower() in {'.pyc', '.pt', '.pth', '.ckpt', '.safetensors'}:
            continue
        files.append(path)
for name in ['README.md', 'IMITATOR_RESEARCH_PLAN.md', 'download-manifest-15task-rgb.json']:
    files.append(WORKSPACE / name)

copied = []
for source in sorted(set(files)):
    if not source.exists():
        continue
    relative = source.relative_to(WORKSPACE)
    target = TARGET / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target, follow_symlinks=False)
    # Convert machine-specific Markdown navigation to GitHub relative links.
    # Preserve shell commands and local-path descriptions as experiment evidence.
    if target.suffix == '.md':
        import os
        text = target.read_text()
        def local_link(match):
            path = match.group(1)
            line = re.search(r':(\d+)$', path)
            if line:
                path = path[:line.start()]
            destination = TARGET / Path(path).relative_to(WORKSPACE)
            link = os.path.relpath(destination, target.parent)
            if line:
                link += '#L' + line.group(1)
            return '](' + link + ')'
        text = re.sub(r'\]\((/home/zxc/Imitator/[^)]+)\)', local_link, text)
        target.write_text(text)
    copied.append(dict(path=str(relative), bytes=target.stat().st_size))

shutil.copy2(SOURCE / 'LICENSE', TARGET / 'LICENSE')
(TARGET / '.gitignore').write_text('''# Local-only caches, datasets and trained weights
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
**/te_cache/
**/checkpoints/
**/runs/
**/.cache/
/data/
/demos/
/hf-cache/
/uv-cache/
.DS_Store
''')
manifest = dict(
    target_repository='sxh-kk/My_Imitator', visibility='public',
    upstream_repository='https://github.com/imitator-game/The-Imitator-Game',
    upstream_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=SOURCE).decode().strip(),
    snapshot_layout='The-Imitator-Game is a regular directory, not a submodule.',
    file_count=len(copied), copied_bytes=sum(x['bytes'] for x in copied),
    exclusions=['datasets', 'downloaded task assets', 'trained checkpoints', 'feature caches',
                'environment caches', 'local storage-cleanup audits'],
    publication_only_changes=['Markdown source links converted from local absolute paths to GitHub relative links'],
)
(TARGET / 'SOURCE_SNAPSHOT.json').write_text(json.dumps(manifest, indent=2)+'\n')
print(json.dumps(dict(target=str(TARGET), **manifest), indent=2))
