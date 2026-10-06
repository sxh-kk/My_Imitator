"""Download pinned official RGB chunks and selectively extract official task assets."""
import argparse
import hashlib
import json
import shutil
import tarfile
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = Path('/var/tmp/imitator-game-zxc/data')
ASSETS = Path('/var/tmp/imitator-game-zxc/maniskill/data')
REV = '2d7a339b27aff14a0780db4bd4d5e3a68c6358bc'


def dataset():
    manifest = json.loads((HERE / 'download-manifest.json').read_text())
    def fetch(item):
        target = DATA / item['path']
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists() or target.stat().st_size != item['size']:
            url = f"https://huggingface.co/datasets/{manifest['repository']}/resolve/{manifest['revision']}/{item['path']}"
            with urllib.request.urlopen(url, timeout=120) as src, target.with_suffix(target.suffix + '.part').open('wb') as dst:
                shutil.copyfileobj(src, dst, 1024 * 1024)
            target.with_suffix(target.suffix + '.part').replace(target)
        assert target.stat().st_size == item['size'], target
        raw = target.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        if item.get('lfs'):
            assert digest == item['lfs']['oid'], target
        else:
            assert hashlib.sha1(f'blob {len(raw)}\0'.encode() + raw).hexdigest() == item['oid'], target
        print('VERIFIED', target.stat().st_size, item['path'], flush=True)
        return dict(path=str(target), bytes=target.stat().st_size, sha256=digest)
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(fetch, manifest['files']))
    (HERE / 'download-verified.json').write_text(json.dumps(results, indent=2))


def assets():
    import zstandard
    manifest_path = HERE / 'assets-verified.json'
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text())
        if all(Path(x['path']).is_file() and hashlib.sha256(Path(x['path']).read_bytes()).hexdigest() == x['sha256'] for x in existing['files']):
            print('ASSETS: existing selected files verified', flush=True)
            return
    url = f'https://huggingface.co/datasets/imitator-game/IG-10K-Assets/resolve/{REV}/robotwin.tar.zst'
    selected = {'039_mug', '040_rack'}
    required = {
        f'objects/{obj}/{suffix}'
        for obj, model in [('039_mug', 11), ('040_rack', 0)]
        for suffix in [f'model_data{model}.json', f'visual/base{model}.glb', f'collision/base{model}.glb']
    }
    seen = set()
    extracted = []
    started = last_print = time.monotonic()
    class Counted:
        def __init__(self, source):
            self.source, self.n = source, 0
            self.sha = hashlib.sha256()
        def read(self, n=-1):
            nonlocal last_print
            block = self.source.read(n)
            self.n += len(block)
            self.sha.update(block)
            now = time.monotonic()
            if now - last_print > 25:
                print(f'ASSET STREAM {self.n / 1e9:.3f} GB, {now-started:.0f}s, extracted {len(extracted)} files', flush=True)
                last_print = now
            return block
    with urllib.request.urlopen(url, timeout=120) as response:
        counted = Counted(response)
        with zstandard.ZstdDecompressor().stream_reader(counted) as stream:
            with tarfile.open(fileobj=stream, mode='r|') as archive:
                for member in archive:
                    parts = Path(member.name).parts
                    if 'objects' not in parts:
                        continue
                    obj_index = parts.index('objects')
                    if len(parts) <= obj_index + 1 or parts[obj_index + 1] not in selected:
                        continue
                    if not member.isfile():
                        continue
                    relative = Path(*parts[obj_index:])
                    assert '..' not in relative.parts
                    target = ASSETS / 'robotwin' / relative
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with archive.extractfile(member) as src, target.open('wb') as dst:
                        shutil.copyfileobj(src, dst, 1024 * 1024)
                    extracted.append(dict(path=str(target), bytes=target.stat().st_size, sha256=hashlib.sha256(target.read_bytes()).hexdigest()))
                    seen.add(str(relative))
                    print('EXTRACT', member.name, member.size, flush=True)
                    if required <= seen:
                        break
    result = dict(repository='imitator-game/IG-10K-Assets', revision=REV, archive='robotwin.tar.zst', full_archive_downloaded=False, full_archive_hash_verified=False, transferred_bytes=counted.n, elapsed_seconds=time.monotonic()-started, files=extracted)
    assert required <= seen, 'Some required task asset files missing'
    manifest_path.write_text(json.dumps(result, indent=2))
    print('ASSETS DONE', len(extracted), sum(x['bytes'] for x in extracted), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['dataset', 'assets'])
    {'dataset': dataset, 'assets': assets}[parser.parse_args().mode]()
