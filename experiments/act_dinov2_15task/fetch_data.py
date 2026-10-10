"""Download the pinned official 15-task and ten-task evaluation RGB manifest."""
from common import *
import shutil
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed


def main():
    manifest = json.loads((WORKSPACE/'download-manifest-15task-rgb.json').read_text())
    assert manifest['revision'] == DATA_REVISION
    write_json(ROOT/'download-manifest.json', manifest)
    missing = [x for x in manifest['files'] if not (DATA/x['path']).is_file() or (DATA/x['path']).stat().st_size != x['size']]
    assert shutil.disk_usage(DATA).free > sum(x['size'] for x in missing) + 20*1024**3
    started = time.time()
    set_status('downloading_data', status='running', pid=os.getpid(), total_files=len(manifest['files']), completed_files=0, total_bytes=manifest['bytes'])

    def fetch(item):
        target = DATA/item['path']
        target.parent.mkdir(parents=True, exist_ok=True)
        for attempt in range(5):
            try:
                if not target.is_file() or target.stat().st_size != item['size']:
                    url = f"https://huggingface.co/datasets/{manifest['repository']}/resolve/{DATA_REVISION}/{item['path']}?download=true&attempt={attempt}"
                    request = urllib.request.Request(url, headers={'User-Agent': 'Imitator-reproduction/1.0'})
                    partial = target.with_suffix(target.suffix+'.part')
                    with urllib.request.urlopen(request, timeout=90) as src, partial.open('wb') as dst:
                        shutil.copyfileobj(src, dst, 1024*1024)
                    assert partial.stat().st_size == item['size'], (item['path'], partial.stat().st_size, item['size'])
                    partial.replace(target)
                digest = sha256(target)
                if item.get('lfs'):
                    assert digest == item['lfs']['oid'], f'Checksum mismatch: {target}'
                else:
                    raw = target.read_bytes()
                    assert hashlib.sha1(f'blob {len(raw)}\0'.encode()+raw).hexdigest() == item['oid'], f'Git blob checksum mismatch: {target}'
                return {'path': item['path'], 'bytes': item['size'], 'sha256': digest}
            except Exception as exc:
                print('RETRY', item['path'], attempt+1, type(exc).__name__, str(exc)[:200], flush=True)
                if attempt == 4:
                    raise
                time.sleep(min(2**attempt, 10))
    verified = []
    with ThreadPoolExecutor(max_workers=6) as pool:
        futures = [pool.submit(fetch, item) for item in manifest['files']]
        for future in as_completed(futures):
            result = future.result()
            verified.append(result)
            print('VERIFIED', len(verified), len(futures), result['path'], result['bytes'], flush=True)
            set_status('downloading_data', completed_files=len(verified), verified_bytes=sum(x['bytes'] for x in verified))
    verified.sort(key=lambda x: x['path'])
    write_json(ROOT/'checks/data-download.json', {'status':'passed', 'revision':DATA_REVISION, 'elapsed_seconds':time.time()-started, 'files':verified})
    set_status('data_download_complete', status='ready', completed_files=len(verified))


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        set_status('data_download_failed', status='needs_diagnosis', error=str(exc))
        raise
