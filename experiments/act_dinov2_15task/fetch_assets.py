"""Pin asset archives, verify their upstream SHA256, extract required objects."""
from common import *
import ast
import shutil
import tarfile
import urllib.request
import subprocess
from concurrent.futures import ThreadPoolExecutor

ARCHIVES = Path('/var/tmp/imitator-game-zxc/asset-archives')


def selections():
    ids = {x['repo_id'].split('_',1)[1] for name in ['sim_train_config_15.json','sim_test_config_unseen.json'] for x in json.loads((CONFIG/name).read_text())}
    objects, partnet = set(), set()
    files = []
    for path in (SOURCE/'mani_skill/envs/tasks/tabletop/dual_tasks').glob('*.py'):
        text = path.read_text()
        if not any(f'"{env}"' in text or repr(env) in text for env in ids):
            continue
        files.append(str(path.relative_to(SOURCE)))
        for node in ast.walk(ast.parse(text)):
            if not isinstance(node,ast.Constant) or not isinstance(node.value,str):
                continue
            val = node.value
            if len(val)<65 and len(val)>4 and val[:3].isdigit() and val[3]=='_':
                objects.add(val)
            elif val.isdigit() and len(val)>=5:
                partnet.add(val)
    assert len(files) == 20
    return {'objects':sorted(objects), 'partnet_ids':sorted(partnet), 'task_files':files}


def download(item):
    ARCHIVES.mkdir(parents=True,exist_ok=True)
    path = ARCHIVES/item['path']
    size = item['size']
    expected = item.get('lfs',{}).get('oid')
    assert expected
    if path.exists() and path.stat().st_size==size and sha256(path)==expected:
        return path
    part = path.with_suffix(path.suffix+'.part')
    for attempt in range(6):
        try:
            offset = part.stat().st_size if part.exists() else 0
            url=f"https://huggingface.co/datasets/imitator-game/IG-10K-Assets/resolve/{ASSET_REVISION}/{item['path']}?download=true&attempt={attempt}"
            headers={'User-Agent':'Imitator-reproduction/1.0'}
            if offset: headers['Range']=f'bytes={offset}-'
            with urllib.request.urlopen(urllib.request.Request(url,headers=headers),timeout=90) as src:
                resumed=offset>0 and src.status==206
                with part.open('ab' if resumed else 'wb') as dst:
                    last=time.monotonic()
                    while True:
                        block=src.read(4*1024*1024)
                        if not block:break
                        dst.write(block)
                        if time.monotonic()-last>20:
                            print('ASSET DOWNLOAD',item['path'],dst.tell(),size,flush=True)
                            last=time.monotonic()
            assert part.stat().st_size==size,(item['path'],part.stat().st_size,size)
            assert sha256(part)==expected,f'Checksum mismatch: {part}'
            part.replace(path)
            return path
        except Exception as exc:
            print('ASSET RETRY',item['path'],attempt,str(exc)[:200],flush=True)
            if attempt==5:raise
            time.sleep(min(2**attempt,10))


def extract(path,selected):
    extracted=[]
    archive_kind=path.name.split('.tar')[0]
    decoder=subprocess.Popen(['/home/zxc/miniconda3/bin/zstd','-dc',str(path)],stdout=subprocess.PIPE)
    with tarfile.open(fileobj=decoder.stdout,mode='r|') as archive:
        for member in archive:
            if not member.isfile():continue
            parts=Path(member.name).parts
            assert '..' not in parts
            relative=None
            if archive_kind=='robotwin' and 'objects' in parts:
                idx=parts.index('objects')
                if len(parts)>idx+1 and parts[idx+1] in selected['objects']:
                    relative=Path('robotwin',*parts[idx:])
            elif archive_kind=='partnet_mobility' and 'dataset' in parts:
                idx=parts.index('dataset')
                if len(parts)>idx+2 and parts[idx+2] in selected['partnet_ids']:
                    relative=Path('partnet_mobility',*parts[idx:])
            elif archive_kind=='sketchfab':
                idx=parts.index('sketchfab') if 'sketchfab' in parts else None
                relative=Path(*parts[idx:]) if idx is not None else Path('sketchfab',*parts)
            if relative is None:continue
            target=ASSETS/relative
            target.parent.mkdir(parents=True,exist_ok=True)
            assert shutil.disk_usage(target.parent).free>member.size+10*1024**3
            content=archive.extractfile(member)
            if target.exists():
                # Existing assets from finished experiments must never change.
                h=hashlib.sha256()
                for chunk in iter(lambda:content.read(8*1024*1024),b''):h.update(chunk)
                assert target.stat().st_size==member.size and sha256(target)==h.hexdigest(),target
            else:
                tmp=target.with_suffix(target.suffix+'.part')
                with tmp.open('wb') as dst:shutil.copyfileobj(content,dst,1024*1024)
                tmp.replace(target)
            extracted.append({'path':str(target),'bytes':member.size,'sha256':sha256(target)})
    decoder.stdout.read()  # Drain tar padding so zstd exits normally.
    decoder.stdout.close()
    assert decoder.wait()==0
    return extracted


def main():
    selected=selections()
    write_json(ROOT/'asset-selection.json',selected)
    url=f'https://huggingface.co/api/datasets/imitator-game/IG-10K-Assets/tree/{ASSET_REVISION}?recursive=false&limit=1000'
    with urllib.request.urlopen(url,timeout=30) as response:items=json.load(response)
    items=[x for x in items if x['path'] in ['robotwin.tar.zst','partnet_mobility.tar.zst','sketchfab.tar.zst']]
    assert len(items)==3
    assert shutil.disk_usage(ASSETS).free>sum(x['size'] for x in items)+35*1024**3
    write_json(ROOT/'asset-manifest.json',{'revision':ASSET_REVISION,'files':items})
    with ThreadPoolExecutor(max_workers=3) as pool:paths=list(pool.map(download,items))
    files=[]
    for path in paths:
        print('EXTRACTING',path.name,flush=True)
        files.extend(extract(path,selected))
    assert files
    write_json(ROOT/'checks/assets-download.json',{'status':'passed','revision':ASSET_REVISION,'files':files,'archives_checksum_verified':True})
    print('ASSETS VERIFIED',len(files),sum(x['bytes'] for x in files),flush=True)


if __name__=='__main__':main()
