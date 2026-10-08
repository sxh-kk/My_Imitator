"""Pinned task-only downloads and training-only shared level normalizers."""
from common import *
import argparse
import copy
import shutil
import subprocess
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor


def freeze_inputs():
    path=ROOT/'checks/input-provenance.json'
    if path.exists(): return json.loads(path.read_text())
    selection=json.loads((OLD/'final-test-selection.json').read_text())
    selected=[x for x in selection['selections'] if x['run'].startswith('A50_')]
    for x in selected:
        assert sha256(x['checkpoint'])==x['checkpoint_sha256']
    tracked=subprocess.check_output(['git','ls-files','*.py'],cwd=SOURCE,text=True).splitlines()
    core={p:sha256(SOURCE/p) for p in tracked}
    assets=json.loads((WORKSPACE/'experiments/act_placemugrack_smoke/assets-verified.json').read_text())
    required=[]
    for obj,model in [('039_mug',11),('039_mug',4),('040_rack',0)]:
        for suffix in [f'model_data{model}.json',f'visual/base{model}.glb',f'collision/base{model}.glb']:
            p=Path('/var/tmp/imitator-game-zxc/maniskill/data/robotwin/objects')/obj/suffix
            assert p.is_file(),p
            old=next(x for x in assets['files'] if x['path']==str(p))
            assert sha256(p)==old['sha256']
            required.append({'path':str(p),'sha256':old['sha256'],'bytes':p.stat().st_size})
    r={'status':'passed','frozen_at':time.time(),'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=SOURCE,text=True).strip(),
       'upstream_python':core,'git_status_before':subprocess.check_output(['git','status','--short'],cwd=SOURCE,text=True),
       'old_scripts':{p.name:sha256(p) for p in OLD.glob('*.py')},'a50_selections':selected,'assets':required,
       'human':{str(p.relative_to(OLD/'prepared')):sha256(p) for p in [OLD/'prepared/human_ep0.pt',OLD/'prepared/human_ep1.pt',OLD/'prepared/human_cache_ep0/backbone_dinov2_vitl14/raw_features.pt',OLD/'prepared/human_cache_ep1/backbone_dinov2_vitl14/raw_features.pt']},
       'a50_stats_sha256':sha256(OLD/'prepared/A50/sim'/TASK/'meta/stats.json')}
    write_json(path,r)
    return r


def human_views():
    dest=ROOT/'prepared'; dest.mkdir(exist_ok=True)
    for name in ['human_ep0.pt','human_ep1.pt','human_cache_ep0','human_cache_ep1']:
        p=dest/name
        if not p.exists(): p.symlink_to(OLD/'prepared'/name,target_is_directory='cache' in name)
    for ep in [0,1]:
        write_json(dest/f'human_ep{ep}.json',[{'repo_id':HUMAN,'root':HUMAN,'train':f'{ep}:{ep+1}','test':''}])
    shutil.copyfile(dest/'human_ep0.json',dest/'human_train.json')


def alias_view(condition,stats,training_levels):
    dest=ROOT/'prepared'/condition
    for level in LEVELS:
        raw=DATA/'imitator_sim_v1_zed2i'/task(level)
        metadata=raw/'meta' if level in training_levels else DATA/'imitator_sim_v1_zed2i'/TASK/'meta'
        view=dest/'sim'/task(level)
        view.mkdir(parents=True,exist_ok=True)
        shutil.copytree(metadata,view/'meta',dirs_exist_ok=True)
        if isinstance(stats,Path): shutil.copyfile(stats,view/'meta/stats.json')
        else: write_json(view/'meta/stats.json',stats)
        if level in training_levels:
            for name in ['data','videos']:
                p=view/name
                if not p.exists(): p.symlink_to(raw/name,target_is_directory=True)
    cfg=lambda ls:[{'repo_id':task(l),'root':task(l),'train':'0:50','test':''} for l in ls]
    write_json(dest/'sim_train.json',cfg(training_levels))
    write_json(dest/'sim_eval.json',cfg(LEVELS))
    hashes=[sha256(stats_path(condition,l)) for l in LEVELS]
    assert len(set(hashes))==1,hashes


def phase_a():
    freeze_inputs(); human_views()
    alias_view('A50',OLD/'prepared/A50/sim'/TASK/'meta/stats.json',[])
    assert sha256(stats_path('A50'))==json.loads((ROOT/'checks/input-provenance.json').read_text())['a50_stats_sha256']
    print('PHASE A INPUTS VERIFIED',flush=True)


def download():
    repo='imitator-game/IG-10K-Dataset'; files=[]
    for level in ['L1','L2']:
        prefix=f'imitator_sim_v1_zed2i/{task(level)}'
        url=f'https://huggingface.co/api/datasets/{repo}/tree/{REVISION}/{prefix}?recursive=true&limit=1000'
        with urllib.request.urlopen(url,timeout=60) as r:
            entries=json.load(r)
            assert not r.headers.get('Link'),'Pagination requires handling'
        files.extend(x for x in entries if x['type']=='file')
    write_json(ROOT/'download-manifest.json',{'repository':repo,'revision':REVISION,'files':files,'bytes':sum(x['size'] for x in files)})
    def fetch(item):
        p=DATA/item['path']; p.parent.mkdir(parents=True,exist_ok=True)
        for attempt in range(3):
            try:
                if not p.exists() or p.stat().st_size!=item['size']:
                    url=f"https://huggingface.co/datasets/{repo}/resolve/{REVISION}/{item['path']}"
                    with urllib.request.urlopen(url,timeout=120) as r,p.with_suffix(p.suffix+'.part').open('wb') as f:
                        shutil.copyfileobj(r,f,1024*1024)
                    p.with_suffix(p.suffix+'.part').replace(p)
                assert p.stat().st_size==item['size']
                digest=sha256(p)
                if item.get('lfs'): assert digest==item['lfs']['oid']
                else:
                    raw=p.read_bytes()
                    assert hashlib.sha1(f'blob {len(raw)}\0'.encode()+raw).hexdigest()==item['oid']
                print('VERIFIED',item['path'],item['size'],flush=True)
                return {'path':str(p),'bytes':p.stat().st_size,'sha256':digest,'upstream_checksum_verified':True}
            except Exception:
                if attempt==2: raise
                time.sleep(2)
    with ThreadPoolExecutor(max_workers=4) as pool: verified=list(pool.map(fetch,files))
    write_json(ROOT/'checks/download-verified.json',{'status':'passed','files':verified})


def training():
    import numpy as np
    import pyarrow.parquet as pq
    import torch
    from examples.baselines.lerobot_dataset.normalizer import ActionNormalizer
    human_views()
    keys=['observation.qpos_gripper_states','action.qpos_gripper_actions']
    raw={}; frame_counts={}; ranges={}
    for level in LEVELS:
        base=DATA/'imitator_sim_v1_zed2i'/task(level)
        table=pq.read_table(base/'data/chunk-000/file-000.parquet').to_pydict()
        ep=np.array(table['episode_index']); assert set(ep)==set(range(50))
        raw[level]={k:np.asarray(table[k],dtype=np.float32) for k in keys}
        assert raw[level][keys[0]].shape==(len(ep),18) and raw[level][keys[1]].shape==(len(ep),16)
        assert all(np.isfinite(x).all() for x in raw[level].values())
        assert len(ep)==json.loads((base/'meta/info.json').read_text())['total_frames']
        frame_counts[level]=len(ep)
        ranges[level]={k:{'min':raw[level][k].min(0).tolist(),'max':raw[level][k].max(0).tolist()} for k in keys}
    source_stats=json.loads((OLD/'prepared/A50/sim'/TASK/'meta/stats.json').read_text())
    manifest={'status':'passed','frames':frame_counts,'raw_ranges':ranges,'conditions':{},'revision':REVISION}
    for condition,levels in CONDITIONS.items():
        stats=copy.deepcopy(source_stats)
        for k in keys:
            x=np.concatenate([raw[l][k] for l in levels]); lo,hi=np.quantile(x,[.01,.99],axis=0,method='linear')
            values={'q01':lo.tolist(),'q99':hi.tolist(),'min':x.min(0).tolist(),'max':x.max(0).tolist(),'mean':x.mean(0).tolist(),'std':x.std(0).tolist(),'count':[len(x)]}
            if condition=='B0':
                for name,v in values.items(): assert np.array_equal(np.asarray(stats[k][name]),np.asarray(v)),(k,name)
            else: stats[k].update(values)
        alias_view(condition,OLD/'prepared/A50/sim'/TASK/'meta/stats.json' if condition=='B0' else stats,levels)
        normal=ActionNormalizer(); normal.add_dataset_stats(0,TASK,stats,keys[0],keys[1])
        checks={}
        for k,fn,back in [(keys[0],normal.normalize_state,normal.denormalize_state),(keys[1],normal.normalize_action,normal.denormalize_action)]:
            x=torch.tensor(np.concatenate([raw[l][k] for l in levels])); n=fn(x,0); y=back(n,0)
            lo=torch.tensor(stats[k]['q01'],dtype=x.dtype); hi=torch.tensor(stats[k]['q99'],dtype=x.dtype)
            clipped=torch.maximum(torch.minimum(x,hi),lo)
            err=float((y-clipped).abs().max()); assert err<1e-5 and torch.isfinite(n).all()
            checks[k]={'zero_width_dims':torch.where(lo==hi)[0].tolist(),'clipped_fraction_per_dim':((x<lo)|(x>hi)).float().mean(0).tolist(),'official_roundtrip_max_error':err,'q01':lo.tolist(),'q99':hi.tolist()}
        manifest['conditions'][condition]={'levels':levels,'episodes_per_level':list(range(50)),'frames':sum(frame_counts[l] for l in levels),'stats_sha256':sha256(stats_path(condition)),'checks':checks}
    write_json(ROOT/'prepared/manifest.json',manifest)
    print('TRAINING VIEWS VERIFIED',frame_counts,flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('mode',choices=['phase-a','download','training'])
    {'phase-a':phase_a,'download':download,'training':training}[ap.parse_args().mode]()
