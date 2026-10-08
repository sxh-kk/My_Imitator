"""Derive training-only normalizers and immutable human-video caches."""
from common import *
import copy
import time
import numpy as np
import pyarrow.parquet as pq
import torch


def main():
    start=time.time()
    target=ROOT/'prepared'
    if (target/'manifest.json').exists():
        raise RuntimeError('Preparation already complete; refusing to overwrite')
    target.mkdir(exist_ok=True)
    plan=json.loads((ROOT/'study-plan.json').read_text())
    source=DATA/'imitator_sim_v1_zed2i'/TASK
    rows=pq.read_table(source/'data/chunk-000/file-000.parquet')
    eps=np.array(rows['episode_index'])
    keys=['observation.qpos_gripper_states','action.qpos_gripper_actions']
    original_stats=json.loads((source/'meta/stats.json').read_text())
    manifest={'status':'passed','conditions':{},'human':{},'data_revision':plan['dataset_revision']}
    for condition, selected in [('A50',list(range(50))),('A10',plan['conditions']['A10']['episodes']),('D2',[0,1])]:
        view=target/condition/'sim'/TASK
        view.mkdir(parents=True,exist_ok=True)
        # Link the large original arrays; copy only metadata that will be derived.
        for name in ['data','videos']:
            (view/name).symlink_to(source/name,target_is_directory=True)
        import shutil
        shutil.copytree(source/'meta',view/'meta',dirs_exist_ok=True)
        stats=copy.deepcopy(original_stats)
        mask=np.isin(eps,selected)
        checks={}
        for key in keys:
            x=np.array(rows[key].to_pylist(),dtype=np.float32)[mask]
            low,high=np.quantile(x,[.01,.99],axis=0,method='linear')
            stats[key].update(q01=low.tolist(),q99=high.tolist(),min=x.min(0).tolist(),max=x.max(0).tolist(),mean=x.mean(0).tolist(),std=x.std(0).tolist(),count=[len(x)])
            denom=high-low
            constant=denom==0
            effective=np.where(constant,1.,denom)
            normalized=np.clip(2*(x-low)/effective-1,-1,1)
            restored=(normalized+1)/2*effective+low
            clipped=np.clip(x,low,high)
            assert np.isfinite(normalized).all()
            assert np.max(np.abs(restored-clipped))<1e-5
            checks[key]={'shape':list(x.shape),'constant_dims':np.where(constant)[0].tolist(),'clipped_fraction':float(((x<low)|(x>high)).mean()),'roundtrip_to_clipped_max_error':float(np.max(np.abs(restored-clipped)))}
        write_json(view/'meta/stats.json',stats)
        write_json(target/condition/'sim.json',[{'repo_id':TASK,'root':TASK,'train':str(selected),'test':''}])
        manifest['conditions'][condition]={'episodes':selected,'frames':int(mask.sum()),'stats_sha256':sha256(view/'meta/stats.json'),'checks':checks}
    for ep in [0,1,2,3]:
        write_json(target/f'human_ep{ep}.json',[{'repo_id':HUMAN,'root':HUMAN,'train':f'{ep}:{ep+1}','test':''}])
    shutil.copyfile(target/'human_ep0.json',target/'human_train.json')
    from examples.baselines.encoders.task_encoder.video_backbone import build_video_backbone
    backbone=build_video_backbone(backbone_type='dinov2_vitl14',latent_dim=256,max_seq_patches=32,adapter_layers=1,num_sampled_frames=10).to('cuda')
    backbone._load_backbone()
    backbone.to('cuda').eval()
    # Precompute without running the trainable Adapter; record actual decoded frames.
    for ep in [0,1]:
        seed_all(20261006+ep)
        proc=processor('A50',ep)
        reads=[]
        reader=proc.human_dataset.video_reader
        original=reader.read_frames
        def recorded(path, indices):
            reads.append({'path':str(path),'absolute_frames':[int(i) for i in indices]})
            return original(path,indices)
        reader.read_frames=recorded
        video=proc.get_video(TASK,1)
        assert list(video.shape)==[1,10,224,224,3]
        torch.save(video.clone(),target/f'human_ep{ep}.pt')
        with torch.no_grad():
            cls,seq=backbone._encode_dino(video.to('cuda'))
        cache=target/f'human_cache_ep{ep}'/'backbone_dinov2_vitl14'
        cache.mkdir(parents=True,exist_ok=True)
        torch.save({HUMAN:{'cls':cls[0].detach().cpu().clone(),'seq':seq[0].detach().cpu().clone()}},cache/'raw_features.pt')
        manifest['human'][str(ep)]={'sampling_seed':20261006+ep,'decode_reads':reads,'video_shape':list(video.shape),'video_sha256':sha256(target/f'human_ep{ep}.pt'),'cache_sha256':sha256(cache/'raw_features.pt'),'raw_cls_shape':list(cls.shape),'raw_seq_shape':list(seq.shape),'preprocessing':'official evaluate processor deterministic resize and normalization; fixed once'}
    manifest['elapsed_seconds']=time.time()-start
    manifest['frozen_model_files']=[{'path':str(p),'sha256':sha256(p)} for p in Path(os.environ['HF_HOME']).glob('hub/models--facebook--dinov2-large/snapshots/*/model.safetensors')]
    write_json(target/'manifest.json',manifest)
    print('PREPARATION PASSED',json.dumps(manifest['conditions']),flush=True)

if __name__=='__main__':
    main()
