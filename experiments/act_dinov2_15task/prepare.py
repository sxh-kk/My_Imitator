"""Audit official episode boundaries and freeze the published task configs."""
from common import *
import subprocess
import shutil
import numpy as np
import pyarrow.parquet as pq


def main():
    verified=json.loads((ROOT/'checks/data-download.json').read_text())
    assert verified['status']=='passed'
    dest=ROOT/'prepared'
    dest.mkdir(exist_ok=True)
    names={'human_train_config_15.json':'human_train_15.json','sim_train_config_15.json':'sim_train_15.json',
           'human_test_config_unseen.json':'human_fewshot.json','sim_test_config_unseen.json':'sim_fewshot.json',
           'human_eval_config_seen_plus_unseen_10tasks.json':'human_eval.json','sim_eval_config_seen_plus_unseen_10tasks.json':'sim_eval.json'}
    for src, name in names.items():shutil.copyfile(CONFIG/src,dest/name)
    groups={name:json.loads((dest/name).read_text()) for name in names.values()}
    trainids={x['repo_id'] for x in groups['sim_train_15.json']}
    fewids={x['repo_id'] for x in groups['sim_fewshot.json']}
    assert len(trainids)==60 and len(fewids)==20 and not trainids & fewids
    # The released loader checks every metadata video before selecting cameras.
    # A separate RGB view avoids requiring unused depth/mask videos; raw data and
    # statistics remain byte-identical. No loader/model source is changed.
    for repo in sorted(trainids|fewids):
        raw=DATA/'imitator_sim_v1_zed2i'/repo
        view=dest/'sim_rgb'/repo
        view.mkdir(parents=True,exist_ok=True)
        shutil.copytree(raw/'meta',view/'meta',dirs_exist_ok=True)
        info=json.loads((view/'meta/info.json').read_text())
        info['features']={k:v for k,v in info['features'].items() if v.get('dtype')!='video' or k=='observation.images.zed2i'}
        write_json(view/'meta/info.json',info)
        for folder in ['data','videos']:
            alias=view/folder
            if not alias.exists():alias.symlink_to(raw/folder,target_is_directory=True)
        assert sha256(view/'meta/stats.json')==sha256(raw/'meta/stats.json')
    rows=[]
    for condition, name in [('pretrain15','sim_train_15.json'),('scratch5','sim_fewshot.json')]:
        for cfg in groups[name]:
            root=DATA/'imitator_sim_v1_zed2i'/cfg['root']
            paths=sorted((root/'data').rglob('*.parquet'))
            table=pq.read_table(paths,columns=['episode_index','observation.qpos_gripper_states','action.qpos_gripper_actions'])
            ep=np.asarray(table['episode_index']);start,end=map(int,cfg['train'].split(':'))
            mask=(ep>=start)&(ep<end)
            assert set(ep[mask])==set(range(start,end)),cfg['repo_id']
            state=np.asarray(table['observation.qpos_gripper_states'].to_pylist(),dtype=np.float32)
            action=np.asarray(table['action.qpos_gripper_actions'].to_pylist(),dtype=np.float32)
            assert state.shape==(len(ep),18) and action.shape==(len(ep),16)
            assert np.isfinite(state).all() and np.isfinite(action).all()
            rows.append({'condition':condition,'repo_id':cfg['repo_id'],'episodes':end-start,'training_frames':int(mask.sum()),
                         'metadata_stats_sha256':sha256(root/'meta/stats.json'),'metadata_total_episodes':int(len(set(ep)))})
    assert sum(x['episodes'] for x in rows if x['condition']=='pretrain15')==3000
    assert sum(x['episodes'] for x in rows if x['condition']=='scratch5')==200
    tracked=subprocess.check_output(['git','ls-files','*.py'],cwd=SOURCE,text=True).splitlines()
    source={p:sha256(SOURCE/p) for p in tracked}
    provenance={'status':'passed','created_at':time.time(),'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=SOURCE,text=True).strip(),
                'upstream_python':source,'official_configs':{src:sha256(CONFIG/src) for src in names},
                'prepared_configs':{name:sha256(dest/name) for name in names.values()},'rows':rows,
                'normalization_protocol':'Released per-repo meta/stats.json unchanged; unseen zero-shot uses target metadata; few-shot data is 0:10 but metadata may summarize 50 demos.',
                'pretrain_frames':sum(x['training_frames'] for x in rows if x['condition']=='pretrain15'),
                'fewshot_frames':sum(x['training_frames'] for x in rows if x['condition']=='scratch5')}
    write_json(ROOT/'checks/data-boundaries.json',provenance)
    print('OFFICIAL DATA BOUNDARIES VERIFIED',provenance['pretrain_frames'],provenance['fewshot_frames'],flush=True)


if __name__=='__main__':offline();os.chdir(SOURCE);main()
