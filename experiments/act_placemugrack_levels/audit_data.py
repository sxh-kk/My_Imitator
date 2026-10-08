"""Actual paired samples, trajectory boundaries and shared stats checks."""
from common import *
import numpy as np
import pyarrow.parquet as pq
import torch


def main():
    prepared=json.loads((ROOT/'prepared/manifest.json').read_text())
    raw={l:pq.read_table(DATA/'imitator_sim_v1_zed2i'/task(l)/'data/chunk-000/file-000.parquet').to_pydict() for l in LEVELS}
    reports={}; examples={}
    for condition,levels in CONDITIONS.items():
        args=build_args(condition,1,64,0,ROOT/'prepared/human_cache_ep0')
        paired=build_dataset(args);paired.skip_human_video=True
        paired.sim_dataset.config.enable_augmentation=False
        paired.sim_dataset._setup_transforms_sim(paired.sim_dataset.config)
        assert len(paired)==sum(prepared['frames'][l] for l in levels)
        assert paired.valid_indices==list(range(len(paired)))
        seen=0;checked=0;offset=0
        for dsidx,level in enumerate(levels):
            ds=paired.sim_dataset.lerobot_dataset.datasets[dsidx]
            assert ds.repo_id==task(level)
            local_eps=np.asarray(ds.hf_dataset['episode_index'],dtype=int)
            assert set(local_eps)==set(range(50))
            array=raw[level]
            states=np.asarray(array['observation.qpos_gripper_states'],dtype=np.float32)
            actions=np.asarray(array['action.qpos_gripper_actions'],dtype=np.float32)
            for ep in range(50):
                positions=np.where(local_eps==ep)[0]
                assert np.array_equal(np.asarray(array['frame_index'])[positions],np.arange(len(positions)))
                for local in [positions[0],positions[-1]]:
                    sample=paired[offset+int(local)]
                    assert sample['sim_task_id']==task(level) and sample['human_repo_id']==HUMAN
                    assert int(sample['dataset_idx'])==dsidx
                    future=np.minimum(local+np.arange(24),positions[-1])
                    expected=paired.sim_dataset.normalizer.normalize_action(torch.tensor(actions[future]),dsidx)
                    expected_state=paired.sim_dataset.normalizer.normalize_state(torch.tensor(states[local:local+1]),dsidx)
                    assert torch.equal(sample['robot_actions'],expected),(condition,level,ep,int(local))
                    assert torch.equal(sample['robot_obs']['states'],expected_state)
                    assert sample['robot_obs']['view_1'].shape==(1,224,224,3)
                    assert np.all(local_eps[future]==ep)
                    assert int(sample['robot_actions'].shape[-1])==16
                    checked+=1
                seen+=1
            if condition=='B012':
                positions=np.where(local_eps==0)[0]
                examples[level]=[]
                for fraction in [0.,.33,.67,1.]:
                    local=int(positions[int((len(positions)-1)*fraction)])
                    image=paired[offset+local]['robot_obs']['view_1'][0].numpy()
                    examples[level].append(image)
            offset+=len(ds)
        proc=processor(condition)
        for l in LEVELS:
            assert proc._resolve_dataset_idx(task(l))==LEVELS.index(l)
            values=proc.normalizer.dataset_stats[LEVELS.index(l)]
            train_stats=json.loads(stats_path(condition).read_text())
            for key in ['observation.qpos_gripper_states','action.qpos_gripper_actions']:
                assert np.array_equal(np.asarray(values[key]['q01']),np.asarray(train_stats[key]['q01']))
                assert np.array_equal(np.asarray(values[key]['q99']),np.asarray(train_stats[key]['q99']))
        reports[condition]={'frames':len(paired),'episodes':seen,'boundary_samples_checked':checked,'exact_pairing_and_labels':True,'eval_stats_identical_all_levels':True,'stats_sha256':sha256(stats_path(condition))}
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(3,4,figsize=(12,9))
    for row,level in enumerate(LEVELS):
        for col,img in enumerate(examples[level]):
            axes[row,col].imshow(img);axes[row,col].axis('off');axes[row,col].set_title(f'{level}, episode 0, {col}/3')
    fig.tight_layout();fig.savefig(ROOT/'checks/official-demo-contact-sheet.png',dpi=140);plt.close(fig)
    write_json(ROOT/'checks/data-boundaries.json',{'status':'passed','conditions':reports})
    print('DATA AUDIT PASSED',reports,flush=True)


if __name__=='__main__':main()
