"""Verify sparse episode selection and horizon padding against original parquet."""
from common import *
import numpy as np
import pyarrow.parquet as pq
import torch


def main():
    raw=pq.read_table(DATA/'imitator_sim_v1_zed2i'/TASK/'data/chunk-000/file-000.parquet').to_pydict()
    eps=np.array(raw['episode_index']); frames=np.array(raw['frame_index'])
    state=np.array(raw['observation.qpos_gripper_states'],dtype=np.float32)
    action=np.array(raw['action.qpos_gripper_actions'],dtype=np.float32)
    results={}
    for condition in ['A50','A10']:
        args=build_args(condition,1,64,0,ROOT/'prepared/human_cache_ep0')
        paired=build_dataset(args); paired.skip_human_video=True
        ds=paired.sim_dataset.main_dataset
        selected=json.loads((ROOT/'prepared/manifest.json').read_text())['conditions'][condition]['episodes']
        local_eps=np.array([int(v) for v in ds.hf_dataset['episode_index']])
        assert set(local_eps)==set(selected)
        stats=json.loads((Path(args.sim_root)/TASK/'meta/stats.json').read_text())
        checked=0
        for ep in selected:
            positions=np.where(local_eps==ep)[0]
            for local in [positions[0],positions[-1]]:
                metadata=ds.hf_dataset[int(local)]
                frame=int(metadata['frame_index'])
                global_index=int(metadata['index'])
                sample=paired[int(local)]
                ep_indices=np.where(eps==ep)[0]
                future=np.minimum(global_index+np.arange(24),ep_indices[-1])
                assert np.all(eps[future]==ep)
                expected=paired.sim_dataset.normalizer.normalize_action(torch.tensor(action[future]),0)
                assert torch.equal(sample['robot_actions'],expected),(condition,ep,frame)
                expected_state=paired.sim_dataset.normalizer.normalize_state(torch.tensor(state[global_index:global_index+1]),0)
                assert torch.equal(sample['robot_obs']['states'],expected_state)
                checked+=1
        results[condition]={'frames':len(paired),'episodes':selected,'episode_start_and_end_samples_checked':checked,'actions_match_original_parquet_and_pad_within_episode':True,'states_match_original_parquet':True}
    write_json(ROOT/'checks/data-boundaries.json',{'status':'passed','conditions':results})
    print('DATA BOUNDARY AUDIT PASSED',flush=True)

if __name__=='__main__': main()
