"""Repair excluded ACT15 L3 cells without interrupting the training service."""
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]/'experiments/act_dinov2_15task'
sys.path.insert(0,str(ROOT))
from common import *
import fcntl
import subprocess
from fetch_assets import extract, ARCHIVES


def run(script,args,log_name):
    with (ROOT/'logs'/log_name).open('a',buffering=1) as log:
        log.write('\nOFFICIAL L3 REPAIR '+time.strftime('%Y-%m-%d %H:%M:%S')+'\n')
        subprocess.run([str(PYTHON),str(ROOT/script),*map(str,args)],cwd=SOURCE,
                       stdout=log,stderr=subprocess.STDOUT,check=True)


def main():
    lock=(ROOT/'l3-repair.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    marker=ROOT/'checks/l3-repair.json'
    plan=json.loads((ROOT/'plan.json').read_text())
    envs=[env for env in plan['seen_envs']+plan['unseen_envs'] if env.startswith('L3_')]
    checkpoint=ROOT/'runs/pretrain15/checkpoints/final_model.pt'
    expected=json.loads((ROOT/'runs/pretrain15/complete.json').read_text())['checkpoint_sha256']
    assert sha256(checkpoint)==expected
    try:
        write_json(marker,{'status':'running','stage':'additional_asset','pid':os.getpid(),'started_at':time.time()})
        # The official L3 FoldBox environment uses a different articulation.
        manifest=json.loads((ROOT/'asset-manifest.json').read_text())
        archive=next(x for x in manifest['files'] if x['path']=='partnet_mobility.tar.zst')
        path=ARCHIVES/archive['path']
        assert path.stat().st_size==archive['size'] and sha256(path)==archive['lfs']['oid']
        files=extract(path,{'objects':[],'partnet_ids':['100141']})
        assert files and any(x['path'].endswith('/100141/mobility.urdf') for x in files)
        write_json(ROOT/'checks/l3-additional-assets.json',{'status':'passed','revision':ASSET_REVISION,
            'archive_sha256':archive['lfs']['oid'],'files':files})
        for env in envs:
            run('probe_env.py',['--env',env],f'scene_{env}.log')
        for index,env in enumerate(envs):
            output=ROOT/'evaluation/pretrain15'/env
            result=output/'result.json'
            if result.exists():
                report=json.loads(result.read_text())
                assert report['registered_env']==env.split('_',1)[1].replace('-v','L3-v')
                continue
            write_json(marker,{'status':'running','stage':'evaluation','pid':os.getpid(),
                'env':env,'cell':index+1,'cells':len(envs),'updated_at':time.time()})
            run('evaluate.py',['--condition','pretrain15','--checkpoint',checkpoint,'--env',env,
                '--out',output,'--episodes',10],f'eval_pretrain15_{env}.log')
        for env in envs:
            report=json.loads((ROOT/'evaluation/pretrain15'/env/'result.json').read_text())
            old=json.loads((ROOT/'checks/protocol-amendments/01_official_l3_dispatch/excluded_evaluation'/env/'result.json').read_text())
            assert report['status']=='passed' and len(report['episodes'])==10
            assert report['checkpoint_sha256']==old['checkpoint_sha256']==expected
            assert report['human_video_sha256']==old['human_video_sha256']
            assert [x['seed'] for x in report['episodes']]==[x['seed'] for x in old['episodes']]==list(range(6000,6010))
            assert report['registered_env']==env.split('_',1)[1].replace('-v','L3-v')
        write_json(marker,{'status':'passed','corrected_cells':10,'corrected_episodes':100,
            'checkpoint_sha256':expected,'same_checkpoint_video_and_seeds':True,'finished_at':time.time()})
        print('OFFICIAL L3 REPAIR COMPLETE: 10 cells / 100 episodes',flush=True)
    except Exception as exc:
        write_json(marker,{'status':'needs_diagnosis','error':str(exc),'failed_at':time.time()})
        raise


if __name__=='__main__':main()
