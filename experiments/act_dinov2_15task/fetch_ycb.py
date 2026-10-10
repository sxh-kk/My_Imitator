"""Use the released checksum-pinned ManiSkill YCB asset downloader."""
from common import *
import subprocess


if __name__=='__main__':
    os.chdir(SOURCE)
    target=ASSETS/'assets/mani_skill2_ycb'
    marker=ROOT/'checks/ycb-download.json'
    if not target.exists():
        subprocess.run([str(PYTHON),'-m','mani_skill.utils.download_asset','ycb','--non-interactive'],check=True)
    models=target/'models'
    required=['011_banana','013_apple','014_lemon','015_peach','028_skillet_lid']
    for model in required:assert (models/model).is_dir(),(models/model)
    files=[{'path':str(p),'sha256':sha256(p),'bytes':p.stat().st_size} for model in required for p in (models/model).rglob('*') if p.is_file()]
    write_json(marker,{'status':'passed','official_archive_sha256':'1551724fd1ac7bad9807ebcf46dd4a788caed5c9499c1225b9bfa080ffbefcb3','files':files})
