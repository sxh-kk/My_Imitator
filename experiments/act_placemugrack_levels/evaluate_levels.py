"""Sequential level evaluation with exact IDs and one fixed stats set."""
from common import *
import argparse
import subprocess


if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--checkpoint',required=True);ap.add_argument('--condition',required=True)
    ap.add_argument('--out',required=True);ap.add_argument('--seed-start',type=int,required=True)
    ap.add_argument('--episodes',type=int,required=True);ap.add_argument('--num-envs',type=int,default=4)
    ap.add_argument('--no-video',action='store_true')
    cli=ap.parse_args()
    for level in LEVELS:
        out=Path(cli.out)/level
        if (out/'result.json').exists(): continue
        command=[sys.executable,str(ROOT/'evaluate.py'),'--checkpoint',cli.checkpoint,'--condition',cli.condition,'--level',level,'--out',str(out),'--seed-start',str(cli.seed_start),'--episodes',str(cli.episodes),'--num-envs',str(cli.num_envs)]
        if cli.no_video: command.append('--no-video')
        subprocess.run(command,check=True)
