"""Standalone research plots from completed local results only."""
from common import *
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    results=json.loads((ROOT/'results.json').read_text())
    failure=json.loads((ROOT/'failure_analysis.json').read_text())
    figures=ROOT/'figures'; figures.mkdir(exist_ok=True)
    colors={'A50':'#1f77b4','A10':'#d97706'}
    fig,axes=plt.subplots(2,2,figsize=(11,7),constrained_layout=True)
    for run in results['runs']:
        name=run['run']; condition=name.split('_')[0]; seed=int(name[-1])
        rows=[json.loads(s) for s in (ROOT/'runs'/name/'train.jsonl').read_text().splitlines()]
        steps=[r['step'] for r in rows]
        for ax,key in zip(axes.flat[:3],['loss','l1','kl']):
            ax.plot(steps,[r[key] for r in rows],color=colors[condition],alpha=.45+.15*seed,label=name,linewidth=1)
            ax.set(xlabel='Optimizer updates',ylabel=key,title=f'Training {key}')
            ax.set_yscale('log'); ax.grid(alpha=.2)
        dev=run['development_results']
        axes[1,1].plot([r['step'] for r in dev],[100*r['success'] for r in dev],marker='o',color=colors[condition],alpha=.45+.15*seed,label=name)
    axes[1,1].set(xlabel='Optimizer updates',ylabel='Development success (%)',title='20 fixed development episodes per checkpoint',ylim=(-3,103))
    axes[1,1].grid(alpha=.2)
    axes[0,0].legend(fontsize=8,ncol=2)
    fig.suptitle('PlaceMugRack ACT: fixed batch 64 and 18,000 optimizer updates')
    fig.savefig(figures/'learning_curves.png',dpi=180); fig.savefig(figures/'learning_curves.pdf'); plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
    for ax,key,title in zip(axes,['success_once_count','success_at_end_count'],['Success at least once','Success at episode end']):
        for x,condition in enumerate(['A10','A50']):
            r=failure['conditions'][condition][key]
            rates=np.array(r['per_seed_rates'])*100
            ax.bar(x,np.mean(rates),color=colors[condition],alpha=.25,width=.55)
            ax.errorbar(x,np.mean(rates),yerr=r['sample_std']*100,color=colors[condition],capsize=5,fmt='s',label=f'{condition}: mean ± training-seed SD')
            ax.scatter(x+np.array([-.12,0,.12]),rates,color=colors[condition])
            for i,y in enumerate(rates): ax.annotate(f's{i+1}: {y:.0f}%',(x+[-.12,0,.12][i],y),xytext=([-12,0,12][i],-14-17*(i==1)),textcoords='offset points',ha='center',va='top',fontsize=8)
        ax.set(xticks=[0,1],xticklabels=['10 demonstrations','50 demonstrations'],ylabel='Final-test success (%)',ylim=(-3,113),title=title)
        ax.legend(loc='lower left',fontsize=8); ax.grid(axis='y',alpha=.2)
    fig.suptitle('100 held-out environment seeds per trained checkpoint')
    fig.savefig(figures/'final_success.png',dpi=180); fig.savefig(figures/'final_success.pdf'); plt.close(fig)
    print('Saved learning_curves and final_success as PNG/PDF')

if __name__=='__main__': main()
