"""Report only completed official protocol cells and audit paired rollouts."""
from common import *
import csv
import subprocess
import numpy as np


def main():
    plan=json.loads((ROOT/'plan.json').read_text())
    cells=[]
    expected={'pretrain15':plan['seen_envs']+plan['unseen_envs'],'scratch5':plan['unseen_envs'],'finetune5':plan['unseen_envs']}
    reports={}
    for cond,envs in expected.items():
        run=json.loads((ROOT/'runs'/cond/'complete.json').read_text())
        assert run['status']=='complete' and run['epochs']==100
        assert sha256(run['checkpoint'])==run['checkpoint_sha256']
        reports[cond]=run
        for env in envs:
            r=json.loads((ROOT/'evaluation'/cond/env/'result.json').read_text())
            assert r['status']=='passed' and len(r['episodes'])==10
            assert [x['seed'] for x in r['episodes']]==list(range(6000,6010))
            assert r['checkpoint_sha256']==run['checkpoint_sha256']
            cells.append({'condition':cond,'env':env,'group':r['group'],'SR':r['SR'],
                          'Sub_SR_tau_095':r['Sub_SR_tau_095'],'success_once':r['success_once']})
    for env in plan['unseen_envs']:
        rs=[json.loads((ROOT/'evaluation'/cond/env/'result.json').read_text()) for cond in expected]
        assert len({r['human_video_sha256'] for r in rs})==1,(env,'human video differs')
        for i in range(10):
            for field in ['initial_rgb_sha256','initial_state','robot_roots','seed']:
                assert all(r['episodes'][i][field]==rs[0]['episodes'][i][field] for r in rs),(env,i,field)
    aggregates={}
    for cond,group in [('pretrain15','seen'),('pretrain15','unseen'),('scratch5','unseen'),('finetune5','unseen')]:
        part=[c for c in cells if c['condition']==cond and c['group']==group]
        assert len(part)==20
        aggregates[f'{cond}_{group}']={metric:float(np.mean([r[metric] for r in part])) for metric in ['SR','Sub_SR_tau_095','success_once']}
    gain={metric:aggregates['finetune5_unseen'][metric]-aggregates['scratch5_unseen'][metric] for metric in ['SR','Sub_SR_tau_095']}
    provenance=json.loads((ROOT/'checks/data-boundaries.json').read_text())
    changed=[p for p,h in provenance['upstream_python'].items() if sha256(SOURCE/p)!=h]
    assert not changed,changed
    assert reports['scratch5']['frames']==reports['finetune5']['frames']==provenance['fewshot_frames']
    assert reports['scratch5']['iterations']==reports['finetune5']['iterations']
    source=json.loads((ROOT/'execution-source-hashes.json').read_text())
    assert {p.name:sha256(p) for p in ROOT.glob('*.py')}==source
    result={'status':'complete','scope':plan['scope'],'cells':cells,'aggregates':aggregates,'PFT_minus_Scratch':gain,
            'formal_episodes':800,'training_runs':reports,'audit':{'upstream_unchanged':True,'outer_scripts_unchanged':True,
            'paired_reset_and_video':True,'equal_fewshot_budget':True},'limitations':[
            'one training seed and ten trials per cell','only 15-task scale; paper Table3 also averages 30/45',
            'unpublished Sub-SR threshold replaced by explicitly authorized 0.95',
            'target task metadata statistics available, faithful to released evaluator',
            'Transformer size, warmup and few-shot training duration defaulted where paper is unspecified'],
            'finished_at':time.time()}
    write_json(ROOT/'results.json',result)
    with (ROOT/'results.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(cells[0]));writer.writeheader();writer.writerows(cells)
    lines=['# ACT＋DINOv2 15 任务仿真复现结果','',
           '状态：训练与 800 个正式 rollout 已完成，源文件和配对审计通过。',
           '', '协议详见 [第十一篇](11_ACT_DINOV2_15TASK_REPRODUCTION.md)。这里只覆盖一个 seed、15 任务规模、ACT＋DINOv2 的仿真结果。',
           '', '## 主要结果','']
    labels={'pretrain15_seen':'预训练：已见任务','pretrain15_unseen':'预训练：未见任务零样本','scratch5_unseen':'未见任务：少样本从零训练','finetune5_unseen':'未见任务：预训练后少样本微调'}
    for key,row in aggregates.items():lines.append(f"- {labels[key]}：SR {row['SR']:.1%}，Sub-SR@0.95 {row['Sub_SR_tau_095']:.1%}，success_once {row['success_once']:.1%}。")
    lines+=['',f"P+FT−Scratch：SR {gain['SR']*100:+.1f} 个百分点，Sub-SR@0.95 {gain['Sub_SR_tau_095']*100:+.1f} 个百分点。",'',
            '## 逐任务、逐级别结果','',
            '全部 80 个单元的数值保存在 [results.csv](../experiments/act_dinov2_15task/results.csv) 和 [results.json](../experiments/act_dinov2_15task/results.json)。原始 episode、阶段峰值和固定第一个 episode 的视频位于 evaluation/ 对应目录。',
            '', '## 解释边界','',
            '已见、未见零样本和少样本结果分开统计。仅当本次 P+FT 高于 Scratch 时，才称本次协议下预训练有收益；如果不高于，照实报告，不挑 checkpoint 或更改预算。单 seed 结果不能证明稳定的总体趋势。',
            '', 'Sub-SR 使用用户同意的临时阈值 0.95，官方阈值尚未公开；原始阶段峰值可重算。官方目标任务元数据统计保持可用，这里的零样本描述权重没有目标训练，不能称统计信息也完全零样本。',
            '', '论文主表还平均了 30／45 任务规模，本次不会将单档结果当作主表的等价数值复现。架构、warmup 和少样本时长采用源码默认值的部分在第十一篇明确列出。',
            '', '## 审计','', '三个运行均完成 100 epochs；Scratch 和 P+FT 的帧数、更新数、示范 indices、统计和测试 reset 相同。未见任务没有进入预训练；测试没有参与 checkpoint 选择；所有已登记核心 Python 源码和外层执行脚本 hash 保持一致。','']
    (WORKSPACE/'LearningDocs/12_ACT_DINOV2_15TASK_RESULTS.md').write_text('\n'.join(lines))
    write_json(ROOT/'checks/completion-verification.json',{'status':'passed','formal_cells':80,'formal_episodes':800,'upstream_unchanged':True,'finished_at':time.time()})
    set_status('complete',status='complete',finished_at=time.time(),formal_episodes=800)


if __name__=='__main__':main()
