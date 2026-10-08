"""Standalone scientific figures and a Chinese repository experiment report."""
from common import *
import csv
import datetime as dt
import time
import numpy as np


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    r=json.loads((ROOT/'results.json').read_text());audit=json.loads((ROOT/'checks/final-audit.json').read_text())
    failures=json.loads((ROOT/'failure_analysis.json').read_text())['cells'];plan=json.loads((ROOT/'plan.json').read_text())
    prepared=json.loads((ROOT/'prepared/manifest.json').read_text());state=json.loads((ROOT/'status.json').read_text())
    figroot=ROOT/'figures';figroot.mkdir(exist_ok=True)
    colors={'B0':'#4c78a8','B01':'#f58518','B012':'#54a24b'}
    fig,axes=plt.subplots(1,2,figsize=(12,4),gridspec_kw={'width_ratios':[1,2]})
    transfer=r['frozen_transfer']['conditions']['A50']
    axes[0].bar(np.arange(3),[transfer[l]['success_at_end']['mean']*100 for l in LEVELS],yerr=[transfer[l]['success_at_end']['sample_std']*100 for l in LEVELS],color='#999999',capsize=4)
    axes[0].set_xticks(range(3),LEVELS);axes[0].set_title('Frozen A50 transfer (diagnostic resets)')
    for i,(c,color) in enumerate(colors.items()):
        axes[1].bar(np.arange(3)+(i-1)*.24,[r['conditions'][c][l]['success_at_end']['mean']*100 for l in LEVELS],width=.23,yerr=[r['conditions'][c][l]['success_at_end']['sample_std']*100 for l in LEVELS],color=color,label=c,capsize=3)
    axes[1].set_xticks(range(3),LEVELS);axes[1].set_title('Supervised coverage (held-out resets)');axes[1].legend()
    for ax in axes:ax.set_ylim(0,105);ax.set_ylabel('Terminal success (%)');ax.grid(axis='y',alpha=.25);ax.set_axisbelow(True)
    fig.tight_layout();fig.savefig(figroot/'terminal-success.png',dpi=180);fig.savefig(figroot/'terminal-success.pdf');plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(13,4),sharey=True)
    for ax,l in zip(axes,LEVELS):
        for c,color in colors.items():
            series=[]
            for seed in [1,2,3]:
                s=json.loads((ROOT/'runs'/f'{c}_s{seed}'/'selection.json').read_text())
                series.append([x['levels'][l]['success_at_end']*100 for x in s['development']])
            array=np.array(series);mean=array.mean(0);std=array.std(0,ddof=1)
            ax.plot(plan['milestones'],mean,'o-',label=c,color=color);ax.fill_between(plan['milestones'],mean-std,mean+std,alpha=.15,color=color)
        ax.set_title(l);ax.set_xlabel('Optimizer updates');ax.grid(alpha=.2);ax.set_ylim(0,105)
    axes[0].set_ylabel('Development terminal success (%)');axes[-1].legend();fig.tight_layout();fig.savefig(figroot/'learning-curves.png',dpi=180);plt.close(fig)
    categories=['success_at_end','transient_success_only','never_grasped','grasped_without_lift','lifted_without_rack_contact','rack_contact_without_success']
    palette=['#54a24b','#b9d88a','#e45756','#f58518','#b279a2','#72b7b2']
    fig,axes=plt.subplots(1,3,figsize=(13,4),sharey=True)
    for ax,l in zip(axes,LEVELS):
        bottom=np.zeros(3)
        for cat,color in zip(categories,palette):
            x=np.array([np.mean([failures[f'{c}_s{s}/{l}']['counts'].get(cat,0) for s in [1,2,3]]) for c in colors])
            ax.bar(range(3),x,bottom=bottom,color=color,label=cat);bottom+=x
        ax.set_xticks(range(3),list(colors));ax.set_title(l);ax.set_ylim(0,100)
    axes[0].set_ylabel('Episodes (%)');fig.legend(loc='lower center',ncol=3,fontsize=8);fig.tight_layout(rect=(0,.15,1,1));fig.savefig(figroot/'failure-stages.png',dpi=180);plt.close(fig)
    gpu=[]
    for row in csv.reader((ROOT/'gpu-during-study.csv').read_text().splitlines()):
        try:gpu.append([float(x.strip()) for x in row[1:]])
        except (ValueError,IndexError):continue
    resource={'telemetry_samples':len(gpu),'gpu_utilization_mean_percent':float(np.mean([x[0] for x in gpu])),'gpu_memory_peak_GiB':float(np.max([x[1] for x in gpu])/1024),'controller':json.loads((ROOT/'controller.json').read_text()),'phase_a_started_at':state['started_at'],'wall_to_report_seconds':time.time()-state['started_at']}
    write_json(ROOT/'resource-summary.json',resource)
    lines=['# PlaceMugRack：L0／L1／L2 数据覆盖实验结果','',f'日期：{dt.datetime.now().date()}。状态：九组训练及全部 **6,300 条主实验 rollout** 已完成，最终审计通过。','',
        '本轮沿用官方 ACT 和 human video 条件，比较单任务三个场景级别的数据覆盖；核心源码和上一轮脚本、A50 权重均保持一致。方案见 [09](09_PLACEMUGRACK_LEVEL_EXPERIMENT_PLAN.md)。','',
        '## 1. 冻结 A50 的直接迁移','',
        '保持三个 A50 selected checkpoint、human_H57 episode 1、L0 训练统计量不变，在每个级别用 seeds 3000–3099 测试 100 次。以下均为三个训练 seed 的均值 ± 样本标准差；这不是置信区间。','']
    def metric(x):return f"{x['mean']*100:.2f}% ± {x['sample_std']*100:.2f} 个百分点"
    for l in LEVELS:lines.append(f"- **{l}**：终端成功 {metric(transfer[l]['success_at_end'])}；曾成功 {metric(transfer[l]['success_once'])}。")
    lines+=['','[冻结迁移逐模型记录](../experiments/act_placemugrack_levels/phase_a/summary.json)。该部分与新训练的最终测试使用不同 reset 集及 checkpoint 选择协议，不直接作为配对训练增益。','',
        '## 2. 三组固定预算训练的独立最终测试','',
        'B0 使用 50 条 L0；B01 使用 L0/L1 各 50 条；B012 使用 L0/L1/L2 各 50 条。每组三个训练 seed，每个 run B64、18,000 次更新；数据级别均衡采样，ACT/Adapter 从头初始化。','']
    for c in colors:
        lines.append(f'### {c}');lines.append('')
        for l in LEVELS:
            m=r['conditions'][c][l];counts=[f"{x['success_at_end_count']}/100" for x in sorted(r['rows'],key=lambda x:x['training_seed']) if x['condition']==c and x['level']==l]
            lines.append(f"- **{l}**：终端成功 {metric(m['success_at_end'])}（seed1/2/3：{', '.join(counts)}）；曾成功 {metric(m['success_once'])}。")
        macro=np.mean([r['conditions'][c][l]['success_at_end']['mean'] for l in LEVELS])*100
        lines+=['',f'三个级别等权终端成功均值：**{macro:.2f}%**。','']
    lines+=['![终端成功率](../experiments/act_placemugrack_levels/figures/terminal-success.png)','',
        '## 3. 配对增益与原有级别保持','']
    for pair,levels in r['paired_differences'].items():
        lines.append(f'### {pair}');lines.append('')
        for l in LEVELS:
            x=np.array(levels[l]['success_at_end'])*100
            lines.append(f"- {l}：终端成功平均变化 **{x.mean():+.2f} 个百分点**；配对 seed1/2/3：{', '.join(f'{v:+.1f}' for v in x)}。")
        lines.append('')
    lines+=['差值在相同训练 seed、同级别同 reset 初始状态下比较。100 条 rollout 不能替代多个独立训练模型，三个训练 seed 也不足以据此轻率宣称统计显著。','',
        '## 4. 失败阶段与动作尺度','',
        '阶段分类互斥，优先判断第500步成功，其次短暂成功，再按是否抓取、抬升、接触挂架定位。官方成功谓词为接触挂架、杯子超过高度阈值且当前未被夹持；终端成功不等于长期静止悬挂。','']
    for c in colors:
        for l in LEVELS:
            counts=collections_counts(failures,c,l,categories)
            desc='；'.join(f'{k}={v}' for k,v in counts.items())
            arms=[sum(x['arm_ever_grasped_counts'][a] for x in r['rows'] if x['condition']==c and x['level']==l) for a in [0,1]]
            lines.append(f'- **{c}/{l}**，300 条：{desc}。两臂曾抓取次数：{arms}（可能同一条两臂都抓过）。')
    lines+=['','![失败阶段](../experiments/act_placemugrack_levels/figures/failure-stages.png)','',
        'A50/B0 的第15维动作 q01=q99=1，反归一化后第二臂夹爪始终张开。B01 是否仍退化、B012 是否获得活动范围，以实际统计如下为准：','']
    for c in colors:
        stats=prepared['conditions'][c]['checks']['action.qpos_gripper_actions']
        lines.append(f"- {c}：第二臂夹爪 q01/q99={stats['q01'][15]:.6g}/{stats['q99'][15]:.6g}；零宽动作维={stats['zero_width_dims']}。")
    lines+=['',
        '每个模型跨 L0/L1/L2 只使用一套来自自身训练数据的共享 stats；三组之间 stats 不同。因此改善同时包含合法训练尺度覆盖和网络参数学习，不能只归因于视觉表征。部署截断比例、夹爪命令范围和两臂抓取逐 run/level 见 [results.json](../experiments/act_placemugrack_levels/results.json)。','',
        '## 5. 学习曲线、选择与结论边界','',
        '五个 checkpoint 在每个级别用 seeds 4000–4019、20次开发评估；按三个级别等权终端成功选择，平分时看曾成功，再取较早步数。九个选择全部冻结后，才使用 seeds 5000–5099 完成最终测试。','',
        '![开发学习曲线](../experiments/act_placemugrack_levels/figures/learning-curves.png)','',
        'B0/B01 的未训练级别开发评分参与了 checkpoint 选择，所以最终测试不称为从未接触目标级别反馈的严格 zero-shot。完整冻结的 A50 迁移单独报告。','',
        'B012 测试的是训练已覆盖的官方替代物体、场景级别中的新 reset；未留出新的杯子模型、语义任务或真实机器人。50/100/150 条示范总量及各级别重复呈现次数也变化，主实验回答的是固定计算预算下整套监督覆盖方案的收益。','',
        '## 6. 正确性与资源','',
        '- 600个真实轨迹首尾样本核对，动作padding、标签来源、human配对和stats映射通过。',
        '- 新评估包装器与旧包装器在相同四条轨迹上的动作、阶段和指标完全一致；训练更新与实际官方epoch源代码的参数更新完全一致。',
        '- 九组共162,000次更新、10,368,000次样本呈现；45份checkpoint回读及hash通过；主实验900次冻结评估、2,700次开发、2,700次最终测试。',
        '- 同级别同seed的场景、关节、RGB和机器人root pose配对检查通过；72段主录像均为501帧。',
        '- 执行前修正了方案对镜像的描述：官方ACT入口保持机器人底座，L2仅镜像场景物体；首次原型断言失败归档，未计入主实验。',
        f"- telemetry记录GPU利用率均值 {resource['gpu_utilization_mean_percent']:.1f}%，显存峰值 {resource['gpu_memory_peak_GiB']:.2f} GiB；采样自controller启动起，不覆盖最早的准备/部分阶段A。",
        f"- 从阶段A启动至报告生成约 {resource['wall_to_report_seconds']/3600:.2f} 小时，包含准备阶段重叠与计算；详细吞吐和并发选择见 [calibration/results.json](../experiments/act_placemugrack_levels/calibration/results.json)。",'',
        '完整审计：[final-audit.json](../experiments/act_placemugrack_levels/checks/final-audit.json)。权重索引：[final-test-selection.json](../experiments/act_placemugrack_levels/final-test-selection.json)。录像在每个 `runs/<run>/test/<level>/videos/`；失败补录若另行执行须单独标注。','']
    (WORKSPACE/'LearningDocs/10_PLACEMUGRACK_LEVEL_EXPERIMENT_RESULTS.md').write_text('\n'.join(lines)+'\n')
    print('REPORT WRITTEN',flush=True)


def collections_counts(failures,c,l,categories):
    return {k:sum(failures[f'{c}_s{s}/{l}']['counts'].get(k,0) for s in [1,2,3]) for k in categories}


if __name__=='__main__':main()
