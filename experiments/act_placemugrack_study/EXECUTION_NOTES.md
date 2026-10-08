# 执行记录

## 实施前验证

- 所有新增代码位于本实验目录。上游核心 Python 文件未修改。
- 固定步数训练与官方 epoch 更新语句进行同输入、同权重、同 RNG 对照，损失和 253 个状态 tensor 完全一致。
- 两个数据条件分别计算逐维 q01/q99；原始下载文件未修改。另准备 D2 诊断条件。
- human episode 0/1 的采样帧、视频 tensor、DINO raw feature 和预训练模型文件均记录 hash。
- 脚本尚未实现 DataLoader worker RNG 的精确中断恢复；正式 run 若中断，优先从确定起点重跑，避免误称精确 resume。

## 标定中的修正

1. 评估包装器调用官方 parser 时遗漏其必填 output-dir 参数；已补齐。此时尚未启动正式训练。
2. 共用随机种子 helper 同时开启了训练的 TF32 开关，最初的 N=1/2 第一动作最大差异约 0.002535。修正为独立官方 evaluator 的默认推理设置：关闭 matmul/cudnn TF32。旧记录保存在 `calibration/eval_tf32_n1`、`eval_tf32_n2`。
3. 修正后，N=1/2/4 在同一组 reset seed 上的初始物体位姿、关节状态和 RGB hash 全部一致；N=2/4 的第一动作相对 N=1 最大差异约 0.000020，满足预设 atol=1e-5、rtol=1e-4。物理接触会放大浮点差异，因此不声称完整轨迹逐 bit 一致；正式条件统一使用同一种 num_envs。
4. 主实验关闭辅助 DTW/TSS，所有条件一致。成功判定、动作聚合与物理设置不变。
5. 每次主评估只预先选取前两批的第一个环境录像，其余 episode 完整评分、不录视频。N=4 时开发录像 seeds 为 1000/1004，最终录像 seeds 为 2000/2004。需要展示额外成功或失败案例时另行补录，并标明补录。

## 标定初步结果

- workers=4 时 batch 32/64/128 的吞吐分别约 944/921/931 samples/s。
- batch 64、workers=0 约 272 samples/s，workers=8 约 1543 samples/s。
- 500 步稳态复测：单训练约 1478 samples/s；两个训练合计约 1753 samples/s，提升约 18.6%。
- 不录视频的四条 rollout：N=1/2/4 分别约 14.51/8.93/6.47 秒；这些时间不含全部进程和环境初始化。

最终选择记录于 `execution-config.json`；完整原始测量见 `calibration/`。
