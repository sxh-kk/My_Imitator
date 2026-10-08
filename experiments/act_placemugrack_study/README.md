# PlaceMugRack ACT 实验执行目录

执行方案：[LearningDocs/06_4090_EXPERIMENT_DESIGN.md](../../LearningDocs/06_4090_EXPERIMENT_DESIGN.md)。本目录只增加实验包装器，官方核心 Python 源码保持原样。

## 文件

- `study-plan.json`：数据条件、训练预算和开发/测试协议。
- `prepare.py`：派生 A50/A10/D2 的训练统计，生成固定 human episode 0/1 及冻结特征缓存。
- `train.py`：调用官方 ACTAgent 和 compute_loss 的固定步数训练、原子 checkpoint 保存和开发评估。每次更新沿用官方 epoch 分支的优化器分组、BF16 和梯度裁剪。
- `evaluate.py`：调用官方 evaluate_with_task_encoder；增加实际 reset seed、初始场景、参数回读、动作和计分检查。
- `calibrate.py`、`calibrate_runtime.py`：batch/workers、并发训练、并行环境及训练/评估重叠的实测。
- `run_study.py`：先完成全量试运行，再执行固定实验矩阵，冻结 checkpoint 选择后运行最终测试。
- `summarize.py`：聚合结果、核验配对初始化、报告三次训练的均值及样本标准差。
- `progress.py`：读取训练和评估进度，显示队列状态。
- `analyze_results.py`、`plot_results.py`、`write_report.py`：最终测试完成后生成失败分类、图件和第七篇结果报告。
- `final_audit.py`：最终核对六组训练计数、30 份权重 hash、1,200 条 rollout 和 72 段视频；建立 best/last/final 权重符号链接。
- `collect_examples.py`、`record_example.py`：索引成功和失败视频；缺少所需类别的录像时补录同一四环境批次，补录明确排除在主实验计分之外。
- `finish_study.py`：等待主队列完成后依次运行后处理和审计，再更新报告与文档状态；进度保存在 `postprocess-status.json`。

## 当前运行与复查

使用 `/home/zxc/miniconda3/envs/imitator/bin/python`。`common.py` 读取现有 conda 环境变量，使用已下载的模型、数据和资产，默认离线运行。

观察 `status.json`、`runs/<run>/train.jsonl` 和各评估目录的 `progress.json`；`result.json` 只在该次评估全部检查通过后产生。训练末尾的 `complete.json` 不等于任务成功，成功率在独立评估结果内。

本轮还启动了独立的 `finish_study.py` 后处理队列。`status.json` 的 complete 表示主训练/测试已完成；`postprocess-status.json` 的 complete 表示分析、视频示例、审计和报告也已完成。主队列出现 failed/needs_diagnosis 时后处理停止，避免把不完整实验写成已完成。

短测目录和正式 `runs/` 相互独立。已有完成文件时脚本拒绝覆盖。主调度支持跳过已完整完成的 run；部分完成的训练需先诊断，再从确定起点重跑。checkpoint 保存 RNG 与采样游标，但本实现没有恢复 DataLoader worker 内部随机状态的功能，因此不声称逐 bit 精确续训。

评估默认固定 human episode 1；DTW/TSS 不计入本实验，所有条件一致关闭。额外 `SceneProbe` 只读取初始物体位姿，不修改场景、观测、控制或成功判定。

## 完成后的权重和复查

每组 `runs/<run>/checkpoints/best_model.pt` 指向仅由开发集选出的权重；`last.pt` 与 `final_model.pt` 指向 18,000 步权重。链接由最终审计脚本生成，不重复占用权重空间。checkpoint 包含策略、适配层、优化器和实验参数；冻结 DINO 仍从既有预训练模型缓存加载，数据 processor 还需要对应条件的统计文件。

主队列完成后依次执行 `analyze_results.py`、`plot_results.py`、`collect_examples.py`、`final_audit.py`、`write_report.py`。不继续训练或重新选择权重；补录单独记录原始与重放指标是否一致。结果报告保存在 `LearningDocs/07_PLACEMUGRACK_EXPERIMENT_RESULTS.md`。

## 前置验证证据

- `prepared/manifest.json`：50 条/11,228 帧、10 条/2,241 帧的统计范围、裁剪比例、反归一化检查、人类采样帧、DINO 文件 hash。
- `checks/update_equivalence/update-equivalence.json`：同一 batch、同一初始化/RNG 下，执行从官方 epoch 分支提取的实际更新语句，与本驱动的损失及 253 个状态 tensor 完全一致；另外检查 warmup/cosine 关键步数。
- `calibration/`：吞吐、显存、批量推理与种子一致性记录；未进入正式模型选择。

准备阶段固定人类视频时使用官方 evaluation processor 的 resize/归一化，关闭一次性的随机图像增强；训练期间机器人图像仍使用官方默认增强。此项已在视频缓存 manifest 中明确记录。
