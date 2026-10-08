# PlaceMugRack：L0／L1／L2 实验运行记录

状态：**实验运行中，正式九组对照及独立最终测试尚未完成。** 本文件将由实验队列在全部计算及审计通过后替换为最终报告。

2026-10-08进度：B012_s1已完成18,000次更新、五个checkpoint和300次开发评估；B0_s1、B01_s1已自动启动并行训练。

**中断与恢复**：10月8日上午检查发现旧控制器及GPU实验进程已退出，退出原因未被日志记录。B0_s1与B01_s1各停在9,000次更新，已归档并从相同seed重跑；两次中断运行合计320条已完成开发评估仅作诊断，不计入正式主实验。09:28已通过独立systemd用户服务恢复队列；正式完成数仍为1/9模型、900条冻结评估和300条开发评估，重跑的实时增量以本机记录为准。详见[恢复记录](../experiments/act_placemugrack_levels/RECOVERY.md)。

方案：[09_PLACEMUGRACK_LEVEL_EXPERIMENT_PLAN.md](09_PLACEMUGRACK_LEVEL_EXPERIMENT_PLAN.md)。实时状态：[status.json](../experiments/act_placemugrack_levels/status.json)。

## 已完成的冻结 A50 评估

保持权重、human_H57 episode 1 视频与原 L0 训练统计量不变，每个模型、每个级别测试100次，共900次。

- A50_s1：L0终端成功94/100；L1、L2均0/100。
- A50_s2：L0终端成功98/100；L1、L2均0/100。
- A50_s3：L0终端成功92/100；L1、L2均0/100。

同级别同seed的初始场景、关节状态、RGB和robot root pose配对检查通过。逐模型记录见 [phase_a/summary.json](../experiments/act_placemugrack_levels/phase_a/summary.json)。

## 首个 B012 模型的开发结果

第5,400次更新，开发seeds4000–4019，每级别20次：

- L0：终端成功20/20。
- L1：终端成功18/20。
- L2：终端成功18/20；20条均由第二臂实际抓取，夹爪已能闭合。

按预先声明的宏平均终端成功率、宏平均曾成功率、较早更新数顺序，从五个checkpoint中选中第12,600次更新的checkpoint：L0终端成功20/20，L1为19/20，L2为18/20；三个级别曾成功率均为20/20。选择及五轮开发分数见 [selection.json](../experiments/act_placemugrack_levels/runs/B012_s1/selection.json)。

这些是开发评分，用于验证训练、checkpoint、共享统计量及三级仿真闭环；尚不能作为三组训练的独立最终收益结论。该模型已完成统一18,000次更新，队列继续剩余八个run，全部模型及选择冻结后才执行独立最终测试。

数据校验、600个真实边界样本、官方更新一致性、旧评估包装器一致性及新checkpoint三级执行检查均通过。标定中双训练总稳态吞吐相对单训练提高约17.2%，已配置最多两个并发训练任务和全局评估锁。

查看当前进度：

```bash
/home/zxc/miniconda3/envs/imitator/bin/python /home/zxc/Imitator/scripts/check_levels_status.py
/home/zxc/miniconda3/envs/imitator/bin/python /home/zxc/Imitator/experiments/act_placemugrack_levels/progress.py
```
