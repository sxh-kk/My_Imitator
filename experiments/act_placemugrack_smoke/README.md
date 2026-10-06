# PlaceMugRack ACT smoke run

已完成：2 条机器人轨迹、1 条训练 human demo、batch 8、3 epoch、171 optimizer steps；checkpoint 保存和逐 tensor 回读核验；2 个 500-step 仿真 episode 与录像。

任务成功率为 **0/2**，用途是验证运行流程。

完整协议、结果、存储位置和复跑命令见 [LearningDocs/04_ACT_SMOKE_RUN.md](../../LearningDocs/04_ACT_SMOKE_RUN.md)。

主要入口：`prepare_data.py`、`check_data.py`、`check_env.py`、`train.sh`、`evaluate.sh`。

首次运行结果：`train.log`、`training-metrics.json`、`evaluation-check.json`、`evaluation/`。后续评估默认使用新的 `evaluation_<时间>/` 目录，并在该目录内保存检查记录。
