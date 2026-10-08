# PlaceMugRack 多级别 ACT 实验

执行协议见 [LearningDocs/09](../../LearningDocs/09_PLACEMUGRACK_LEVEL_EXPERIMENT_PLAN.md)。

状态：**2026-10-08 13:54:56 CST全部完成**。九组训练、6,300条主实验rollout、45个checkpoint及最终审计通过；结果见 [LearningDocs/10](../../LearningDocs/10_PLACEMUGRACK_LEVEL_EXPERIMENT_RESULTS.md)。用户服务已正常退出，勿重复启动已完成队列。

本目录使用原 ACT、paired dataset、human encoder 和 evaluator；新代码仅负责固定预算、级别均衡采样、训练数据共享统计量、显式仿真级别、记录与审计。

查看实验状态和数值记录：

```bash
/home/zxc/miniconda3/envs/imitator/bin/python /home/zxc/Imitator/scripts/check_levels_status.py
/home/zxc/miniconda3/envs/imitator/bin/python /home/zxc/Imitator/experiments/act_placemugrack_levels/progress.py
```

先用 `check_levels_status.py` 检查真实控制器进程和telemetry更新时间，再读 `progress.py` 的数值进度。状态文件可能在控制器被外部终止后仍保留 `running`。

2026-10-08恢复后，队列由独立systemd用户服务承载。检查服务可运行 `systemctl --user show imitator-placemugrack-levels.service -p ActiveState -p MainPID`；正常首启入口为 `bash scripts/start_levels_service.sh`（在工作区根目录执行）。已存在中断运行时须先按协议归档，不能直接重复启动。背景与计数边界见 [RECOVERY.md](RECOVERY.md)。

`status.json` 为整体阶段；`runs/<run>/train.jsonl` 为真实更新日志；`phase_a/<A50_run>/<level>/progress.json` 与 `runs/<run>/dev|test/.../progress.json` 为逐批评估记录。

`execute.py` 接续阶段A，依次完成校验、标定、pilot、剩余训练、全部模型冻结后的最终测试、审计和报告。它通过 `execution.lock` 避免重复启动。部分正式run不做近似恢复：保存采样计划、消费游标、RNG和optimizer/scheduler用于诊断；若worker增强状态无法精确恢复，归档中断run后从同一seed重跑。

不要在已有正式进程运行时另启一套队列；旧experiment目录不被本队列写入。
