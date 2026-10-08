# 2026-10-08 中断恢复及 GitHub 同步记录

09:26 CST检查发现：原控制器PID和GPU实验进程均不存在，GPU telemetry最后更新时间距检查约8.9小时。`status.json`仍记录running，不能据此判断进程存活。原日志没有记录退出原因；尚无证据确定是应用会话退出、外部终止或其他原因。

## 保留和重跑

- B012_s1已完整完成18,000次更新、五个checkpoint及300条开发评估，保留全部原产物和选择。
- B0_s1、B01_s1均停在9,000次更新，各已完成160条开发评估。完整文件移动到本机 `interrupted_runs/20261008_092805/`；checkpoint和训练日志hash见 [recovery-history.json](recovery-history.json)。
- 两个中断run从原seed 1、相同初始化及原冻结协议重新训练。未从9,000步近似续训：checkpoint没有精确恢复DataLoader worker图像增强状态的能力。
- 中断run的18,000次合计更新及320条开发评估均不计入正式九组对照，不混入新run的开发选择和最终测试。原始记录仍可用于诊断。

## 承载方式及复核

09:28 CST通过 `scripts/start_levels_service.sh` 启动独立systemd用户服务 `imitator-placemugrack-levels.service`。控制器及其所有训练、评估worker属于该服务的cgroup，脱离当前工具会话。服务没有失败自动重启：部分run必须先归档，从同一seed重跑。用户登出或机器关机仍可能停止用户服务，应检查真实进程，不能假定永不退出。

14个已冻结执行脚本、plan、execution-config、训练数据统计量和human输入hash在重启前均与冻结记录一致；已选B012 checkpoint hash也一致。未修改核心ACT、仿真源码或训练协议。新增加的服务启动、进程检查及GitHub同步工具均位于工作区 `scripts/`。

```bash
python scripts/check_levels_status.py
python experiments/act_placemugrack_levels/progress.py
systemctl --user show imitator-placemugrack-levels.service -p ActiveState -p SubState -p MainPID
journalctl --user -u imitator-placemugrack-levels.service -n 20 --no-pager
```

最终报告中的 `wall_to_report_seconds` 从原阶段A启动时间计算，包含此次中断等待及重跑开销，不能解释为连续主动计算时长。GPU telemetry均值按实际采样记录计算，中断期间缺失的采样不补零。最终审计和主实验更新／rollout数只统计完整正式run。

## GitHub 上传范围

上传源码、依赖配置、中文文档、图件、数值结果、完整行训练日志、训练stats及checkpoint校验元数据。权重、视频、原始数据和缓存保留在本机；中断run仅上传本恢复说明及hash记录。

仓库是上传时间的静态快照，采集时间和实际进程状态见仓库根目录 `PUBLICATION_STATUS.json`。上午提交 `19168cc` 时实验仍未完成，阶段报告明确区分开发评分和独立最终测试。同步脚本 `scripts/sync_github_snapshot.py` 只更新发布checkout，git commit和push须由调用者显式执行。

## 完成复核

2026-10-08 **13:54:56 CST**：九组训练、900次冻结评估、2,700次开发评估和2,700次独立最终测试全部完成，45个checkpoint和72段主录像审计通过。服务退出码0，控制器进程正常结束。恢复服务后的剩余流程用时约4.45小时；正式162,000次更新不包含归档run的18,000次额外更新。

完成后已复核14个执行脚本hash未变、状态与汇总及最终审计一致，并将报告、导航和发布状态更新为完成。最终报告见 [LearningDocs/10](../../LearningDocs/10_PLACEMUGRACK_LEVEL_EXPERIMENT_RESULTS.md)。
