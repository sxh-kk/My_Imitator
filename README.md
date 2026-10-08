# My_Imitator

Imitator / The Imitator Game 的个人复现工作区，包含源码、本机环境配置、中文源码导读、ACT 充分训练及 L0／L1／L2 数据覆盖实验。

上游：[The-Imitator-Game](https://github.com/imitator-game/The-Imitator-Game) · [项目主页](https://imitator-game.github.io/)。本快照基于 commit `d6d16ec511bc389e0a207692730c137bc022ef14`，保留上游及第三方许可证。

## 当前进展

截至 **2026-10-08**：

- 最小 ACT training → checkpoint → evaluation 已跑通，初始 smoke 为 0/2；参数和过程见 [04](LearningDocs/04_ACT_SMOKE_RUN.md)。
- L0 的 A10／A50 数据量对照已完成六次训练、600 次开发评估及600次独立最终测试。A50 终端成功率为 **95.33% ± 1.15 个百分点**，详见 [07](LearningDocs/07_PLACEMUGRACK_EXPERIMENT_RESULTS.md)。
- 新一轮 L0／L1／L2 覆盖实验**尚未完成**。冻结 A50 的900次迁移评估完成：L0终端成功率均值94.67%，L1／L2均0%。B012_s1完成18,000次更新和300次开发评估；其余八次训练及独立最终测试由队列接续执行。
- 10月8日上午发现旧控制器已退出，已归档中断的 B0_s1／B01_s1，并按原seed从头重跑。队列改由独立 systemd 用户服务运行；详见 [恢复记录](experiments/act_placemugrack_levels/RECOVERY.md)。

各实验使用不同的评估seed与checkpoint选择规则，分开报告。上述结果是本机实验结果；论文指标复现还需要匹配论文完整协议。核心 ACT 和仿真源码保持上游实现，本机环境兼容调整记录在依赖文件和锁文件中。

GitHub 内容是上传时的静态快照；上传时间及实际进程检查见 `PUBLICATION_STATUS.json`。本机可使用 `python scripts/check_levels_status.py` 同时检查进度文件与控制器进程，避免把旧的 `running` 记录当作当前运行状态。

## 目录导航

- [The-Imitator-Game/](The-Imitator-Game/)：上游源码及本机依赖配置、`uv.lock`。
- [LearningDocs/](LearningDocs/)：训练入口、数据流、human demo encoder、ACT、simulator、evaluation 的源码导读。
- [最小实验报告](LearningDocs/04_ACT_SMOKE_RUN.md)：实际运行参数、结果、文件位置和复跑命令。
- [environment/](environment/)：环境配置与检查记录。
- [experiments/act_placemugrack_smoke/](experiments/act_placemugrack_smoke/)：下载、训练、评估脚本与实验记录。
- [充分训练及数据量对照](LearningDocs/07_PLACEMUGRACK_EXPERIMENT_RESULTS.md)：A10／A50 六组结果与正确性检查。
- [L0／L1／L2 实验方案](LearningDocs/09_PLACEMUGRACK_LEVEL_EXPERIMENT_PLAN.md)及[阶段记录／最终报告](LearningDocs/10_PLACEMUGRACK_LEVEL_EXPERIMENT_RESULTS.md)：九组固定预算训练和独立测试。
- [experiments/act_placemugrack_levels/](experiments/act_placemugrack_levels/)：多级别实验执行脚本、训练统计量和小型数值记录。
- [研究复现规划](IMITATOR_RESEARCH_PLAN.md)：阶段方案和本机资源规划。

## 使用说明

先阅读 [环境说明](environment/README.md) 和 [最小实验报告](LearningDocs/04_ACT_SMOKE_RUN.md)。当前脚本记录了原实验机器上的绝对路径；在其他机器运行前，需要按实际位置调整 Conda、源码、数据和缓存路径。

Git 仓库包含源码、配置、文档、图件、数值评估、训练日志和checkpoint校验元数据。IG-10K 数据、下载的任务资产、checkpoint权重、视频、特征缓存及 Conda／Hugging Face／uv 缓存保存在实验机器上；下载脚本、版本清单及权重hash记录提供对应来源。中断运行的完整产物归档在本机，只上传恢复记录，其结果不计入正式对照。

发布目录保留当前工作区的 `The-Imitator-Game/` 子目录布局，源码作为普通文件提交，构成固定上游版本的完整快照。

## 许可证与引用

上游源码使用 [MIT License](The-Imitator-Game/LICENSE)。各第三方组件继续遵循其目录中的许可证；论文信息和引用方式见 [上游 README](The-Imitator-Game/README.md)。
