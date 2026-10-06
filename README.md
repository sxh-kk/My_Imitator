# My_Imitator

Imitator / The Imitator Game 的个人复现工作区，包含源码、本机环境配置、源码学习记录与最小 ACT 闭环实验。

上游：[The-Imitator-Game](https://github.com/imitator-game/The-Imitator-Game) · [项目主页](https://imitator-game.github.io/)。本快照基于 commit `d6d16ec511bc389e0a207692730c137bc022ef14`，保留上游及第三方许可证。

## 当前进展

已使用官方 `L0_TwoRobotPlaceMugRack-v1` 数据完成：

- 2 条机器人轨迹、1 条人类训练示范，batch 8、3 epoch，共 171 次参数更新。
- ACT checkpoint 保存、加载和 692 个 state tensor 的逐元素一致性核验。
- 2 个仿真 episode，共 1,000 步动作执行，生成评估结果和录像。

任务成功率为 **0/2**。本轮验证运行流程，不代表论文性能复现。核心策略和仿真代码保持上游实现；本机环境兼容调整记录在依赖文件和锁文件中。

## 目录导航

- [The-Imitator-Game/](The-Imitator-Game/)：上游源码及本机依赖配置、`uv.lock`。
- [LearningDocs/](LearningDocs/)：训练入口、数据流、human demo encoder、ACT、simulator、evaluation 的源码导读。
- [最小实验报告](LearningDocs/04_ACT_SMOKE_RUN.md)：实际运行参数、结果、文件位置和复跑命令。
- [environment/](environment/)：环境配置与检查记录。
- [experiments/act_placemugrack_smoke/](experiments/act_placemugrack_smoke/)：下载、训练、评估脚本与实验记录。
- [研究复现规划](IMITATOR_RESEARCH_PLAN.md)：阶段方案和本机资源规划。

## 使用说明

先阅读 [环境说明](environment/README.md) 和 [最小实验报告](LearningDocs/04_ACT_SMOKE_RUN.md)。当前脚本记录了原实验机器上的绝对路径；在其他机器运行前，需要按实际位置调整 Conda、源码、数据和缓存路径。

Git 仓库包含源码、配置、文档和小型运行证据。IG-10K 数据、下载的任务资产、训练 checkpoint、特征缓存以及 Conda / Hugging Face / uv 缓存保存在实验机器上；下载脚本与版本清单提供了对应来源。

发布目录保留当前工作区的 `The-Imitator-Game/` 子目录布局，源码作为普通文件提交，构成固定上游版本的完整快照。

## 许可证与引用

上游源码使用 [MIT License](The-Imitator-Game/LICENSE)。各第三方组件继续遵循其目录中的许可证；论文信息和引用方式见 [上游 README](The-Imitator-Game/README.md)。
