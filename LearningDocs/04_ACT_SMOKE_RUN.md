# ACT 最小真实任务闭环实测

实测日期：2026-10-06。Conda 环境：`imitator`。

**已完成官方数据 → ACT 训练 → checkpoint 保存与回读 → 官方仿真评估 → 录像。**
选用 `L0_TwoRobotPlaceMugRack-v1`。训练完成 171 次参数更新，评估完成 2 个 episode、共 1,000 步；任务成功率 **0/2**。这是流程验证结果，不是论文性能复现结果。

本轮没有修改仓库核心代码。实验配置、下载和检查脚本位于 [experiments/act_placemugrack_smoke](../experiments/act_placemugrack_smoke/)。仓库现有的 `.gitignore`、`pyproject.toml`、`uv.lock` 环境配置改动保持原状。

## 1. 固定的任务、数据与训练配置

- 官方任务：数据 ID `L0_TwoRobotPlaceMugRack-v1`；Gym ID `TwoRobotPlaceMugRack-v1`；对应人类任务 `human_H57`。
- 机器人训练轨迹：episode **0、1**，分别 224、229 帧，总计 **453 个训练样本**。
- 人类训练示范：episode **0**，原始 47 帧，每次加载采样 10 帧。
- 人类评估示范：episode **1**，原始 61 帧，采样 10 帧。没有把训练的人类示范直接用于本次评估。
- 图像：单路 `zed2i` RGB，224×224；双臂 qpos state 18 维、action 16 维。
- 模型：官方 ACT；机器人图像 ResNet18；人类视频冻结 DINOv2-L；task latent 256；ACT hidden 256、encoder 2 层、decoder 4 层；预测 24 步。
- 训练：seed 1，batch **8**，3 epoch，每 epoch 57 batch，最后一个 batch 为 5；共 **171 optimizer steps**。
- 优化器：AdamW，主学习率 `1e-4`，backbone 参数组学习率 `1e-5`，cosine scheduler，**warmup_epochs=0**；KL 权重 10；沿用官方 BF16 autocast 与梯度裁剪。
- 数据增强、动作归一化和缓存沿用官方实现。归一化使用发布数据的 `meta/stats.json`，没有用这两条轨迹重新估计统计量。
- 训练期间关闭在线评估，训练后在独立进程加载 checkpoint 做评估。

episode 范围由四份独立 JSON 控制：[训练 human](../experiments/act_placemugrack_smoke/configs/human_train.json)、[训练 sim](../experiments/act_placemugrack_smoke/configs/sim_train.json)、[评估 human](../experiments/act_placemugrack_smoke/configs/human_eval.json)、[评估 sim](../experiments/act_placemugrack_smoke/configs/sim_eval.json)。官方独立评估入口读取配置中的 `train` 字段，因此评估 human JSON 的 `train` 写成 `1:2`；这不表示 episode 1 参与了训练。

发布的数据将多条 episode 合并到同一个 Parquet/MP4 块中。本轮保留原始块，所以物理下载包含该 task 更多 episode；实际训练范围由配置限制为上述两条机器人轨迹和一条人类示范。

## 2. 实测结果与证据

三个 epoch 的平均训练 loss 分别为 **8.5693 → 2.0815 → 1.6757**。这只能说明本次小样本优化过程正常，不能代替在线任务成功率。

checkpoint 回读检查通过：

- 保存的 `iteration=171`、`epoch=3`，scheduler 的 `last_epoch=171`。
- 有 optimizer state 的参数，其 step 均为 171，且存在非零一阶动量。
- 评估模型重建后，**692 个 state_dict tensor 均与 checkpoint 逐元素完全相等**，并通过有限值检查。
- checkpoint SHA256：`502114b33cbc2b5fa70e76cc65a07f4267facb29100461b1008780f4ae353d9c`。

评估使用官方 `eval_act_imitator.main()`、官方环境工厂和控制/计分链路：

- `physx_cpu` 物理后端、GPU `rt-fast` 渲染、`pd_joint_pos` 控制；1 个并行环境。
- 2 个 episode，每个上限 500 步，两个都执行满 500 步。
- 使用官方默认轻量 temporal aggregation：每 4 步查询策略，窗口为 4。
- 总计 **250 次策略查询、1,000 次 env.step**；预测动作、传入模拟器的双臂动作、返回的观测和 reward 均通过有限值检查。
- `success_once=0.0`，`success_at_end=0.0`；两次 return 为 47.1370、46.6825，均值 **46.9098**。
- 两个 MP4 均为 H.264、512×512、30 fps、501 帧、16.7 秒。501 帧包括初始画面与 500 次 step。
- 当前录像显示机械臂靠近杯子但未完成任务；本轮没有针对失败表现修改策略或延长训练。

官方结果 JSON 的 `status: "success"` 表示评估程序完成；任务是否成功应读取 `success_once_mean` / `success_at_end_mean`。本次这两个值均为 0。

评估入口默认还计算 DTW/TSS，本轮保留其输出，但不以这些辅助指标声称性能复现。这里的两次在线 rollout 也不构成正式 benchmark 测试集。

原始证据：

- [真实 DataLoader shapes](../experiments/act_placemugrack_smoke/data-check.json)、[场景与控制检查](../experiments/act_placemugrack_smoke/env-check.json)。
- [训练日志](../experiments/act_placemugrack_smoke/train.log)、[TensorBoard 导出的指标](../experiments/act_placemugrack_smoke/training-metrics.json)。
- [checkpoint 与 rollout 检查](../experiments/act_placemugrack_smoke/evaluation-check.json)、[评估日志](../experiments/act_placemugrack_smoke/evaluate.log)。
- [官方评估结果 JSON](../experiments/act_placemugrack_smoke/evaluation/eval_envs_video_only_20261006_215521.json)、[CSV](../experiments/act_placemugrack_smoke/evaluation/eval_envs_video_only_20261006_215521.csv)。
- Episode 0 录像（本机文件：`experiments/act_placemugrack_smoke/evaluation/videos/L0_TwoRobotPlaceMugRack-v1/0.mp4`）、Episode 1 录像（本机文件：`experiments/act_placemugrack_smoke/evaluation/videos/L0_TwoRobotPlaceMugRack-v1/1.mp4`）。
- [checkpoint 与视频文件校验记录](../experiments/act_placemugrack_smoke/artifact-check.json)。

## 3. 本次真实运行中的关键 shape

训练 DataLoader，`B=8`：

```text
human_video                  [8, 10, 224, 224, 3]
robot_obs.states             [8, 1, 18]
robot_obs.view_1              [8, 1, 224, 224, 3]
robot_actions                [8, 24, 16]
robot_first_frame.view_1      [8, 1, 3, 224, 224]
```

上述为缓存启用前的真实 batch 检查。训练入口预计算人类任务特征后启用 `skip_human_video`，随后从缓存取 raw CLS `[8,1024]` 和 patch sequence `[8,32,1024]`，Adapter 继续训练。缓存以 `human_H57` 为 key，共 1 项。本轮只有一条训练 human episode，因此缓存取样来源明确。磁盘缓存中的单项 shape 记录在 `artifact-check.json`。

独立评估，`N=1`：

```text
归一化后的 obs.state         [1, 1, 18]
归一化后的 obs.rgb           [1, 1, 224, 224, 3]
ACT get_action               [1, 24, 16]
聚合、反归一化后的 action     [1, 16]
panda_wristcam-0              [1, 8]
panda_wristcam-1              [1, 8]
```

ACT 内部张量链路仍见 [02_SAMPLE_TO_ACT.md](02_SAMPLE_TO_ACT.md)，控制执行链见 [03_EVALUATION_TO_ROBOT.md](03_EVALUATION_TO_ROBOT.md)。本篇补上真实文件读取、参数更新和机器人闭环的运行证据。

## 4. 本轮遇到的兼容细节

1. **只下载 RGB 时，LeRobot 完整性检查先于相机筛选执行。** Sim metadata 声明了 `zed2i_depth`，缺少该文件会触发远端下载分支。补齐官方深度视频块 17,498,968 bytes 后解决；模型仍使用 RGB，没有修改 metadata 或 loader。
2. **资产按需流式解包。** 官方 `robotwin.tar.zst` 为 14,764,813,696 bytes。本轮从固定 revision 的压缩流中取出 `039_mug` 与 `040_rack`，必需 JSON 和 GLB 到齐后停止读取；没有保存整个压缩包，也没有下载其他资产包。实际所需 GLB 已检查文件长度和内嵌资源，真实场景加载通过。记录的是选出文件的本地 SHA256，**没有声称校验整个压缩包的 SHA256**。
3. **checkpoint 初次加载会报告 439 个 frozen DINO unexpected keys。** 原因是评估时 DINO 延迟加载，而 checkpoint 已包含它。本轮先使用官方加载函数，再显式触发同一预训练 DINO 的延迟加载，随后比较所有 692 个 state tensor，确认完全一致；没有忽略可训练参数缺失问题。
4. **检查脚本只增加观测和断言。** [evaluate_checked.py](../experiments/act_placemugrack_smoke/evaluate_checked.py) 调用官方评估入口，在 checkpoint 加载、策略输出和环境 step 外记录 shape、有限值与计数；没有替换动作聚合、归一化、控制器或成功判定。Python / NumPy / Torch seed 固定为 1，环境沿用 ManiSkill 默认主 RNG 初始 seed 2022 及后续 reset 序列。

仓库当前 ACT decoder 层选择等实现行为保持原样；此次不引入新方法或核心逻辑修复。

## 5. 文件位置与存储预算

- 数据：`/var/tmp/imitator-game-zxc/data/`，13 个原始发布文件，合计 **180,989,653 bytes ≈ 172.61 MiB**；`du -sh` 为 **173 MiB**。
- 任务资产：`/var/tmp/imitator-game-zxc/maniskill/data/robotwin/objects/`，44 个提取文件，合计 **48,037,502 bytes ≈ 45.81 MiB**；`du -sh` 为 **46 MiB**。
- checkpoint：[final_model.pt](../The-Imitator-Game/runs/act_placemugrack_smoke-20261006/checkpoints/final_model.pt)，**1,422,994,538 bytes ≈ 1.33 GiB**；`du -sh` 显示约 **1.4 GiB**。
- 实验目录：`experiments/act_placemugrack_smoke/`，约 12 MiB，包括约 10 MiB feature cache、配置、日志、预览图和两段录像。
- 初次运行的 TensorBoard events：`The-Imitator-Game/runs/act_placemugrack_smoke-20261006/`。

本轮新增文件总量约 **1.55 GiB**，不含此前已经安装的环境与预训练权重缓存。2026-10-06 检查时，数据所在根分区剩余约 **200 GiB**，checkpoint 所在 `/home` 剩余约 **98 GiB**，本轮运行空间充足。

## 6. 复查与复跑

只查看已有结果和录像不需要再次训练。以下命令使用现有环境与数据。

```bash
source /home/zxc/Imitator/scripts/activate_imitator.sh
export PYTHONPATH=/home/zxc/Imitator/The-Imitator-Game

# 真实数据/场景预检查
python /home/zxc/Imitator/experiments/act_placemugrack_smoke/check_data.py
python /home/zxc/Imitator/experiments/act_placemugrack_smoke/check_env.py

# 用已保存的 checkpoint 再跑两个 episode；默认创建新的 evaluation_<时间> 目录
bash /home/zxc/Imitator/experiments/act_placemugrack_smoke/evaluate.sh
```

重新训练用新的实验名，避免覆盖已有 checkpoint：

```bash
IMITATOR_EXP_NAME=act_placemugrack_repeat \
  bash /home/zxc/Imitator/experiments/act_placemugrack_smoke/train.sh

# 将日期替换为新训练的实际日期；评估新 checkpoint
IMITATOR_CHECKPOINT=/home/zxc/Imitator/The-Imitator-Game/runs/act_placemugrack_repeat-YYYYMMDD/checkpoints/final_model.pt \
  bash /home/zxc/Imitator/experiments/act_placemugrack_smoke/evaluate.sh
```

脚本参数见 [train.sh](../experiments/act_placemugrack_smoke/train.sh) 和 [evaluate.sh](../experiments/act_placemugrack_smoke/evaluate.sh)。评估检查器的 171 steps / 2 episodes / 1,000 env steps 断言是针对本 smoke 配置的；改变训练时长或 episode 数量后应同步调整断言。新评估的检查记录写入其输出目录下的 `evaluation-check.json`。

重新校验或补齐下载文件：

```bash
# base Python 已有 zstandard；不需要改动 imitator 环境
/home/zxc/miniconda3/bin/python /home/zxc/Imitator/experiments/act_placemugrack_smoke/prepare_data.py dataset
/home/zxc/miniconda3/bin/python /home/zxc/Imitator/experiments/act_placemugrack_smoke/prepare_data.py assets
```

数据文件逐个核对官方 LFS SHA256 或 Git blob SHA1；资产仅下载并提取任务所需目录。配置和校验清单见 [download-manifest.json](../experiments/act_placemugrack_smoke/download-manifest.json)、[download-verified.json](../experiments/act_placemugrack_smoke/download-verified.json)、[assets-verified.json](../experiments/act_placemugrack_smoke/assets-verified.json)。

## 7. 固定来源

- 代码 commit：`d6d16ec511bc389e0a207692730c137bc022ef14`。
- [官方数据集固定 revision](https://huggingface.co/datasets/imitator-game/IG-10K-Dataset/tree/57fa861d911afe899da5c0f28d973151411d46d1)：`57fa861d911afe899da5c0f28d973151411d46d1`。
- [官方资产固定 revision](https://huggingface.co/datasets/imitator-game/IG-10K-Assets/tree/2d7a339b27aff14a0780db4bd4d5e3a68c6358bc)：`2d7a339b27aff14a0780db4bd4d5e3a68c6358bc`。
- 环境与依赖记录：[environment/README.md](../environment/README.md)。
