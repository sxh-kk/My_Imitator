# 仓库地图：六个核心模块

本文以 `examples/baselines/act/train_act_imitator.py` 对应的视频条件 ACT 为主线。路径均对应本机固定 commit，链接可定位到源码。

## 1. 训练入口

主文件：[train_act_imitator.py](../The-Imitator-Game/examples/baselines/act/train_act_imitator.py#L53)。

- `TrainingArgs`，第 53 行：模型维度、数据路径、训练参数、视频编码器和评估参数。
- `train()`，第 730 行：解析配置、构建 dataset、DataLoader、ACTAgent、optimizer、训练循环。
- 第 761 行创建 `PairedDatasetConfig`；第 793 行选择 `HumanSimPairedDataset`；第 828 行创建 DataLoader。
- 第 860 行创建 ACTAgent；冻结编码器预计算缓存后，第 864 行附近可能重建 DataLoader。
- 第 914 行开始 epoch 内循环：`batch → compute_loss → backward → optimizer step`。
- `save_checkpoint()`，第 654 行：保存 agent、optimizer、scheduler、参数配置等。

参考启动器：[run_exp_act.sh](../The-Imitator-Game/examples/baselines/exp_scripts/act/run_exp_act.sh#L12)。它传入 15/30/45-task JSON、`video_only`、10 帧编码、24 步预测等参数，并默认并行启动三个 GPU 作业。本次仅阅读该脚本。

区分三类配置来源：`TrainingArgs` 的默认值、启动脚本覆盖值、checkpoint 中保存的值。例如默认编码 4 帧，而启动脚本覆盖为 10 帧；默认 Transformer hidden dim 为 256，ACT README 另列有 1024 的消融配置。本文不会混用这些规模。

## 2. Dataset 与 DataLoader

外层配对：[lerobot_paired_dataset.py](../The-Imitator-Game/examples/baselines/lerobot_dataset/lerobot_paired_dataset.py#L241)。

- `PairedDatasetConfig`，第 44 行：人类与机器人数据共用配置。
- `TaskMapper`，第 111 行：读 `task_mapping.json`，建立 human、sim、real robot 的任务对应。
- `HumanSimPairedDataset.__getitem__`，第 434 行：取一个仿真时刻及其动作块，找对应的人类任务，组合为一个样本。
- `collate_paired_batch`，第 771 行：将样本堆成 batch；`get_collate_fn`，第 856 行：按输入模式选择 collate。
- `HumanRobotPairedDataset`，第 493 行：真实机器人数据分支。本次主线使用 sim 分支。

底层实现：

- [lerobot_dataloader.py](../The-Imitator-Game/examples/baselines/lerobot_dataset/lerobot_dataloader.py#L68)：`build_lerobot_dataset()` 按 `source_type` 构建 human/sim/robot dataset；它不是训练循环里的 PyTorch DataLoader 本体。
- [lerobot_sim_dataset.py](../The-Imitator-Game/examples/baselines/lerobot_dataset/lerobot_sim_dataset.py#L157)：`LeRobotSimDataset`，时间窗、图像变换、状态/动作字段选择、归一化。
- [lerobot_human_dataset.py](../The-Imitator-Game/examples/baselines/lerobot_dataset/lerobot_human_dataset.py#L320)：`HumanVideoDataset`，按任务挑选 episode，再采样视频帧。
- [lerobot_dataset.py](../The-Imitator-Game/examples/baselines/lerobot_dataset/lerobot_dataset.py#L1082)：仓库自带的 `LeRobotDataset` 实现，读取表格、视频和未来动作，处理 episode 边界。
- [video_utils.py](../The-Imitator-Game/examples/baselines/lerobot_dataset/video_utils.py#L127)：仿真数据路径的视频解码；human loader 另有 `VideoFrameReader`。
- [normalizer.py](../The-Imitator-Game/examples/baselines/lerobot_dataset/normalizer.py#L5)：`ActionNormalizer`，按 dataset 的统计量归一化与反归一化。
- [h5_to_lerobot.py](../The-Imitator-Game/examples/baselines/lerobot_dataset/h5_to_lerobot.py#L31)：数据转换与字段语义，可核查 18 维 state、16 维 action 的排列。

用于本文样本的配置：

- [task_mapping.json](../The-Imitator-Game/examples/baselines/lerobot_dataset/task_mapping.json#L752)：`human_H57 ↔ L0/L1/L2/L3_TwoRobotPlaceMugRack-v1`。
- [human_train_config_15.json](../The-Imitator-Game/examples/baselines/lerobot_dataset/config/exp_configs/human_train_config_15.json#L27)：人类任务数据目录与 episode 范围。
- [sim_train_config_15.json](../The-Imitator-Game/examples/baselines/lerobot_dataset/config/exp_configs/sim_train_config_15.json#L99)：对应仿真数据目录与 episode 范围。

## 3. Human demo encoder

核心文件：[video_backbone.py](../The-Imitator-Game/examples/baselines/encoders/task_encoder/video_backbone.py#L181)。

- `FrozenVideoBackbone`：包含预训练视觉主干与可训练 Adapter。
- `_load_dino()`，第 348 行：加载 `facebook/dinov2-large` 及 image processor，冻结主干参数。
- `_sample_frames()`，第 478 行：在 loader 给出的帧中再按编码帧数取样。
- `_encode_dino()`，第 530 行：逐帧 DINO 编码，时间维平均 CLS token，采样 patch tokens。
- `forward()`，第 685 行：控制冻结/梯度状态，执行 Adapter 和位置编码。
- `encode()`，第 755 行：返回 `{"z": ..., "z_seq": ...}`。

缓存文件：[frozen_backbone_cache.py](../The-Imitator-Game/examples/baselines/encoders/task_encoder/frozen_backbone_cache.py#L233)。

- `RawFeatureCache` 缓存 Adapter **之前**的特征。
- `CachedFrozenBackbone` 对缓存特征执行仍可训练的 Adapter。
- `setup_frozen_backbone_cache()`，第 564 行：加载或预计算缓存。
- `enable_skip_human_video()`，第 662 行：缓存可用后让 dataset 跳过 human video 解码。

与 ACT 的连接点是 [ACTAgent._encode_task()](../The-Imitator-Game/examples/baselines/act/train_act_imitator.py#L345)：当前只取 `enc_out["z"]`，再经过 `ftask_norm`。`z_seq` 没有传给当前 ACT 模型。

## 4. ACT policy

外层策略：[ACTAgent](../The-Imitator-Game/examples/baselines/act/train_act_imitator.py#L259)，与训练入口在同一文件中。

- `_preprocess_obs_for_model()`，第 374 行：统一图像维度，取最后一帧 state/RGB。
- `compute_loss()`，第 420 行：编码任务、预测动作块、计算 L1 与 KL。
- `prepare_for_eval()`，第 492 行：在 episode 开始时计算并保存任务向量。
- `get_action()`，第 515 行：用当前机器人观测输出整个动作块。

网络内部：

- [detr_video/backbone.py](../The-Imitator-Game/examples/baselines/act/act/detr_video/backbone.py#L85)：机器人当前 RGB 的 ResNet18 编码器。
- [detr_video/position_encoding.py](../The-Imitator-Game/examples/baselines/act/act/detr_video/position_encoding.py#L30)：视觉特征图的二维位置编码。
- [detr_video/detr_vae.py](../The-Imitator-Game/examples/baselines/act/act/detr_video/detr_vae.py#L33)：`DETRVAE`，训练时的 CVAE latent、任务条件、状态条件、动作头。
- [detr_video/transformer.py](../The-Imitator-Game/examples/baselines/act/act/detr_video/transformer.py#L21)：编码器、解码器及 learned action queries。

这里有两个视觉网络：DINOv2 编码**人类示范**，ResNet18 编码**机器人当前观测**。它们的参数、预处理和输出位置不同。`act/detr/` 也是现有目录，但本入口明确导入的是 `act/detr_video/`。

## 5. Simulator 与机器人控制

环境工厂：[act/make_env.py](../The-Imitator-Game/examples/baselines/act/act/make_env.py#L38) 的 `make_eval_envs()`，负责 `gym.make`、wrapper、vector env 和录像。

仿真主体：

- [sapien_env.py](../The-Imitator-Game/mani_skill/envs/sapien_env.py#L1174)：`BaseEnv.step()` / `_step_action()`，将 action 送入 agent、推进物理、生成下一帧观测和指标。
- [PlaceMugRack 任务](../The-Imitator-Game/mani_skill/envs/tasks/tabletop/dual_tasks/_038_place_mug_rack.py#L37)：`TwoRobotPlaceMugRackEnv`，场景、相机、reset、成功判定和奖励。
- [PlaceMugRack L3 任务](../The-Imitator-Game/mani_skill/envs/tasks/tabletop/dual_tasks_l3/_038_place_mug_rack_l3.py)：L3 环境入口。
- [L0_L3_utils.py](../The-Imitator-Game/mani_skill/envs/tasks/tabletop/utils/L0_L3_utils.py)：任务使用的级别开关和场景配置。
- [multi_agent.py](../The-Imitator-Game/mani_skill/agents/multi_agent.py#L73)：按机器人 UID 分发双臂动作。
- [panda_wristcam.py](../The-Imitator-Game/mani_skill/agents/robots/panda/panda_wristcam.py#L13) 与 [panda.py](../The-Imitator-Game/mani_skill/agents/robots/panda/panda.py#L77)：机器人模型与控制模式配置。
- [base_controller.py](../The-Imitator-Game/mani_skill/agents/controllers/base_controller.py#L277)：`CombinedController`，将单臂 8 维动作再拆成 arm / gripper。
- [pd_joint_pos.py](../The-Imitator-Game/mani_skill/agents/controllers/pd_joint_pos.py#L73)：设置关节位置目标，驱动模拟机器人。

项目中的 ManiSkill Python 层使用已安装的 SAPIEN；物理推进和渲染发生在该仿真栈内。离线 DataLoader 读取记录好的仿真轨迹，本身不会调用 `env.step()`。

## 6. Evaluation

主文件：[eval_act_imitator.py](../The-Imitator-Game/examples/baselines/act/eval_act_imitator.py#L825)。

- `main()`，第 825 行：读评估列表、加载模型与数据处理器、遍历任务。
- `load_agent_from_checkpoint()`，第 579 行：优先按 checkpoint 保存的参数重建 ACTAgent。
- `evaluate_single_env()`，第 678 行：创建任务环境，准备 rollout 参数，汇总结果。
- `evaluate_with_task_encoder()`，第 152 行：选择人类视频、调用策略、聚合动作、反归一化、`step()`、收集指标。
- `extract_base_env_name()`，第 547 行：把数据/评估 ID 转为 Gym 环境 ID；L3 会转到带 `L3` 后缀的环境。

共享处理文件：

- [evaluate_processor.py](../The-Imitator-Game/examples/baselines/lerobot_dataset/evaluate_processor.py#L54)：`HumanVideoSimEvaluateProcessor`，从任务 ID 找人类示范、从 sim metadata 找归一化统计。
- [flatten.py](../The-Imitator-Game/mani_skill/utils/wrappers/flatten.py#L14) 和 [frame_stack.py](../The-Imitator-Game/mani_skill/utils/wrappers/frame_stack.py#L11)：仿真观测转 RGB/state，并增加时间窗维度。
- [CPU Gym wrapper](../The-Imitator-Game/mani_skill/utils/wrappers/gymnasium.py#L55) / [GPU vector wrapper](../The-Imitator-Game/mani_skill/vector/wrappers/gymnasium.py#L110)：累计 `success_once`、`success_at_end` 等 episode 指标。
- [trajectory_metrics.py](../The-Imitator-Game/examples/baselines/lerobot_dataset/trajectory_metrics.py)：可选 DTW/TSS 诊断；不负责预测动作。
- [parallel_eval_act.py](../The-Imitator-Game/examples/baselines/act/parallel_eval_act.py)：评估调度入口；实际闭环仍需看上面的评估函数。

训练文件还内置 `evaluate()`（第 534 行），与独立评估入口的动作聚合默认值不同。本文执行链按独立的 `eval_act_imitator.py` 展开。

其他 baseline 的入口可以从 [examples/baselines/README.md](../The-Imitator-Game/examples/baselines/README.md) 导航；本次形状结论仅适用于上述 ACT 路径。
