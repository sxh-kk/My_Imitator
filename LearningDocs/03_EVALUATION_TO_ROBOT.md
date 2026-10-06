# 从 ACT 预测到 robot action：仿真评估路径

本文继续第二篇的配置，使用 `L0_TwoRobotPlaceMugRack-v1`、`physx_cpu`、`pd_joint_pos`、单路 zed2i RGB。`N` 表示并行环境数，与训练 batch 大小 `B` 分开记。

ACT 输出的是未来 24 步的动作块；模拟器每次 `step()` 接收左右臂各 8 维的当前步控制命令。

## 1. 独立评估的入口与环境 ID

从 [eval_act_imitator.main()](../The-Imitator-Game/examples/baselines/act/eval_act_imitator.py#L825) 出发：

```text
读取 eval-config 中的环境列表
  → load_agent_from_checkpoint(checkpoint)
  → HumanVideoSimEvaluateProcessor(...)
  → evaluate_single_env(env_id)
  → make_eval_envs_with_level(...)
  → evaluate_with_task_encoder(...)
```

模型结构优先取 checkpoint 的 `args`，不是简单使用评估 CLI 默认值。`load_state_dict(..., strict=False)` 会打印 missing/unexpected keys；“成功调用加载函数”本身不能说明所有权重都匹配。

数据/评估 ID 与 Gym ID 的对应由 [extract_base_env_name()](../The-Imitator-Game/examples/baselines/act/eval_act_imitator.py#L547) 处理：

- `L0_TwoRobotPlaceMugRack-v1` → `TwoRobotPlaceMugRack-v1`。
- L1、L2 同样去掉前缀，并设置相应场景级别开关。
- `L3_TwoRobotPlaceMugRack-v1` → `TwoRobotPlaceMugRackL3-v1`，同时设置 L3 开关。

本文具体的机器人、观测维度与成功判定按 L0 任务代码解释。

## 2. 环境如何包装观测

CPU 路径见 [make_eval_envs()](../The-Imitator-Game/examples/baselines/act/act/make_env.py#L62)：

```text
gym.make(TwoRobotPlaceMugRack-v1)
  → FlattenRGBDObservationWrapper
  → FrameStack(obs_horizon=1)
  → CPUGymWrapper(ignore_terminations=True, record_metrics=True)
  → RecordEpisode（设置 video_dir 时）
  → SyncVectorEnv（N=1）/ AsyncVectorEnv（N>1）
```

SAPIEN/ManiSkill 内部通常已有环境 batch 维，CPU wrapper 会处理单环境转换，外层 Gym vector env 再组织成 `N` 个环境。到策略前的维度如下：

| 边界 | RGB | state |
| --- | --- | --- |
| 任务默认相机 | 每环境 `(224,224,3)` | 原始 agent/extra 字典 |
| flatten + 时间窗 + vector env | `(N,1,224,224,3)` | `(N,1,D_raw)` |
| `normalize_state_rgb` 后 | `(N,1,224,224,3)` | `(N,1,18)` |
| ACT 观测预处理后 | `(N,1,3,224,224)` | `(N,18)` |

默认 [PlaceMugRack 相机配置](../The-Imitator-Game/mani_skill/envs/tasks/tabletop/dual_tasks/_038_place_mug_rack.py#L120) 是 224×224 的 `zed2i`；`hi_res=False`、`wrist_sensor=False`。Flatten wrapper 会把多台相机的图像沿最后一维拼接，因此上述 3 通道结论依赖单路相机条件。

## 3. 原始 state 怎么变成 18 维

[BaseAgent.get_proprioception()](../The-Imitator-Game/mani_skill/agents/base_agent.py#L336) 返回 `qpos` 与 `qvel`；[MultiAgent](../The-Imitator-Game/mani_skill/agents/multi_agent.py#L34) 按机器人顺序组织。

在本文的双 Panda、`pd_joint_pos` 条件下，flatten 后开头是：

```text
索引  0:9    panda_wristcam-0 的 qpos  (9,)
索引  9:18   panda_wristcam-0 的 qvel  (9,)
索引 18:27   panda_wristcam-1 的 qpos  (9,)
索引 27:36   panda_wristcam-1 的 qvel  (9,)
其后         任务 extra 字段
```

此 L0 任务的 RGB 模式 extra 包含四个 7 维 pose 和两个布尔值；由 [任务 _get_obs_extra()](../The-Imitator-Game/mani_skill/envs/tasks/tabletop/dual_tasks/_038_place_mug_rack.py#L286) 推导 `D_raw=36+28+2=66`。这是源码推导，未用本地任务资产实测；其他任务的 `D_raw` 可以不同。

[normalize_state_rgb()](../The-Imitator-Game/examples/baselines/lerobot_dataset/evaluate_processor.py#L174) 在最后一维大于 27 时执行：

```text
cat(state[...,0:9], state[...,18:27], dim=-1)
    (N,1,D_raw) → (N,1,18)
```

随后根据该 sim dataset 的 q01/q99 归一化。这个 qpos 路径保留双臂关节位置，丢掉 qvel 和任务 extra；即使原始观测含 mug/rack pose，它们也不会沿本文这条 state 输入进入 ACT。

RGB 在该函数中仅执行 `float()/255`；没有对机器人图像再次 resize。当前 224×224 来自任务的默认相机配置。

## 4. 评估时如何提供 human demo

[evaluate_with_task_encoder()](../The-Imitator-Game/examples/baselines/act/eval_act_imitator.py#L217) 在 rollout 循环外调用：

```text
evaluate_processor.get_video(env_id, num_envs=N)
  → sim env ID 映射为 human_H57
  → 每环境一个视频 tensor
  → human_video (N,T,224,224,3)
```

`get_video()` 的旧 docstring 写“返回 4D”，但 [实际返回值](../The-Imitator-Game/examples/baselines/lerobot_dataset/evaluate_processor.py#L242) 是 `torch.stack(videos)`，因此这里为 **5D**。

训练 `HumanVideoDataset._get_target_item()` 和评估处理器的 [_get_video()](../The-Imitator-Game/examples/baselines/lerobot_dataset/evaluate_processor.py#L293) 都会从配置筛选后的可用 episode 中随机选择，再采样帧。需要区分的是调用频率：视频在一次 `evaluate_with_task_encoder()` 调用中取一次，后续 episode reset 不重新调用 `get_video()`，所以同一批 rollout 会复用已经取到的视频。

每次 episode 的 `ts=0`：

```text
agent.prepare_for_eval(human_video, robot_obs)
  → _encode_task(...)
  → _cached_task_z (N,256)
```

后续 `get_action(obs)` 使用保存的向量，避免每个控制步重复运行 DINO。这里是**一次 episode 的 task_z 缓存**；第二篇的训练缓存则是**按 human repo 存储的 Adapter 前特征**，两者层级不同。

## 5. ACT 输出整个 chunk

[ACTAgent.get_action()](../The-Imitator-Game/examples/baselines/act/train_act_imitator.py#L515)：

```text
当前 RGB / state
  → _preprocess_obs_for_model
  → model(obs, actions=None, video_feature=_cached_task_z)
  → a_hat (N,24,16)
```

推理不输入未来真实动作；CVAE 的 32 维 latent 置零。机器人观测和 task_z 仍参与预测。模型输出来自线性 action head，不能假定数值天然限制在 `[-1,1]` 内。

## 6. Chunk 怎么变成当前步 action

独立评估默认开启 [light_temporal_agg](../The-Imitator-Game/examples/baselines/act/eval_act_imitator.py#L175)，`tagg_window=4`：

- 每 **4 个控制步**查询一次策略，每次仍预测 **24 个动作**。
- 最多保留最近 4 次查询产生的 chunk。
- 从每个 chunk 中取出对应当前时刻的那一个动作。
- 按 chunk 的年龄加权求和，较新的预测权重更大。

例子：在 `ts=4` 时，已有 `ts=0` 与 `ts=4` 的两个预测，使用新 chunk 的第 0 项和旧 chunk 的第 4 项：

```text
new_chunk[:,0,:]                         (N,16)
old_chunk[:,4,:]                         (N,16)
stacked                                 (N,2,16)
权重 ∝ [exp(0), exp(-0.1)]，再归一化
sum over chunk dimension → raw_action   (N,16)
```

代码位置：[动作聚合](../The-Imitator-Game/examples/baselines/act/eval_act_imitator.py#L272)。`raw_action` 此时仍处于训练标签的归一化空间。

函数另有完整 temporal aggregation 和直接按 chunk 顺序执行两个分支。当前 `light_temporal_agg` 优先级更高；CLI 将其定义为 `store_true, default=True`，所以不能仅看到 `--temporal-agg` 就假定启用了完整分支。这里只说明现状。

## 7. 反归一化与左右臂拆分

`pd_joint_pos` 不含字符串 `delta`，因此 [评估循环](../The-Imitator-Game/examples/baselines/act/eval_act_imitator.py#L321) 会反归一化：

```text
raw_action (N,16)
  → evaluate_processor.denormalize_action(raw_action, env_id)
  → 选择此任务 dataset 的 action q01/q99
  → action = 0.5 * (raw_action + 1) * (q99 - q01) + q01
  → (N,16)
```

反归一化函数本身没有再 clamp。CPU 物理路径将 tensor 转成 NumPy，随后直接切片：

```python
action = {
    "panda_wristcam-0": action_array[:, :8],   # (N,8)，本文任务的左臂
    "panda_wristcam-1": action_array[:, 8:16], # (N,8)，本文任务的右臂
}
obs, reward, terminated, truncated, info = eval_envs.step(action)
```

左/右与 agent 0/1 的对应见 [PlaceMugRack.left_agent / right_agent](../The-Imitator-Game/mani_skill/envs/tasks/tabletop/dual_tasks/_038_place_mug_rack.py#L251)。

这里有两种不同的归一化：dataset 的 q01/q99 缩放，以及机器人 gripper controller 的动作缩放。恢复到数据集动作空间之后，臂关节命令是绝对关节角目标；夹爪那一维仍是控制器约定的输入，不能把所有 16 维一概理解成物理关节位置。

## 8. `env.step()` 之后的控制调用链

```text
vector env / wrappers
  → BaseEnv.step(action)
  → BaseEnv._step_action(action)
  → MultiAgent.set_action(action_dict)
  → 每个 BaseAgent.set_action(action_for_this_robot)
  → CombinedController.set_action(...)
  → arm PDJointPosController + gripper PDJointPosMimicController
  → articulation.set_joint_drive_targets(...)
  → scene.step() 推进物理
```

对应 [BaseEnv](../The-Imitator-Game/mani_skill/envs/sapien_env.py#L1205)、[MultiAgent](../The-Imitator-Game/mani_skill/agents/multi_agent.py#L73)、[CombinedController](../The-Imitator-Game/mani_skill/agents/controllers/base_controller.py#L287)、[PDJointPosController](../The-Imitator-Game/mani_skill/agents/controllers/pd_joint_pos.py#L73)。

CPU vector env 将 `(N,8)` 分给各个单环境时，单环境接收 `(8,)`；`BaseEnv` 将其恢复为带环境维的 `(1,8)`。随后：

| 单臂控制边界 | shape | 语义 |
| --- | --- | --- |
| CombinedController 输入 | `(1,8)` | 本臂的完整命令 |
| arm 子控制器 | `(1,7)` | 7 个臂关节的目标位置 |
| gripper 子控制器输入 | `(1,1)` | 1 个夹爪控制量 |
| gripper mimic 目标 | `(1,2)` | 同一目标广播给两个手指关节 |

[Panda 的控制配置](../The-Imitator-Game/mani_skill/agents/robots/panda/panda.py#L81) 对 arm 使用 `normalize_action=False`、默认 `use_delta=False`。夹爪保留 controller 的归一化处理，将输入裁剪并缩放到配置的 `[-0.01,0.04]` 目标范围，再给两个手指关节使用。

这些是 PD 控制的目标，机器人通过物理模拟向目标运动。`BaseEnv._step_action()` 在一个控制周期内多次调用 `scene.step()`，然后读取下一时刻的观测。

## 9. Success 与评价结果来自哪里

策略不输出成功标签。每个任务通过 `evaluate()` 产生 `info["success"]`，wrapper 再累计 episode 指标。

本文 L0 任务的 [evaluate()](../The-Imitator-Game/mani_skill/envs/tasks/tabletop/dual_tasks/_038_place_mug_rack.py#L259) 当前使用：

```text
success = mug 接触 rack
          AND mug 高于其参考高度 + 0.08
          AND 没有被任一机器人抓住
```

函数也计算 `is_obj_placed` 和 `is_robot_static`，但此版本最终 `success` 表达式没有使用这两项。阅读代码时应以最终逻辑为准。

[CPUGymWrapper](../The-Imitator-Game/mani_skill/utils/wrappers/gymnasium.py#L55) 记录：

- `success_once`：episode 内至少一次成功。
- `success_at_end`：episode 最后一步是否成功。
- `return`、`episode_len`：累计奖励与控制步数。

评估设置 `ignore_terminations=True`；循环主要在 `truncated` 时提取 `final_info["episode"]`、计数、清除 task 缓存并 reset。随后 [evaluate_single_env()](../The-Imitator-Game/examples/baselines/act/eval_act_imitator.py#L749) 写出指标均值、标准差与任务信息，录像由 `RecordEpisode` 管理。

结果 JSON 中 `status="success"` 表示此次评估流程完成；机器人任务表现应看 `success_once_mean` 等数值字段。可选 DTW/TSS 由 `trajectory_metrics.py` 提供，独立于 ACT 动作生成。

## 10. 这条链路的验证边界

本次合成样本动态核查实际覆盖了配对/collate、视频与机器人观测编码、CVAE 训练分支、推理分支、动作头以及 `(1,16) → 两个 (1,8)` 的切片。反归一化演示使用合成统计量。

真实任务的 env 创建、机器人动作执行、任务成功率没有在本次文档工作中运行；controller、环境和评分部分来自固定版本源码阅读。基础 Panda 渲染在先前环境配置阶段验证过，但不等于 PlaceMugRack 闭环已经验证。
