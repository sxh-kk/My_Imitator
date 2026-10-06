# 一条样本：从 DataLoader 到 ACT 动作块

## 0. 本文采用的配置与符号

主线是 `data_source=sim`、`input_mode=video_only`、`state_type=qpos`、双臂、RGB、`cameras=["zed2i"]`。
采用默认 ACT 网络规模，并按官方启动脚本把 `frozen_backbone_num_frames` 从默认 4 改为 10。

| 符号 | 含义 | 本文数值 |
| --- | --- | --- |
| `B` | 训练 batch 大小 | 一般写 `B`，动态核查为 1 |
| `T` | human dataset 返回的帧数 | `task_num_frames=10` |
| `F` | DINO 实际编码帧数 | `min(frozen_backbone_num_frames,T)=10` |
| `O` | 机器人观测时间窗 | `obs_horizon=1` |
| `Q` | 预测动作块长度 | `pred_horizon=24` |
| `M` | 策略读取的相机数 | 1 |
| `H,W,C` | 图像高、宽、通道 | 224、224、3 |
| `D_s,D_a` | state、action 维度 | 18、16 |
| `d` | ACT Transformer hidden dim | 256 |
| `D_h,D_task,D_vae` | DINO、任务向量、CVAE latent 维度 | 1024、256、32 |
| `L_d` | ACT decoder 层数 | 4 |

这些数值是明确选择的一条代码路径；其他 backbone、RGBD、多相机、真实机器人或不同 checkpoint 不应直接套用全部 shape。

## 1. 先确定“一个样本”是什么

用 `L0_TwoRobotPlaceMugRack-v1` 的某个机器人时刻 `k` 作为例子。它通过 [任务映射](../The-Imitator-Game/examples/baselines/lerobot_dataset/task_mapping.json#L752) 对应 `human_H57`。

[HumanSimPairedDataset.__getitem__()](../The-Imitator-Game/examples/baselines/lerobot_dataset/lerobot_paired_dataset.py#L434) 的调用顺序是：

```text
paired_dataset[i]
  → actual_sim_idx = valid_indices[i]
  → sim_dataset[actual_sim_idx]
  → sim_task_id → human_task_id → human_repo_id
  → human_dataset._get_target_item(human_repo_id)
  → 组合 robot_obs、robot_actions、human_video 等字段
```

配对依据是**同一任务的映射关系**。人类示范由 human loader 选择 episode、采样整段视频中的帧；代码没有把人类第 k 帧与机器人第 k 帧逐时刻对齐。

## 2. 底层 sim loader：当前观测与未来动作

源码：[LeRobotSimDataset.__getitem__()](../The-Imitator-Game/examples/baselines/lerobot_dataset/lerobot_sim_dataset.py#L356)。

`state_type=qpos` 选择两个原始字段：

```text
observation.qpos_gripper_states   单时刻 (18,)
action.qpos_gripper_actions      单动作 (16,)
```

18 维 state 的排列是左臂 `7 个臂关节 + 2 个手指关节`，再接右臂同样的 9 维。
16 维 action 的排列是左臂 `7 个臂关节命令 + 1 个夹爪命令`，再接右臂的 8 维。
依据是 [h5_to_lerobot.py 的字段命名](../The-Imitator-Game/examples/baselines/lerobot_dataset/h5_to_lerobot.py#L73)。因此 state 和 action 的维度本来就不同。

未来动作读取通过 [delta_timestamps](../The-Imitator-Game/examples/baselines/lerobot_dataset/lerobot_sim_dataset.py#L185) 实现：

```text
offsets = [0/30, 1/30, ..., 23/30]
action_sequence = [a_k, a_(k+1), ..., a_(k+23)]   → (24,16)
```

从当前时刻开始，包含 `a_k`。底层 [LeRobotDataset._get_query_indices()](../The-Imitator-Game/examples/baselines/lerobot_dataset/lerobot_dataset.py#L972) 会将超出 episode 边界的索引截到边界，重复边界动作并生成 `*_is_pad` 标记。当前 sim 包装层没有把该 mask 传给 ACT，后面的 L1 loss 也没有用它屏蔽尾部重复动作。

机器人图像经过 resize、训练增强、除以 255 和 CHW 排列；state/action 使用本数据集的统计量。最终 `sim_sample` 为：

| 字段 | shape | 含义 |
| --- | --- | --- |
| `states` | `(O,18) = (1,18)` | 归一化后的观测时间窗 |
| `view_1` | `(O,3,224,224)` | 机器人 zed2i RGB，float32，约 `[0,1]` |
| `actions` | `(24,16)` | 从当前时刻开始的动作标签 |
| `dataset_idx` | `()` | int64 标量，指向该 dataset 的统计量 |
| `repo_id` | 字符串 | 此例为 `L0_TwoRobotPlaceMugRack-v1` |

默认 `bounds_q99` 归一化来自 [ActionNormalizer](../The-Imitator-Game/examples/baselines/lerobot_dataset/normalizer.py#L52)：

```text
x_norm = clamp(2 * (x - q01) / (q99 - q01) - 1, -1, 1)
```

零宽度区间的分母替换为 1；缺少对应统计量时原样返回。shape 正确并不能证明所用统计量正确。

## 3. Human loader：取整段任务的示范信息

源码：[HumanVideoDataset._get_target_item()](../The-Imitator-Game/examples/baselines/lerobot_dataset/lerobot_human_dataset.py#L861)。

它先从 `human_H57` 的可用 episode 中随机挑一个，再用 `uniform_jitter` 等策略采样 `T=10` 帧。实际文件位置通过 LeRobot 的视频 chunk/file 元数据和 episode 起始时间定位。

```text
选中的视频帧              (10,H_original,W_original,3), uint8
resize / 颜色增强 / 转换   (10,3,224,224), float32
permute 后返回 video      (10,224,224,3), float32，[0,1]
```

依据：[human 图像变换](../The-Imitator-Game/examples/baselines/lerobot_dataset/lerobot_human_dataset.py#L625)、[返回 video 的排列](../The-Imitator-Game/examples/baselines/lerobot_dataset/lerobot_human_dataset.py#L1038)。当前多路 human camera 分支最后仍选 `all_view_tensors[0]`；本文使用一台相机。

## 4. 配对样本与 collate 后的 batch

配对层将 sim 的 `view_1` 从 `(O,C,H,W)` 换成 `(O,H,W,C)`；`robot_first_frame_obs` 则保持 CHW。

| 字段 | 一条配对样本 | collate 后 |
| --- | --- | --- |
| `human_video` | `(10,224,224,3)` | `(B,10,224,224,3)` |
| `robot_obs["states"]` | `(1,18)` | `(B,1,18)` |
| `robot_obs["view_1"]` | `(1,224,224,3)` | `(B,1,224,224,3)` |
| `robot_actions` | `(24,16)` | `(B,24,16)` |
| `robot_first_frame_obs["states"]` | `(1,18)` | `(B,1,18)` |
| `robot_first_frame_obs["view_1"]` | `(1,3,224,224)` | `(B,1,3,224,224)` |
| `dataset_idx` | `()` | `(B,)` |
| `human_repo_id` | 字符串 | 长度 B 的字符串列表 |

源码：[配对层第 443 行](../The-Imitator-Game/examples/baselines/lerobot_dataset/lerobot_paired_dataset.py#L443)、[collate 第 771 行](../The-Imitator-Game/examples/baselines/lerobot_dataset/lerobot_paired_dataset.py#L771)。

“first frame”在此处是 `sim_sample` **观测窗口的第一帧**。`O=1` 时就是当前观测，不能仅凭名称将它理解成机器人整段 episode 的初始画面。

配对样本还带 `sample_id`、`human_task_id` 等字段，但 `video_only` 的 collate 并没有把它们全部保留。缓存路径实际使用 `human_repo_id`。

## 5. 进入 ACTAgent：统一机器人观测的 shape

调用：[ACTAgent.compute_loss()](../The-Imitator-Game/examples/baselines/act/train_act_imitator.py#L420)。

`robot_obs`、动作标签、首帧观测和存在时的人类视频被移到 agent 的 device。
[观测预处理](../The-Imitator-Game/examples/baselines/act/train_act_imitator.py#L374) 执行：

```text
states: (B,1,18)               → 取最后一个时刻 → (B,18)
view_1: (B,1,224,224,3)        → 取最后时刻、CHW、增加相机维
                               → (B,1,3,224,224)
```

传给 `DETRVAE` 的字典变成：

```python
obs_for_model = {
    "state": ...,  # (B,18)
    "rgb": ...,    # (B,M=1,3,224,224)
}
```

这里 RGB 的第二维是**相机维**；DataLoader 的 `view_1` 第二维是**时间维**。它们在本文都等于 1，但语义不同。

当前代码虽然定义了 `img_normalize = T.Normalize(ImageNet mean/std)`，上述预处理和 `compute_loss()` 没有调用它。机器人 RGB 沿此路径保持 loader 给出的 `[0,1]` 数值；人类 DINO 输入则由其自己的 processor 处理。

## 6. Human demo encoder：视频到 task_z

调用链：[ACTAgent._encode_task()](../The-Imitator-Game/examples/baselines/act/train_act_imitator.py#L345) → `FrozenVideoBackbone.encode()` → [forward / _encode_dino()](../The-Imitator-Game/examples/baselines/encoders/task_encoder/video_backbone.py#L530)。

### 6.1 视频与 DINO 主干

```text
human_video                         (B,10,224,224,3)
转 uint8，二次采样 F=10 帧          (B,10,224,224,3)
合并 batch × frame                  (B*10,224,224,3)
HF image processor                  (B*10,3,224,224)
DINOv2-L last_hidden_state           (B*10,257,1024)
```

本机缓存的 DINO processor 会 resize 到短边 256、中心裁剪到 224，并执行自己的归一化。224×224 输入、patch size 14，产生 `16×16=256` 个 patch token，加 1 个 CLS token，共 257 个。此处使用的是不含 register tokens 的 `dinov2_vitl14`。

### 6.2 帧特征汇总与 Adapter

```text
每帧 CLS                  (B,10,1024)
沿时间维平均 raw_cls      (B,1024)
每帧 patches              (B,10,256,1024)
拼接所有帧 patches        (B,2560,1024)
按步长取最多 32 个        (B,32,1024) = raw_seq

cls_adapter: Linear + LN  (B,1024)    → (B,256)    = z
seq_adapter + PE          (B,32,1024) → (B,32,256) = z_seq
ACTAgent.ftask_norm       (B,256)     → (B,256)    = task_z
```

`task_z` 是当前 ACT 实际使用的人类示范条件；`z_seq` 虽然生成，但没有传入 `DETRVAE`。
`_encode_task()` 的签名包含 `robot_first_frame`、`human_desc`，此实现调用 encoder 时只传 `human_video` 和 `human_vl_ids`。本条路径不使用机器人首帧或语言来联合编码任务。

### 6.3 默认冻结训练中的缓存路径

正常训练会将 DataLoader 传给 ACTAgent。`lora_rank=0` 且使用默认缓存目录时，在训练循环前运行 [setup_frozen_backbone_cache()](../The-Imitator-Game/examples/baselines/encoders/task_encoder/frozen_backbone_cache.py#L564)：

```text
每个唯一 human_repo_id 预计算一次
  → 从该 repo 选一个 human episode / 采样帧
  → raw_cls (1024,), raw_seq (32,1024)
  → 以 human_repo_id 为 key 存储

正式训练 batch
  → human_repo_id: list[str]，可能没有 human_video 字段
  → 缓存查询得到 (B,1024)、(B,32,1024)
  → 可训练 Adapter → task_z (B,256)
```

依据：[预计算单条样本](../The-Imitator-Game/examples/baselines/encoders/task_encoder/frozen_backbone_cache.py#L397)、[命中缓存后的 forward](../The-Imitator-Game/examples/baselines/encoders/task_encoder/frozen_backbone_cache.py#L269)。

缓存 key 的粒度是 **human repo**，不包含机器人时刻或 human episode 编号。冷读 human loader 会随机挑 episode，但默认预计算后，同一 repo 的训练样本读取缓存中已存好的那一份特征，不能理解成每个训练 step 都重新随机编码一段人类视频。

## 7. ACT 的 CVAE latent：与 task_z 分开看

源码：[DETRVAE.forward()](../The-Imitator-Game/examples/baselines/act/act/detr_video/detr_vae.py#L68)。

训练时传入 `actions`，所以会调用 CVAE encoder：

```text
CLS embedding                 (B,1,256)
state Linear(18→256)           (B,1,256)
action Linear(16→256)          (B,24,256)
拼接并转为 sequence-first     (26,B,256)
CVAE encoder                  (26,B,256)
取 CLS                        (B,256)
latent_proj                   (B,64)
拆 mu、logvar                 各 (B,32)
重参数采样 latent_sample      (B,32)
latent_out_proj               (B,256) = latent_input
```

这里的 32 维 latent 表达训练动作序列的潜变量；人类示范产生的是前面的 256 维 `task_z`。二者分别投影为 Transformer 的条件 token。

推理时 `actions=None`，**不调用 CVAE encoder**，构造全零 `(B,32)` latent，再经过 `latent_out_proj`。线性层有 bias，所以“32 维 latent 为零”不代表投影后的条件 token 也全为零。

## 8. 机器人 RGB、条件 token 与 action queries

源码：[机器人 backbone](../The-Imitator-Game/examples/baselines/act/act/detr_video/backbone.py#L85)、[DETRVAE 的视觉分支](../The-Imitator-Game/examples/baselines/act/act/detr_video/detr_vae.py#L108)、[Transformer.forward()](../The-Imitator-Game/examples/baselines/act/act/detr_video/transformer.py#L50)。

```text
robot RGB                        (B,1,3,224,224)
取单路相机                       (B,3,224,224)
ResNet18 layer4                  (B,512,7,7)
1×1 Conv: 512→256                (B,256,7,7)
flatten / permute                (49,B,256)

video_feature_proj(task_z)       (B,256)
latent_input                     (B,256)
input_proj_robot_state(state)    (B,256)
三个条件 token + 49 视觉 token    (52,B,256)
ACT Transformer encoder memory  (52,B,256)

24 个 learned query embeddings  (24,256) → (24,B,256)
decoder 的每层输出               (24,B,256)
叠加层维并 transpose             (L_d=4,B,24,256)
```

Transformer 输入中的三个额外 token 顺序为 `[video_feature, latent_input, proprio_input]`。这里没有把整段 human patch token 序列拼进 Transformer；它使用的是投影后的一个视频条件 token。

**当前代码取哪一层输出：**`build_transformer()` 设置 `return_intermediate_dec=True`，`Transformer.forward()` 直接返回 tensor；`DETRVAE` 随后使用 `self.transformer(...)[0]`。因此此版本实际选取第一个 decoder 层的 `(B,24,256)` 输出，再执行 `Linear(256→16)`：

```text
a_hat = action_head(hs[0])   → (B,24,16)
```

这不是对模型意图的推测：合成样本 hook 实测返回 `(4,1,24,256)`，动作头输入与第一层输出相同、与最后一层不同。这里只记录行为，未修改实现。

## 9. 训练终点与执行起点

`compute_loss()` 返回三个标量：

```text
l1   = mean(abs(a_hat - robot_actions))
kl   = mean_batch(sum_latent(-0.5 * (1 + logvar - mu² - exp(logvar))))
loss = l1 + kl_weight * kl       # 默认 kl_weight=10
```

训练循环拿标量 loss 反向传播并更新参数；预测动作不会在这一批训练中直接送入模拟器。

训练保存 checkpoint 后，评估端通过 `prepare_for_eval()` 缓存 `task_z`，再用 `get_action(obs)` 输出 `(N,24,16)`；其中 `N` 是并行仿真环境数。动作块如何变成机器人实际执行的当步动作，继续看 [03_EVALUATION_TO_ROBOT.md](03_EVALUATION_TO_ROBOT.md)。

## 10. 动态核查记录怎么读

[shape_trace.json](shape_trace.json) 中 `paired_sample`、`collated_batch`、`training`、`inference` 分别对应上面的边界。实际核查调用了仓库原有配对、collate 和 ACT 方法，未修改这些函数。

底层 human/sim 数据、归一化统计是合成输入；没有加载真实 IG-10K episode，没有训练优化器更新，也没有执行机器人。动态核查关闭了特征缓存，使用现有预训练权重观察完整编码过程；默认缓存路径由源码核对。核查的 dtype 也不等同于训练循环开启 BF16 autocast 后每个中间量的 dtype。
