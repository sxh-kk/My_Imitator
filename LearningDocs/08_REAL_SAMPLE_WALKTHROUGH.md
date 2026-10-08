# 真实训练样本：七个暂停点跟通 ACT

核对日期：2026-10-07。对应上游 commit：`d6d16ec511bc389e0a207692730c137bc022ef14`。

本教程使用本机已有的 PlaceMugRack 官方数据、A50 的训练归一化统计、human episode 0 原始特征缓存，以及 A50 seed 1 的开发集所选 checkpoint（step 5,400）。调试 batch 默认为 1；正式训练 batch 为 64。

外层脚本：[trace_real_act_sample.py](trace_real_act_sample.py)。它读取真实文件，执行一次带标签的训练前向与一次不带标签的推理前向，不执行 backward、optimizer step 或 simulator step。不会保存、改写 checkpoint 或特征缓存。上游核心源码保持原样。

脚本已经配有中文注释、变量含义和每站的“中文解释”。交互模式还会显示可以直接输入的 `p ...` 命令及对应中文说明。变量名仍与原仓库一致，便于在源码中搜索。

先记住这些词：

- `sample`：一条机器人时刻样本，带当前观测、未来动作标签和同任务人类示范。
- `batch / collate`：一批样本，以及把样本堆成张量的操作。
- `shape / dtype / device`：张量的形状、数值类型、所在设备。
- `raw / normalized`：归一化前的值，以及经过本数据集统计量缩放的值。
- `feature / token / embedding`：编码后的特征；网络处理的特征单元；向量表示。
- `Adapter / LayerNorm`：可训练的特征变换模块；对特征执行归一化的层。
- `task_z`：由人类视频生成的 256 维任务条件。
- `latent / mu / logvar`：CVAE 潜变量、其分布均值、其分布的对数方差；这与 `task_z` 不同。
- `action chunk / query`：连续多步动作块；每个时间位置对应的可训练查询向量。
- `forward / backward / optimizer step`：前向计算、反向计算梯度、按梯度更新参数。本脚本只执行前向。
- `hook`：附在现有模块上的观察回调，用来记录输入输出，不替换网络计算。

已有 [trace_act_shapes.py](trace_act_shapes.py) 使用合成的底层样本，本教程改用真实数据。先读 [01 仓库地图](01_REPO_MAP.md) 看文件关系；需要解释某个中间 shape 时，对照 [02 样本到 ACT](02_SAMPLE_TO_ACT.md)。

## 0. 启动方式

先在终端运行一次完整跟踪，确认本机依赖和数据可用：

```bash
source /home/zxc/Imitator/scripts/activate_imitator.sh
cd /home/zxc/Imitator
which python
python LearningDocs/trace_real_act_sample.py
```

`which python` 应指向 `/home/zxc/miniconda3/envs/imitator/bin/python`。初始化日志较多，等待出现 `STEP 1` 至 `STEP 7`；结束时应输出 `TRACE PASSED`。结果写入 [real_sample_trace.json](real_sample_trace.json)，再次运行会覆盖这个学习记录，不影响实验结果。

然后开启交互跟踪：

```bash
python LearningDocs/trace_real_act_sample.py --interactive
```

脚本在七个位置进入 `(Pdb)`。每个暂停点先执行下文的 `p ...` 检查变量，再用 `c` 到下一个暂停点。代码块不含 `(Pdb)` 前缀，可以逐行复制。

常用命令：

- `p expression`：打印表达式；`pp expression`：展开字典。
- `n`：执行当前行；`s`：进入当前行调用的函数。
- `r`：运行到当前函数返回；`c`：继续到下一个断点。
- `w`：看调用栈；`u` / `d`：切换到调用方 / 被调用方栈帧。
- `b 文件路径:行号`：设置源码断点；`b`：列出现有断点。
- `cl 断点编号`：删除手动设置的断点；`q`：退出跟踪。

首次只用 `p` 和 `c` 完成七站，第二次再进入源码。脚本使用 `num_workers=0`，Dataset 和 collate 都在当前进程，方便断点跟踪。

## 1. 暂停点 STEP 1：任务如何配对

此时 dataset 已构建，尚未读取选中的样本：

```python
p len(paired)
p paired.paired_tasks
p paired.task_mapper.get_human_task_from_sim(TASK)
p paired.valid_indices[:5]
p args.human_dataset_file
p args.sim_dataset_file
```

预期：

```text
11228
[('human_H57', 'L0_TwoRobotPlaceMugRack-v1')]
'human_H57'
[0, 1, 2, 3, 4]
```

接着打开 [lerobot_paired_dataset.py](../The-Imitator-Game/examples/baselines/lerobot_dataset/lerobot_paired_dataset.py:434) 的 `HumanSimPairedDataset.__getitem__()`，按这个顺序阅读：

```text
idx -> valid_indices[idx] -> sim_dataset[actual_sim_idx]
    -> sim_task_id -> task_mapper -> human_task_id
    -> human_repo_id -> human_dataset._get_target_item(human_repo_id)
```

任务映射见 [task_mapping.json](../The-Imitator-Game/examples/baselines/lerobot_dataset/task_mapping.json:752)。本机训练的人类配置只允许 episode 0，机器人配置允许 episode 0–49。机器人 episode 7 也会使用 `human_H57` 的人类示范；配对依据不是 episode 编号相等，也不是两段视频逐帧同步。

用 `c` 到 STEP 2。

## 2. 暂停点 STEP 2：当前时刻与未来动作标签

```python
p sample['sample_id']
p int(row['episode_index']), int(row['frame_index']), int(row['index'])
p sample['sim_task_id'], sample['human_repo_id']
p raw_state.shape, raw_action.shape, raw_actions.shape
p sample['robot_obs']['states'].shape
p sample['robot_actions'].shape
p sample['human_video'].shape
p future_indices
p padding['action.qpos_gripper_actions_is_pad'].tolist()
```

默认样本为机器人 episode 0 / frame 0。预期：

```text
raw_state                  (18,)
raw_action                 (16,)
raw_actions                (24,16)
sample robot state         (1,18)
sample robot_actions       (24,16)
sample human_video         (10,224,224,3)
future_indices             [0,1,...,23]
padding                    全为 False
```

### 18 维 state 与 16 维 action

字段语义依据 [h5_to_lerobot.py](../The-Imitator-Game/examples/baselines/lerobot_dataset/h5_to_lerobot.py:73)。用原始值查看各部分：

```python
p raw_state[:7]      # 左臂 7 个关节位置
p raw_state[7:9]     # 左夹爪两个手指关节位置
p raw_state[9:16]    # 右臂 7 个关节位置
p raw_state[16:18]   # 右夹爪两个手指关节位置
p raw_action[:7]    # 左臂 7 个关节控制命令
p raw_action[7]     # 左夹爪一个控制命令
p raw_action[8:15]  # 右臂 7 个关节控制命令
p raw_action[15]    # 右夹爪一个控制命令
```

state 记录两个手指的位置；夹爪控制只需要一个命令，通过 mimic controller 驱动两个手指，所以每臂 state 为 9 维、action 为 8 维。当前 `pd_joint_pos` 路径中，臂动作是关节位置目标，不是末端位姿或力矩；夹爪命令还经过自己的 controller 缩放，不能把整条 action 当作同一种物理单位。

`raw_state` / `raw_actions` 来自原始 parquet；`sample` 内对应数值已归一化。查看：

```python
p sample['robot_obs']['states'][0]
p sample['robot_actions'][0]
```

归一化在 [lerobot_sim_dataset.py](../The-Imitator-Game/examples/baselines/lerobot_dataset/lerobot_sim_dataset.py:534)，默认公式为 `clamp(2*(x-q01)/(q99-q01)-1, -1, 1)`，常量维度有单独的分母处理。脚本已经核对这条 state 和 24 步动作与原始值归一化后的结果完全相等。

### 为什么有 24 步标签

打开 [LeRobotSimDataset 的动作时间窗](../The-Imitator-Game/examples/baselines/lerobot_dataset/lerobot_sim_dataset.py:185)：`horizon=24`，采样偏移为 `0/30,...,23/30`，得到从当前时刻开始的 `[a_k,...,a_(k+23)]`。这是离线训练标签，不是读取未来图像，也不是在线推理能够看到未来动作。

底层 [LeRobotDataset._get_query_indices](../The-Imitator-Game/examples/baselines/lerobot_dataset/lerobot_dataset.py:972) 把超出本 episode 的索引截到末尾，并产生 padding mask。当前 sim/ACT 路径未将这个 mask 用于 L1 屏蔽。

用 `c` 到 STEP 3。

## 3. 暂停点 STEP 3：collate 如何增加 batch 维

```python
p list(batch_before_cache)
p batch_before_cache['robot_obs']['states'].shape
p batch_before_cache['robot_obs']['view_1'].shape
p batch_before_cache['robot_actions'].shape
p batch_before_cache['human_video'].shape
p batch_before_cache['human_repo_id']
```

默认 `B=1`：

```text
states          (1,1,18)
view_1          (1,1,224,224,3)
robot_actions   (1,24,16)
human_video     (1,10,224,224,3)
human_repo_id   ['human_H57']
```

打开 [collate_paired_batch](../The-Imitator-Game/examples/baselines/lerobot_dataset/lerobot_paired_dataset.py:771)，找到几处 `torch.stack()`。它只负责把各样本堆叠，不负责视频编码或模型预测。`view_1` 的第二维此时是机器人观测时间窗；下一站的模型 RGB 第二维则是相机数。

直接读取 `sample`、再通过 DataLoader 读取同一索引，图像增强可能不同，state 与动作标签保持一致。当前未启用缓存的 human loader 也会重新采样视频帧，因此不要把本暂停点的视频当作正式训练中固定缓存对应的那一份帧。

用 `c` 到 STEP 4。

## 4. 暂停点 STEP 4：正式训练的缓存路径

ACTAgent 已按正式训练方式构建并加载所选 checkpoint。其构造函数启用了跳过 human 视频解码：

```python
p checkpoint_step
p type(agent.task_encoder).__name__
p paired.skip_human_video
p 'human_video' in batch
p batch['human_repo_id']
p cached_raw['cls'].shape, cached_raw['seq'].shape
p obs_for_model['state'].shape, obs_for_model['rgb'].shape
```

预期：

```text
5400
'CachedFrozenBackbone'
True
False
['human_H57']
(1,1024), (1,32,1024)
(1,18), (1,1,3,224,224)
```

打开 [CachedFrozenBackbone.forward](../The-Imitator-Game/examples/baselines/encoders/task_encoder/frozen_backbone_cache.py:269)，跟踪：

```text
human_repo_id -> 缓存 get_batch -> raw_cls/raw_seq
             -> 可训练 cls_adapter/seq_adapter -> z/z_seq
```

冻结 DINO 的 raw features 已准备好，因此正式训练 step 没有再次执行 DINO。Adapter 仍是可训练参数；查：

```python
p backbone.cls_adapter[0].weight.requires_grad
```

应为 `True`。缓存 key 是 human repo，本轮所有 PlaceMugRack 样本共享 episode 0 的固定特征。视频并未按机器人时刻重新编码。

机器人输入预处理见 [ACTAgent._preprocess_obs_for_model](../The-Imitator-Game/examples/baselines/act/train_act_imitator.py:374)：取观测窗最后时刻，将图像转成 CHW，并增加相机维。

用 `c` 到 STEP 5。

## 5. 暂停点 STEP 5：亲眼查看 DINO 原始特征

为展示缓存的来源，脚本加载本轮训练实际固定的 human_ep0.pt（本机文件：`experiments/act_placemugrack_study/prepared/human_ep0.pt`），重新调用一次 DINO。视频文件和帧索引来源见 [prepared/manifest.json](../experiments/act_placemugrack_study/prepared/manifest.json) 的 `human.0`。

```python
p fixed_video.shape
p report['hooks']['dino_precompute']['dino'][0]['kwargs']['pixel_values']['shape']
p report['hooks']['dino_precompute']['dino'][0]['output']['shape']
p raw_cls.shape, raw_seq.shape
p cls_error, seq_error
```

预期：

```text
fixed_video       (1,10,224,224,3)
DINO pixel_values [10,3,224,224]
DINO output       [10,257,1024]
raw_cls           (1,1024)
raw_seq           (1,32,1024)
cls/seq error     0.0, 0.0（本机此次核查）
```

打开 [FrozenVideoBackbone._encode_dino](../The-Imitator-Game/examples/baselines/encoders/task_encoder/video_backbone.py:530)，观察它如何将 `batch × frame` 合并、取每帧 CLS 并沿时间平均，以及从全部 patch tokens 中选取最多 32 个 token。

257 是 `1 个 CLS + 16×16 个 patch token`；1024 是 DINOv2-L 的特征宽度。这里的 DINO 只编码人类视频，机器人 RGB 后面由 ResNet18 编码。

这一步额外重算 DINO 是学习检查。正式训练热路径仍直接用 STEP 4 的缓存。

用 `c` 到 STEP 6。

## 6. 暂停点 STEP 6：task_z、CVAE 与动作块

脚本已经执行一次原版 `agent.compute_loss(batch)`。这里保留 autograd 图，尚未 backward 或更新参数。训练前向使用 BF16。

```python
p task_z.shape
p pred_train.shape
p mu.shape, logvar.shape
p {k:float(v.detach()) for k,v in losses.items()}
p losses['loss'].requires_grad
p report['hooks']['training']['cvae_encoder'][0]['input'][0]['shape']
p report['hooks']['training']['act_encoder'][0]['output']['shape']
p report['hooks']['training']['transformer'][0]['output']['shape']
p 'dino' in report['hooks']['training']
```

预期：

```text
task_z                 (1,256)
pred_train             (1,24,16)
mu / logvar            (1,32) / (1,32)
loss.requires_grad     True
CVAE input             [26,1,256]
ACT encoder output     [52,1,256]
ACT decoder stack      [4,1,24,256]
DINO called here       False
```

### DINO 特征在哪里进入 ACT

依次阅读这三个位置：

1. [ACTAgent._encode_task](../The-Imitator-Game/examples/baselines/act/train_act_imitator.py:345)：encoder 的 `z` 经过 `ftask_norm`，得到 `task_z`。当前没有使用 `z_seq`。
2. [ACTAgent.compute_loss](../The-Imitator-Game/examples/baselines/act/train_act_imitator.py:465)：通过 `self.model(..., video_feature=task_z)` 把它传给策略。
3. [DETRVAE.forward](../The-Imitator-Game/examples/baselines/act/act/detr_video/detr_vae.py:68)：`video_feature_proj` 将它投影成一个视频条件 token，送入 ACT Transformer。

条件链为：

```text
raw_cls: (B,1024)
  -> cls_adapter: (B,256)
  -> ftask_norm: task_z (B,256)
  -> video_feature_proj: (B,256)
  -> ACT encoder 的一个条件 token
```

ACT encoder 还有 latent token、state token，以及 ResNet18 产生的 `7×7=49` 个机器人视觉 token，共 `3+49=52` 个。人类视频不是直接拼成十帧机器人观测，也没有把那 32 个 `z_seq` token 传入当前 ACT。

### 训练 CVAE 看到了什么

CVAE encoder 把一个 CLS、一个 state 和 24 个真实动作标签嵌入拼接，所以序列长度为 `1+1+24=26`。它输出 `mu/logvar`，采样 32 维 latent，再投影到 256 维。这个 latent 与人类视频得到的 `task_z` 是两种不同的条件。

训练损失为 `L1(pred_train, robot_actions) + 10 × KL`。本脚本的单 batch loss 只用于理解执行链，不能代替训练曲线或闭环成功率。

### 为什么预测 B×24×16

```python
p agent.model.query_embed.weight.shape
p agent.model.action_head.in_features, agent.model.action_head.out_features
p report['first_decoder_layer_selected']
```

预期是 `(24,256)`、`(256,16)`、`True`。24 个 learned queries 分别预测块中的 24 个时间位置，每个输出 16 维双臂动作，batch 堆叠得到 `(B,24,16)`。

当前源码 Transformer 返回 `(4,B,24,256)`，`DETRVAE` 的 `[0]` 选择第一 decoder 层，再进入 action head。本教程按现有行为记录。模型返回完整块，后续 evaluator 才负责取当前时刻动作与聚合。

用 `c` 到 STEP 7。

## 7. 暂停点 STEP 7：推理为何不再调用 CVAE encoder

脚本用同一个离线机器人观测，执行 `prepare_for_eval()` 和 `get_action()`，用于比较模型接口；这里没有启动 simulator。

```python
p agent.training
p eval_task_z.shape
p prediction.shape
p latent_inference.shape
p torch.count_nonzero(latent_inference).item()
p 'cvae_encoder' in report['hooks']['inference']
p live['inference']['model']['kwargs']['actions']
p live['inference']['model']['output'][1]
```

默认 B=1 的预期：

```text
agent.training       False
eval_task_z          (1,256)
prediction           (1,24,16)
latent_inference     (1,32)
nonzero elements     0
CVAE encoder called  False
actions              None
mu/logvar            [None,None]
```

阅读 [ACTAgent.get_action](../The-Imitator-Game/examples/baselines/act/train_act_imitator.py:515) 和 [DETRVAE.forward](../The-Imitator-Game/examples/baselines/act/act/detr_video/detr_vae.py:68)。切换 CVAE 分支的判断是 `actions is not None`，不是单独根据 `agent.train()` / `agent.eval()`：

```text
训练：机器人当前观测 + task_z + 真实动作标签
      -> CVAE encoder -> mu/logvar -> 采样 latent -> 预测动作块

推理：机器人当前观测 + task_z，actions=None
      -> 跳过 CVAE encoder -> 零 latent -> 预测动作块
```

零的是投影前的 32 维 latent；`latent_out_proj` 有 bias，其 256 维输出不必为零。推理没有动作标签参与，所以训练前向的重建误差不能单独说明部署效果。

本教程在 `prepare_for_eval` 使用 episode 0 固定视频，方便与训练缓存对照；正式开发/最终测试使用 episode 1。不要把这里的一次离线推理当作正式测试复跑。视频在 prepare 阶段编码一次，随后 get_action 使用缓存的 task_z。

用 `c` 结束，保存 JSON 记录。

## 8. 第二遍：进入源码内部

在 STEP 1 的 `(Pdb)` 提示符下可以设置配对层断点：

```text
b /home/zxc/Imitator/The-Imitator-Game/examples/baselines/lerobot_dataset/lerobot_paired_dataset.py:434
c
```

到 `__getitem__` 后用 `n` 查看 `actual_sim_idx`、`sim_sample`、`human_task_id` 的产生，适时用 `p`；可用 `s` 进入 sim loader。下列位置同样可作为手动断点：

- `lerobot_sim_dataset.py:356`：当前机器人帧、图像和动作窗。
- `lerobot_human_dataset.py:861`：人类 episode 和视频采样。
- `lerobot_paired_dataset.py:771`：collate。
- `frozen_backbone_cache.py:269`：raw cache 命中和 Adapter。
- `video_backbone.py:530`：DINO 编码；重算阶段才会走到这里。
- `train_act_imitator.py:420`：训练 compute_loss。
- `detr_video/detr_vae.py:68`：训练/推理两次 forward。
- `train_act_imitator.py:515`：推理 get_action。

文件根目录是 `/home/zxc/Imitator/The-Imitator-Game/examples/baselines/`，对照 [01 的链接](01_REPO_MAP.md) 拼接完整路径。设置某个断点后先观察一次，再用 `b` 查编号、`cl 编号` 删除，避免反复停在后续 DataLoader 读取中。

## 9. 两个小练习

先改 batch 为 2，观察 B 维如何变化：

```bash
python LearningDocs/trace_real_act_sample.py --batch-size 2 --interactive \
  --out LearningDocs/real_sample_trace_b2.json
```

机器人 batch、训练 task_z、预测块都会变为 B=2；准备推理只输入一份固定人类视频，因此 `_cached_task_z` 是 `(1,256)`，get_action 将它 expand 成 `(2,256)`。

再查看整份 A50 数据的最后一条样本，观察 episode 尾部 padding：

```bash
python LearningDocs/trace_real_act_sample.py --sample-index 11227 --interactive \
  --out LearningDocs/real_sample_trace_last.json
```

在 STEP 2 查看 `future_indices` 和 `padding`。它们不会跨到下一条 episode；当前动作块后续位置重复本 episode 最后动作。

完成后，用自己的话回答：任务如何配对；缓存如何生成并被查询；18/16 维分别表示什么；24 个查询如何产生动作块；以及真实动作标签在哪个分支出现、推理时为何消失。
