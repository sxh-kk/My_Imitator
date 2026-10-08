# PlaceMugRack：充分训练与评估方案

日期：2026-10-06。状态：**方案，尚未启动本轮训练，也未修改核心代码。**

后续更新：下一轮执行以 [06_4090_EXPERIMENT_DESIGN.md](06_4090_EXPERIMENT_DESIGN.md) 为准，扩展为固定训练预算的 10/50 条示范对照、三个训练 seed 和实测单卡调度。本篇保留为前一版设计记录。

目标是得到可重复评估的单任务 ACT baseline，判断当前实现能否学会 PlaceMugRack，并形成有证据的失败分析。100 epoch 是首轮训练预算上限，不代表已经保证收敛或保证某个成功率。

## 1. 当前起点与固定范围

已完成的 smoke run 见 [04_ACT_SMOKE_RUN.md](04_ACT_SMOKE_RUN.md)：2 条机器人轨迹、batch 8、3 epoch、171 次更新；checkpoint 回读通过，2 次仿真评估成功率 0/2。

本次只做 `L0_TwoRobotPlaceMugRack-v1`，保持已验证的官方 ACT 路径：

- 冻结 DINOv2-L，人类视频采样 10 帧；ResNet18 编码机器人 RGB。
- 单路 zed2i、224×224，state 18 维、action 16 维，obs horizon 1、预测 24 步。
- ACT hidden 256、encoder 2 层、decoder 4 层、feedforward 512、8 heads。
- 保留官方 Adapter、KL 权重、数据增强、动作归一化、动作聚合与控制器实现。
- 独立评估使用 `physx_cpu`、`rt-fast`、`pd_joint_pos`，每 episode 上限 500 步。

这是单任务训练诊断配置。官方 ACT 文档另有多任务训练及较大架构配置，因此本轮结果不能直接当作论文指标复现。[官方 ACT 文档](https://github.com/imitator-game/The-Imitator-Game/blob/d6d16ec511bc389e0a207692730c137bc022ef14/examples/baselines/act/README.md)

## 2. 数据安排

### 机器人数据

正式训练使用本地已有的 **全部 50 条 L0 轨迹，episode 0–49，共 11,228 帧**，配置范围为 `0:50`。无需追加下载。

本轮不划分离线机器人验证集：这 50 条全部属于训练数据，发布的 `meta/stats.json` 也固定作为训练/评估共同的归一化依据。训练样本上的动作误差只用于优化诊断，不能报告为验证误差。模型选择和最终评价使用独立的在线环境 reset seed。

如果以后需要离线 train/validation split，必须按 episode 划分，并另行处理只从训练部分计算统计量的问题。

### 人类示范

- 训练仍固定 `human_H57` episode **0**。
- 主开发评估与主最终测试固定 episode **1**，使用同一份记录好的采样帧。
- 主结果完成后，可分别用 episode **2、3** 各做 20 次附加评估，观察同一任务更换示范后的变化；单独报告，不混入主成功率。

当前官方冻结编码器缓存以 human repo/task 为 key，一个任务只缓存一份 raw feature。把 human 配置从 `0:1` 改成 `0:50` 不等于训练时利用了 50 份独立示范。本轮保持这一实现，每个训练 run 使用独立缓存目录，并记录其 hash，避免误用 smoke 或其他配置的旧缓存。

单任务成功本身不能证明模型具备从视频辨别不同任务的能力；本阶段先验证动作学习与闭环控制。

## 3. 执行顺序

### A. 一次吞吐标定

在正式训练前做一次短测，保留 batch 64 为首选配置：

- 比较 batch 32 / 64 / 128，先固定其余训练参数。
- 主配置候选分别检查 `num_dataload_workers=0` 与 `4`；有明确解码瓶颈时再试 `8`。
- 使用真实全量 L0 loader；每组先预热，再统计稳定阶段的 samples/s、step 时间、数据等待时间和显存峰值。
- 选稳定且吞吐合理的组合，并为串行评估子进程保留显存余量。显存占满不是配置选择标准。
- 短测参数不作为正式 checkpoint 起点。正式 run 从明确的初始化重新开始。

默认方案以下按 **batch 64、workers 4** 计算。若标定后更改 batch，记录真实 optimizer step 数；同样 100 epoch 的数据曝光量相同，但参数更新次数会变化。

### B. 小数据充分拟合诊断

先用已有 2 条轨迹做一个短而充分的诊断：

- 453 帧、batch 8、**50 epoch = 2,850 次更新**。
- seed 1，warmup 1 epoch；ACT 重新初始化，视觉主干仍使用现有预训练权重。
- 分别使用 human episode 0 与 1 各评估 10 次，采用相同的一组环境 seed，区分训练视频条件与更换示范的影响。
- 检查训练 L1 / KL，同时在**不输入真实 action、使用部署推理路径**时检查预测动作。

这一阶段回答“代码是否能够充分拟合少量轨迹，推理是否输出有结构的动作”。两条轨迹在随机场景下仍然 0 成功，不足以单独断言代码有 bug；若训练误差明显下降、动作与抓取阶段有进步，继续全量训练。

若出现 NaN、关节/夹爪语义异常，或训练误差下降而部署预测完全失效，则先核对归一化、动作语义和 CVAE 训练/推理路径。不要直接复制三份长训练。

### C. 全量 L0 主训练，先 seed 1

推荐配置：

```text
sim episodes              0:50（11,228 帧）
human episodes            0:1
batch size                64
epochs                    100
batches per epoch         ceil(11,228 / 64) = 176
optimizer updates         17,600
seed                      1
optimizer                 AdamW，weight_decay=1e-4
lr                        1e-4
lr_backbone               1e-5
warmup_epochs             5
lr scheduler              cosine
kl_weight                 10
precision                 官方 BF16 autocast
num_dataload_workers       4（经短测确认）
training mode             epoch
in-training evaluator     关闭；统一调用独立评估入口
save_epoch_freq           10
log_freq                  20
```

从与 smoke 相同的预训练视觉主干初始化，重新训练 ACT 和可训练 Adapter。**不直接 resume 已结束的 3-epoch smoke checkpoint**：它的 optimizer / scheduler 已走完原来的预算，无法通过只改 total_epochs 来构成清晰的对照实验。

当前源码以 0 起始 epoch 命名 checkpoint：`epoch_10.pt` 实际已经完成 11 个 epoch。按现有保存逻辑，重点评估完成 **11、21、41、61、81、100** 个 epoch 的 checkpoint。以 checkpoint 的 iteration 和每 epoch batch 数确认进度；不把文件名误当已完成的 epoch 数。

每个指定 checkpoint 保存后，由外层包装脚本串行调用独立评估进程，训练等待该次评估结束再继续。算法和 loss 不变；短测需要确认此时训练模型保留在显存内与评估进程共存的内存预算。

### D. 固定配置后重复训练

seed 1 出现可重复的任务成功，且训练/评估行为可信后，再按固定配置依次运行 seed **2、3**。三个训练 seed 使用相同的数据范围、模型规模与开发评估协议，各自选择开发集最优 checkpoint。

若 seed 1 全程无有效进步，优先定位失败并与作者 checkpoint 对齐，而不是立即重复同一失败三次。

## 4. 评估协议

### 开发评估：用于挑选 checkpoint

- 固定环境 reset seed **1000–1019**，共 **20 次**。
- 固定 human episode 1 及其采样帧；不同 checkpoint 不重新随机选择人类条件。
- 每个 checkpoint 在同一批 seed 上运行，保存每个 episode 的结果。
- 主指标：`success_once`；同时记录 `success_at_end`、return、episode_len 和失败阶段。
- checkpoint 选择规则：先比较 success_once，再比较 success_at_end；仍相同时选择训练步数较少者。

### 最终测试：不参与调参和模型选择

- 配置与 checkpoint 冻结后，使用环境 seed **2000–2099**，共 **100 次 / 训练 seed**。
- 每个训练 seed 都评估其开发集选出的 checkpoint；完成三个 seed 时总计 300 次测试。
- 报告每个 seed 的成功次数/总次数，以及跨 seed 的成功率均值和标准差；保留失败阶段分布与代表视频。
- 如果查看最终测试后继续调整训练或挑选 checkpoint，必须将这轮结果归为开发结果，并另设新的最终测试 seed 集。

开发和最终评估均使用已验证的独立 `eval_act_imitator.py` 路径：light temporal aggregation、窗口 4、每 4 步查询策略。训练入口内置 evaluator 的默认聚合方式不同，本轮不混用两套结果。

需要在外层显式设置、记录每个**实际 rollout**的 reset seed。只设置 Python/NumPy/Torch seed 不足以代替这个操作；当前 vector env 自动 reset 与 evaluator 显式 reset 并存，调度脚本必须核对实际开始 rollout 的 seed。每次评估使用独立输出目录，防止官方的已有结果跳过机制混用不同 checkpoint。

辅助记录优先保留失败发生在哪一步：靠近杯子、抓取、抬升、对齐杯架、挂放/释放。判断依据是环境 info 与录像；loss 和 dense reward 用于解释过程，不代替成功率。

## 5. 停止与继续条件

- **数值或执行异常**：发现非有限 loss/action、加载的可训练参数不一致、环境执行异常时立即停止该 run，完成定位后再继续。
- **早期无进展**：完成 21 和 41 epoch 的开发评估均为 0/20，且抓取/抬升等阶段也无改善时，停止该 run 做诊断。
- **阶段有进展**：即使还没有完整成功，只要 L1、部署动作和抓取/放置阶段提供一致的改善证据，可继续到 100 epoch 的预算上限。
- **达到预算**：100 epoch 后保存结果与失败报告；不自动追加数百 epoch。下一轮预算由学习曲线、在线结果和作者 baseline 对齐情况决定。

首轮交付不预设“必须达到 80%”等成功率。希望获得可重复成功的模型；如无法达到，也应交付经过检查的失败证据。

## 6. 外层脚本需要适配什么

实施时在新的 `experiments/act_placemugrack_full/` 中准备配置与调度脚本，保留原 smoke 材料：

1. 参数化 checkpoint 检查，去掉旧 `evaluate_checked.py` 中固定 `171`、`1000 steps`、`2 episodes` 的断言，改为核对本轮配置与 checkpoint metadata。
2. 保留可训练参数逐 tensor 回读核验。冻结 DINO 按固定权重 revision 检查；考虑缓存复用时 checkpoint 可能没有实例化并保存 DINO 参数的情况。
3. 为训练 checkpoint 增加外层保存回调和串行评估调度；不改 ACT、dataset、encoder 或 simulator 核心代码。
4. 显式调度开发/测试 seed；固定评估视频采样；记录 checkpoint hash、配置、数据和缓存来源。
5. 记录实际 epoch、iteration、学习率、L1/KL、运行时间、显存与吞吐。
6. 中断恢复须核对训练进度：当前中间 checkpoint 的 epoch 是正在完成的 0 起始索引，直接 resume 会重做该 epoch。由外层恢复逻辑明确下一 epoch 与 scheduler 状态，避免重复训练或误计预算。

不要直接把旧 `evaluate.sh` 加上 `--num-episodes 100` 就当成正式评估流程；其检查器仍针对 smoke 设计。

## 7. 4090 与存储安排

本机驱动报告约 48 GiB 显存，当前任务已有全部所需 L0 数据和资产。单卡顺序运行“训练 → 指定 checkpoint 评估 → 继续训练”，先获得可解释的结果，再考虑并发。

- 沿用 BF16 和冻结特征缓存；先测 loader 吞吐，减少 GPU 等待视频解码的时间。
- 主训练单 seed 处理约 112 万帧次；实际耗时用短测得到的稳定 step 时间乘以 17,600，再加初始化、保存和评估时间估算。
- 当前 smoke 实测两个 500-step episode 的 rollout 约 20 秒。沿该速度，20 次评估约 3–4 分钟，100 次约 17 分钟，仅作排期参考；需另加初始化和 IO，实际物理接触复杂度也会影响时间。
- 每份完整 checkpoint 约 1.33 GiB。每 run 最终保留 best / last / final，重复文件可用链接避免复制；只清理本轮新建且已评估、不再需要的中间 checkpoint。
- 三个 run 的保留 checkpoint 按约 12 GiB 预算，再预留中间文件空间；当前 `/home` 约 97 GiB、根分区约 200 GiB 空闲，能够覆盖本轮。
- 结果、checkpoint 清单和固定配置保存在 workspace 的实验目录；只发布小型文档与结果文件，权重仍留在本机。

首次执行顺序：**吞吐标定 → 两轨迹拟合诊断 → 全量 seed 1 → 开发评估选择 → 视结果启动 seed 2/3 → 冻结配置后最终测试。**

最终交付包括：固定配置、训练曲线、所选 checkpoint 及 hash、逐 episode 成功率记录、失败阶段统计、代表视频与一页结论。作者 checkpoint 到位后，使用匹配其训练范围与评估配置的单独实验进行对齐，保留本轮结果作为自己的基线记录。
