# PlaceMugRack：单卡 4090 实验设计

日期：2026-10-06。执行完成：2026-10-07。状态：**六个训练 run、600 次开发评估和 600 次最终测试已完成；结果见 [07](07_PLACEMUGRACK_EXPERIMENT_RESULTS.md)。**

本方案作为下一轮执行依据，替代 [05 的单组训练排期](05_PLACEMUGRACK_TRAINING_PLAN.md)。只围绕本地训练与评估；保持 ACT、human encoder、loss、动作聚合和 simulator 核心实现不变。实验包装脚本放在 `experiments/act_placemugrack_study/`。

## 1. 目标与先后顺序

第一目标是让本地 ACT 在 `L0_TwoRobotPlaceMugRack-v1` 上经过充分优化后产生可重复的闭环成功，并解释失败发生在哪个阶段。随后回答：**在相同训练计算预算下，50 条机器人示范相对 10 条示范是否有稳定收益？**

执行顺序：吞吐标定 → 全量 50 条示范的 seed 1 试运行 → 固定实验配置 → 完成 10/50 条示范各 3 个训练 seed → 独立最终测试 → 失败分析。

18,000 次更新是首轮预算，不代表预先保证收敛。用学习曲线、闭环成功率和失败阶段判断训练是否充分；本轮不自动无限加训。

当前已完成的 [smoke run](04_ACT_SMOKE_RUN.md) 只有 2 条轨迹、171 次更新、成功率 0/2，说明流程能够执行，还不能判断任务能否学会。

## 2. 本机资源与利用原则

本机核对结果：RTX 4090，驱动报告 49,140 MiB 显存（约 48 GiB）；7950X、32 线程；内存约 187 GiB；`/home` 可用约 97 GiB，根分区约 200 GiB。使用已经配置好的 conda 环境 `imitator`。

所有必需的本任务数据和资产已在本地。优先提高**每小时完成的有效训练样本和评估 episode 数量**，同时记录 GPU 利用率、显存、CPU 和数据等待时间。

### 吞吐标定：先串行，再测并发

1. 用真实 50 条示范的 loader，在 workers=4 下比较 batch 32、64、128；每组预热 50 步、测量 200 步，初始化及首次特征缓存耗时单列。波动明显时将测量延长至 500 步。
2. 优先采用 batch 64，再比较 workers=0、4、8。记录 samples/s、step 时间中位数/p95、数据等待时间、显存峰值。保留已有 BF16、pin memory、冻结特征缓存；避免每个 step 强制 CUDA 同步造成测量失真。
3. 在候选配置上比较单训练与两个独立训练并发，统计**总 samples/s**及完成同样两份训练工作的总耗时。只有总吞吐至少增加 15%、显存峰值不超过约 40 GiB、结果检查正常时才启用双训练。
4. 比较评估 num_envs=1、2、4，保持 CPU physics、shader、控制器和每条 rollout 的 seed 一致。确认初始状态对应关系、计分、episode 数及逐环境动作分配正确，再选择吞吐最好的配置。物理数值差异须记录；同一正式对照统一评估配置。
5. 比较“训练后评估”与“一个训练进程＋一个评估任务重叠”。用同量训练步骤和 rollout 的总完成时间判断，达到至少 15% 改善且无资源异常才启用重叠。

默认运行一个训练任务；标定后最多同时调度两个顶层 GPU 任务，例如两个训练，或一个训练加一个评估。评估的多个环境还会产生渲染子进程，显存统计须覆盖全部进程。初始 OMP/MKL 线程数设为 4，loader 每个训练任务先用 4 workers，再按测量调整。

**全套正式实验统一 batch。** 默认 B=64、18,000 次更新；只有 B=32/128 实测确有优势时，才在所有正式训练开始前统一替换，并保持每个 run 的样本呈现总量 1,152,000 不变：B=32 对应 36,000 步，B=128 对应 9,000 步。warmup 和评估点按预算比例同步换算。标定产生的权重不进入正式结果。

## 3. 正式实验：两个数据条件、三个训练 seed

六个 run：`A50_s1`、`A50_s2`、`A50_s3`、`A10_s1`、`A10_s2`、`A10_s3`。训练 seed 分别为 1、2、3；两个数据条件按相同 seed 配对，并在模型初始化前明确重置 RNG。

- **A50**：episode 0–49，共 50 条、11,228 帧。
- **A10**：固定嵌套子集 `[5, 12, 14, 18, 31, 32, 33, 35, 43, 46]`，共 10 条、2,241 帧。由 `random.Random(20261006).sample(range(50), 10)` 后排序产生。
- 相同模型、batch、优化器、更新次数、学习率计划、视频条件及开发/测试协议。
- 每次遍历随机打乱，`drop_last=True`，耗尽 loader 后重新遍历，保证每次更新恰好 B 个样本。
- 不留离线机器人验证集：各组列出的轨迹全部用于训练。模型选择和最终测试依靠独立环境 reset seed。

默认每个 run 呈现 64 × 18,000 = **1,152,000 个样本**，A50 约相当于 102.6 次数据遍历，A10 约 514.1 次。六个 run 共 108,000 次更新、6,912,000 个样本呈现。有效遍历数按呈现量/总帧数计算，实际每轮末尾会丢弃不足一个 batch 的部分。

本轮控制训练步数与样本呈现量，避免用相同 epoch 使 A50 获得约五倍计算量。数据多样性和重复采样次数随条件变化，正是需要观察的因素。

### 归一化与人类示范

两个条件分别只用自己的训练轨迹计算 state/action 的逐维 q01、q99，使用同一种统计方法；保留官方 `bounds_q99` 变换。A10 不借用 A50 的统计量。派生统计文件独立保存，原始下载数据不改动；训练、部署、动作误差分析使用同一份对应统计量并核对 hash。

这是一项明确记录的实验预处理设置，与 smoke 使用发布版 stats 的做法不同。准备阶段须核验非退化维度、裁剪比例、反归一化以及 18 维 state/16 维 action 的顺序，不能把裁剪后的 roundtrip 误差当实现错误。

训练固定 `human_H57` episode 0，所有 run 共用同一份已记录采样帧和冻结 DINO raw feature；每个 run 使用独立缓存路径，内容 hash 一致。开发和最终测试固定 episode 1 及其采样帧，不随 checkpoint 或训练 seed 改变。

当前冻结缓存以 human repo/task 为 key，一项任务只有一份 raw feature；扩大 human episode 配置不会自动变成逐示范条件训练。本轮保留此行为。单任务成功只能支持动作学习与控制结论，不能证明模型能够辨别不同任务意图。

## 4. 固定训练配置

```text
task                        L0_TwoRobotPlaceMugRack-v1
robot observation           zed2i RGB 224×224, state_dim=18
action                      action_dim=16, pred_horizon=24, obs_horizon=1
robot image encoder         ResNet18
human encoder               frozen DINOv2-L, 10 frames, existing Adapter
ACT                         hidden=256, feedforward=512, heads=8
                            encoder_layers=2, decoder_layers=4
batch_size                  64 (default; freeze after calibration)
optimizer updates           18000
warmup steps                900
optimizer                   AdamW, weight_decay=1e-4
learning rate               other=1e-4, backbone-named params=1e-5
scheduler                   cosine, total_steps=18000
gradient clipping           backbone=0.1, other=1.0
KL weight                   10
precision                   existing BF16 autocast path
internal training evaluator disabled
```

ACT 和可训练 Adapter 重新初始化，视觉主干使用本地已有的同版本预训练权重。所有正式 run 不从 smoke 权重继续训练。

优化器分组按当前源码的参数名规则复用：名称包含 `backbone` 的可训练参数进入低学习率组，该组可能包括 task encoder 内 Adapter 参数，不能凭模块称呼重新分组。保留当前 decoder 输出索引行为，详见 [源码追踪](02_SAMPLE_TO_ACT.md)。

### 固定步数包装器是实施前置工作

当前官方 iteration 分支把 scheduler 总步数设为 `total_iters // 5`，warmup 固定为 1000，梯度裁剪也与已验证的 epoch 分支不同。**不能直接设置 `--total-iters 18000` 就认为执行了上面的协议。**

实施时在实验目录新增外层训练包装器，调用官方 dataset、ACTAgent 和 `compute_loss`，沿用 epoch 分支的更新过程，仅显式控制 loader 重启、满 batch、真实 optimizer step、scheduler、保存与日志。开始长训练前，对同一初始权重和同一输入 batch 验证损失、梯度处理、参数更新与原 epoch 路径一致；再核对第 900/18,000 步学习率计划。核心源码不改。

## 5. 试运行与扩展条件

先跑 `A50_s1`，在总预算 30% 和 50%（默认 5,400、9,000 步）做判断：

- 出现非有限 loss/action、参数回读不一致、归一化/动作维度异常、环境执行错误：立即停止该 run，定位后重新开始受影响的实验。
- 两次开发评估均为 0/20，且录像中的抓取、抬升等阶段没有改善：暂不扩展其余五个 run，先做两轨迹充分拟合诊断。
- 任务已有成功，或阶段行为持续改善：继续完成预算；配置固定后启动剩余 run。

两轨迹诊断沿用 episode 0/1、453 帧、B=8、50 epoch（保留末尾小 batch，共 2,850 次更新），单独记录。检查训练 L1/KL，也检查**不提供真实 action 的部署推理路径**；CVAE 带真实动作的训练损失下降不能单独证明部署预测有效。

试运行若没有引发配置/代码改动，可以作为正式 `A50_s1`。若做了实质调整，旧试运行仅归档为探索结果，六个正式 run 全部按最终协议重新开始。

正式矩阵冻结后，两组都跑完相同预算；不能依据成功率仅提前停止 A10。计算异常和资源中断另记，未完成的实验不当作零成功完整实验。

## 6. 评估与 checkpoint 选择

统一使用独立评估路径的设置：`physx_cpu`、`rt-fast`、`pd_joint_pos`、每 episode 500 步、light temporal aggregation 窗口 4、每 4 步查询策略。保留已验证的动作反归一化与左右臂拆分。训练内部 evaluator 默认聚合方式不同，结果不混用。

### 开发评估

- checkpoint 位于训练预算的 **10%、30%、50%、70%、100%**：默认 1,800 / 5,400 / 9,000 / 12,600 / 18,000 步。
- 每个 checkpoint 固定环境 seed **1000–1019**，20 个 episode。
- 首选最高 `success_once`，再比 `success_at_end`，仍相同时选训练步数较少者。
- 每个 run 共 100 个开发 episode；六个 run 共 **600 个**。

### 最终测试

- 六个 run 的配置及所选 checkpoint 全部冻结后，再打开最终测试。
- 每个 run 固定 seed **2000–2099**，100 个 episode；六个 run 共 **600 个**。
- 报告每个 run 的成功次数/100、三次训练的成功率均值及样本标准差；同时列出相同训练 seed 的 A50−A10 差值。
- 对照共享环境 seeds；仍需记录实际初始状态，以检查 seed 配对是否真正对应。100 次 rollout 不能替代多个独立训练 seed。

主实验合计 **1,200 个 episode、最多 600,000 个环境 step**，不含标定和诊断。每个 episode 都保存指标；代表视频在运行前按固定 seed 选取，失败分析需要时补录并标明补录。视频采集设置在两个条件间一致。

每次评估写入唯一目录，附 checkpoint SHA256、配置、stats hash、human 帧索引与环境 seed。只设置 Python/Torch 随机种子不够：必须显式安排实际 rollout 的 reset seed，检查 vector env 自动 reset 与外层显式 reset 的相互作用。

主指标为任务成功率，辅助记录最终成功、return、episode_len、策略查询数及失败阶段：接近 → 抓取 → 抬升 → 对齐 → 挂放/释放。辅助 DTW/TSS 不参与模型选择；如果为吞吐而关闭，其开关需全实验统一且在报告中说明。

旧 smoke 检查器硬编码了 171 次更新、2 个 episode 和单环境 tensor shape，实施时必须参数化；不能只给旧脚本增加 `--num-episodes 100`。

## 7. 单卡调度、耗时与空间

建议队列：标定 → A50_s1 试运行 → A10_s1 → A50_s2 → A10_s2 → A50_s3 → A10_s3。通过并发标定后，队列可同时执行两个已满足前置检查的任务。开发评估优先于下一轮扩展，避免尚未发现的错误复制到多个 run。

checkpoint 保存到临时文件，完成落盘和 hash 校验后原子改名并产生 ready 标记，评估进程只读取 ready 文件。每个 run 最多积压一个待评估 checkpoint；积压时暂停该 run 的进一步调度。保留 trial、正式 run、吞吐测试各自独立的目录。

恢复训练须同时保存 optimizer、scheduler、已完成更新数、采样顺序/游标和 RNG 状态，并验证恢复后的更新计数及学习率。当前官方中间 epoch checkpoint 的 resume 会重复该 epoch，不能直接沿用其计数。如果不能恢复一致的采样过程，应明确标记为非精确恢复；正式可比结果优先从确定起点重跑受影响的 run。

### 时间预算

设计时以完成全矩阵为目标，首轮排期预留 **24 小时窗口**；下面保留执行前的耗时估算。本轮实际使用 B64、workers 8、两个训练 run 的队列和 4 个评估环境，正式训练/开发评估/最终测试共 **2.11 小时**，详细实测见 [07](07_PLACEMUGRACK_EXPERIMENT_RESULTS.md)。实现、标定和测试后的报告/视频整理另计。

- 训练时间估算：`108000 × 实测秒/step`，再加初始化、缓存、保存、数据处理等开销；若更改 batch，同步使用新总步数。
- 现有 smoke 每个 500-step rollout 约 10 秒；按此粗算，1,200 次评估串行约 **3.3 小时**，尚未包含全部初始化和 IO。复杂接触及并发渲染会改变实际耗时。
- 举例：若 B64 稳态为 0.10 秒/step，纯训练约 3 小时；若为 0.30 秒/step，约 9 小时。这是排期示例，**不是本机大 batch 已测性能**。
- 实测超出窗口时保留队列和完整协议，优先交付全量单 seed 的训练/开发曲线；后续补齐矩阵，不能偷偷缩短某一组预算。

### 空间预算

- 本轮无新增大规模数据下载；派生统计文件和配置很小。
- 按已生成的完整 checkpoint 约 1.33 GiB 保守估计，六个 run 最终各保留 best/last/final，最多约 **24 GiB**；相同 checkpoint 可用链接引用。
- 暂存五个 milestone 加一份恢复点，六个 run 的理论高峰约 **48 GiB**；加日志和视频，设置本轮 **60 GiB 总预算**。当前 `/home` 约 97 GiB 空闲，可留约 37 GiB 余量。
- 仅在 checkpoint 已评估、结果完整且不是 best/last/final 后，清理本轮自行创建的中间权重。保存新 checkpoint 前检查空间，避免先写满磁盘。
- 使用 raw feature 缓存时冻结 DINO 可能没有实例化，实际 checkpoint 会更小；以实际字节数为准。可训练参数必须回读一致，冻结主干另核对固定 revision/hash。

## 8. 交付与结论边界

每个 run 交付配置、数据 episode 清单、训练曲线、逐 episode 指标、checkpoint 清单/hash、成功及失败代表视频。汇总报告包含数据量对照、随更新次数变化的开发成功率、GPU/loader 吞吐、训练与评估时间占比，以及失败阶段分布。

判断下一步：

- **A50 稳定优于 A10**：支持该任务和预算下增加数据有帮助；后续再加 25 条条件或其他 10 条子集，确认趋势。
- **A10 和 A50 都稳定成功**：本任务当前设置已足以建立可用 baseline，之后扩展任务或层级。
- **训练 loss 降低、闭环仍普遍失败**：优先定位部署预测、动作控制和失败阶段，再设计针对性诊断。
- **不同训练 seed 差异大**：先检查训练曲线和数据覆盖，必要时增加独立训练次数。

本轮只有一个固定 10 条子集，三个训练 seed 衡量训练随机性的影响，不能代表不同数据子集的方差。各 milestone 同属一条 18,000-step cosine 训练轨迹，不能当作“独立训练 1,800/9,000 步”的预算消融。单任务、较小 ACT 配置和自定数据协议的结果，作为本地 baseline 报告。

全部主实验完成后，可选用 human episode 2/3，各在 20 个固定环境 seeds 上评估三个 A50 checkpoint，共 120 次附加 rollout，用于观察更换同任务人类示范的敏感性；不计入主矩阵或最终成功率。

实验参数清单见 [study-plan.json](../experiments/act_placemugrack_study/study-plan.json)。该文件记录实验协议；实际配置、标定记录及执行结果保存在同一实验目录。完成后另写结果报告。
