# PlaceMugRack：L0／L1／L2 数据覆盖实验计划

日期：2026-10-07。执行进度更新：2026-10-08。状态：**900条冻结跨级别评估已完成，数据、真实样本、更新／评估一致性检查及标定通过；B012_s1 已完成18,000次更新、五个checkpoint及300次开发评估，当前并行执行 B0_s1、B01_s1。控制器将自动接续其余训练、独立最终测试与审计。运行进度见 [status.json](../experiments/act_placemugrack_levels/status.json)。**

10月8日09:28恢复说明：旧控制器退出、进度文件陈旧；已将中断的 B0_s1／B01_s1 归档，并从原seed重新训练。已完成的 B012_s1 保留，实验源码和冻结协议不变。队列现由 `imitator-placemugrack-levels.service` 承载；详见[恢复记录](../experiments/act_placemugrack_levels/RECOVERY.md)。检查当前真实进程请运行 `python scripts/check_levels_status.py`，仅阅读静态状态文件无法判断进程是否存活。

本轮围绕同一个任务 PlaceMugRack，先测试现有 L0 策略跨级别执行的能力，再比较机器人监督数据覆盖范围的收益。沿用当前官方 ACT、冻结 DINOv2-L、human video 条件和仿真实现；新增工作限定在实验包装脚本、数据配置、采样与记录。既有 A50 权重、数据及 [上一轮结果](07_PLACEMUGRACK_EXPERIMENT_RESULTS.md) 保留。

**执行顺序：冻结 A50 → 测 L0/L1/L2 → 检查对应机器人示范 → 准备三组数据与统计量 → 单 seed 试运行 → 完成九组训练 → 独立最终测试 → 分析收益与失败。**

## 1. 这轮要回答什么

1. **直接迁移**：只经过 L0 训练的现有 A50，保持权重、视频和统计量不变，在 L1、L2 上还能完成多少轨迹？失败发生在抓取、抬升、接触挂架还是释放后保持？
2. **位置覆盖**：在同样计算预算下，加入 L1 机器人轨迹，是否改善 L1，并保留 L0 能力？
3. **物体与动作角色覆盖**：继续加入 L2 机器人轨迹，是否改善 L2 的抓取、搬运和终端成功？是否影响原有级别？

这是一项**单任务、多个场景级别的监督覆盖实验**。三组训练都执行“把杯子放到挂架”这一任务；并不是学习三个不同任务，也不是多任务共享研究。

假设可以被结果否定：加入对应级别数据未必在固定预算下提高成功率；新增数据也可能带来原有级别退步。18,000 次更新是本轮统一预算，不预先视为所有条件都已收敛。

## 2. 先理解本任务的实际级别变化

论文将 L1 定义为保留物体身份、改变空间布局；L2 为保持语义角色、改变外观或几何。论文附录 C.5 还明确说明，**仿真 L2 同时改变承担操作的手臂**。因此，L2 结果要同时检查物体适应和动作角色变化。[论文定义及仿真构造说明](https://arxiv.org/html/2608.22301v1)

本地 commit `d6d16ec511bc389e0a207692730c137bc022ef14` 的实际配置见 [任务文件](../The-Imitator-Game/mani_skill/envs/tasks/tabletop/dual_tasks/_038_place_mug_rack.py)、[级别工具](../The-Imitator-Game/mani_skill/envs/tasks/tabletop/utils/L0_L3_utils.py) 和 [镜像实现](../The-Imitator-Game/mani_skill/envs/sapien_env.py)：

- **L0**：杯子 `039_mug/model11`，尺度 `(1,1,1)`；挂架 `040_rack/model0`，尺度 `(1.5,1.5,1.5)`。
- **L1**：保留上述资产，杯子 XY 增加 `(-0.1,-0.1)` 偏移；仍有初始化随机扰动。
- **L2**：杯子替换为 `039_mug/model4`，尺度 `(0.08,0.08,0.08)`；挂架仍为 model0；启用场景物体左右镜像。执行前检查确认，官方 ACT evaluator 在入口显式设置 `set_lr_mirror_robot_pose_enabled(False)`，故机器人底座保持原位置；工具模块的默认 True 不代表本评估入口的行为。不得关闭物体镜像后仍把结果记作官方 L2。
- 这些级别都存在小范围 reset 随机化。L0 也不意味着每条 rollout 初始状态完全相同。

仿真成功谓词是：

```text
success = 与挂架发生接触 AND 杯子高于阈值 AND 当前未被夹持
```

`is_obj_placed`、`is_robot_static` 是辅助记录，不在当前 `success` 的布尔表达式里。因此本文的“终端成功”只表示第 500 步满足上述官方谓词；不能自动解释成持续静止或长期稳定悬挂。

### 必须单独记录的归一化限制

现有 A50 的 [训练统计量](../experiments/act_placemugrack_study/prepared/A50/sim/L0_TwoRobotPlaceMugRack-v1/meta/stats.json) 中：

- action 第 15 维（从 0 开始，第二臂／通常所称右臂的夹爪命令）满足 `q01=q99=1`。
- state 第 16、17 维是第二臂两个手指位置，`q01=q99≈0.04`。
- [官方 normalizer](../The-Imitator-Game/examples/baselines/lerobot_dataset/normalizer.py) 在输入归一化时截断到 `[-1,1]`；动作反归一化采用 `0.5 × (prediction+1) × (q99-q01) + q01`。

**因此，不论模型如何预测第 15 维，现有 A50 送往第二臂夹爪的命令都会变成 1，即保持张开。** 在需要第二臂闭合抓取的 L2 示范行为上，这是当前部署流程的明确限制。评估中机器人底座不镜像；仍核对 root pose，日志以 agent index 区分两臂。

不能据此预先宣称所有 L2 rollout 都必然失败：其他手臂或接触行为仍可能满足成功谓词。但若 L2 抓取失败，不能只归因于视觉编码或模型容量。对于非零动作范围，预测可以超出 `[-1,1]` 后再反归一化；“训练范围窄”与“该动作维被固定”是两种不同情况。

## 3. 阶段 A：现有 A50 的冻结跨级别测试

### 使用哪些权重

使用上一轮已冻结的三个 A50 selected checkpoint，不重新选择：

- `A50_s1`：step 5,400。
- `A50_s2`：step 12,600。
- `A50_s3`：step 5,400。

完整路径和 SHA256 以 [final-test-selection.json](../experiments/act_placemugrack_study/final-test-selection.json) 为准。执行前后核对文件 hash；所有级别均使用对应的同一份权重。

### 固定什么、改变什么

固定 checkpoint、`human_H57` episode 1 的十帧视频／冻结特征、A50 的 L0 训练 stats、图像预处理、控制器、物理后端、动作聚合和计分。只改变 simulator level 及其官方场景配置。

每个模型分别测试：

```text
L0：reset seeds 3000–3099，100 episodes
L1：reset seeds 3000–3099，100 episodes
L2：reset seeds 3000–3099，100 episodes

3 个 A50 模型 × 3 个级别 × 100 = 900 episodes
```

L0 也重测一遍，获得与本轮 L1/L2 对应的参考。上一轮 `2000–2099` 结果仅作历史记录，不拿它直接与本轮新 seeds 的 L1/L2 当作配对差值。

**配对检查只要求同一级别、同一 reset seed 在不同模型间具有相同初始场景、state 和 RGB hash。** 不要求 L0/L1/L2 之间相同；跨级别初始场景本来就应该变化。

### 实施前置工作

现有 [evaluate.py](../experiments/act_placemugrack_study/evaluate.py) 的 level、task ID、prepared 路径仍固定为 L0；当前不能只换一个命令参数就完成本轮。新建独立实验目录 `experiments/act_placemugrack_levels/`，增加外层评估入口，显式接收：

```text
checkpoint / level / exact repo_id / stats view / human video
output root / reset seed range / episode count / num_envs
```

调用原 ACT evaluator 和 [环境工厂](../The-Imitator-Game/examples/baselines/act/act/make_env.py)，在每个子进程中传入真正的 `l_level`。只在主进程设置环境变量，可能无法正确切换已有 forkserver 创建的 worker。

阶段 A 的三个级别建立**统计量别名视图**：各 level repo ID 映射到同一份 A50 L0 metadata/stats。使用 [evaluate_processor](../The-Imitator-Game/examples/baselines/lerobot_dataset/evaluate_processor.py) 的 metadata 加载路径，禁止按测试 level 自动读入 L1/L2 的发布版 stats。此阶段不需要 L1/L2 机器人训练数据。

先做每级别 4 条流水线检查，使用 `2800–2803`，不计入正式 900 条。核对模型/尺度、mirror 是否实际生效、两臂 root pose、reset seed、动作拆分、stats hash 和评分。L2 所需 model4 JSON、visual/collision GLB 当前均已在本地；仍要核验文件及资产 provenance。

### 本阶段交付

- 每个 A50 seed 在 L0/L1/L2 的 `success_at_end`、`success_once`、成功次数／100。
- 跨级别差值，以及相对本轮 L0 的下降。
- 互斥失败分类、两臂抓取情况、state 截断比例和夹爪原始命令范围。
- 每个模型×级别预选两条录像，共 18 条；失败代表可在分析后补录，标注为补录且不追加计分。

这一步回答的是“**现有完整 L0 部署流程能迁移多少**”。此时不修复 normalizer、不扩大动作范围、不改变 human 条件；任何修改都会改变被测试对象。

## 4. 阶段 B：用官方机器人数据覆盖变化

### 数据范围与下载量

继续固定官方数据 revision：`57fa861d911afe899da5c0f28d973151411d46d1`。仅使用 `imitator_sim_v1_zed2i/` 下三个 PlaceMugRack repo；各自 episodes `0:50`，即 0–49。

2026-10-07 只读检查官方 Hugging Face tree/API 与 metadata 后确认：

- **L0**：已在本地；50 条、11,228 帧。
- **L1**：50 条、12,079 帧；目录全部 8 个文件共 **46,549,243 bytes，约 44.39 MiB**。
- **L2**：50 条、11,304 帧；目录全部 8 个文件共 **43,978,397 bytes，约 41.94 MiB**。
- L1+L2 新增数据共 **90,527,640 bytes，约 86.33 MiB**，不包含已有 human 数据、模型与资产缓存。

官方目录：[L1](https://huggingface.co/datasets/imitator-game/IG-10K-Dataset/tree/57fa861d911afe899da5c0f28d973151411d46d1/imitator_sim_v1_zed2i/L1_TwoRobotPlaceMugRack-v1)、[L2](https://huggingface.co/datasets/imitator-game/IG-10K-Dataset/tree/57fa861d911afe899da5c0f28d973151411d46d1/imitator_sim_v1_zed2i/L2_TwoRobotPlaceMugRack-v1)。大小来自对应固定 revision 的 API 文件清单求和；上述核对未下载视频或机器人轨迹。

每个级别包括动作/state parquet、episode metadata、info/stats/tasks、RGB 和 depth 视频及 `.complete`。虽然本轮策略只用 RGB，当前 loader 的完整性检查涉及 depth 文件，按这套完整的小目录准备。下载到现有 `/var/tmp/imitator-game-zxc/data/`，验证文件大小与上游可用 checksum，保留清单；不复制整套 IG-10K。

### 先验证机器人示范确实覆盖失败行为

在训练前分别读取每级别所有轨迹首尾及若干中间样本，核对：

1. state `(18,)`、action `(16,)`、每条 trajectory 边界以及 action chunk `(24,16)` 的 padding；没有跨 episode 拼接。
2. timestamp/frame index 与 RGB 对齐；实际 episode 数和帧数与 metadata 一致。
3. L1 的抓取、搬运轨迹是否区别于 L0；检查关节动作范围及可用末端轨迹信息，并查看预选示范视频。
4. L2 的替代杯子是否对应当前资产，活动手臂是否与仿真一致；第 7/15 维夹爪标签是否具备对应闭合／释放过程。不能只凭 repo 名称确认角色切换。
5. 不同级别的 state/action 排列、agent index 与 PD controller 语义一致。
6. 三个 robot repo 都映射到 `human_H57`，没有因 level 而换 human 视频或描述。

保留官方已有图像增强和预处理，全组一致，不新增外观增强策略。新增抓取与搬运动作通过对应 robot action 监督学习。若数据与当前场景构造不对应，先解决版本／加载问题；不能把错误数据下的训练记作正式覆盖实验。

### 三组正式训练数据

```text
B0   ：L0                  →  50 条，11,228 帧
B01  ：L0 + L1             → 100 条，23,307 帧
B012 ：L0 + L1 + L2        → 150 条，34,611 帧
```

L0 的 50 条在所有组完全相同，L1 的 50 条在 B01/B012 完全相同，不按本轮评估成败挑选机器人示范。每组分别使用训练 seeds 1、2、3，共 **9 个 run**。

上述机器人轨迹全部用于训练，不额外留出离线 robot 验证集；开发和最终指标来自新的在线 reset。不同 reset 仍属于同一官方 level 的场景分布，不是新的物体类别或任务。

所有 ACT 和可训练 Adapter 从头初始化，视觉主干使用现有同版本预训练权重。不从 A50 checkpoint 接着训练：否则新组混入额外的历史 L0 更新预算。

## 5. 计算预算、级别采样与统计量

### 相同总预算

统一 batch 64，每 run 18,000 次 optimizer update，共 **1,152,000 次训练样本呈现**。九个 run 合计 **162,000 次更新、10,368,000 次样本呈现**。

多个级别采用外层 batch sampler，不改官方 `HumanSimPairedDataset` 或 ACT loss：

- **B0**：每 batch 64 个 L0。
- **B01**：每 batch 32 个 L0、32 个 L1。
- **B012**：每 batch 21/21/22 个，额外一个样本在 L0、L1、L2 间轮换；每 3 个 batch，各 level 恰好出现 64 次。
- 每个 level 内对真实帧索引随机打乱、耗尽后继续下一次打乱；拼成满 batch。记录每个 level 的采样顺序、游标及累计计数。这里是**级别均衡、级别内按帧采样**，不声称每条长度不同的 trajectory 同频。
- 只有一个语义任务，任务均衡自然成立；level 不作为额外标签、prompt 或 policy 输入。

18,000 能被 3 整除，故每个 run 的计数应精确满足：

```text
B0  ：L0 = 1,152,000
B01 ：L0 =   576,000；L1 = 576,000
B012：L0 =   384,000；L1 = 384,000；L2 = 384,000
```

这项设计比较的是**相同总计算预算下的数据覆盖收益**。B012 分给每个级别的更新样本更少，不能用相同 epoch 给大数据组额外计算量。若 B012 尚在改善而 B0 已饱和，只能说本预算下结果如何；后续等比例加训另立一轮。

### 每个条件只有一套共享坐标尺度

继续使用官方 `bounds_q99`。为了避免评估器通过测试 level 选择不同的动作尺度，同一个模型训练／部署时，三个 level 都必须使用**完全相同的 state/action stats**。

- B0：沿用并复核现有 A50 的 L0 训练 stats。
- B01：仅合并 L0、L1 训练轨迹，计算一套逐维 q01/q99。
- B012：仅合并 L0、L1、L2 训练轨迹，计算一套逐维 q01/q99。
- 计算方式与已有 `prepare.py` 一致：原始训练数值转 float32，拼接所有所选 level 的训练帧，`np.quantile(..., method='linear')`。这里每个原始帧等权，统计量不是级别加权 quantile；训练的级别均衡由 sampler 实现。
- 将同一份派生 state/action stats 放入该条件所有 level 的 metadata 视图，使用链接访问原始 data/videos。原始下载 metadata 不改。
- exact repo ID 分别为 `L0_...`、`L1_...`、`L2_...`，避免共用 task basename 时 normalizer 映射被覆盖。核对训练、评估 state 处理和 action 反归一化实际解析到的 stats hash 一致。
- 不使用 dev/test rollout 计算 stats；B0/B01 不借用其未训练级别的发布版 stats。

每组记录逐维 q01/q99、零宽范围、训练标签截断比例、部署 state 截断比例，以及反归一化后动作范围。roundtrip 必须调用真正的官方 normalizer，按当前裁剪和零宽维规则核验，不能只用一个近似公式代替。

**三组统计量会随合法训练数据覆盖而变化。** 尤其 B0/B01 的第二臂夹爪可能仍被固定，B012 应通过 L2 标签获得其活动范围。因此主实验的结论是“扩大监督覆盖后，完整 ACT 训练／部署流程是否改善”，同时包含尺度覆盖和参数学习的影响；不能把全部收益归因于网络表征变化。

若后续需要隔离尺度影响，可以另做三组共用一份 L0+L1+L2 训练池校准 stats 的对照，并全部从头训练。该对照必须注明 B0/B01 已获得额外级别的统计信息，不能称为严格未接触这些级别的迁移实验。本轮不混入这项额外矩阵。

### human video 条件

训练统一 `human_H57` episode 0；开发／最终测试统一 episode 1。直接复用上一轮十帧采样结果及冻结 DINO raw feature，核对视频、采样索引与缓存 hash。

三个 level 共享同一个 human task key，既不重采样视频，也不换成更贴近 L2 的 human 条件。Adapter 在各 run 内训练，DINO 主干保持冻结。此设置控制视频变量，但单任务正确执行不能证明策略通过视频理解了不同任务意图。

## 6. 固定模型和训练参数

沿用上一轮已经验证的配置：

```text
robot RGB                  zed2i，224×224
state / action             18 / 16，双臂 qpos 控制
ACT output                 B×24×16，obs_horizon=1
robot backbone             ResNet18
human backbone             frozen DINOv2-L，10 frames + 可训练 Adapter
ACT                        hidden=256，feedforward=512，heads=8
                           encoder_layers=2，decoder_layers=4
batch / updates            64 / 18000
AdamW                      lr=1e-4，backbone-named lr=1e-5，wd=1e-4
scheduler                  cosine，总步数18000，warmup=900
gradient clipping          backbone-named=0.1，other=1.0
KL weight                  10
precision                  BF16，沿用现有路径
loader                     初始 workers=8，pin memory，persistent workers
internal evaluator         disabled；使用独立闭环 evaluator
```

保持参数名决定的 optimizer 分组、CVAE train/inference 行为和当前第一个 decoder 输出索引。损失中的 L1 指动作回归误差，不是场景级别 L1。不同组归一化尺度不同，normalized L1 loss 不直接跨组排名。

同一训练 seed 的三个条件在第一次更新前，ACT/Adapter 所有可训练参数 hash 应相同。视频缓存准备、loader 初始化可能消耗 RNG，须显式检查初始化，不能只确认都写了 `seed=1`。

先用真实 batch 验证外层更新与当前官方 epoch 分支的 loss、参数更新一致；核对第 900/18,000 步的学习率。正式长训练前，对三个条件及所有 level 检查真实 batch 的标签来源、norm stats、padding mask 和有限数值。

### 试运行与正式队列

先完成 B012_s1：覆盖全部级别，最容易暴露多 dataset 映射和双臂动作问题。在 1,800/5,400/9,000 步看开发结果、训练 loss 与各臂动作。

- 如出现 NaN、标签／动作顺序错误、错误 stats、level 未切换或 checkpoint 回读不一致，先修正包装流程；受影响正式 run 重跑。
- 如果部署长期为零成功，但训练 loss 下降，先检查无真实 action 输入的 CVAE 推理、归一化和失败阶段；不直接扩大其余八个 run。
- 在流程正确的前提下，低成功率本身不触发单独给某组加训、换视频或修改模型。
- 如果试运行没有引发实质配置变更，B012_s1 纳入九组正式结果。若改动训练协议，归档试运行，统一冻结新协议后九组重新开始。

随后建议队列：`B0_s1 → B01_s1 → B0_s2 → B01_s2 → B012_s2 → B0_s3 → B01_s3 → B012_s3`。B012_s1 已先完成。并发只改变执行时间，不改变每个 run 的预算。

## 7. 阶段 C：开发选择与独立最终测试

### 所有模型统一评估 L0、L1、L2

固定当前闭环设置：`physx_cpu`、`rt-fast`、`pd_joint_pos`、num_envs=4、每 episode 500 步、每 4 步查询策略、light temporal aggregation 窗口 4。DTW/TSS 不参与选择，统一关闭。

即使 B0 未训练 L1/L2，也评估全部级别，始终使用 B0 的同一套训练 stats；其他条件同理。所有 run 的相同 level/reset seed 对齐初始状态与 RGB hash。

### 开发集：选择 checkpoint

- 保存步数：**1,800 / 5,400 / 9,000 / 12,600 / 18,000**。
- 每 checkpoint、每 level 使用 **4000–4019**，20 个 episode，即每 checkpoint 60 个。
- 主选择分数：三个 level 的 `success_at_end` 等权平均。
- 分数相同时，比较三个 level 的 `success_once` 平均，再取较早 checkpoint。
- 每 run 300 个开发 episode，九个 run 共 **2,700 个**。
- 保留所有五个 checkpoint 的逐级别学习曲线，避免只展示选出的最高点。

本轮优先终端成功，选择规则与上一轮 A50 的 `success_once` 优先不同。因此旧 A50 是冻结迁移参照，**新的 B0 才是本轮三组训练的正式基线**。

20 次／level 的开发成功率粒度是 5 个百分点，checkpoint 选择可能有噪声；用独立最终测试验证，不把 dev 峰值当性能结论。

**B0/B01 会接触未纳入其机器人训练数据的 level 开发评分。** 这用于统一三组部署目标和选择预算，不加入这些 level 的 action 监督；但它们在阶段 C 的最终结果不能称为“目标级别从未参与任何模型选择”的严格 zero-shot。阶段 A 才保持原有 L0 选择的 A50 完全冻结。

### 最终集：只在九个模型全部冻结后打开

- 每个 selected checkpoint、每 level 使用 **5000–5099**，100 个 episode。
- 9 个模型 × 3 个 level × 100 = **2,700 个最终 episode**。
- 使用共同的 level/seed 清单；保存逐 episode 初始场景、终端 flags、首次阶段事件和两臂记录。
- 正式测试前冻结 config、sampler、stats、human 条件、代码 hash 和九个 checkpoint SHA256。
- 最终测试结果不反馈到本轮参数、选择规则或继续训练决策；需要下一轮时另建协议和新测试 seeds。

测试是在相同官方 level 的新 reset 上闭环执行，没有留出新的物体模型。B012 的 L2 成功支持已覆盖替代杯子场景中的执行能力，不能推出对任意未见杯子、未见任务或真实机器人均能泛化。

## 8. 指标与失败分析

**主指标 `success_at_end`；辅助指标 `success_once`。** 每个 run/level 报告成功次数／100、比例；每条件/level 报告三个训练 seed 的均值和样本标准差。标准差表示训练间波动，不当作 95% 置信区间。

比较相同训练 seed 下的三组差值：

```text
位置覆盖收益       ΔL1 = B01(L1)  − B0(L1)
L2覆盖收益         ΔL2 = B012(L2) − B01(L2)
原有级别保持       B01(L0) − B0(L0)
                   B012(L0/L1) − B01(L0/L1)
综合部署表现       三个级别 success_at_end 的等权平均
```

100 次 reset 衡量一个模型的 rollout 波动；三个训练 seed 衡量训练重复性。不能把三次训练的 300 条 rollout 当作 300 个独立训练模型。仅三个训练 seed 时，以配对差值和一致性描述证据，不轻率声称统计显著。

### 互斥失败分类

沿用上一轮分类，按以下优先顺序，每条轨迹只进入一类：

1. 第 500 步成功：`success_at_end`。
2. 曾经成功但第 500 步失败：`transient_success_only`。
3. 从未抓到：`never_grasped`。
4. 抓过但从未达到抬升阈值：`grasped_without_lift`。
5. 抬升过但从未接触挂架：`lifted_without_rack_contact`。
6. 接触挂架但从未成功：`rack_contact_without_success`。

首次事件是便于定位的摘要，并不保证按正确顺序发生。补充记录首次成功、成功总步数、最后成功时刻、当前是否夹持及视频，区分意外接触、短暂悬挂和持续成功。

每臂额外记录抓取首次发生时间、是否承担搬运、夹爪反归一化命令 min/max；读取 episode 内 flags/状态，记录每维 state 超出训练 q01/q99 的比例。不要仅凭模型预测的 normalized action 就断言夹爪实际闭合。

### 根据结果做判断

- **A50 L1 低，B01 L1 提高**：支持当前任务的位置相关机器人动作覆盖有效；结合失败从“抓不到”向后移、关节范围与轨迹差异分析。不是新增方法贡献。
- **A50/B01 L2 低，B012 L2 提高**：支持加入替代物体／另一臂动作及其训练统计量的完整流程改善；明确 normalizer 零宽维解除也参与了变化。
- **B012 L2 提高但 L0/L1 下降**：存在固定预算下的能力取舍；查看各 level 样本呈现量、学习曲线和终端不稳，不只报告 L2 最高点。
- **训练 loss 低、闭环仍差**：检查 inference latent、动作尺度、时序执行和分布偏移；低离线误差不足以证明部署成功。
- **全部条件 L2 均差**：先看示范是否匹配当前模型／镜像、动作角色是否正确、常数维是否解除、部署阶段有无改善。流程通过后，再讨论预算或其他实验。
- **现有 A50 已在 L1/L2 很好**：仍按预注册矩阵记录增益和退步；若接近上限，有限提高不代表数据无用。不得再按测试分数临时加难场景来放大收益。

可将对应级别平均提高至少 10 个百分点、三个训练 seed 的配对差值都为正、原有级别平均下降不超过 5 个百分点，作为后续工程投入的一个筛选目标。它不是显著性标准；最终报告仍完整展示连续数值与未达目标结果。

本轮 50/100/150 条数据量也发生变化。因此结论是“嵌套覆盖方案在固定计算预算下的效果”，不能单独证明是场景覆盖而非示范总量带来的收益。若出现明确收益，下一轮可固定总示范数 50，预先选定 L0=50、L0/L1=25/25、L0/L1/L2=17/17/16 的补充对照，单独处理数据量因素。

## 9. 4090 调度、时间与空间

### 本机与吞吐

2026-10-07 核对：驱动报告 RTX 4090、49,140 MiB 显存；7950X／32 线程，约 187 GiB RAM；`/home` 可用约 **89 GiB**，根分区 `/var/tmp` 可用约 **201 GiB**。使用 conda `imitator`。

优先复用上一轮实测配置：B64、workers8、四个评估环境。上一轮单训练约 1,478 samples/s，双训练合计约 1,753 samples/s；双训练提升 18.6%，整个正式队列 GPU 利用率均值 61.1%、显存峰值 20.83 GiB。以上是 L0 旧实验的测量，不保证多级别 loader 和 L2 接触的吞吐相同。

在新数据上做小规模吞吐复核，不再盲目遍历所有 batch：

- 单 B012 预热 50、测量 200 updates，必要时延长。
- 比较单训练、两个训练，以及一个训练加一个评估的总有效吞吐。
- 总并发最多两个顶层 GPU 任务；所有 evaluator 用全局锁串行，CPU physics worker 统一线程设置。
- 同时统计显存、CPU、step 时间和 loader 等待。若并发相对串行总吞吐提升不足 15% 或资源异常，保持串行；不靠 GPU 利用率百分比决定是否有效。
- 维持 B64 与 18,000 steps。调整 workers／并发须在正式矩阵前统一冻结，避免 run 间资源设置变化影响比较。

### 主实验工作量

```text
冻结A50跨级别          900 episodes
九组开发选择         2700 episodes
九组最终测试         2700 episodes
合计                 6300 episodes

按每条500步：3,150,000 个个体环境 step
按每4步查询：787,500 个个体 action-chunk 预测
num_envs=4时：787,500 次 vector step，196,875 次批量 policy 查询

新训练：162,000 updates，10,368,000 次样本呈现
```

每次评估检查 `vector_steps=episodes/4×500`、`policy_queries=episodes/4×125`、episode 数与 reset seed 清单；阶段 A 流水线检查、吞吐标定和失败视频补录另计，不混入正式分母。

沿用上一轮累计训练更新耗时，162,000 steps 的训练进程工作量约 **3 小时量级**。上一轮开发／最终评估含初始化累计约 1.04 小时／1,200 episodes，按该平均吞吐外推本轮评估约 **5.5 小时进程工作量**。这些工作可部分重叠，且实际 L2、level 切换、无录像开发评估会改变耗时；它们不是墙钟承诺。

建议预留 **一个工作日窗口**完成正式计算；脚本实现、数据核验、故障修复与报告另计。阶段 A 先标定三个级别真实 seconds/episode，再更新排期。资源中断保留完整协议，不悄悄缩短某组更新数或测试数。

### 空间

- L1+L2 数据新增约 86.33 MiB，保存在 `/var/tmp`。杯子 model4 必需文件已在本地，本轮按当前状态无新增大资产下载需求。
- 现有完整 checkpoint 约 205,374,913 bytes。9 个 run × 5 份约 **8.61 GiB**；每个 run 的 best/last/selected 用链接指向已有文件，不额外复制。
- 开发评估统一不录视频；阶段 A 每 cell 两条共 18 条，最终每 cell 两条共 54 条，主实验共 **72 条代表录像**。全部 6,300 条仍完整计分。
- 日志、一次恢复点、记录与必要补录留余量，本轮实验目录预算 **30 GiB**。以当前 `/home` 约 89 GiB 空闲计算，仍留约 59 GiB；每次保存前检查实际空间。
- 不删除上一轮权重、原始数据或模型缓存。中间 checkpoint 均先保留供学习曲线复查；后续清理单独安排。

## 10. 包装器实现清单与完成标准

下列包装器已经实现；结果文件随队列完成产生，完整验收标准保持不变：

```text
experiments/act_placemugrack_levels/
  plan.json                    固定矩阵、seeds、预算、选择规则
  prepare.py                   下载清单、派生stats、视频/cache引用
  evaluate.py                  level参数化、冻结stats、逐episode诊断
  train.py                     官方ACT更新 + 均衡level sampler
  run_queue.py                 最多两个GPU任务、全局评估锁
  prepared/                    三个条件的metadata/数据视图
  phase_a/                     已有A50的跨级别结果
  runs/B0_s1 ... B012_s3/       九组checkpoint、曲线、dev、final
  checks/                      数据/采样/更新/载入/初始场景审计
  results.json                 逐run/level及配对汇总
  failure_analysis.json        互斥分类、两臂、归一化诊断
```

实现时复用旧实验已验证的纯逻辑，参数化新研究的 root/task/level。不能直接导入会把 cwd、ROOT、TASK 固定为旧 study 的辅助模块后误写原目录。实验脚本冻结后记录 hash，官方核心源码执行前后核对一致；保留本机原有 `.gitignore`、`pyproject.toml` 和 `uv.lock` 状态。

checkpoint 原子落盘、回读 tensor、核对 optimizer/scheduler step 后再标记 ready。恢复训练需要保存三个 level 的 sampler 游标、轮换位置、loader/RNG 状态以及 optimizer/scheduler；不能仅加载模型后把未完成更新数当作已完成。

完成本轮必须具备：

1. 阶段 A 900 条冻结迁移记录，权重／human／stats hash 与旧 A50 一致。
2. 数据三套边界、动作顺序、角色切换及统计量审计通过。
3. 九个 run 各完成 18,000 次更新，level 呈现计数精确符合预算。
4. 45 份 milestone checkpoint 的开发记录完整，按统一规则选出九个模型。
5. 全部选择冻结后完成 2,700 次最终测试；测试输出目录不覆盖已有结果。
6. 同级别同 seed 的初始场景配对、模型载入、normalizer 映射及环境级别检查通过。
7. 每条件/级别的成功率、seed 波动、配对差值、失败阶段、夹爪范围、截断比例及资源记录完整。
8. 报告收益、退步、未收敛或不确定结果，并明确统计量变化、示范总量、开发选择与未见场景边界。

优先交付阶段 A，因为它立即定位现有策略的限制。随后按已冻结协议执行三组覆盖对照；不以已有 L0 高成功率或训练 loss 的单一数值替代跨级别闭环证据。

查看当前运行进度：

```bash
/home/zxc/miniconda3/envs/imitator/bin/python /home/zxc/Imitator/experiments/act_placemugrack_levels/progress.py
```

执行准备中发现的镜像描述差异已按官方 ACT 入口修正。首个原型检查的失败输出保存在 `checks/prototype_smoke_L0_failed/`，不计入主实验。训练 worker 的增强状态未实现精确恢复，因此正式中断 run 保存诊断信息后从相同 seed 重跑，禁止近似续训混入可比结果。
