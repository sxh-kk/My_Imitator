# ACT＋DINOv2 15 任务仿真复现结果

状态：训练与 800 个正式 rollout 已完成，源清单、L3 环境 ID 和配对审计通过。

协议详见 [第十一篇](11_ACT_DINOV2_15TASK_REPRODUCTION.md)。这里只覆盖一个 seed、15 任务规模、ACT＋DINOv2 的仿真结果。

## 主要结果

- 预训练：已见任务：SR 78.0%，Sub-SR@0.95 85.2%，success_once 81.0%。
- 预训练：未见任务零样本：SR 4.0%，Sub-SR@0.95 4.0%，success_once 4.0%。
- 未见任务：少样本从零训练：SR 85.5%，Sub-SR@0.95 89.5%，success_once 85.5%。
- 未见任务：预训练后少样本微调：SR 73.0%，Sub-SR@0.95 79.8%，success_once 73.0%。

P+FT−Scratch：SR -12.5 个百分点，Sub-SR@0.95 -9.7 个百分点。

## 逐任务、逐级别结果

全部 80 个单元的数值保存在 [results.csv](../experiments/act_dinov2_15task/results.csv) 和 [results.json](../experiments/act_dinov2_15task/results.json)。原始 episode、阶段峰值和固定第一个 episode 的视频位于 evaluation/ 对应目录。

## 解释边界

已见、未见零样本和少样本结果分开统计。仅当本次 P+FT 高于 Scratch 时，才称本次协议下预训练有收益；如果不高于，照实报告，不挑 checkpoint 或更改预算。单 seed 结果不能证明稳定的总体趋势。

Sub-SR 使用用户同意的临时阈值 0.95，官方阈值尚未公开；原始阶段峰值可重算。官方目标任务元数据统计保持可用，这里的零样本描述权重没有目标训练，不能称统计信息也完全零样本。

论文主表还平均了 30／45 任务规模，本次不会将单档结果当作主表的等价数值复现。架构、warmup 和少样本时长采用源码默认值的部分在第十一篇明确列出。

## 审计

三个运行均完成 100 epochs；Scratch 和 P+FT 的帧数、更新数、示范 indices、统计和测试 reset 相同。未见任务没有进入预训练；测试没有参与 checkpoint 选择；已登记核心 Python 源码和训练脚本 hash 保持一致。

外层评估在运行期间修正了 L3 环境 ID：使用官方 extract_base_env_name 创建独立 L3 环境。旧的 100 个预训练 L3 episode 归档排除，使用同一 checkpoint、视频条件、统计和 seeds 重跑。修正前后源清单与原因保存在 checks/protocol-amendments/01_official_l3_dispatch/；最终结果仅包含经过环境 ID 审计的 L3 单元。

## 这次进行了什么实验

预训练一个共享 ACT＋DINOv2 模型：官方 15 任务、每任务 L0–L3、每任务—级别 50 条机器人示范，共 3,000 条、1,033,711 帧；100 epochs、403,800 次更新。使用这一个最终 checkpoint 分别测五个已见任务和五个未见任务。

少样本对照在五个未见任务上训练两个共享模型，各使用每任务—级别 10 条、合计 200 条、60,246 帧；各 100 epochs、23,600 次更新。Scratch 的策略没有 IG-10K 预训练，但保留官方预训练视觉主干；P+FT 继承预训练权重，优化器、scheduler 和计数从零开始，所有可训练 tensor 的加载逐项验证一致。两组示范、视频缓存、统计和测试场景相同。这里不是分别为每个目标任务训练一个单任务策略。

三个模型都只取固定最后 epoch；测试结果不参与 checkpoint 选择。预训练评估 400 条，Scratch 和 P+FT 各评估 200 条，共 800 条有效正式 episode。

## 支持的结论与不能支持的结论

已见任务 SR 78% 说明当前共享模型具备已监督任务上的闭环执行能力。按级别平均为 L0 94%、L1 84%、L2 62%、L3 72%；这不是级别越高就必然越难的单调关系。训练已覆盖各级别的固定替代模式，这些分数不能称为完全未见场景或未见物体泛化。

未见任务的权重零样本 SR 4% 来自 200 条中的 8 条成功，全部集中在 FoldBox L1；其他十九个任务—级别单元均为零。这说明本次配置下没有广泛的新任务零样本转移。仍然不能凭这一结果断定 DINO 特征无用或视频条件没有作用，因为没有去视频、错配视频等独立对照。

Scratch 成功 171/200，P+FT 成功 146/200。配对同一初始场景，双方成功 141 条，仅 P+FT 成功 5 条，仅 Scratch 成功 30 条，双方失败 24 条。本次固定协议下，预训练初始化后的终端策略比 Scratch 低 12.5 个百分点；这是本轮观察到的迁移结果，不是预训练普遍有害的证明。

按任务平均四级 SR：PickRemoteControl 97.5%→97.5%，ScanMilkBox 100%→70%，PourKettle 42.5%→50%，PickFood 87.5%→67.5%，FoldBox 100%→80%（箭头为 Scratch→P+FT）。下降主要来自 ScanMilkBox、PickFood、FoldBox；PourKettle 有小幅收益。

第 100 epoch 的平均训练 L1 为 Scratch 0.03647、P+FT 0.02386；P+FT 的训练动作误差更低，闭环 SR 却更低。训练误差是在示范状态分布上计算，闭环会访问自己的动作造成的新状态，所以不能用更低训练 loss 代替更好策略。仅凭该结果不能确定过拟合、遗忘、错误表征或接触时序中的哪一种是根因。

## 与论文的关系

论文 [§4.1／B.2／F.7／表 3](https://arxiv.org/html/2608.22301v1) 的任务划分、分级监督预算、ACT＋DINOv2 输入输出和显式训练参数已按本轮记录执行。它覆盖官方 15 任务这一档的仿真实验结构，并没有覆盖全部模型、30／45 任务、真实机器人或 Arena。

论文表 3 的 ACT＋DINOv2 SR 参考值为 Seen 81%、ZS 2%、Scratch 76%、P+FT 84%；Seen、ZS、P+FT 是 15／30／45 规模平均，不能直接与本轮 15 档的一次运行作严格数值差值。本次“已见较高、零样本很低”的定性方向一致；“预训练后微调提升”的结果尚未复现。论文打印的列值存在四舍五入，其收益列为 +0.09，不能用打印值 0.84−0.76 推断更多精度。

精确模型尺寸、warmup、少样本学习率及时长未全部确认：本次使用源码默认 hidden256／FF512／8头／encoder2／decoder4、warmup5、两组少样本均100epochs及lr1e-4。README 的参考命令为 10epochs／B128／10编码帧／warmup1，还列有 1024／4096／16／12／12 的 ablation 参数；launcher 与论文的 epochs／batch／帧数也不一致。这些差异应先核对对应实验，不能把某一组参数直接认定为唯一官方设置。

Sub-SR@0.95 为用户同意的临时诊断指标，论文阈值尚未确认；目标任务发布统计仍可用。以上限制意味着当前应称为“发布源码与显式假设下的代表性复现”，不能称为论文全部配置和主表数值的严格复现。

## decoder 输出选层：已确认行为与未确认意图

本轮保留了发布源码的行为：[transformer.py](../The-Imitator-Game/examples/baselines/act/act/detr_video/transformer.py) 在 `build_transformer` 中设置 `return_intermediate_dec=True`，返回所有 decoder 层输出；[detr_vae.py](../The-Imitator-Game/examples/baselines/act/act/detr_video/detr_vae.py) 在调用 Transformer 后取 `[0]`，再交给动作头。因此动作预测依赖第一个 decoder 层，后面三层没有贡献到该动作输出。

2026-10-09 使用最终 P+FT checkpoint、真实官方少样本 batch B=2 做前向／反向诊断，未执行 optimizer.step，未修改任何核心源码或 checkpoint。动作 shape 为 2×24×16；decoder 的内部完整输出 shape 为 4×24×2×256。每层有 18 个参数 tensor，第 1 层 18 个均有非零梯度；第 2／3／4 层的 18 个梯度 tensor 均为零。完整记录见 [decoder-gradient-diagnosis.json](../experiments/act_dinov2_15task/checks/decoder-gradient-diagnosis.json)，可用 [诊断脚本](../scripts/diagnose_act15_decoder.py) 重现。

因此“配置四层 decoder”不能解释为四层均参与动作预测；本轮仍是该发布实现的有效基线，但模型结构是否符合作者预期需要确认。它是两组共有的行为，尚未证明它导致 P+FT−Scratch 为负。不得把这个梯度诊断直接写成“已经查明负迁移根因”。

进一步核对发现，[ACT 官方上游 transformer.py](https://raw.githubusercontent.com/tonyzhaozh/act/main/detr/models/transformer.py) 也设置 `return_intermediate_dec=True` 且直接返回层输出 tensor；[上游 detr_vae.py](https://raw.githubusercontent.com/tonyzhaozh/act/main/detr/models/detr_vae.py) 同样取 `[0]` 后进入动作头。因此，前文“源码结构问题”的措辞已收紧为“已确认的实现行为”：它不能被认定为 Imitator 特有的错误，也不能仅凭此宣称复现失效。若论文使用同一实现，保留这一行为才是对应基线的复现；采用最后一层属于需要单独登记、重新训练的实现对照。

从 shape 可以明确索引含义：decoder 内部返回 `(4,24,B,256)`；Transformer 的 `transpose(1,2)` 把它转成 `(4,B,24,256)`；`[0]` 选层轴的第一项，得到 `(B,24,256)`；动作头 `Linear(256,16)` 输出 `(B,24,16)`。它没有选择 batch 的第一个样本，也没有选择动作块的第一个时间步。

反向的零梯度来自依赖关系：预测只依赖第一层输出，第一层输出不依赖后面三层参数。KL 来自独立的 CVAE encoder 分支，不会为后面三层提供辅助动作监督。诊断中后三层的 `grad` tensor 实际存在，只是全部元素为零；不能描述成 `grad is None`。AdamW 的 weight decay 仍可能改变这些参数，所以准确表述是“没有来自本动作损失的学习信号”，而非“参数从未变化”。

另一个待确认的表述差异：论文 F.7 描述视频 z 加入 decoder token，而当前源码把视频 z、CVAE latent 和 state 拼入 encoder 源序列，再由 decoder 的动作查询访问 memory。本次没有自行改变条件注入方式。

## 失败阶段与下一步

按临时 0.95 阈值，ScanMilkBox 的 12 个 P+FT 失败 episode 均未达到 bring_milkbox 阶段阈值，部分已经达到接近与抓取阈值。PickFood 的 P+FT 在 L2 为 0/10，失败轨迹中第二个物体相关阶段均未越过阈值。FoldBox 的 8 个新增失败全部在 L0，P+FT 2/10、Scratch 10/10。上述峰值可缩小看录像的范围，但不能单独确诊动作、抓取或碰撞原因。

优先冻结本轮结果，核对 decoder 选层与官方架构／微调 recipe，再用固定数据和独立开发场景做受控实现回归。随后补训练 seeds，验证终端 SR 和适应速度。评估第 10／50／100 epoch 的开发曲线可检验学习过程，不能用最终测试选择有利 checkpoint 后改写本轮结果。单一训练 seed 下的 800 条评估轨迹不等于 800 次独立训练复现。
