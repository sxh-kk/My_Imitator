# ACT＋DINOv2 官方 15 任务仿真复现

准备日期：2026-10-08。完成日期：2026-10-09 19:00。三个条件均完成 100 epochs、800 个有效正式 episode，最终审计通过。结果及复现边界见 [第十二篇](12_ACT_DINOV2_15TASK_RESULTS.md)，其中新增真实 batch 的 decoder 梯度诊断；核心实现与 checkpoint 保留本轮版本。

## 目标与完成条件

使用一个共享 ACT 模型完成官方 15 任务预训练，并在官方 5 个已见任务、5 个未见任务上逐级评估 L0–L3。随后在五个未见任务的同一批少样本上训练两个共享模型，比较从零训练与预训练后微调。

这是 ACT＋DINOv2、15 任务规模、仿真域、一个训练 seed 的代表性复现。它不覆盖其他模型、30／45 任务规模、真实机器人与 Arena 人类评价，不能据此声称完整复现论文全部结论。

完成条件：三个训练条件均完成既定预算并通过 checkpoint 回读；80 个条件—任务—级别评估单元各完成 10 个 episode，共 800 个正式 rollout；逐单元保存 SR、临时 Sub-SR、阶段峰值、配对种子和代表录像；数据边界、源文件、优化器步数与配对关系审计通过；生成第十二篇最终报告。

## 协议来源及明确存在的差异

论文依据：[论文 §4.1、附录 B.2／B.3／F.7](https://arxiv.org/html/2608.22301v1)。源码 commit 固定为 `d6d16ec511bc389e0a207692730c137bc022ef14`，数据 revision 固定为 `57fa861d911afe899da5c0f28d973151411d46d1`。

论文 F.7 写明 ACT 使用 100 epochs、batch 256、DINOv2-L/14 编码 4 帧、256 维任务嵌入、动作块 24、L1＋10×KL、AdamW、学习率 1e-4。仓库 `run_exp_act.sh` 则设置 10 epochs、batch 128、10 帧。此次以论文的三个显式数值为主；不会把 100-epoch 运行表述为启动脚本原封不动执行。

论文未完整说明 ACT Transformer 尺寸、warmup 和少样本优化时长。此次采用源码默认值：hidden 256、FFN 512、8 heads、2 encoder／4 decoder layers、5 warmup epochs；少样本两组均用 100 epochs、batch 256、相同优化器和 schedule。README 另有 1024／4096／16／12／12 的 ablation 参数，但启动脚本和论文没有明确对应，故没有混入此次主运行。

核心 dataset、ACT、encoder、simulator 和官方 rollout 函数保持原样。外层脚本负责固定配置、记录进度、只加载权重进行微调、保存 checkpoint、读取阶段峰值与审计。

## 数据与任务划分

直接复制官方 `human_train_config_15.json`、`sim_train_config_15.json` 和官方未见任务 `*_test_config_unseen.json` 的 episode 范围。

预训练任务：StirSpoon、PlaceClothBasket、PlaceMagazineFolder、PickWash、PlaceChipsRack、PlaceFruitBox、PlacePlateRack、CutFruit、PlaceFileFolder、PlaceBrushRest、CleanCup、GrindFood、LiftLidFromSkillet、FoldTowel、PlaceMugRack。

已见评估任务：StirSpoon、PlacePlateRack、PlaceFileFolder、FoldTowel、PlaceMugRack。

未见评估任务：PickRemoteControl、ScanMilkBox、PourKettle、PickFood、FoldBox。它们不进入预训练。

机器人预训练为 15×4×50＝3,000 个 episode，episode indices 0–49，共 1,033,711 帧。少样本为 5×4×10＝200 个 episode，indices 0–9，共 60,246 帧。“10 条”是每个任务—级别的预算，并不是四级合计 10 条。

Scratch 和 P+FT 使用同一批示范与同一人类视频缓存，均为五个目标任务共享一个模型。按官方代码在全部有效机器人帧上 shuffle，不加入此前单任务实验的级别均衡 sampler。

官方冻结 encoder 缓存以 human repo ID 为 key，每个任务复用一个缓存嵌入，不是每条配对轨迹独立缓存一个人类视频。保留这一实现并记录选取来源。

## 归一化与信息边界

此次沿用仓库按任务—级别读取的原始 `meta/stats.json`，不套用此前 PlaceMugRack 的 pooled normalizer。

官方评估器会读未见目标任务的统计；官方少样本配置虽然仅训练 0–9 个 episode，原始元数据统计可能覆盖完整 50 条示范。因此结果应表述为“沿用发布实现、目标任务元数据可用的权重零样本／少样本”，不能称完全不使用目标机器人信息。Scratch 与 P+FT 共享相同统计，预训练差异以外的条件一致；严格仅十条统计的版本属于另一个协议，此次不混用。

RGB-only 下载约 9.33 GB。发布的 loader 在筛选摄像头前会检查所有元数据视频，所以另建 `prepared/sim_rgb/`：只从副本的 `info.json.features` 移除未使用的视频字段，数据与 RGB 视频链接到原文件，stats 保持逐字节一致。原始数据及此前实验不会修改。

## 训练与 checkpoint

训练顺序为 `pretrain15 → 评估 Seen／ZS → scratch5 → 评估 Scr. → finetune5 → 评估 P+FT`。

精度沿用官方 BF16 autocast。动作模型和 adapter 更新，DINO 冻结。优化器参数分组、backbone 学习率 1e-5、weight decay 1e-4、梯度裁剪 0.1／1.0 和 cosine schedule 与源码一致。只通过增加 DataLoader workers 改善吞吐，batch 保持 256。

正式评估固定使用第 100 个 epoch 完成后的模型，不按最终测试结果选 checkpoint。第 10、50、100 epoch 另存用于审计，逐 epoch 更新 `latest.pt`。

P+FT 只继承预训练模型和 adapter 权重；optimizer、scheduler、epoch 与 iteration 从零开始。直接使用官方 `--resume-from` 会继承优化器和已完成 epoch，不适合作为这里的少样本初始化，因此在外层明确处理。所有可训练 tensor 必须逐项验证回读一致。

Checkpoint 省略不可训练的 DINO 主干权重，ACT／ResNet、adapter、norm 和 buffers 原样保存，依赖本机固定的官方 DINO 权重恢复。文件仍兼容官方 evaluator，省略内容和依赖在清单中注明。

## 评估与指标

每个条件—任务—级别 10 个 episode，配对 reset seeds 固定 6000–6009；500 步、physx_cpu、rt-fast、pd_joint_pos、单环境。调用官方 `evaluate_with_task_encoder`，沿用 light temporal aggregation、window 4、每 4 步查询一次动作块；不计算 DTW。

SR 使用官方 `success_at_end`，额外保存 `success_once`，两者不能互换。

论文 Sub-SR 是阶段 shaped reward 的 episode 峰值超过阈值后，计算完成阶段占比；仓库有 RewardTracker，但没有发布该阈值。经用户明确同意，暂固定 threshold＝0.95，报告字段名为 `Sub_SR_tau_095`，明确不是已确认的论文原阈值。保存全部阶段峰值以便以后重算，不将连续 dense reward 平均值冒充 Sub-SR。

评估 wrapper 只在 `env.step` 返回后读取 tracker，物理状态、动作与奖励计算保持原样。各单元先保存第一个 episode 录像，再关闭后续录像。所有正式单元必须成功执行；基础设施报错不会记作任务失败率。

已见／未见分别对任务和级别等权平均。P+FT−Scr. 用相同场景与示范的配对差异计算，并同时展示逐任务、逐级别结果。单 seed 与每单元十次评估存在明显方差，只解释本次结果，不直接与论文表 3 的 15／30／45 三档均值作数值等价比较。

## 执行与观察

数据与资产位于根分区 `/var/tmp/imitator-game-zxc/`，其可用空间比 `/home` 充裕；启动时根分区约 200 GB、home 约 74 GB。机器人资产仅抽取所需模型，已存在的资产必须校验相同后保留。

训练和下载使用独立 systemd 用户 service，避免交互工具会话退出杀死任务。控制器会自动顺序执行全部阶段，并写 `status.json`、逐步 JSONL、checkpoint 清单和最后审计；出错会留下明确状态与日志。电脑需保持开机且不进入睡眠；这不是开机自启动服务。

启动整个实验：

```bash
cd /home/zxc/Imitator
bash scripts/start_act15_service.sh
```

启动入口先检查数据和标定，再验证 checkpoint 的官方闭环加载，随后启动预训练。资产可并行下载，正式多任务评估前会等待资产校验并逐场景检查。已在运行时重复调用会返回现有 service 状态，不会重复训练；不完整的训练记录会要求先诊断，避免误覆盖。

只检查启动条件，不启动：

```bash
/home/zxc/miniconda3/envs/imitator/bin/python experiments/act_dinov2_15task/execute.py --check-only
```

查看真实状态：

```bash
/home/zxc/miniconda3/envs/imitator/bin/python scripts/check_act15_status.py
systemctl --user status imitator-act15.service --no-pager
```

查看控制器日志：`journalctl --user -u imitator-act15.service -f`。训练日志：`tail -f experiments/act_dinov2_15task/logs/train_pretrain15.log`；第一阶段完成后相应日志分别为 `train_scratch5.log` 与 `train_finetune5.log`。训练逐步数值另存在 `runs/<condition>/train.jsonl`。

本机 8-worker 标定为约 0.135 s/update、1,903 samples/s、6.46 GiB 峰值 PyTorch 显存；预训练预计约 15.1 小时，不含后续评估、Scratch 和 P+FT。这个短标定估计会随实际持续吞吐修正。

复现脚本和配置：[experiments/act_dinov2_15task](../experiments/act_dinov2_15task)。最终结果完成后写入 `12_ACT_DINOV2_15TASK_RESULTS.md`；完整对照结束前不预填最终成功率。

## 2026-10-09 进度与 L3 评估修正

15 任务预训练已于 10 月 9 日 14:36 完成 100 epochs、403,800 次更新，用时约 21.76 小时；最终 epoch 的平均 loss 为 0.01761。短标定的 15.1 小时估计偏乐观，以实测时间为准。最终 checkpoint 位于 `runs/pretrain15/checkpoints/final_model.pt`，SHA256 记录在 `runs/pretrain15/complete.json`。

首轮完成了 40 个单元、400 个 episode，但检查发现外层 `evaluate.py`／`probe_env.py` 的 L3 环境名称转换漏用了官方 `extract_base_env_name`。官方 L3 使用独立注册环境，例如 `L3_TwoRobotPlaceMugRack-v1` 必须创建 `TwoRobotPlaceMugRackL3-v1`，不能仅创建基础任务并打开 L3 开关。旧的 10 个 L3 单元、100 个 episode 归档排除；L0–L2 的 30 个单元、300 个 episode 保留。

修正只涉及外层评估、场景检查和最终审计；核心源码、训练脚本、数据、归一化统计、权重和训练预算保持原样。修正前源清单、原始源码、旧评估结果及原因记录在 `checks/protocol-amendments/01_official_l3_dispatch/`，新的源清单明确登记修正；最终审计不会再声称所有外层脚本自首次冻结以来完全未变。

已保存的预训练 checkpoint 用相同人类视频条件、目标统计和 seeds 6000–6009 重跑全部 10 个 L3 单元；后续 Scratch／P+FT 自动使用修正入口。FoldBox L3 所需 PartNet 100141 从已校验的本机资产包补取，不重新下载整个包。修正队列由 `imitator-act15-l3-repair.service` 执行，进度在 `checks/l3-repair.json`；主训练 service 持续运行。

## 完成状态：2026-10-09 19:00:43

三个条件均已完成 100 epochs，正式评估完成 80 个单元、800 个 episode；L3 的 100 个重评估 episode 已替换旧结果。配对视频与 reset、相同少样本预算、源清单和 checkpoint 审计通过，两个 service 均以 exit code 0 正常结束。预训练共 403,800 次更新；Scratch 和 P+FT 各 23,600 次更新。最终结果见 [第十二篇](12_ACT_DINOV2_15TASK_RESULTS.md)，原始数值见 `experiments/act_dinov2_15task/results.json` 与 `results.csv`。
